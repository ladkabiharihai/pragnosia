"""Train the LM on the tokenized slice. One file, one loop, checkpoints so intermediate states can be probed.

Run:  .venv/bin/python -m src.train --steps 120000 --out /mnt/kv_cache/run1
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .common import Run, cap_gpu, seed_all, require_prod, eval_bpb
from .model import LM
from .optim import Fabric

MIX = {"dclm": .413, "fineweb-edu": .322, "starcoderdata": .130, "finemath": .036,
       "open-web-math": .030, "arxiv": .030, "wikipedia": .023, "algebraic-stack": .015}


def loaders(data, only=None):
    """only: restrict to these sources, renormalising the mix. Used for a long-context phase, where the point
    is to train on windows that lie inside ONE document: 21% of documents in the full mix exceed 2048 tokens,
    but arxiv's median is 12,710 and algebraic-stack's 10,375."""
    bins = {k: np.memmap(f"{data}/{k}.bin", dtype=np.uint16, mode="r")
            for k in MIX if os.path.exists(f"{data}/{k}.bin") and (only is None or k in only)}
    ks = list(bins)
    ps = np.array([MIX[k] for k in ks], dtype=np.float64); ps /= ps.sum()
    return bins, ks, ps


def batch(bins, ks, ps, rng, B, T):
    o = np.empty((B, T + 1), dtype=np.int64)
    for i in range(B):
        a = bins[ks[rng.choice(len(ks), p=ps)]]
        j = rng.integers(0, len(a) - T - 2)
        o[i] = a[j:j + T + 1]
    # pinned + non-blocking: a pageable .cuda() waits for every kernel already queued before it returns, so
    # the host could never run ahead of the card and the card idled through each step's sampling.
    return torch.from_numpy(o).pin_memory().cuda(non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--mlp", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--data", default="/mnt/kv_cache/tok1b_v8192")
    ap.add_argument("--sources", default="", help="comma-separated subset of the mix (long-context phase)")
    ap.add_argument("--vocab", type=int, default=8192, help="must match the tokenizer the data was made with")
    ap.add_argument("--chunks", type=int, default=1, help="head chunks for the loss; >1 never holds all logits at once")
    ap.add_argument("--accum", type=int, default=1, help="gradient-accumulation steps: --batch is the MICRO-batch; tokens per update = accum*batch*ctx. At ctx 2048 the 78M carrier peaks at 20.2 GiB with batch 16 and 5.9 GiB with batch 4 x accum 4, at 88% of the speed")
    ap.add_argument("--scan-chunk", type=int, default=0, help="associative scan chunk (0: the carrier's default)")
    ap.add_argument("--recompute", type=int, default=0, help="recompute each scan chunk in backward: at ctx 8192 this halves memory (25.4 -> 14.9 GiB) for 2% of speed")
    ap.add_argument("--out", default="/mnt/kv_cache/run1")
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--grow-at", type=int, default=0, help="step at which to widen the model in place (0: never)")
    ap.add_argument("--grow-to", type=int, default=0, help="width to grow to at --grow-at")
    ap.add_argument("--fused", type=int, default=1, help="one compiled kernel group per parameter update")
    ap.add_argument("--graph", type=int, default=1, help="replay the optimizer step as a CUDA graph")
    ap.add_argument("--graph-step", type=int, default=1,
                    help="capture forward+loss+backward as ONE CUDA graph and replay it per micro-batch")
    ap.add_argument("--resume", default="")
    ap.add_argument("--mixer", default="spin", choices=["spin", "gspin", "fspin", "aspin", "dspin", "wspin", "xspin", "w2spin", "hspin", "hdspin", "sdspin", "altspin", "attn"])
    ap.add_argument("--spike", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    require_prod()
    with Run("train", note=f"d{a.d} L{a.layers} mlp{a.mlp} ctx{a.ctx} B{a.batch} "
                           f"{a.steps} steps -> {a.out}", seed=a.seed) as r:
        cap_gpu(a.cap)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"/mnt/kv_cache/tok_v{a.vocab}.json")
        assert tok.get_vocab_size() == a.vocab, (tok.get_vocab_size(), a.vocab)
        bins, ks, ps = loaders(a.data, only=[x for x in a.sources.split(",") if x] or None)
        r.log(f"corpora {ks}")
        m = LM(vocab=a.vocab, d=a.d, layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer).cuda()
        # The carrier is 94% of an uncompiled step (195 ms of 206), because its cross-chunk pass is a python
        # loop issuing many tiny kernels. Compiling fuses them: 40 -> 154 ktok/s measured, a 3.8x speedup.
        for mx in m.mix:
            if a.scan_chunk and hasattr(mx, "chunk"): mx.chunk = a.scan_chunk
            if a.recompute and hasattr(mx, "recompute"): mx.recompute = True
        fwd = torch.compile(m) if a.compile else m
        trunk = torch.compile(m.trunk) if a.compile else m.trunk
        P = sum(p.numel() for p in m.parameters())
        car = sum(p.numel() for n, p in m.named_parameters() if "mix." in n)
        base = [a.lr, 1e-3]
        # The eager optimizer was CPU-bound: 19.5 ms issuing ~4,500 kernels per step. Fused and graphed it is
        # a few hundred kernels replayed without python; grads must then be zeroed in place, not set to None.
        opt = Fabric(m.param_groups(lr=a.lr), lr=a.lr, fused=bool(a.fused), cuda_graph=bool(a.graph and a.fused))
        set_none = not opt.cuda_graph
        rng = np.random.default_rng(a.seed)

        # THE MICRO-STEP AS ONE CUDA GRAPH. With the scan in Triton the step was launch-bound: 1.3 ms of
        # kernels inside a 23.9 ms micro-step at micro-batch 4, the rest python and launch gaps. The same
        # recipe as the optimizer and the decode step: a static input buffer, grads zeroed in place, the whole
        # forward + chunked loss + backward captured once and replayed per accumulation step. Measured at 78M
        # ctx 2048: delta 143 -> 264 ktok/s, plain 282 -> 295, attention 312 -> 325. torch.compile's own
        # "reduce-overhead" mode could not do this (it refuses grads that outlive the replay); manual capture
        # can. Copying the batch into the static buffer is the only per-step host work left.
        def capture():
            if not (a.graph_step and a.compile): return None
            xs = torch.empty((a.batch, a.ctx + 1), dtype=torch.long, device="cuda")
            xs.copy_(batch(bins, ks, ps, np.random.default_rng(11), a.batch, a.ctx))
            for p_ in m.parameters():
                if p_.grad is None: p_.grad = torch.zeros_like(p_)
            out = {}
            def micro():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = trunk(xs[:, :a.ctx])
                l_ = m.loss(h, xs[:, 1:a.ctx + 1], chunks=max(1, a.chunks))
                (l_ / a.accum).backward()
                out["loss"] = l_.detach()
            s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(3): micro()                       # compile and warm on a side stream
            torch.cuda.current_stream().wait_stream(s_)
            for p_ in m.parameters(): p_.grad.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                micro()
            for p_ in m.parameters(): p_.grad.zero_()
            return g, xs, out["loss"]
        tok_total = a.steps * a.batch * a.ctx * a.accum
        r.log(f"{P/1e6:.1f}M params, carrier {100*car/P:.1f}%, "
              f"{tok_total/1e9:.2f}B tokens = {tok_total/P:.1f} tok/param")

        # RESUME. The last run diverged at step 14,400 and the trainer then wrote NaN weights over a healthy
        # checkpoint, so the only survivor was a copy taken by accident for a probe.
        start = 0; grew_ctx = None
        if a.resume and os.path.exists(a.resume):
            sd = torch.load(a.resume, map_location="cuda")
            if sd["cfg"].get("ctx", a.ctx) != a.ctx:
                # A LONG-CONTEXT PHASE. The positional table is sized to the old window; tile the trained
                # pattern to fill the new one. That is exactly what inference already does when it processes a
                # long input as consecutive windows with the state carried, so training this way teaches the
                # model to use the state ACROSS a window boundary rather than only within one.
                old_ctx = sd["cfg"]["ctx"]
                reps = -(-a.ctx // old_ctx)
                w = sd["model"]["p.weight"]
                sd["model"]["p.weight"] = w.repeat(reps, 1)[: a.ctx]
                r.log(f"extended the positional table {old_ctx} -> {a.ctx} by tiling the trained pattern {reps}x")
                sd["cfg"]["ctx"] = a.ctx
                grew_ctx = (old_ctx, sd["cfg"]["d"])
            if sd["cfg"]["d"] != m.d:                      # a grown run: rebuild at the checkpoint's width
                m = LM(vocab=a.vocab, d=sd["cfg"]["d"], layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer).cuda()
                fwd = torch.compile(m) if a.compile else m
                trunk = torch.compile(m.trunk) if a.compile else m.trunk
                opt = Fabric(m.param_groups(lr=a.lr), lr=a.lr, fused=bool(a.fused), cuda_graph=bool(a.graph and a.fused))
            m.load_state_dict(sd["model"]); start = sd["step"]
            # RESTORE THE OPTIMIZER TOO. Loading only the weights leaves the optimizer cold, with its second
            # moment at zero, so its first steps after a resume are near-maximal on a model that was already
            # converged to that point. This project measured the same failure at growth events: a rebuilt
            # optimizer takes a 1.5x larger first step and costs about a nat of shock. A resume is the same
            # event. Recorded because the first version of this resume shipped without it.
            if "opt" in sd:
                try:
                    opt.load_state_dict(sd["opt"])
                    if grew_ctx is not None:
                        # the positional table grew with the window, so its second moment is still sized to
                        # the old one. Fabric.grew extends it with the median of the existing rows; without
                        # this the fused update tried to add a 2048-vector to an 8192-vector and the compile
                        # failed rather than silently doing something wrong.
                        opt.grew(m.p.weight, grew_ctx)
                        r.log(f"extended the optimizer's statistics for the positional table to {a.ctx} rows")
                    r.log(f"resumed optimizer state ({len(sd['opt'].get('state', {}))} tensors)")
                except Exception as e:
                    r.log(f"optimizer state present but not loadable ({type(e).__name__}); starting cold")
            else:
                r.log("WARNING: checkpoint carries no optimizer state; the first steps will be a shock")
            r.log(f"resumed from {a.resume} at step {start}, loss {sd['loss']:.4f}")
        gs = capture()
        if gs is not None:
            set_none = False
            r.log(f"micro-step captured as a CUDA graph; {torch.cuda.memory_reserved()/2**30:.1f} GiB reserved after capture")

        t0 = time.time(); ema = None
        # THE GUARD LIVES ON THE CARD. Reading the gradient norm and the loss every step were the only two
        # syncs left in the loop, and each one drained the queue and left the card idle through the host's
        # bookkeeping: 2 ms of a 128 ms update measured piece by piece. The norm history, its median, the
        # keep/skip decision and the loss average are device tensors now; the host reads them once per 200
        # steps. A skipped step zeroes its gradients through one foreach multiply instead of a branch, so the
        # optimizer sees a zero gradient (its moments decay once) rather than being skipped outright; on
        # steps that do not spike, which is all but a few per ten thousand, nothing changes.
        hist = torch.zeros(1000, device="cuda"); skips_t = torch.zeros((), device="cuda"); ema_t = None
        nonfin = torch.zeros((), device="cuda"); nf_prev = 0
        marks = []; bpbs = []; skipped = 0
        nxt = [batch(bins, ks, ps, rng, a.batch, a.ctx) for _ in range(a.accum)]
        for i in range(start, a.steps):
            w = min(1.0, (i + 1) / a.warm)
            # cosine over the whole budget: a flat rate lets a model equilibrate at that rate's noise floor,
            # which is what drove the previous project's BPB upward for hundreds of thousands of steps.
            f = min(1.0, max(0.0, (i - a.warm) / max(1, a.steps - a.warm)))
            sched = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f))
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = base[gi] * sched * w
            if a.grow_at and i == a.grow_at and m.d < a.grow_to:
                # GROWTH (vision goal 7): widen in place, function preserved, optimizer state carried. The
                # loss right before and right after is logged so the cost of the event is a number.
                xg = batch(bins, ks, ps, np.random.default_rng(7), a.batch, a.ctx)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lb = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, a.vocab), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                for p_, (old_p, sh) in m.grow(a.grow_to).items():
                    opt.grew(p_, sh, replaces=old_p)
                opt.zero_grad(set_to_none=True)             # grad buffers are the old shape
                torch._dynamo.reset()                        # compiled graphs hold the old parameters
                fwd = torch.compile(m) if a.compile else m
                trunk = torch.compile(m.trunk) if a.compile else m.trunk
                gs = capture()                                # the graph held the old parameters
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    la = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, a.vocab), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                P = sum(p.numel() for p in m.parameters())
                r.log(f"    GREW to d={m.d} at step {i}: {P/1e6:.1f}M params, same batch loss {lb:.4f} -> {la:.4f}, "
                      f"optimizer state {opt.state_bytes()/2**20:.0f} MiB carried")
            opt.zero_grad(set_to_none=set_none)
            loss = None
            for _acc in range(a.accum):
                x = nxt[_acc]
                if gs is not None:
                    gs[1].copy_(x); gs[0].replay(); l_ = gs[2].clone()
                elif a.chunks > 1:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = trunk(x[:, :a.ctx])
                    l_ = m.loss(h, x[:, 1:a.ctx + 1], chunks=a.chunks)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l_ = F.cross_entropy(fwd(x[:, :a.ctx]).reshape(-1, a.vocab), x[:, 1:a.ctx + 1].reshape(-1))
                if gs is None: (l_ / a.accum).backward()      # the graph replay already did
                loss = l_.detach() if loss is None else loss + l_.detach()
            loss = loss / a.accum
            # the next step's batches are sampled NOW, while the card is still busy with this one; the
            # gradient-norm read below is a sync and everything after it is a gap the card sits through
            nxt = [batch(bins, ks, ps, rng, a.batch, a.ctx) for _ in range(a.accum)]
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)          # a device scalar, not read
            # SPIKE GUARD. The previous run went from loss 4.0998 to NaN between step 14,200 and 14,400, a
            # sudden spike rather than a drift. Clipping bounds the step but still applies a direction taken
            # from a pathological batch, and the optimizer then carries that direction in its state for
            # thousands of steps. Compared against a running median so it needs no retuning as loss scale
            # changes. This guard existed in the previous project's trainer and I failed to carry it here.
            j = i - start; fin = torch.isfinite(gn)
            med = hist[:min(j, 1000)].median() if j else gn
            keep = fin & ((gn <= a.spike * med) if j >= 200 else torch.ones((), dtype=torch.bool, device="cuda"))
            hist[j % 1000] = torch.where(keep, gn, hist[j % 1000])            # only kept steps enter the history
            skips_t.add_(~keep)
            # zero the gradients of a skipped step; a non-finite one becomes NaN * 0 here and is zeroed for
            # good inside the fused update (nan_to_num), so the optimizer state never sees it
            torch._foreach_mul_([p_.grad for p_ in m.parameters() if p_.grad is not None], keep.float())
            opt.step()
            # A NON-FINITE LOSS MUST NOT POISON THE AVERAGE. The first version of this device-side guard folded
            # every loss into the EMA, so one NaN batch made the average NaN for ever and the run stopped at the
            # next log line even though the guard had already zeroed that step's gradients and the weights were
            # still finite: run20 died that way at step 32,600 on 2026-09-17 with a healthy checkpoint on disk.
            # The average now skips non-finite losses and they are counted instead; what decides a real
            # divergence is the weights, checked below.
            fin_l = torch.isfinite(loss)
            nonfin.add_(~fin_l)
            ema_t = loss.clone() if ema_t is None else torch.where(fin_l, ema_t * 0.99 + loss * 0.01, ema_t)
            if (i + 1) % 200 == 0:
                ema, nsk, medv = ema_t.item(), int(skips_t.item()), float(med.item())   # the one sync
                nnf = int(nonfin.item())
                el = time.time() - t0
                # Rate and ETA must count from the RESUME point, not from step zero, or a resumed run
                # reports a throughput it never achieved and an ETA that is wildly optimistic.
                done = i + 1 - start
                r.log(f"{i+1}/{a.steps} loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {done*a.batch*a.ctx*a.accum/el/1e3:.0f} ktok/s "
                      f"eta {(a.steps-i-1)*el/max(1,done)/60:.0f}m"
                      + (f" {torch.cuda.memory_reserved()/2**30:.1f} GiB" if done == 200 else "")
                      + (f" recaptures {opt.recaptures}" if opt.recaptures else ""))
                if nnf > nf_prev:
                    # the guard zeroed those steps; the run continues unless the weights themselves went bad
                    alive = bool(torch.stack([p_.isfinite().all() for p_ in m.parameters()]).all().item())
                    recent_nf = nnf - nf_prev; nf_prev = nnf
                    r.log(f"    {recent_nf} non-finite loss(es) in the last 200 steps (total {nnf}); "
                          f"weights {'still finite' if alive else 'NON-FINITE'}")
                    if not alive or recent_nf > 100:
                        r.log(f"    DIVERGED: the weights are not finite; stopping rather than burning the card")
                        break
                if not math.isfinite(ema):
                    r.log(f"    NON-FINITE RUNNING LOSS with finite weights; stopping, this should not happen")
                    break
                if nsk > skipped:
                    r.log(f"    {nsk - skipped} spike(s) skipped in the last 200 steps (median |g| {medv:.2f}, total {nsk})")
                marks.append((i + 1, nsk)); skipped = nsk
                # DIVERGED, not spiking: the attention control blew up at step ~11,000 and the guard then
                # skipped every one of the next 27,000 steps while the log showed a healthy-looking rate.
                # A guard that skips everything is a frozen model burning the card; stop and say so.
                if len(marks) >= 4 and nsk - marks[-4][1] >= 250:
                    r.log(f"    DIVERGED: {nsk} skips, {nsk - marks[-4][1]} of them within the last 600 steps; stopping")
                    break
            if (i + 1) % a.eval_every == 0:
                # BITS PER BYTE, not loss per token. Per-token loss depends on the vocabulary and is
                # comparable to nothing; bits per byte removes the tokenizer and is the only number that can
                # be set against another model.
                b, nb = eval_bpb(m, tok, bins, ks, ps, np.random.default_rng(1234),
                                 n_batches=8, B=8, T=a.ctx, vocab=a.vocab)
                r.log(f"  eval step {i+1}: BPB {b:.4f}  ({nb/1e3:.0f}k bytes)")
                # DEGRADING, not frozen: fspin at 6e-3 went 1.7075 -> 1.7534 -> 2.1130 over three evals while
                # the skip guard saw only 200 skips in 15k steps. Three consecutive rises, each above the
                # eval's own noise (measured spread ~0.005 on this fixed sample), is a run going backwards.
                bpbs.append(b)
                if len(bpbs) >= 4 and all(bpbs[-k] > bpbs[-k-1] + 0.01 for k in (1, 2, 3)):
                    r.log(f"    DEGRADING: BPB rose over the last three evals {bpbs[-4:]}; stopping")
                    break
            if (i + 1) % a.save_every == 0 or i + 1 == a.steps:
                # NEVER OVERWRITE A GOOD CHECKPOINT WITH A BROKEN ONE. The previous run saved NaN weights over
                # a healthy checkpoint and only a stray copy survived. Verify before writing, and keep the
                # previous file as last-known-good.
                bad = sum(1 for v in m.state_dict().values() if not torch.isfinite(v).all())
                if bad:
                    r.log(f"    refusing to checkpoint: {bad} non-finite tensors")
                else:
                    if os.path.exists(f"{a.out}/ckpt.pt"):
                        os.replace(f"{a.out}/ckpt.pt", f"{a.out}/ckpt_prev.pt")
                    torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                                "step": i + 1, "loss": ema_t.item(),   # the current average, not the last log's
                                "cfg": dict(vocab=a.vocab, d=m.d, layers=a.layers,
                                            mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer)},
                               f"{a.out}/ckpt.pt")
        if ema_t is None:
            r.log(f"NOTHING TO DO: resumed at step {start} with --steps {a.steps}. A resumed run counts from the "
                  f"checkpoint's step, so --steps must exceed it.")
        else:
            ema = ema_t.item(); skipped = int(skips_t.item())
            r.log(f"final loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f}, {skipped} spike-skips")


if __name__ == "__main__":
    main()
