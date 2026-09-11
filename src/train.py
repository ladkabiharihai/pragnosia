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


def loaders(data):
    bins = {k: np.memmap(f"{data}/{k}.bin", dtype=np.uint16, mode="r")
            for k in MIX if os.path.exists(f"{data}/{k}.bin")}
    ks = list(bins)
    ps = np.array([MIX[k] for k in ks], dtype=np.float64); ps /= ps.sum()
    return bins, ks, ps


def batch(bins, ks, ps, rng, B, T):
    o = np.empty((B, T + 1), dtype=np.int64)
    for i in range(B):
        a = bins[ks[rng.choice(len(ks), p=ps)]]
        j = rng.integers(0, len(a) - T - 2)
        o[i] = a[j:j + T + 1]
    return torch.from_numpy(o).cuda()


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
    ap.add_argument("--out", default="/mnt/kv_cache/run1")
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--grow-at", type=int, default=0, help="step at which to widen the model in place (0: never)")
    ap.add_argument("--grow-to", type=int, default=0, help="width to grow to at --grow-at")
    ap.add_argument("--fused", type=int, default=1, help="one compiled kernel group per parameter update")
    ap.add_argument("--graph", type=int, default=1, help="replay the optimizer step as a CUDA graph")
    ap.add_argument("--resume", default="")
    ap.add_argument("--mixer", default="spin", choices=["spin", "gspin", "fspin", "attn"])
    ap.add_argument("--spike", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    require_prod()
    with Run("train", note=f"d{a.d} L{a.layers} mlp{a.mlp} ctx{a.ctx} B{a.batch} "
                           f"{a.steps} steps -> {a.out}", seed=a.seed) as r:
        cap_gpu(a.cap)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
        bins, ks, ps = loaders(a.data)
        r.log(f"corpora {ks}")
        m = LM(vocab=8192, d=a.d, layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer).cuda()
        # The carrier is 94% of an uncompiled step (195 ms of 206), because its cross-chunk pass is a python
        # loop issuing many tiny kernels. Compiling fuses them: 40 -> 154 ktok/s measured, a 3.8x speedup.
        fwd = torch.compile(m) if a.compile else m
        P = sum(p.numel() for p in m.parameters())
        car = sum(p.numel() for n, p in m.named_parameters() if "mix." in n)
        base = [a.lr, 1e-3]
        # The eager optimizer was CPU-bound: 19.5 ms issuing ~4,500 kernels per step. Fused and graphed it is
        # a few hundred kernels replayed without python; grads must then be zeroed in place, not set to None.
        opt = Fabric(m.param_groups(lr=a.lr), lr=a.lr, fused=bool(a.fused), cuda_graph=bool(a.graph and a.fused))
        set_none = not opt.cuda_graph
        rng = np.random.default_rng(a.seed)
        tok_total = a.steps * a.batch * a.ctx
        r.log(f"{P/1e6:.1f}M params, carrier {100*car/P:.1f}%, "
              f"{tok_total/1e9:.2f}B tokens = {tok_total/P:.1f} tok/param")

        # RESUME. The last run diverged at step 14,400 and the trainer then wrote NaN weights over a healthy
        # checkpoint, so the only survivor was a copy taken by accident for a probe.
        start = 0
        if a.resume and os.path.exists(a.resume):
            sd = torch.load(a.resume, map_location="cuda")
            if sd["cfg"]["d"] != m.d:                      # a grown run: rebuild at the checkpoint's width
                m = LM(vocab=8192, d=sd["cfg"]["d"], layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer).cuda()
                fwd = torch.compile(m) if a.compile else m
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
                    r.log(f"resumed optimizer state ({len(sd['opt'].get('state', {}))} tensors)")
                except Exception as e:
                    r.log(f"optimizer state present but not loadable ({type(e).__name__}); starting cold")
            else:
                r.log("WARNING: checkpoint carries no optimizer state; the first steps will be a shock")
            r.log(f"resumed from {a.resume} at step {start}, loss {sd['loss']:.4f}")

        t0 = time.time(); ema = None
        gnorms = []; skipped = 0; recent = []; bpbs = []
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
                    lb = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, 8192), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                for p_, (old_p, sh) in m.grow(a.grow_to).items():
                    opt.grew(p_, sh, replaces=old_p)
                opt.zero_grad(set_to_none=True)             # grad buffers are the old shape
                torch._dynamo.reset()                        # compiled graphs hold the old parameters
                fwd = torch.compile(m) if a.compile else m
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    la = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, 8192), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                P = sum(p.numel() for p in m.parameters())
                r.log(f"    GREW to d={m.d} at step {i}: {P/1e6:.1f}M params, same batch loss {lb:.4f} -> {la:.4f}, "
                      f"optimizer state {opt.state_bytes()/2**20:.0f} MiB carried")
            x = batch(bins, ks, ps, rng, a.batch, a.ctx)
            opt.zero_grad(set_to_none=set_none)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(fwd(x[:, :a.ctx]).reshape(-1, 8192), x[:, 1:a.ctx + 1].reshape(-1))
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            # SPIKE GUARD. The previous run went from loss 4.0998 to NaN between step 14,200 and 14,400, a
            # sudden spike rather than a drift. Clipping bounds the step but still applies a direction taken
            # from a pathological batch, and the optimizer then carries that direction in its state for
            # thousands of steps. Compared against a running median so it needs no retuning as loss scale
            # changes. This guard existed in the previous project's trainer and I failed to carry it here.
            skip = len(gnorms) >= 200 and (not math.isfinite(gn) or gn > a.spike * float(np.median(gnorms)))
            if skip:
                opt.zero_grad(set_to_none=set_none); skipped += 1
                recent.append(i)
                # DIVERGED, not spiking: the attention control blew up at step ~11,000 and the guard then
                # skipped every one of the next 27,000 steps while the log showed a healthy-looking rate.
                # A guard that skips everything is a frozen model burning the card; stop and say so.
                if len(recent) >= 250 and i - recent[-250] < 500:
                    r.log(f"    DIVERGED: {len(recent)} skips, 250 of them within the last 500 steps; stopping")
                    break
                if skipped <= 10 or skipped % 200 == 0:
                    r.log(f"    spike at step {i+1}: |g|={gn:.2f} vs median {float(np.median(gnorms)):.2f}"
                          f" -- SKIPPED (total {skipped})")
            else:
                opt.step()
                if math.isfinite(gn):
                    gnorms.append(gn)
                    if len(gnorms) > 1000: gnorms.pop(0)
            v = loss.item()
            if not math.isfinite(v):
                r.log(f"    NON-FINITE LOSS at step {i+1}; stopping rather than burning the card on NaN")
                break
            ema = v if ema is None else 0.99 * ema + 0.01 * v
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                # Rate and ETA must count from the RESUME point, not from step zero, or a resumed run
                # reports a throughput it never achieved and an ETA that is wildly optimistic.
                done = i + 1 - start
                r.log(f"{i+1}/{a.steps} loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {done*a.batch*a.ctx/el/1e3:.0f} ktok/s "
                      f"eta {(a.steps-i-1)*el/max(1,done)/60:.0f}m"
                      + (f" recaptures {opt.recaptures}" if opt.recaptures else ""))
            if (i + 1) % a.eval_every == 0:
                # BITS PER BYTE, not loss per token. Per-token loss depends on the vocabulary and is
                # comparable to nothing; bits per byte removes the tokenizer and is the only number that can
                # be set against another model.
                b, nb = eval_bpb(m, tok, bins, ks, ps, np.random.default_rng(1234),
                                 n_batches=8, B=8, T=a.ctx)
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
                                "step": i + 1, "loss": ema,
                                "cfg": dict(vocab=8192, d=m.d, layers=a.layers,
                                            mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer)},
                               f"{a.out}/ckpt.pt")
        r.log(f"final loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f}, {skipped} spike-skips")


if __name__ == "__main__":
    main()
