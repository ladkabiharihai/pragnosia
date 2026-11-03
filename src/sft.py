"""Teach a pretrained checkpoint to ANSWER: supervised fine-tuning on question-answer pairs.

Loss is on the answer tokens only (the mask built by eval/build_sft.py), so the model learns to respond rather
than to continue the question. The optimiser is the project's own trust-ratio Fabric, the one the model was
pretrained with: AdamW at a small rate damaged this architecture's general text badly in eval/continual.py
(1.07 -> 1.56 bits per byte in 150 steps), because its per-coordinate normalisation fights the weight scales a
trust-ratio optimiser produces.

Compiled by default. Eager runs the carrier's cross-chunk pass as a python loop and measured 15k tokens/s here.
The buffer-reuse fault that a micro-batch above one hits under torch.compile (PROGRESS.md, 2026-09-22) appeared
at an 8,192-token window; at these 1,024-token rows the per-chunk state is the size run33 trained with for 10B
tokens, and --compile 0 remains as the fallback.
usage: python -m src.sft --init <ckpt> --out <dir> [--lr 1.5e-4] [--epochs 2] [--batch 8]"""
import argparse, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from .common import cap_gpu, seed_all
from .model import LM
from .optim import Fabric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/mnt/kv_cache/sft_v8192_r1025")
    ap.add_argument("--row", type=int, default=1025, help="packed row length; 1,025 gives 1,024-token inputs, a multiple of the 512 attention window")
    ap.add_argument("--fused", type=int, default=1, help="the fused optimizer step instead of the per-parameter python loop")
    ap.add_argument("--lr", type=float, default=1.5e-4, help="trust fraction, peak")
    ap.add_argument("--spectral-lr", type=float, default=1e-4, help="absolute step for timescales and frequencies, peak")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--prodigy-steps", type=int, default=0,
                    help="psf: freeze Prodigy's own step size after this many steps (0: never). 2026-10-14: it only ratchets up "
                         "(2.7e-5 -> 3.2e-4 by ~5,000 steps on the replay mix) and three long replay fine-tunes diverged; "
                         "pretraining holds its step size for the same reason")
    ap.add_argument("--rehearse", default="",
                    help="comma-separated data dirs of EARLIER fine-tunes (2026-10-16, goal 7 for skills): their held-out answer loss is "
                         "the model's self-test; a set whose loss has risen above its step-0 value is FADING and joins the sampling "
                         "pool beside the new data (one source drawn uniformly per row) until it recovers. No mixing ratio is set: "
                         "the pool's make-up follows from which sets fade. eval/continual3.py's rule, applied to whole skills")
    ap.add_argument("--cap", type=float, default=24.0)
    ap.add_argument("--optim", default="fabric", choices=["fabric", "psf"],
                    help="psf: schedule-free Prodigy at its published defaults, the same as pretraining: no rate, "
                         "no schedule, no warm-up; the saved checkpoint holds the averaged weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1, help="compile the forward; eager runs the carrier's chunk loop in python, ~10x slower")
    a = ap.parse_args()
    cap_gpu(a.cap); seed_all(a.seed); os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/sft.log", "a")
    def say(s):
        line = f"[{time.time() - t0:8.1f}s] {s}"; print(line, flush=True); log.write(line + "\n"); log.flush()
    t0 = time.time()

    sd = torch.load(a.init, map_location="cuda", weights_only=False)
    m = LM(**sd["cfg"]).cuda(); m.load_state_dict(sd["model"]); m.train()
    # THE FUSED SCAN, as in training: the python chunk scan's backward allocates ~10 GiB outside the capped allocator, so
    # this fine-tune held 15.9 GiB under a 7 GiB cap beside the training and prod (2026-10-06)
    import src.assoc as A_
    A_.FUSED_SCAN = True
    for mx_ in m.mix:
        if hasattr(mx_, "need_sent"): mx_.need_sent = False
    ROW = a.row
    tr = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r").reshape(-1, ROW)
    trm = np.memmap(f"{a.data}/train.mask", dtype=np.uint8, mode="r").reshape(-1, ROW)
    va = np.memmap(f"{a.data}/val.bin", dtype=np.uint16, mode="r").reshape(-1, ROW)
    vam = np.memmap(f"{a.data}/val.mask", dtype=np.uint8, mode="r").reshape(-1, ROW)
    RH = []                                            # (name, train rows, train mask, val rows, val mask)
    for d_ in [d_ for d_ in a.rehearse.split(",") if d_]:
        RH.append((os.path.basename(d_), *(np.memmap(f"{d_}/{f_}", dtype=dt_, mode="r").reshape(-1, ROW)
                                           for f_, dt_ in (("train.bin", np.uint16), ("train.mask", np.uint8), ("val.bin", np.uint16), ("val.mask", np.uint8)))))
    steps = int(a.epochs * len(tr) / a.batch)
    say(f"init {a.init}  {sum(p.numel() for p in m.parameters())/1e6:.1f}M params  {len(tr):,} train rows  "
        f"{steps:,} steps of {a.batch}x{ROW} = {steps * a.batch * ROW / 1e6:.1f}M tokens ({a.epochs} epochs)  lr {a.lr}")

    if a.optim == "psf":
        from prodigyplus import ProdigyPlusScheduleFree
        # empty groups skipped: a transformer has no spectral parameters, and the package fails on an empty group
        opt = ProdigyPlusScheduleFree([{"params": g["params"]} for g in m.param_groups(lr=a.lr, spectral_lr=a.spectral_lr)
                                       if g["params"]], prodigy_steps=a.prodigy_steps)
        opt.train()
    else:
        opt = Fabric(m.param_groups(lr=a.lr, spectral_lr=a.spectral_lr), lr=a.lr, fused=bool(a.fused))
    base = [g["lr"] for g in opt.param_groups]

    trunk = torch.compile(m.trunk) if a.compile else m.trunk

    def loss_on(x, msk):
        # the trunk, then the chunked head: at most one slice of the logits exists at a time, instead of
        # the whole B x T x V block in fp32. Masked positions are -100, which cross-entropy skips; m.loss
        # divides by every position, so rescale to the mean over the answer tokens.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = trunk(x[:, :-1])
        y = x[:, 1:].clone(); y[msk[:, 1:] == 0] = -100
        valid = (y != -100).sum().clamp(min=1)
        return m.loss(h, y, chunks=4) * (y.numel() / valid)

    @torch.no_grad()
    def evaluate(va=va, vam=vam, n_rows=256):
        if a.optim == "psf": opt.eval()
        m.eval(); tot = 0.0; n = 0
        for i in range(0, min(len(va), n_rows), a.batch):
            x = torch.from_numpy(va[i:i + a.batch].astype(np.int64)).cuda()
            k = torch.from_numpy(vam[i:i + a.batch].astype(np.int64)).cuda()
            tot += float(loss_on(x, k)) * x.shape[0]; n += x.shape[0]
        m.train()
        if a.optim == "psf": opt.train()
        return tot / n

    rng = np.random.default_rng(a.seed)
    best = evaluate(); best_step = 0
    say(f"step 0: answer loss on held-out pairs {best:.4f}")
    r0 = [evaluate(r[3], r[4], 64) for r in RH]; fading = []
    if RH: say("rehearsal self-test at step 0: " + ", ".join(f"{r[0]} {v:.4f}" for r, v in zip(RH, r0)))
    nrh = [0] * len(RH)
    torch.save({"cfg": sd["cfg"], "model": m.state_dict(), "step": 0, "sft_from": a.init}, f"{a.out}/ckpt.pt")
    tl = []; tt = time.time()
    # DIVERGENCE GUARDS (2026-10-14). Two replay fine-tunes under schedule-free Prodigy went to NaN (one at step ~3,000, one at
    # ~31,400 after 3.7 h -- its held-out loss 1.04 at 20k, 1.33 at 30k, then NaN) and trained on NaN to the end, the last good
    # weights overwritten at every evaluation. Now: a step whose loss is not finite is skipped; the gradient is clipped to the
    # 90th percentile of its own recent norms (AutoClip, Seetharaman et al. 2020 -- no fixed threshold); ckpt.pt keeps the BEST
    # held-out weights; a held-out loss that is not finite or 1.5x the best ends the run with the best kept.
    gn_hist = []; skipped = 0
    for step in range(1, steps + 1):
        if a.optim != "psf":
            f = min(1.0, step / a.warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))))
            for g, b in zip(opt.param_groups, base): g["lr"] = b * f
        src_ = int(rng.integers(0, 1 + len(fading)))  # 0 = the new data; j > 0 = the j-th fading earlier set
        T_, TM_ = (tr, trm) if src_ == 0 else (RH[fading[src_ - 1]][1], RH[fading[src_ - 1]][2])
        if src_: nrh[fading[src_ - 1]] += 1
        idx = rng.integers(0, len(T_), a.batch)
        x = torch.from_numpy(T_[idx].astype(np.int64)).cuda()
        k = torch.from_numpy(TM_[idx].astype(np.int64)).cuda()
        loss = loss_on(x, k)
        if not torch.isfinite(loss):
            skipped += 1; opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        if a.optim != "psf": torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        else:
            lim = float(np.percentile(gn_hist, 90)) if len(gn_hist) >= 100 else float("inf")
            gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), lim))
            if not math.isfinite(gn): skipped += 1; opt.zero_grad(set_to_none=True); continue
            gn_hist.append(gn); gn_hist = gn_hist[-1000:]
        opt.step()
        tl.append(float(loss.detach()))
        if step % 100 == 0:
            dt = time.time() - tt; tt = time.time()
            dl_ = f"  d {opt.param_groups[0].get('d', float('nan')):.2e}" if a.optim == "psf" else ""   # Prodigy's own step size (2026-10-14: logged to see the drift before a divergence)
            say(f"{step:,}/{steps:,}  answer loss {np.mean(tl[-100:]):.4f}{dl_}  {100 * a.batch * ROW / dt / 1e3:.0f} ktok/s  "
                f"eta {(steps - step) * dt / 100 / 60:.0f}m")
        if step % a.eval_every == 0 or step == steps:
            v = evaluate(); say(f"step {step:,}: answer loss on held-out pairs {v:.4f}  (best {best:.4f} at {best_step:,}; skipped steps {skipped})")
            if RH:
                rv = [evaluate(r[3], r[4], 64) for r in RH]; fading = [j for j in range(len(RH)) if rv[j] > r0[j]]
                say("  rehearsal self-test: " + ", ".join(f"{r[0]} {x_:.4f} (start {y_:.4f}{', FADING' if x_ > y_ else ''}; {n_:,} rows rehearsed)"
                                                         for r, x_, y_, n_ in zip(RH, rv, r0, nrh)))
            if not math.isfinite(v) or v > 1.5 * best:
                say(f"DIVERGING at step {step:,} (held-out {v:.4f} vs best {best:.4f}): stopping; ckpt.pt keeps step {best_step:,}")
                break
            if v < best:
                best, best_step = v, step
                if a.optim == "psf": opt.eval()          # save the averaged weights
                torch.save({"cfg": sd["cfg"], "model": m.state_dict(), "step": step, "sft_from": a.init}, f"{a.out}/ckpt.pt")
                if a.optim == "psf": opt.train()
    say(f"done: best held-out answer loss {best:.4f} at step {best_step:,} (ckpt.pt); {skipped} non-finite steps skipped")


if __name__ == "__main__":
    main()
