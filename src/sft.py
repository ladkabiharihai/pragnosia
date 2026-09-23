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
    ap.add_argument("--cap", type=float, default=24.0)
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
    ROW = a.row
    tr = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r").reshape(-1, ROW)
    trm = np.memmap(f"{a.data}/train.mask", dtype=np.uint8, mode="r").reshape(-1, ROW)
    va = np.memmap(f"{a.data}/val.bin", dtype=np.uint16, mode="r").reshape(-1, ROW)
    vam = np.memmap(f"{a.data}/val.mask", dtype=np.uint8, mode="r").reshape(-1, ROW)
    steps = int(a.epochs * len(tr) / a.batch)
    say(f"init {a.init}  {sum(p.numel() for p in m.parameters())/1e6:.1f}M params  {len(tr):,} train rows  "
        f"{steps:,} steps of {a.batch}x{ROW} = {steps * a.batch * ROW / 1e6:.1f}M tokens ({a.epochs} epochs)  lr {a.lr}")

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
    def evaluate():
        m.eval(); tot = 0.0; n = 0
        for i in range(0, min(len(va), 256), a.batch):
            x = torch.from_numpy(va[i:i + a.batch].astype(np.int64)).cuda()
            k = torch.from_numpy(vam[i:i + a.batch].astype(np.int64)).cuda()
            tot += float(loss_on(x, k)) * x.shape[0]; n += x.shape[0]
        m.train(); return tot / n

    rng = np.random.default_rng(a.seed)
    say(f"step 0: answer loss on held-out pairs {evaluate():.4f}")
    tl = []; tt = time.time()
    for step in range(1, steps + 1):
        f = min(1.0, step / a.warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))))
        for g, b in zip(opt.param_groups, base): g["lr"] = b * f
        idx = rng.integers(0, len(tr), a.batch)
        x = torch.from_numpy(tr[idx].astype(np.int64)).cuda()
        k = torch.from_numpy(trm[idx].astype(np.int64)).cuda()
        loss = loss_on(x, k)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        tl.append(float(loss.detach()))
        if step % 100 == 0:
            dt = time.time() - tt; tt = time.time()
            say(f"{step:,}/{steps:,}  answer loss {np.mean(tl[-100:]):.4f}  {100 * a.batch * ROW / dt / 1e3:.0f} ktok/s  "
                f"eta {(steps - step) * dt / 100 / 60:.0f}m")
        if step % a.eval_every == 0 or step == steps:
            v = evaluate(); say(f"step {step:,}: answer loss on held-out pairs {v:.4f}")
            torch.save({"cfg": sd["cfg"], "model": m.state_dict(), "step": step, "sft_from": a.init}, f"{a.out}/ckpt.pt")
    say(f"done: {steps:,} steps, final held-out answer loss {evaluate():.4f}")


if __name__ == "__main__":
    main()
