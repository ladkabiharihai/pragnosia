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
        m = LM(vocab=8192, d=a.d, layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx).cuda()
        # The carrier is 94% of an uncompiled step (195 ms of 206), because its cross-chunk pass is a python
        # loop issuing many tiny kernels. Compiling fuses them: 40 -> 154 ktok/s measured, a 3.8x speedup.
        fwd = torch.compile(m) if a.compile else m
        P = sum(p.numel() for p in m.parameters())
        car = sum(p.numel() for n, p in m.named_parameters() if "mix." in n)
        base = [a.lr, 1e-3]
        opt = Fabric(m.param_groups(lr=a.lr), lr=a.lr)
        rng = np.random.default_rng(a.seed)
        tok_total = a.steps * a.batch * a.ctx
        r.log(f"{P/1e6:.1f}M params, carrier {100*car/P:.1f}%, "
              f"{tok_total/1e9:.2f}B tokens = {tok_total/P:.1f} tok/param")

        t0 = time.time(); ema = None
        for i in range(a.steps):
            w = min(1.0, (i + 1) / a.warm)
            # cosine over the whole budget: a flat rate lets a model equilibrate at that rate's noise floor,
            # which is what drove the previous project's BPB upward for hundreds of thousands of steps.
            f = min(1.0, max(0.0, (i - a.warm) / max(1, a.steps - a.warm)))
            sched = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f))
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = base[gi] * sched * w
            x = batch(bins, ks, ps, rng, a.batch, a.ctx)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(fwd(x[:, :a.ctx]).reshape(-1, 8192), x[:, 1:a.ctx + 1].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            v = loss.item(); ema = v if ema is None else 0.99 * ema + 0.01 * v
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                r.log(f"{i+1}/{a.steps} loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {(i+1)*a.batch*a.ctx/el/1e3:.0f} ktok/s "
                      f"eta {(a.steps-i-1)*el/(i+1)/60:.0f}m")
            if (i + 1) % a.eval_every == 0:
                # BITS PER BYTE, not loss per token. Per-token loss depends on the vocabulary and is
                # comparable to nothing; bits per byte removes the tokenizer and is the only number that can
                # be set against another model.
                b, nb = eval_bpb(m, tok, bins, ks, ps, np.random.default_rng(1234),
                                 n_batches=8, B=8, T=a.ctx)
                r.log(f"  eval step {i+1}: BPB {b:.4f}  ({nb/1e3:.0f}k bytes)")
            if (i + 1) % a.save_every == 0 or i + 1 == a.steps:
                torch.save({"model": m.state_dict(), "step": i + 1, "loss": ema,
                            "cfg": dict(vocab=8192, d=a.d, layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx)},
                           f"{a.out}/ckpt.pt")
        r.log(f"final loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f}")


if __name__ == "__main__":
    main()
