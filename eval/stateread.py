"""IS THE STATE-READ PATH WORTH ITS DATA MOVEMENT?

The three attention layers can query the carrier's per-chunk memory ("read_state"). That path costs a projection
and a matmul per attention layer, and -- the expensive part -- it forces every carrier layer to write its state
at every chunk boundary to VRAM (~25 MB per layer per step at an 8,192-token window), which is also what stops
the carrier moving to the fused flash-linear-attention kernel (measured 1.50-1.67x faster for identical maths),
because that kernel never materialises those states.

Its gates have closed to sigmoid(rg) ~ 0.003 and it changes no induction item. This measures what it is worth
where it matters: bits per byte on held-out text at several windows, and exact recall at a distance, with the
read on and with it switched off exactly (rg = -30, sigmoid 1e-13). If both are unchanged, it can go.
usage: eval/stateread.py [checkpoint]"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(14.0)
import numpy as np, torch
from tokenizers import Tokenizer
from src.bench import Ours
from src.common import eval_bpb
from src.train import loaders

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
m = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
bins, ks, ps = loaders("/mnt/kv_cache/tokfull_v8192")
readers = [mx for mx in m.m.mix if getattr(mx, "read_state", 0)]
saved = [mx.rg.data.clone() for mx in readers]
print(f"{len(readers)} attention layers read the carrier state; their gates sigmoid(rg) = "
      + ", ".join(f"{torch.sigmoid(r).item():.4f}" for r in saved))
def bpb(T):
    b, _ = eval_bpb(m.m, tk, bins, ks, ps, np.random.default_rng(31), n_batches=8, B=4, T=T)
    return b
print(f"\n{'window':>8s} {'read on':>10s} {'read off':>10s} {'cost of removing':>17s}")
for T in (1024, 2048, 8192):
    on = bpb(T)
    for mx in readers: mx.rg.data.fill_(-30.0)
    off = bpb(T)
    for mx, r in zip(readers, saved): mx.rg.data.copy_(r)
    print(f"{T:>8,d} {on:10.4f} {off:10.4f} {off - on:+17.4f} bits/byte", flush=True)
