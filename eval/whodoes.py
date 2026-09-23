"""WHO IS DOING THE WORK, THE CARRIER OR THE ATTENTION? Every layer here is residual, so silencing one by
zeroing its output projection turns it into the identity and leaves everything else untouched. The rise in
bits per byte is then that layer's contribution, in the only currency that matters.

Three questions: what do the 3 attention layers contribute together, what do the 9 carrier layers contribute
together, and what does each single layer contribute on its own."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(14.0)
import numpy as np, torch
from src.bench import Ours
from src.common import eval_bpb
from src.train import loaders
from tokenizers import Tokenizer
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
m = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
bins, ks, ps = loaders("/mnt/kv_cache/tokfull_v8192")
T = 2048
def bpb():
    b, _ = eval_bpb(m.m, tk, bins, ks, ps, np.random.default_rng(99), n_batches=6, B=4, T=T)
    return b
def kinds():
    return [("attn" if type(mx).__name__ == "Attn" else "carrier") for mx in m.m.mix]
K = kinds()
saved = [mx.o.weight.data.clone() for mx in m.m.mix]
base = bpb()
print(f"the whole model: {base:.4f} bits per byte at {T} tokens")
n_par = {}
for i, mx in enumerate(m.m.mix):
    n_par.setdefault(K[i], 0)
    n_par[K[i]] += sum(p.numel() for p in mx.parameters())
tot = sum(p.numel() for p in m.m.parameters())
print(f"parameters: carrier {n_par['carrier']/1e6:.1f}M, attention {n_par['attn']/1e6:.1f}M, "
      f"everything else {(tot - sum(n_par.values()))/1e6:.1f}M of {tot/1e6:.1f}M total\n")
for tag in ("attn", "carrier"):
    for i, mx in enumerate(m.m.mix):
        if K[i] == tag: mx.o.weight.data.zero_()
    b = bpb()
    for i, mx in enumerate(m.m.mix): mx.o.weight.data.copy_(saved[i])
    n = sum(1 for k in K if k == tag)
    print(f"  silence all {n} {tag:<8s} layers: {b:8.4f}   (+{b-base:.4f})", flush=True)
print()
for i, mx in enumerate(m.m.mix):
    mx.o.weight.data.zero_()
    b = bpb()
    mx.o.weight.data.copy_(saved[i])
    print(f"  silence layer {i:2d} ({K[i]:<7s}):      {b:8.4f}   (+{b-base:.4f})", flush=True)
