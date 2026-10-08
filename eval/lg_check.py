"""LONG GENERATION WITH THE DRY PENALTY ACTUALLY APPLIED (2026-10-08). The eager decode (every recall model) sampled plain
until today, so the final scorecard's "DRY 0.4" row was plain sampling. Here: 16,384 tokens from each of three prompts at
temperature 0.85 with DRY 0.4 (the strength real text's own 4-gram repeat rate chose on 368k, eval/longgen3.py), GPT-2's
bits per byte per 4,096-token segment (real held-out text 0.95-1.19), and collapsed segments (< 0.5: loops).
usage: eval/lg_check.py <ckpt>[@k] ...    @k: recall k at generation (0 = off; default: the checkpoint's own)"""
import os, sys, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(5.0)
import numpy as np, torch
import src.bench as SB
from src.bench import Ours, GPT2
import src.assoc as A
gpt2 = GPT2(); SB.DRY_MULT = float(os.environ.get("DRY", "0.4")); T = float(os.environ.get("TEMP", "0.85"))
N = int(os.environ.get("NGEN", "16384"))
PROMPTS = ["The history of the city begins", "In this paper we study", "The river rises in the mountains"]
def score(txt):
    idg = gpt2.tok.encode(txt)[:1000]
    x = torch.tensor([idg], device="cuda")
    with torch.no_grad(): lg = gpt2.m(x).logits.float().log_softmax(-1)
    return float(-lg[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).sum()) / (math.log(2) * max(1, len(gpt2.tok.decode(idg[1:]).encode())))
for spec in sys.argv[1:]:
    ck, k = (spec.split("@") + [None])[:2]
    o = Ours(ck, "/mnt/kv_cache/tok_v8192.json"); A.FUSED_SCAN = True
    for mx in o.m.mix:
        if hasattr(mx, "need_sent"): mx.need_sent = False
    if k is not None: o.m.set_recall(int(k), span=0)
    segs_all = []
    for i, p in enumerate(PROMPTS):
        out = o.generate(o.encode(p), N, temp=T, seed=i)
        segs = [score(o.decode(out[s:s + 4096])) for s in range(0, len(out), 4096)]
        segs_all += segs
        print(f"  {os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} recall {getattr(o.m, 'recall_k', 0)} {p[:20]!r}: "
              + " ".join(f"{s:.2f}" for s in segs), flush=True)
    s = np.array(segs_all)
    print(f"{spec}: temp {T} DRY {SB.DRY_MULT}: {int(((s > 0.95) & (s < 1.19)).sum())}/{len(s)} segments in band, "
          f"{int((s < 0.5).sum())}/{len(s)} collapsed, median {np.median(s):.2f}", flush=True)
    del o; torch.cuda.empty_cache()
