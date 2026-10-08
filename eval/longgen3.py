"""LONG GENERATION, III: a DRY penalty (only tokens that extend an earlier sequence) whose strength is chosen by real text's own repetition rate (2026-10-06).
eval/longgen.py: run85's 368k checkpoint collapses into loops after a few thousand tokens at plain temperature (GPT-2 scores
segments at 0.02-0.28 bits/byte, real text 0.97-1.21) and min-p makes it worse. A loop is repetition, so the decoder's
penalty on recently used tokens is CHOSEN by matching how often real text repeats itself: the share of positions whose
4-gram already occurred in the previous 4,096 tokens, on 8,192-token spans of held-out text (dclm, wikipedia, arxiv) vs
on 8,192 generated tokens from three prompts. Then the chosen setting is judged as eval/nine.py judges: 32,768 tokens
from three prompts, GPT-2's bits per byte per 4,096-token segment, horizon = tokens before the first segment out of band.
usage: eval/longgen2.py <ckpt>"""
import os, sys, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(6.0)
import numpy as np, torch
import src.bench as SB
from src.bench import Ours, GPT2
from tokenizers import Tokenizer
ours = Ours(sys.argv[1], "/mnt/kv_cache/tok_v8192.json"); gpt2 = GPT2()
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
def score(txt):
    idg = gpt2.tok.encode(txt)[:1000]
    if len(idg) < 32: return float("nan")
    x = torch.tensor([idg], device="cuda")
    with torch.no_grad(): lg = gpt2.m(x).logits.float().log_softmax(-1)
    nats = float(-lg[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).sum())
    return nats / (math.log(2) * max(1, len(gpt2.tok.decode(idg[1:]).encode())))
def rep4(ids, back=4096):
    """share of positions whose 4-gram already occurred within the previous `back` tokens"""
    last = {}; hit = 0
    for i in range(3, len(ids)):
        g = tuple(ids[i - 3:i + 1]); j = last.get(g)
        hit += j is not None and i - j <= back; last[g] = i
    return hit / max(1, len(ids) - 3)
real = []
for src in ("dclm", "wikipedia", "arxiv"):
    a = np.memmap(f"/mnt/kv_cache/tokfull_v8192/{src}.bin", dtype=np.uint16, mode="r")
    for o in (10**7, 3 * 10**7, 5 * 10**7):
        f = np.asarray(a[o:o + 12000]); f = f[f != 8191][:8192].tolist(); real.append(rep4(f))
band = []
a = np.memmap("/mnt/kv_cache/tokfull_v8192/dclm.bin", dtype=np.uint16, mode="r")
band = [score(tk.decode([int(t) for t in a[i:i + 1500]])) for i in range(10**7, 13 * 10**7, 10**7)]
lo, hi = np.nanpercentile(band, [10, 90]); mid = float(np.nanmean(band))
R0 = float(np.median(real))
print(f"real text: 4-gram repeat rate {R0:.3f} (median of {len(real)} spans; {min(real):.3f}-{max(real):.3f}); GPT-2 band "
      f"{mid:.3f} ({lo:.3f}-{hi:.3f})  {sys.argv[1]}", flush=True)
PROMPTS = ["The history of the city begins", "In this paper we study", "The river rises in the mountains"]
rows = []
for t in (0.85, 0.9):
    for fp in (0.0, 0.4, 0.8, 1.6):
        SB.DRY_MULT = fp
        outs = [ours.generate(ours.encode(p), 8192, temp=t, seed=1) for p in PROMPTS]
        rr = float(np.mean([rep4(o) for o in outs]))
        sc = float(np.nanmean([score(ours.decode(o[:2048])) for o in outs]))
        rows.append((abs(math.log(max(rr, 1e-4) / R0)) + abs(sc - mid), t, fp, rr, sc))
        print(f"  temp {t:4.2f} DRY {fp:4.2f}: 4-gram repeat {rr:.3f} (real {R0:.3f}), GPT-2 score at the start {sc:.3f}", flush=True)
rows.sort()
for _, t, fp, rr, sc in rows[:2]:
    SB.DRY_MULT = fp; hz = []
    for p in PROMPTS:
        out = ours.generate(ours.encode(p), 32768, temp=t, seed=1); h = 0; segs = []
        for st in range(0, len(out), 4096):
            s_ = score(ours.decode(out[st:st + 4096])[:4000]); segs.append(s_)
            if lo <= s_ <= hi and h == st: h = st + 4096
        hz.append(h)
        print(f"    temp {t} DRY {fp} {p[:24]!r}: horizon {h:,}; segments " + " ".join(f"{x:.2f}" for x in segs)
              + f"; repeat {rep4(out):.3f}", flush=True)
    print(f"  CHOSEN-BY-DATA temp {t} DRY {fp}: HORIZON median {int(np.median(hz)):,} ({', '.join(f'{x:,}' for x in hz)})", flush=True)
    print(f"  sample: {ours.decode(out[16384:16384 + 120])!r}", flush=True)
