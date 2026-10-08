"""LONG GENERATION: does min-p sampling hold the horizon? (2026-10-06; user: "also long generation")
At plain temperature 0.9, run85's 368k checkpoint left the band of real text after ~12k tokens on every one of three prompts
(runs of non-breaking spaces, newlines, dashes) and two prompts started too surprising (eval/nine_r85_s368000_v2.txt).
Judged exactly as eval/nine.py judges (GPT-2's bits per byte on each 4,096-token segment, band = 10-90% of real held-out
text). Each (temperature, min-p) setting is first scored on 2,048 tokens from three prompts; the settings closest to real
text are then run to 32,768 tokens from the same three prompts. The setting is CHOSEN by matching real text's statistics,
not set by hand. usage: eval/longgen.py <ckpt>"""
import os, sys, math, collections, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(6.0)
import numpy as np, torch
import src.bench as SB
from src.bench import Ours, GPT2
ours = Ours(sys.argv[1], "/mnt/kv_cache/tok_v8192.json"); gpt2 = GPT2()
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
def score(txt):
    idg = gpt2.tok.encode(txt)[:1000]
    if len(idg) < 32: return float("nan")
    x = torch.tensor([idg], device="cuda")
    with torch.no_grad(): lg = gpt2.m(x).logits.float().log_softmax(-1)
    nats = float(-lg[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).sum())
    return nats / (math.log(2) * max(1, len(gpt2.tok.decode(idg[1:]).encode())))
a = np.memmap("/mnt/kv_cache/tokfull_v8192/dclm.bin", dtype=np.uint16, mode="r")
band = [score(tk.decode([int(t) for t in a[i:i + 1500]])) for i in range(10**7, 13 * 10**7, 10**7)]
lo, hi = np.nanpercentile(band, [10, 90]); mid = float(np.nanmean(band))
print(f"band: real text {mid:.3f} (10-90%: {lo:.3f}-{hi:.3f}, {len(band)} samples)  {sys.argv[1]}", flush=True)
PROMPTS = ["The history of the city begins", "In this paper we study", "The river rises in the mountains"]
res = []
for mp in (0.0, 0.05, 0.1):
    SB.MIN_P = mp
    for t in (0.8, 0.85, 0.9, 1.0):
        s = float(np.nanmean([score(ours.decode(ours.generate(ours.encode(p), 2048, temp=t, seed=1))) for p in PROMPTS]))
        res.append((abs(s - mid), t, mp, s)); print(f"  temp {t:4.2f} min-p {mp:4.2f}: {s:.3f}", flush=True)
res.sort()
for _, t, mp, s0 in res[:3]:
    SB.MIN_P = mp; hz = []; filler = []
    for p in PROMPTS:
        out = ours.generate(ours.encode(p), 32768, temp=t, seed=1); h = 0; segs = []
        for st in range(0, len(out), 4096):
            sc = score(ours.decode(out[st:st + 4096])[:4000]); segs.append(sc)
            if lo <= sc <= hi and h == st: h = st + 4096
        top = max(collections.Counter(out[st:st + 4096]).most_common(1)[0][1] / 4096 for st in range(0, len(out), 4096))
        hz.append(h); filler.append(top)
        print(f"    temp {t} min-p {mp} {p[:24]!r}: horizon {h:,}; segments " + " ".join(f"{x:.2f}" for x in segs)
              + f"; worst commonest-token share {100 * top:.0f}%", flush=True)
    print(f"  temp {t} min-p {mp}: HORIZON median {int(np.median(hz)):,} ({', '.join(f'{x:,}' for x in hz)}); in band at 2k: {s0:.3f}", flush=True)
