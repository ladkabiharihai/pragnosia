"""NEEDLE IN A HAYSTACK: can the model retrieve one specific fact from far back in its context?

This is the test a technical reviewer runs first for "context rot", and it measures something different from
bits per byte improving with context. Bits per byte can fall because distant text helps prediction in bulk;
retrieval asks whether ONE planted fact can be recovered exactly. eval/incontext.py suggested the carrier's
exact recall dies near its 512-token attention window, so this maps where.

One sentence -- "The secret code for <place> is <number>." -- is planted at a chosen depth in real held-out
prose, and after the haystack the model is asked for the code. Four-way: the true code against three other
four-digit numbers, scored by total log-probability. 40 needles per cell, chance 25%. Both models, GPT-2 only
where the whole thing fits in its 1,024 positions.
usage: eval/needle.py [checkpoint]"""
import sys, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(14.0)
import numpy as np, torch
from tokenizers import Tokenizer
from src.bench import Ours, GPT2

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
hay = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
HAY = tk.decode([int(v) for v in hay[3 * 10**6: 3 * 10**6 + 60000]])
PLACES = ["the harbour", "the old mill", "the north gate", "the library", "the tower", "the orchard",
          "the bridge", "the chapel", "the market", "the lighthouse"]
N = 40


def trial(model, enc, distance, seed, cap):
    r = random.Random(seed)
    place = r.choice(PLACES); code = r.randint(1000, 9999)
    wrongs = set()
    while len(wrongs) < 3:
        w = r.randint(1000, 9999)
        if w != code: wrongs.add(w)
    needle = f" The secret code for {place} is {code}."
    q = f" Question: what is the secret code for {place}? Answer: The secret code for {place} is"
    start = r.randint(0, 40000)
    filler = enc(HAY[start:start + distance * 8])[:distance]      # distance = tokens of prose AFTER the needle
    lead = enc(HAY[start + distance * 8: start + distance * 8 + 800])[:120]
    ctx = lead + enc(needle) + filler
    lps = []
    for c in [code] + sorted(wrongs):
        ids = ctx + enc(q) + enc(f" {c}")
        if cap and len(ids) > cap: return None
        lg = model.logits_last(ids, 1).float().log_softmax(-1)
        nc = min(len(enc(f" {c}")), lg.shape[1] - 1)
        y = torch.tensor(ids[-nc:], device=lg.device)
        lps.append(float(lg[0, -nc - 1:-1].gather(-1, y.unsqueeze(-1)).sum()))
    return int(np.argmax(lps)) == 0


models = [("Carrier", Ours(CK, "/mnt/kv_cache/tok_v8192.json"), None), ("GPT-2 124M", GPT2(), 1024)]
print(f"tokens of prose between the needle and the question; {N} needles per cell; chance 25%\n")
print(f"{'distance':>10s}" + "".join(f"{lab:>14s}" for lab, _, _ in models))
for dist in (0, 64, 256, 512, 768, 1024, 2048, 4096, 8192, 16384):
    row = f"{dist:>10,d}"
    for lab, m, cap in models:
        enc = m.encode if hasattr(m, "encode") else m.tok.encode
        res = [trial(m, enc, dist, s, cap) for s in range(N)]
        row += f"{'—':>14s}" if any(x is None for x in res) else f"{100 * np.mean(res):>13.1f}%"
    print(row, flush=True)
print("\nThe carrier's attention layers see 512 tokens directly; past that, recall must come from the carrier state.")
