"""A QUALITY MEASURE THAT CANNOT BE FOOLED BY SYMBOL SOUP.

The bits-per-byte judge was necessary and is not sufficient. It catches loops, which are too predictable, but
it scored 'C ...\nC ...\nS ...\nG ...' at 0.96 and 'Nn....  ª ..  ª ..' at 1.14 -- both inside the band real
text occupies -- because unusual junk is about as surprising as prose. Every "blocks in band" number reported
for long generation on 2026-09-22 is unreliable for that reason.

So three things must hold together, and the weakest decides:
  WORDS      the share of whitespace-separated pieces that are ordinary alphabetic words of 2+ letters.
             Real English text sits near 0.85; 'C ... G ...' sits near 0.0.
  SURPRISE   bits per byte under GPT-2, inside the band real text occupies. Catches loops.
  VARIETY    distinct 4-grams as a share of 4-grams. Catches repetition the other two miss.
A block counts as TEXT only if words >= 0.60 and 0.954 <= bpb <= 1.188 and variety >= 0.50.
The same three are measured on real held-out text first, so the thresholds are anchored and not invented."""
import sys, math, re, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(16.0)
import numpy as np, torch
from tokenizers import Tokenizer
from src.bench import Ours, GPT2
g = GPT2(); tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
WORD = re.compile(r"^[A-Za-z][A-Za-z'’-]{1,}$")
def judge(txt):
    idg = g.tok.encode(txt)[:1000]
    if len(idg) < 32: return float("nan")
    x = torch.tensor([idg], device="cuda")
    with torch.no_grad(): lg = g.m(x).logits.float().log_softmax(-1)
    return float(-lg[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).sum()) / (math.log(2) * max(1, len(g.tok.decode(idg[1:]).encode())))
def metrics(txt):
    w = txt.split()
    words = sum(1 for t in w if WORD.match(t)) / max(1, len(w))
    four = [tuple(w[i:i+4]) for i in range(max(0, len(w) - 3))]
    var = len(set(four)) / max(1, len(four))
    return words, judge(txt[:4000]), var
def is_text(m): return m[0] >= 0.60 and 0.954 <= m[1] <= 1.188 and m[2] >= 0.50
a = np.memmap("/mnt/kv_cache/tokfull_v8192/dclm.bin", dtype=np.uint16, mode="r")
print("REAL HELD-OUT TEXT, to anchor the thresholds:")
for i in (10**7, 3*10**7, 5*10**7):
    t = tk.decode([int(v) for v in a[i:i + 1500]])
    mm = metrics(t)
    print(f"   words {mm[0]:.2f}  bpb {mm[1]:.2f}  variety {mm[2]:.2f}   -> {'TEXT' if is_text(mm) else 'fails'}")
print()
for ck, lab in ((sys.argv[1], sys.argv[2]),):
    m = Ours(ck, "/mnt/kv_cache/tok_v8192.json")
    for t in (0.8, 0.85, 0.9):
        out = m.generate(m.encode("The history of the city begins"), 32768, temp=t, seed=0)
        segs = [m.decode(out[s:s + 4096]) for s in range(0, len(out), 4096)]
        mm = [metrics(s) for s in segs]
        ok = [is_text(x) for x in mm]
        print(f"{lab} temp {t}: {sum(ok)}/8 blocks are TEXT", flush=True)
        print("    words  " + " ".join(f"{x[0]:5.2f}" for x in mm))
        print("    bpb    " + " ".join(f"{x[1]:5.2f}" for x in mm))
        print("    variety" + " ".join(f"{x[2]:5.2f}" for x in mm), flush=True)
