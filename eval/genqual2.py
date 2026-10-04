"""GENERATION QUALITY, SHORT, against GPT-2, on the corrected measure.

The bits-per-byte judge alone put us marginally ahead of GPT-2 at each model's best temperature. It also
scored symbol soup inside the band, so that verdict has to be re-taken with a measure junk cannot pass:
the share of pieces that are ordinary words, the bits-per-byte band, and 4-gram variety, all three at once.
1,024 generated tokens, five prompts, both models, thresholds anchored on real text."""
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
def met(txt):
    w = txt.split()
    words = sum(1 for t in w if WORD.match(t)) / max(1, len(w))
    four = [tuple(w[i:i+4]) for i in range(max(0, len(w) - 3))]
    return words, judge(txt[:4000]), len(set(four)) / max(1, len(four))
PROMPTS = ["The history of the city begins", "In 1847 the railway reached the town, and",
           "Photosynthesis is the process by which", "The report concluded that",
           "She had never seen the sea before, and when"]
a = np.memmap("/mnt/kv_cache/tokfull_v8192/dclm.bin", dtype=np.uint16, mode="r")
ref = [met(tk.decode([int(v) for v in a[i:i + 1500]])) for i in (10**7, 3*10**7, 5*10**7, 7*10**7, 9*10**7)]
R = np.array(ref)
print(f"{'':<22s} {'words':>7s} {'bpb':>7s} {'variety':>8s}   (1,024 generated tokens, mean of 5 prompts)")
print(f"{'REAL HELD-OUT TEXT':<22s} {R[:,0].mean():7.2f} {R[:,1].mean():7.2f} {R[:,2].mean():8.2f}")
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
m = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
NAME = (CK.split("/")[-2] if CK.endswith("/ckpt.pt") else CK.split("/")[-1].removesuffix(".pt"))[:12]   # was a fixed "run44"
for t in (0.8, 0.85, 0.9):
    v = np.array([met(m.decode(m.generate(m.encode(p), 1024, temp=t, seed=1))) for p in PROMPTS])
    print(f"{'ours (' + NAME + ') t=' + str(t):<22s} {v[:,0].mean():7.2f} {v[:,1].mean():7.2f} {v[:,2].mean():8.2f}", flush=True)
for t in (0.8, 0.9, 1.0):
    vs = []
    for p in PROMPTS:
        ids = g.tok.encode(p); x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            o = g.m.generate(x, max_new_tokens=1000 - len(ids), do_sample=True, temperature=t, top_k=0, top_p=1.0,
                             pad_token_id=g.tok.eos_token_id)
        vs.append(met(g.tok.decode(o[0][len(ids):])))
    v = np.array(vs)
    print(f"{'GPT-2 124M t=' + str(t):<22s} {v[:,0].mean():7.2f} {v[:,1].mean():7.2f} {v[:,2].mean():8.2f}", flush=True)
print("\n  a sample from each at its best setting:")
print(f"  ours : {m.decode(m.generate(m.encode(PROMPTS[0]), 80, temp=0.85, seed=1))!r}")
ids = g.tok.encode(PROMPTS[0]); x = torch.tensor([ids], device="cuda")
with torch.no_grad():
    o = g.m.generate(x, max_new_tokens=80, do_sample=True, temperature=0.9, top_k=0, top_p=1.0, pad_token_id=g.tok.eos_token_id)
print(f"  GPT-2: {g.tok.decode(o[0][len(ids):])!r}")
