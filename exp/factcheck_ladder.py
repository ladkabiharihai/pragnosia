"""GENERATION QUALITY AND FACTUAL ACCURACY ACROSS EVERY RUNG OF THE LADDER.

HOW A BASE MODEL SHOULD BE FACT-CHECKED. These models have no instruction tuning, so asking a question and
grading the free text measures prompt-following, not knowledge. The standard base-model method is used instead:
score the log-probability the model assigns to each candidate completion of the same stem and take the argmax.
That isolates what the weights actually encode. Chance is 25% with four options, and the Wilson interval is
reported so a 4-point gap on 48 items is not mistaken for a difference.

Scoring is the MEAN log-probability over the completion tokens, not the sum, so a longer correct answer is not
penalised for being longer. Distractors are drawn from the same category as the answer (a city for a city, an
element for an element), because a model can pick the right answer for the wrong reason when the wrong options
are not even the right kind of thing.

Free generation is also shown at low temperature, where a base model is most likely to state facts rather than
wander, and at the temperature we ship.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("factcheck_ladder")
cap_gpu(float(os.environ.get("CAP", "7")))
import numpy as np, torch
import core as C
from tokenizers import Tokenizer

DEV = C.DEV; TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D = "/mnt/kv_cache/pragnosia_data"
MODELS = [("125M", f"{D}/p124f.pt"), ("191M", f"{D}/p400f_before_grow2.pt"),
          ("279M", f"{D}/p400f_before_grow3.pt"), ("405M", f"{D}/p400f_405M_snap.pt")]

from facts_set import FACTS

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

@torch.no_grad()
def score(m, cfg, stem, comp):
    si = TOK.encode(stem).ids; ci = TOK.encode(comp).ids
    x = torch.tensor([si + ci], device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp = m(x[:, :-1]).float().log_softmax(-1)
    return float(lp[0, -len(ci):].gather(-1, x[0, 1:][-len(ci):].unsqueeze(-1)).mean())

GEN_PROMPTS = [
 "The capital of France is Paris. The capital of Germany is",
 "Water boils at 100 degrees Celsius and freezes at",
 "The planets of the solar system, in order from the Sun, are",
]

print(f"{'model':>7s}{'params':>9s}{'facts correct':>16s}{'accuracy':>11s}{'95% CI':>18s}   (chance 25%)")
print("-" * 76)
results = {}
for tag, path in MODELS:
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=2048, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    P = sum(p.numel() for p in m.parameters())
    ok = []
    for stem, right, wrongs in FACTS:
        cands = [right] + wrongs
        sc = [score(m, cfg, stem, c) for c in cands]
        ok.append(int(np.argmax(sc) == 0))
    k, n = sum(ok), len(ok)
    lo, hi = wilson(k, n)
    results[tag] = (P, k, n, ok)
    print(f"{tag:>7s}{P/1e6:>8.1f}M{f'{k}/{n}':>16s}{100*k/n:>10.1f}%"
          f"{f'[{100*lo:.1f}, {100*hi:.1f}]':>18s}", flush=True)
    del m; torch.cuda.empty_cache()

# PAIRED TEST. Every model answers the SAME 48 items, so comparing two independent Wilson intervals throws
# away the pairing and is far less powerful than it needs to be. McNemar looks only at the items where the two
# models DISAGREE, which is the only place the evidence lives.
print("\n" + "=" * 76)
print("  PAIRED COMPARISON vs the 125M (McNemar, only the items where the two disagree)")
print("=" * 76)
base_ok = results["125M"][3]
for tag in ("191M", "279M", "405M"):
    o = results[tag][3]
    b = sum(1 for x, y in zip(base_ok, o) if x == 0 and y == 1)   # 125M wrong, this model right
    c = sum(1 for x, y in zip(base_ok, o) if x == 1 and y == 0)   # 125M right, this model wrong
    if b + c == 0:
        print(f"  {tag}: no disagreements"); continue
    chi = (abs(b - c) - 1) ** 2 / (b + c)                          # continuity-corrected
    from math import erfc, sqrt
    p = erfc(sqrt(chi / 2))
    print(f"  {tag}: gained {b}, lost {c} against the 125M   chi2 {chi:.2f}  p {p:.4f}   "
          f"{'SIGNIFICANT' if p < 0.05 else 'not significant'} at p<0.05")

print("\n  Wilson intervals overlap heavily at n=48, so treat differences here as suggestive unless a pair is")
print("  clearly separated. What the interval rules out is the strong claim, not the weak one.")

print("\n" + "=" * 96)
print("  FACT-BEARING GENERATION, greedy (temp 0) and shipped sampling (temp 0.9)")
print("=" * 96)
for tag, path in MODELS:
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=2048, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    print(f"\n########## {tag} ({results[tag][0]/1e6:.0f}M) ##########", flush=True)
    for p in GEN_PROMPTS:
        ids = TOK.encode(p).ids
        for lbl, kw in [("greedy", dict(greedy=True)), ("t=0.9", dict(temp=0.9, top_p=0.95))]:
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = m.generate(torch.tensor([ids], device=DEV), 45, **kw)
            print(f"  [{lbl:>6s}] {p!r}\n            -> {TOK.decode([int(t) for t in out])[:210]!r}", flush=True)
    del m; torch.cuda.empty_cache()
