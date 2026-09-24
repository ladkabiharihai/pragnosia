"""VISION GOAL 7, THE WAY THIS ARCHITECTURE IS BUILT FOR: LEARNING BY READING, NOT BY TRAINING.

Continual learning by fine-tuning failed for both models (eval/continual.py): neither kept what it knew, and the
carrier did not even learn the new facts. But a model with a fixed-size state and no context limit has another
way to learn: read the new facts, and answer from its memory. The weights never change, so nothing already
known can be forgotten -- by construction, not by a regulariser.

The same 120 invented facts as eval/continual.py, one sentence each, placed in the context; then a question in
a phrasing that was not in the context. Four-way, chance 25%. Measured at growing numbers of facts, and for
the largest set by how far back the fact sits. GPT-2 is measured wherever the facts fit in its 1,024 positions,
and marked where they do not.
usage: eval/incontext.py [checkpoint]"""
import sys, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import numpy as np, torch
from src.bench import Ours, GPT2

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
rng = random.Random(1234)
SYL = ["vel", "dra", "osk", "arth", "mir", "tan", "quel", "lin", "zor", "va", "bre", "nth", "kal", "iso", "rum",
       "pel", "dor", "gha", "sen", "tri", "wex", "ulm", "fae", "rok", "yst", "cor", "nib", "sha", "lom", "dex"]
seen = set()
def fresh(n):
    while True:
        w = "".join(rng.choice(SYL) for _ in range(n)).capitalize()
        if w not in seen and len(w) >= 5: seen.add(w); return w
countries = [fresh(3) for _ in range(60)]; capitals = [fresh(2) for _ in range(60)]
machines = [fresh(2) for _ in range(60)]; people = [f"{fresh(2)} {fresh(2)}" for _ in range(60)]
FACTS = []                                   # (sentence placed in context, question, answer, distractors)
for i in range(60):
    FACTS.append((f"The capital of {countries[i]} is {capitals[i]}.",
                  f"Question: which city is the capital of {countries[i]}? Answer:", f" {capitals[i]}",
                  [f" {o}" for o in rng.sample([x for x in capitals if x != capitals[i]], 3)]))
    FACTS.append((f"The {machines[i]} engine was invented by {people[i]}.",
                  f"Question: who invented the {machines[i]} engine? Answer:", f" {people[i]}",
                  [f" {o}" for o in rng.sample([x for x in people if x != people[i]], 3)]))
order = list(range(len(FACTS))); random.Random(9).shuffle(order)
# HARD mode (argv[2] == "hard"): every wrong option is ANOTHER ANSWER FROM THE SAME CONTEXT. Measured 2026-09-24:
# in the default mode the wrong options were drawn from all 60 names, so with 10 facts in context most of them
# had never appeared and "pick the name that was in the text" scored well without any lookup. Here only lookup
# scores: every option was in the text.
HARD = len(sys.argv) > 2 and sys.argv[2] == "hard"
def hard(sub):
    out = []
    for f in sub:
        pool = [g[2] for g in sub if g is not f and ("capital" in g[1]) == ("capital" in f[1])]
        out.append((f[0], f[1], f[2], rng.sample(pool, 3) if len(pool) >= 3 else f[3]))
    return out


def ask(model, enc, context_ids, q, ans, wrongs, max_ctx=None):
    lps = []
    for c in [ans] + wrongs:
        ids = context_ids + enc(" " + q) + enc(c)
        if max_ctx and len(ids) > max_ctx: return None
        lg = model.logits_last(ids, 1).float().log_softmax(-1)
        nc = min(len(enc(c)), lg.shape[1] - 1)
        y = torch.tensor(ids[-nc:], device=lg.device)
        lps.append(float(lg[0, -nc - 1:-1].gather(-1, y.unsqueeze(-1)).sum()))
    return int(np.argmax(lps)) == 0


models = [("Carrier", Ours(CK, "/mnt/kv_cache/tok_v8192.json"), None), ("GPT-2 124M", GPT2(), 1024)]
print("Facts are read, not trained on: the weights never change, so nothing known can be forgotten."
      + ("  HARD: every wrong option is another answer from the same context." if HARD else "") + "\n")
print(f"{'facts in context':<18s}" + "".join(f"{lab:>22s}" for lab, _, _ in models))
for n in (10, 30, 60, 120):
    sub = [FACTS[i] for i in order[:n]]
    if HARD: sub = hard(sub)
    row = f"{n:<18d}"
    for lab, m, cap in models:
        enc = m.encode if hasattr(m, "encode") else m.tok.encode
        ctx = enc(" ".join(f[0] for f in sub))
        res = [ask(m, enc, ctx, f[1], f[2], f[3], cap) for f in sub]
        if any(r is None for r in res):
            row += f"{'does not fit (' + str(len(ctx)) + ' tok)':>22s}"
        else:
            row += f"{100 * np.mean(res):>13.1f}% ({len(ctx):>4d} tok)"
    print(row, flush=True)
print("\nNo facts in context at all (what each model guesses cold): ", end="")
for lab, m, cap in models:
    enc = m.encode if hasattr(m, "encode") else m.tok.encode
    print(f"{lab} {100 * np.mean([ask(m, enc, [], f[1], f[2], f[3], cap) for f in FACTS]):.1f}%   ", end="")
print("(chance 25%)")
lab, m, _ = models[0]
sub = [FACTS[i] for i in order]
if HARD: sub = hard(sub)
ctx = m.encode(" ".join(f[0] for f in sub))
res = np.array([ask(m, m.encode, ctx, f[1], f[2], f[3]) for f in sub])
print(f"\nCarrier with all 120 facts in context ({len(ctx)} tokens), accuracy by how far back the fact sits:")
for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120)):
    print(f"   facts {lo + 1:>3d}-{hi:<3d} (the {'most recent' if lo == 90 else 'earliest' if lo == 0 else 'middle'} quarter): "
          f"{100 * res[lo:hi].mean():.1f}%")
