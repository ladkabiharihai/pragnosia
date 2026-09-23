"""VISION GOAL 7, SECOND HALF: CONTINUOUS LEARNING. Learn something new; keep what was known.

120 facts neither model can have seen -- invented countries with invented capitals, invented machines with
invented inventors -- are written out in three phrasings each and trained on briefly. Before and after, three
things are measured on BOTH models with the identical procedure:
  NEW         four-way accuracy on the invented facts, asked in a phrasing not trained on (chance 25%)
  OLD FACTS   four-way accuracy on the 48 real facts (what was known must survive)
  TEXT        bits per byte on held-out real text (general ability must survive)
Learning without forgetting means NEW rises while OLD FACTS and TEXT hold. The same optimiser, learning rate,
steps and data go to both models; neither is tuned for this.
usage: eval/continual.py [checkpoint]"""
import sys, math, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(20.0)
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from src.bench import Ours, GPT2
from src.facts import FACTS

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
STEPS, LR, BATCH, SEQ = 150, 5e-5, 8, 256

rng = random.Random(1234)
SYL = ["vel", "dra", "osk", "arth", "mir", "tan", "quel", "lin", "zor", "va", "bre", "nth", "kal", "iso", "rum",
       "pel", "dor", "gha", "sen", "tri", "wex", "ulm", "fae", "rok", "yst", "cor", "nib", "sha", "lom", "dex"]
def name(n):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()
seen = set()
def fresh(n):
    while True:
        w = name(n)
        if w not in seen and len(w) >= 5: seen.add(w); return w
countries = [fresh(3) for _ in range(60)]; capitals = [fresh(2) for _ in range(60)]
machines = [fresh(2) for _ in range(60)]; people = [f"{fresh(2)} {fresh(2)}" for _ in range(60)]
TRAIN, TEST = [], []
for i in range(60):
    c, k = countries[i], capitals[i]
    TRAIN += [f"The capital of {c} is {k}.", f"{k} is the capital city of {c}.", f"{c}'s capital, {k}, is its largest city."]
    others = rng.sample([x for x in capitals if x != k], 3)
    TEST.append((f"Question: which city is the capital of {c}? Answer:", f" {k}", [f" {o}" for o in others]))
    mch, p = machines[i], people[i]
    TRAIN += [f"The {mch} engine was invented by {p}.", f"{p} invented the {mch} engine.", f"The inventor of the {mch} engine was {p}."]
    others = rng.sample([x for x in people if x != p], 3)
    TEST.append((f"Question: who invented the {mch} engine? Answer:", f" {p}", [f" {o}" for o in others]))


def mc(model, items, enc):
    right = 0
    for stem, ans, wrongs in items:
        lps = []
        for c in [ans] + wrongs:
            ids = enc(stem) + enc(c)
            lg = model.logits_last(ids, 1).float().log_softmax(-1)
            nc = min(len(enc(c)), lg.shape[1] - 1)
            y = torch.tensor(ids[-nc:], device=lg.device)
            lps.append(float(lg[0, -nc - 1:-1].gather(-1, y.unsqueeze(-1)).sum()))
        right += int(np.argmax(lps)) == 0
    return right / len(items)


tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
held = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
HELD_TXT = [tk.decode([int(v) for v in held[i:i + 700]]) for i in range(2 * 10**6, 2 * 10**6 + 12 * 10**5, 10**5)]


def text_bpb(model, enc, fwd):
    nats = 0.0; nby = 0
    for t in HELD_TXT:
        ids = enc(t)[:512]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(x).float().log_softmax(-1)
        nats += float(-lg[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).sum())
        nby += len(t.encode()) * (len(ids) - 1) / max(1, len(enc(t)))
    return nats / (math.log(2) * nby)


def train(params, fwd, enc):
    """The same loop for both models: fixed steps, fixed rate, sequences packed from the shuffled sentences."""
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    r = random.Random(7)
    for step in range(STEPS):
        batch = []
        for _ in range(BATCH):
            ids = []
            while len(ids) < SEQ + 1:
                ids += enc(" " + r.choice(TRAIN))
            batch.append(ids[:SEQ + 1])
        x = torch.tensor(batch, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(x[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return float(loss)


print(f"{len(TEST)} invented facts, trained on {len(TRAIN)} sentences for {STEPS} steps at lr {LR}, "
      f"asked in a phrasing not trained on\n")
print(f"{'':<14s} {'NEW before':>11s} {'NEW after':>10s} {'OLD before':>11s} {'OLD after':>10s} {'TEXT before':>12s} {'TEXT after':>11s}")
for lab in ("Carrier", "GPT-2 124M"):
    if lab == "Carrier":
        m = Ours(CK, "/mnt/kv_cache/tok_v8192.json"); enc = m.encode
        fwd = lambda x: m.m(x); params = m.m.parameters(); m.m.train()
    else:
        m = GPT2(); enc = m.tok.encode
        fwd = lambda x: m.m(x).logits; params = m.m.parameters(); m.m.train()
    torch.manual_seed(0)
    nb, ob, tb = mc(m, TEST, enc), mc(m, FACTS, enc), text_bpb(m, enc, fwd)
    last = train(params, fwd, enc)
    if lab == "Carrier": m.graphable = False                  # weights changed; do not reuse a captured decode graph
    na, oa, ta = mc(m, TEST, enc), mc(m, FACTS, enc), text_bpb(m, enc, fwd)
    print(f"{lab:<14s} {nb:>10.1%} {na:>10.1%} {ob:>10.1%} {oa:>10.1%} {tb:>12.4f} {ta:>11.4f}   (final train loss {last:.3f})", flush=True)
    del m; torch.cuda.empty_cache()
print("\nLearning without forgetting: NEW rises, OLD and TEXT hold. Chance on NEW is 25%.")
