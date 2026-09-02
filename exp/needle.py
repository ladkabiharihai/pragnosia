"""NEEDLE-IN-A-HAYSTACK, DONE PROPERLY -- the missing half of north-star flag 1.

Flag 1 is "constant memory/compute AND needle/long-doc accuracy HELD". We have measured the cost half exhaustively
(flat to 1M tokens). The ACCURACY half has never been measured honestly: evaluate.py's recall test reports 0% at
every distance, but that test asks a 54M model to GENERATE an answer greedily from a ZERO state -- the exact
out-of-distribution setting its own docstring warns against. A 0% there says almost nothing about whether the
memory retained the fact.

THIS TEST ISOLATES MEMORY FROM QA ABILITY. Plant a fact, bury it under D tokens of real text, then measure the
model's LOG-PROBABILITY of the correct continuation against distractors. No generation, no answer formatting, no
greedy decoding. If the state retained the needle, the correct token scores higher. That is the whole question.

  accuracy    -- how often the true value beats all distractors (chance = 1/(1+n_distract))
  margin      -- logprob(true) - best logprob(distractor), in nats; > 0 means retained
Swept over distance so the result is a CURVE, which is what flag 1 actually claims.
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("needle")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
NTRIAL = 24
# distance 0 and 20 are CONTROLS: if the model cannot pick the name it saw five tokens ago, the test is measuring
# copying ability (or a bug), not memory retention over distance.
DISTS = [0, 20, 200, 1000, 4000, 16000, 64000]

sd = torch.load(CK, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
print(f"=== {P/1e6:.0f}M | step {sd['step']} | {sd['step']*32768*(sd.get('B',6))/1e9:.2f}B tokens | "
      f"D={cfg.D}x{cfg.NL}L DIN={cfg.DIN} gated={cfg.gated} ===\n", flush=True)

np.random.seed(3)
dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
# one long pool of real text to bury the needle in
pool = []
while len(pool) < max(DISTS) + 4096:
    pool.extend(dm.batch(1)[0].tolist())

NAMES = ["Aurelia", "Kestrel", "Vandermeer", "Okonkwo", "Thackeray", "Zoravar",
         "Lindqvist", "Mbeki", "Castellan", "Yusupov"]
_first = [TOK.encode(f" {n}").ids[0] for n in NAMES]
assert len(set(_first)) == len(_first), "candidates share a first token -- scoring it would be meaningless"


def zstate(b=1):
    return [(torch.zeros(b, cfg.FE, cfg.DV, device=DEV), torch.zeros(b, cfg.FE, device=DEV))
            for _ in range(cfg.NL)]


@torch.no_grad()
def score_next(ids, cue, cands):
    """Read fact+filler+cue as ONE stream with prev threaded across every chunk boundary, then score the next
    token. prefill() takes no `prev`, so calling it on the cue separately would drop the boundary key -- the very
    bug that made the chunked recurrence 8.1e-02 wrong. Threading forward() avoids it entirely."""
    full = ids + cue
    st = zstate(); pv = None
    for i in range(0, len(full) - 1, CH):
        x = torch.tensor([full[i:i + CH]], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, st, pv = m(x, states=st, return_states=True, prev=pv, return_prev=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, _, _ = m.step(torch.tensor([[full[-1]]], device=DEV), st, pv)
    lp = lg[0].float().log_softmax(-1)
    return [float(lp[c[0]]) for c in cands]


# PAIRED DESIGN. Comparing the true name against DISTRACTOR names confounds retrieval with token-frequency
# priors: rare names score low no matter what, which is exactly the uniform -1.5 nat margin seen at EVERY
# distance including 0. The clean measurement compares the SAME token to ITSELF with the fact present vs absent,
# so every prior cancels:
#     lift = logP(name | fact planted) - logP(name | a DIFFERENT name planted)
# A positive lift means reading the fact raised that name's probability -- retrieval, with nothing else it can be.
print(f"  {'distance':>9s} {'lift (nats)':>12s} {'>0 in':>8s} {'accuracy':>9s} {'chance':>7s}", flush=True)
for D in DISTS:
    lifts = []; hits = 0
    for t in range(NTRIAL):
        rng = np.random.RandomState(1000 + t)
        true_name = NAMES[t % len(NAMES)]
        other = NAMES[(t + 3) % len(NAMES)]
        distract = [n for n in NAMES if n != true_name]
        rng.shuffle(distract); distract = distract[:4]
        start = rng.randint(0, len(pool) - D - 64)
        filler = pool[start:start + D]
        cue = TOK.encode(" The vault keeper is").ids
        cands = [TOK.encode(f" {n}").ids for n in [true_name] + distract]
        with_fact = TOK.encode(f" The vault keeper is {true_name}. ").ids
        without   = TOK.encode(f" The vault keeper is {other}. ").ids
        sc_w = score_next(with_fact + filler, cue, cands)
        sc_o = score_next(without + filler, cue, cands)
        lifts.append(sc_w[0] - sc_o[0])          # same token, same everything else, fact swapped
        hits += (int(np.argmax(sc_w)) == 0)
    lifts = np.array(lifts)
    print(f"  {D:9d} {lifts.mean():+12.3f} {100*(lifts>0).mean():7.0f}% "
          f"{100*hits/NTRIAL:8.0f}% {20:6d}%", flush=True)

print(f"\n  LIFT > 0 means reading the fact raised that exact name's probability: retrieval, priors cancelled.", flush=True)
print(f"  A lift near 0 at distance 0 would mean the model cannot copy at all, independent of memory.", flush=True)
