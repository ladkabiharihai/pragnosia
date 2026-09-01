"""PICK write_scale PROPERLY. The earlier sweep used ONE document and gave contradictory readings between runs
(write_scale=0.25 looked clean once and produced digit garbage the next time), so it cannot decide anything.

What is now established:
  - the collapse is STRUCTURAL: a 202M model trained on 5.4B tokens degenerates too, and so does an UNGATED one,
    so neither scale nor the gate is the cause
  - freezing the state (write_scale=0) generates coherent English for the full 1200 tokens
  - scheduled sampling FAILED (repetition 0.55 vs 0.48 untouched, letters 0.84 vs 0.95)

So the remaining question is only: how much of its own output should generation write back? Enough samples this
time to tell signal from noise -- 6 documents x 2 seeds = 12 generations per setting, with the spread reported.

QUALITY IS NOT ONE NUMBER. Three metrics together, because each alone is gameable:
  4gram-rep  -- degeneration into loops        (but digit spam scores well)
  letters    -- fraction of characters that are letters, catches digit/punctuation spam
  distinct-1 -- vocabulary diversity           (but random tokens score perfectly)
A setting only wins if all three sit near the real-text reference.
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("write_scale_sweep")
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048; NGEN = 1200; NDOC = 6; NSEED = 2
CK = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/claude-1000/-opt-code-membrae/ab0dd831-724a-4bd0-be9e-dd7e1ea6f491/scratchpad/pinned.pt"


def stats(t):
    d1 = len(set(t)) / max(1, len(t))
    g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return d1, 1 - len(set(g)) / max(1, len(g))


def letters(txt):
    c = [x for x in txt if not x.isspace()]
    return sum(x.isalpha() for x in c) / max(1, len(c))


np.random.seed(11)
dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
docs = [dm.batch(1)[0].tolist() for _ in range(NDOC)]
np.random.seed(9)
refs = [dm.batch(1)[0][:NGEN].tolist() for _ in range(24)]
rr = [stats(r)[1] for r in refs]; rd = [stats(r)[0] for r in refs]
rl = [letters(TOK.decode(r)) for r in refs]

sd = torch.load(CK, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
print(f"SUBJECT: {P/1e6:.0f}M, step {sd['step']}, {NDOC}x{NSEED}={NDOC*NSEED} generations of {NGEN} tokens each\n",
      flush=True)
print(f"  {'write_scale':>12s} {'4gram-rep':>16s} {'letters':>15s} {'distinct-1':>15s} {'tail-rep':>9s}", flush=True)
print(f"  {'REAL TEXT':>12s} {f'{np.mean(rr):.2f} +-{np.std(rr):.2f}':>16s} "
      f"{f'{np.mean(rl):.2f} +-{np.std(rl):.2f}':>15s} {f'{np.mean(rd):.2f} +-{np.std(rd):.2f}':>15s} "
      f"{np.mean(rr):9.2f}", flush=True)

best = None
for w in (1.0, 0.5, 0.25, 0.1, 0.05, 0.0):
    R, L, D, T2 = [], [], [], []
    for di, doc in enumerate(docs):
        for s in range(NSEED):
            torch.manual_seed(di * 10 + s)
            o, _ = m.generate(doc[:1024], NGEN, temp=0.85, write_scale=w)
            d1, r4 = stats(o)
            R.append(r4); D.append(d1); L.append(letters(TOK.decode(o)))
            T2.append(stats(o[-400:])[1])                       # the last 400 tokens: where collapse shows first
    # a setting is only good if it is near real text on ALL THREE
    score = abs(np.mean(R) - np.mean(rr)) + abs(np.mean(L) - np.mean(rl)) + abs(np.mean(T2) - np.mean(rr))
    print(f"  {w:12.2f} {f'{np.mean(R):.2f} +-{np.std(R):.2f}':>16s} "
          f"{f'{np.mean(L):.2f} +-{np.std(L):.2f}':>15s} {f'{np.mean(D):.2f} +-{np.std(D):.2f}':>15s} "
          f"{np.mean(T2):9.2f}", flush=True)
    if best is None or score < best[0]: best = (score, w)

print(f"\n  closest to real text on all three metrics: write_scale = {best[1]}", flush=True)
for w in (1.0, best[1]):
    torch.manual_seed(0)
    o, _ = m.generate(docs[0][:1024], NGEN, temp=0.85, write_scale=w)
    print(f"\n  write_scale={w}, last 240 chars: {TOK.decode(o)[-240:]!r}", flush=True)
