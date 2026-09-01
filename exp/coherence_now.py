"""COHERENCE AND GENERATION STABILITY AT THE CURRENT SIZE. The claim that generation can only be judged at 200M is
not defensible: a 21.6M model of this same architecture already beat a transformer on generation (ppl 37.8 vs 47.7,
distinct-1 0.72 vs 0.66, zero repetition). So it must be demonstrable here, at 54M, or something is wrong.

Everything is run the way the model was TRAINED, which is what earlier mistakes taught:
  - continue REAL text at the trained context length (a bare prompt with a zero state is out of distribution)
  - temperature sampling, never greedy (top-1 is often ~0.09; argmax on a flat distribution locks into a loop)
  - Core.generate(), which prefills all-but-the-last token (the old code double-processed the last prompt token)
  - no hand-set decay: this model is GATED, so the state already decays inside the layer

  1. SHORT-RANGE COHERENCE -- continue real text, measure diversity and repetition against real text itself
  2. LONG-RANGE STABILITY -- 1200 tokens, windowed, to see whether it degenerates as it runs
  3. TEMPERATURE SENSITIVITY -- is there a setting where it holds together
  4. THE ACTUAL TEXT -- shown in full, so quality is judged and not inferred from a number
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("coherence_now")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
CH = 2048

sd = torch.load(CK, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
print(f"=== {P/1e6:.0f}M | step {sd['step']} | {sd['step']*32768*8/1e9:.2f}B tokens | "
      f"D={cfg.D}x{cfg.NL}L DIN={cfg.DIN} gated={cfg.gated} ===\n", flush=True)


def stats(t):
    d1 = len(set(t)) / max(1, len(t))
    g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return d1, 1 - len(set(g)) / max(1, len(g))


np.random.seed(11)
dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
docs = [dm.batch(1)[0].tolist() for _ in range(4)]

# what REAL text scores on the same metrics -- the only meaningful reference
np.random.seed(9)
rd, rr = zip(*[stats(dm.batch(1)[0][:120].tolist()) for _ in range(30)])
print(f"REFERENCE -- real text from the same corpus: distinct-1 {np.mean(rd):.2f}  4gram-rep {np.mean(rr):.2f}\n",
      flush=True)

print("=" * 100)
print("1) SHORT-RANGE COHERENCE -- continue 1024 tokens of real text, 120 generated")
print("=" * 100, flush=True)
d1s, r4s = [], []
for i, doc in enumerate(docs):
    torch.manual_seed(i)
    out, _ = m.generate(doc[:1024], 120, temp=0.85)
    d1, r4 = stats(out); d1s.append(d1); r4s.append(r4)
    print(f"\n  --- sample {i+1}  (distinct-1 {d1:.2f}, 4gram-rep {r4:.2f}) ---", flush=True)
    print(f"  [context ends] ...{TOK.decode(doc[976:1024])}", flush=True)
    print(f"  [model writes] {TOK.decode(out)}", flush=True)
print(f"\n  MEAN over {len(docs)} samples: distinct-1 {np.mean(d1s):.2f}  4gram-rep {np.mean(r4s):.2f}"
      f"   (real text {np.mean(rd):.2f} / {np.mean(rr):.2f})", flush=True)

print("\n" + "=" * 100)
print("2) LONG-RANGE STABILITY -- 1200 tokens continuous, measured in 200-token windows")
print("=" * 100, flush=True)
torch.manual_seed(0)
long_out, _ = m.generate(docs[0][:1024], 1200, temp=0.85)
print(f"\n  {'window':>12s} {'distinct-1':>11s} {'4gram-rep':>10s}", flush=True)
for a in range(0, 1200, 200):
    w = long_out[a:a + 200]; d1, r4 = stats(w)
    print(f"  {f'{a}-{a+200}':>12s} {d1:11.2f} {r4:10.2f}", flush=True)
od1, or4 = stats(long_out)
print(f"  {'OVERALL':>12s} {od1:11.2f} {or4:10.2f}", flush=True)
print(f"\n  a RISING 4gram-rep across windows = it degenerates as it runs. Flat = stable.", flush=True)
print(f"\n  last 300 chars: {TOK.decode(long_out)[-300:]!r}", flush=True)

print("\n" + "=" * 100)
print("3) TEMPERATURE SENSITIVITY")
print("=" * 100, flush=True)
print(f"\n  {'temp':>6s} {'distinct-1':>11s} {'4gram-rep':>10s}", flush=True)
for t in (0.6, 0.7, 0.85, 1.0):
    ds, rs = [], []
    for i, doc in enumerate(docs[:3]):
        torch.manual_seed(i)
        o, _ = m.generate(doc[:1024], 120, temp=t)
        a, b = stats(o); ds.append(a); rs.append(b)
    print(f"  {t:6.2f} {np.mean(ds):11.2f} {np.mean(rs):10.2f}", flush=True)
