"""DOES THE SELF-WRITE COLLAPSE GET BETTER OR WORSE WITH TRAINING AND SCALE?

Scheduled sampling FAILED: 4-gram repetition 0.55 at the tail versus 0.48 untouched, and the output turned to
digit garbage (letters 0.84 vs 0.95). Before attempting another blind fix, establish the one fact that decides
whether an architecture change is needed at all -- the TRAJECTORY.

  falling repetition across checkpoints -> training resolves it, no fix needed, spend the compute
  flat or rising                        -> structural, must be fixed BEFORE committing to a 90M run

The most informative row is the 235M / 12.3B-token model from run 3: nine times the tokens and four times the
parameters of the current checkpoint. If that one also collapses, scale is definitively not the answer.

CAVEAT, stated because it matters: the run-3 checkpoints are UNGATED and were trained with a hand-set decay
applied between chunks, which m.step() does not reproduce during generation. Their absolute numbers are therefore
not directly comparable to the gated rows -- they answer "does a much bigger, much longer-trained model of this
family degenerate", not "is it the gate".

Cheap: generation only, no training. Same documents, same seeds, same metric for every checkpoint.
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("stability_traj")
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
NGEN = 1200


def stats(t):
    d1 = len(set(t)) / max(1, len(t))
    g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return d1, 1 - len(set(g)) / max(1, len(g))


def letters(txt):
    c = [x for x in txt if not x.isspace()]
    return sum(x.isalpha() for x in c) / max(1, len(c))


np.random.seed(11)
dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
docs = [dm.batch(1)[0].tolist() for _ in range(3)]
np.random.seed(9)
refs = [dm.batch(1)[0][:NGEN].tolist() for _ in range(12)]
rw = [np.mean([stats(r[a:a + 200])[1] for r in refs]) for a in range(0, NGEN, 200)]
rd1 = np.mean([stats(r)[0] for r in refs])
rlt = np.mean([letters(TOK.decode(r)) for r in refs])

hdr = " ".join(f"{f'{a}-{a+200}':>9s}" for a in range(0, NGEN, 200))
print("STABILITY vs TRAINING AND SCALE: 4-gram repetition by window, 3 documents averaged\n", flush=True)
print(f"  {'checkpoint':>28s} {hdr} {'d1':>6s} {'letters':>8s}", flush=True)
print(f"  {'REAL TEXT':>28s} " + " ".join(f"{r:9.2f}" for r in rw) + f" {rd1:6.2f} {rlt:8.2f}", flush=True)

for ck in sys.argv[1:]:
    try:
        sd = torch.load(ck, map_location=DEV)
        cfg = cfg_from_ckpt(sd, T=CH)
        m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    except Exception as e:
        print(f"  {ck.split('/')[-1][:28]:>28s}  SKIPPED ({type(e).__name__})", flush=True)
        continue
    P = sum(p.numel() for p in m.parameters())
    allw, d1s, lts = [], [], []
    for di, doc in enumerate(docs):
        torch.manual_seed(di)
        o, _ = m.generate(doc[:1024], NGEN, temp=0.85, write_scale=1.0)
        allw.append([stats(o[a:a + 200])[1] for a in range(0, NGEN, 200)])
        d1s.append(stats(o)[0]); lts.append(letters(TOK.decode(o)))
    w = np.mean(allw, axis=0)
    g = 'gated' if cfg.gated else 'UNGATED'
    tag = f"{P/1e6:.0f}M {g} {sd['step']*32768*8/1e9:.1f}B"
    print(f"  {tag:>28s} " + " ".join(f"{r:9.2f}" for r in w) +
          f" {np.mean(d1s):6.2f} {np.mean(lts):8.2f}", flush=True)
    del m; torch.cuda.empty_cache()

print("\n  FALLING repetition across rows = training/scale resolves it, no architecture change needed.", flush=True)
print("  FLAT or RISING = structural, and must be fixed before spending compute on a 90M run.", flush=True)
