"""IS THE 405M's LONGER RETRIEVAL REAL? n=8 gave a monotone trend with error bars too wide to claim it.

At n=8 the d=1000 lift rose from +0.542+-0.281 (125M) to +1.422+-0.639 (405M), which is about 1.3 sigma: a
trend, not a result. VISION 5e exists because exactly this shape of evidence was believed once before. This
runs only the two endpoint models at only the two distances that matter, with enough trials to decide.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("needle_focus")
cap_gpu(float(os.environ.get("CAP", "18")))
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV; TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D = "/mnt/kv_cache/pragnosia_data"
MODELS = [("125M", f"{D}/p124f.pt"), ("405M", f"{D}/p400f_405M_snap.pt")]
DISTS = (1000, 2000)
N = int(os.environ.get("N", "40"))

dm = DataMix(ctx=2049, verbose=False, lang_frac=0.70)
np.random.seed(5)
pool = []
while len(pool) < 3000 + N * 2200: pool.extend(dm.batch(1)[0].tolist())

@torch.no_grad()
def lifts(m, d):
    cue = TOK.encode(" The vault keeper is").ids; ans = TOK.encode(" Okonkwo").ids
    def sc(ctx):
        x = torch.tensor([ctx + cue + ans], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = m(x[:, :-1]).float().log_softmax(-1)
        return float(lp[0, -len(ans):].gather(-1, x[0, 1:][-len(ans):].unsqueeze(-1)).mean())
    out = []
    for t in range(N):
        f = pool[t*2200: t*2200 + d]
        out.append(sc(TOK.encode(" The vault keeper is Okonkwo. ").ids + f)
                   - sc(TOK.encode(" The vault keeper is Vandermeer. ").ids + f))
    return np.array(out)

got = {}
for tag, path in MODELS:
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=2048, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    for d in DISTS:
        v = lifts(m, d); got[(tag, d)] = v
        print(f"  {tag} d={d}: mean {v.mean():+.4f} +- {v.std(ddof=1)/np.sqrt(N):.4f}  (n={N})", flush=True)
    del m; torch.cuda.empty_cache()

print("\n" + "=" * 84)
print("  PAIRED-DIFFERENCE TEST, same fill text for both models at each trial, so the comparison is paired")
print("=" * 84)
for d in DISTS:
    a, b = got[("125M", d)], got[("405M", d)]
    diff = b - a                                   # trial i uses identical fill for both models
    se = diff.std(ddof=1)/np.sqrt(N)
    t = diff.mean()/se if se > 0 else float('nan')
    print(f"  d={d}:  405M - 125M = {diff.mean():+.4f} +- {se:.4f}   t = {t:+.2f}   "
          f"{'SIGNIFICANT' if abs(t) >= 2.0 else 'not significant'} at 2 sigma")
print("\n  A paired test is the right one here: both models see the identical filler at each trial, so the")
print("  trial-to-trial variance in the filler cancels instead of being counted twice.")
