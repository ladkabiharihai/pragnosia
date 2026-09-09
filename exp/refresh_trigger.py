"""DOES A CLOSED-LOOP REFRESH FIX THE LOW-TEMPERATURE COLLAPSE?

The periodic state refresh holds long generation together at temperature 0.9 but not at 0.8 with a short
prompt, where 4-gram repetition reached 0.468 over 9000 tokens against real web text's 0.068. The periodic
refresh is open-loop: it fires on a 1000-token schedule whether or not anything has gone wrong, so a loop that
begins at token 100 runs 900 tokens before the state is touched.

This tests closing the loop: measure repetition over the last 256 generated tokens every 64 tokens, and refresh
immediately when it crosses a threshold.

The test is a matrix, not a single cell, because the only interesting question is whether the fix helps the bad
condition WITHOUT hurting the good ones. A change that rescues temperature 0.8 by making temperature 0.9 worse
is not a fix, it is a trade, and a single-cell result would hide that.
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("refresh_trigger")
cap_gpu(8.0)
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CKPT = os.environ.get("PRAGNOSIA_CKPT", "/mnt/kv_cache/pragnosia_data/p124f.pt")
NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
WIN = 3000

sd = torch.load(CKPT, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=512, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())

def rep4(ids):
    g = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return 1 - len(set(g))/max(1, len(g))
def dn(ids, n):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(g))/max(1, len(g))

np.random.seed(11)
pd = DataMix(ctx=1025, verbose=False, only='dclm')
LONG = [int(t) for t in pd.batch(1)[0][:256]]
SHORT = TOK.encode(" The following is a detailed account of").ids

# real-text reference on the SAME metric and window, so the target is a measured line and not zero
np.random.seed(0)
rp = DataMix(ctx=3001, verbose=False, only='dclm')
R = [[int(t) for t in rp.batch(1)[0][:3000]] for _ in range(8)]
ref = (np.mean([rep4(r) for r in R]), np.mean([dn(r,1) for r in R]), np.mean([dn(r,3) for r in R]))
print(f"REAL WEB TEXT (n=8 windows of 3000):  4g-repeat {ref[0]:.3f}   distinct-1 {ref[1]:.3f}   "
      f"distinct-3 {ref[2]:.3f}\n", flush=True)

print(f"{'prompt':>24s}{'temp':>6s}{'trigger':>10s}{'4g repeat':>11s}{'distinct-1':>12s}"
      f"{'distinct-3':>12s}{'fires':>7s}   per-window 4g repeat")
print("-" * 108)
res = {}
for pname, pid in [("256-token real doc", LONG), ("6-token generic", SHORT)]:
    for temp in (0.9, 0.8):
        for thr in (0.0, 0.35):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = m.generate(torch.tensor([pid], device=DEV), NGEN, temp=temp, top_p=0.95,
                                    rep_thresh=thr)
            g = [int(t) for t in out][len(pid):] if len(out) > NGEN else [int(t) for t in out]
            assert len(g) > NGEN * 0.9, f"got {len(g)} tokens"
            wins = [rep4(g[w:w+WIN]) for w in range(0, len(g)-WIN+1, WIN)]
            r, d1, d3 = np.mean(wins), dn(g, 1), dn(g, 3)
            res[(pname, temp, thr)] = r
            print(f"{pname:>24s}{temp:>6.1f}{('off' if thr==0 else f'{thr:.2f}'):>10s}"
                  f"{r:>11.3f}{d1:>12.3f}{d3:>12.3f}{getattr(m,'last_refresh_triggers',0):>7d}   "
                  + "  ".join(f"{v:.2f}" for v in wins), flush=True)

print("\n" + "=" * 108)
print("  VERDICT")
print("=" * 108)
worse = better = 0
for (pname, temp, thr), v in res.items():
    if thr == 0.0: continue
    base = res[(pname, temp, 0.0)]
    d = v - base
    tag = "BETTER" if d < -0.02 else ("WORSE" if d > 0.02 else "unchanged")
    if d < -0.02: better += 1
    if d > 0.02: worse += 1
    print(f"  {pname:>24s}  temp {temp}   {base:.3f} -> {v:.3f}   {d:+.3f}   {tag}")
print(f"\n  {better} cells better, {worse} cells worse, out of 4.")
print("  Ship only if it helps where it was broken and harms nothing where it worked. Anything else is a trade")
print("  dressed up as a fix, and the honest move is to report it and leave the default off.")
