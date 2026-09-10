"""DOES BANNING A SELF-REPEATING TOKEN FIX THE TEMPERATURE-0.8 COLLAPSE?

The closed-loop refresh repairs the recurrent state and measurably helps, but at temperature 0.8 the 191M still
collapses into runs of newlines with the trigger firing. Refresh cannot reach that failure: it is driven by the
CURRENT TOKEN, and "the current token is a newline" predicts another newline no matter what the state says.

run_ban=N zeroes a token that has just repeated N times in a row, for one step. The question is not only
whether it lowers repetition, but whether it does so WITHOUT distorting healthy generation, because real text
does contain short runs and a sampler that forbids them is measuring its own intervention.

Reported per condition: 4-gram repetition, the LONGEST RUN of a single token (the actual failure being fixed),
distinct-3, and how many times each mechanism fired. Real text is measured on the same metrics so "longest run
of 14" can be read against what English actually does.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("run_ban_test")
cap_gpu(8.0)
import shutil
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LIVE = "/mnt/kv_cache/pragnosia_data/p400f.pt"
TMP = "/mnt/kv_cache/pragnosia_data/_run_ban_copy.pt"
shutil.copyfile(LIVE, TMP)

sd = torch.load(TMP, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=512, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())

def rep4(ids):
    g = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return 1 - len(set(g))/max(1, len(g))
def dn(ids, n):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(g))/max(1, len(g))
def longest_run(ids):
    best = cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i-1] else 1
        best = max(best, cur)
    return best

np.random.seed(0)
rp = DataMix(ctx=NGEN+1, verbose=False, only='dclm')
R = [[int(t) for t in rp.batch(1)[0][:NGEN]] for _ in range(8)]
print(f"REAL WEB TEXT (n=8):  4g-repeat {np.mean([rep4(r) for r in R]):.3f}   "
      f"longest same-token run {np.mean([longest_run(r) for r in R]):.1f} (max {max(longest_run(r) for r in R)})   "
      f"distinct-3 {np.mean([dn(r,3) for r in R]):.3f}\n", flush=True)

np.random.seed(11)
pd = DataMix(ctx=1025, verbose=False, only='dclm')
LONG = [int(t) for t in pd.batch(1)[0][:256]]
SHORT = TOK.encode(" The following is a detailed account of").ids

print(f"{'prompt':>22s}{'temp':>6s}{'run_ban':>9s}{'4g rep':>9s}{'longest run':>13s}"
      f"{'distinct-3':>12s}{'refresh':>9s}{'bans':>6s}", flush=True)
print("-" * 92)
res = {}
for pname, pid in [("256-tok real doc", LONG), ("6-tok generic", SHORT)]:
    for temp in (0.9, 0.8):
        for rb in (0, 12):
            torch.manual_seed(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = m.generate(torch.tensor([pid], device=DEV), NGEN, temp=temp, top_p=0.95, run_ban=rb)
            g = [int(t) for t in out]
            r, lr_, d3 = rep4(g), longest_run(g), dn(g, 3)
            res[(pname, temp, rb)] = (r, lr_)
            print(f"{pname:>22s}{temp:>6.1f}{('off' if rb==0 else str(rb)):>9s}{r:>9.3f}{lr_:>13d}"
                  f"{d3:>12.3f}{getattr(m,'last_refresh_triggers',0):>9d}{getattr(m,'last_run_bans',0):>6d}",
                  flush=True)

print("\n" + "=" * 92)
better = worse = 0
for (pn, t, rb), (r, lr_) in res.items():
    if rb == 0: continue
    br, blr = res[(pn, t, 0)]
    tag = "BETTER" if r < br - 0.02 else ("WORSE" if r > br + 0.02 else "unchanged")
    if r < br - 0.02: better += 1
    if r > br + 0.02: worse += 1
    print(f"  {pn:>22s} temp {t}   4g {br:.3f} -> {r:.3f}   longest run {blr} -> {lr_}   {tag}")
print(f"\n  {better} better, {worse} worse, out of 4. Ship only if it fixes the broken cells and leaves the")
print("  healthy ones alone; a sampler that changes good generations is measuring itself, not the model.")
try: os.remove(TMP)
except OSError: pass
