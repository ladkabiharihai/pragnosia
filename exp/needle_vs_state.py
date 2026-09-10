"""DOES RETRIEVAL DISTANCE SCALE WITH STATE CAPACITY?

VISION 5i concluded the ~2000-token retrieval horizon is intrinsic to carrying a fixed-size state, after ruling
out training context, write interference, capacity-as-parameters, and credit assignment. It named one direction
left untested: a DIFFERENT state design. The ladder accidentally ran that experiment, because each growth
enlarges the state, and the first pass showed in-context lift at d=1000 rising monotonically with model size
while BPB got worse.

That is worth measuring properly rather than reading off one distance. The state here is FE x DV per layer,
where FE = 1 + DIN + DIN^2 is the feature dimension of the second-order map and DV is the value width, so
growth in either DIN or D enlarges it.

Paired protocol: identical query scored with the fact present versus a different fact present, so token
frequency priors cancel exactly and the number is the lift attributable to the fact being in context.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("needle_vs_state")
cap_gpu(float(os.environ.get("CAP", "18")))
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D = "/mnt/kv_cache/pragnosia_data"
MODELS = [("125M", f"{D}/p124f.pt"), ("191M", f"{D}/p400f_before_grow2.pt"),
          ("279M", f"{D}/p400f_before_grow3.pt"), ("405M", f"{D}/p400f_405M_snap.pt")]
DISTS = (0, 200, 500, 1000, 2000, 4000, 8000)
NTRIAL = int(os.environ.get("NTRIAL", "8"))

dm = DataMix(ctx=2049, verbose=False, lang_frac=0.70)
np.random.seed(5)
pool = []
while len(pool) < 9000 + NTRIAL * 1000: pool.extend(dm.batch(1)[0].tolist())

@torch.no_grad()
def lift(m, d):
    cue = TOK.encode(" The vault keeper is").ids
    ans = TOK.encode(" Okonkwo").ids
    def sc(ctx):
        x = torch.tensor([ctx + cue + ans], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = m(x[:, :-1]).float().log_softmax(-1)
        return float(lp[0, -len(ans):].gather(-1, x[0, 1:][-len(ans):].unsqueeze(-1)).mean())
    L = []
    for t in range(NTRIAL):
        fill = pool[t*1000: t*1000 + d]
        L.append(sc(TOK.encode(" The vault keeper is Okonkwo. ").ids + fill)
                 - sc(TOK.encode(" The vault keeper is Vandermeer. ").ids + fill))
    return float(np.mean(L)), float(np.std(L)/np.sqrt(len(L)))

print(f"{'model':>7s}{'params':>9s}{'DIN':>5s}{'DV':>6s}{'state/layer':>13s}{'x125M':>7s}  "
      + "".join(f"{('d='+str(d)):>15s}" for d in DISTS), flush=True)
print("-" * (47 + 15*len(DISTS)))
base = None
for tag, path in MODELS:
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=2048, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    P = sum(p.numel() for p in m.parameters())
    FE = 1 + cfg.DIN + cfg.DIN*cfg.DIN
    st = FE * cfg.DV
    if base is None: base = st
    cells = []
    for d in DISTS:
        mu, se = lift(m, d)
        cells.append(f"{mu:>+9.3f}+-{se:.3f}")
    print(f"{tag:>7s}{P/1e6:>8.1f}M{cfg.DIN:>5d}{cfg.DV:>6d}{st/1e6:>11.2f}M{st/base:>7.2f}  "
          + "".join(f"{c:>15s}" for c in cells), flush=True)
    del m; torch.cuda.empty_cache()

print("\n  Lift is nats. Paired, so frequency priors cancel and 0 means the fact in context bought nothing.")
print("  If the far columns rise with state size while BPB does not, then the retrieval horizon is set by")
print("  STATE CAPACITY rather than being intrinsic to recurrence, and VISION 5i's conclusion needs amending.")
