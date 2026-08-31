"""IS THE GROWN RETRIEVAL CAPACITY ACTUALLY BEING USED? The width ablation on run 2 showed grown DIMENSIONS were
nearly dead weight (all 640 grown dims worth 0.59 nats against the seed's 8.13). DIN growth is new, so test it the
same honest way: zero the key/query rows added at each growth and measure the damage.

If the newest DIN dims carry real load soon after being added, the retrieval fix is working as intended -- unlike
the width growth, whose capacity took billions of tokens to recruit."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("din_recruited")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from util import cfg_from_ckpt
DEV=C.DEV; CH=1024
sd=torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=CH)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
np.random.seed(1234); dm=DataMix(ctx=CH+1,verbose=False,only='window2')
EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(30)]
@torch.no_grad()
def evl():
    t=0
    for X in EV:
        with torch.autocast("cuda",dtype=torch.bfloat16):
            t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    return t/len(EV)
base=evl()
P=sum(p.numel() for p in m.parameters())
print(f"@ step {sd['step']}  {P/1e6:.0f}M  D={cfg.D} NL={cfg.NL} DIN={cfg.DIN} (FE={cfg.FE})  tied={cfg.tie_emb}")
print(f"  baseline loss {base:.4f}\n")
print(f"  {'DIN dims':>12s} {'when added':>22s} {'damage if zeroed':>18s}")
bands=[(0,8,"original seed (DIN=8)"),(8,12,"growth #1"),(12,16,"growth #2 (newest)")]
for lo,hi,tag in bands:
    if hi>cfg.DIN: continue
    sav={}
    with torch.no_grad():
        for i,L in enumerate(m.layers):
            sav[i]=(L.Wk.weight[lo:hi].clone(),L.Wq.weight[lo:hi].clone())
            L.Wk.weight[lo:hi]=0; L.Wq.weight[lo:hi]=0
    d=evl()-base
    with torch.no_grad():
        for i,L in enumerate(m.layers):
            a,b=sav[i]; L.Wk.weight[lo:hi]=a; L.Wq.weight[lo:hi]=b
    print(f"  {f'{lo}-{hi}':>12s} {tag:>22s} {d:>+17.4f}")
print(f"\n  For contrast, run 2's WIDTH growth: the newest band was worth +0.0011 after 2.6B tokens.")
