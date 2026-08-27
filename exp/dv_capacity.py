"""Does a BIGGER STATE (DV) actually buy more usable context? Grow DV on a trained model (function-preserving, so
quality is unchanged at t=0) and re-measure the memory gain at long range. If gain improves with DV, then growth
genuinely scales context capacity -- the user's point (1)."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("dv_capacity")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G; from data_mix import DataMix
DEV=C.DEV; CH=2048
def load():
    sd=torch.load("/mnt/kv_cache/pragnosia_data/snap_6000.pt",map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg
@torch.no_grad()
def gain(m,cfg,decay,targets,seed=11):
    np.random.seed(seed); dm=DataMix(ctx=CH+1,verbose=False); st=None; seen=0; out={}
    while seen<max(targets):
        X=torch.tensor(dm.batch(1),device=DEV); xi=X[:,:CH]; yi=X[:,1:CH+1]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg,st=m(xi,states=st,return_states=True)
        st=m.decay_states(st,decay); seen+=CH
        if seen in targets:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lw=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
                lo=F.cross_entropy(m(xi).reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
            out[seen]=lo-lw
    return out
if __name__=="__main__":
    T={8192,32768,131072}
    print("Does growing DV extend usable context? (function-preserving grow, so t=0 quality is identical)",flush=True)
    print(f"  {'DV':>6s} {'state/layer':>12s} "+" ".join(f"{t:>9d}" for t in sorted(T)),flush=True)
    for add in (0,512,1536):
        m,cfg=load()
        if add: G.grow_state(m,add)                      # DV only, function-preserving
        g=gain(m,m.cfg,0.05,T)
        print(f"  {m.cfg.DV:6d} {m.cfg.FE*m.cfg.DV*4/2**20:11.2f}M "+" ".join(f"{g[t]:+9.4f}" for t in sorted(T)),flush=True)
        del m; torch.cuda.empty_cache()
