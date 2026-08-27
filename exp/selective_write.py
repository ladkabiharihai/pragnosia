"""(3) SELECTIVE WRITING: does gating memory writes by the model's OWN SURPRISE extend usable context?
Every token currently writes, so N tokens superimpose in a fixed state and any one fact is ~1/N of the signal.
Write only the top-p most surprising tokens (the vision's propose gate) and re-measure the memory gain at long range."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("selective_write")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; CH=2048
sd=torch.load("/mnt/kv_cache/pragnosia_data/snap_6000.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
@torch.no_grad()
def run(frac,decay,targets,seed=11):
    """frac=None -> write everything (current). frac=p -> write only the top-p most surprising tokens."""
    np.random.seed(seed); dm=DataMix(ctx=CH+1,verbose=False); st=None; seen=0; out={}
    while seen<max(targets):
        X=torch.tensor(dm.batch(1),device=DEV); xi=X[:,:CH]; yi=X[:,1:CH+1]
        gate=None
        if frac is not None:                                  # the model's own surprise decides what is worth storing
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lp=m(xi,states=st)[:, :-1].log_softmax(-1)
            sur=-lp.gather(-1,xi[:,1:].unsqueeze(-1)).squeeze(-1).float()
            sur=torch.cat([sur[:,:1],sur],1)
            k=max(1,int(frac*CH)); thr=sur.topk(k,dim=1).values[:,-1:]
            gate=(sur>=thr).float()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg,st=m(xi,states=st,return_states=True,wgate=gate)
        st=m.decay_states(st,decay); seen+=CH
        if seen in targets:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lw=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
                lo=F.cross_entropy(m(xi).reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
            out[seen]=lo-lw
    return out
if __name__=="__main__":
    T={8192,32768,131072,524288}
    print("SELECTIVE WRITING: memory gain when only the most SURPRISING tokens write to state",flush=True)
    print(f"  {'writes':>12s} "+" ".join(f"{t:>9d}" for t in sorted(T)),flush=True)
    for frac,lab in [(None,"all (current)"),(0.25,"top 25%"),(0.10,"top 10%"),(0.03,"top 3%")]:
        g=run(frac,0.05,T)
        print(f"  {lab:>12s} "+" ".join(f"{g[t]:+9.4f}" for t in sorted(T)),flush=True)
    print("\n  (decay 0.05 throughout; a HIGHER gain at long range = selective writing extends usable context)",flush=True)
