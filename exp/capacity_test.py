"""WHY does usefulness stop beyond ~32k, and is it fixable? Three candidate causes:
  C1 STATE SIZE: capacity is FE x DV; too small -> facts interfere. Test: does a bigger DV extend the useful range?
  C2 DECAY: decay=0.2/chunk gives an effective window of ~chunk/(1-decay); older writes are geometrically erased.
     Test: sweep decay and see where the memory gain is maximized at LONG range.
  C3 TRAINING HORIZON: the model was only ever trained on 32k documents, so it never learned to use anything beyond.
     Test: is the decay that is BEST at long range different from the one it was trained with?
Measures memory gain at increasing context, per decay setting."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("capacity_test")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; CH=2048
sd=torch.load("/mnt/kv_cache/pragnosia_data/snap_6000.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
@torch.no_grad()
def gain_at(decay,targets,seed=11):
    np.random.seed(seed); dm=DataMix(ctx=CH+1,verbose=False)
    st=None; seen=0; out={}
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
    print(f"CAPACITY / DECAY SWEEP @ step {sd['step']} (D={cfg.D} DV={cfg.DV}, state {cfg.FE}x{cfg.DV} = "
          f"{cfg.FE*cfg.DV*4/2**20:.2f} MiB per layer)",flush=True)
    T=[8192,32768,131072,524288]
    print(f"\n  memory GAIN (higher = memory helping more) at each context length:",flush=True)
    print(f"  {'decay':>7s} "+" ".join(f"{t:>9d}" for t in T),flush=True)
    for d in (1.0,0.9,0.5,0.2,0.05):
        g=gain_at(d,set(T))
        print(f"  {d:7.2f} "+" ".join(f"{g[t]:+9.4f}" for t in T),flush=True)
    print("\n  effective window ~ chunk/(1-decay):",flush=True)
    for d in (1.0,0.9,0.5,0.2,0.05):
        w="unbounded (all writes kept)" if d>=1.0 else f"{CH/(1-d):.0f} tokens"
        print(f"    decay={d:.2f} -> {w}",flush=True)
