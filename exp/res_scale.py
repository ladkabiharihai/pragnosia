"""ANOTHER UNQUESTIONED CHOICE: res_scale. Every layer adds its output to the residual stream as `a + rs*o`, with
rs = TARGET_NL**-0.5 = 16**-0.5 = 0.25 at our target depth. That is a PERMANENT attenuation: each layer contributes
only a quarter of what it computes, forever.

A standard transformer uses `x + f(x)` with NO scaling -- the 1/sqrt(depth) factor appears only in the INITIALISATION
of the output projections, so the network can learn to use its full depth. A permanent scale cannot be undone by
training the layer weights, because it multiplies whatever they produce.

Worse for us: rs is set from TARGET_NL (16), not the CURRENT depth, so the infant at 4 layers is already attenuating
by 0.25 when 1/sqrt(4) = 0.5 would be the matched value. Test whether this is costing us."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("res_scale")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; D=384; NL=6; STEPS=3000
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(rs,steps=STEPS):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=12,DV=D,NL=NL,T=CH,res_scale=rs,tie_emb=True)
    m=C.Core(cfg).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for i in range(steps):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    out=evl(m,cfg); del m,opt; torch.cuda.empty_cache(); return out
if __name__=="__main__":
    print(f"DOES THE PERMANENT RESIDUAL SCALE COST US? D={D}, {NL} layers, {STEPS} steps\n",flush=True)
    print(f"  {'res_scale':>28s} {'value':>7s} {'loss':>8s} {'vs ours':>9s}",flush=True)
    base=None
    for tag,rs in [("16**-0.5 (ours: TARGET_NL)",16**-0.5),
                   ("NL**-0.5 (matched depth)",NL**-0.5),
                   ("2**-0.5",2**-0.5),
                   ("1.0 (transformer-style)",1.0)]:
        l=run(rs)
        if base is None: base=l
        print(f"  {tag:>28s} {rs:7.4f} {l:8.4f} {l-base:+9.4f}",flush=True)
    print(f"\n  ours is set from TARGET_NL=16 regardless of current depth, so even the 4-layer infant",flush=True)
    print(f"  attenuates by 0.25 when its matched value would be 0.5.",flush=True)
