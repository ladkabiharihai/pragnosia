"""DOES THE RETRIEVAL BOTTLENECK GET WORSE AS THE MODEL GROWS? DIN=8 (45 features) costs -0.11 nats at D=384.
If the penalty GROWS with D, that explains why our 235M model underperforms its parameter count: it grew 2.7x wider
while its retrieval space stayed identical. Test the same DIN sweep at two widths and compare the gaps."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("din_scaling")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; NL=6; STEPS=3000
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(D,DIN,steps=STEPS):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5)
    m=C.Core(cfg).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for i in range(steps):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    out=evl(m,cfg); del m,opt; torch.cuda.empty_cache(); return out
if __name__=="__main__":
    print(f"DOES THE BOTTLENECK WORSEN WITH WIDTH? {NL} layers, {STEPS} steps\n",flush=True)
    print(f"  {'D':>5s} {'DIN=8':>9s} {'DIN=16':>9s} {'DIN=32':>9s} {'gain 8->32':>11s} {'D/FE ratio':>11s}",flush=True)
    for D in (384,768):
        r={}
        for DIN in (8,16,32):
            r[DIN]=run(D,DIN)
        fe=1+8+8*9//2
        print(f"  {D:5d} {r[8]:9.4f} {r[16]:9.4f} {r[32]:9.4f} {r[8]-r[32]:+11.4f} {D/fe:11.1f}",flush=True)
    print(f"\n  If 'gain 8->32' is LARGER at D=768 than D=384, the bottleneck tightens as the model grows --",flush=True)
    print(f"  which is exactly what happened to our 235M model (D=1024 with the infant's 45 features).",flush=True)
