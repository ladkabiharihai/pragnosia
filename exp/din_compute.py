"""RETRIEVAL WIDTH vs TOKENS AT MATCHED COMPUTE. Raising DIN improves loss but costs throughput (DIN 16->32 is
-0.096 nats at -28% ktok/s). The previous sweep compared at matched STEPS, which quietly gives the wide model more
compute. The decision we actually face is how to spend a FIXED GPU budget: on a wider retrieval space, or on more
tokens through a narrower one.

So run each DIN for the SAME WALL-CLOCK TIME and compare. Whichever is lower at the buzzer is the better use of the
same hardware -- which is the only comparison that matters for a run that is throughput-bound."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("din_compute")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; D=512; NL=8; BUDGET=300.0     # seconds per arm
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(DIN,budget=BUDGET):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=16**-0.5,tie_emb=True)
    m=C.Core(cfg).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    t0=time.time(); steps=0
    while time.time()-t0 < budget:
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); steps+=1
    out=evl(m,cfg); fe=cfg.FE
    del m,opt; torch.cuda.empty_cache()
    return out,fe,steps
if __name__=="__main__":
    print(f"SAME GPU-SECONDS, DIFFERENT RETRIEVAL WIDTHS ({BUDGET:.0f}s each, D={D}, {NL} layers)\n",flush=True)
    print(f"  {'DIN':>4s} {'FE':>5s} {'steps done':>11s} {'tokens':>9s} {'loss':>8s} {'vs DIN=16':>10s}",flush=True)
    base=None
    for DIN in (16,24,32):
        l,fe,st=run(DIN)
        if base is None: base=l
        print(f"  {DIN:4d} {fe:5d} {st:11d} {st*B*CH/1e6:8.1f}M {l:8.4f} {l-base:+10.4f}",flush=True)
    print(f"\n  At matched STEPS, DIN=32 won by 0.096 nats. At matched COMPUTE the wide model gets fewer",flush=True)
    print(f"  tokens, so this is the honest comparison for a throughput-bound run.",flush=True)
