"""IS DIN=24 THE RIGHT TARGET? Our plan ends at DIN=24 (FE=325), giving D/FE = 3.2 at D=1024. A transformer has
D/FE = 1. The earlier sweep showed DIN=32 still improving at D=768, so the target may still be too low.
But FE grows QUADRATICALLY (FE = 1 + DIN + DIN(DIN+1)/2), so both the state and the attention compute scale with it.
Measure the tradeoff: loss gained vs throughput lost."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("din_target")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; D=512; NL=8; STEPS=2500
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(DIN):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=16**-0.5,tie_emb=True)
    m=C.Core(cfg).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    t0=time.time()
    for i in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    dt=time.time()-t0
    out=evl(m,cfg); fe=cfg.FE
    del m,opt; torch.cuda.empty_cache()
    return out,fe,STEPS*B*CH/dt
if __name__=="__main__":
    print(f"HOW HIGH SHOULD DIN GO? D={D}, {NL} layers, {STEPS} steps, tied\n",flush=True)
    print(f"  {'DIN':>4s} {'FE':>5s} {'D/FE':>6s} {'loss':>8s} {'gain':>8s} {'ktok/s':>8s} {'state/layer':>12s}",flush=True)
    base=None
    for DIN in (16,24,32,48):
        l,fe,tps=run(DIN)
        if base is None: base=l
        print(f"  {DIN:4d} {fe:5d} {D/fe:6.1f} {l:8.4f} {l-base:+8.4f} {tps/1e3:8.0f} "
              f"{fe*D*4/2**20:11.2f}M",flush=True)
    print(f"\n  Our plan targets DIN=24. If 32 or 48 buys much more loss than it costs in speed,",flush=True)
    print(f"  the target is still too conservative.",flush=True)
