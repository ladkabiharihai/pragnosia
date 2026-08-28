"""DOES RECRUITED CAPACITY ACTUALLY WIN? fix_growth.py showed net2net recruits new dims 30x better (+0.047 vs
-0.002) but its final loss was still marginally behind, because the copy+split takes a loss spike at growth.
The question that decides everything: train LONGER and see whether recruiting the capacity repays the spike.
Also test net2net + a higher LR on the new dims, since D showed extra LR helps recruitment."""
import sys,copy; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("fix_growth2")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; PRE=1200; ADD=128; D0=256
CHECKS=[1500,3000,5000,7000]
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(6)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def contrib(m,cfg,lo,hi):
    base=evl(m,cfg); sl=slice(lo,hi); sav={}
    with torch.no_grad():
        for i,L in enumerate(m.layers):
            sav[i]=(L.Wo.weight[sl,:].clone(),L.m2.weight[sl,:].clone())
            L.Wo.weight[sl,:]=0; L.m2.weight[sl,:]=0
    d=evl(m,cfg)-base
    with torch.no_grad():
        for i,L in enumerate(m.layers):
            a,b=sav[i]; L.Wo.weight[sl,:]=a; L.m2.weight[sl,:]=b
    return base,d
def grow_net2net(m,add,noise=0.01):
    D=m.cfg.D
    with torch.no_grad():
        act=m.emb.weight.float().pow(2).mean(0); src=torch.topk(act,add).indices
    G.grow_width(m,add)
    with torch.no_grad():
        m.emb.weight[:,D:]=m.emb.weight[:,src]*0.5; m.emb.weight[:,src]*=0.5
        for L in m.layers:
            L.Wo.weight[D:]=L.Wo.weight[src]*0.5; L.Wo.weight[src]*=0.5
            L.m2.weight[D:]=L.m2.weight[src]*0.5; L.m2.weight[src]*=0.5
            for nm in ('Wk','Wq','Wv'):
                W=getattr(L,nm).weight; W[:,D:]=W[:,src]+torch.randn_like(W[:,src])*noise
            L.m1.weight[:,D:]=L.m1.weight[:,src]
        m.head.weight[:,D:]=m.head.weight[:,src]*0.5; m.head.weight[:,src]*=0.5
    return m.cfg.D
def run(tag,growfn,newlr_mult,sd0):
    torch.manual_seed(1); np.random.seed(1)
    cfg=C.Config(D=D0,DIN=8,DV=D0,NL=4,T=CH,res_scale=4**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd0); growfn(m); cfg=m.cfg
    if newlr_mult==1.0: opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    else:
        newp=[]; 
        for L in m.layers: newp+=[L.Wo.weight,L.m2.weight]
        ids={id(p) for p in newp}; old=[p for p in m.parameters() if id(p) not in ids]
        opt=torch.optim.Adam([{'params':old,'lr':1e-3},{'params':newp,'lr':1e-3*newlr_mult}])
    out={}
    for step in range(1,max(CHECKS)+1):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if step in CHECKS:
            lo,d=contrib(m,cfg,D0,D0+ADD); out[step]=(lo,d)
            print(f"    {tag:24s} @{step:5d}: loss {lo:.4f}  new-dim load {d:+.4f}",flush=True)
    del m; torch.cuda.empty_cache(); return out
if __name__=="__main__":
    torch.manual_seed(0); np.random.seed(0)
    cfg0=C.Config(D=D0,DIN=8,DV=D0,NL=4,T=CH,res_scale=4**-0.5)
    seed=C.Core(cfg0).to(DEV)
    o=torch.optim.Adam(seed.parameters(),lr=1e-3)
    for it in range(PRE):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(seed(X[:,:CH]).reshape(-1,cfg0.vocab),X[:,1:CH+1].reshape(-1))
        o.zero_grad(); l.backward(); o.step()
    sd0=copy.deepcopy(seed.state_dict())
    print(f"DOES RECRUITED CAPACITY WIN? seed loss {evl(seed,cfg0):.4f}, grow +{ADD} onto D={D0}\n",flush=True)
    del seed; torch.cuda.empty_cache()
    R={}
    R['zero (current)']  = run("A zero (current)",   lambda m:G.grow_width(m,ADD), 1.0, sd0)
    R['net2net']         = run("C net2net",          lambda m:grow_net2net(m,ADD), 1.0, sd0)
    R['net2net + 3x LR'] = run("E net2net + 3x LR",  lambda m:grow_net2net(m,ADD), 3.0, sd0)
    print(f"\n  {'steps after grow':>18s} " + " ".join(f"{k:>20s}" for k in R),flush=True)
    for s in CHECKS:
        print(f"  {s:18d} " + " ".join(f"{R[k][s][0]:9.4f} ({R[k][s][1]:+.3f})" for k in R),flush=True)
    fin={k:R[k][max(CHECKS)][0] for k in R}
    b=min(fin,key=fin.get)
    print(f"\n  BEST FINAL LOSS: {b} at {fin[b]:.4f}",flush=True)
    print(f"  vs current growth: {fin['zero (current)']-fin[b]:+.4f} nats",flush=True)
