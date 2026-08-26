"""DECISIVE: does the layerwise-local gap stay FLAT with depth? (= true depth unlocked under pure local learning)"""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L; from data_mix import DataMix
DEV=C.DEV; T=32; B=32; STEPS=400; dm=DataMix(ctx=T,verbose=False)
def fresh(NL):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T,res_scale=NL**-0.5); return cfg,C.Core(cfg).to(DEV)
def evl(m,cfg,nb=8):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def run(NL,bptt,lr):
    cfg,m=fresh(NL); Bfb=L.make_feedback(cfg,NL); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    opt=torch.optim.Adam(m.parameters(),lr=lr) if bptt else C.LocalAdam(m.param_dict(),lr=lr)
    for i in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV)
        if bptt:
            loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        else: opt.step(L.lw_grads(m,X,Bfb))
    return evl(m,cfg),time.time()-t0
print("DEPTH TEST: layerwise-local vs BPTT, each at its own best LR, 400 steps, real 290B mix",flush=True)
print(f"  {'NL':>3s} {'BPTT':>8s} {'LAYERWISE':>10s} {'gap':>7s} {'cost':>6s}",flush=True)
for NL in (2,4,6,12):
    b,bt=run(NL,True,1e-2); l,lt=run(NL,False,3e-3)
    print(f"  {NL:3d} {b:8.4f} {l:10.4f} {l-b:+7.3f} {lt/bt:5.2f}x",flush=True)
