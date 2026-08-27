"""LONG-HORIZON CONFIRMATION of the final rule (local objectives + residual-highway feedback) vs BPTT, 4000 steps.
The previous rule looked fine early then plateaued -- this checks the new rule holds over a long run."""
import sys,time,json; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L; from data_mix import DataMix
DEV=C.DEV; NL=6; T=32; B=32; STEPS=4000; RS=NL**-0.5; dm=DataMix(ctx=T,verbose=False)
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T,res_scale=RS); return cfg,C.Core(cfg).to(DEV)
def evl(m,cfg,nb=16):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def run(arm,lr):
    cfg,m=fresh(); torch.manual_seed(1); np.random.seed(1); t0=time.time(); cur=[]
    opt=torch.optim.Adam(m.parameters(),lr=lr) if arm=='bptt' else C.LocalAdam(m.param_dict(),lr=lr)
    for i in range(1,STEPS+1):
        X=torch.tensor(dm.batch(B),device=DEV)
        if arm=='bptt':
            loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        else: opt.step(L.lw_grads(m,X))
        if i%500==0:
            e=evl(m,cfg); cur.append((i,e,time.time()-t0)); print(f"    {arm:10s} {i:5d} heldout={e:.4f} ({time.time()-t0:.0f}s)",flush=True)
    return cur
print(f"LONG-HORIZON CONFIRMATION @ NL={NL}, {STEPS} steps: final local rule vs BPTT",flush=True)
lw=run('local',3e-3); bp=run('bptt',1e-2)
print("\n  step |    BPTT | LOCAL-rule |    gap",flush=True)
for (i,b,_),(j,l,_) in zip(bp,lw): print(f"  {i:5d} | {b:7.4f} | {l:10.4f} | {l-b:+7.3f}",flush=True)
print(f"\nfinal gap = {lw[-1][1]-bp[-1][1]:+.3f}  (negative = local BEATS backprop)",flush=True)
json.dump({'bptt':bp,'local':lw},open('long3.json','w'))
