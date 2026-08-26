"""LONG-HORIZON v2 on the NORMALIZED core: layerwise-local vs BPTT, 4000 steps. Does the gap shrink with training?"""
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
    cfg,m=fresh(); Bfb=L.make_feedback(cfg,NL); torch.manual_seed(1); np.random.seed(1); t0=time.time(); cur=[]
    opt=torch.optim.Adam(m.parameters(),lr=lr) if arm=='bptt' else C.LocalAdam(m.param_dict(),lr=lr)
    for i in range(1,STEPS+1):
        X=torch.tensor(dm.batch(B),device=DEV)
        if arm=='bptt':
            loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        else: opt.step(L.lw_grads(m,X,Bfb))
        if i%500==0:
            e=evl(m,cfg); cur.append((i,e,time.time()-t0)); print(f"    {arm:10s} {i:5d} heldout={e:.4f} ({time.time()-t0:.0f}s)",flush=True)
    return cur
print(f"LONG HORIZON v2 (normalized core) @ NL={NL}, {STEPS} steps",flush=True)
bp=run('bptt',1e-2); lw=run('layerwise',3e-3)
print("\n  step |    BPTT | LAYERWISE |    gap",flush=True)
for (i,b,_),(j,l,_) in zip(bp,lw): print(f"  {i:5d} | {b:7.4f} | {l:9.4f} | {l-b:+7.3f}",flush=True)
g0=lw[0][1]-bp[0][1]; g1=lw[-1][1]-bp[-1][1]
print(f"\ngap @{bp[0][0]}: {g0:+.3f}  ->  gap @{bp[-1][0]}: {g1:+.3f}   (change {g1-g0:+.3f})",flush=True)
bt=bp[-1][2]; eq=min([l for (i,l,t) in lw if t<=bt] or [lw[-1][1]])
print(f"EQUAL WALL-CLOCK ({bt:.0f}s): BPTT={bp[-1][1]:.4f} LAYERWISE={eq:.4f} gap={eq-bp[-1][1]:+.3f}",flush=True)
print(("VERDICT: gap CLOSING -> proceed to build" if g1<g0-0.1 else "VERDICT: gap flat/growing -> grind stability"),flush=True)
json.dump({'bptt':bp,'layerwise':lw},open('long2.json','w'))
