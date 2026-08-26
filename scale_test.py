"""SCALE + LONG-HORIZON: does the local rule still beat backprop at larger scale over a long run, or does BPTT
catch up (the gap has been shrinking ~x0.6 per doubling)? D=256, NL=6, T=128, 20000 steps (~82M tokens).
Third arm checks that stochastic local objectives (k_local=2) preserve quality at much lower cost."""
import sys,time,json; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L; from data_mix import DataMix
DEV=C.DEV; D=256; NL=6; T=128; B=32; STEPS=20000; RS=NL**-0.5; dm=DataMix(ctx=T,verbose=False)
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=RS); return cfg,C.Core(cfg).to(DEV)
def evl(m,cfg,nb=12):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def run(arm,lr,k=None):
    cfg,m=fresh(); torch.manual_seed(1); np.random.seed(1); t0=time.time(); cur=[]
    opt=torch.optim.Adam(m.parameters(),lr=lr) if arm=='bptt' else C.LocalAdam(m.param_dict(),lr=lr)
    for i in range(1,STEPS+1):
        X=torch.tensor(dm.batch(B),device=DEV)
        if arm=='bptt':
            loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        else: opt.step(L.lw_grads(m,X,k_local=k))
        if i%2000==0:
            e=evl(m,cfg); cur.append((i,e,time.time()-t0)); print(f"    {arm}{'' if k is None else '-k%d'%k:>4s} {i:6d} heldout={e:.4f} ({time.time()-t0:.0f}s)",flush=True)
    return cur
print(f"SCALE TEST D={D} NL={NL} T={T} B={B} steps={STEPS} (~{STEPS*B*T/1e6:.0f}M tokens)",flush=True)
res={}
res['local']=run('local',3e-3); res['local_k2']=run('local',3e-3,k=2); res['bptt']=run('bptt',1e-2)
print("\n   step |    BPTT |   LOCAL |  gap  | LOCAL-k2 | gap-k2",flush=True)
for (i,b,_),(j,l,_),(q,l2,_) in zip(res['bptt'],res['local'],res['local_k2']):
    print(f"  {i:6d} | {b:7.4f} | {l:7.4f} | {l-b:+.3f} | {l2:8.4f} | {l2-b:+.3f}",flush=True)
gs=[l-b for (i,b,_),(j,l,_) in zip(res['bptt'],res['local'])]
print(f"\ngap trend: {' -> '.join(f'{g:+.3f}' for g in gs)}",flush=True)
print(("VERDICT: local still AHEAD at scale+long horizon" if gs[-1]<0 else "VERDICT: BPTT CAUGHT UP -- user's hypothesis confirmed"),flush=True)
json.dump(res,open('scale_test.json','w'))
