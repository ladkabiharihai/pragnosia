"""Train at scale with the best local settings, then GENERATION-test it (loss is not enough).
Also trains a BPTT model for the same steps so generation quality can be compared head-to-head."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L, gen_eval as G; from data_mix import DataMix
DEV=C.DEV; D=256; NL=6; T=128; B=32; STEPS=6000; dm=DataMix(ctx=T,verbose=False)
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); return cfg,C.Core(cfg).to(DEV)
def evl(m,cfg,nb=12):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def train(arm):
    cfg,m=fresh(); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    opt=torch.optim.Adam(m.parameters(),lr=1e-3) if arm=='bptt' else C.LocalAdam(m.param_dict(),lr=3e-3,adaptive=True)
    for i in range(1,STEPS+1):
        X=torch.tensor(dm.batch(B),device=DEV)
        if arm=='bptt':
            l=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)); opt.zero_grad(); l.backward(); opt.step()
        else: opt.step(L.lw_grads(m,X,lam=4.0,k_local=2))
        if i%2000==0: print(f"    {arm} {i} loss={evl(m,cfg):.4f} ({time.time()-t0:.0f}s)",flush=True)
    return m,cfg,evl(m,cfg)
print(f"GENERATION TEST: D={D} NL={NL} T={T}, {STEPS} steps",flush=True)
for arm in ('local','bptt'):
    m,cfg,l=train(arm); m.eval()
    print(f"\n=== {arm.upper()}  heldout-loss={l:.4f}  ppl={np.exp(l):.1f} ===",flush=True)
    G.report(m,tag=arm)
