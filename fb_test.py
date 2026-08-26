"""GRIND: fix the greedy PLATEAU. Idea: with residual layers + a shared readout, dL/da_top already lives in every
layer's space (it IS the residual-highway term that dominates the true gradient). So broadcast it DIRECTLY (identity
feedback) instead of through a random DFA matrix -- still a global error broadcast, no weight transport."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L; from data_mix import DataMix
DEV=C.DEV; NL=6; T=32; B=32; RS=NL**-0.5; STEPS=2000; dm=DataMix(ctx=T,verbose=False)
def grads(m,X,Bfb,lam,mode,wloc=1.0):
    cfg=m.cfg; V=cfg.vocab; D=cfg.D; Y=X[:,1:T+1]; Bs=X.shape[0]; ar=torch.arange(Bs*T,device=DEV)
    with torch.no_grad():
        acts=[]; a_in=m.emb(X[:,:T]); caches=[]
        for l in range(NL):
            acts.append(a_in); a_out,c=m.layers[l](a_in); caches.append(c); a_in=a_out
        acts.append(a_in)
        g={}; dW=torch.zeros_like(m.head.weight); db=torch.zeros_like(m.head.bias)
        dw_t,db_t,dtop=L._readout(m,acts[NL],Y,ar,Bs,T,V,D)
        demb=torch.zeros_like(acts[0])
        for l in range(NL):
            dwl,dbl,dout=L._readout(m,acts[l+1],Y,ar,Bs,T,V,D)
            dW+=dwl*wloc; db+=dbl*wloc
            fb = dtop if mode=='identity' else (dtop@Bfb[l])
            pg,din=m.layers[l].local_vjp(acts[l],caches[l],wloc*dout+lam*fb)
            for n in ('Wk','Wq','Wv','Wo'): g[f'{l}.{n}']=pg[n]
            g[f'{l}.bo']=pg['bo']
            if l==0: demb=demb+din
        dwe,dbe,de=L._readout(m,acts[0],Y,ar,Bs,T,V,D); dW+=dwe*wloc; db+=dbe*wloc
        demb=demb+wloc*de+lam*(dtop if mode=='identity' else dtop@Bfb[NL])
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),demb.reshape(-1,D))
        g['emb']=gemb; g['hW']=dW; g['hb']=db
    return g
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T,res_scale=RS); return cfg,C.Core(cfg).to(DEV)
def evl(m,cfg,nb=16):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def run(tag,lam,mode,wloc=1.0,lr=3e-3):
    cfg,m=fresh(); Bfb=L.make_feedback(cfg,NL); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    opt=C.LocalAdam(m.param_dict(),lr=lr)
    for i in range(1,STEPS+1):
        X=torch.tensor(dm.batch(B),device=DEV); opt.step(grads(m,X,Bfb,lam,mode,wloc))
    e=evl(m,cfg); print(f"  {tag:30s} heldout={e:.4f} ({time.time()-t0:.0f}s)",flush=True); return e
print(f"PLATEAU GRIND @ NL={NL}, {STEPS} steps. BPTT ref @2000 = 5.5359; layerwise(random DFA,l=1) = 6.1968",flush=True)
run("random DFA lam=1 (current)",1.0,'random')
run("IDENTITY fb lam=1",1.0,'identity')
run("IDENTITY fb lam=2",2.0,'identity')
run("IDENTITY fb lam=4",4.0,'identity')
run("IDENTITY lam=4, local w=0.25",4.0,'identity',wloc=0.25)
run("IDENTITY lam=1, NO local (w=0)",1.0,'identity',wloc=0.0)
