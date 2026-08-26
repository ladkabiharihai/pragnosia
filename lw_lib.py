"""Layerwise-local learning rule (the depth solution): per-layer local next-token objective through the shared
readout + DFA broadcast of the top error via FIXED RANDOM feedback + an embedding local objective.
NO backprop, NO settling, NO weight transport. Within-layer credit is EXACT (eligibility traces)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV
def make_feedback(cfg,NL,seed=7):
    gen=torch.Generator(device=DEV); gen.manual_seed(seed)
    return [torch.randn(cfg.D,cfg.D,generator=gen,device=DEV)/cfg.D**0.5 for _ in range(NL+1)]
def lw_grads(m,X,Bfb,lam=1.0,emb_obj=True):
    cfg=m.cfg; NL=cfg.NL; T=cfg.T; V=cfg.vocab; D=cfg.D
    Y=X[:,1:T+1]; Bs=X.shape[0]; ar=torch.arange(Bs*T,device=DEV)
    with torch.no_grad():
        acts=[]; a_in=m.emb(X[:,:T]); caches=[]
        for l in range(NL):
            acts.append(a_in); a_out,c=m.layers[l](a_in); caches.append(c); a_in=a_out
        acts.append(a_in)
        lgt=a_in@m.head.weight.T+m.head.bias; pt=lgt.softmax(-1)
        et=pt.clone(); et.view(-1,V)[ar,Y.reshape(-1)]-=1.0; et=et/(Bs*T); dtop=et@m.head.weight
        g={}; dW=torch.zeros_like(m.head.weight); db=torch.zeros_like(m.head.bias); demb=torch.zeros_like(acts[0])
        for l in range(NL):
            a_out=acts[l+1]
            lg=a_out@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,V)[ar,Y.reshape(-1)]-=1.0; e=e/(Bs*T)
            dW+=e.reshape(-1,V).T@a_out.reshape(-1,D); db+=e.reshape(-1,V).sum(0)
            pg,din=m.layers[l].local_vjp(acts[l],caches[l],e@m.head.weight+lam*(dtop@Bfb[l]))
            for n in ('Wk','Wq','Wv','Wo'): g[f'{l}.{n}']=pg[n]
            g[f'{l}.bo']=pg['bo']
            if l==0: demb=demb+din
        if emb_obj:
            a0=acts[0]; lg0=a0@m.head.weight.T+m.head.bias; p0=lg0.softmax(-1)
            e0=p0.clone(); e0.view(-1,V)[ar,Y.reshape(-1)]-=1.0; e0=e0/(Bs*T)
            dW+=e0.reshape(-1,V).T@a0.reshape(-1,D); db+=e0.reshape(-1,V).sum(0); demb=demb+e0@m.head.weight
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),demb.reshape(-1,D))
        g['emb']=gemb; g['hW']=dW; g['hb']=db
    return g
