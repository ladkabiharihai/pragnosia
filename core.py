"""Pragnosia CORE (fast, vectorized). Recurrent linear-attention with a symmetric 2nd-order Taylor feature map and a
key->NEXT-value SHIFT, residual + RMS pre-norm, trained by PURE LOCAL PLASTICITY (no backprop, no autograd).
The causal recurrence  num_t = sum_{s<=t} (qf_t . kf_s) v_s  is computed in PARALLEL as a masked quadratic form
(no Python time-loop, no per-step state materialization) -> orders of magnitude faster and far less memory.
`Layer.local_vjp` is the EXACT within-layer credit (eligibility traces in closed form), verified against autograd.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda" if torch.cuda.is_available() else "cpu"

class Config:
    def __init__(s, D=96, DIN=8, DV=96, NL=4, T=32, vocab=32768, res_scale=1.0):
        s.D=D; s.DIN=DIN; s.DV=DV; s.NL=NL; s.T=T; s.vocab=vocab; s.res_scale=res_scale
        s.iu=torch.triu_indices(DIN,DIN,device=DEV); s.FE=1+DIN+s.iu.shape[1]

def rmsnorm(x,eps=1e-6):
    n=(x.pow(2).mean(-1,keepdim=True)+eps).sqrt(); return x/n, n
def rmsnorm_vjp(x,n,dy):
    D_=x.shape[-1]; return dy/n - x*((dy*x).sum(-1,keepdim=True)/(D_*n.pow(3)))
def taylor(x,cfg):                                            # [...,DIN] -> [...,FE], symmetric 2nd-order
    qd=(x.unsqueeze(-1)*x.unsqueeze(-2))[...,cfg.iu[0],cfg.iu[1]]
    return torch.cat([torch.ones(*x.shape[:-1],1,device=x.device,dtype=x.dtype),x,qd],-1)
def tvjp(x,g,cfg):                                            # exact vjp of taylor
    dx=g[...,1:1+cfg.DIN].clone(); gq=g[...,1+cfg.DIN:]
    dx.index_add_(-1,cfg.iu[0],gq*x[...,cfg.iu[1]]); dx.index_add_(-1,cfg.iu[1],gq*x[...,cfg.iu[0]]); return dx
def _shift(x):                                                # key source is the PREVIOUS token (binds key_t -> value_{t+1})
    return torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],1)

class Layer(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg; s.rs=getattr(cfg,'res_scale',1.0)
        s.Wk=nn.Linear(cfg.D,cfg.DIN,bias=False); s.Wq=nn.Linear(cfg.D,cfg.DIN,bias=False)
        s.Wv=nn.Linear(cfg.D,cfg.DV,bias=False); s.Wo=nn.Linear(cfg.DV,cfg.D)
    def forward(s,a):
        cfg=s.cfg; B,T,_=a.shape; H,nrm=rmsnorm(a); Hs=_shift(H)
        xk=Hs@s.Wk.weight.T; kf=taylor(xk,cfg)                 # writes (shifted keys)
        xq=H@s.Wq.weight.T;  qf=taylor(xq,cfg)                 # reads
        v=H@s.Wv.weight.T
        A=torch.einsum('btf,bsf->bts',qf,kf)                   # retrieval weights beta_{t,s} = qf_t . kf_s
        A=A.tril()                                             # causal: read at t sees writes s<=t
        den=A.sum(-1,keepdim=True)+1e-5; num=torch.einsum('bts,bsd->btd',A,v)
        r=num/den; o=r@s.Wo.weight.T+s.Wo.bias
        return a+s.rs*o, (H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r)
    def local_vjp(s,a,cache,dout):
        cfg=s.cfg; H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r=cache
        Wk,Wq,Wv,Wo=(s.Wk.weight,s.Wq.weight,s.Wv.weight,s.Wo.weight)
        dO=dout*s.rs
        dWo=torch.einsum('btd,bte->de',dO,r); dbo=dO.sum((0,1)); dr=dO@Wo
        dnum=dr/den; dden=-(dr*num).sum(-1,keepdim=True)/den.pow(2)
        dA=(torch.einsum('btd,bsd->bts',dnum,v)+dden).tril()
        dv=torch.einsum('bts,btd->bsd',A,dnum)
        dqf=torch.einsum('bts,bsf->btf',dA,kf); dkf=torch.einsum('bts,btf->bsf',dA,qf)
        dxk=tvjp(xk,dkf,cfg); dxq=tvjp(xq,dqf,cfg)
        dWv=torch.einsum('btd,bte->de',dv,H); dWq=torch.einsum('btd,bte->de',dxq,H); dWk=torch.einsum('btd,bte->de',dxk,Hs)
        dH=dv@Wv+dxq@Wq
        dHs=dxk@Wk; dH=dH+torch.cat([dHs[:,1:],torch.zeros_like(dHs[:,:1])],1)   # un-shift
        return {'Wk':dWk,'Wq':dWq,'Wv':dWv,'Wo':dWo,'bo':dbo}, dout+rmsnorm_vjp(a,nrm,dH)

class Core(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg; s.emb=nn.Embedding(cfg.vocab,cfg.D)
        s.layers=nn.ModuleList([Layer(cfg) for _ in range(cfg.NL)]); s.head=nn.Linear(cfg.D,cfg.vocab)
        nn.init.normal_(s.emb.weight,0,0.02)
    def forward(s,X):
        a=s.emb(X)
        for L in s.layers: a,_=L(a)
        h,_=rmsnorm(a); return s.head(h)
    def param_dict(s):
        p={'emb':s.emb.weight,'hW':s.head.weight,'hb':s.head.bias}
        for li,L in enumerate(s.layers):
            for n in ('Wk','Wq','Wv','Wo'): p[f'{li}.{n}']=getattr(L,n).weight
            p[f'{li}.bo']=L.Wo.bias
        return p

class LocalAdam:
    def __init__(s,prm,lr=3e-3,b1=0.9,b2=0.999,eps=1e-8):
        s.prm=prm; s.lr=lr; s.b1=b1; s.b2=b2; s.eps=eps; s.i=0
        s.m={k:torch.zeros_like(p) for k,p in prm.items()}; s.v={k:torch.zeros_like(p) for k,p in prm.items()}
    def step(s,g):
        s.i+=1
        with torch.no_grad():
            for k in s.prm:
                s.m[k]=s.b1*s.m[k]+(1-s.b1)*g[k]; s.v[k]=s.b2*s.v[k]+(1-s.b2)*g[k]**2
                s.prm[k]-=s.lr*(s.m[k]/(1-s.b1**s.i))/((s.v[k]/(1-s.b2**s.i)).sqrt()+s.eps)
