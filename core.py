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
    def forward(s,a,state=None):
        """`state`=(M,zM): the PERSISTENT fast-weight memory carried from previous chunks. Reading it costs nothing
        extra and is what gives UNBOUNDED context at CONSTANT memory; carrying it during training is what teaches the
        model to use its own memory. The returned new state is detached (no cross-chunk backprop)."""
        cfg=s.cfg; B,T,_=a.shape; H,nrm=rmsnorm(a); Hs=_shift(H)
        xk=Hs@s.Wk.weight.T; kf=taylor(xk,cfg)                 # writes (shifted keys)
        xq=H@s.Wq.weight.T;  qf=taylor(xq,cfg)                 # reads
        v=H@s.Wv.weight.T
        A=torch.einsum('btf,bsf->bts',qf,kf)                   # retrieval weights beta_{t,s} = qf_t . kf_s
        A=A.tril()                                             # causal: read at t sees writes s<=t
        den=A.sum(-1,keepdim=True); num=torch.einsum('bts,bsd->btd',A,v)
        if state is not None:                                  # + read the persistent store
            M,zM=state
            num=num+torch.einsum('btf,bfd->btd',qf,M); den=den+torch.einsum('btf,bf->bt',qf,zM).unsqueeze(-1)
        den=den+1e-5
        r=num/den; o=r@s.Wo.weight.T+s.Wo.bias
        newM=torch.einsum('btf,btd->bfd',kf,v).detach(); newz=kf.sum(1).detach()
        if state is not None: newM=state[0]+newM; newz=state[1]+newz
        return a+s.rs*o, (H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r,(newM,newz))
    def local_vjp(s,a,cache,dout):
        cfg=s.cfg; H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r,_st=cache
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
    def forward(s,X,states=None,return_states=False):
        """states: list of per-layer (M,zM) carried from the previous chunk (persistent memory / long context)."""
        a=s.emb(X); ns=[]
        for i,L in enumerate(s.layers):
            a,c=L(a,None if states is None else states[i]); ns.append(c[-1])
        h,_=rmsnorm(a); lg=s.head(h)
        return (lg,ns) if return_states else lg
    def decay_states(s,states,d):
        return None if states is None else [(M*d,z*d) for (M,z) in states]
    def param_dict(s):
        p={'emb':s.emb.weight,'hW':s.head.weight,'hb':s.head.bias}
        for li,L in enumerate(s.layers):
            for n in ('Wk','Wq','Wv','Wo'): p[f'{li}.{n}']=getattr(L,n).weight
            p[f'{li}.bo']=L.Wo.bias
        return p

class LocalAdam:
    """Applies the LOCAL gradients. `adaptive=True` uses a SCALE-FREE relative step (trust ratio, LAMB-style):
    each parameter moves a fixed FRACTION of its own norm, so the hyperparameter does not need re-tuning when the
    model grows or scales (VISION 0a: no hand-tuned per-scale constants). `min_norm` lets zero-initialized new
    capacity (function-preserving growth) start moving instead of being frozen at trust=0."""
    def __init__(s,prm,lr=3e-3,b1=0.9,b2=0.999,eps=1e-8,adaptive=False,min_norm=1e-3,cap=10.0):
        s.prm=prm; s.lr=lr; s.b1=b1; s.b2=b2; s.eps=eps; s.i=0
        s.adaptive=adaptive; s.min_norm=min_norm; s.cap=cap
        s.m={k:torch.zeros_like(p) for k,p in prm.items()}; s.v={k:torch.zeros_like(p) for k,p in prm.items()}
    def step(s,g):
        s.i+=1
        with torch.no_grad():
            for k in s.prm:
                s.m[k]=s.b1*s.m[k]+(1-s.b1)*g[k]; s.v[k]=s.b2*s.v[k]+(1-s.b2)*g[k]**2
                u=(s.m[k]/(1-s.b1**s.i))/((s.v[k]/(1-s.b2**s.i)).sqrt()+s.eps)
                if s.adaptive:
                    wn=s.prm[k].norm().clamp_min(s.min_norm); un=u.norm().clamp_min(s.eps)
                    u=u*torch.clamp(wn/un,max=s.cap)   # relative step: move a fixed fraction of ||w||
                s.prm[k]-=s.lr*u
