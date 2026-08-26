"""Pragnosia CORE: deep-narrow recurrent VChunkRecall, trained by PURE LOCAL PLASTICITY (no backprop, no autograd).
Generalizes the proven de-risk math to N layers. Each layer = recurrent linear-attn (2nd-order Taylor feature map)
with key->NEXT-value SHIFT (proven required) + residual. Temporal credit = eligibility traces (EXACT). Depth credit =
Predictive Coding settling (feedforward-init, K steps, small eta -> ~backprop). This module IS the reusable body; the
self-test scales DEPTH (2..8 layers) to check PC-depth holds and to find K(depth) for §0b2 efficiency.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda" if torch.cuda.is_available() else "cpu"

class Config:
    def __init__(s, D=96, DIN=8, DV=96, NL=4, T=32, vocab=32768, res_scale=1.0):
        s.D=D; s.DIN=DIN; s.DV=DV; s.NL=NL; s.T=T; s.vocab=vocab; s.res_scale=res_scale
        s.iu=torch.triu_indices(DIN,DIN,device=DEV); s.FE=1+DIN+s.iu.shape[1]

def taylor(x,cfg):
    B=x.shape[0]; qd=(x.unsqueeze(2)*x.unsqueeze(1))[:,cfg.iu[0],cfg.iu[1]]
    return torch.cat([torch.ones(B,1,device=x.device),x,qd],1)
def tvjp(x,g,cfg):
    dx=g[:,1:1+cfg.DIN].clone(); gq=g[:,1+cfg.DIN:]
    dx.index_add_(1,cfg.iu[0],gq*x[:,cfg.iu[1]]); dx.index_add_(1,cfg.iu[1],gq*x[:,cfg.iu[0]]); return dx

class Layer(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg; s.rs=getattr(cfg,'res_scale',1.0)
        s.Wk=nn.Linear(cfg.D,cfg.DIN,bias=False); s.Wq=nn.Linear(cfg.D,cfg.DIN,bias=False)
        s.Wv=nn.Linear(cfg.D,cfg.DV,bias=False); s.Wo=nn.Linear(cfg.DV,cfg.D)
    def forward(s,a):
        cfg=s.cfg; B,Tt,_=a.shape; S=torch.zeros(B,cfg.FE,cfg.DV,device=DEV); z=torch.zeros(B,cfg.FE,device=DEV); outs=[]; cache=[]
        for t in range(Tt):
            src=a[:,t-1] if t>0 else torch.zeros_like(a[:,0]); xk=s.Wk(src); kf=taylor(xk,cfg); v=s.Wv(a[:,t])
            S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
            xq=s.Wq(a[:,t]); qf=taylor(xq,cfg); num=torch.einsum('bf,bfd->bd',qf,S); den=(qf*z).sum(-1,keepdim=True)+1e-5
            o=s.Wo(num/den); outs.append(a[:,t]+s.rs*o); cache.append((src,xk,kf,v,xq,qf,num,den,S.clone()))
        return torch.stack(outs,1), cache
    def local_vjp(s,a,cache,dout):                            # dout=dL/da_out -> (param grads, dL/da_in). Pure local, no autograd.
        cfg=s.cfg; B,Tt,_=a.shape
        W={'Wk':s.Wk.weight.detach(),'Wq':s.Wq.weight.detach(),'Wv':s.Wv.weight.detach(),'Wo':s.Wo.weight.detach()}
        dWk=torch.zeros_like(W['Wk']);dWq=torch.zeros_like(W['Wq']);dWv=torch.zeros_like(W['Wv']);dWo=torch.zeros_like(W['Wo']);dbo=torch.zeros_like(s.Wo.bias)
        din=dout.clone(); e_num=[];g_den=[]
        for t in range(Tt):
            (src,xk,kf,v,xq,qf,num,den,S)=cache[t]; r=num/den; ert=dout[:,t]*s.rs
            dWo+=ert.T@r; dbo+=ert.sum(0); err=ert@W['Wo']; e_num.append(err/den); g_den.append(-(err*num).sum(-1,keepdim=True)/den**2)
        zt=[];zr=torch.zeros(B,cfg.FE,device=DEV)
        for t in range(Tt): zr=zr+cache[t][2]; zt.append(zr.clone())
        P=torch.zeros(B,cfg.FE,cfg.DV,device=DEV); Q=torch.zeros(B,cfg.FE,device=DEV)
        for t in reversed(range(Tt)):
            (src,xk,kf,v,xq,qf,num,den,S)=cache[t]
            P=P+torch.einsum('bf,bd->bfd',qf,e_num[t]); Q=Q+g_den[t]*qf
            dv=torch.einsum('bf,bfd->bd',kf,P); dWv+=dv.T@a[:,t]; din[:,t]+=dv@W['Wv']
            dkf=torch.einsum('bfd,bd->bf',P,v)+Q; dxk=tvjp(xk,dkf,cfg); dWk+=dxk.T@src
            if t>0: din[:,t-1]+=dxk@W['Wk']
            dqf=torch.einsum('bfd,bd->bf',S,e_num[t])+g_den[t]*zt[t]; dxq=tvjp(xq,dqf,cfg); dWq+=dxq.T@a[:,t]; din[:,t]+=dxq@W['Wq']
        return {'Wk':dWk,'Wq':dWq,'Wv':dWv,'Wo':dWo,'bo':dbo}, din

class Core(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg; s.emb=nn.Embedding(cfg.vocab,cfg.D); s.layers=nn.ModuleList([Layer(cfg) for _ in range(cfg.NL)])
        s.head=nn.Linear(cfg.D,cfg.vocab); nn.init.normal_(s.emb.weight,0,0.02)
    def forward(s,X):
        a=s.emb(X)
        for L in s.layers: a,_=L(a)
        return s.head(a)
    def param_dict(s):
        p={'emb':s.emb.weight,'hW':s.head.weight,'hb':s.head.bias}
        for li,L in enumerate(s.layers):
            for n in ('Wk','Wq','Wv','Wo'): p[f'{li}.{n}']=getattr(L,n).weight
            p[f'{li}.bo']=L.Wo.bias
        return p

def local_grads(m,X,K=None,eta=0.05):
    """Pure-local gradient of next-token loss: eligibility (time, exact) + PC settling (depth, ~backprop). No autograd."""
    cfg=m.cfg; NL=cfg.NL; T=cfg.T; D=cfg.D; K=K or max(6,3*NL); Y=X[:,1:T+1]; B=X.shape[0]
    with torch.no_grad():
        a0=m.emb(X[:,:T]); ahat=[a0]; a=a0
        for L in m.layers: o,_=L(a); ahat.append(o); a=o
        aL=[ai.clone() for ai in ahat]
        for _ in range(K):                                    # PC settling: descend energy; error front propagates 1 layer/step
            lg=aL[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,cfg.vocab)[torch.arange(B*T,device=DEV),Y.reshape(-1)]-=1.0; rd=(e@m.head.weight)/(B*T)
            epsl=[None]*(NL+1); pca=[None]*NL
            for li,L in enumerate(m.layers): o,ca=L(aL[li]); pca[li]=ca; epsl[li+1]=aL[li+1]-o
            aL[NL]=aL[NL]-eta*(epsl[NL]+rd)
            for li in range(NL-1,0,-1):
                _,din=m.layers[li].local_vjp(aL[li],pca[li],epsl[li+1]); aL[li]=aL[li]-eta*(epsl[li]-din)
        g={}; lg=aL[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
        e=p.clone(); e.view(-1,cfg.vocab)[torch.arange(B*T,device=DEV),Y.reshape(-1)]-=1.0; e=e/(B*T)
        g['hW']=e.reshape(-1,cfg.vocab).T@aL[NL].reshape(-1,D); g['hb']=e.reshape(-1,cfg.vocab).sum(0); din0=None
        for li,L in enumerate(m.layers):
            o,ca=L(aL[li]); eps_l=aL[li+1]-o; pg,din=L.local_vjp(aL[li],ca,-eps_l)
            for n in ('Wk','Wq','Wv','Wo'): g[f'{li}.{n}']=pg[n]
            g[f'{li}.bo']=pg['bo']
            if li==0: din0=din
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); g['emb']=gemb
    return g

class LocalAdam:                                              # optimizer = substrate math applied to LOCAL grads (not backprop)
    def __init__(s,prm,lr=3e-3,b1=0.9,b2=0.999,eps=1e-8):
        s.prm=prm; s.lr=lr; s.b1=b1; s.b2=b2; s.eps=eps; s.i=0
        s.m={k:torch.zeros_like(p) for k,p in prm.items()}; s.v={k:torch.zeros_like(p) for k,p in prm.items()}
    def step(s,g):
        s.i+=1
        with torch.no_grad():
            for k in s.prm:
                s.m[k]=s.b1*s.m[k]+(1-s.b1)*g[k]; s.v[k]=s.b2*s.v[k]+(1-s.b2)*g[k]**2
                s.prm[k]-=s.lr*(s.m[k]/(1-s.b1**s.i))/((s.v[k]/(1-s.b2**s.i)).sqrt()+s.eps)

if __name__=="__main__":
    import sys; sys.path.insert(0,"/opt/code/membrae"); from data_mix import DataMix
    T=32; dm=DataMix(ctx=T, verbose=False)
    print("DEPTH-SCALING: does PC-depth credit hold as layers grow? cosine(local-grad, backprop-grad) per depth",flush=True)
    print("  (K=3*NL, eta=0.05; min-cosine over all params; >0.97 = PC-depth holds at that depth)",flush=True)
    for NL in (2,4,6,8):
        cfg=Config(D=96,DIN=8,DV=96,NL=NL,T=T); m=Core(cfg).to(DEV); X=torch.tensor(dm.batch(24),device=DEV)
        # true grad
        Y=X[:,1:T+1]; lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),Y.reshape(-1)); m.zero_grad(); loss.backward()
        tg={'emb':m.emb.weight.grad.clone(),'hW':m.head.weight.grad.clone()}
        for li,L in enumerate(m.layers):
            for n in ('Wk','Wv','Wo'): tg[f'{li}.{n}']=getattr(L,n).weight.grad.clone()
        K=3*NL; g=local_grads(m,X,K=K,eta=0.05)
        cs=[F.cosine_similarity(g[k].flatten(),tg[k].flatten(),dim=0).item() for k in tg]
        print(f"  NL={NL:<2d} K={K:<3d}: min-cos={min(cs):.4f} mean={np.mean(cs):.4f}  (bottom-layer 0.Wk={F.cosine_similarity(g['0.Wk'].flatten(),tg['0.Wk'].flatten(),dim=0).item():.3f})",flush=True)
