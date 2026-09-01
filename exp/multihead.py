"""OUR ARCHITECTURE IS 1.02 NATS BEHIND A TRANSFORMER 9.6x SMALLER TRAINED ON 29x FEWER TOKENS. Not a token
problem. The biggest structural difference: a transformer runs H INDEPENDENT attention heads per layer, each free
to retrieve a different relation. We run ONE. A single head per layer can only express one retrieval pattern, no
matter how many features it has.

Test multi-head in our mixer: split the key/query projection into H groups, each with its own Taylor map and its
own slice of the value space, exactly as multi-head attention does. Same total parameters, same FE budget."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("multihead")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; CH=512; B=16; D=384; NL=6; STEPS=4000
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
def taylor_h(x,iu):
    """Taylor map applied per head: x is [B,T,H,dh] -> [B,T,H,fe]"""
    qd=(x.unsqueeze(-1)*x.unsqueeze(-2))[...,iu[0],iu[1]]
    return torch.cat([torch.ones(*x.shape[:-1],1,device=x.device,dtype=x.dtype),x,qd],-1)
class MHLayer(nn.Module):
    """our mixer with H independent heads"""
    def __init__(s,D,dh,H,DV,rs,mult=4):
        super().__init__(); s.H=H; s.dh=dh; s.rs=rs; s.dv=DV//H
        s.Wk=nn.Linear(D,H*dh,bias=False); s.Wq=nn.Linear(D,H*dh,bias=False)
        s.Wv=nn.Linear(D,DV,bias=False); s.Wo=nn.Linear(DV,D)
        s.m1=nn.Linear(D,mult*D); s.m2=nn.Linear(mult*D,D)
        s.register_buffer('iu',torch.triu_indices(dh,dh),persistent=False)
    def forward(s,a):
        Bs,T,_=a.shape; H,dh=s.H,s.dh
        Hn,_=C.rmsnorm(a); Hs=torch.cat([torch.zeros_like(Hn[:,:1]),Hn[:,:-1]],1)
        kf=taylor_h(s.Wk(Hs).view(Bs,T,H,dh),s.iu)          # [B,T,H,fe]
        qf=taylor_h(s.Wq(Hn).view(Bs,T,H,dh),s.iu)
        v=s.Wv(Hn).view(Bs,T,H,s.dv)
        A=torch.einsum('bthf,bshf->bhts',qf,kf).tril()
        den=A.sum(-1,keepdim=True)+1e-5
        num=torch.einsum('bhts,bshd->bthd',A,v)
        r=(num/den.permute(0,2,1,3)).reshape(Bs,T,-1)
        a1=a+s.rs*(r@s.Wo.weight.T+s.Wo.bias)
        H2,_=C.rmsnorm(a1)
        return a1+s.rs*(F.gelu(H2@s.m1.weight.T+s.m1.bias)@s.m2.weight.T+s.m2.bias)
class MHCore(nn.Module):
    def __init__(s,D,dh,H,NL,V=32768):
        super().__init__(); s.emb=nn.Embedding(V,D)
        s.layers=nn.ModuleList([MHLayer(D,dh,H,D,NL**-0.5) for _ in range(NL)])
        s.head=nn.Linear(D,V); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0,0.02)
    def forward(s,X):
        a=s.emb(X)
        for L in s.layers: a=L(a)
        h,_=C.rmsnorm(a); return s.head(h)
def evl(fwd):
    t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,32768).float(),X[:,1:CH+1].reshape(-1)).item()
    return t/len(EV)
def train(m,tag):
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95)); t0=time.time()
    for i in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,32768),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return evl(m), STEPS*B*CH/(time.time()-t0)
if __name__=="__main__":
    def fe(d): return 1+d+d*(d+1)//2
    print(f"MULTI-HEAD vs SINGLE HEAD in our mixer. D={D}, {NL} layers, {STEPS} steps\n",flush=True)
    print(f"  {'config':>26s} {'heads':>6s} {'dh':>4s} {'FE/head':>8s} {'FE total':>9s} {'loss':>8s} {'ktok/s':>8s}",flush=True)
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=24,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    o=C.Core(cfg).to(DEV); l0,s0=train(o,"single")
    print(f"  {'ours now (1 head, DIN=24)':>26s} {1:6d} {24:4d} {fe(24):8d} {fe(24):9d} {l0:8.4f} {s0/1e3:8.0f}",flush=True)
    del o; torch.cuda.empty_cache()
    for H,dh in ((4,12),(8,8),(6,10)):
        torch.manual_seed(0); np.random.seed(0)
        m=MHCore(D,dh,H,NL).to(DEV); l,sp=train(m,f"{H}h")
        print(f"  {f'{H} heads x dh={dh}':>26s} {H:6d} {dh:4d} {fe(dh):8d} {H*fe(dh):9d} {l:8.4f} {sp/1e3:8.0f}",flush=True)
        del m; torch.cuda.empty_cache()
    print(f"\n  transformer control at this budget: 3.15 (21M params, 20 tok/param)",flush=True)
