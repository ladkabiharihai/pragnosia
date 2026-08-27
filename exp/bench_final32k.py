"""DEFINITIVE: training at 32768-token documents with the OPTIMAL chunk size, ours vs a param-matched transformer,
across the growth ladder 40M -> 200M -> 400M -> 1B. Answers: how efficient, how it scales, how much faster."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
DEV=C.DEV; V=32768; DOC=32768
class TBlock(nn.Module):
    def __init__(s,d,h=8):
        super().__init__(); s.h=h
        s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
        s.n2=nn.LayerNorm(d); s.m1=nn.Linear(d,4*d); s.m2=nn.Linear(4*d,d)
    def forward(s,x):
        B,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).chunk(3,-1)
        q,k,v=[t.view(B,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        x=x+s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D))
        return x+s.m2(F.gelu(s.m1(s.n2(x))))
class TLM(nn.Module):
    def __init__(s,d,nl):
        super().__init__(); s.emb=nn.Embedding(V,d); s.bl=nn.ModuleList([TBlock(d) for _ in range(nl)])
        s.nf=nn.LayerNorm(d); s.head=nn.Linear(d,V)
    def forward(s,X):
        x=s.emb(X)
        for b in s.bl: x=b(x)
        return s.head(s.nf(x))
def npar(m): return sum(p.numel() for p in m.parameters())
def t(f):
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    f(); torch.cuda.synchronize(); t0=time.time(); f(); torch.cuda.synchronize()
    return time.time()-t0, torch.cuda.max_memory_allocated()/2**30
if __name__=="__main__":
    print(f"TRAINING AT {DOC}-TOKEN DOCUMENTS -- ours (optimal chunk) vs param-matched transformer",flush=True)
    print(f"  {'scale':>6s} {'ours D x NL':>14s} {'params':>8s} {'ch':>5s} {'B':>2s} | {'OURS s':>7s} {'GiB':>6s} {'ktok/s':>7s} | {'TF s':>7s} {'GiB':>6s} {'ktok/s':>7s} | {'FASTER':>7s} {'LIGHTER':>8s}",flush=True)
    for tag,d,nl,B,CH in (("40M",512,12,2,2048),("200M",1536,24,1,1024),("400M",2048,36,1,1024),("1B",3072,40,1,1024)):
        cfg=C.Config(D=d,DIN=8,DV=d,NL=nl,T=CH,res_scale=nl**-0.5)
        om=C.Core(cfg).to(DEV); oo=torch.optim.Adam(om.parameters(),lr=1e-4); P=npar(om); n=DOC//CH
        def ostep():
            X=torch.randint(0,V,(B,DOC+1),device=DEV); st=None; oo.zero_grad()
            for i in range(n):
                xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
                lg,st=om(xi,states=st,return_states=True); st=om.decay_states(st,0.2)
                (F.cross_entropy(lg.reshape(-1,V),yi.reshape(-1))/n).backward()
            oo.step()
        try: ot,og=t(ostep)
        except RuntimeError: print(f"  {tag:>6s} OURS OOM",flush=True); del om,oo; torch.cuda.empty_cache(); continue
        del oo; torch.cuda.empty_cache(); gc.collect()
        tnl=max(1,round((P-2*V*d)/(12*d*d)))
        try:
            tm=TLM(d,tnl).to(DEV); to=torch.optim.Adam(tm.parameters(),lr=1e-4)
            def tstep():
                X=torch.randint(0,V,(B,DOC+1),device=DEV)
                F.cross_entropy(tm(X[:,:DOC]).reshape(-1,V),X[:,1:DOC+1].reshape(-1)).backward(); to.step(); to.zero_grad()
            tt,tg=t(tstep)
            print(f"  {tag:>6s} {'D%d x %dL'%(d,nl):>14s} {P/1e6:7.0f}M {CH:5d} {B:2d} | {ot:7.2f} {og:6.2f} {B*DOC/ot/1e3:7.1f} | {tt:7.2f} {tg:6.2f} {B*DOC/tt/1e3:7.1f} | {tt/ot:6.2f}x {tg/og:7.2f}x",flush=True)
            del tm,to
        except RuntimeError:
            print(f"  {tag:>6s} {'D%d x %dL'%(d,nl):>14s} {P/1e6:7.0f}M {CH:5d} {B:2d} | {ot:7.2f} {og:6.2f} {B*DOC/ot/1e3:7.1f} | {'OOM':>7s} {'--':>6s} {'--':>7s} | transformer CANNOT RUN",flush=True)
        del om; torch.cuda.empty_cache(); gc.collect()
