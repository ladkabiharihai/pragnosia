"""SCALING PROJECTION: train at 32k-token documents, from ~40M up to ~1B params, ours vs a PARAM-MATCHED transformer.
Ours chunks the document (constant memory); the transformer must attend over all 32768 tokens. Param-matched means
the transformer gets fewer, fatter layers (its layer is ~12*D^2; ours is ~2*D^2), so the comparison is fair on size.
Reports s/step, peak memory, tok/s, and the resulting speed/memory advantage at each scale."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
DEV=C.DEV; V=32768; DOC=32768; CH=128
class TBlock(nn.Module):
    def __init__(s,d,h=8):
        super().__init__(); s.h=h
        s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
        s.n2=nn.LayerNorm(d); s.m1=nn.Linear(d,4*d); s.m2=nn.Linear(4*d,d)
    def forward(s,x):
        B,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).chunk(3,-1)
        q,k,v=[t.view(B,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+s.o(a.transpose(1,2).reshape(B,T,D))
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
def tf_layers_for(target,d):                                   # pick transformer depth to match param count
    fixed=2*V*d; per=12*d*d
    return max(1,round((target-fixed)/per))
def timed(f,n=3):
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    f(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t0)/n, torch.cuda.max_memory_allocated()/2**30
if __name__=="__main__":
    print(f"SCALING at {DOC}-token documents (ours chunked at {CH}; transformer attends over all {DOC})",flush=True)
    print(f"  {'target':>8s} {'D x NL (ours)':>15s} {'params':>8s} {'B':>2s} | {'OURS s/step':>11s} {'GiB':>6s} {'ktok/s':>7s} | {'TF s/step':>10s} {'GiB':>6s} {'ktok/s':>7s} | {'speedup':>8s} {'mem':>7s}",flush=True)
    for tag,d,nl,B in (("40M",512,12,2),("200M",1536,24,1),("400M",2048,36,1),("1B",3072,40,1)):
        cfg=C.Config(D=d,DIN=8,DV=d,NL=nl,T=CH,res_scale=nl**-0.5)
        try:
            om=C.Core(cfg).to(DEV); oo=torch.optim.Adam(om.parameters(),lr=1e-4); P=npar(om)
            NCH_=DOC//CH
            def ostep():
                """ONE optimizer step per document (gradients accumulated over chunks) -- like-for-like with the
                transformer, which also does one step per document. Per-chunk stepping was 256x the optimizer work."""
                X=torch.randint(0,V,(B,DOC+1),device=DEV); st=None; oo.zero_grad()
                for i in range(NCH_):
                    xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
                    lg,st=om(xi,states=st,return_states=True); st=om.decay_states(st,0.2)
                    (F.cross_entropy(lg.reshape(-1,V),yi.reshape(-1))/NCH_).backward()
                oo.step()
            ot,og=timed(ostep,1)
        except RuntimeError as e:
            print(f"  {tag:>8s} {'D%d x %dL'%(d,nl):>15s} {'-':>8s} {B:2d} | OURS OOM: {str(e)[:40]}",flush=True); continue
        del oo; torch.cuda.empty_cache(); gc.collect()
        tnl=tf_layers_for(P,d)
        try:
            tm=TLM(d,tnl).to(DEV); to=torch.optim.Adam(tm.parameters(),lr=1e-4)
            def tstep():
                X=torch.randint(0,V,(B,DOC+1),device=DEV)
                loss=F.cross_entropy(tm(X[:,:DOC]).reshape(-1,V),X[:,1:DOC+1].reshape(-1)); to.zero_grad(); loss.backward(); to.step()
            tt,tg=timed(tstep,1)
            print(f"  {tag:>8s} {'D%d x %dL'%(d,nl):>15s} {P/1e6:7.0f}M {B:2d} | {ot:11.2f} {og:6.2f} {B*DOC/ot/1e3:7.1f} | {tt:10.2f} {tg:6.2f} {B*DOC/tt/1e3:7.1f} | {tt/ot:7.2f}x {tg/og:6.2f}x",flush=True)
            del tm,to
        except RuntimeError as e:
            print(f"  {tag:>8s} {'D%d x %dL'%(d,nl):>15s} {P/1e6:7.0f}M {B:2d} | {ot:11.2f} {og:6.2f} {B*DOC/ot/1e3:7.1f} | {'OOM':>10s} (TF {tnl}L cannot fit {DOC} tokens) | ours-only",flush=True)
        del om; torch.cuda.empty_cache(); gc.collect()
