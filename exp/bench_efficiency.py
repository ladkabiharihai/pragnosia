"""EFFICIENCY BENCHMARK vs a real transformer baseline (the comparison the vision has never actually run).
Q1 training memory, Q2 how it scales with growth, Q3 training efficiency vs an existing-LLM-style transformer.
Baseline is a STRONG one: same D/NL/vocab, pre-norm, GQA-free MHA using torch's fused scaled_dot_product_attention
(flash kernels when available), trained with autograd. Ours trains on documents in CHUNKS with carried state, so the
honest comparison is 'train on a document of length L' -- the transformer must attend over all L, we never do."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
DEV=C.DEV; V=32768
class TBlock(nn.Module):
    def __init__(s,d,h=8):
        super().__init__(); s.h=h; s.d=d
        s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
        s.n2=nn.LayerNorm(d); s.m1=nn.Linear(d,4*d); s.m2=nn.Linear(4*d,d)
    def forward(s,x):
        B,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).chunk(3,-1)
        q,k,v=[t.view(B,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True)           # fused/flash attention
        x=x+s.o(a.transpose(1,2).reshape(B,T,D))
        return x+s.m2(F.gelu(s.m1(s.n2(x))))
class TLM(nn.Module):
    def __init__(s,d,nl):
        super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(8192,d)
        s.bl=nn.ModuleList([TBlock(d) for _ in range(nl)]); s.nf=nn.LayerNorm(d); s.head=nn.Linear(d,V)
    def forward(s,X):
        x=s.emb(X)+s.pos(torch.arange(X.shape[1],device=X.device))[None]
        for b in s.bl: x=b(x)
        return s.head(s.nf(x))
def nparams(m): return sum(p.numel() for p in m.parameters())
def measure(fn):
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    for _ in range(2): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(4): fn()
    torch.cuda.synchronize()
    return (time.time()-t0)/4, torch.cuda.max_memory_allocated()/2**20
def ours_step(m,cfg,opt,L,B,chunk=128):
    def f():
        X=torch.randint(0,V,(B,L+1),device=DEV); st=None
        for i in range(L//chunk):
            xi=X[:,i*chunk:(i+1)*chunk]; yi=X[:,i*chunk+1:(i+1)*chunk+1]
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,0.2)
            loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return f
def tf_step(m,opt,L,B):
    def f():
        X=torch.randint(0,V,(B,L+1),device=DEV)
        loss=F.cross_entropy(m(X[:,:L]).reshape(-1,V),X[:,1:L+1].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    return f
if __name__=="__main__":
    D,NL,B=256,6,8
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=128,res_scale=NL**-0.5); om=C.Core(cfg).to(DEV); oo=torch.optim.Adam(om.parameters(),lr=1e-4)
    tm=TLM(D,NL).to(DEV); to=torch.optim.Adam(tm.parameters(),lr=1e-4)
    print(f"Q3 TRAINING EFFICIENCY vs a transformer (D={D}, {NL} layers, vocab {V}, batch {B})",flush=True)
    print(f"  params: ours={nparams(om)/1e6:.1f}M  transformer={nparams(tm)/1e6:.1f}M",flush=True)
    print(f"\n  {'doc len':>8s} | {'OURS s/step':>11s} {'peak MiB':>9s} {'ktok/s':>8s} | {'TF s/step':>10s} {'peak MiB':>9s} {'ktok/s':>8s} | {'mem ratio':>9s} {'speed':>7s}",flush=True)
    for L in (128,512,2048,8192):
        try: ot,om_=measure(ours_step(om,cfg,oo,L,B))
        except RuntimeError as e: print(f"  {L:8d} | OURS OOM"); continue
        try: tt,tm_=measure(tf_step(tm,to,L,B))
        except RuntimeError as e:
            print(f"  {L:8d} | {ot:11.3f} {om_:9.1f} {B*L/ot/1e3:8.1f} | {'OOM':>10s} {'-':>9s} {'-':>8s} | transformer OOM",flush=True); continue
        print(f"  {L:8d} | {ot:11.3f} {om_:9.1f} {B*L/ot/1e3:8.1f} | {tt:10.3f} {tm_:9.1f} {B*L/tt/1e3:8.1f} | {tm_/om_:8.2f}x {tt/ot:6.2f}x",flush=True)
    print("\nQ1/Q2 OUR TRAINING MEMORY as the model GROWS (doc len 2048, chunked):",flush=True)
    print(f"  {'config':>22s} {'params':>9s} {'peak MiB':>9s} {'s/step':>8s}",flush=True)
    for (d,nl) in ((128,4),(256,6),(384,8),(512,12)):
        c2=C.Config(D=d,DIN=8,DV=d,NL=nl,T=128,res_scale=nl**-0.5); m2=C.Core(c2).to(DEV); o2=torch.optim.Adam(m2.parameters(),lr=1e-4)
        t,mem=measure(ours_step(m2,c2,o2,2048,B))
        print(f"  {'D%d x %dL'%(d,nl):>22s} {nparams(m2)/1e6:8.1f}M {mem:9.1f} {t:8.3f}",flush=True)
        del m2,o2
