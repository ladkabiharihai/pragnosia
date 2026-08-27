"""ARCHITECTURE DECISION before any long run: our layer is attention-only (~2*D^2). A transformer layer is ~12*D^2
because of its MLP, and MLPs are where most of a model's knowledge is usually stored.
At a MATCHED TOTAL PARAMETER BUDGET, is it better to spend params on (A) more/wider attention-only layers, or
(B) fewer layers that each have an MLP? This decides the shape of the real model."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log,pbar; setup_log("arch_mlp")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; T=256; B=16; STEPS=4000; LR=1e-3
dm=DataMix(ctx=T+1,verbose=False)
class MLPLayer(nn.Module):
    """our recurrent mixer + a standard feedforward block (pre-norm), i.e. a full transformer-style layer."""
    def __init__(s,cfg,mult=4):
        super().__init__(); s.inner=C.Layer(cfg); s.cfg=cfg
        s.m1=nn.Linear(cfg.D,mult*cfg.D); s.m2=nn.Linear(mult*cfg.D,cfg.D)
    def forward(s,a,state=None):
        a,c=s.inner(a,state)
        h,_=C.rmsnorm(a)
        return a+s.cfg.res_scale*s.m2(F.gelu(s.m1(h))), c
def build(D,NL,mlp):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
    if mlp: m.layers=nn.ModuleList([MLPLayer(cfg).to(DEV) for _ in range(NL)])
    return cfg,m
def npar(m): return sum(p.numel() for p in m.parameters())
def run(tag,D,NL,mlp):
    cfg,m=build(D,NL,mlp); P=npar(m)
    torch.manual_seed(1); np.random.seed(1); opt=torch.optim.Adam(m.parameters(),lr=LR); t0=time.time()
    for it in pbar(range(STEPS),tag):
        X=torch.tensor(dm.batch(B),device=DEV)
        loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    tot=0.
    with torch.no_grad():
        for _ in range(12):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    e=tot/12; nonemb=P-2*cfg.vocab*D
    print(f"  {tag:26s} params={P/1e6:6.1f}M (non-emb {nonemb/1e6:5.1f}M)  loss={e:.4f}  ({time.time()-t0:.0f}s)",flush=True)
    del m,opt; torch.cuda.empty_cache(); return e
if __name__=="__main__":
    print("ARCHITECTURE: attention-only vs with-MLP at a matched total parameter budget (~92M)",flush=True)
    a=run("A attention-only D1024x12L",1024,12,False)
    b=run("B with-MLP     D640x12L",640,12,True)
    print(f"\n  attention-only={a:.4f}   with-MLP={b:.4f}   difference={b-a:+.4f}",flush=True)
    print(("  => ADD THE MLP: it buys more quality per parameter" if b<a-0.02 else
           "  => KEEP ATTENTION-ONLY: the MLP does not pay for itself" if b>a+0.02 else
           "  => TIE: keep the simpler attention-only design"),flush=True)
