"""DOES A BETTER OPTIMIZER CUT THE TRAINING BUDGET? Adam vs Muon, matched steps.

WHY THIS AND NOT MORE KERNEL WORK. Everything tried so far shaved the COST OF A STEP: torch.compile (2.24x),
fused Adam (9%), and three ideas that failed. After all of it, matmuls are 36% of a step -- that is real
arithmetic and near the floor. The only remaining large lever is taking FEWER STEPS to the same loss, which is
the optimizer's job. Adam treats every weight as an independent scalar; it ignores that a weight MATRIX has
structure. Muon orthogonalises the momentum of 2D parameters (Newton-Schulz, no matrix inverse), which keeps
updates from collapsing into a few dominant directions.

Honest framing: this is a well-known method, not our invention, and the interesting question is whether it helps
THIS architecture -- a recurrent linear-attention model with a fixed-size state, which is not what it was tuned on.
Judged on BPB at matched STEPS and on wall-clock, because an optimizer that needs fewer steps but costs more per
step may be no better.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
setup_log("muon")
cap_gpu(float(os.environ.get("CAP","30")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6; D=384; NL=8; DIN=32
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 10.0
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]

def zeropower_ns(G,steps=5):
    """Newton-Schulz iteration for the orthogonal factor of G. Quintic, runs in bf16, no inverse, no SVD."""
    a,b,c=(3.4445,-4.7750,2.0315)
    X=G.bfloat16()
    if G.size(0)>G.size(1): X=X.T
    X=X/(X.norm()+1e-7)
    for _ in range(steps):
        A=X@X.T; Bm=b*A+c*A@A; X=a*X+Bm@X
    if G.size(0)>G.size(1): X=X.T
    return X

class Muon(torch.optim.Optimizer):
    """Muon for 2D params; everything else (biases, norms, embeddings, scalars) falls back to Adam."""
    def __init__(s,params,lr=0.02,momentum=0.95,nesterov=True):
        super().__init__(params,dict(lr=lr,momentum=momentum,nesterov=nesterov))
    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                st=s.state[p]
                if 'm' not in st: st['m']=torch.zeros_like(p.grad)
                m=st['m']; m.mul_(g['momentum']).add_(p.grad)
                upd=p.grad.add(m,alpha=g['momentum']) if g['nesterov'] else m
                o=zeropower_ns(upd)
                # scale so the update magnitude is comparable across shapes
                p.add_(o.to(p.dtype),alpha=-g['lr']*max(1.0,p.size(0)/p.size(1))**0.5)

def bpb(m,cfg):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

def run(tag,kind,lr):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True
    m=C.Core(cfg).to(DEV)
    STEPS=max(1,int(TPP*25_000_000)//(B*CH))
    fwd=torch.compile(m)
    if kind=="adam":
        opts=[torch.optim.Adam(m.parameters(),lr=lr,betas=(0.9,0.95),fused=True)]
    else:
        two=[p for p in m.parameters() if p.ndim==2 and p.numel()>4096]
        rest=[p for p in m.parameters() if not (p.ndim==2 and p.numel()>4096)]
        opts=[Muon(two,lr=lr),torch.optim.Adam(rest,lr=3e-3,betas=(0.9,0.95),fused=True)]
        print(f"    Muon on {len(two)} matrices ({sum(p.numel() for p in two)/1e6:.1f}M params), "
              f"Adam on {len(rest)} others ({sum(p.numel() for p in rest)/1e6:.1f}M)",flush=True)
    warm=300; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
    EVERY=max(400,STEPS//10); ema=None
    bar=pbar(range(STEPS),tag,total=STEPS)
    for i in bar:
        sc=((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for o in opts:
            for g in o.param_groups: g['lr']=(lr if o is opts[0] else 3e-3)*sc
        X=torch.tensor(dm.batch(B),device=DEV)
        for o in opts: o.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        for o in opts: o.step()
        v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
        if (i+1)%50==0:
            bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s")
        if (i+1)%EVERY==0 or i+1==STEPS:
            bar.write(f"    [{tag[:10]}] {(i+1)*B*CH/1e6:5.0f}M tok  BPB {bpb(m,cfg):.4f}  ({(time.time()-t0)/60:.1f} min)")
    bb=bpb(m,cfg); mins=(time.time()-t0)/60
    print(f"  {tag:26s} BPB {bb:.4f}   ({mins:.1f} min)",flush=True)
    del m,fwd,opts; torch.cuda.empty_cache(); return bb,mins

print(f"ADAM vs MUON -- {D}x{NL}L DIN{DIN}, {TPP:.0f} tok/param, matched steps\n",flush=True)
a=run("Adam (current, fused)","adam",1e-3)
b=run("Muon on 2D + Adam rest","muon",0.02)
print(f"\n  Adam {a[0]:.4f} in {a[1]:.1f} min   |   Muon {b[0]:.4f} in {b[1]:.1f} min   "
      f"({b[0]-a[0]:+.4f} BPB, {b[1]/a[1]:.2f}x wall-clock)",flush=True)
