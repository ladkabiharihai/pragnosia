"""BRAIN-REGION ARCHITECTURE: constant per-token load while capacity grows.

THE GOAL. Today, growing the model grows the bill: 125M runs at 208 ktok/s, 405M at 83. Cost scales with size at
BOTH training and inference, which makes growth a liability at serving time and contradicts what the project
claims. A brain has 86 billion neurons and only a small fraction fire for any given input. We want that: ADD
REGIONS WITHOUT ADDING COST.

THE STRUCTURE (regions specialised by function, not identical layers):
  PITUITARY   a global modulator. One small projection produces a per-token control vector that biases routing --
              the "hormone" signal that sets what the rest of the brain is disposed to do. Always runs, tiny.
  AMYGDALA    salience. Always runs, cheap. Decides how strongly a token engages cortex at all. We already have
              this mechanism working -- the selective-write gate learned a 24x spread across layers unsupervised.
  CEREBELLUM  fast procedural. Always runs, narrow. Handles the routine so cortex does not have to.
  CORTEX      four specialised regions, ROUTED -- only k of them see any given token:
      occipital  local pattern     small DIN, narrow MLP
      temporal   language + memory  large DIN (big state)
      parietal   integration        wide MLP
      frontal    reasoning          deep, standard DIN

WHY FIXED-CAPACITY ROUTING AND NOT MoE-STYLE GATHER. Our layers are RECURRENT over the sequence, so you cannot
gather arbitrary tokens the way an MoE gathers them for a feedforward expert -- the state has to see its tokens in
order. So each region takes its top-`cap` tokens BY ROUTER SCORE, in sequence order, and runs its recurrence over
that subsequence. Causality is preserved; the region simply attends to less of the stream. This is what makes the
scheme implementable at all for this architecture.
VISION 6 rejected sparse MoE at ~136M active, and this is deliberately NOT that: no expert duplication, no added
parameters per expert, no top-1 collapse mode. It SKIPS computation on parameters that already exist.
"""
import sys, math
sys.path.insert(0,"/opt/code/membrae/brainexp")
import torch, torch.nn as nn, torch.nn.functional as F
import core as C
DEV=C.DEV

REGIONS=[  # name        DIN  mlp_mult  routed?
    ("cerebellum",        16,   2,      False),   # always-on, cheap, procedural
    ("occipital",         16,   2,      True ),   # local pattern
    ("temporal",          40,   4,      True ),   # language + memory: the big state
    ("parietal",          24,   6,      True ),   # integration: wide MLP
    ("frontal",           32,   4,      True ),   # reasoning
]

class Pituitary(nn.Module):
    """Global modulator. Produces a per-token control vector that BIASES routing -- it does not process content,
    it sets disposition. Deliberately tiny: this is a gland, not a lobe."""
    def __init__(s,D,nreg,h=32):
        super().__init__(); s.f=nn.Sequential(nn.Linear(D,h),nn.GELU(),nn.Linear(h,nreg))
        nn.init.zeros_(s.f[2].weight); nn.init.zeros_(s.f[2].bias)   # starts neutral: no bias on any region
    def forward(s,h): return s.f(h)

class Brain(nn.Module):
    def __init__(s,D=512,V=32768,T=2048,depth=2,topk=2,cap_frac=None,dense=False):
        """depth  = stacks of the region block (so total layers = depth * regions_run_per_token)
           topk   = how many ROUTED cortical regions see each token   <- this is what holds cost constant
           dense  = ablation: every region sees every token (the control)"""
        super().__init__()
        s.D=D; s.V=V; s.topk=topk; s.dense=dense; s.depth=depth
        s.emb=nn.Embedding(V,D); nn.init.normal_(s.emb.weight,0,0.02)
        s.head=nn.Linear(D,V); s.head.weight=s.emb.weight
        s.names=[r[0] for r in REGIONS]
        s.routed=[i for i,r in enumerate(REGIONS) if r[3]]
        s.always=[i for i,r in enumerate(REGIONS) if not r[3]]
        s.blocks=nn.ModuleList()
        s.cfgs=[]
        for d in range(depth):
            layer=nn.ModuleList()
            for (nm,din,mult,_r) in REGIONS:
                cfg=C.Config(D=D,DIN=din,DV=D,NL=depth*len(REGIONS),T=T,vocab=V,
                             res_scale=(depth*len(REGIONS))**-0.5,tie_emb=True)
                cfg.gated=True; cfg.qknorm=True; cfg.qk_learn_scale=True; cfg.sel_write=True
                cfg.mlp_mult=mult
                layer.append(C.Layer(cfg)); s.cfgs.append(cfg)
            s.blocks.append(layer)
        s.router=nn.ModuleList([nn.Linear(D,len(s.routed)) for _ in range(depth)])
        s.pit=Pituitary(D,len(s.routed))
        s.cfg=s.cfgs[0]

    def forward(s,X):
        B,T=X.shape; a=s.emb(X)
        for d,layer in enumerate(s.blocks):
            h,_=C.rmsnorm(a)
            # PITUITARY biases the router: a global disposition signal added to the local routing decision
            score=s.router[d](h)+s.pit(h)                       # [B,T,nrouted]
            for i in s.always:                                  # cerebellum: every token, always
                a,_=layer[i](a)
            if s.dense:
                for j,i in enumerate(s.routed):
                    a,_=layer[i](a)
                continue
            cap=max(1,int(T*s.topk/len(s.routed)))              # tokens each region may take
            for j,i in enumerate(s.routed):
                sc=score[...,j]                                 # [B,T]
                idx=sc.topk(cap,dim=1).indices.sort(dim=1).values   # keep SEQUENCE ORDER for the recurrence
                g=torch.gather(a,1,idx.unsqueeze(-1).expand(-1,-1,s.D))
                y,_=layer[i](g)
                # straight-through gate so the router gets gradient from how useful its picks were
                w=torch.sigmoid(torch.gather(sc,1,idx)).unsqueeze(-1)
                a=a.scatter_add(1,idx.unsqueeze(-1).expand(-1,-1,s.D),(y-g)*w)
        return s.head(C.rmsnorm(a)[0])

    def flops_per_token(s):
        """analytic per-token cost: what actually runs, not what exists"""
        tot=0
        for d in range(s.depth):
            for k,(nm,din,mult,routed) in enumerate(REGIONS):
                c=2*s.D*din*2 + 2*s.D*s.D*2 + 2*mult*s.D*s.D*2   # qk, v+o, mlp
                if routed and not s.dense: c*= s.topk/len(s.routed)
                tot+=c
        return tot
    def n_params(s): return sum(p.numel() for p in s.parameters())
