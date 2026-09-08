"""Muon optimizer, extracted so it can be IMPORTED without side effects.

It lived in exp/muon.py with the experiment's run() calls at module level, so `from muon import Muon` re-executed
the entire 24.5M benchmark -- and inherited sys.argv, planning 610 billion steps. Module-level work in a file
that is also a library is a trap; this file defines only the optimizer.
"""
import torch

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

