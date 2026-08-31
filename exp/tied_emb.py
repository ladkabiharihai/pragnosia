"""ANOTHER UNQUESTIONED CHOICE: the embedding and the output head are SEPARATE matrices. At vocab 32768 and D=1024
that is 33.5M + 33.5M = 67M parameters, 29% of a 235M model, spent twice on the same vocabulary.
Weight TYING (head.weight = emb.weight) is standard for small models -- it halves that cost and the freed budget can
go into layers, which is where reasoning lives. Test whether tying helps, hurts, or is neutral at matched TOTAL
parameters: the tied model gets extra layers/width bought with the saved embedding budget."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("tied_emb")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; STEPS=3000
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(D,NL,tie,steps=STEPS,DIN=12):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5)
    m=C.Core(cfg).to(DEV)
    if tie:
        m.head.weight=m.emb.weight                      # share one matrix for input and output
    P=sum({id(p):p.numel() for p in m.parameters()}.values())
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    for i in range(steps):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    out=evl(m,cfg); del m,opt; torch.cuda.empty_cache(); return out,P
if __name__=="__main__":
    print(f"DOES TYING THE EMBEDDING AND HEAD HELP? {STEPS} steps, DIN=12\n",flush=True)
    print(f"  {'config':>26s} {'params':>8s} {'loss':>8s}",flush=True)
    l1,p1=run(384,6,False); print(f"  {'384x6L untied (current)':>26s} {p1/1e6:7.1f}M {l1:8.4f}",flush=True)
    l2,p2=run(384,6,True);  print(f"  {'384x6L TIED':>26s} {p2/1e6:7.1f}M {l2:8.4f}",flush=True)
    # tied frees ~12.6M params at D=384; spend them on depth and width
    l3,p3=run(448,8,True);  print(f"  {'448x8L TIED (spent free)':>26s} {p3/1e6:7.1f}M {l3:8.4f}",flush=True)
    print(f"\n  tying alone: {l2-l1:+.4f} nats at {(p1-p2)/1e6:.1f}M fewer params",flush=True)
    print(f"  tying + spending the savings on layers: {l3-l1:+.4f} nats at {(p3-p1)/1e6:+.1f}M params",flush=True)
