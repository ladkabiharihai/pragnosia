"""FORGETTING: the carried state accumulated unboundedly (decay=1.0), diluting retrieval and hurting absolute loss.
Biology forgets for exactly this reason. Sweep the decay applied to the state between chunks and measure BOTH
absolute loss AND the memory gain -- we want the best of both, not one at the other's expense."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; D=256; NL=6; T=128; NCH=4; B=16; STEPS=3000
dm=DataMix(ctx=T*NCH+1,verbose=False)
def long_batch(): return torch.tensor(dm.batch(B),device=DEV)
def chunks(X): return [(X[:,i*T:(i+1)*T], X[:,i*T+1:(i+1)*T+1]) for i in range(NCH)]
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); return cfg,C.Core(cfg).to(DEV)
@torch.no_grad()
def ev(m,cfg,decay,nb=8):
    wi=np.zeros(NCH); wo=np.zeros(NCH)
    for _ in range(nb):
        X=long_batch(); st=None
        for i,(xi,yi) in enumerate(chunks(X)):
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,decay)
            wi[i]+=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)).item()
            wo[i]+=F.cross_entropy(m(xi).reshape(-1,cfg.vocab),yi.reshape(-1)).item()
    return wi/nb, wo/nb
def train(decay,steps=STEPS,lr=1e-3):
    cfg,m=fresh(); opt=torch.optim.Adam(m.parameters(),lr=lr); torch.manual_seed(1); np.random.seed(1)
    for it in range(steps):
        X=long_batch(); st=None
        for xi,yi in chunks(X):
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,decay)
            loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m,cfg
print("FORGETTING SWEEP (decay applied to the carried state between chunks). decay=1.0 was: loss 5.59-5.83, gain +0.22",flush=True)
print(f"  {'decay':>6s} {'mean loss (with mem)':>21s} {'memory gain':>12s}",flush=True)
for d in (1.0,0.9,0.7,0.5,0.2):
    t0=time.time(); m,cfg=train(d); wi,wo=ev(m,cfg,d)
    gain=np.mean([wo[i]-wi[i] for i in range(1,NCH)])
    print(f"  {d:6.2f} {np.mean(wi):21.4f} {gain:+12.4f}   ({time.time()-t0:.0f}s)",flush=True)
