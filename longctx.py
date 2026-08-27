"""LONG / UNBOUNDED CONTEXT AT CONSTANT MEMORY, and teaching the model to USE its own memory.
Train on CONTIGUOUS multi-chunk documents, carrying the fast-weight state across chunks (detached between chunks:
no cross-chunk backprop, so memory stays CONSTANT no matter how long the document is).
DECISIVE TEST: does per-chunk loss DROP when the carried state is available vs zeroed? If yes, the model has learned
to write and read its own persistent memory -- the prerequisite for continual learning at inference."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; D=256; NL=6; T=128; NCH=4; B=16; STEPS=3000
dm=DataMix(ctx=T*NCH+1,verbose=False)
def long_batch():
    return torch.tensor(dm.batch(B),device=DEV)                  # [B, NCH*T+1] contiguous
def chunks(X):
    return [(X[:,i*T:(i+1)*T], X[:,i*T+1:(i+1)*T+1]) for i in range(NCH)]
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); return cfg,C.Core(cfg).to(DEV)
@torch.no_grad()
def eval_carry(m,cfg,nb=8,decay=1.0):
    """per-chunk loss WITH carried state vs WITHOUT (state zeroed each chunk)."""
    wi=np.zeros(NCH); wo=np.zeros(NCH)
    for _ in range(nb):
        X=long_batch(); st=None
        for i,(xi,yi) in enumerate(chunks(X)):
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,decay)
            wi[i]+=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)).item()
            wo[i]+=F.cross_entropy(m(xi).reshape(-1,cfg.vocab),yi.reshape(-1)).item()
    return wi/nb, wo/nb
def train(carry,steps=STEPS,lr=1e-3):
    cfg,m=fresh(); opt=torch.optim.Adam(m.parameters(),lr=lr); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    for it in range(steps):
        X=long_batch(); st=None
        for xi,yi in chunks(X):
            if carry:
                lg,st=m(xi,states=st,return_states=True)
            else: lg=m(xi); st=None
            loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        if (it+1)%1000==0: print(f"    {'carry' if carry else 'nocarry'} {it+1} loss={loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.eval(); return m,cfg
if __name__=="__main__":
    print(f"LONG-CONTEXT: {NCH} chunks x T={T} = {NCH*T} tokens per document, state carried, constant memory",flush=True)
    for carry in (True,False):
        m,cfg=train(carry)
        wi,wo=eval_carry(m,cfg)
        tag='TRAINED WITH CARRY' if carry else 'trained without carry'
        print(f"\n  === {tag} ===",flush=True)
        print(f"    chunk:        "+"  ".join(f"{i}" for i in range(NCH)),flush=True)
        print(f"    with memory:  "+"  ".join(f"{x:.3f}" for x in wi),flush=True)
        print(f"    no memory:    "+"  ".join(f"{x:.3f}" for x in wo),flush=True)
        print(f"    MEMORY GAIN:  "+"  ".join(f"{(b-a):+.3f}" for a,b in zip(wi,wo))+"   (positive = memory HELPS)",flush=True)
        print(f"    mean gain on chunks 1..{NCH-1}: {np.mean([(wo[i]-wi[i]) for i in range(1,NCH)]):+.4f}",flush=True)
