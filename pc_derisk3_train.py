"""PC DE-RISK #3: does pure LOCAL plasticity TRAIN end-to-end on the FULL 290B 32K mix, tracking BPTT?
De-risk #2 proved the local gradient == true gradient (cosine 1.0). This proves it OPTIMIZES: run the SAME tiny core
on the SAME data mix for the same steps, once with BPTT (Adam on autograd grads) and once with the pure-LOCAL rule
(Adam on the manual eligibility-trace grads, NO autograd), and compare LM loss curves. Success = local loss tracks
BPTT loss (they should be ~identical since the grads are identical; small diff only from optimizer-state paths).
"""
import warnings,os,time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
import sys; sys.path.insert(0,"/opt/code/membrae")
from data_mix import DataMix
from pc_derisk2_seqlm import Core, local_grads, VOCAB, T, FE, DIN, DV
dm=DataMix(ctx=T, verbose=True)
def adam_state(prm): return ({k:torch.zeros_like(p) for k,p in prm.items()},{k:torch.zeros_like(p) for k,p in prm.items()})
def bptt_step(m,opt,X):
    lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,VOCAB),X[:,1:T+1].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step(); return loss.item()
def local_step(m,st,v,i,X,lr=3e-3,b1=0.9,b2=0.999,eps=1e-8):
    g=local_grads(m,X); prm={'emb':m.emb.weight,'Wk':m.Wk.weight,'Wq':m.Wq.weight,'Wv':m.Wv.weight,'Wo':m.Wo.weight,'bo':m.Wo.bias}
    with torch.no_grad():
        for k in prm:
            st[k]=b1*st[k]+(1-b1)*g[k]; v[k]=b2*v[k]+(1-b2)*g[k]**2
            prm[k]-=lr*(st[k]/(1-b1**i))/((v[k]/(1-b2**i)).sqrt()+eps)
    # report loss (forward only, no autograd needed but cheap)
    with torch.no_grad():
        lg=m(X[:,:T]); return F.cross_entropy(lg.reshape(-1,VOCAB),X[:,1:T+1].reshape(-1)).item()
def run(mode, steps=600, B=32, lr=3e-3):
    torch.manual_seed(0); np.random.seed(0); m=Core().to(DEV)
    if mode=="bptt":
        opt=torch.optim.Adam(m.parameters(),lr=lr); curve=[]
        for i in range(1,steps+1):
            X=torch.tensor(dm.batch(B),device=DEV); l=bptt_step(m,opt,X)
            if i%50==0: curve.append((i,l)); print(f"  BPTT  step {i} loss={l:.3f} ppl={np.exp(min(20,l)):.1f}",flush=True)
        return curve
    else:
        prm={'emb':m.emb.weight,'Wk':m.Wk.weight,'Wq':m.Wq.weight,'Wv':m.Wv.weight,'Wo':m.Wo.weight,'bo':m.Wo.bias}
        st,v=adam_state(prm); curve=[]
        for i in range(1,steps+1):
            X=torch.tensor(dm.batch(B),device=DEV); l=local_step(m,st,v,i,X,lr=lr)
            if i%50==0: curve.append((i,l)); print(f"  LOCAL step {i} loss={l:.3f} ppl={np.exp(min(20,l)):.1f}",flush=True)
        return curve
if __name__=="__main__":
    print(f"device={DEV} VOCAB={VOCAB} T={T}  -- end-to-end training, full 290B 32K mix",flush=True)
    print("BPTT baseline:",flush=True); t0=time.time(); bp=run("bptt"); print(f"  ({time.time()-t0:.0f}s)",flush=True)
    print("PURE LOCAL (eligibility traces, NO autograd):",flush=True); t0=time.time(); lo=run("local"); print(f"  ({time.time()-t0:.0f}s)",flush=True)
    bpf=bp[-1][1]; lof=lo[-1][1]
    print(f"\nfinal loss  BPTT={bpf:.3f} (ppl {np.exp(bpf):.1f})   LOCAL={lof:.3f} (ppl {np.exp(lof):.1f})   gap={lof-bpf:+.3f}",flush=True)
    print(("PASS -> pure local plasticity TRAINS on real 290B mix, tracks BPTT (§0c end-to-end)" if abs(lof-bpf)<0.15 else
           "close -> minor optimizer-path diff" if abs(lof-bpf)<0.4 else "diverged -> investigate"),flush=True)
