"""TRAINING COMPARISON: does the ~0.985-cosine pure-local rule actually TRAIN as well as backprop, at depth?
Same architecture (muPC residual scaling), same init, same data (real 290B 32K mix), same optimizer/lr. Three arms:
  1) BPTT            -- autograd backprop (reference)
  2) LOCAL muPC+STOP -- pure local plasticity, NO backprop/autograd (the fix from build step 1e)
  3) LOCAL naive-PC  -- pure local, pre-lever settling (shows the levers matter for TRAINING, not just cosine)
"""
import warnings,time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import sys; sys.path.insert(0,"/opt/code/membrae")
import core as C, pcalm3 as P3
from data_mix import DataMix
DEV=C.DEV; NL=6; T=32; B=32; LR=3e-3; STEPS=400; RS=NL**-0.5
dm=DataMix(ctx=T,verbose=False)
def fresh():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T,res_scale=RS); return cfg,C.Core(cfg).to(DEV)
def evloss(m,cfg,nb=8):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV)
            lg=m(X[:,:T]); tot+=F.cross_entropy(lg.reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def run(arm):
    cfg,m=fresh(); t0=time.time(); curve=[]
    torch.manual_seed(1); np.random.seed(1)                       # identical data stream across arms
    if arm=='bptt': opt=torch.optim.Adam(m.parameters(),lr=LR)
    else: opt=C.LocalAdam(m.param_dict(),lr=LR)
    for i in range(1,STEPS+1):
        X=torch.tensor(dm.batch(B),device=DEV)
        if arm=='bptt':
            lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        elif arm=='mupc_stop':
            opt.step(P3.pcalm3_grads(m,X,T_settle=12,eta_a=0.8,rho=0.5,stop=True))
        else:
            opt.step(C.local_grads(m,X,K=18,eta=0.15))
        if i%100==0:
            l=evloss(m,cfg); curve.append((i,l)); print(f"    {arm:11s} step {i:4d} heldout-loss={l:.4f} ({time.time()-t0:.0f}s)",flush=True)
    return curve,time.time()-t0
if __name__=="__main__":
    print(f"TRAIN COMPARE: NL={NL} D=96 T={T} B={B} steps={STEPS}, muPC arch, real 290B mix",flush=True)
    res={}
    for arm in ('bptt','mupc_stop','naive_pc'):
        c,el=run(arm); res[arm]=(c[-1][1],el)
        print(f"  == {arm}: final heldout-loss={c[-1][1]:.4f}  ppl={np.exp(c[-1][1]):.1f}  ({el:.0f}s)",flush=True)
    b=res['bptt'][0]
    print(f"\nRESULT (vs BPTT {b:.4f}):",flush=True)
    for arm in ('mupc_stop','naive_pc'):
        print(f"  {arm:11s} {res[arm][0]:.4f}  gap={res[arm][0]-b:+.4f}  cost={res[arm][1]/res['bptt'][1]:.1f}x BPTT wall-time",flush=True)
    g=res['mupc_stop'][0]-b
    print(f"\n{'PASS: pure local plasticity trains ~= backprop at depth 6' if g<0.15 else 'CLOSE' if g<0.4 else 'GAP remains'}",flush=True)
