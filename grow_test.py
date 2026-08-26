"""GROWTH + LOCAL RULE: does an INFANT grown to the target match a model trained AT the target size, for less compute?
Arm A (fixed): train at the target size the whole way.
Arm B (grown): start small (D=96, NL=2) and grow WIDTH+DEPTH to the target during training (function-preserving).
Both use the pure-local rule. Reports final loss AND wall-time (growth should reach the target for LESS compute)."""
import sys,time,json; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L, growth as G; from data_mix import DataMix
DEV=C.DEV; T=64; B=32; STEPS=6000; LR=3e-3
TD,TNL=192,6                                              # target: D=192, 6 layers
dm=DataMix(ctx=T,verbose=False)
def evl(m,cfg,nb=12):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def make(D,NL):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=TNL**-0.5); return cfg,C.Core(cfg).to(DEV)
def run(arm):
    if arm=='fixed': cfg,m=make(TD,TNL); sched={}
    else:
        cfg,m=make(96,2)
        sched={1200:('w',48),2400:('d',2),3600:('w',48),4800:('d',2)}   # infant 96/2 -> 192/6
    torch.manual_seed(1); np.random.seed(1); opt=C.LocalAdam(m.param_dict(),lr=LR); t0=time.time(); cur=[]
    for i in range(1,STEPS+1):
        if i in sched:
            kind,amt=sched[i]
            new=G.grow_width(m,amt) if kind=='w' else G.grow_depth(m,amt)
            opt=C.LocalAdam(m.param_dict(),lr=LR)                       # params changed -> fresh optimizer state
            print(f"      [grow @{i}] {'width' if kind=='w' else 'depth'} -> D={m.cfg.D} NL={m.cfg.NL}",flush=True)
        X=torch.tensor(dm.batch(B),device=DEV); opt.step(L.lw_grads(m,X))
        if i%1000==0:
            e=evl(m,m.cfg); cur.append((i,e,time.time()-t0))
            print(f"    {arm:6s} {i:5d} heldout={e:.4f} D={m.cfg.D} NL={m.cfg.NL} ({time.time()-t0:.0f}s)",flush=True)
    return cur,time.time()-t0
print(f"GROWTH TEST: infant(D96,NL2) grown -> target(D{TD},NL{TNL}) vs fixed target, {STEPS} steps, pure-local rule",flush=True)
fx,ft=run('fixed'); gr,gt=run('grown')
print(f"\n  FIXED  final={fx[-1][1]:.4f}  ({ft:.0f}s)",flush=True)
print(f"  GROWN  final={gr[-1][1]:.4f}  ({gt:.0f}s, {gt/ft:.2f}x compute)",flush=True)
d=gr[-1][1]-fx[-1][1]
print(f"\n  grown - fixed = {d:+.4f}   compute saved = {(1-gt/ft)*100:.0f}%",flush=True)
print(("GROWTH WINS: same/better quality for less compute" if d<0.05 else
       "growth close" if d<0.2 else "growth costs quality"),flush=True)
json.dump({'fixed':fx,'grown':gr},open('grow_test.json','w'))
