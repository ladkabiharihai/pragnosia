"""ACCEPTANCE 5 -- ONE GENUINELY INTRINSIC DECISION (VISION 0a): growth fires on the MODEL'S OWN saturation signal,
not on a schedule I wrote. Three arms, same budget:
  A FIXED    -- trained at the target size the whole way (quality reference)
  B SCHEDULED-- grows on hand-written step numbers (what we did before: scaffolding)
  C INTRINSIC-- grows when the model's OWN predictive entropy stops falling faster than its own noise
If C matches B, the hand-written schedule can be deleted and the decision belongs to the model."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, pbar; setup_log("acceptance5_intrinsic_growth")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G; from data_mix import DataMix
DEV=C.DEV; T=256; B=16; STEPS=6000; LR=1e-3
TD,TNL=192,6                                    # target: D=192, 6 layers  (infant: D=96, 2 layers)
dm=DataMix(ctx=T+1,verbose=False)
def make(D,NL):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=TNL**-0.5); return cfg,C.Core(cfg).to(DEV)
def evl(m,cfg,nb=10):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def plan_step(m,kind,amt):
    return G.grow_width(m,amt) if kind=='w' else G.grow_depth(m,amt)
PLAN=[('w',48),('d',2),('w',48),('d',2)]        # infant 96/2 -> target 192/6 (same total growth in every arm)
def run(mode):
    cfg,m=(make(TD,TNL) if mode=='fixed' else make(96,2))
    torch.manual_seed(1); np.random.seed(1)
    opt=torch.optim.Adam(m.parameters(),lr=LR); t0=time.time(); k=0; fired=[]
    sched={1200:0,2400:1,3600:2,4800:3}
    sensor=G.SaturationSensor(window=150,t_stat=1.0,min_gap=500)
    bar=pbar(range(STEPS),f"{mode:9s}")
    for it in bar:
        X=torch.tensor(dm.batch(B),device=DEV)
        lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        grow=None
        if mode=='scheduled' and it in sched and k<len(PLAN): grow=PLAN[k]
        elif mode=='intrinsic' and k<len(PLAN):
            if sensor.observe(G.SaturationSensor.entropy(lg.detach()),it): grow=PLAN[k]
        if grow:
            plan_step(m,*grow); opt=torch.optim.Adam(m.parameters(),lr=LR); k+=1; fired.append(it)
            bar.write(f"    [{mode}] grow #{k} at step {it}: D={m.cfg.D} NL={m.cfg.NL}")
        if (it+1)%200==0: bar.set_postfix_str(f"loss={loss.item():.3f} D={m.cfg.D} NL={m.cfg.NL}")
    return evl(m,cfg), time.time()-t0, fired, m.cfg.D, m.cfg.NL
if __name__=="__main__":
    print(f"ACCEPTANCE 5: does the model's OWN saturation signal replace my hand-written growth schedule?",flush=True)
    res={}
    for mode in ('fixed','scheduled','intrinsic'):
        l,dt,fired,D_,NL_=run(mode); res[mode]=(l,dt,fired,D_,NL_)
        print(f"  {mode:10s} final={l:.4f}  ({dt:.0f}s)  ended D={D_} NL={NL_}  grows at {fired}",flush=True)
    f,s,i=res['fixed'][0],res['scheduled'][0],res['intrinsic'][0]
    print(f"\n  fixed={f:.4f} | scheduled={s:.4f} ({s-f:+.3f}) | INTRINSIC={i:.4f} ({i-f:+.3f} vs fixed, {i-s:+.3f} vs scheduled)",flush=True)
    print(f"  compute: scheduled {res['scheduled'][1]/res['fixed'][1]:.2f}x  intrinsic {res['intrinsic'][1]/res['fixed'][1]:.2f}x of fixed",flush=True)
    print(("  => INTRINSIC TRIGGER WORKS: the schedule can be deleted" if i-s<0.1 else
           "  => intrinsic trails the schedule; sensor needs work"),flush=True)
