"""PURE REPLAY damages the model even with backprop and even teaching nothing (+0.094 backprop, +0.166 local).
The substrate was trained on 262,144 tokens per step; consolidation used 256 -- a 1000x smaller batch. Tiny batches
give gradients dominated by noise, and a noisy gradient applied to a CONVERGED model is a random walk away from the
optimum. That, not the learning rule, is the suspect.

TEST: pure replay (no facts) at increasing EFFECTIVE batch via gradient accumulation. If damage falls toward zero as
the batch grows, consolidation just needs a big enough batch and the vision's local rule is fine."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("batch_noise")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L
from data_mix import DataMix
DEV=C.DEV; T=256
SUB="/mnt/kv_cache/pragnosia_data/pragnosia.pt"
def load():
    sd=torch.load(SUB,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=T,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg
@torch.no_grad()
def general(m,cfg,n=6):
    np.random.seed(21); dm=DataMix(ctx=T+1,verbose=False,only='window2'); t=0
    for _ in range(n):
        X=torch.tensor(dm.batch(2),device=DEV)
        t+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab).float(),X[:,1:T+1].reshape(-1)).item()
    return t/n
def pure_replay(acc,updates=50,lr=1e-4,micro=2):
    """`acc` micro-batches accumulated per update -> effective batch = acc*micro sequences."""
    torch.manual_seed(0); np.random.seed(0)
    m,cfg=load(); g0=general(m,cfg)
    dm=DataMix(ctx=T+1,verbose=False,only='window2')
    opt=C.LocalAdam(m.param_dict(),lr=lr,adaptive=True)
    for u in range(updates):
        agg=None
        for a in range(acc):
            g=L.lw_grads(m,torch.tensor(dm.batch(micro),device=DEV),lam=4.0,k_local=1)
            if agg is None: agg={k:v.clone() for k,v in g.items()}
            else:
                for k in agg: agg[k]+=g[k]
        for k in agg: agg[k]/=acc
        opt.step(agg)
    d=general(m,cfg)-g0
    del m; torch.cuda.empty_cache()
    return d
if __name__=="__main__":
    print("PURE REPLAY DAMAGE vs EFFECTIVE BATCH (50 updates, no facts taught, local rule)\n",flush=True)
    print(f"  {'accum':>6s} {'eff. batch':>11s} {'tokens/update':>14s} {'DAMAGE':>9s}",flush=True)
    for acc in (1,4,16,48):
        d=pure_replay(acc)
        print(f"  {acc:6d} {acc*2:11d} {acc*2*T:14d} {d:+9.4f}",flush=True)
    print(f"\n  training used 262144 tokens/step. If damage -> 0 as the batch grows, the fix is batch size,",flush=True)
    print(f"  not the learning rule -- and the vision's local plasticity is intact.",flush=True)
