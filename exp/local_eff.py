"""LOCAL PLASTICITY AS A TRAINING-EFFICIENCY LEVER (VISION 0c), measured on memory and throughput, not just loss.

THE CASE, WHICH IS ABOUT MEMORY NOT ACCURACY. Backprop must retain the activation graph for the whole network to
run its backward pass -- ~9.3 GiB of the 405M's footprint. Local plasticity computes each layer's update from
locally available signals (lw_lib: per-layer readout + an identity error broadcast + exact within-layer
eligibility traces), so it never builds that graph. If it frees enough memory, the batch can grow, and on a card
shared with prod the batch is exactly what we cannot grow today.

VISION 0c already records the accuracy story honestly: pure-local BEATS backprop at small scale and LOSES at
scale. So the question is NOT "is local as accurate" -- it is "how much memory does it free, and is the batch
that buys worth the accuracy it costs". Measure peak memory, throughput, and the largest batch each can fit.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu
setup_log("local_eff")
cap_gpu(float(os.environ.get("CAP","30")))
import numpy as np, torch, torch.nn.functional as F, core as C
import lw_lib as LW
DEV=C.DEV; V=32768

def build(D,NL,DIN,T):
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=T,vocab=V,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True
    return C.Core(cfg).to(DEV),cfg

def try_batch(mode,D,NL,DIN,T,B):
    """can this mode fit batch B, and how fast is it?"""
    torch.manual_seed(0)
    m,cfg=build(D,NL,DIN,T)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,fused=True)
    X=torch.randint(0,V,(B,T+1),device=DEV)
    Bfb=LW.make_feedback(cfg,NL) if mode=="local" else None
    def step_bp():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:T]).reshape(-1,V),X[:,1:T+1].reshape(-1))
        l.backward(); opt.step()
    def step_lw():
        opt.zero_grad(set_to_none=True)
        g=LW.lw_grads(m,X,Bfb=Bfb)
        opt.step()
    step=step_bp if mode=="backprop" else step_lw
    try:
        for _ in range(2): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.time()
        for _ in range(3): step()
        torch.cuda.synchronize(); ms=(time.time()-t0)*1000/3
        pk=torch.cuda.max_memory_allocated()/2**30
        r=(ms,pk,B*T/ms*1e3/1e3)
    except (torch.cuda.OutOfMemoryError,RuntimeError) as e:
        r=None
    except Exception as e:
        print(f"    {mode} B={B}: {type(e).__name__}: {str(e)[:70]}",flush=True); r=None
    del m,opt,X; torch.cuda.empty_cache(); return r

D,NL,DIN,T=832,14,32,2048
print(f"  shape D{D}x{NL}L DIN{DIN} T={T}   (the finished 125M)\n")
print(f"  {'mode':>10s} {'batch':>6s} {'ms/step':>9s} {'peak GiB':>9s} {'ktok/s':>8s}")
best={}
for mode in ("backprop","local"):
    for B in (1,2,4,8,12,16):
        r=try_batch(mode,D,NL,DIN,T,B)
        if r is None:
            print(f"  {mode:>10s} {B:6d}      OOM / unsupported",flush=True); break
        print(f"  {mode:>10s} {B:6d} {r[0]:9.1f} {r[1]:9.2f} {r[2]:8.0f}",flush=True)
        best[mode]=(B,)+r
print()
for k,v in best.items(): print(f"  best {k:>9s}: batch {v[0]}, {v[3]:.0f} ktok/s at {v[2]:.2f} GiB")
if len(best)==2:
    bp,lw=best["backprop"],best["local"]
    print(f"\n  local fits {lw[0]/bp[0]:.1f}x the batch and runs at {lw[3]/bp[3]:.2f}x the throughput")
    print(f"  VISION 0c: pure-local loses ACCURACY at scale, so this is the memory/throughput side only --")
    print(f"  the trade is only worth taking if the accuracy cost is smaller than the throughput gain.")
