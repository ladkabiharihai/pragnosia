"""CAN THE ROUTED MODEL CONVERT ITS IDLE GPU INTO THROUGHPUT?

Measured during the quality run: DENSE sits at ~100% GPU utilisation, ROUTED top-2 at 64-74%. So routed is NOT
compute-bound -- it does 1.77x less arithmetic but leaves a third of the card idle, because gather/scatter and the
extra kernel launches make it launch-bound. The FLOP saving is real and UNCONVERTED.

The way to collect it is more work per launch: a bigger batch. Dense is already saturated and cannot take one.
This measures throughput and peak memory across batch sizes for both, which is the honest test of whether
"constant load" is a real serving property or only a spreadsheet one.
"""
import sys, os, time; sys.path.insert(0,"/opt/code/membrae/brainexp"); sys.path.insert(0,"/opt/code/membrae")
from util import cap_gpu; cap_gpu(float(os.environ.get("CAP","26")))
import torch, torch.nn.functional as F, brain as BR
DEV="cuda"; V=32768; CH=1024
def bench(dense,topk,B):
    torch.manual_seed(0)
    m=BR.Brain(D=512,V=V,T=CH,depth=2,topk=topk,dense=dense).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,fused=True)
    X=torch.randint(0,V,(B,CH+1),device=DEV)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); opt.step()
    try:
        for _ in range(3): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.time()
        for _ in range(5): step()
        torch.cuda.synchronize(); ms=(time.time()-t0)*1000/5
        r=(ms, B*CH/ms*1e3/1e3, torch.cuda.max_memory_allocated()/2**30)
    except torch.cuda.OutOfMemoryError: r=None
    del m,opt,X; torch.cuda.empty_cache(); return r
print(f"  {'arm':>14s} {'batch':>6s} {'ms/step':>9s} {'ktok/s':>8s} {'peak GiB':>9s}")
best={}
for tag,dense,topk in (("DENSE",True,4),("ROUTED k=2",False,2),("ROUTED k=1",False,1)):
    for B in (8,16,24,32,48):
        r=bench(dense,topk,B)
        if r is None: print(f"  {tag:>14s} {B:6d}      OOM",flush=True); break
        print(f"  {tag:>14s} {B:6d} {r[0]:9.1f} {r[1]:8.0f} {r[2]:9.2f}",flush=True)
        if tag not in best or r[1]>best[tag][1]: best[tag]=(B,r[1])
print()
for k,v in best.items(): print(f"  best {k:>11s}: batch {v[0]}, {v[1]:.0f} ktok/s")
if "DENSE" in best and "ROUTED k=2" in best:
    print(f"\n  routed k=2 at its best batch is {best['ROUTED k=2'][1]/best['DENSE'][1]:.2f}x dense throughput")
    print(f"  (analytic FLOP saving is 1.77x -- the gap between those two numbers is what the")
    print(f"   gather/scatter overhead costs, and what a fused kernel would recover.)")
