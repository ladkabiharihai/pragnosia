"""WHY IS IT SLOW? Measure the real levers: dtype (fp32 vs bf16 tensor cores), chunk size, doc length, batch size.
Reports tokens/sec and peak memory so the configuration can be chosen on data, not on my guess."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("bench_config")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV; V=32768
def bench(D,NL,DOC,CH,B,amp,steps=3):
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-4); n=DOC//CH
    def step():
        X=torch.randint(0,V,(B,DOC+1),device=DEV); st=None; opt.zero_grad()
        for i in range(n):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp):
                lg,st=m(xi,states=st,return_states=True)
                loss=F.cross_entropy(lg.reshape(-1,V).float(),yi.reshape(-1))/n
            st=m.decay_states(st,0.2); loss.backward()
        opt.step()
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    try:
        step(); torch.cuda.synchronize(); t0=time.time()
        for _ in range(steps): step()
        torch.cuda.synchronize(); dt=(time.time()-t0)/steps
        mem=torch.cuda.max_memory_allocated()/2**30
        del m,opt; torch.cuda.empty_cache(); gc.collect()
        return B*DOC/dt/1e3, mem, dt
    except RuntimeError as e:
        del m,opt; torch.cuda.empty_cache(); gc.collect(); return None,None,str(e)[:30]
if __name__=="__main__":
    print("LEVER 1: DTYPE -- fp32 (what we were doing) vs bf16 tensor cores.  D=512 NL=4, doc 8192, chunk 1024, B=4",flush=True)
    for amp in (False,True):
        k,mem,dt=bench(512,4,8192,1024,4,amp)
        print(f"  {'bf16' if amp else 'fp32'}: {k:8.1f} ktok/s  {mem:5.2f} GiB  {dt:.3f}s/step",flush=True)
    print("\nLEVER 2: CHUNK + DOC + BATCH (all bf16).  D=512 NL=4",flush=True)
    print(f"  {'doc':>6s} {'chunk':>6s} {'B':>3s} | {'ktok/s':>9s} {'GiB':>6s}",flush=True)
    for DOC,CH,B in ((8192,1024,4),(8192,2048,4),(32768,2048,4),(32768,2048,8),(32768,4096,8),(32768,2048,16)):
        k,mem,dt=bench(512,4,DOC,CH,B,True)
        print(f"  {DOC:6d} {CH:6d} {B:3d} | "+(f"{k:9.1f} {mem:6.2f}" if k else f"{'OOM':>9s}"),flush=True)
    print("\nLEVER 3: at the TARGET size D=1024 NL=16 (bf16)",flush=True)
    for DOC,CH,B in ((32768,2048,4),(32768,2048,8)):
        k,mem,dt=bench(1024,16,DOC,CH,B,True)
        print(f"  {DOC:6d} {CH:6d} {B:3d} | "+(f"{k:9.1f} {mem:6.2f}" if k else f"{'OOM':>9s} {dt}"),flush=True)
