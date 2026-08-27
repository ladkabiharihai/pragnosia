"""Is our 32k slowdown really CHUNK GRANULARITY (many tiny sequential kernel launches) rather than real FLOPs?
Sweep the chunk size at a fixed 32768-token document. Bigger chunks = fewer sequential steps and bigger kernels,
at the cost of O(chunk^2) within-chunk mixing. Find the chunk size that maximizes throughput."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV; V=32768; DOC=32768
def run(d,nl,CH,B):
    cfg=C.Config(D=d,DIN=8,DV=d,NL=nl,T=CH,res_scale=nl**-0.5); m=C.Core(cfg).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4); n=DOC//CH
    def step():
        X=torch.randint(0,V,(B,DOC+1),device=DEV); st=None; opt.zero_grad()
        for i in range(n):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,0.2)
            (F.cross_entropy(lg.reshape(-1,V),yi.reshape(-1))/n).backward()
        opt.step()
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize(); t0=time.time(); step(); torch.cuda.synchronize()
    dt=time.time()-t0; mem=torch.cuda.max_memory_allocated()/2**30
    del m,opt; torch.cuda.empty_cache(); gc.collect()
    return dt,mem
if __name__=="__main__":
    print(f"CHUNK-SIZE SWEEP at {DOC}-token documents (transformer reference at 40M: 0.81 s/step, 29.06 GiB)",flush=True)
    for d,nl,B,tag in ((512,12,2,"40M"),(1536,24,1,"200M")):
        print(f"\n  --- {tag} (D{d} x {nl}L, batch {B}) ---",flush=True)
        print(f"  {'chunk':>6s} {'steps/doc':>9s} {'s/step':>8s} {'GiB':>6s} {'ktok/s':>8s}",flush=True)
        for CH in (128,512,1024,2048,4096):
            try:
                dt,mem=run(d,nl,CH,B)
                print(f"  {CH:6d} {DOC//CH:9d} {dt:8.2f} {mem:6.2f} {B*DOC/dt/1e3:8.1f}",flush=True)
            except RuntimeError as e:
                print(f"  {CH:6d} {DOC//CH:9d} {'OOM':>8s}",flush=True); torch.cuda.empty_cache(); gc.collect()
