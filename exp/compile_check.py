"""Does torch.compile help on top of bf16? We are at 99% of a 400W cap, so gains must come from doing LESS work
(kernel fusion / less memory traffic), not from filling idle time. Measured with training running, so both arms
are equally loaded and the RATIO is what matters."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("compile_check")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV; V=32768
def bench(D,NL,DOC,CH,B,compile_,steps=3):
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=CH,res_scale=16**-0.5); m=C.Core(cfg).to(DEV)
    if compile_:
        try: m=torch.compile(m)
        except Exception as e: return None,f"compile failed: {str(e)[:40]}"
    opt=torch.optim.Adam(m.parameters(),lr=1e-4); n=DOC//CH
    def step():
        X=torch.randint(0,V,(B,DOC+1),device=DEV); st=None; opt.zero_grad()
        for i in range(n):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg,st=m(xi,states=st,return_states=True)
                l=F.cross_entropy(lg.reshape(-1,V).float(),yi.reshape(-1))/n
            st=[(a.detach(),b.detach()) for a,b in st]; l.backward()
        opt.step()
    try:
        torch.cuda.empty_cache(); gc.collect()
        t0=time.time(); step(); warm=time.time()-t0                 # includes compilation
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(steps): step()
        torch.cuda.synchronize(); dt=(time.time()-t0)/steps
        del m,opt; torch.cuda.empty_cache(); gc.collect()
        return B*DOC/dt/1e3, f"warmup {warm:.0f}s"
    except Exception as e:
        del m,opt; torch.cuda.empty_cache(); gc.collect(); return None,str(e)[:60]
if __name__=="__main__":
    print("torch.compile on top of bf16 (doc 32768, chunk 2048)",flush=True)
    for D,NL,B in ((384,4,8),(1024,16,4)):
        a,na=bench(D,NL,32768,2048,B,False)
        b,nb=bench(D,NL,32768,2048,B,True)
        if a and b: print(f"  D{D}x{NL}L B{B}: eager {a:7.1f} ktok/s | compiled {b:7.1f} ktok/s | {b/a:.2f}x  ({nb})",flush=True)
        elif a: print(f"  D{D}x{NL}L B{B}: eager {a:7.1f} ktok/s | compiled FAILED: {nb}",flush=True)
        else: print(f"  D{D}x{NL}L B{B}: eager failed {na}",flush=True)
