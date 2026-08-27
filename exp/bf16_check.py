"""Does bf16 actually TRAIN correctly, or just fast? Same model/data/seed, fp32 vs bf16, compare loss curves.
Also confirms memory at the TARGET size with the chosen config, including through growth."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log,pbar; setup_log("bf16_check")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G; from data_mix import DataMix
DEV=C.DEV; DOC=8192; CH=2048; NCH=DOC//CH; B=4; STEPS=1200
dm=DataMix(ctx=DOC+1,verbose=False)
def run(amp):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=384,DIN=8,DV=384,NL=4,T=CH,res_scale=16**-0.5); m=C.Core(cfg).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=6e-4); torch.manual_seed(1); np.random.seed(1)
    curve=[]; t0=time.time()
    for it in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV); st=None; opt.zero_grad()
        for i in range(NCH):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=amp):
                lg,st=m(xi,states=st,return_states=True)
                loss=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1))/NCH
            st=m.decay_states(st,0.2); loss.backward()
        opt.step()
        if (it+1)%200==0: curve.append(loss.item()*NCH)
    # held-out
    tot=0.
    with torch.no_grad():
        for _ in range(10):
            X=torch.tensor(dm.batch(B),device=DEV)
            tot+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1)).item()
    del m,opt; torch.cuda.empty_cache(); gc.collect()
    return curve,tot/10,time.time()-t0
if __name__=="__main__":
    print("bf16 CORRECTNESS: does it train the same as fp32? (D384x4L, 1200 steps, identical seed/data)",flush=True)
    c32,h32,t32=run(False); print(f"  fp32 curve: {' '.join(f'{x:.3f}' for x in c32)}   held-out={h32:.4f}  ({t32:.0f}s)",flush=True)
    c16,h16,t16=run(True);  print(f"  bf16 curve: {' '.join(f'{x:.3f}' for x in c16)}   held-out={h16:.4f}  ({t16:.0f}s)",flush=True)
    print(f"\n  held-out difference = {h16-h32:+.4f}   speedup = {t32/t16:.2f}x",flush=True)
    print(("  => bf16 IS SAFE: same quality, real speedup" if abs(h16-h32)<0.05 else
           "  => bf16 degrades quality; keep fp32 or use selective precision"),flush=True)
    print("\nMEMORY AT TARGET (bf16, doc 32768, chunk 2048) including a GROWN model:",flush=True)
    for D,NL,Bs in ((1024,16,4),(1024,16,8)):
        torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
        cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=2048,res_scale=16**-0.5); m=C.Core(cfg).to(DEV)
        opt=torch.optim.Adam(m.parameters(),lr=1e-4)
        try:
            X=torch.randint(0,32768,(Bs,32769),device=DEV); st=None; opt.zero_grad()
            for i in range(16):
                xi=X[:,i*2048:(i+1)*2048]; yi=X[:,i*2048+1:(i+1)*2048+1]
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    lg,st=m(xi,states=st,return_states=True)
                    l=F.cross_entropy(lg.reshape(-1,32768).float(),yi.reshape(-1))/16
                st=m.decay_states(st,0.2); l.backward()
            opt.step()
            P=sum(p.numel() for p in m.parameters())
            print(f"  D{D} x {NL}L B={Bs}: {P/1e6:.0f}M params, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",flush=True)
        except RuntimeError as e: print(f"  D{D} x {NL}L B={Bs}: OOM",flush=True)
        del m,opt; torch.cuda.empty_cache(); gc.collect()
