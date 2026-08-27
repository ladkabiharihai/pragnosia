"""Train a stronger substrate with memory carry, for the verification/identity work in later loop iterations."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; D=256; NL=6; T=128; NCH=4; B=16; DECAY=0.2; STEPS=20000
OUT="/mnt/kv_cache/pragnosia_data/substrate_20k.pt"
dm=DataMix(ctx=T*NCH+1,verbose=False)
torch.manual_seed(0); np.random.seed(0)
cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
opt=torch.optim.Adam(m.parameters(),lr=1e-3); torch.manual_seed(1); np.random.seed(1); t0=time.time()
for it in range(STEPS):
    X=torch.tensor(dm.batch(B),device=DEV); st=None
    for i in range(NCH):
        xi=X[:,i*T:(i+1)*T]; yi=X[:,i*T+1:(i+1)*T+1]
        lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,DECAY)
        loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    if (it+1)%4000==0:
        torch.save(m.state_dict(),OUT); print(f"  {it+1} loss={loss.item():.4f} ({time.time()-t0:.0f}s) saved",flush=True)
torch.save(m.state_dict(),OUT); print(f"[done] {time.time()-t0:.0f}s -> {OUT}",flush=True)
