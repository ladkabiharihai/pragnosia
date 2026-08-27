"""Train a substrate with memory carry. tqdm progress -> logs/substrate.log"""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix; from util import pbar
DEV=C.DEV; D=256; NL=6; T=128; NCH=4; B=16; DECAY=0.2
STEPS=int(sys.argv[1]) if len(sys.argv)>1 else 20000
OUT=sys.argv[2] if len(sys.argv)>2 else "/mnt/kv_cache/pragnosia_data/substrate_20k.pt"
dm=DataMix(ctx=T*NCH+1,verbose=False)
torch.manual_seed(0); np.random.seed(0)
cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
opt=torch.optim.Adam(m.parameters(),lr=1e-3); torch.manual_seed(1); np.random.seed(1)
ema=None
bar=pbar(range(STEPS),f"substrate D{D}/{NL}L")
for it in bar:
    X=torch.tensor(dm.batch(B),device=DEV); st=None
    for i in range(NCH):
        xi=X[:,i*T:(i+1)*T]; yi=X[:,i*T+1:(i+1)*T+1]
        lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,DECAY)
        loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    ema=loss.item() if ema is None else 0.99*ema+0.01*loss.item()
    if (it+1)%200==0: bar.set_postfix_str(f"loss={ema:.4f} ppl={np.exp(min(20,ema)):.1f}")
    if (it+1)%4000==0: torch.save(m.state_dict(),OUT)
torch.save(m.state_dict(),OUT); print(f"\n[done] -> {OUT}",flush=True)
