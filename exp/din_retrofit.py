"""CAN WE FIX THE LIVE 235M MODEL? It has plateaued (loss moved 0.002 over 4.7B tokens) and no learning rate helps,
which said "saturated". But the saturation may be the RETRIEVAL BOTTLENECK, not the parameter count: DIN has been 8
since the infant, so 1024 residual dims are served by 45 retrieval features.

grow_din is exactly function-preserving (verified 0.000e+00), so we can retrofit the trained model and continue.
If the plateaued model starts improving again with a wider retrieval space, the ceiling was the bottleneck."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("din_retrofit")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix
DEV=C.DEV; CH=512; B=1; STEPS=600   # small footprint: the trainer and prod own most of the card
CK="/mnt/kv_cache/pragnosia_data/pragnosia.pt"
np.random.seed(1234); _dm=DataMix(ctx=CH+1,verbose=False,only='window2')
EV=[torch.tensor(_dm.batch(2),device=DEV) for _ in range(40)]
def load():
    sd=torch.load(CK,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); return m,cfg
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(add_din,steps=STEPS,lr=6e-5):
    torch.manual_seed(0); np.random.seed(0)
    m,cfg=load()
    if add_din: G.grow_din(m,add_din); cfg=m.cfg
    dm=DataMix(ctx=CH+1,verbose=False,only='window2')
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    for i in range(steps):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        del X
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    out=evl(m,cfg); fe=cfg.FE
    del m,opt; torch.cuda.empty_cache(); return out,fe
if __name__=="__main__":
    m,cfg=load(); base=evl(m,cfg); del m; torch.cuda.empty_cache()
    print(f"PLATEAUED 235M MODEL: held-out loss {base:.4f}\n",flush=True)
    print(f"  continuing {STEPS} steps ({STEPS*B*CH/1e6:.1f}M tokens) at lr 6e-5:\n",flush=True)
    print(f"  {'retrieval':>18s} {'FE':>5s} {'loss after':>11s} {'change':>9s}",flush=True)
    l0,fe0=run(0)
    print(f"  {'DIN=8 (current)':>18s} {fe0:5d} {l0:11.4f} {l0-base:+9.4f}",flush=True)
    for add in (8,16):
        l,fe=run(add)
        print(f"  {f'DIN={8+add} (grown)':>18s} {fe:5d} {l:11.4f} {l-base:+9.4f}",flush=True)
    print(f"\n  If the grown ones improve while DIN=8 does not, the ceiling was the retrieval bottleneck.",flush=True)
