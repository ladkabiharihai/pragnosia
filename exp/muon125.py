"""DOES MUON HOLD AT 125M, AND DOES IT SURVIVE A WARM START? The experiment that gates the flagship restart.

TWO THINGS THE 24.5M RESULT DOES NOT TELL US.
1. SCALE. Muon beat Adam by 0.0774 BPB at 24.5M from scratch. Small-scale results have misled this project twice
   today, so a 24.5M optimiser win is not licence to commit a 24-hour run.
2. WARM START -- and this is the real question. Optimiser gains are usually LARGEST EARLY, because they come from
   keeping updates well-conditioned from initialisation onward. Our flagship would not start from initialisation:
   it continues the FINISHED 125M, whose weights Adam has already shaped into its own basin. Muon resuming there
   may capture only part of the benefit, or none. The 24.5M number is an UPPER BOUND, not a prediction.

So this measures the ACTUAL situation: take the finished 125M and continue it two ways, same data, same steps,
same seed, only the optimiser differing. Whatever the gap is here is what a flagship restart would actually buy.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar, cfg_from_ckpt
setup_log("muon125")
cap_gpu(float(os.environ.get("CAP","32")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
from muon_opt import Muon                   # side-effect-free import of the exact class measured at 24.5M
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6
TOK_TARGET=float(sys.argv[1]) if len(sys.argv)>1 else 0.30e9      # tokens of continuation per arm
SRC="/mnt/kv_cache/pragnosia_data/p124f.pt"
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]

def bpb(m,cfg):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

def run(tag,kind,lr):
    torch.manual_seed(0); np.random.seed(0)
    sd=torch.load(SRC,map_location=DEV); cfg=cfg_from_ckpt(sd,T=CH,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
    b0=bpb(m,cfg)
    STEPS=int(TOK_TARGET)//(B*CH)
    fwd=torch.compile(m)
    if kind=="adam":
        opts=[torch.optim.Adam(m.parameters(),lr=lr,betas=(0.9,0.95),fused=True)]
    else:
        two=[p for p in m.parameters() if p.ndim==2 and p.numel()>4096]
        rest=[p for p in m.parameters() if not (p.ndim==2 and p.numel()>4096)]
        opts=[Muon(two,lr=lr),torch.optim.Adam(rest,lr=3e-4,betas=(0.9,0.95),fused=True)]
    warm=200; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
    bar=pbar(range(STEPS),tag[:20],total=STEPS); ema=None; EVERY=max(300,STEPS//8)
    print(f"  {tag}: starting from the finished 125M at BPB {b0:.4f}",flush=True)
    for i in bar:
        sc=((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for o in opts:
            base=lr if o is opts[0] else 3e-4
            for g in o.param_groups: g['lr']=base*sc
        X=torch.tensor(dm.batch(B),device=DEV)
        for o in opts: o.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        for o in opts: o.step()
        v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
        if (i+1)%50==0: bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s")
        if (i+1)%EVERY==0 or i+1==STEPS:
            bar.write(f"    [{tag[:10]}] {(i+1)*B*CH/1e6:4.0f}M tok  BPB {bpb(m,cfg):.4f}  (start {b0:.4f})")
    bb=bpb(m,cfg)
    print(f"  {tag:28s} {b0:.4f} -> {bb:.4f}   ({bb-b0:+.4f} in {(time.time()-t0)/60:.1f} min)",flush=True)
    del m,fwd,opts; torch.cuda.empty_cache(); return b0,bb

print(f"MUON AT 125M, WARM START from the finished flagship -- {TOK_TARGET/1e6:.0f}M tokens per arm\n",flush=True)
a=run("Adam (fused) continuation","adam",3e-4)
b=run("Muon continuation","muon",0.01)
print(f"\n  Adam {a[0]:.4f} -> {a[1]:.4f}  |  Muon {b[0]:.4f} -> {b[1]:.4f}   "
      f"gap {b[1]-a[1]:+.4f}")
print(f"  (at 24.5M FROM SCRATCH Muon won by -0.0774; if the warm-start gap is much smaller, that is the")
print(f"   early-training advantage not transferring, and the flagship should simply RESUME on Adam.)",flush=True)
