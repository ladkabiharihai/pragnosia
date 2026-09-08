"""THE RUN-TO-RUN SEED NOISE FLOOR ON BPB. Never measured, and several shipped decisions sit inside it.

WHY THIS MATTERS AND IS NOT ACADEMIC. Decisions taken this week whose margins may be noise:
  brain routed-8 vs dense-4 at matched compute   +0.0082
  per-layer vs per-token read sharpness          +0.0068
  DIN 16 vs 32                                   +0.0102  (called a win)
  reference-scored token selection vs uniform    +0.0046  (called parity)
I have been writing "inside plausible noise" all day WITHOUT HAVING MEASURED PLAUSIBLE. This measures it.

Identical config, identical token budget, identical everything -- only the seed differs, and it varies BOTH the
model init AND the data order, because that is what actually differs between two runs of the same experiment.
Reports the spread and the 2-sigma threshold that any future claim has to clear.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
setup_log("seedfloor")
cap_gpu(float(os.environ.get("CAP","20")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6; D=384; NL=8; DIN=32
NSEED=int(sys.argv[1]) if len(sys.argv)>1 else 4
TPP=float(sys.argv[2]) if len(sys.argv)>2 else 10.0
# the EVAL set is fixed across seeds -- only training varies, or we would be measuring eval noise too
np.random.seed(777); _e=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
EV=[torch.tensor(_e.batch(2),device=DEV) for _ in range(8)]

def bpb(m,cfg):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)      # fresh mixer so data ORDER varies with the seed too
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True
    m=C.Core(cfg).to(DEV)
    STEPS=max(1,int(TPP*25_000_000)//(B*CH))
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95),fused=True)
    warm=300; t0=time.time()
    torch.manual_seed(seed+1000); np.random.seed(seed+1000)
    bar=pbar(range(STEPS),f"seed {seed}",total=STEPS); ema=None
    for i in bar:
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g in opt.param_groups: g['lr']=lr
        X=torch.tensor(dm.batch(B),device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
        if (i+1)%50==0: bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s")
    b=bpb(m,cfg)
    print(f"  seed {seed}: BPB {b:.4f}   ({(time.time()-t0)/60:.1f} min)",flush=True)
    del m,fwd,opt; torch.cuda.empty_cache(); return b

print(f"SEED NOISE FLOOR -- {D}x{NL}L DIN{DIN}, {TPP:.0f} tok/param, {NSEED} seeds, identical but for the seed\n",flush=True)
R=[run(s) for s in range(NSEED)]
a=np.array(R); sd=float(a.std(ddof=1)); rng=float(a.max()-a.min())
print(f"\n  BPB per seed: "+"  ".join(f"{x:.4f}" for x in R))
print(f"  mean {a.mean():.4f}   std {sd:.4f}   min-max spread {rng:.4f}")
print(f"\n  => a difference between two single-seed runs must exceed ~{2*sd*math.sqrt(2):.4f} BPB to be 2-sigma.")
print(f"  Against that bar, decisions taken this week:")
for nm,d in (("brain routed-8 vs dense-4 (matched compute)",0.0082),
             ("per-layer vs per-token sharpness",0.0068),
             ("DIN 16 vs 32 -- called a WIN",0.0102),
             ("reference-scored selection vs uniform",0.0046),
             ("read sharpness fix (shipped)",0.0518),
             ("Muon vs Adam from scratch",0.0774)):
    print(f"    {nm:46s} {d:.4f}  {'CLEARS' if d>2*sd*math.sqrt(2) else 'INSIDE NOISE'}")
