"""REFERENCE-SCORED SURPRISE (the Rho-1 / Selective-LM form), which is what my first version got wrong.

WHY THE FIRST VERSION FAILED. Selecting the top-k tokens by RAW loss conflates two different things:
  - learnable but not yet learned  -> real surprise, worth gradient
  - inherently unpredictable       -> a random surname, an arbitrary number, a URL fragment
The second kind has permanently high loss and can never be learned, so raw-loss selection preferentially feeds
the model noise. Measured: top-60%-by-loss was 0.17-0.27 BPB WORSE than uniform at every matched checkpoint.

THE FIX. Score each token against a REFERENCE model that already knows the data -- our FINISHED 125M -- and keep
tokens with the largest EXCESS loss (student - reference). A token the reference also finds hard is unlearnable
noise and gets dropped; a token the reference predicts easily and the student does not is exactly the learnable
frontier. This is the intrinsic version of VISION 5's answer-masking: the mask is drawn by a model, not by hand.

COST NOTE, stated honestly: this runs a 125M reference forward per step, which is ~5x the 24.5M student's
forward, so WALL-CLOCK here is not the production cost. In a real run the reference scores are precomputed once
over the corpus. So the comparison that matters is BPB at MATCHED TOKENS, and that is what is reported.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar, cfg_from_ckpt
setup_log("surprise_ref")
cap_gpu(float(os.environ.get("CAP","12")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6; D=384; NL=8; DIN=32
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 10.0
KEEP=float(sys.argv[2]) if len(sys.argv)>2 else 0.60
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]

rsd=torch.load("/mnt/kv_cache/pragnosia_data/p124f.pt",map_location=DEV)
rcfg=cfg_from_ckpt(rsd,T=CH,res_scale=rsd['res_scale'])
ref=C.Core(rcfg).to(DEV); ref.load_state_dict(rsd['model']); ref.eval()
for p in ref.parameters(): p.requires_grad_(False)
print(f"  reference: the FINISHED 125M ({sum(p.numel() for p in ref.parameters())/1e6:.0f}M, 9.12B tokens)",flush=True)

def bpb(m,cfg):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

torch.manual_seed(0); np.random.seed(0)
cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True
m=C.Core(cfg).to(DEV)
STEPS=max(1,int(TPP*25_000_000)//(B*CH))
fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95),fused=True)
warm=300; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
EVERY=max(400,STEPS//12); ema=None; dropped=[]
tag=f"D) reference-scored top {KEEP:.0%}"
bar=pbar(range(STEPS),tag,total=STEPS)
for i in bar:
    lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
    for g in opt.param_groups: g['lr']=lr
    X=torch.tensor(dm.batch(B),device=DEV); y=X[:,1:CH+1].reshape(-1)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        rl=F.cross_entropy(ref(X[:,:CH]).reshape(-1,V),y,reduction='none')
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        per=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),y,reduction='none')
        excess=(per.detach()-rl)                       # learnable-but-unlearned, not merely hard
        k=max(1,int(KEEP*per.numel()))
        idx=torch.topk(excess,k,largest=True).indices
        l=per[idx].mean()
    l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
    if (i+1)%50==0:
        bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s {(i+1)*B*CH/1e6:.0f}M tok")
    if (i+1)%EVERY==0 or i+1==STEPS:
        bar.write(f"    [REF-SCORED] step {i+1}/{STEPS}  {(i+1)*B*CH/1e6:5.0f}M tok  BPB {bpb(m,cfg):.4f}  "
                  f"({(time.time()-t0)/60:.1f} min)   [uniform at this point: see surprise.log]")
print(f"\n  {tag}: BPB {bpb(m,cfg):.4f}  ({(time.time()-t0)/60:.1f} min, {STEPS*B*CH/1e9:.2f}B tokens)",flush=True)
