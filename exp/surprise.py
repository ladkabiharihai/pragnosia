"""LEARN FROM SURPRISE: does gating the loss on prediction error beat uniform next-token training?

THE CASE. Measured on our own model: the top 20% of tokens carry 62% of the loss and the bottom 34% carry 0.8%.
Uniform training spends a third of its gradient on tokens that are already learned. A brain does not do this --
plasticity is gated by prediction error; you do not rewire on the expected. VISION 0f makes biology a design
source, and VISION 5 already found the objective (not the architecture) is what unlocks faculties: answer-masking
the recall data took needle recall 0% -> 96%. This is that finding generalised from a hand-drawn mask to an
intrinsic one -- the model's OWN error decides what it learns from, which is 0a's "everything intrinsic".

THE TEST. Identical models, data, seed, and STEP COUNT -- the only difference is which tokens contribute gradient.
Cost per step is unchanged (the forward is needed anyway; only the loss reduction differs), so equal steps IS
equal wall-clock, and any BPB difference is pure efficiency.

THE RISK, stated up front: selecting the highest-loss tokens can chase noise -- unlearnable tokens (random names,
corrupted text) have permanently high loss and would soak up all the gradient. If the aggressive arm underperforms
the moderate one, that is the mechanism, not a bug.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
setup_log("surprise")
cap_gpu(float(os.environ.get("CAP","7")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling=True; _ic.compile_threads=6
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6; D=384; NL=8; DIN=32
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 10.0
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

def run(tag,keep):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True
    m=C.Core(cfg).to(DEV); P=sum(p.numel() for p in m.parameters())
    STEPS=max(1,int(TPP*25_000_000)//(B*CH))
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95),fused=True)
    warm=300; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
    EVERY=max(400,STEPS//12); ema=None
    bar=pbar(range(STEPS),tag,total=STEPS)
    for i in bar:
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g in opt.param_groups: g['lr']=lr
        X=torch.tensor(dm.batch(B),device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=fwd(X[:,:CH])
            per=F.cross_entropy(lg.reshape(-1,V),X[:,1:CH+1].reshape(-1),reduction='none')
            if keep>=1.0:
                l=per.mean()
            else:
                k=max(1,int(keep*per.numel()))
                # top-k by loss, detached selection: the CHOICE is not differentiated, only the kept losses are
                idx=torch.topk(per.detach(),k,largest=True).indices
                l=per[idx].mean()
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
        if (i+1)%50==0:
            bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s "
                                f"{(i+1)*B*CH/1e6:.0f}M tok")
        # LIVE BPB, so the arms can be compared WHILE they run instead of only at the end
        if (i+1)%EVERY==0 or i+1==STEPS:
            bar.write(f"    [{tag[:12]}] step {i+1}/{STEPS}  {(i+1)*B*CH/1e6:5.0f}M tok  "
                      f"BPB {bpb(m,cfg):.4f}  ({(time.time()-t0)/60:.1f} min)")
    bb=bpb(m,cfg); mins=(time.time()-t0)/60
    print(f"  {tag:44s} BPB {bb:.4f}   ({mins:.1f} min, {STEPS} steps, {STEPS*B*CH/1e9:.2f}B tokens seen)",flush=True)
    del m,fwd,opt; torch.cuda.empty_cache(); return bb,mins

print(f"LEARN FROM SURPRISE -- {D}x{NL}L DIN{DIN} at {TPP:.0f} tok/param, identical but for which tokens carry gradient")
print(f"  measured on the 125M: top 20% of tokens hold 62% of the loss, bottom 34% hold 0.8%\n",flush=True)
r={}
r['uniform']=run("A) UNIFORM -- every token contributes (current)",1.0)
r['60']=run("B) top 60% by prediction error",0.60)
r['30']=run("C) top 30% by prediction error",0.30)
print(f"\n  {'arm':>34s} {'BPB':>8s} {'vs uniform':>12s}")
for k,lab in (('uniform','uniform'),('60','top 60%'),('30','top 30%')):
    print(f"  {lab:>34s} {r[k][0]:8.4f} {r[k][0]-r['uniform'][0]:+12.4f}")
