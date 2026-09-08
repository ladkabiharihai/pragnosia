"""IS THE DATA MIX ANY GOOD? It was set once while the architecture was being debugged and never revisited.

WHAT WE ALREADY KNOW ABOUT IT. window2 carries **39.9% of the sampling weight** and is **10.5% synthetic template
boilerplate** (kinship-QA and sequence-completion, measured over 400 random windows; dclm is 0.0%). That
boilerplate is verbatim what leaks out of our generations as `<assistant>` tags, "B: ... C: ..." multiple choice,
and "Answer: entailment" -- the model is not degenerating, it is reproducing a tenth of its training distribution.
So the single largest corpus is also the most contaminated, and nobody has tested whether that weighting is right.

WHY THIS MATTERS MORE THAN AN OPTIMISER. A better mix costs NOTHING at inference and nothing at training time --
it is the same number of tokens, just better ones. If it is worth even 0.02 BPB it is free money.

Arms judged on BPB, and on GENERATION contamination (rate of the template markers), because the mix that scores
best on BPB is not automatically the one that stops the model emitting <assistant> at a user.
"""
import sys, os, time, math, re; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
setup_log("datamix")
cap_gpu(float(os.environ.get("CAP","30")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6; D=384; NL=8; DIN=32
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 10.0

# HELD-OUT EVAL IS FIXED ACROSS ARMS and deliberately NOT the training mix -- otherwise each arm would be scored
# on its own distribution and the arm that trains on the easiest corpus would "win" by definition.
np.random.seed(777)
_ev=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
EV=[torch.tensor(_ev.batch(2),device=DEV) for _ in range(8)]
np.random.seed(778)
_evw=DataMix(ctx=CH+1,verbose=False,only='dclm')
EVW=[torch.tensor(_evw.batch(2),device=DEV) for _ in range(6)]
np.random.seed(11); _pd=DataMix(ctx=1025,verbose=False,only='dclm')
PROMPTS=[_pd.batch(1)[0].tolist()[:256] for _ in range(3)]
MARKERS=("<assistant>","<user>","Answer: entailment","\nB:","\nC:","Step-by-step")

def bpb(m,cfg,ev):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in ev:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

@torch.no_grad()
def contamination(m):
    """how often does the model emit the template boilerplate, per 1000 generated tokens"""
    m.eval(); hits=0; tot=0
    for p in PROMPTS:
        torch.manual_seed(0)
        o,_=m.generate(torch.tensor([p],device=DEV),n=800,temp=0.8,top_p=0.92)
        t=TOK.decode(o); tot+=len(o)
        hits+=sum(t.count(k) for k in MARKERS)
    m.train(); return 1000*hits/max(1,tot)

def run(tag,lang_frac,override):
    torch.manual_seed(0); np.random.seed(0)
    dm=DataMix(ctx=CH+1,verbose=False,lang_frac=lang_frac,override=override)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True
    m=C.Core(cfg).to(DEV)
    STEPS=max(1,int(TPP*25_000_000)//(B*CH))
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95),fused=True)
    warm=300; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
    bar=pbar(range(STEPS),tag[:22],total=STEPS); ema=None; EVERY=max(400,STEPS//5)
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
        if (i+1)%EVERY==0: bar.write(f"    [{tag[:16]}] {(i+1)*B*CH/1e6:4.0f}M tok  BPB {bpb(m,cfg,EV):.4f}")
    a=bpb(m,cfg,EV); w=bpb(m,cfg,EVW); c=contamination(m)
    print(f"  {tag:34s} mix-BPB {a:.4f}  web-BPB {w:.4f}  template-markers {c:.2f}/1k tok  "
          f"({(time.time()-t0)/60:.1f} min)",flush=True)
    del m,fwd,opt; torch.cuda.empty_cache(); return a,w,c

print(f"DATA MIX -- {D}x{NL}L at {TPP:.0f} tok/param, identical models, only the sampling weights differ")
print(f"  eval is FIXED across arms (the current mix + a pure-web set), so no arm is scored on its own data\n",flush=True)
R={}
R['current']  = run("A) current (window2 39.9%)",0.70,None)
R['w2_half']  = run("B) window2 halved -> dclm",0.70,{"window2":0.20})
R['w2_quart'] = run("C) window2 quartered -> dclm",0.70,{"window2":0.10})
R['lang85']   = run("D) more language (lang_frac 0.85)",0.85,None)
print(f"\n  {'arm':>34s} {'mix-BPB':>9s} {'web-BPB':>9s} {'markers/1k':>11s}")
for k,lab in (('current','A) current'),('w2_half','B) window2 halved'),
              ('w2_quart','C) window2 quartered'),('lang85','D) lang_frac 0.85')):
    a,w,c=R[k]; print(f"  {lab:>34s} {a:9.4f} {w:9.4f} {c:11.2f}")
