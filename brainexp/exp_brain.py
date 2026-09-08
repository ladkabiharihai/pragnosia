"""DOES CONSTANT-LOAD ROUTING PAY? Two questions, one run.

Q1 THE GROWTH PROPERTY. Add cortical regions at fixed top-k: do PARAMETERS grow while FLOPS/TOKEN stay flat?
   That is the whole claim -- grow the brain without growing the bill. It is arithmetic, so it is checked first
   and cheaply, before spending GPU on quality.

Q2 THE PRICE. Skipping computation must cost something. Measured against DENSE at MATCHED PARAMETERS -- not
   matched FLOPs, which would flatter the sparse arm by construction. If routing at top-k=2 costs less quality
   than the 1.8x compute it saves, the direction is live.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae/brainexp")
sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
setup_log("exp_brain")
cap_gpu(float(os.environ.get("CAP","30")))
import numpy as np, torch, torch.nn.functional as F
import brain as BR, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=128
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=1024; B=8; Dm=512; DEPTH=2
TOKENS=float(sys.argv[1]) if len(sys.argv)>1 else 150e6
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]

print("="*96); print("Q1  THE GROWTH PROPERTY -- add regions at fixed top-k, does cost stay flat?")
print("="*96)
base=list(BR.REGIONS)
print(f"  {'cortical regions':>17s} {'top-k':>6s} {'params':>9s} {'MFLOPs/token':>13s}")
for extra in (0,2,4):
    BR.REGIONS=base+[(f"assoc{i}",32,4,True) for i in range(extra)]
    m=BR.Brain(D=Dm,depth=DEPTH,topk=2).to(DEV)
    nr=len([r for r in BR.REGIONS if r[3]])
    print(f"  {nr:17d} {2:6d} {m.n_params()/1e6:8.1f}M {m.flops_per_token()/1e6:13.1f}")
    del m; torch.cuda.empty_cache()
BR.REGIONS=base
print("  -> params grow, cost does not. That is the claim, and it is arithmetic.\n",flush=True)

def bpb(m):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

def run(tag,topk,dense):
    torch.manual_seed(0); np.random.seed(0)
    m=BR.Brain(D=Dm,V=V,T=CH,depth=DEPTH,topk=topk,dense=dense).to(DEV)
    P=m.n_params(); fl=m.flops_per_token()
    STEPS=int(TOKENS)//(B*CH)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95),fused=True)
    warm=200; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
    bar=pbar(range(STEPS),tag[:20],total=STEPS); ema=None; EVERY=max(200,STEPS//6)
    for i in bar:
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g in opt.param_groups: g['lr']=lr
        X=torch.tensor(dm.batch(B),device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
        if (i+1)%50==0: bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s")
        if (i+1)%EVERY==0: bar.write(f"    [{tag[:14]}] {(i+1)*B*CH/1e6:4.0f}M tok  BPB {bpb(m):.4f}")
    bb=bpb(m); mins=(time.time()-t0)/60; tps=STEPS*B*CH/(mins*60)/1e3
    # what did the router actually learn -- or did it collapse?
    with torch.no_grad():
        X=EV[0][:,:CH]; h,_=C.rmsnorm(m.emb(X)); sc=(m.router[0](h)+m.pit(h))
        share=torch.zeros(len(m.routed),device=DEV)
        cap=max(1,int(CH*topk/len(m.routed)))
        for j in range(len(m.routed)):
            share[j]=sc[...,j].topk(cap,dim=1).indices.numel()
        ent=float(-(sc.softmax(-1).mean((0,1))*sc.softmax(-1).mean((0,1)).log()).sum())
    print(f"  {tag:26s} BPB {bb:.4f}  {P/1e6:5.1f}M params  {fl/1e6:5.1f} MFLOP/tok  "
          f"{tps:5.0f} ktok/s  router-entropy {ent:.3f}/{math.log(len(m.routed)):.3f}  ({mins:.1f} min)",flush=True)
    del m,opt; torch.cuda.empty_cache(); return bb,fl,tps

print("="*96); print(f"Q2  THE PRICE -- matched PARAMETERS, {TOKENS/1e6:.0f}M tokens each")
print("="*96,flush=True)
R={}
R['dense']=run("DENSE (all 4, every token)",4,True)
R['k2']   =run("ROUTED top-2 of 4",2,False)
R['k1']   =run("ROUTED top-1 of 4",1,False)
print(f"\n  {'arm':>26s} {'BPB':>8s} {'vs dense':>9s} {'MFLOP/tok':>10s} {'compute saved':>14s}")
for k,lab in (('dense','DENSE'),('k2','top-2'),('k1','top-1')):
    b,f,t=R[k]; print(f"  {lab:>26s} {b:8.4f} {b-R['dense'][0]:+9.4f} {f/1e6:10.1f} {R['dense'][1]/f:13.2f}x")
