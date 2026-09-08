"""THE COMPARISON THAT ACTUALLY ANSWERS THE QUESTION: matched COMPUTE, not matched parameters.

Matched-parameters asks "what does sparsity cost?" -- measured: +0.0503 BPB for 1.77x fewer FLOPs.
But the point of constant load is not to run the SAME model cheaper. It is to run a BIGGER model at the SAME
cost. So: give the routed model more regions until its FLOPs/token match dense, and see which is better.

  DENSE    4 regions, every token         41.2M params, 48.8 MFLOP/tok
  ROUTED   8 regions, top-4 per token     62.5M params, ~48.8 MFLOP/tok   <- 52% more capacity, same bill

If routed-8 beats dense-4 at equal compute, constant-load routing is a win and growth stops costing anything at
serving time. If it loses, sparsity is not paying for itself here and the honest answer is that dense is right
for this architecture at this scale -- which is what VISION 6 already concluded for MoE.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae/brainexp"); sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
from muon_opt import Muon
setup_log("exp_muonbrain")
cap_gpu(float(os.environ.get("CAP","28")))
import numpy as np, torch, torch.nn.functional as F
import brain as BR, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=1024; B=8; Dm=512; DEPTH=2
TOKENS=float(sys.argv[1]) if len(sys.argv)>1 else 150e6
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]
BASE=list(BR.REGIONS)

def bpb(m):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

def run(tag,regions,topk,dense):
    BR.REGIONS=regions
    torch.manual_seed(0); np.random.seed(0)
    m=BR.Brain(D=Dm,V=V,T=CH,depth=DEPTH,topk=topk,dense=dense).to(DEV)
    P=m.n_params(); fl=m.flops_per_token(); STEPS=int(TOKENS)//(B*CH)
    # MUON on the 2D weight matrices, Adam on biases/norms/scalars -- the split measured at 24.5M.
    two=[p for p in m.parameters() if p.ndim==2 and p.numel()>4096]
    rest=[p for p in m.parameters() if not (p.ndim==2 and p.numel()>4096)]
    opts=[Muon(two,lr=0.02),torch.optim.Adam(rest,lr=3e-3,betas=(0.9,0.95),fused=True)]
    warm=200; t0=time.time(); torch.manual_seed(1); np.random.seed(1)
    bar=pbar(range(STEPS),tag[:20],total=STEPS); ema=None; EVERY=max(200,STEPS//5)
    for i in bar:
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for o in opts:
            for g in o.param_groups: g['lr']=(0.02 if o is opts[0] else 3e-3)*(lr/1e-3)
        X=torch.tensor(dm.batch(B),device=DEV)
        for o in opts: o.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        for o in opts: o.step()
        v=float(l.detach()); ema=v if ema is None else 0.99*ema+0.01*v
        if (i+1)%50==0: bar.set_postfix_str(f"loss={ema:.3f} {(i+1)*B*CH/(time.time()-t0)/1e3:.0f}ktok/s")
        if (i+1)%EVERY==0: bar.write(f"    [{tag[:16]}] {(i+1)*B*CH/1e6:4.0f}M tok  BPB {bpb(m):.4f}")
    bb=bpb(m)
    print(f"  {tag:30s} BPB {bb:.4f}  {P/1e6:5.1f}M params  {fl/1e6:5.1f} MFLOP/tok  ({(time.time()-t0)/60:.1f} min)",flush=True)
    BR.REGIONS=BASE; del m,opts; torch.cuda.empty_cache(); return bb,P,fl

print(f"MATCHED COMPUTE, MUON -- {TOKENS/1e6:.0f}M tokens each (dense+Adam was 1.2737)\n",flush=True)
big=BASE+[("assoc0",32,4,True),("assoc1",24,4,True),("assoc2",40,2,True),("assoc3",16,6,True)]
# dense+MUON is run for real here. Comparing routed+Muon against dense+ADAM would flatter routing by giving only
# one arm the better optimiser; the 2x2 has to be complete or the claim is not fair.
a=run("DENSE 4 regions + MUON",BASE,4,True)
b=run("ROUTED 8 regions, top-4",big,4,False)
print(f"\n  dense-4  BPB {a[0]:.4f}  {a[1]/1e6:.1f}M params  {a[2]/1e6:.1f} MFLOP/tok")
print(f"  routed-8 BPB {b[0]:.4f}  {b[1]/1e6:.1f}M params  {b[2]/1e6:.1f} MFLOP/tok")
print(f"  -> at {b[2]/a[2]:.2f}x the compute, routed-8 has {b[1]/a[1]:.2f}x the capacity and is "
      f"{b[0]-a[0]:+.4f} BPB")
