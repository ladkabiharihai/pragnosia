"""DOES CENTERING THE KEYS HELP WHEN TRAINED IN? The end-to-end test of the key-collapse diagnosis.

THE CHAIN, measured on the finished 125M:
  hidden states anisotropic (mean pairwise cosine 0.976 early) -> Wk inherits it -> keys cluster around a MEAN
  (uncentered rank ~6 of 32, centered ~20) -> |cos(q,k)| averages 0.30 instead of ~0 -> the 3:1 algebraic
  ceiling binds at 1.45x instead of near 3.00x -> state 1.2% occupied -> retrieval dies at ~2000 tokens.
Removing the mean measured 1.35x more selectivity POST HOC (1.45 -> 1.97). Post-hoc is not proof: the same trick
on read sharpness tripled needle lift and destroyed BPB, because the model had trained without it. So train it in.

Judged on BPB *and* on the quantities the diagnosis predicts should move: key rank, state occupancy, and needle
lift. If BPB improves but key rank does not, the diagnosis is wrong even if the change helps.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu, pbar
setup_log("keycenter")
cap_gpu(float(os.environ.get("CAP","20")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=6; D=384; NL=8; DIN=32
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 10.0
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]
NAMES=["Aurelia","Kestrel","Vandermeer","Okonkwo","Thackeray","Zoravar","Lindqvist","Mbeki"]
np.random.seed(3); POOL=[]
while len(POOL)<40000: POOL.extend(dm.batch(1)[0].tolist())   # occupancy walks 32768; 20000 sliced EMPTY past that

def bpb(m,cfg):
    m.eval(); n=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return n/(np.log(2)*nb)

def er(A):
    A=A.float(); S=torch.linalg.svdvals(A).double()
    return float((S.sum()**2)/(S.pow(2).sum()+1e-12))

@torch.no_grad()
def diagnostics(m,cfg):
    """key rank (UNCENTERED -- the quantity that was 5.6-21 and bounds the state) and state occupancy"""
    m.eval(); X=EV[0][:,:CH]
    a=m.emb(X); ks=[]
    for L in m.layers:
        H,_=C.rmsnorm(a); Hs=C._shift(H)
        xk=Hs@L.Wk.weight.T
        if L.qk_center: xk=xk-L.k_mu.to(xk.dtype)
        if L.qknorm: xk=xk*torch.rsqrt(xk.pow(2).sum(-1,keepdim=True)+1e-6)*L.qk_s(Hs)
        ks.append(er(xk[0])); a,_=L(a)
    st=[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
    pv=None
    for i in range(0,min(32768,len(POOL)-CH),CH):   # never slice past the pool
        with torch.autocast("cuda",dtype=torch.bfloat16):
            xb=torch.tensor([POOL[i:i+CH]],dtype=torch.long,device=DEV)
            assert xb.shape[1]==CH, f"short slice {xb.shape} -- POOL too small"
            _,st,pv=m(xb,states=st,return_states=True,prev=pv,return_prev=True)
    srs=[er(M[0]) for M,_ in st]
    m.train(); return float(np.mean(ks)), float(np.mean(srs))

@torch.no_grad()
def needle(m,cfg,n=12):
    m.eval()
    def zs(): return [(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
    def score(ids,cue,tok):
        full=ids+cue; st=zs(); pv=None
        for i in range(0,len(full)-1,CH):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                _,st,pv=m(torch.tensor([full[i:i+CH]],device=DEV),states=st,return_states=True,prev=pv,return_prev=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg,_,_=m.step(torch.tensor([[full[-1]]],device=DEV),st,pv)
        return float(lg[0].float().log_softmax(-1)[tok])
    out=[]
    for Dd in (0,200,1000):
        L=[]
        for t in range(n):
            tn=NAMES[t%8]; ot=NAMES[(t+3)%8]
            cue=TOK.encode(" The vault keeper is").ids; tok=TOK.encode(f" {tn}").ids[0]
            rng=np.random.RandomState(500+t); s0=rng.randint(0,len(POOL)-Dd-64)
            L.append(score(TOK.encode(f" The vault keeper is {tn}. ").ids+POOL[s0:s0+Dd],cue,tok)
                    -score(TOK.encode(f" The vault keeper is {ot}. ").ids+POOL[s0:s0+Dd],cue,tok))
        out.append(float(np.mean(L)))
    m.train(); return out

def run(tag,center,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True;cfg.qknorm=True;cfg.qk_learn_scale=True;cfg.sel_write=True;cfg.qk_center=center
    m=C.Core(cfg).to(DEV)
    STEPS=max(1,int(TPP*25_000_000)//(B*CH))
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95),fused=True)
    warm=300; t0=time.time(); torch.manual_seed(seed+1); np.random.seed(seed+1)
    bar=pbar(range(STEPS),tag[:20],total=STEPS); ema=None
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
    b=bpb(m,cfg); kr,sr=diagnostics(m,cfg); nd=needle(m,cfg)
    print(f"  {tag:26s} BPB {b:.4f}  key-rank {kr:5.1f}/32  state-rank {sr:5.1f}/{D}  "
          f"needle d0 {nd[0]:+.3f} d200 {nd[1]:+.3f} d1000 {nd[2]:+.3f}  ({(time.time()-t0)/60:.1f}m)",flush=True)
    del m,fwd,opt; torch.cuda.empty_cache(); return b,kr,sr,nd

print(f"KEY CENTERING -- {D}x{NL}L DIN{DIN} at {TPP:.0f} tok/param, identical but for the running mean")
print(f"  seed floor is 0.0120 BPB, so a BPB claim under that needs 3+ seeds\n",flush=True)
a=run("A) baseline (no centering)",False)
b=run("B) key+query centering",True)
print(f"\n  BPB      {a[0]:.4f} -> {b[0]:.4f}   ({b[0]-a[0]:+.4f}, seed bar 0.0120)")
print(f"  key rank {a[1]:.1f} -> {b[1]:.1f}   ({b[1]/max(1e-9,a[1]):.2f}x)   <- the diagnosis predicts THIS moves")
print(f"  state rank {a[2]:.1f} -> {b[2]:.1f}   ({b[2]/max(1e-9,a[2]):.2f}x)")
print(f"  needle d0 {a[3][0]:+.3f} -> {b[3][0]:+.3f}   d200 {a[3][1]:+.3f} -> {b[3][1]:+.3f}")
