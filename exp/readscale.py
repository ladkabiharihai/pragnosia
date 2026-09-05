"""DOES A LEARNED READ SHARPNESS RECOVER INDUCTION WITHOUT COSTING ANYTHING? The controlled test of the fix.

THE DIAGNOSIS. qk-norm forces |xq|=|xk|=1, so the retrieval weight is a polynomial in the cosine ALONE:
A = qf.kf = 1 + c + c^2. A perfect key match scores 3, an orthogonal key scores 1 -- the selectivity of this
kernel is capped at 3:1 whatever Wq/Wk learn, and at L=256 the right key gets 3/(3+255) = 1.2% of the read.
Measured inside the finished 124M: mean A_max/A_median = 2.08x, i.e. the layers are already pressed against the
3.00x ceiling. Sweeping a scale at INFERENCE on those same weights takes induction from 0% to 21%/9% at
L=64/128 -- the circuit was always there and the kernel was drowning it.

THE FIX UNDER TEST. Scale the normalised q,k by a per-layer LEARNED s: A = 1 + s^2 c + s^4 c^2, ratio
(1+s^2+s^4):1. Learned rather than set, so the model can keep some layers soft (smooth generation) and make
others sharp (retrieval) -- which is how transformers end up with specialised induction heads. Init s=1, so arm B
starts as an EXACT copy of arm A and any difference is something the model chose.

Same size, data, tokens, seed. Judged on induction AND BPB AND retrieval AND generation, because the earlier
inference-time sharpening tripled needle lift and destroyed generation, and a fix that repeats that is no fix.
"""
import sys, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log
setup_log("readscale")
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling=True; _ic.compile_threads=8
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=8; D=448; NL=10; DIN=16
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 30.0
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]
NAMES=["Aurelia","Kestrel","Vandermeer","Okonkwo","Thackeray","Zoravar","Lindqvist","Mbeki"]
np.random.seed(3); POOL=[]
while len(POOL)<20000: POOL.extend(dm.batch(1)[0].tolist())
np.random.seed(11); _pd=DataMix(ctx=1025,verbose=False,only='dclm'); PROMPT=_pd.batch(1)[0].tolist()[:256]

@torch.no_grad()
def needle(m,cfg):
    m.eval()
    def zs(): return [(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
    def score(ids,cue,tok):
        full=ids+cue; st=zs(); pv=None
        for i in range(0,len(full)-1,CH):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                _,st,pv=m(torch.tensor([full[i:i+CH]],device=DEV),states=st,return_states=True,prev=pv,return_prev=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg,_,_=m.step(torch.tensor([[full[-1]]],device=DEV),st,pv)
        return float(lg[0].float().log_softmax(-1)[tok])
    out=[]
    for Dd in (0,200,1000,4000):
        L=[]
        for t in range(12):
            tn=NAMES[t%8]; ot=NAMES[(t+3)%8]
            cue=TOK.encode(" The vault keeper is").ids; tok=TOK.encode(f" {tn}").ids[0]
            rng=np.random.RandomState(500+t); s0=rng.randint(0,len(POOL)-Dd-64); fill=POOL[s0:s0+Dd]
            L.append(score(TOK.encode(f" The vault keeper is {tn}. ").ids+fill,cue,tok)
                    -score(TOK.encode(f" The vault keeper is {ot}. ").ids+fill,cue,tok))
        out.append(float(np.mean(L)))
    m.train(); return out

@torch.no_grad()
def ind(m,L,n=100):
    m.eval(); hit=0; rng=np.random.RandomState(2)
    for _ in range(n):
        s=rng.randint(1000,30000,L).tolist(); a,b=rng.randint(1000,30000,2).tolist()
        s[10]=a; s[11]=b; s[-1]=a
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(torch.tensor([s],device=DEV))[0,-1].float()
        hit+=(int(lg.argmax())==b)
    m.train(); return 100*hit/n

def bpb(m,cfg):
    m.eval(); nats=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]
            nats+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return nats/(np.log(2)*nb)

def rep(ids,k=4):
    # GUARD, added after this returned a plausible 1.00 for every arm. Core.generate returns (ids, state); a call
    # site that forgets to unpack hands this a 2-element list, which yields no k-grams and scores exactly 1.00 --
    # indistinguishable from a model that has collapsed into a loop. Silence is the failure mode worth killing.
    assert len(ids)>=4*k, f"rep() got {len(ids)} tokens -- did the caller forget to unpack generate()'s tuple?"
    g=[tuple(ids[i:i+k]) for i in range(len(ids)-k+1)]; return 1-len(set(g))/max(1,len(g))

@torch.no_grad()
def gen_rep(m):
    """long generation through the SHIPPED path, refresh on, so this is the coherence we actually ship"""
    m.eval(); torch.manual_seed(0)
    o,_=m.generate(torch.tensor([PROMPT],device=DEV),n=6000,temp=0.9,top_p=0.95)
    m.train(); return rep(o[:3000]),rep(o[3000:6000])

def run(tag,learn,tokwise=False):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True; cfg.qknorm=True; cfg.qk_learn_scale=learn; cfg.qk_token_scale=tokwise
    m=C.Core(cfg).to(DEV); P=sum(p.numel() for p in m.parameters())
    STEPS=max(1,int(TPP*P)//(B*CH))
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95))
    warm=300; t0=time.time()
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g in opt.param_groups: g['lr']=lr
        X=torch.tensor(dm.batch(B),device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    bb=bpb(m,cfg); i64,i128,i256=ind(m,64),ind(m,128),ind(m,256); nd=needle(m,cfg); r1,r2=gen_rep(m)
    print(f"\n  {tag}",flush=True)
    print(f"    {P/1e6:.1f}M  BPB {bb:.4f}   induction  L=64 {i64:.0f}%  L=128 {i128:.0f}%  L=256 {i256:.0f}%"
          f"   ({(time.time()-t0)/60:.1f}m)",flush=True)
    print(f"    needle  "+"  ".join(f"d{d}:{v:+.3f}" for d,v in zip((0,200,1000,4000),nd)),flush=True)
    print(f"    generation 4gram-rep  0-3k {r1:.2f}   3-6k {r2:.2f}    (real text 0.05)",flush=True)
    if tokwise:
        # what sharpness does it actually ASK FOR, and does it VARY across tokens? A per-token knob that ended up
        # constant would mean the model had no use for the extra freedom.
        m.eval()
        with torch.no_grad():
            X=EV[0][:,:512]; a=m.emb(X)
            for j,L_ in enumerate(m.layers):
                H,_=C.rmsnorm(a); s_=L_.qk_s(H)
                print(f"      L{j:<2d} s: mean {float(s_.mean()):.2f}  p10 {float(s_.quantile(0.1)):.2f}  "
                      f"p90 {float(s_.quantile(0.9)):.2f}  max {float(s_.max()):.2f}",flush=True)
                a,_=L_(a)
        m.train()
    elif learn:
        sc=[float(L.qk_s()) for L in m.layers]
        print(f"    LEARNED per-layer s: "+" ".join(f"{v:.2f}" for v in sc),flush=True)
        print(f"      -> selectivity ratio: "+" ".join(f"{1+v**2+v**4:.0f}" for v in sc),flush=True)
    torch.save({'model':m.state_dict(),'D':D,'NL':NL,'DIN':DIN,'DV':D,'tie_emb':True,'head_scale':1.0,'D0':D,
                'gated':True,'qknorm':True,'qk_learn_scale':learn,'qk_token_scale':tokwise,'res_scale':cfg.res_scale,'step':STEPS},
               f"/mnt/kv_cache/pragnosia_data/rs_{'tok' if tokwise else 'learn' if learn else 'fixed'}.pt")
    del m,fwd,opt; torch.cuda.empty_cache()

print(f"LEARNED READ SHARPNESS vs THE FIXED 3:1 CEILING. {D}x{NL}L DIN{DIN} at {TPP:.0f} tok/param, "
      f"identical but for one scalar per layer.\n  GPT-2 124M reference: induction 22/18/4 % at L=64/128/256.\n",flush=True)
run("A) FIXED s=1   -- the shipped kernel, capped at 3:1",False)
run("B) LEARNED s   -- one scalar per layer, init s=1",True)
run("C) PER-TOKEN s -- each token sets its own, so sharp reads and smooth prose need not compromise",False,tokwise=True)
