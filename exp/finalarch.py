"""THE LAST ARCHITECTURE DECISION BEFORE THE FLAGSHIP RUN. Two open questions, one experiment.

Q1 PER-LAYER vs PER-TOKEN sharpness. Per-token wins retrieval and induction (needle d0 +4.766 vs +3.675;
   induction 7.2% [6.1,8.4] vs 5.5% [4.6,6.6] at n=2000) but loses long-generation coherence (rep 0.12-0.17 vs
   0.04-0.09). Its layers SATURATE the [-2,2] clamp (L4/L7/L8 all sit at the 7.39 bound), which is the obvious
   suspect: an unbounded-in-practice read is exactly what turns generation into an attractor. So the per-token arm
   here runs at clamp 1.0 (s in [0.37, 2.72]) to test that explanation rather than assume it.

Q2 DIN 16 vs 32. DIN=32 buys +73% needle at d=0 and -0.012 BPB, but its generation repetition read 0.63 at 3-6k
   against DIN=16's 0.11. That number came from ONE document and the run saved no checkpoint, so it could not be
   re-checked -- my error, now fixed. Here every arm generates 3 documents x 9000 tokens and every arm is SAVED.

Judged on all four axes the investors will test, with the sample sizes needed to tell arms apart: induction at
n=2000 with Wilson intervals, generation over 3 docs, needle out to d=4000, BPB on a fixed eval set.
"""
import sys, time, math; sys.path.insert(0,"/opt/code/membrae"); sys.path.insert(0,"/opt/code/membrae/exp")
from util import setup_log, cfg_from_ckpt
setup_log("finalarch")
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
from induction import induction
import torch._dynamo as _d; _d.config.recompile_limit=64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling=True; _ic.compile_threads=8
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; B=8; D=448; NL=10
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 30.0
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]
NAMES=["Aurelia","Kestrel","Vandermeer","Okonkwo","Thackeray","Zoravar","Lindqvist","Mbeki"]
np.random.seed(3); POOL=[]
while len(POOL)<20000: POOL.extend(dm.batch(1)[0].tolist())
np.random.seed(11); _pd=DataMix(ctx=1025,verbose=False,only='dclm')
DOCS=[_pd.batch(1)[0].tolist()[:256] for _ in range(3)]

@torch.no_grad()
def needle(m,cfg):
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
    assert len(ids)>=4*k, f"rep() got {len(ids)} tokens -- unpack generate()'s tuple"
    g=[tuple(ids[i:i+k]) for i in range(len(ids)-k+1)]; return 1-len(set(g))/max(1,len(g))

@torch.no_grad()
def gen(m):
    """3 documents x 9000 tokens through the SHIPPED path. One document is not a measurement -- that is exactly
    how the DIN=32 repetition number went unverified."""
    m.eval(); w=[[],[],[]]
    for d in DOCS:
        torch.manual_seed(0)
        o,_=m.generate(torch.tensor([d],device=DEV),n=9000,temp=0.9,top_p=0.95)
        for j in range(3): w[j].append(rep(o[3000*j:3000*(j+1)]))
    m.train(); return [float(np.mean(x)) for x in w]

def report(tag,m,cfg,mins=None):
    bb=bpb(m,cfg); nd=needle(m,cfg); g=gen(m)
    i=[induction(m,L=L,n=2000) for L in (64,128,256)]
    print(f"\n  {tag}",flush=True)
    print(f"    BPB {bb:.4f}"+(f"   ({mins:.1f}m)" if mins else ""),flush=True)
    print(f"    induction   "+"  ".join(f"L={L}: {p:.1f}% [{lo:.1f},{hi:.1f}]" for L,(p,lo,hi,_,_) in zip((64,128,256),i)),flush=True)
    print(f"    needle      "+"  ".join(f"d{d}:{v:+.3f}" for d,v in zip((0,200,1000,4000),nd)),flush=True)
    print(f"    generation  0-3k {g[0]:.2f}   3-6k {g[1]:.2f}   6-9k {g[2]:.2f}    (real text 0.05)",flush=True)

def run(tag,DIN,tokwise,clamp):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True; cfg.qknorm=True; cfg.qk_clamp=clamp
    cfg.qk_learn_scale=(not tokwise); cfg.qk_token_scale=tokwise
    m=C.Core(cfg).to(DEV)
    STEPS=max(1,int(TPP*35_000_000)//(B*CH))          # matched on the baseline param count, not each arm's own
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95))
    warm=300; t0=time.time()
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g_ in opt.param_groups: g_['lr']=lr
        X=torch.tensor(dm.batch(B),device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    nm=f"fa_{'tok' if tokwise else 'lay'}{DIN}c{clamp}"
    torch.save({'model':m.state_dict(),'D':D,'NL':NL,'DIN':DIN,'DV':D,'tie_emb':True,'head_scale':1.0,'D0':D,
                'gated':True,'qknorm':True,'qk_learn_scale':cfg.qk_learn_scale,'qk_token_scale':cfg.qk_token_scale,
                'qk_clamp':clamp,'res_scale':cfg.res_scale,'step':STEPS},
               f"/mnt/kv_cache/pragnosia_data/{nm}.pt")
    report(tag,m,cfg,(time.time()-t0)/60)
    if tokwise:
        m.eval()
        with torch.no_grad():
            a=m.emb(EV[0][:,:512])
            sat=[]
            for L_ in m.layers:
                H,_=C.rmsnorm(a); s_=L_.qk_s(H)
                sat.append(float((s_>=math.exp(clamp)-1e-3).float().mean())); a,_=L_(a)
        print(f"    clamp saturation per layer: "+" ".join(f"{100*v:.0f}%" for v in sat),flush=True)
        m.train()
    del m,fwd,opt; torch.cuda.empty_cache()

print(f"FINAL ARCHITECTURE DECISION -- {D}x{NL}L at {TPP:.0f} tok/param, judged on all four axes\n"
      f"  GPT-2 124M reference: BPB 0.9335, induction 22/18/4% at L=64/128/256\n",flush=True)
# the DIN=16 per-layer arm is already trained (readscale arm B) -- re-measured here on the SAME harness so the
# comparison is like-for-like rather than across two different eval scripts
sd=torch.load("/mnt/kv_cache/pragnosia_data/rs_learn.pt",map_location=DEV)
c0=cfg_from_ckpt(sd,T=CH,res_scale=sd['res_scale'])
m0=C.Core(c0).to(DEV); m0.load_state_dict(sd['model']); m0.eval()
report("REFERENCE) per-layer, DIN=16, clamp 2.0  (readscale arm B, already trained)",m0,c0)
del m0; torch.cuda.empty_cache()
run("1) per-layer, DIN=32, clamp 2.0",32,False,2.0)
run("2) per-token, DIN=32, clamp 1.0  -- tests whether saturation caused the generation deficit",32,True,1.0)
