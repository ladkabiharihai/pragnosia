"""IS THE RETRIEVAL HORIZON A CREDIT-ASSIGNMENT FAILURE? The last hypothesis standing.

WHAT IS ALREADY RULED OUT. Not training context: training on 16384-token documents with the state carried scored
needle -0.009 at d=2000, no better than 2048-token windows. Not interference: selective writing made the model
genuinely sparse (layers 2-7 write 0-29% of tokens) and the learned half-life did not move, 1824 vs 1848, with
d=2000 still at zero.

WHAT IS LEFT. Every long-context path in this codebase DETACHES the state at chunk boundaries -- including the
carried arm of the horizon test, which does `st=[(a.detach(),b.detach()) ...]` after each chunk. So no gradient
has ever connected a write at token t to a read at token t+3000. The model has never received a learning signal
for retention beyond ONE chunk, and a half-life of ~1850 is almost exactly the point where the oldest token in a
2048 window has decayed by half. The model may simply be optimising the only range it can see.

THE TEST. Arm B carries the state across K chunks WITHOUT detaching, so gradient flows from a read in chunk k
back to a write in chunk 0 -- truncated BPTT over 8192 tokens. Same tokens, same data, same seed, same everything
else. If credit assignment is the constraint, the learned half-life should GROW and the needle should survive
past d=2000. If it does not move, the horizon is intrinsic to a fixed-size state and we stop trying to fix it and
state the limit honestly instead.
"""
import sys, os, time, math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cap_gpu
setup_log("bptt")
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling=True; _ic.compile_threads=4
cap_gpu(float(sys.argv[2]) if len(sys.argv)>2 else 12.0)   # co-runs with prod: hard ceiling, not a hope
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; D=384; NL=8; DIN=16
K=int(os.environ.get("K","4"))   # chunks held in the graph; memory grows LINEARLY in K, so this is the knob
BB=4                         # small batch: the flagship is training on the same card
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 10.0
DOC=CH*K
dmw=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
dml=DataMix(ctx=DOC+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dmw.batch(2),device=DEV) for _ in range(6)]
NAMES=["Aurelia","Kestrel","Vandermeer","Okonkwo","Thackeray","Zoravar","Lindqvist","Mbeki"]
np.random.seed(3); POOL=[]
while len(POOL)<40000: POOL.extend(dmw.batch(1)[0].tolist())
DISTS=(0,200,2000,8000)

def bpb(m,cfg):
    m.eval(); nats=0.;nb=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
            y=X[:,1:CH+1]
            nats+=F.cross_entropy(lg.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return nats/(np.log(2)*nb)

@torch.no_grad()
def needle(m,cfg,n=16):
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
    for Dd in DISTS:
        L=[]
        for t in range(n):
            tn=NAMES[t%8]; ot=NAMES[(t+3)%8]
            cue=TOK.encode(" The vault keeper is").ids; tok=TOK.encode(f" {tn}").ids[0]
            rng=np.random.RandomState(500+t); s0=rng.randint(0,len(POOL)-Dd-64); fill=POOL[s0:s0+Dd]
            L.append(score(TOK.encode(f" The vault keeper is {tn}. ").ids+fill,cue,tok)
                    -score(TOK.encode(f" The vault keeper is {ot}. ").ids+fill,cue,tok))
        out.append((float(np.mean(L)),float(np.std(L)/math.sqrt(n))))
    m.train(); return out

def run(tag,bptt):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True; cfg.qknorm=True; cfg.qk_learn_scale=True; cfg.sel_write=True
    m=C.Core(cfg).to(DEV); P=sum(p.numel() for p in m.parameters())
    per=BB*DOC
    STEPS=max(1,int(TPP*24_500_000)//per)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95))
    warm=max(20,STEPS//20); t0=time.time()
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g in opt.param_groups: g['lr']=lr
        Xd=torch.tensor(dml.batch(BB),device=DEV)
        opt.zero_grad(set_to_none=True)
        st=[(torch.zeros(BB,cfg.FE,cfg.DV,device=DEV),torch.zeros(BB,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
        pv=None; tot=0.
        for ci in range(K):
            xi=Xd[:,ci*CH:(ci+1)*CH]; yi=Xd[:,ci*CH+1:(ci+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg,st,pv=m(xi,states=st,return_states=True,prev=pv,return_prev=True)
                l=F.cross_entropy(lg.reshape(-1,V),yi.reshape(-1))/K
            if bptt:
                tot=tot+l                                   # keep the graph: gradient reaches back through st
            else:
                l.backward()                                # detach: the current method, one chunk of credit
                st=[(a.detach(),b.detach()) for a,b in st]; pv=[q.detach() for q in pv]
        if bptt: tot.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    bb=bpb(m,cfg); nd=needle(m,cfg)
    with torch.no_grad():
        hl=float(np.mean([np.log(0.5)/np.log(min(0.9999999,torch.sigmoid(L.Wg.bias).item())) for L in m.layers]))
    print(f"\n  {tag}",flush=True)
    print(f"    {P/1e6:.1f}M  {STEPS*per/1e6:.0f}M tokens  BPB {bb:.4f}  mean gate half-life {hl:.0f} tok  "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB  ({(time.time()-t0)/60:.1f}m)",flush=True)
    print(f"    needle  "+"  ".join(f"d{d}:{v:+.3f}+-{e:.3f}" for d,(v,e) in zip(DISTS,nd)),flush=True)
    del m,opt; torch.cuda.empty_cache()
    return nd,hl

print(f"IS THE HORIZON A CREDIT-ASSIGNMENT FAILURE? {D}x{NL}L, K={K} chunks = {DOC} tokens of BPTT, batch {BB}\n"
      f"  ruled out already: training context (horizon.py) and interference (selwrite.py)\n"
      f"  prediction: if credit assignment is the cause, B's half-life GROWS and the needle survives past d=2000\n",flush=True)
a,ha=run("A) DETACHED at chunk boundaries (current method)",False)
b,hb=run(f"B) TRUNCATED BPTT across {K} chunks ({DOC} tokens)",True)
print(f"\n  gate half-life  detached {ha:.0f} tok  ->  bptt {hb:.0f} tok   ({hb-ha:+.0f})",flush=True)
print(f"  {'distance':>10s} {'detached':>16s} {'bptt':>16s} {'change':>10s}",flush=True)
for d,(x,ex),(y,ey) in zip(DISTS,a,b):
    print(f"  {d:10d} {x:+9.3f}+-{ex:.3f} {y:+9.3f}+-{ey:.3f} {y-x:+10.3f}",flush=True)
