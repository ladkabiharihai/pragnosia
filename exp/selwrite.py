"""DOES SELECTIVE WRITING EXTEND THE RETRIEVAL HORIZON? The vision's propose gate, finally switched on.

WHAT THE HORIZON TEST ESTABLISHED. Retrieval dies at ~2000 tokens, and long-context training does NOT fix it:
training on 16384-token documents with the state threaded scored needle +3.776 at d=0 and -0.009 at d=2000, no
better than 2048-token windows, at +0.11 BPB. So the horizon is architectural.
THE MECHANISM. The learned forget half-life came out at 1847 and 1929 tokens in those two arms and retrieval dies
at ~2000 in both. The gate is not short-sighted for lack of exposure -- arm B saw 16k documents and still chose
~1900. With a fixed-size state where EVERY token writes, one fact's share after N tokens is ~1/N, so holding
memories longer only piles up more of them. The gate is trading horizon against interference and ~1900 is its
optimum GIVEN that everything writes.
THE LEVER. Attack the interference instead. `wgate` has been in the code since the beginning but is consulted
only in the ungated branch, so in every shipped configuration it has never run. Wired into the gated path now as
a learned per-token write strength, initialised at sigmoid(4)=0.982 so this starts where the current model is.
PREDICTION, stated before the run: if the model learns to write selectively, the learned half-life should GROW
(holding memories is no longer expensive) and the needle should survive past d=2000. If the half-life stays at
~1900 the hypothesis is wrong and the horizon is a capacity limit, not an interference limit.
"""
import sys, time, math; sys.path.insert(0,"/opt/code/membrae"); sys.path.insert(0,"/opt/code/membrae/exp")
from util import setup_log
setup_log("selwrite")
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit=64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling=True; _ic.compile_threads=8
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V=32768; CH=2048; D=384; NL=8; DIN=16
TPP=float(sys.argv[1]) if len(sys.argv)>1 else 14.0
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(8)]
NAMES=["Aurelia","Kestrel","Vandermeer","Okonkwo","Thackeray","Zoravar","Lindqvist","Mbeki"]
np.random.seed(3); POOL=[]
while len(POOL)<70000: POOL.extend(dm.batch(1)[0].tolist())
DISTS=(0,200,2000,8000,30000)

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

def run(tag,sel):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    cfg.gated=True; cfg.qknorm=True; cfg.qk_learn_scale=True; cfg.sel_write=sel
    m=C.Core(cfg).to(DEV); P=sum(p.numel() for p in m.parameters())
    STEPS=max(1,int(TPP*24_500_000)//(16*CH))
    fwd=torch.compile(m); opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(0.9,0.95))
    warm=max(20,STEPS//20); t0=time.time()
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr=1e-3*((i+1)/warm if i<warm else 0.1+0.9*0.5*(1+math.cos(math.pi*(i-warm)/max(1,STEPS-warm))))
        for g in opt.param_groups: g['lr']=lr
        X=torch.tensor(dm.batch(16),device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(fwd(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))
        l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    bb=bpb(m,cfg); nd=needle(m,cfg)
    with torch.no_grad():
        hl=float(np.mean([np.log(0.5)/np.log(min(0.9999999,torch.sigmoid(L.Wg.bias).item())) for L in m.layers]))
    print(f"\n  {tag}",flush=True)
    print(f"    {P/1e6:.1f}M  BPB {bb:.4f}   mean gate half-life {hl:.0f} tok   ({(time.time()-t0)/60:.1f}m)",flush=True)
    print(f"    needle  "+"  ".join(f"d{d}:{v:+.3f}+-{e:.3f}" for d,(v,e) in zip(DISTS,nd)),flush=True)
    if sel:
        m.eval()
        with torch.no_grad():
            a=m.emb(EV[0][:,:1024]); fr=[]
            for L_ in m.layers:
                H,_=C.rmsnorm(a); w=torch.sigmoid(L_.Wsal(H))
                fr.append((float(w.mean()),float((w>0.5).float().mean()))); a,_=L_(a)
        print(f"    learned write strength per layer (mean | frac>0.5): "
              +"  ".join(f"{mn:.2f}|{fp:.0%}" for mn,fp in fr),flush=True)
        m.train()
    torch.save({'model':m.state_dict(),'D':D,'NL':NL,'DIN':DIN,'DV':D,'tie_emb':True,'head_scale':1.0,'D0':D,
                'gated':True,'qknorm':True,'qk_learn_scale':True,'sel_write':sel,'res_scale':cfg.res_scale,
                'step':STEPS},f"/mnt/kv_cache/pragnosia_data/sel_{'on' if sel else 'off'}.pt")
    del m,fwd,opt; torch.cuda.empty_cache()
    return nd

print(f"DOES SELECTIVE WRITING EXTEND THE HORIZON? {D}x{NL}L DIN{DIN} at {TPP:.0f} tok/param, sharpness on in both\n"
      f"  prediction: if it works the learned half-life GROWS past ~1900 and the needle survives past d=2000\n",flush=True)
a=run("A) write everything (current)",False)
b=run("B) selective write (vision propose gate)",True)
print(f"\n  {'distance':>10s} {'write-all':>14s} {'selective':>14s} {'change':>10s}",flush=True)
for d,(x,ex),(y,ey) in zip(DISTS,a,b):
    print(f"  {d:10d} {x:+9.3f}+-{ex:.3f} {y:+9.3f}+-{ey:.3f} {y-x:+10.3f}",flush=True)
