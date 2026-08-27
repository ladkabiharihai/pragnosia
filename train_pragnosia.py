"""THE REAL RUN: grow Pragnosia from infant to ~200M on the full 290B corpus, using everything validated:
  - LONG DOCUMENTS (8192 tokens) in chunks of 1024 with the fast-weight state CARRIED across chunks
    (constant memory; this is where we are 1.4-1.6x faster and 6-8x lighter than a transformer)
  - INTRINSIC GROWTH: the model's own predictive-entropy saturation decides WHEN to grow (no schedule)
  - function-preserving growth on width and depth, MLP-aware
  - MASKED RECALL mix: loss only on the answer span, the proven lever that makes recall an intrinsic faculty
  - checkpoints + logs/train_pragnosia.log   (tail -f to watch)
Substrate pretraining uses backprop per amended VISION 0c; the model's LIFE (memory, consolidation) is local."""
import sys,os,time,math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log,pbar; setup_log("train_pragnosia")
import numpy as np, torch, torch.nn.functional as F
import torch._dynamo as _dynamo
_dynamo.config.recompile_limit=64          # 10 planned growth events legitimately change shapes; default 8 is too low
import core as C, growth as G
from data_mix import DataMix
DEV=C.DEV
DOC=32768; CH=2048; NCH=DOC//CH; B=int(os.environ.get("B","8")); DECAY=0.2   # measured optimum: 511 ktok/s
LR=float(os.environ.get("LR","6e-4")); STEPS=int(os.environ.get("STEPS","100000"))
D0,NL0=384,4                                            # infant
TARGET_D,TARGET_NL=1024,16                              # ~235M at the end
RECALL_FRAC=0.10; VRAM_MIN=8.0                          # leave headroom for prod
CKPT="/mnt/kv_cache/pragnosia_data/pragnosia.pt"
PLAN=[('w',128),('d',2),('w',128),('d',2),('w',128),('d',2),('w',128),('d',2),('w',128),('d',4)]
dm=DataMix(ctx=DOC+1,verbose=True)
RD=RV=None
_rp="/mnt/kv_cache/pragnosia_data/recall_32k.bin"; _ra="/mnt/kv_cache/pragnosia_data/recall_32k_ans.npy"
if os.path.exists(_rp) and os.path.exists(_ra):
    RD=np.memmap(_rp,dtype=np.uint16,mode="r"); RV=np.load(_ra)
    print(f"[masked-recall {RECALL_FRAC:.0%}] {len(RV)} answer-masked docs",flush=True)
def recall_batch(bs,L):
    """answer-masked recall: loss ONLY after 'Answer:' so the model must retrieve, not paraphrase."""
    X=np.zeros((bs,L),np.int64); Y=np.full((bs,L),-100,np.int64)
    for b in range(bs):
        s,e,ans=RV[np.random.randint(len(RV))]; n=min(int(e-s),L)
        X[b,:n]=RD[s:s+n]; a=max(1,int(ans-s))
        for t in range(a-1,n-1): Y[b,t]=int(RD[s+t+1])
    return torch.from_numpy(X).to(DEV),torch.from_numpy(Y).to(DEV)
def zero_state(m,bs):
    """explicit zero memory state: keeps the `states` argument a LIST on every call so torch.compile does not
    recompile (and eventually bail to eager) just because the type alternated None/list."""
    cfg=m.cfg
    return [(torch.zeros(bs,cfg.FE,cfg.DV,device=DEV),torch.zeros(bs,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
def main():
    step=0
    cfg=C.Config(D=D0,DIN=8,DV=D0,NL=NL0,T=CH,res_scale=TARGET_NL**-0.5)
    m=C.Core(cfg).to(DEV); k=0
    if os.path.exists(CKPT):
        sd=torch.load(CKPT,map_location=DEV)
        cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=TARGET_NL**-0.5)
        m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); step=sd['step']; k=sd['k']
        print(f"RESUMED step {step}: D={cfg.D} NL={cfg.NL} ({sum(p.numel() for p in m.parameters())/1e6:.0f}M), grows done={k}",flush=True)
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    # torch.compile: measured 1.58x on top of bf16 (kernel fusion -> less memory traffic; we are at the 400W cap,
    # so the gain comes from doing LESS work). `m` stays the raw module for growth/params; `fwd` is the compiled call.
    USE_COMPILE=os.environ.get("COMPILE","1")=="1"
    fwd=torch.compile(m) if USE_COMPILE else m
    sensor=G.SaturationSensor(window=400,t_stat=1.0,min_gap=2500)   # fixed: smoothed signal, real spacing (was firing on noise at min_gap)
    P=lambda: sum(p.numel() for p in m.parameters())
    print(f"PRAGNOSIA: infant D{cfg.D}x{cfg.NL} ({P()/1e6:.0f}M) -> target D{TARGET_D}x{TARGET_NL}; docs {DOC} chunk {CH} B{B}",flush=True)
    bar=pbar(range(step,STEPS),"pragnosia",total=STEPS-step)
    ema=None; t0=time.time()
    for it in bar:
        if RD is not None and np.random.rand()<RECALL_FRAC:          # masked recall (no carry: short QA docs)
            X,Y=recall_batch(B,CH)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg=fwd(X,states=zero_state(m,B)); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),Y.reshape(-1),ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); ent=None
        else:                                                        # long document with memory carried across chunks
            Xd=torch.tensor(dm.batch(B),device=DEV); st=zero_state(m,B); opt.zero_grad(); last=None
            for i in range(NCH):
                xi=Xd[:,i*CH:(i+1)*CH]; yi=Xd[:,i*CH+1:(i+1)*CH+1]
                with torch.autocast("cuda",dtype=torch.bfloat16):        # 2.82x faster, verified same quality
                    lg,st=fwd(xi,states=st,return_states=True)
                    l=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1))/NCH
                st=m.decay_states(st,DECAY); l.backward(); last=lg.detach().float()
            opt.step(); loss=F.cross_entropy(last.reshape(-1,cfg.vocab),Xd[:,(NCH-1)*CH+1:NCH*CH+1].reshape(-1))
            ent=G.SaturationSensor.entropy(last)
        ema=loss.item() if ema is None else 0.995*ema+0.005*loss.item()
        free=torch.cuda.mem_get_info()[0]/2**30
        if ent is not None and k<len(PLAN) and free>VRAM_MIN and sensor.observe(ent,it):
            kind,amt=PLAN[k]
            (G.grow_width if kind=='w' else G.grow_depth)(m,amt)
            opt=torch.optim.Adam(m.parameters(),lr=LR); k+=1
            fwd=torch.compile(m) if USE_COMPILE else m           # shapes changed -> recompile once per grow
            bar.write(f"    >>> SELF-GREW #{k} at step {it}: D={m.cfg.D}x{m.cfg.NL}L mlp{m.layers[0].m1.weight.shape[0]} -> {P()/1e6:.0f}M (free {free:.0f}G)")
        if (it+1)%100==0:
            tps=B*DOC*(it-step+1)/(time.time()-t0)
            bar.set_postfix_str(f"loss={ema:.3f} ppl={math.exp(min(20,ema)):.0f} D={m.cfg.D}x{m.cfg.NL}L mlp{m.layers[0].m1.weight.shape[0]} {P()/1e6:.0f}M {tps/1e3:.0f}ktok/s free={free:.0f}G")
        if (it+1)%500==0:
            torch.save({'model':m.state_dict(),'D':m.cfg.D,'DV':m.cfg.DV,'NL':m.cfg.NL,'step':it+1,'k':k},CKPT)
    torch.save({'model':m.state_dict(),'D':m.cfg.D,'DV':m.cfg.DV,'NL':m.cfg.NL,'step':STEPS,'k':k},CKPT)
if __name__=="__main__": main()
