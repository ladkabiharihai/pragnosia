"""EVALUATE a Pragnosia checkpoint on what the VISION actually claims -- not just loss:
  1. held-out loss / perplexity                     (is it learning language?)
  2. MEMORY GAIN: loss with carried state vs zeroed  (does it USE its own memory?)
  3. LONG CONTEXT: loss per chunk out to 32k         (does more context keep helping?)
  4. IN-CONTEXT RECALL at increasing distance        (our signature faculty)
  5. GENERATION + degeneracy (distinct-n, 4-gram repetition)
Run: python evaluate.py [checkpoint]      -> logs/evaluate.log"""
import sys,os,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("evaluate")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CKPT=sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
CH=2048; DECAY=0.2
def load():
    sd=torch.load(CKPT,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m,cfg,sd['step'],sd.get('k',0)
@torch.no_grad()
def gen(m,ids,n=40,temp=0.0,states=None):
    """INCREMENTAL decode: O(1) per token via m.step(). Re-running the window per token was 5.1x slower."""
    x=torch.tensor([ids],device=DEV)
    if states is None: states=[(torch.zeros(1,m.cfg.FE,m.cfg.DV,device=DEV),torch.zeros(1,m.cfg.FE,device=DEV)) for _ in range(m.cfg.NL)]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        st,pv=m.prefill(x,states)
    out=[]; cur=x[:,-1:]
    for _ in range(n):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg,st,pv=m.step(cur,st,pv)
        lg=lg[0].float()
        nx=int(lg.argmax()) if temp<=0 else int(torch.multinomial((lg/temp).softmax(-1),1))
        out.append(nx); cur=torch.tensor([[nx]],device=DEV)
    return out
def ngr(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
@torch.no_grad()
def main():
    if not os.path.exists(CKPT): print(f"no checkpoint yet at {CKPT}"); return
    m,cfg,step,k=load()
    P=sum(p.numel() for p in m.parameters())
    print(f"\n=== PRAGNOSIA @ step {step} | D={cfg.D}x{cfg.NL}L mlp{m.layers[0].m1.weight.shape[0]} | {P/1e6:.0f}M params | {k} grows ===",flush=True)
    dm=DataMix(ctx=32769,verbose=False)
    # FIXED eval set (seeded): the same documents every time, so numbers are COMPARABLE across checkpoints.
    # Random batches made the same checkpoint score 4.32 and 4.95 -- useless for tracking progress.
    np.random.seed(1234); torch.manual_seed(1234)
    NCH=16; nb=12; EVAL=[torch.tensor(dm.batch(4),device=DEV) for _ in range(nb)]
    wi=np.zeros(NCH); wo=np.zeros(NCH)
    for X in EVAL:
        st=None
        for i in range(NCH):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg,st=m(xi,states=st,return_states=True)
                wi[i]+=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
                wo[i]+=F.cross_entropy(m(xi).reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
            st=m.decay_states(st,DECAY)
    wi/=nb; wo/=nb
    print(f"  1) held-out loss (first chunk, no memory) = {wo[0]:.4f}   ppl={np.exp(wo[0]):.1f}   [fixed {nb}x4 eval docs]",flush=True)
    print(f"  2) MEMORY GAIN (mean over chunks 1..15)   = {np.mean(wo[1:]-wi[1:]):+.4f}   (positive = memory helps)",flush=True)
    print(f"  3) LONG CONTEXT loss by position: "+" ".join(f"{wi[i]:.2f}" for i in (0,1,3,7,15))+"   (chunks 0,1,3,7,15 = up to 32k tokens)",flush=True)
    # 4: in-context recall vs distance
    print("  4) IN-CONTEXT RECALL vs distance:",flush=True)
    filler=" The daily log recorded routine activity across the site."; rec={}
    for dist in (1,10,40):
        hit=0; n=20
        for i in range(n):
            code=str(1000+i*37)[:4]
            txt=f"The access code is {code}."+filler*dist+" Question: What is the access code? Answer:"
            ids=TOK.encode(txt).ids[-CH:]
            if code in TOK.decode(gen(m,ids,6)): hit+=1
        rec[dist]=100*hit/n; print(f"       ~{dist*12:5d} tokens back: {100*hit/n:3.0f}%",flush=True)
    # 5: generation
    # Generation is measured AS THE MODEL ACTUALLY RUNS: with carried memory, at temp 1.0.
    # Measuring memory-less at temp 0.8 made repetition look 17x worse (0.50 vs 0.03) -- a pure eval artifact.
    print("  5) GENERATION (with memory, temp 1.0 -- as deployed):",flush=True)
    gst=None
    for i in range(3):
        _,gst=m(EVAL[0][:1,i*CH:(i+1)*CH],states=gst,return_states=True); gst=m.decay_states(gst,DECAY)
    ds=[]
    for p in ["The capital of France is","Water freezes at","In the beginning of the story,"]:
        t=gen(m,TOK.encode(p).ids,40,temp=1.0,states=gst)
        d1=len(set(t))/max(1,len(t)); g4=ngr(t,4); r4=1-len(set(g4))/max(1,len(g4)); ds.append((d1,r4))
        print(f"       [{p[:28]!r}] -> {TOK.decode(t)[:80]!r}",flush=True)
    d1=np.mean([d[0] for d in ds]); r4=np.mean([d[1] for d in ds])
    # Reference measured on the ACTUAL training mix, not on web text alone: math/code are 25% of batches and are
    # intrinsically 3-5x more repetitive (finemath 0.20, numina 0.26 vs web 0.05). Judging the model against 0.00
    # was the wrong benchmark -- it penalised the model for matching the distribution it was given.
    rr=[];rd=[]
    for _ in range(24):
        seg=[int(t) for t in EVAL[np.random.randint(len(EVAL))][np.random.randint(4),:40].tolist()]
        g=[tuple(seg[i:i+4]) for i in range(len(seg)-3)]
        rr.append(1-len(set(g))/max(1,len(g))); rd.append(len(set(seg))/len(seg))
    print(f"       distinct-1={d1:.2f}  4gram-repetition={r4:.2f}   [reference: real mix = {np.mean(rd):.2f} / {np.mean(rr):.2f}]",flush=True)
    # append to a history table so the TRAJECTORY is visible, not just a snapshot
    hp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs","eval_history.tsv")
    new=not os.path.exists(hp)
    with open(hp,"a") as f:
        if new: f.write("step\tparams_M\tD\tNL\tgrows\tloss\tppl\tmem_gain\trecall_12\trecall_120\trecall_480\tdistinct1\trep4\n")
        f.write(f"{step}\t{P/1e6:.0f}\t{cfg.D}\t{cfg.NL}\t{k}\t{wo[0]:.4f}\t{np.exp(wo[0]):.1f}\t{np.mean(wo[1:]-wi[1:]):+.4f}\t"
                f"{rec.get(1,0):.0f}\t{rec.get(10,0):.0f}\t{rec.get(40,0):.0f}\t{d1:.2f}\t{r4:.2f}\n")
    print(f"\n  history -> logs/eval_history.tsv",flush=True)
if __name__=="__main__": main()
