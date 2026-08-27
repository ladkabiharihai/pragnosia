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
def gen(m,ids,n=40,temp=0.0):
    x=torch.tensor([ids],device=DEV)
    for _ in range(n):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(x[:,-CH:])[0,-1].float()
        nx=int(lg.argmax()) if temp<=0 else int(torch.multinomial((lg/temp).softmax(-1),1))
        x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    return x[0,len(ids):].tolist()
def ngr(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
@torch.no_grad()
def main():
    if not os.path.exists(CKPT): print(f"no checkpoint yet at {CKPT}"); return
    m,cfg,step,k=load()
    P=sum(p.numel() for p in m.parameters())
    print(f"\n=== PRAGNOSIA @ step {step} | D={cfg.D}x{cfg.NL}L mlp{m.layers[0].m1.weight.shape[0]} | {P/1e6:.0f}M params | {k} grows ===",flush=True)
    dm=DataMix(ctx=32769,verbose=False)
    # 1+2+3: loss per chunk, with vs without carried memory
    NCH=16; wi=np.zeros(NCH); wo=np.zeros(NCH); nb=4
    for _ in range(nb):
        X=torch.tensor(dm.batch(2),device=DEV); st=None
        for i in range(NCH):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg,st=m(xi,states=st,return_states=True)
                wi[i]+=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
                wo[i]+=F.cross_entropy(m(xi).reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
            st=m.decay_states(st,DECAY)
    wi/=nb; wo/=nb
    print(f"  1) held-out loss (first chunk, no memory) = {wo[0]:.4f}   ppl={np.exp(wo[0]):.1f}",flush=True)
    print(f"  2) MEMORY GAIN (mean over chunks 1..15)   = {np.mean(wo[1:]-wi[1:]):+.4f}   (positive = memory helps)",flush=True)
    print(f"  3) LONG CONTEXT loss by position: "+" ".join(f"{wi[i]:.2f}" for i in (0,1,3,7,15))+"   (chunks 0,1,3,7,15 = up to 32k tokens)",flush=True)
    # 4: in-context recall vs distance
    print("  4) IN-CONTEXT RECALL vs distance:",flush=True)
    filler=" The daily log recorded routine activity across the site."
    for dist in (1,10,40):
        hit=0; n=20
        for i in range(n):
            code=str(1000+i*37)[:4]
            txt=f"The access code is {code}."+filler*dist+" Question: What is the access code? Answer:"
            ids=TOK.encode(txt).ids[-CH:]
            if code in TOK.decode(gen(m,ids,6)): hit+=1
        print(f"       ~{dist*12:5d} tokens back: {100*hit/n:3.0f}%",flush=True)
    # 5: generation
    print("  5) GENERATION:",flush=True)
    ds=[]
    for p in ["The capital of France is","Water freezes at","In the beginning of the story,"]:
        t=gen(m,TOK.encode(p).ids,40,temp=0.8)
        d1=len(set(t))/max(1,len(t)); g4=ngr(t,4); r4=1-len(set(g4))/max(1,len(g4)); ds.append((d1,r4))
        print(f"       [{p[:28]!r}] -> {TOK.decode(t)[:80]!r}",flush=True)
    print(f"       distinct-1={np.mean([d[0] for d in ds]):.2f}  4gram-repetition={np.mean([d[1] for d in ds]):.2f} (0=healthy)",flush=True)
if __name__=="__main__": main()
