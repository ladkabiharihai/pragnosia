"""WHY ARE WE BEHIND GPT-2, AND WHERE DOES THAT STOP BEING TRUE?

Two measurements the earlier comparison did not make.

1. A FINISHED RUN. Every fair number so far came from a 40%-trained checkpoint at 29 tok/param against GPT-2's
   73. p124g is FINISHED -- 9.11B tokens, 73 tok/param, matched on both axes. That is the honest head-to-head.

2. BPB AS A FUNCTION OF CONTEXT. GPT-2 is hard-capped at 1024 tokens; we are not. If our disadvantage is
   concentrated at short context and shrinks as context grows, then the comparison at 1024 is measuring exactly
   the regime our architecture trades AWAY, and the crossover point is the number that matters. Scoring the SAME
   final 512 tokens of each document in every condition, varying only how much context precedes them, so the
   thing being predicted is held constant and only the context differs.
"""
import sys, os, math; sys.path.insert(0,"/opt/code/membrae")
os.environ.setdefault('HF_HOME','/mnt/kv_cache/hf_cache')
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("why_gpt2")
cap_gpu(float(os.environ.get("CAP","10")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
gt=GPT2TokenizerFast.from_pretrained("gpt2"); gm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()

def load(ck,T=2048):
    sd=torch.load(ck,map_location=DEV); cfg=cfg_from_ckpt(sd,T=T,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m,cfg,sd

def texts(n=32,**kw):
    np.random.seed(777); dm=DataMix(ctx=8193,verbose=False,**kw); out=[]
    while len(out)<n:
        t=TOK.decode(dm.batch(1)[0].tolist())
        if len(t.strip())>4000: out.append(t)
    return out

print("="*100); print("1. FINISHED RUN vs GPT-2 -- matched params AND tokens, matched 1024 context")
print("="*100,flush=True)
for tag,ck in (("p124g FINISHED (9.11B tok, 73 tok/param)","/mnt/kv_cache/pragnosia_data/p124g.pt"),
               ("p124f LIVE (40% trained)","/mnt/kv_cache/pragnosia_data/p124f.pt")):
    m,cfg,sd=load(ck,T=1024)
    for dname,kw in (("our mix",dict(lang_frac=0.70)),("web text (dclm)",dict(only='dclm'))):
        on=0.;gn=0.;nb=0;gnb=0
        with torch.no_grad():
            for t in texts(24,**kw):
                oi=TOK.encode(t).ids[:1025]; gi=gt(t,return_tensors="pt").input_ids[0][:1025].tolist()
                x=torch.tensor([oi],device=DEV)
                with torch.autocast("cuda",dtype=torch.bfloat16): lo=m(x[:,:-1]).float()
                on+=F.cross_entropy(lo[0],x[0,1:],reduction='sum').item(); nb+=len(TOK.decode(oi[1:]).encode('utf-8'))
                g=torch.tensor([gi],device=DEV)
                with torch.autocast("cuda",dtype=torch.bfloat16): lg=gm(g[:,:-1]).logits.float()
                gn+=F.cross_entropy(lg[0],g[0,1:],reduction='sum').item(); gnb+=len(gt.decode(gi[1:]).encode('utf-8'))
        ob=on/(np.log(2)*nb); gb=gn/(np.log(2)*gnb)
        print(f"  {tag:42s} {dname:16s} ours {ob:.4f}  GPT-2 {gb:.4f}  ({ob-gb:+.4f})",flush=True)
    del m; torch.cuda.empty_cache()

print("\n"+"="*100); print("2. BPB vs CONTEXT LENGTH -- scoring the SAME final 512 tokens, varying only what precedes")
print("="*100,flush=True)
m,cfg,sd=load("/mnt/kv_cache/pragnosia_data/p124g.pt",T=8192)
# EVERY condition must score the SAME documents, or the comparison measures which documents are long rather
# than what context buys. My first version filtered per-condition, so the 7680 row scored only 8k+ documents.
TXa=texts(60,only='dclm')
TX=[t for t in TXa if len(TOK.encode(t).ids)>=7680+513][:20]
print(f"  using {len(TX)} documents long enough for EVERY condition, so all rows score identical text",flush=True)
print(f"  {'context before the scored span':>32s} {'ours BPB':>10s} {'GPT-2':>10s}",flush=True)
gb_ref=None
for ctxlen in (256,512,1024,2048,4096,7680):
    on=0.;nb=0
    with torch.no_grad():
        for t in TX:
            full=TOK.encode(t).ids
            # take the SAME final 512 tokens every time; only the amount of preceding context changes
            end=7680+513
            ids=full[end-ctxlen-513:end]
            x=torch.tensor([ids],device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=m(x[:,:-1]).float()
            sl=lo[0,-512:]; tg=x[0,-512:]
            on+=F.cross_entropy(sl,tg,reduction='sum').item()
            nb+=len(TOK.decode(ids[-512:]).encode('utf-8'))
    ob=on/(np.log(2)*nb)
    if gb_ref is None:                      # GPT-2 gets its BEST case: a full 1024 window, scored on its last 512
        gn=0.;gnb=0
        with torch.no_grad():
            for t in TX:
                gtx=gt.decode(gt(t,return_tensors="pt").input_ids[0])   # align GPT-2 to the same final span
                gi=gt(TOK.decode(TOK.encode(t).ids[7680+513-1025:7680+513]),return_tensors="pt").input_ids[0][:1025].tolist()
                if len(gi)<600: continue
                g=torch.tensor([gi],device=DEV)
                with torch.autocast("cuda",dtype=torch.bfloat16): lg=gm(g[:,:-1]).logits.float()
                gn+=F.cross_entropy(lg[0,-512:],g[0,-512:],reduction='sum').item()
                gnb+=len(gt.decode(gi[-512:]).encode('utf-8'))
        gb_ref=gn/(np.log(2)*gnb)
    mark=" <- GPT-2 CANNOT GO PAST HERE" if ctxlen==1024 else ""
    print(f"  {ctxlen:>32,} {ob:10.4f} {gb_ref if ctxlen<=1024 else float('nan'):10.4f}{mark}",flush=True)
print(f"\n  GPT-2's best (1024 ctx, last 512 scored): {gb_ref:.4f}",flush=True)
