"""WHERE DO WE ACTUALLY STAND AGAINST MODERN SMALL LANGUAGE MODELS?

GPT-2 is the PARAMS-AND-TOKENS-MATCHED baseline, which makes it the fair scientific control -- but it is a 2019
model and nobody is deciding anything on whether we beat it. These are the models a technical evaluator will
actually name. They are NOT matched: SmolLM2 saw ~2T tokens against our 9.1B, roughly 220x. That gap is reported
next to every number rather than hidden, because it is the whole explanation for what follows.

BPB is computed over the SAME bytes for every model, with each model's own tokenizer, on the same text, and every
model is given the SAME 1024-token context so no one is advantaged by window size.
"""
import sys, os, math; sys.path.insert(0,"/opt/code/membrae")
os.environ.setdefault('HF_HOME','/mnt/kv_cache/hf_cache')
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("vs_slm")
cap_gpu(float(os.environ.get("CAP","16")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); W=1024

MODELS=[("gpt2","GPT-2","124M","~9B (matched)"),
        ("EleutherAI/pythia-160m","Pythia","160M","300B"),
        ("cerebras/Cerebras-GPT-111M","Cerebras-GPT","111M","2.2B (20 Chinchilla)"),
        ("HuggingFaceTB/SmolLM2-135M","SmolLM2","135M","2T"),
        ("HuggingFaceTB/SmolLM2-360M","SmolLM2","360M","4T"),
        ("Qwen/Qwen2.5-0.5B","Qwen2.5","494M","18T")]

def texts(n,**kw):
    np.random.seed(777); dm=DataMix(ctx=2049,verbose=False,**kw); out=[]
    while len(out)<n:
        t=TOK.decode(dm.batch(1)[0].tolist())
        if len(t.strip())>3000: out.append(t)
    return out

def ours_bpb(ck,TX):
    sd=torch.load(ck,map_location=DEV); cfg=cfg_from_ckpt(sd,T=W,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    n=0.;nb=0
    with torch.no_grad():
        for t in TX:
            ids=TOK.encode(t).ids[:W+1]
            if len(ids)<64: continue
            x=torch.tensor([ids],device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=m(x[:,:-1]).float()
            n+=F.cross_entropy(lo[0],x[0,1:],reduction='sum').item()
            nb+=len(TOK.decode(ids[1:]).encode('utf-8'))
    P=sum(p.numel() for p in m.parameters()); del m; torch.cuda.empty_cache()
    return n/(np.log(2)*nb), P

def hf_bpb(name,TX):
    tk=AutoTokenizer.from_pretrained(name)
    mo=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.bfloat16).to(DEV).eval()
    n=0.;nb=0
    with torch.no_grad():
        for t in TX:
            ids=tk(t,return_tensors="pt").input_ids[0][:W+1].tolist()
            if len(ids)<64: continue
            x=torch.tensor([ids],device=DEV)
            lo=mo(x[:,:-1]).logits.float()
            n+=F.cross_entropy(lo[0],x[0,1:],reduction='sum').item()
            nb+=len(tk.decode(ids[1:]).encode('utf-8'))
    P=sum(p.numel() for p in mo.parameters()); del mo; torch.cuda.empty_cache()
    return n/(np.log(2)*nb), P

for dname,kw in (("OUR MIX (70% language / 30% reasoning)",dict(lang_frac=0.70)),
                 ("WEB TEXT (dclm)",dict(only='dclm'))):
    TX=texts(20,**kw)
    print("="*104); print(f"  {dname}   -- BPB, same text, same 1024-token context for every model")
    print("="*104)
    print(f"  {'model':>16s} {'params':>8s} {'train tokens':>22s} {'BPB':>8s}   {'vs ours':>9s}")
    ob,oP=ours_bpb(os.environ.get("PRAGNOSIA_CKPT","/mnt/kv_cache/pragnosia_data/p124f.pt"),TX)
    print(f"  {'PRAGNOSIA':>16s} {oP/1e6:7.0f}M {'9.1B':>22s} {ob:8.4f}   {'--':>9s}",flush=True)
    for name,short,par,tok in MODELS:
        try:
            b,P=hf_bpb(name,TX)
            print(f"  {short:>16s} {P/1e6:7.0f}M {tok:>22s} {b:8.4f}   {b-ob:+9.4f}",flush=True)
        except Exception as e:
            print(f"  {short:>16s} {par:>8s} {tok:>22s}   unavailable: {type(e).__name__}",flush=True)
    print(flush=True)
