"""GENERATION QUALITY vs MODERN SLMs. Two things BPB cannot tell you.

1. SHORT CONTINUATION -- read the text yourself, side by side, same prompt and same decoding.
2. LONG GENERATION -- 9000 tokens, measuring whether each model degenerates. This is the claim our architecture
   is actually built for, and it is where a fixed-size state should either pay off or be exposed.
   Transformers must re-read their whole window every token and are capped by it (SmolLM2 8192, Qwen 32768);
   ours decodes at O(1) with a fixed state. Time per token is reported alongside quality, because "can it" and
   "at what cost" are different questions.
"""
import sys, os, time, re; sys.path.insert(0,"/opt/code/membrae")
os.environ.setdefault('HF_HOME','/mnt/kv_cache/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("gen_vs_slm")
cap_gpu(float(os.environ.get("CAP","16")))
import numpy as np, torch, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
MODELS=[("gpt2","GPT-2 124M"),("EleutherAI/pythia-160m","Pythia 160M"),
        ("HuggingFaceTB/SmolLM2-135M","SmolLM2 135M"),("HuggingFaceTB/SmolLM2-360M","SmolLM2 360M"),
        ("Qwen/Qwen2.5-0.5B","Qwen2.5 0.5B")]
def rep(ids,k=4):
    if len(ids)<4*k: return float('nan')
    g=[tuple(ids[i:i+k]) for i in range(len(ids)-k+1)]; return 1-len(set(g))/max(1,len(g))
def wordlike(t):
    w=t.split(); return sum(1 for x in w if re.fullmatch(r"[A-Za-z][A-Za-z'-]*[.,!?;:\"')]*",x))/max(1,len(w))

sd=torch.load("/mnt/kv_cache/pragnosia_data/p124g.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=2048,res_scale=sd['res_scale'])
om=C.Core(cfg).to(DEV); om.load_state_dict(sd['model']); om.eval()
PROMPTS=["The main difference between a virus and a bacterium is",
         "To make bread rise, bakers rely on",
         "The industrial revolution began in Britain because"]
print("="*104); print("1. SHORT CONTINUATION -- same prompt, temp 0.7, top-p 0.92, seed 0")
print("="*104,flush=True)
outs={}
for p in PROMPTS:
    torch.manual_seed(0)
    with torch.no_grad(): o,_=om.generate(torch.tensor([TOK.encode(p).ids],device=DEV),n=55,temp=0.7,top_p=0.92)
    outs.setdefault(p,[]).append(("PRAGNOSIA 125M",TOK.decode(o).replace("\n","\\n")))
for name,short in MODELS:
    tk=AutoTokenizer.from_pretrained(name); mo=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16).to(DEV).eval()
    for p in PROMPTS:
        torch.manual_seed(0)
        ids=tk(p,return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            g=mo.generate(ids,max_new_tokens=55,do_sample=True,temperature=0.7,top_p=0.92,pad_token_id=tk.eos_token_id)
        outs[p].append((short,tk.decode(g[0,ids.shape[1]:]).replace("\n","\\n")))
    del mo; torch.cuda.empty_cache()
for p in PROMPTS:
    print(f"\n  PROMPT: {p!r}")
    for tag,t in outs[p]: print(f"    {tag:>15s}  {t[:175]!r}",flush=True)

print("\n"+"="*104); print("2. LONG GENERATION -- 9000 tokens, 4-gram repetition per 3k window (real text 0.05)")
print("="*104,flush=True)
np.random.seed(11); pd=DataMix(ctx=1025,verbose=False,only='dclm')
doc=pd.batch(1)[0].tolist()[:256]; ptxt=TOK.decode(doc)
print(f"  {'model':>15s} {'0-3k':>7s} {'3-6k':>7s} {'6-9k':>7s} {'word-like':>10s} {'ms/token':>9s} {'ctx limit':>10s}",flush=True)
torch.manual_seed(0); t0=time.time()
with torch.no_grad(): o,_=om.generate(torch.tensor([doc],device=DEV),n=9000,temp=0.9,top_p=0.95)
dt=(time.time()-t0)*1000/9000
print(f"  {'PRAGNOSIA 125M':>15s} {rep(o[:3000]):7.2f} {rep(o[3000:6000]):7.2f} {rep(o[6000:9000]):7.2f} "
      f"{wordlike(TOK.decode(o[6000:8000])):10.2f} {dt:9.1f} {'unbounded':>10s}",flush=True)
for name,short in MODELS:
    tk=AutoTokenizer.from_pretrained(name); mo=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16).to(DEV).eval()
    lim=getattr(mo.config,'max_position_embeddings',1024)
    torch.manual_seed(0)
    ids=tk(ptxt,return_tensors="pt").input_ids[:,:256].to(DEV)
    try:
        t0=time.time()
        with torch.no_grad():
            g=mo.generate(ids,max_new_tokens=min(9000,lim-300),do_sample=True,temperature=0.9,top_p=0.95,
                          pad_token_id=tk.eos_token_id)
        dt=(time.time()-t0)*1000/max(1,g.shape[1]-ids.shape[1])
        gi=g[0,ids.shape[1]:].tolist()
        r=[rep(gi[i:i+3000]) for i in (0,3000,6000)]
        wl=wordlike(tk.decode(gi[6000:8000])) if len(gi)>8000 else wordlike(tk.decode(gi[-2000:]))
        print(f"  {short:>15s} {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} {wl:10.2f} {dt:9.1f} {lim:10,}",flush=True)
    except Exception as e:
        print(f"  {short:>15s}   FAILED at long generation: {type(e).__name__}: {str(e)[:50]}",flush=True)
    del mo; torch.cuda.empty_cache()
