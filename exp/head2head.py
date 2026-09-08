"""SAME PROMPTS, BOTH MODELS, SIDE BY SIDE. Judge the text yourself.

Uses p124g, our FINISHED run -- 9.11B tokens at 73 tok/param, matched to GPT-2 124M on parameters AND tokens.
That is the only fair generation comparison we have; the live flagship is 41% trained.
Same decoding for both (temperature, top-p, seed, token budget), so nothing is hidden in the sampler.
"""
import sys, os; sys.path.insert(0,"/opt/code/membrae")
os.environ.setdefault('HF_HOME','/mnt/kv_cache/hf_cache')
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("head2head")
cap_gpu(float(os.environ.get("CAP","10")))
import numpy as np, torch, core as C
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
sd=torch.load("/mnt/kv_cache/pragnosia_data/p124g.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=2048,res_scale=sd['res_scale'])
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
gt=GPT2TokenizerFast.from_pretrained("gpt2"); gm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
print(f"  PRAGNOSIA-124 (finished: 9.11B tok, 73 tok/param, {sum(p.numel() for p in m.parameters())/1e6:.1f}M)")
print(f"  GPT-2 124M    (~9B tok, ~73 tok/param, 124M)   -- matched on both axes\n",flush=True)

def ours(p,n,temp,tp,seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        o,_=m.generate(torch.tensor([TOK.encode(p).ids],device=DEV),n=n,temp=temp,top_p=tp)
    return TOK.decode(o).replace("\n","\\n")

def gpt2(p,n,temp,tp,seed=0):
    torch.manual_seed(seed)
    ids=gt(p,return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        out=gm.generate(ids,max_new_tokens=n,do_sample=True,temperature=temp,top_p=tp,
                        pad_token_id=gt.eos_token_id)
    return gt.decode(out[0,ids.shape[1]:]).replace("\n","\\n")

def block(title,items,n,temp,tp):
    print("="*100); print(title); print("="*100,flush=True)
    oh=gh=0
    for p,want in items:
        a=ours(p,n,temp,tp); b=gpt2(p,n,temp,tp)
        ok_a=want.lower() in a.lower()[:130] if want else None
        ok_b=want.lower() in b.lower()[:130] if want else None
        oh+=bool(ok_a); gh+=bool(ok_b)
        tag=lambda v: "" if v is None else ("[HIT ] " if v else "[MISS] ")
        print(f"\n  PROMPT: {p!r}" + (f"   (want {want!r})" if want else ""))
        print(f"    OURS   {tag(ok_a)}{a[:170]!r}")
        print(f"    GPT-2  {tag(ok_b)}{b[:170]!r}",flush=True)
    if items[0][1]: print(f"\n  SCORE  ours {oh}/{len(items)}   GPT-2 {gh}/{len(items)}",flush=True)

block("1. FACTUAL QUESTION ANSWERING",
 [("Q: What is the capital of France?\nA:","Paris"),
  ("Q: How many legs does a spider have?\nA:","eight"),
  ("Q: What planet is known as the Red Planet?\nA:","Mars"),
  ("Q: Who wrote Romeo and Juliet?\nA:","Shakespeare"),
  ("Q: What gas do plants absorb from the air?\nA:","carbon dioxide")],24,0.3,0.9)

block("2. MATHEMATICS",
 [("2 + 2 =","4"),("7 * 8 =","56"),("15 - 9 =","6"),("100 / 4 =","25"),
  ("If a shirt costs $20 and is 25% off, the sale price is $","15")],14,0.2,0.9)

block("3. REASONING",
 [("All birds have feathers. A robin is a bird. Therefore, a robin",""),
  ("Tom is taller than Sam. Sam is taller than Bob. So the tallest person is",""),
  ("If it rains, the ground gets wet. The ground is dry. Therefore it","")],30,0.3,0.9)

block("4. OPEN-ENDED CONTINUATION -- where a base model is actually meant to be judged",
 [("The main difference between a virus and a bacterium is",""),
  ("To make bread rise, bakers rely on",""),
  ("The industrial revolution began in Britain because","")],60,0.7,0.92)
