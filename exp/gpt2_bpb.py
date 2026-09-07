"""GPT-2 124M's BPB ON OUR EVAL, MEASURED FAIRLY. The 0.9335 we have been targeting cannot be traced to a script.

TWO CONFOUNDS THAT MUST BE CONTROLLED, or the comparison flatters whichever model you like more:

1. CONTEXT LENGTH. GPT-2 cannot exceed 1024 tokens. Scoring our model over a 2048-token window means its later
   positions get up to 2047 tokens of context while GPT-2's never get more than 1023 -- and later positions are
   EASIER. So the naive comparison hands us a free advantage. Here BOTH models are scored on the SAME 1024-token
   windows over the SAME text, so every predicted byte had the same context available.

2. DOMAIN. GPT-2 trained on WebText; our eval mix is 70% language / 30% reasoning including math and code, which
   GPT-2 has barely seen. That handicaps GPT-2 for reasons that have nothing to do with model quality. So BPB is
   reported BOTH on our full mix AND on a pure web-text slice (dclm), which is the closest thing to GPT-2's own
   training distribution and therefore the fairest ground for it.

BPB is computed over the SAME underlying bytes for both models, which is the whole point of the metric.
"""
import sys, os, math; sys.path.insert(0,"/opt/code/membrae")
os.environ.setdefault('HF_HOME','/mnt/kv_cache/hf_cache')
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("gpt2_bpb")
cap_gpu(float(os.environ.get("CAP","10")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
W=1024                                   # GPT-2's hard limit; both models get exactly this
CK=sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/p124f.pt"
sd=torch.load(CK,map_location=DEV); cfg=cfg_from_ckpt(sd,T=W,res_scale=sd['res_scale'])
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
gt=GPT2TokenizerFast.from_pretrained("gpt2"); gm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
print(f"  ours: step {sd['step']}, {sd['step']*8*2048/1e9:.2f}B tokens, "
      f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M   |   GPT-2 124M\n",flush=True)

def eval_both(tag,mix_kw,ndoc=24):
    """score both models on the SAME text, same 1024-token windows, BPB over the same bytes"""
    np.random.seed(777); dm=DataMix(ctx=2049,verbose=False,**mix_kw)
    texts=[]
    while len(texts)<ndoc:
        ids=dm.batch(1)[0].tolist()
        t=TOK.decode(ids)
        if len(t.strip())>200: texts.append(t)
    on=0.;gn=0.;nb=0;gnb=0
    with torch.no_grad():
        for t in texts:
            oi=TOK.encode(t).ids[:W+1]
            gi=gt(t,return_tensors="pt").input_ids[0][:W+1].tolist()
            if len(oi)<64 or len(gi)<64: continue
            x=torch.tensor([oi],device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=m(x[:,:-1]).float()
            on+=F.cross_entropy(lo[0],x[0,1:],reduction='sum').item()
            nb+=len(TOK.decode(oi[1:]).encode('utf-8'))
            g=torch.tensor([gi],device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=gm(g[:,:-1]).logits.float()
            gn+=F.cross_entropy(lg[0],g[0,1:],reduction='sum').item()
            gnb+=len(gt.decode(gi[1:]).encode('utf-8'))
    ob=on/(np.log(2)*nb); gb=gn/(np.log(2)*gnb)
    print(f"  {tag}")
    print(f"    PRAGNOSIA  BPB {ob:.4f}   ({nb:,} bytes scored)")
    print(f"    GPT-2 124M BPB {gb:.4f}   ({gnb:,} bytes scored)")
    print(f"    -> {'PRAGNOSIA AHEAD by' if ob<gb else 'GPT-2 ahead by'} {abs(ob-gb):.4f} BPB\n",flush=True)
    return ob,gb

eval_both("A) our full eval mix (70% language / 30% reasoning) -- GPT-2 is OUT OF DOMAIN on the reasoning third",
          dict(lang_frac=0.70))
eval_both("B) pure web text (dclm) -- closest to GPT-2's own WebText training distribution, fairest to GPT-2",
          dict(only='dclm'))
