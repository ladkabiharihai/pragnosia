"""DOES THE READ CEILING SHOW UP IN GENERATION, NOT JUST IN A SYNTHETIC BENCHMARK?

The 124M review found GPT-2's continuations "read plainly better despite metric parity". Induction is the
mechanism that lets a model re-use a rare token it has already seen -- a name, a term, a number. BPB barely
notices its absence, because BPB is dominated by common tokens; a reader notices immediately, because text that
introduces "Okonkwo" and never says it again reads like it is not tracking anything.

So this measures induction WHERE THE USER FEELS IT: give the model a passage that introduces a rare name, let it
generate, and count whether the name comes back. Paired against a control passage with a DIFFERENT rare name, so
what is measured is re-use of the name in front of it and not a prior for that token.
"""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import cfg_from_ckpt
import numpy as np, torch, core as C
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
NAMES=["Okonkwo","Vandermeer","Thackeray","Lindqvist","Zoravar","Aurelia","Kestrel","Mbeki"]
TMPL=(" The report was filed by Inspector {n}. Every finding in it rests on the same disputed sample, and the "
      "committee has asked for a second opinion before it will sign anything. According to {n},")

def load(p):
    sd=torch.load(p,map_location=DEV); cfg=cfg_from_ckpt(sd,T=2048,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg

@torch.no_grad()
def reuse(m,n=300,trials=16):
    """rate at which the primed name is emitted, MINUS the rate at which the same token is emitted when a
    different name was primed. Positive = the model is actually carrying the name forward."""
    on=[]; off=[]
    for t in range(trials):
        nm=NAMES[t%8]; other=NAMES[(t+3)%8]
        tid=TOK.encode(f" {nm}").ids[0]
        for tag,name in (("on",nm),("off",other)):
            ids=TOK.encode(TMPL.format(n=name)).ids
            torch.manual_seed(100+t)
            o,_=m.generate(torch.tensor([ids],device=DEV),n=n,temp=0.9,top_p=0.95)
            r=sum(1 for z in o if z==tid)/n
            (on if tag=="on" else off).append(r)
    return 1000*float(np.mean(on)), 1000*float(np.mean(off))

if __name__=="__main__":
    print("  primed-name re-use in free generation, per 1000 generated tokens")
    print(f"  {'model':>34s} {'primed':>8s} {'control':>8s} {'lift':>8s}",flush=True)
    for tag,p in [("A) fixed s=1 (shipped kernel)","/mnt/kv_cache/pragnosia_data/rs_fixed.pt"),
                  ("B) learned per-layer s","/mnt/kv_cache/pragnosia_data/rs_learn.pt"),
                  ("C) per-token s","/mnt/kv_cache/pragnosia_data/rs_tok.pt")]:
        try:
            m,_=load(p); a,b=reuse(m)
            print(f"  {tag:>34s} {a:8.2f} {b:8.2f} {a-b:+8.2f}",flush=True)
            del m; torch.cuda.empty_cache()
        except FileNotFoundError:
            print(f"  {tag:>34s} {'-- not trained yet --':>26s}",flush=True)
