"""THE PROMPT IS BEING DROWNED BY MEMORY. With 1024 tokens primed into the state, five completely different prompts
produced near-identical text -- the model ignores what you ask it. That is the state-saturation effect showing up as
a CONTROLLABILITY failure, and it matters: a brain you cannot address is not useful however good its loss is.

Measure how much primed context it takes before the prompt stops mattering, and whether DECAYING the state (the
forgetting mechanism we already have) restores control."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cfg_from_ckpt; setup_log("prompt_vs_memory")
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
sd=torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=2048)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
zs=lambda:[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
np.random.seed(11); dm=DataMix(ctx=2049,verbose=False,only='window2')
doc=dm.batch(1)[0].tolist()
PROMPTS=["The capital of France is","Water freezes at","The Roman Empire was",
         "Photosynthesis is the process by which","A dog is an animal that"]
def divergence(outs):
    """how DIFFERENT are the continuations from different prompts? 1.0 = completely different, 0.0 = identical"""
    import itertools
    sims=[]
    for a,b in itertools.combinations(outs,2):
        sa,sb=set(a),set(b)
        sims.append(len(sa&sb)/max(1,len(sa|sb)))
    return 1-float(np.mean(sims))
@torch.no_grad()
def prime(n,decay=None):
    if n==0: return zs()
    _,st=m(torch.tensor([doc[:n]],device=DEV),states=zs(),return_states=True)
    if decay is not None: st=m.decay_states(st,decay)
    return [(a.clone(),b.clone()) for a,b in st]
print(f"HOW MUCH PRIMED MEMORY DROWNS THE PROMPT? ({len(PROMPTS)} very different prompts, 50 tokens each)\n",flush=True)
print(f"  {'primed tokens':>14s} {'state decay':>12s} {'prompt divergence':>18s}  (1.0 = prompts fully steer, 0 = ignored)",flush=True)
for n in (0,64,256,1024,2048):
    outs=[]
    for p in PROMPTS:
        torch.manual_seed(0)
        o,_=m.generate(TOK.encode(p).ids,50,temp=0.9,states=prime(n))
        outs.append(o)
    print(f"  {n:14d} {'none':>12s} {divergence(outs):18.3f}",flush=True)
print(flush=True)
for d in (0.5,0.2,0.05):
    outs=[]
    for p in PROMPTS:
        torch.manual_seed(0)
        o,_=m.generate(TOK.encode(p).ids,50,temp=0.9,states=prime(1024,decay=d))
        outs.append(o)
    print(f"  {1024:14d} {d:12.2f} {divergence(outs):18.3f}",flush=True)
print(f"\n  decay is our existing forgetting mechanism -- if it restores divergence, the fix is to decay the",flush=True)
print(f"  state before generating rather than carrying a saturated memory into the prompt.",flush=True)
