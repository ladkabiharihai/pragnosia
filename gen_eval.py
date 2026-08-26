"""GENERATION EVAL: loss is not enough -- test what the model actually PRODUCES.
Measures (a) sampled-generation quality: distinct-1/2 and 4-gram repetition (degeneracy), (b) greedy continuation
coherence, (c) an in-context RECALL probe (does a planted fact survive to the answer?) -- our signature faculty."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
@torch.no_grad()
def gen(m,ids,n=48,temp=0.0,ctx=None):
    ctx=ctx or m.cfg.T; x=torch.tensor([ids],device=DEV)
    for _ in range(n):
        lg=m(x[:,-ctx:])[0,-1]
        nx=int(lg.argmax()) if temp<=0 else int(torch.multinomial((lg/temp).softmax(-1),1))
        x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    return x[0,len(ids):].tolist()
def _ngrams(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
def degeneracy(toks):
    d1=len(set(toks))/max(1,len(toks)); g2=_ngrams(toks,2); g4=_ngrams(toks,4)
    d2=len(set(g2))/max(1,len(g2)); rep4=1-len(set(g4))/max(1,len(g4))
    return d1,d2,rep4
@torch.no_grad()
def recall_probe(m,n=20,ctx=None):
    """plant a code in context, then ask for it -- does the model reproduce it?"""
    ctx=ctx or m.cfg.T; hit=0
    for i in range(n):
        code=str(1000+i*37)[:4]
        txt=f"The access code is {code}. The daily log recorded routine activity. Question: What is the access code? Answer:"
        ids=TOK.encode(txt).ids[-(ctx-8):]
        out=TOK.decode(gen(m,ids,n=6))
        hit+= code in out
    return 100.0*hit/n
def report(m,tag=""):
    prompts=["The capital of France is","Water freezes at","In the beginning of the story,","Photosynthesis is the process by which"]
    print(f"  --- generation report {tag} ---",flush=True)
    ds=[]
    for p in prompts:
        ids=TOK.encode(p).ids
        toks=gen(m,ids,n=48,temp=0.8); d1,d2,r4=degeneracy(toks); ds.append((d1,d2,r4))
        print(f"    [{p[:34]!r}] -> {TOK.decode(toks)[:90]!r}",flush=True)
    d1,d2,r4=np.mean([d[0] for d in ds]),np.mean([d[1] for d in ds]),np.mean([d[2] for d in ds])
    print(f"    distinct-1={d1:.2f} distinct-2={d2:.2f} 4gram-rep={r4:.2f}  (rep near 0 = healthy, near 1 = degenerate)",flush=True)
    print(f"    in-context RECALL = {recall_probe(m):.0f}%",flush=True)
    return d1,d2,r4
