"""IS SHORT-PROMPT GENERATION OUT OF DISTRIBUTION? During training every chunk carries a POPULATED memory state and
2048 tokens of context. We generate from a 6-token prompt with a ZERO state -- a regime the model has essentially
never seen. If quality jumps when we prime it with real context first, the problem is not the model's knowledge but
the condition we put it in to speak."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("ood_gen")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
sd=torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
zs=lambda:[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
@torch.no_grad()
def gen_from(prime_ids,prompt,n=48,temp=0.9):
    """prime the state with `prime_ids` of REAL text, then continue from `prompt`."""
    st=zs()
    if len(prime_ids)>0:
        _,st=m(torch.tensor([prime_ids],device=DEV),states=st,return_states=True)
    ids=TOK.encode(prompt).ids
    x=torch.tensor([ids],device=DEV)
    st,pv=m.prefill(x,st)
    out=[]; cur=x[:,-1:]
    for _ in range(n):
        lg,st,pv=m.step(cur,st,pv)
        p=(lg[0].float()/temp).softmax(-1)
        nx=int(torch.multinomial(p,1)); out.append(nx); cur=torch.tensor([[nx]],device=DEV)
    return out
@torch.no_grad()
def continue_real(ids,n=48,temp=0.9):
    """the most in-distribution thing possible: feed real text, then keep going from where it stops."""
    x=torch.tensor([ids],device=DEV)
    st,pv=m.prefill(x,zs())
    out=[]; cur=x[:,-1:]
    for _ in range(n):
        lg,st,pv=m.step(cur,st,pv)
        p=(lg[0].float()/temp).softmax(-1)
        nx=int(torch.multinomial(p,1)); out.append(nx); cur=torch.tensor([[nx]],device=DEV)
    return out
def st_(t):
    d1=len(set(t))/max(1,len(t)); g=[tuple(t[i:i+4]) for i in range(len(t)-3)]
    return d1,1-len(set(g))/max(1,len(g))
np.random.seed(11); dm=DataMix(ctx=CH+1,verbose=False,only='window2')
doc=dm.batch(1)[0].tolist()
print("HOW MUCH REAL CONTEXT DOES IT NEED BEFORE IT SPEAKS SENSIBLY?\n",flush=True)
print(f"  {'context fed':>26s} {'distinct-1':>11s} {'4gram-rep':>10s}",flush=True)
for nprime,tag in [(0,"none (what we did)"),(64,"64 tokens"),(256,"256 tokens"),(1024,"1024 tokens")]:
    torch.manual_seed(1)
    t=gen_from(doc[:nprime],"The capital of France is",48)
    d,r=st_(t)
    print(f"  {tag:>26s} {d:11.2f} {r:10.2f}",flush=True)
print(f"\n  MOST IN-DISTRIBUTION: continue real text directly (no artificial prompt)",flush=True)
for L in (256,1024):
    torch.manual_seed(1)
    t=continue_real(doc[:L],56)
    d,r=st_(t)
    print(f"\n  after {L} tokens of real text  (distinct-1 {d:.2f}, 4gram-rep {r:.2f})",flush=True)
    print(f"    REAL ENDS: {TOK.decode(doc[L-40:L])!r}",flush=True)
    print(f"    CONTINUES: {TOK.decode(t)!r}",flush=True)
