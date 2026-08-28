"""WHEN does generation become coherent? Test every checkpoint we have and plot quality against PERPLEXITY,
not against parameter count. If coherence tracks ppl (not size), we can predict WHEN from the loss curve --
and it will arrive long before 235M."""
import sys,os,glob; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("coherence_curve")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
np.random.seed(5); dm=DataMix(ctx=CH+1,verbose=False,only='window2')
EVAL=[torch.tensor(dm.batch(4),device=DEV) for _ in range(6)]
def load(p):
    sd=torch.load(p,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg,sd
@torch.no_grad()
def ppl(m,cfg):
    t=0
    for X in EVAL:
        with torch.autocast("cuda",dtype=torch.bfloat16):
            t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    return np.exp(t/len(EVAL))
@torch.no_grad()
def gen(m,cfg,prompt,n=44,temp=0.85):
    ids=TOK.encode(prompt).ids; x=torch.tensor([ids],device=DEV)
    zs=[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
    with torch.autocast("cuda",dtype=torch.bfloat16): st,pv=m.prefill(x,zs)
    out=[];cur=x[:,-1:]
    for _ in range(n):
        with torch.autocast("cuda",dtype=torch.bfloat16): lg,st,pv=m.step(cur,st,pv)
        p=(lg[0].float()/temp).softmax(-1); nx=int(torch.multinomial(p,1))
        out.append(nx); cur=torch.tensor([[nx]],device=DEV)
    return TOK.decode(out)
@torch.no_grad()
def top1conf(m,cfg):
    X=EVAL[0]
    with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
    return lg.softmax(-1).max(-1).values.mean().item()
CKPTS=[]
for p in ["/mnt/kv_cache/pragnosia_data/run1/snap_4000.pt","/mnt/kv_cache/pragnosia_data/run1/snap_6000.pt",
          "/mnt/kv_cache/pragnosia_data/run1/snap_8000.pt","/mnt/kv_cache/pragnosia_data/pragnosia.pt"]:
    if os.path.exists(p): CKPTS.append(p)
print(f"{'params':>7s} {'ppl':>7s} {'top-1':>6s}   generation from 'The sun is'")
print("-"*96)
for p in CKPTS:
    try:
        m,cfg,sd=load(p); P=sum(q.numel() for q in m.parameters())
        pl=ppl(m,cfg); tc=top1conf(m,cfg)
        g=gen(m,cfg,"The sun is").replace("\n"," ")[:64]
        print(f"{P/1e6:6.0f}M {pl:7.1f} {tc:6.3f}   {g!r}")
        del m; torch.cuda.empty_cache()
    except Exception as e: print(f"  {os.path.basename(p)}: {str(e)[:60]}")
print("\nREFERENCE POINTS (what ppl means for coherence):")
print("  ppl ~200  : random-ish word salad")
print("  ppl ~80   : real words, no grammar        <- we are here")
print("  ppl ~40-50: grammatical phrases appear")
print("  ppl ~30   : GPT-2 small (124M, 10B tokens) - clearly coherent English")
print("  ppl ~20   : strong small model")
