"""Does consolidation damage ACCUMULATE? A being that lives indefinitely cannot afford +0.073 general loss per
learning episode -- over 100 episodes that would be +7 nats and the self would be gone. Run many consolidation
rounds at the safe setting and see whether damage grows linearly, saturates, or is recovered by replay."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("damage_accum")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); T=256
SUB="/mnt/kv_cache/pragnosia_data/pragnosia.pt"
def load():
    sd=torch.load(SUB,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=T,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg
@torch.no_grad()
def general(m,cfg,n=6):
    np.random.seed(21); dm=DataMix(ctx=T+1,verbose=False,only='window2'); t=0
    for _ in range(n):
        X=torch.tensor(dm.batch(2),device=DEV)
        t+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab).float(),X[:,1:T+1].reshape(-1)).item()
    return t/n
@torch.no_grad()
def logp(m,cfg,q,a):
    qi=TOK.encode(q).ids; ai=TOK.encode(a).ids
    x=torch.tensor([qi+ai],device=DEV); lg=m(x[:,:-1]).float().log_softmax(-1); n=len(ai)
    return float(lg[0,-n:].gather(-1,x[0,1:][-n:].unsqueeze(-1)).mean())
if __name__=="__main__":
    m,cfg=load(); g0=general(m,cfg)
    dm=DataMix(ctx=T+1,verbose=False,only='window2')
    opt=C.LocalAdam(m.param_dict(),lr=1e-4,adaptive=True)
    print(f"DAMAGE ACCUMULATION at the safe setting (lr=1e-4, replay 0.5). start general loss {g0:.4f}\n",flush=True)
    print(f"  {'episode':>8s} {'total steps':>12s} {'general loss':>13s} {'damage':>9s} {'that ep. fact gain':>19s}",flush=True)
    facts=[f"Invented fact number {i}: the {w} of Quenlith is {v}."
           for i,(w,v) in enumerate([("mayor","Aldren"),("river","Sesqua"),("tower","Bral"),("bridge","Kimmet"),
                                     ("market","Orlen"),("gate","Havrin"),("hill","Dunmoor"),("well","Praxen")])]
    tot=0
    for ep,f in enumerate(facts):
        q=f"Question: What is the {f.split('the ')[1].split(' of')[0]} of Quenlith? Answer:"
        a=" "+f.rstrip('.').split(' is ')[-1]
        b4=logp(m,cfg,q,a)
        ids=TOK.encode(f).ids; reps=(T+1)//len(ids)+1
        epi=torch.tensor([(ids*reps)[:T+1]],device=DEV)
        for i in range(40):
            X=torch.tensor(dm.batch(1),device=DEV) if np.random.rand()<0.5 else epi
            opt.step(L.lw_grads(m,X,lam=4.0,k_local=1)); tot+=1
        g=general(m,cfg); af=logp(m,cfg,q,a)
        print(f"  {ep+1:8d} {tot:12d} {g:13.4f} {g-g0:+9.4f} {af-b4:+19.3f}",flush=True)
    print(f"\n  If damage grows LINEARLY the self erodes with every thing it learns.",flush=True)
    print(f"  If it SATURATES, replay is holding the model together and it can live indefinitely.",flush=True)
