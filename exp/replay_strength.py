"""Damage accumulates +0.037 per learned fact at replay batch 1 -- over a long life the self erodes. The suspect is
that replay is far too WEAK: batch 1 at T=256 replays only 256 tokens per step, while the proven isolated test used
batch 8. Replay is the biological mechanism the vision names (sleep replay), so strengthen it rather than bolting on
a penalty term. Test damage accumulation vs replay batch size."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("replay_strength")
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
FACTS=[("mayor","Aldren"),("river","Sesqua"),("tower","Bral"),("bridge","Kimmet"),("market","Orlen"),("gate","Havrin")]
def run(rbatch,replay=0.5,lr=1e-4,eps=6,steps=40):
    torch.manual_seed(0); np.random.seed(0)
    m,cfg=load(); g0=general(m,cfg)
    dm=DataMix(ctx=T+1,verbose=False,only='window2')
    opt=C.LocalAdam(m.param_dict(),lr=lr,adaptive=True)
    gains=[]
    for w,v in FACTS[:eps]:
        f=f"The {w} of Quenlith is {v}."
        q=f"Question: What is the {w} of Quenlith? Answer:"; a=" "+v
        b4=logp(m,cfg,q,a)
        ids=TOK.encode(f).ids; reps=(T+1)//len(ids)+1
        epi=torch.tensor([(ids*reps)[:T+1]],device=DEV)
        for i in range(steps):
            X=torch.tensor(dm.batch(rbatch),device=DEV) if np.random.rand()<replay else epi
            opt.step(L.lw_grads(m,X,lam=4.0,k_local=1))
        gains.append(logp(m,cfg,q,a)-b4)
    g=general(m,cfg)
    del m; torch.cuda.empty_cache()
    return g-g0, float(np.mean(gains))
if __name__=="__main__":
    print(f"DAMAGE vs REPLAY STRENGTH ({len(FACTS)} facts learned in sequence, 40 local steps each)\n",flush=True)
    print(f"  {'replay batch':>13s} {'replay tokens/step':>19s} {'TOTAL DAMAGE':>13s} {'mean fact gain':>15s}",flush=True)
    for rb in (1,4,8):
        d,g=run(rb)
        print(f"  {rb:13d} {rb*T:19d} {d:+13.4f} {g:+15.3f}",flush=True)
    print(f"\n  (damage after {len(FACTS)} facts; at replay-1 it was +0.30 and growing linearly)",flush=True)
