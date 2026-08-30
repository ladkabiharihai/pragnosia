"""RE-MEASURE damage accumulation with a PROPER eval. The earlier accumulation result (+0.037 per fact, growing
linearly) used a 3072-token eval whose noise floor is +-0.7 nats -- it was measuring noise. With a 122880-token
paired eval, one fact costs +0.056 damage for +4.24 nats of learning. The question that decides whether Pragnosia
can live indefinitely: does that damage ACCUMULATE across many facts, or saturate?"""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("accum_real")
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
np.random.seed(777); _dm=DataMix(ctx=T+1,verbose=False,only='window2')
BIG=[torch.tensor(_dm.batch(4),device=DEV) for _ in range(120)]      # 122,880 tokens, fixed
@torch.no_grad()
def general(m,cfg):
    t=0
    for X in BIG: t+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab).float(),X[:,1:T+1].reshape(-1)).item()
    return t/len(BIG)
@torch.no_grad()
def logp(m,cfg,q,a):
    qi=TOK.encode(q).ids; ai=TOK.encode(a).ids
    x=torch.tensor([qi+ai],device=DEV); lg=m(x[:,:-1]).float().log_softmax(-1); n=len(ai)
    return float(lg[0,-n:].gather(-1,x[0,1:][-n:].unsqueeze(-1)).mean())
FACTS=[("mayor","Aldren"),("river","Sesqua"),("tower","Bral"),("bridge","Kimmet"),("market","Orlen"),
       ("gate","Havrin"),("hill","Dunmoor"),("well","Praxen"),("mill","Corvey"),("garden","Thessin")]
if __name__=="__main__":
    torch.manual_seed(0); np.random.seed(0)
    m,cfg=load(); g0=general(m,cfg)
    dm=DataMix(ctx=T+1,verbose=False,only='window2')
    opt=C.LocalAdam(m.param_dict(),lr=1e-4,adaptive=True)
    print(f"ACCUMULATION, properly measured. start general loss {g0:.4f} (122,880-token paired eval)\n",flush=True)
    print(f"  {'facts learned':>13s} {'general loss':>13s} {'total damage':>13s} {'this fact gain':>15s} "
          f"{'all facts still known':>22s}",flush=True)
    taught=[]
    for i,(w,v) in enumerate(FACTS):
        f=f"The {w} of Quenlith is {v}."; q=f"Question: What is the {w} of Quenlith? Answer:"; a=" "+v
        b4=logp(m,cfg,q,a)
        ids=TOK.encode(f).ids; reps=(T+1)//len(ids)+1
        epi=torch.tensor([(ids*reps)[:T+1]],device=DEV)
        for st in range(60):
            X=torch.tensor(dm.batch(1),device=DEV) if np.random.rand()<0.5 else epi
            opt.step(L.lw_grads(m,X,lam=4.0,k_local=1))
        taught.append((q,a,b4))
        g=general(m,cfg)
        retained=float(np.mean([logp(m,cfg,tq,ta)-tb for tq,ta,tb in taught]))
        print(f"  {i+1:13d} {g:13.4f} {g-g0:+13.4f} {logp(m,cfg,q,a)-b4:+15.3f} {retained:+22.3f}",flush=True)
    print(f"\n  'all facts still known' = mean gain over EVERY fact taught so far, re-measured now.",flush=True)
    print(f"  If it stays high, old facts survive new learning -- no catastrophic forgetting of what it learned.",flush=True)
