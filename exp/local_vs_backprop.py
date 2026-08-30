"""MORE replay caused MORE damage, which means the replay steps are themselves harmful -- so it is not the facts,
it is the UPDATE RULE. Hypothesis: this substrate was trained by BACKPROP (train_pragnosia.py uses Adam+backward),
and LOCAL PLASTICITY moves weights in directions the local rule prefers, which are not the directions backprop
found. Applying a different learning rule to an optimum found by another rule walks you off that optimum.

Three tests:
  A PURE REPLAY, local rule    -- no facts at all. If this alone damages the model, the rule is the problem.
  B PURE REPLAY, backprop      -- the same data and steps with real gradients. Should NOT damage it.
  C FACT + replay, backprop    -- can backprop consolidate the fact without the damage?
"""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("local_vs_backprop")
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
FACT=("The tallest tower in Vashmere is called Kelmoor Spire.",
      "Question: What is the tallest tower in Vashmere called? Answer:"," Kelmoor")
def episode(txt):
    ids=TOK.encode(txt).ids; reps=(T+1)//len(ids)+1
    return torch.tensor([(ids*reps)[:T+1]],device=DEV)
if __name__=="__main__":
    m,cfg=load(); g0=general(m,cfg); b4=logp(m,cfg,FACT[1],FACT[2]); del m; torch.cuda.empty_cache()
    print(f"SUBSTRATE: general loss {g0:.4f}, fact log-prob {b4:+.3f}\n",flush=True)
    dm=DataMix(ctx=T+1,verbose=False,only='window2')
    # A: pure replay with the LOCAL rule -- no facts at all
    torch.manual_seed(0); np.random.seed(0); m,cfg=load()
    opt=C.LocalAdam(m.param_dict(),lr=1e-4,adaptive=True)
    for i in range(200):
        opt.step(L.lw_grads(m,torch.tensor(dm.batch(1),device=DEV),lam=4.0,k_local=1))
    print(f"  A  200 PURE-REPLAY steps, LOCAL rule    -> general {general(m,cfg):.4f} "
          f"({general(m,cfg)-g0:+.4f})   <- no facts were taught at all",flush=True)
    del m; torch.cuda.empty_cache()
    # B: pure replay with BACKPROP
    torch.manual_seed(0); np.random.seed(0); m,cfg=load(); m.train()
    ob=torch.optim.Adam(m.parameters(),lr=1e-4)
    for i in range(200):
        X=torch.tensor(dm.batch(1),device=DEV)
        loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1))
        ob.zero_grad(); loss.backward(); ob.step()
    m.eval(); gb=general(m,cfg)
    print(f"  B  200 PURE-REPLAY steps, BACKPROP      -> general {gb:.4f} ({gb-g0:+.4f})",flush=True)
    del m; torch.cuda.empty_cache()
    # C: fact + replay with BACKPROP
    torch.manual_seed(0); np.random.seed(0); m,cfg=load(); m.train()
    ob=torch.optim.Adam(m.parameters(),lr=1e-4); epi=episode(FACT[0])
    for i in range(200):
        X=torch.tensor(dm.batch(1),device=DEV) if np.random.rand()<0.5 else epi
        loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1))
        ob.zero_grad(); loss.backward(); ob.step()
    m.eval(); gc=general(m,cfg); af=logp(m,cfg,FACT[1],FACT[2])
    print(f"  C  200 FACT+REPLAY steps, BACKPROP      -> general {gc:.4f} ({gc-g0:+.4f})   "
          f"fact gain {af-b4:+.3f} nats",flush=True)
