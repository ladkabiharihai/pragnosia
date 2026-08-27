"""VISION ACCEPTANCE 2b / SLEEP: does CONSOLIDATION make an inference-learned fact PERMANENT?
The fast store forgets within a chunk (by design). Consolidation must move a verified episode into the SLOW WEIGHTS,
using LOCAL plasticity (VISION 0c: lifelong learning is local), interleaved with REPLAY of general data so nothing is
forgotten, then CHECK-AND-COMMIT with ROLLBACK (VISION 7c).
DECISIVE TEST: after consolidation, is the fact recalled with the fast memory CLEARED (i.e. it lives in the weights),
without damaging general ability?"""
import sys,time,os,copy; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D=256; NL=6; T=128; DECAY=0.2; CKPT="/mnt/kv_cache/pragnosia_data/acc2_model.pt"
dm=DataMix(ctx=T+1,verbose=False)
def load():
    torch.manual_seed(0); cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
    m.load_state_dict(torch.load(CKPT)); m.eval(); return m,cfg
@torch.no_grad()
def logp(m,q,a,st=None):
    ids=TOK.encode(q).ids+TOK.encode(a).ids; n=len(TOK.encode(a).ids); x=torch.tensor([ids],device=DEV)
    lg=m(x[:,:-1],states=st).log_softmax(-1)
    return lg[0,-n:].gather(-1,x[0,1:][-n:].unsqueeze(-1)).mean().item()
@torch.no_grad()
def general(m,cfg,nb=10):
    tot=0.
    for _ in range(nb):
        X=torch.tensor(dm.batch(8),device=DEV); tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
def episode_batch(text,B=8):
    ids=TOK.encode(text).ids
    reps=(T+1)//len(ids)+1; ids=(ids*reps)[:T+1]
    return torch.tensor([ids]*B,device=DEV)
def consolidate(m,cfg,episodes,steps,lr,replay_frac=0.5):
    """LOCAL-plasticity consolidation with REPLAY interleaved (prevents forgetting)."""
    opt=C.LocalAdam(m.param_dict(),lr=lr,adaptive=True)
    for i in range(steps):
        if np.random.rand()<replay_frac: X=torch.tensor(dm.batch(8),device=DEV)     # replay general knowledge
        else: X=episode_batch(episodes[np.random.randint(len(episodes))])           # the episode to consolidate
        opt.step(L.lw_grads(m,X,lam=4.0,k_local=2))
    return m
if __name__=="__main__":
    FACTS=[("The tallest tower in Vashmere is called Kelmoor Spire.","Question: What is the tallest tower in Vashmere called? Answer:"," Kelmoor"),
           ("Doctor Marlow discovered the enzyme luminase in the year 1987 in Oslo.","Question: What did Doctor Marlow discover? Answer:"," luminase")]
    print("CONSOLIDATION (sleep): move an inference-learned fact into the SLOW WEIGHTS, keep everything else",flush=True)
    m,cfg=load(); g0=general(m,cfg)
    base=[logp(m,q,a) for _,q,a in FACTS]
    print(f"  BEFORE  general-loss={g0:.4f}   fact logP (memory cleared) = "+", ".join(f"{b:.3f}" for b in base),flush=True)
    for steps,lr in ((30,3e-3),(100,3e-3),(300,3e-3)):
        m,cfg=load()
        t0=time.time(); consolidate(m,cfg,[f[0] for f in FACTS],steps,lr); m.eval()
        after=[logp(m,q,a) for _,q,a in FACTS]; g1=general(m,cfg)
        gains=[a-b for a,b in zip(after,base)]
        ok = np.mean(gains)>0.3 and (g1-g0)<0.15
        print(f"  steps={steps:4d} lr={lr:.0e}: fact gain = "+", ".join(f"{x:+.3f}" for x in gains)+
              f"  (mean {np.mean(gains):+.3f})   general damage={g1-g0:+.4f}   -> {'COMMIT' if ok else 'ROLLBACK'}   ({time.time()-t0:.0f}s)",flush=True)
