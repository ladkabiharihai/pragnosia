"""VISION ACCEPTANCE TEST 2 -- "continual learning at inference" (VISION 1b flag 2), the project's soul.
Previous attempt failed because the model had never learned to use a memory. Now we train WITH memory carry first.
At inference the model reads a fact document; the state update IS the local Hebbian write (no backprop, no grad).
Then we query and measure whether the fact was learned, whether old ability survives, and what noise does.
2a RETENTION: logP(answer) with the fact in memory vs memory zeroed.
2b PERSISTENCE: does it survive N chunks of unrelated text (with forgetting)?
2c INTERFERENCE: does general language ability degrade?
2d POLLUTION (0f): after unrelated noise, does the fact survive?"""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D=256; NL=6; T=128; NCH=4; B=16; DECAY=0.2; STEPS=6000
dm=DataMix(ctx=T*NCH+1,verbose=False)
def train():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    for it in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV); st=None
        for i in range(NCH):
            xi=X[:,i*T:(i+1)*T]; yi=X[:,i*T+1:(i+1)*T+1]
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,DECAY)
            loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        if (it+1)%2000==0: print(f"    train {it+1} loss={loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.eval(); return m,cfg
@torch.no_grad()
def absorb(m,text,st=None,pad=T):
    """read a document at INFERENCE: the state update IS the local Hebbian write (no backprop)."""
    ids=TOK.encode(text).ids[:pad]; ids=ids+[0]*(pad-len(ids))
    x=torch.tensor([ids],device=DEV); _,st=m(x,states=st,return_states=True)
    return m.decay_states(st,DECAY)
@torch.no_grad()
def logp_answer(m,q,ans,st):
    ids=TOK.encode(q).ids+TOK.encode(ans).ids; n=len(TOK.encode(ans).ids)
    x=torch.tensor([ids],device=DEV)
    lg=m(x[:,:-1],states=st).log_softmax(-1)
    return lg[0,-n:].gather(-1,x[0,1:][-n:].unsqueeze(-1)).mean().item()
if __name__=="__main__":
    print(f"ACCEPTANCE 2: continual learning at INFERENCE (train-with-memory first, {STEPS} steps)",flush=True)
    m,cfg=train()
    FACTS=[("The secret access code is 7492. Remember the access code carefully.",
            "Question: What is the access code? Answer:"," 7492"),
           ("Doctor Marlow discovered the enzyme luminase in the year 1987 in Oslo.",
            "Question: What did Doctor Marlow discover? Answer:"," luminase")]
    print("\n  2a RETENTION (fact absorbed at inference vs no memory):",flush=True)
    for fact,q,a in FACTS:
        st=absorb(m,fact)
        with_mem=logp_answer(m,q,a,st); no_mem=logp_answer(m,q,a,None)
        print(f"    {fact[:40]!r:44s} logP: no-mem={no_mem:7.3f} with-mem={with_mem:7.3f}  LEARNED={with_mem-no_mem:+.3f}",flush=True)
    print("\n  2b PERSISTENCE (fact survives N chunks of unrelated text):",flush=True)
    fact,q,a=FACTS[0]
    for n in (0,1,2,4):
        st=absorb(m,fact)
        for _ in range(n):
            X=torch.tensor(dm.batch(1)[:, :T],device=DEV)
            with torch.no_grad(): _,st=m(X,states=st,return_states=True); st=m.decay_states(st,DECAY)
        print(f"    after {n} unrelated chunks: logP={logp_answer(m,q,a,st):7.3f}",flush=True)
    print("\n  2c INTERFERENCE (general ability with the fact in memory):",flush=True)
    st=absorb(m,FACTS[0][0]); tot_w=tot_n=0.
    with torch.no_grad():
        for _ in range(8):
            X=torch.tensor(dm.batch(1),device=DEV)[:, :T+1]
            tot_w+=F.cross_entropy(m(X[:,:T],states=st).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
            tot_n+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    print(f"    held-out loss: no-mem={tot_n/8:.4f} with-fact-in-mem={tot_w/8:.4f}  damage={tot_w/8-tot_n/8:+.4f}",flush=True)
