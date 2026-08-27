"""VISION ACCEPTANCE TEST 2 -- "continual learning at inference" (VISION 1b flag 2). v2: fixes a test bug (v1 padded
the fact to 128 tokens with token 0, so the memory captured PAD, not the fact) and adds proper controls.
The model is trained WITH memory carry first; at inference the state update IS the local Hebbian write (no backprop).
2a RETENTION: logP(answer) with fact in memory vs no memory, and vs a NEUTRAL text of equal length (the control that
   separates 'learned the fact' from 'any state helps/hurts').
2b PERSISTENCE across unrelated chunks. 2c INTERFERENCE vs the neutral-state control. 2d POLLUTION."""
import sys,time,os; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D=256; NL=6; T=128; NCH=4; B=16; DECAY=0.2; STEPS=6000
CKPT="/mnt/kv_cache/pragnosia_data/acc2_model.pt"
dm=DataMix(ctx=T*NCH+1,verbose=False)
def build():
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); return cfg,C.Core(cfg).to(DEV)
def train():
    cfg,m=build()
    if os.path.exists(CKPT):
        m.load_state_dict(torch.load(CKPT)); m.eval(); print("  (loaded cached substrate)",flush=True); return m,cfg
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    for it in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV); st=None
        for i in range(NCH):
            xi=X[:,i*T:(i+1)*T]; yi=X[:,i*T+1:(i+1)*T+1]
            lg,st=m(xi,states=st,return_states=True); st=m.decay_states(st,DECAY)
            loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        if (it+1)%3000==0: print(f"    train {it+1} loss={loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
    m.eval(); torch.save(m.state_dict(),CKPT); return m,cfg
@torch.no_grad()
def absorb(m,text,st=None):
    """read a document at INFERENCE -- NO padding. The state update IS the local Hebbian write."""
    ids=TOK.encode(text).ids
    _,st=m(torch.tensor([ids],device=DEV),states=st,return_states=True)
    return m.decay_states(st,DECAY)
@torch.no_grad()
def logp(m,q,a,st):
    ids=TOK.encode(q).ids+TOK.encode(a).ids; n=len(TOK.encode(a).ids); x=torch.tensor([ids],device=DEV)
    lg=m(x[:,:-1],states=st).log_softmax(-1)
    return lg[0,-n:].gather(-1,x[0,1:][-n:].unsqueeze(-1)).mean().item()
NEUTRAL="The weather report described mild conditions across the region during the afternoon hours."
if __name__=="__main__":
    print(f"ACCEPTANCE 2 v2: continual learning at INFERENCE (fixed: no pad-token contamination)",flush=True)
    m,cfg=train()
    FACTS=[("The secret access code is 7492. Remember the access code carefully.","Question: What is the access code? Answer:"," 7492"),
           ("Doctor Marlow discovered the enzyme luminase in the year 1987 in Oslo.","Question: What did Doctor Marlow discover? Answer:"," luminase"),
           ("The tallest tower in Vashmere is called Kelmoor Spire.","Question: What is the tallest tower in Vashmere called? Answer:"," Kelmoor")]
    print("\n  2a RETENTION  (control = a NEUTRAL text of similar length in memory)",flush=True)
    gains=[]
    for fact,q,a in FACTS:
        n=logp(m,q,a,None); f=logp(m,q,a,absorb(m,fact)); c=logp(m,q,a,absorb(m,NEUTRAL))
        gains.append(f-c)
        print(f"    {fact[:34]!r:38s} no-mem={n:7.3f} neutral-mem={c:7.3f} FACT-mem={f:7.3f}   vs-control={f-c:+.3f}",flush=True)
    print(f"    MEAN fact-vs-control gain = {np.mean(gains):+.3f}  (positive = the fact was genuinely learned)",flush=True)
    print("\n  2b PERSISTENCE (fact vs control, after N unrelated chunks)",flush=True)
    fact,q,a=FACTS[0]
    for n_ in (0,1,2,4):
        stf=absorb(m,fact); stc=absorb(m,NEUTRAL)
        for _ in range(n_):
            X=torch.tensor(dm.batch(1)[:, :T],device=DEV)
            with torch.no_grad():
                _,stf=m(X,states=stf,return_states=True); stf=m.decay_states(stf,DECAY)
                _,stc=m(X,states=stc,return_states=True); stc=m.decay_states(stc,DECAY)
        print(f"    after {n_} chunks: fact={logp(m,q,a,stf):7.3f} control={logp(m,q,a,stc):7.3f}  gain={logp(m,q,a,stf)-logp(m,q,a,stc):+.3f}",flush=True)
    print("\n  2c INTERFERENCE (held-out loss: fact-state vs neutral-state vs no-state)",flush=True)
    stf=absorb(m,fact); stc=absorb(m,NEUTRAL); tw=tc=tn=0.
    with torch.no_grad():
        for _ in range(8):
            X=torch.tensor(dm.batch(1),device=DEV)[:, :T+1]; Y=X[:,1:T+1].reshape(-1)
            tw+=F.cross_entropy(m(X[:,:T],states=stf).reshape(-1,cfg.vocab),Y).item()
            tc+=F.cross_entropy(m(X[:,:T],states=stc).reshape(-1,cfg.vocab),Y).item()
            tn+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),Y).item()
    print(f"    no-state={tn/8:.4f}  neutral-state={tc/8:.4f}  fact-state={tw/8:.4f}   fact-vs-neutral={tw/8-tc/8:+.4f}",flush=True)
