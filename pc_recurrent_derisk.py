"""PC DE-RISK #1 (the REAL one): can PURE LOCAL learning (NO backprop-through-time) train our recurrent linear-attention
core to do ASSOCIATIVE RECALL (MQAR)? The crux: the memory-WRITE projections (W_k,W_v) affect a FUTURE read at the
query; can they get credit without BPTT? Baseline = full BPTT (autograd through the whole sequence). LOCAL learner =
Hebbian state DETACHED across time (no through-time gradient) + a LOCAL objective at EVERY position (self-supervised),
so writes are trained locally, not by the future read. If LOCAL matches BPTT -> pure local plasticity does recall (our
Hebbian memory makes it work); if LOCAL lags -> temporal credit is needed (informative, guides the next step).
NOTE: within-step gradient uses autograd as a stand-in for the PC feedforward rule already de-risked at 97% (#0); the
question here is purely the TEMPORAL / no-BPTT one.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"

# --- MQAR task: N key-value pairs, then a query key -> predict its value ---
NK=8; NV=8; NPAIRS=4                                     # keys 1..NK, values NK+1..NK+NV, 0=pad/query-marker
VOCAB=1+NK+NV
def make_batch(B, npairs=NPAIRS):
    seqs=[]; tgtpos=[]; tgts=[]
    for _ in range(B):
        keys=np.random.choice(range(1,NK+1), npairs, replace=False)
        vals=np.random.choice(range(NK+1,NK+NV+1), npairs, replace=True)
        toks=[]
        for k,v in zip(keys,vals): toks+=[k,v]
        qi=np.random.randint(npairs); toks+=[keys[qi]]      # query = one of the seen keys (last token)
        seqs.append(toks); tgtpos.append(len(toks)-1); tgts.append(vals[qi])
    X=torch.tensor(np.array(seqs),dtype=torch.long,device=DEV)
    return X, torch.tensor(tgtpos,device=DEV), torch.tensor(tgts,device=DEV)

def phi(x): return F.elu(x)+1.0                          # positive feature map (linear-attention)

class LinAttn(nn.Module):
    def __init__(s,d=64,fe=32,dv=64):
        super().__init__(); s.emb=nn.Embedding(VOCAB,d)
        s.Wk=nn.Linear(d,fe,bias=False); s.Wq=nn.Linear(d,fe,bias=False); s.Wv=nn.Linear(d,dv,bias=False)
        s.Wo=nn.Linear(dv,VOCAB); s.fe=fe; s.dv=dv
    def forward(s, X, detach_state=False):
        B,T=X.shape; h=s.emb(X)                                   # [B,T,d]
        k=phi(s.Wk(h)); q=phi(s.Wq(h)); v=s.Wv(h)                  # [B,T,fe],[B,T,fe],[B,T,dv]
        S=torch.zeros(B,s.fe,s.dv,device=X.device); z=torch.zeros(B,s.fe,device=X.device)
        outs=[]
        for t in range(T):
            if detach_state: S=S.detach(); z=z.detach()            # NO backprop-through-time
            S=S+k[:,t].unsqueeze(2)*v[:,t].unsqueeze(1)            # Hebbian write S += k (x) v
            z=z+k[:,t]
            num=torch.einsum('bf,bfd->bd', q[:,t], S)              # read = q . S
            den=(q[:,t]*z).sum(-1,keepdim=True)+1e-5
            outs.append(s.Wo(num/den))
        return torch.stack(outs,1)                                 # [B,T,VOCAB]

def query_acc(m, n=1000):
    m.eval()
    with torch.no_grad():
        X,tp,tg=make_batch(n); lg=m(X)
        pred=lg[torch.arange(n),tp].argmax(-1)
        return (pred==tg).float().mean().item()

def train(detach_state, steps=3000, lr=3e-3, B=128, allpos=False):
    m=LinAttn().to(DEV); opt=torch.optim.Adam(m.parameters(),lr=lr)
    for _ in range(steps):
        X,tp,tg=make_batch(B); lg=m(X,detach_state=detach_state)
        if allpos:                                                 # LOCAL learner: next-token loss at EVERY position (self-supervised)
            loss=F.cross_entropy(lg[:,:-1].reshape(-1,VOCAB), X[:,1:].reshape(-1))
        else:                                                      # BPTT baseline: supervise the query answer
            loss=F.cross_entropy(lg[torch.arange(B),tp], tg)
        opt.zero_grad(); loss.backward(); opt.step()
    return query_acc(m)

if __name__=="__main__":
    print(f"device={DEV}  MQAR: {NPAIRS} pairs, {NK} keys/{NV} vals, vocab={VOCAB}  (chance={100/NV:.0f}%)",flush=True)
    bptt=train(detach_state=False, allpos=False)
    print(f"BPTT baseline (full backprop-through-time): query-acc={bptt*100:.1f}%",flush=True)
    loc=train(detach_state=True, allpos=True)
    print(f"LOCAL (state DETACHED, no BPTT, per-position loss): query-acc={loc*100:.1f}%",flush=True)
    gap=bptt-loc
    print(f"\nGAP BPTT->LOCAL = {gap*100:.1f} pts. "+(
        "LOCAL does recall WITHOUT temporal credit -> pure local plasticity works for our arch (BIG de-risk win)" if gap<0.1 else
        "LOCAL PARTIAL -> some temporal credit helps; tune / add local temporal signal" if gap<0.4 else
        "LOCAL FAILS -> write-projections need future-read credit; need a temporal-local rule"),flush=True)
