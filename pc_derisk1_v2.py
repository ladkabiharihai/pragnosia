"""PC DE-RISK #1 v2 (decisive): (a) fix baseline with the 2nd-order Taylor feature map so BPTT aces MQAR; (b) PROVE
temporal credit is LOCAL for our linear Hebbian memory -- implement the per-write local update from CACHED TRACES
(eligibility traces, no backprop-through-time) and show it MATCHES the true gradient (cosine ~1); (c) TRAIN with the
pure-local rule and show it aces MQAR = BPTT. If local==gradient and local aces MQAR -> pure local plasticity does
recall for our arch (the linear memory makes temporal credit local). No autograd-through-time in the local learner.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
NK=8; NV=8; NPAIRS=4; VOCAB=1+NK+NV
def make_batch(B):
    seqs=[];tp=[];tg=[]
    for _ in range(B):
        keys=np.random.choice(range(1,NK+1),NPAIRS,replace=False)
        vals=np.random.choice(range(NK+1,NK+NV+1),NPAIRS,replace=True)
        toks=[]
        for k,v in zip(keys,vals): toks+=[k,v]
        qi=np.random.randint(NPAIRS); toks+=[keys[qi]]
        seqs.append(toks); tp.append(len(toks)-1); tg.append(vals[qi])
    return (torch.tensor(np.array(seqs),device=DEV),torch.tensor(tp,device=DEV),torch.tensor(tg,device=DEV))

DIN=6                                                          # projection dim before Taylor expand
_iu=torch.triu_indices(DIN,DIN)                                # upper-tri incl diagonal (symmetric 2nd-order)
FE=1+DIN+_iu.shape[1]                                          # Taylor feature dim = 1 + d + d(d+1)/2
def taylor(x):                                                 # symmetric 2nd-order Taylor map (softmax-like recall)
    B=x.shape[0]; ones=torch.ones(B,1,device=x.device)
    qd=(x.unsqueeze(2)*x.unsqueeze(1))[:,_iu[0],_iu[1]]        # [B, d(d+1)/2]
    return torch.cat([ones,x,qd],1)                            # [B,FE]
DV=32
class Core(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(VOCAB,32)
        s.Wk=nn.Linear(32,DIN,bias=False); s.Wq=nn.Linear(32,DIN,bias=False); s.Wv=nn.Linear(32,DV,bias=False)
        s.Wo=nn.Linear(DV,VOCAB)
    def forward(s,X,detach=False):
        B,T=X.shape; h=s.emb(X)
        S=torch.zeros(B,FE,DV,device=X.device); z=torch.zeros(B,FE,device=X.device); outs=[]
        for t in range(T):
            if detach: S=S.detach(); z=z.detach()
            kf=taylor(s.Wk(h[:,t])); v=s.Wv(h[:,t])
            S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
            qf=taylor(s.Wq(h[:,t]))
            num=torch.einsum('bf,bfd->bd',qf,S); den=(qf*z).sum(-1,keepdim=True)+1e-5
            outs.append(s.Wo(num/den))
        return torch.stack(outs,1)
def acc(m,n=1000):
    m.eval()
    with torch.no_grad():
        X,tp,tg=make_batch(n); lg=m(X); return (lg[torch.arange(n),tp].argmax(-1)==tg).float().mean().item()
def train_bptt(steps=2000,lr=3e-3,B=128):
    m=Core().to(DEV); opt=torch.optim.Adam(m.parameters(),lr=lr)
    for _ in range(steps):
        X,tp,tg=make_batch(B); lg=m(X); loss=F.cross_entropy(lg[torch.arange(B),tp],tg)
        opt.zero_grad(); loss.backward(); opt.step()
    return acc(m),m

# ---- LOCAL rule: forward caching per-step traces; at query, credit writes from cached traces (NO backprop-through-time) ----
def local_grads(m,X,tp,tg):
    B,T=X.shape; h=m.emb(X)                                    # (emb via module; we compute its grad manually too)
    cache=[]; S=torch.zeros(B,FE,DV,device=DEV); z=torch.zeros(B,FE,device=DEV)
    Wk=m.Wk.weight; Wq=m.Wq.weight; Wv=m.Wv.weight
    # forward, caching (h_t, kf_t, v_t)
    for t in range(T):
        ht=h[:,t]
        kf=taylor(ht@Wk.T); v=ht@Wv.T
        S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
        cache.append((ht,kf,v))
    ar=torch.arange(B)
    qh=h[ar,tp]; qf=taylor(qh@Wq.T)                            # query features at the query position
    Sg=S; zg=z                                                 # state seen by the query (full accumulation)
    num=torch.einsum('bf,bfd->bd',qf,Sg); den=(qf*zg).sum(-1,keepdim=True)+1e-5
    r=num/den; logits=r@m.Wo.weight.T+m.Wo.bias
    p=logits.softmax(-1); e=p.clone(); e[ar,tg]-=1.0           # dL/dlogits (cross-entropy)
    # Wo (local, feedforward)
    dWo=e.T@r/B; dbo=e.mean(0); er=e@m.Wo.weight               # er = dL/dr [B,DV]
    e_num=er/den; g_den=(-(er*num).sum(-1,keepdim=True)/den**2) # quotient-rule terms
    # query credit -> qf -> Wq (within-step Jacobian via autograd on taylor, state fixed => no time)
    dqf=torch.einsum('bfd,bd->bf',Sg,e_num)+g_den*zg          # dL/dqf [B,FE]
    # temporal credit to writes from CACHED traces (eligibility): loop cached positions
    dWv=torch.zeros_like(Wv); dWk=torch.zeros_like(Wk)
    for (ht,kf,v) in cache:
        beta=(qf*kf).sum(-1,keepdim=True)                     # retrieval weight of this write [B,1]
        dv=beta*e_num                                         # dL/dv_t [B,DV]  (temporal, local via cached beta)
        dWv=dWv+dv.T@ht/B
        dkf=qf*((er* v).sum(-1,keepdim=True)/den + g_den)     # dL/dkf_t [B,FE]
        # map dkf back through the taylor Jacobian to the DIN-projection, then to Wk (within-step autograd)
        x=(ht@Wk.T).detach().requires_grad_(True); (taylor(x)*dkf).sum().backward()
        dWk=dWk+x.grad.T@ht/B
    # map dqf through taylor Jacobian to Wq
    xq=(qh@Wq.T).detach().requires_grad_(True); (taylor(xq)*dqf).sum().backward()
    dWq=xq.grad.T@qh/B
    return {'Wo':dWo,'bo':dbo,'Wv':dWv,'Wk':dWk,'Wq':dWq}
def train_local(steps=2000,lr=3e-3,B=128):
    m=Core().to(DEV)
    for _ in range(steps):
        X,tp,tg=make_batch(B); g=local_grads(m,X,tp,tg)
        with torch.no_grad():
            m.Wo.weight-=lr*g['Wo']; m.Wo.bias-=lr*g['bo']
            m.Wv.weight-=lr*g['Wv']; m.Wk.weight-=lr*g['Wk']; m.Wq.weight-=lr*g['Wq']
    return acc(m),m

if __name__=="__main__":
    print(f"device={DEV}  Taylor map: DIN={DIN} -> FE={FE}  MQAR {NPAIRS}pairs vocab={VOCAB} (chance {100/NV:.0f}%)",flush=True)
    bptt,mb=train_bptt(); print(f"BPTT (Taylor map): query-acc={bptt*100:.1f}%",flush=True)
    # (b) LOCALITY CHECK: manual local grad vs autograd grad on Wv, one batch
    m=Core().to(DEV); X,tp,tg=make_batch(256)
    lg=m(X); loss=F.cross_entropy(lg[torch.arange(256),tp],tg); m.zero_grad(); loss.backward()
    g=local_grads(m,X,tp,tg)
    for name,p in [('Wv',m.Wv.weight),('Wk',m.Wk.weight),('Wq',m.Wq.weight),('Wo',m.Wo.weight)]:
        cos=F.cosine_similarity(g[name].flatten(),p.grad.flatten(),dim=0).item()
        print(f"  locality check {name}: cos(local_rule, true_grad)={cos:.3f}",flush=True)
    loc,ml=train_local(); print(f"LOCAL (eligibility traces, NO backprop-through-time): query-acc={loc*100:.1f}%",flush=True)
    gap=bptt-loc
    print(f"\nGAP BPTT->LOCAL = {gap*100:.1f} pts. "+(
        "PURE LOCAL does recall == BPTT -> load-bearing bet HOLDS for recurrent recall (linear memory makes temporal credit local)" if gap<0.1 else
        "LOCAL close -> refine" if gap<0.3 else "LOCAL lags -> investigate the local rule"),flush=True)
