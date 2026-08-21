"""PC DE-RISK #1 v3 (DECISIVE): pure LOCAL learning (eligibility traces, NO backprop-through-time, manual VJP so no
autograd anywhere) vs BPTT on MQAR, with the CORRECT architecture (key_{t-1} binds value_t via shift; 2nd-order Taylor
map). Insight: our read is r = sum_t beta_t v_t / sum_t beta_t with beta_t = qf . kf_t. So each write leaves an
ELIGIBILITY TRACE kf_t; when the query arrives, credit is assigned locally via beta_t = qf . kf_t (e-prop). This is
EXACTLY local temporal credit, no time-unrolling. Steps: (a) BPTT baseline (target ~100%); (b) LOCALITY CHECK:
analytic local-rule grad vs true autograd grad, cosine ~1; (c) TRAIN with the pure-local rule, expect ~100% == BPTT.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
NK=8; NV=8; NPAIRS=4; VOCAB=1+NK+NV; DE=32; DIN=8; DV=32
_iu=torch.triu_indices(DIN,DIN,device=DEV); NP=_iu.shape[1]; FE=1+DIN+NP
def make_batch(B):
    seqs=[];tp=[];tg=[]
    for _ in range(B):
        keys=np.random.choice(range(1,NK+1),NPAIRS,replace=False)
        vals=np.random.choice(range(NK+1,NK+NV+1),NPAIRS,replace=True)
        toks=[]
        for k,v in zip(keys,vals): toks+=[k,v]
        qi=np.random.randint(NPAIRS); toks+=[keys[qi]]
        seqs.append(toks); tp.append(len(toks)-1); tg.append(vals[qi])
    return torch.tensor(np.array(seqs),device=DEV),torch.tensor(tp,device=DEV),torch.tensor(tg,device=DEV)
def taylor(x):
    B=x.shape[0]; qd=(x.unsqueeze(2)*x.unsqueeze(1))[:,_iu[0],_iu[1]]
    return torch.cat([torch.ones(B,1,device=x.device),x,qd],1)
def taylor_vjp(x,g):                                          # manual VJP: x[N,DIN], g[N,FE] -> dx[N,DIN] (no autograd)
    dx=g[:,1:1+DIN].clone(); gq=g[:,1+DIN:]
    dx.index_add_(1,_iu[0],gq*x[:,_iu[1]]); dx.index_add_(1,_iu[1],gq*x[:,_iu[0]])
    return dx
class Core(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(VOCAB,DE)
        s.Wk=nn.Linear(DE,DIN,bias=False); s.Wq=nn.Linear(DE,DIN,bias=False); s.Wv=nn.Linear(DE,DV,bias=False); s.Wo=nn.Linear(DV,VOCAB)
    def forward(s,X):                                         # query-only readout, shifted key->value binding
        B,T=X.shape; h=s.emb(X); S=torch.zeros(B,FE,DV,device=DEV); z=torch.zeros(B,FE,device=DEV); outs=[]
        for t in range(T):
            src=h[:,t-1] if t>0 else h[:,t]                   # key from PREVIOUS token (binds to value_t)
            kf=taylor(s.Wk(src)); v=s.Wv(h[:,t]); S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
            qf=taylor(s.Wq(h[:,t])); num=torch.einsum('bf,bfd->bd',qf,S); den=(qf*z).sum(-1,keepdim=True)+1e-5
            outs.append(s.Wo(num/den))
        return torch.stack(outs,1)
def acc(m,n=2000):
    with torch.no_grad():
        X,tp,tg=make_batch(n); return (m(X)[torch.arange(n),tp].argmax(-1)==tg).float().mean().item()

def local_grads(m,X,tp,tg):
    """Pure-local eligibility-trace gradient of the QUERY loss. No backprop-through-time, manual VJP."""
    B,T=X.shape; ar=torch.arange(B,device=DEV); h=m.emb(X).detach()
    Wk,Wq,Wv=m.Wk.weight.detach(),m.Wq.weight.detach(),m.Wv.weight.detach()
    Wo,bo=m.Wo.weight.detach(),m.Wo.bias.detach()
    # forward, caching per-write eligibility trace kf_t and (src_t, v_t, hval_t)
    xin=[]; kf=[]; vv=[]; hval=[]; S=torch.zeros(B,FE,DV,device=DEV); z=torch.zeros(B,FE,device=DEV)
    for t in range(T):
        src=h[:,t-1] if t>0 else h[:,t]; xw=src@Wk.T; kft=taylor(xw); vt=h[:,t]@Wv.T
        S=S+kft.unsqueeze(2)*vt.unsqueeze(1); z=z+kft
        xin.append((src,xw)); kf.append(kft); vv.append(vt); hval.append(h[:,t])
    qh=h[ar,tp]; xq=qh@Wq.T; qf=taylor(xq)                    # only the query position reads/learns
    Sg=S; zg=z; num=torch.einsum('bf,bfd->bd',qf,Sg); den=(qf*zg).sum(-1,keepdim=True)+1e-5; r=num/den
    logits=r@Wo.T+bo; p=logits.softmax(-1); e=p.clone(); e[ar,tg]-=1.0
    dWo=e.T@r/B; dbo=e.mean(0); er=e@Wo                        # er=dL/dr [B,DV]
    e_num=er/den; g_den=(-(er*num).sum(-1,keepdim=True)/den**2)  # [B,DV],[B,1]
    dWv=torch.zeros_like(Wv); dWk=torch.zeros_like(Wk); dqf=torch.zeros(B,FE,device=DEV)
    for t in range(T):                                        # credit each cached write via its eligibility trace
        beta=(qf*kf[t]).sum(-1,keepdim=True)                  # retrieval weight [B,1] = qf . trace
        dv=beta*e_num                                         # dL/dv_t  (local: uses cached trace overlap)
        dWv=dWv+dv.T@hval[t]/B
        dbeta=(er*vv[t]).sum(-1,keepdim=True)/den+g_den       # [B,1]
        dkf=dbeta*qf                                          # dL/dkf_t [B,FE]
        dqf=dqf+dbeta*kf[t]                                   # accumulate query credit
        src,xw=xin[t]; dxw=taylor_vjp(xw,dkf); dWk=dWk+dxw.T@src/B
    dxq=taylor_vjp(xq,dqf); dWq=dxq.T@qh/B
    return {'Wo':dWo,'bo':dbo,'Wv':dWv,'Wk':dWk,'Wq':dWq}

def train_bptt(steps=3000,lr=3e-3,B=128):
    m=Core().to(DEV); opt=torch.optim.Adam(m.parameters(),lr=lr)
    for _ in range(steps):
        X,tp,tg=make_batch(B); lg=m(X); loss=F.cross_entropy(lg[torch.arange(B),tp],tg)
        opt.zero_grad(); loss.backward(); opt.step()
    return acc(m)
def train_local(steps=3000,lr=3e-3,B=128):
    m=Core().to(DEV); st={n:torch.zeros_like(p) for n,p in [('Wo',m.Wo.weight),('bo',m.Wo.bias),('Wv',m.Wv.weight),('Wk',m.Wk.weight),('Wq',m.Wq.weight)]}
    b1,b2,eps=0.9,0.999,1e-8; v={k:torch.zeros_like(x) for k,x in st.items()}; i=0
    prm={'Wo':m.Wo.weight,'bo':m.Wo.bias,'Wv':m.Wv.weight,'Wk':m.Wk.weight,'Wq':m.Wq.weight}
    for _ in range(steps):
        i+=1; X,tp,tg=make_batch(B); g=local_grads(m,X,tp,tg)
        with torch.no_grad():                                 # Adam on the LOCAL grads (optimizer = substrate math, not backprop)
            for k in prm:
                st[k]=b1*st[k]+(1-b1)*g[k]; v[k]=b2*v[k]+(1-b2)*g[k]**2
                mh=st[k]/(1-b1**i); vh=v[k]/(1-b2**i); prm[k]-=lr*mh/(vh.sqrt()+eps)
    return acc(m)

if __name__=="__main__":
    print(f"device={DEV}  Taylor DIN={DIN}->FE={FE}  MQAR {NPAIRS}pairs vocab={VOCAB} chance {100/NV:.0f}%  (shifted key->value)",flush=True)
    # (b) LOCALITY CHECK first (cheap, decisive): does the analytic local rule == true gradient?
    m=Core().to(DEV); X,tp,tg=make_batch(256)
    lg=m(X); loss=F.cross_entropy(lg[torch.arange(256),tp],tg); m.zero_grad(); loss.backward()
    g=local_grads(m,X,tp,tg)
    print("LOCALITY CHECK (cosine of local-rule grad vs true autograd grad; ~1.0 => local rule IS the gradient):",flush=True)
    for n,p in [('Wo',m.Wo.weight),('bo',m.Wo.bias),('Wv',m.Wv.weight),('Wk',m.Wk.weight),('Wq',m.Wq.weight)]:
        cos=F.cosine_similarity(g[n].flatten(),p.grad.flatten(),dim=0).item(); print(f"    {n}: {cos:.4f}",flush=True)
    # (a) baseline + (c) local training
    bp=train_bptt(); print(f"BPTT baseline: query-acc={bp*100:.1f}%",flush=True)
    lo=train_local(); print(f"LOCAL (eligibility traces, NO backprop-through-time): query-acc={lo*100:.1f}%",flush=True)
    gap=bp-lo
    print(f"\nGAP BPTT->LOCAL={gap*100:.1f} pts. "+(
      "PURE LOCAL == BPTT on recall -> §0c LOAD-BEARING BET HOLDS for recurrent associative recall" if gap<0.1 else
      "close, refine" if gap<0.3 else "local lags, investigate"),flush=True)
