"""PC DE-RISK #2: pure LOCAL plasticity on REAL LANGUAGE (window2_32k), causal next-token LM, vs BPTT.
Generalizes the eligibility rule to next-token at EVERY position: read r_t = sum_{s<=t} beta_{t,s} v_s / sum beta_{t,s}
(beta_{t,s}=qf_t.kf_s), loss summed over t. Each write s gets credit from ALL future reads t>=s via a
REVERSE-CUMULATIVE learning signal (e-prop) -- still LOCAL (no differentiation through the recurrence across time),
just error-signal traces accumulated backward over cached per-position quantities. Manual VJP, no autograd in the
local path. Decisive test = LOCALITY CHECK: local-rule grad vs true autograd grad, cosine ~1 on all params (incl emb).
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
from tokenizers import Tokenizer
VOCAB=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json").get_vocab_size()
DATA=np.memmap("/mnt/kv_cache/pragnosia_data/window2_32k.bin",dtype=np.uint16,mode="r")
DE=64; DIN=8; DV=64; T=64
_iu=torch.triu_indices(DIN,DIN,device=DEV); FE=1+DIN+_iu.shape[1]
def batch(B):
    ix=np.random.randint(0,len(DATA)-T-1,B); X=np.stack([DATA[i:i+T+1] for i in ix]).astype(np.int64)
    return torch.tensor(X,device=DEV)                          # [B,T+1]; predict X[:,1:] from X[:,:T]
def taylor(x):
    B=x.shape[0]; qd=(x.unsqueeze(2)*x.unsqueeze(1))[:,_iu[0],_iu[1]]
    return torch.cat([torch.ones(B,1,device=x.device),x,qd],1)
def taylor_vjp(x,g):
    dx=g[:,1:1+DIN].clone(); gq=g[:,1+DIN:]
    dx.index_add_(1,_iu[0],gq*x[:,_iu[1]]); dx.index_add_(1,_iu[1],gq*x[:,_iu[0]]); return dx
class Core(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(VOCAB,DE)
        s.Wk=nn.Linear(DE,DIN,bias=False); s.Wq=nn.Linear(DE,DIN,bias=False); s.Wv=nn.Linear(DE,DV,bias=False); s.Wo=nn.Linear(DV,VOCAB)
        nn.init.normal_(s.emb.weight,0,0.02)
    def forward(s,Xin):                                        # Xin [B,T]; returns logits [B,T,VOCAB]
        B,Tt=Xin.shape; h=s.emb(Xin); S=torch.zeros(B,FE,DV,device=DEV); z=torch.zeros(B,FE,device=DEV); outs=[]
        for t in range(Tt):
            src=h[:,t-1] if t>0 else torch.zeros_like(h[:,0]); kf=taylor(s.Wk(src)); v=s.Wv(h[:,t])
            S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
            qf=taylor(s.Wq(h[:,t])); num=torch.einsum('bf,bfd->bd',qf,S); den=(qf*z).sum(-1,keepdim=True)+1e-5
            outs.append(s.Wo(num/den))
        return torch.stack(outs,1)

def local_grads(m,X):
    Xin=X[:,:T]; Y=X[:,1:T+1]; B=Xin.shape[0]; ar=torch.arange(B,device=DEV)
    W={k:getattr(m,k).weight.detach() for k in ('Wk','Wq','Wv','Wo')}; bo=m.Wo.bias.detach(); E=m.emb.weight.detach()
    h=E[Xin]                                                   # [B,T,DE]
    S=torch.zeros(B,FE,DV,device=DEV); z=torch.zeros(B,FE,device=DEV)
    src_c=[];xk_c=[];kf_c=[];v_c=[];xq_c=[];qf_c=[];num_c=[];den_c=[];Ssnap=[]
    for t in range(T):
        src=h[:,t-1] if t>0 else torch.zeros_like(h[:,0]); xk=src@W['Wk'].T; kf=taylor(xk); v=h[:,t]@W['Wv'].T
        S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
        xq=h[:,t]@W['Wq'].T; qf=taylor(xq); num=torch.einsum('bf,bfd->bd',qf,S); den=(qf*z).sum(-1,keepdim=True)+1e-5
        src_c.append(src);xk_c.append(xk);kf_c.append(kf);v_c.append(v);xq_c.append(xq);qf_c.append(qf)
        num_c.append(num);den_c.append(den);Ssnap.append(S.clone())          # S_t (state incl write t) for spatial qf grad
    scale=1.0/(B*T)
    # per-position readout signals
    er=[];e_num=[];g_den=[];dWo=torch.zeros_like(W['Wo']);dbo=torch.zeros_like(bo)
    dE=torch.zeros_like(E)
    for t in range(T):
        r=num_c[t]/den_c[t]; logits=r@W['Wo'].T+bo; p=logits.softmax(-1); e=p; e[ar,Y[:,t]]-=1.0; e=e*scale
        dWo+=e.T@r; dbo+=e.sum(0); ert=e@W['Wo']
        er.append(ert); e_num.append(ert/den_c[t]); g_den.append(-(ert*num_c[t]).sum(-1,keepdim=True)/den_c[t]**2)
    # REVERSE-CUMULATIVE learning signals P (FE x DV) and Q (FE) for write credit
    P=torch.zeros(B,FE,DV,device=DEV); Q=torch.zeros(B,FE,device=DEV)
    dWv=torch.zeros_like(W['Wv']); dWk=torch.zeros_like(W['Wk']); dWq=torch.zeros_like(W['Wq'])
    # first, recompute z_t running for the spatial qf denominator term
    zt=[torch.zeros(B,FE,device=DEV) for _ in range(T)]; zr=torch.zeros(B,FE,device=DEV)
    for t in range(T): zr=zr+kf_c[t]; zt[t]=zr.clone()
    for t in reversed(range(T)):
        # accumulate this read's contribution into reverse sums (credit flows to writes s<=t)
        P=P+torch.einsum('bf,bd->bfd',qf_c[t],e_num[t]); Q=Q+g_den[t]*qf_c[t]
        # write s=t gets credit from all reads >= t (which is exactly current P,Q)
        s=t
        dv=torch.einsum('bf,bfd->bd',kf_c[s],P)                # dL/dv_s = kf_s . P_s  [B,DV]
        dWv+=dv.T@h[:,s]                                       # v_s = Wv h_s
        dkf=torch.einsum('bfd,bd->bf',P,v_c[s])+Q             # dL/dkf_s = P_s . v_s + Q_s  [B,FE]
        dxk=taylor_vjp(xk_c[s],dkf); dWk+=dxk.T@src_c[s]      # kf_s = taylor(Wk src_s)
        # emb credit from key path (src_s = h_{s-1})
        if s>0: dE.index_add_(0,Xin[:,s-1],dxk@W['Wk'])
        # emb credit from value path
        dE.index_add_(0,Xin[:,s],dv@W['Wv'])
        # spatial query grad at position t (uses S_t and z_t)
        dqf=torch.einsum('bfd,bd->bf',Ssnap[t],e_num[t])+g_den[t]*zt[t]
        dxq=taylor_vjp(xq_c[t],dqf); dWq+=dxq.T@h[:,t]
        dE.index_add_(0,Xin[:,t],dxq@W['Wq'])
    return {'emb':dE,'Wk':dWk,'Wq':dWq,'Wv':dWv,'Wo':dWo,'bo':dbo}

if __name__=="__main__":
    print(f"device={DEV} VOCAB={VOCAB} T={T} DE={DE} DIN={DIN}->FE={FE} DV={DV}  data=window2_32k (real language)",flush=True)
    m=Core().to(DEV); X=batch(32)
    lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,VOCAB),X[:,1:T+1].reshape(-1)); m.zero_grad(); loss.backward()
    g=local_grads(m,X)
    print("LOCALITY CHECK (cosine local-rule vs true grad; ~1 => pure-local == backprop for REAL causal LM):",flush=True)
    pairs=[('emb',m.emb.weight),('Wk',m.Wk.weight),('Wq',m.Wq.weight),('Wv',m.Wv.weight),('Wo',m.Wo.weight),('bo',m.Wo.bias)]
    ok=True
    for n,p in pairs:
        cos=F.cosine_similarity(g[n].flatten(),p.grad.flatten(),dim=0).item(); ok&=cos>0.98
        print(f"    {n:4s}: {cos:.4f}",flush=True)
    print(("PASS -> pure local plasticity == backprop on REAL language LM (no autograd, no BPTT)" if ok else
           "MISMATCH -> derivation bug in the reverse-cumulative learning signal, investigate"),flush=True)
