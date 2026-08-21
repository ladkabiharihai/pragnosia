"""PC DE-RISK #4: MULTI-LAYER pure-local learning = Predictive Coding (cross-layer/DEPTH credit) + eligibility traces
(within-layer/TIME credit), TOGETHER, vs BPTT, on real 290B mix. #0 proved PC-depth (feedforward), #2/#3 proved
eligibility-time (1 recurrent layer). This tests the COMBINATION on a 2-layer recurrent core -- the last foundational
validation before a real deep-narrow local-training run. PC is APPROXIMATE (settling), so expect a small gap vs BPTT,
not cosine=1. Success = local-PC final loss tracks BPTT (within PC's approximation) on real language. No global backward:
each layer updates from its LOCAL prediction error; cross-layer errors resolved by SETTLING (forward-weight error prop).
"""
import warnings,os,time; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
import sys; sys.path.insert(0,"/opt/code/membrae")
from data_mix import DataMix
from tokenizers import Tokenizer
VOCAB=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json").get_vocab_size()
D=64; DIN=8; DV=64; T=32; NL=2
_iu=torch.triu_indices(DIN,DIN,device=DEV); FE=1+DIN+_iu.shape[1]
dm=DataMix(ctx=T, verbose=True)
def taylor(x):
    B=x.shape[0]; qd=(x.unsqueeze(2)*x.unsqueeze(1))[:,_iu[0],_iu[1]]
    return torch.cat([torch.ones(B,1,device=x.device),x,qd],1)
def tvjp(x,g):
    dx=g[:,1:1+DIN].clone(); gq=g[:,1+DIN:]
    dx.index_add_(1,_iu[0],gq*x[:,_iu[1]]); dx.index_add_(1,_iu[1],gq*x[:,_iu[0]]); return dx

class Layer(nn.Module):                                            # recurrent linear-attn + residual: a_out = a_in + Wo r
    def __init__(s):
        super().__init__(); s.Wk=nn.Linear(D,DIN,bias=False); s.Wq=nn.Linear(D,DIN,bias=False)
        s.Wv=nn.Linear(D,DV,bias=False); s.Wo=nn.Linear(DV,D)
    def forward(s,a):                                              # a:[B,T,D] -> out:[B,T,D] (+cache for local vjp)
        B,Tt,_=a.shape; S=torch.zeros(B,FE,DV,device=DEV); z=torch.zeros(B,FE,device=DEV); outs=[]; cache=[]
        for t in range(Tt):
            src=a[:,t-1] if t>0 else torch.zeros_like(a[:,0]); xk=s.Wk(src); kf=taylor(xk); v=s.Wv(a[:,t])
            S=S+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf
            xq=s.Wq(a[:,t]); qf=taylor(xq); num=torch.einsum('bf,bfd->bd',qf,S); den=(qf*z).sum(-1,keepdim=True)+1e-5
            o=s.Wo(num/den); outs.append(a[:,t]+o); cache.append((src,xk,kf,v,xq,qf,num,den,S.clone()))
        return torch.stack(outs,1), cache
    def local_vjp(s, a, cache, dout, scale):                      # dout=dL/da_out [B,T,D] -> (param grads, dL/da_in)
        B,Tt,_=a.shape; W={'Wk':s.Wk.weight.detach(),'Wq':s.Wq.weight.detach(),'Wv':s.Wv.weight.detach(),'Wo':s.Wo.weight.detach()}
        bo=s.Wo.bias.detach()
        dWk=torch.zeros_like(W['Wk']);dWq=torch.zeros_like(W['Wq']);dWv=torch.zeros_like(W['Wv']);dWo=torch.zeros_like(W['Wo']);dbo=torch.zeros_like(bo)
        din=dout.clone()*1.0                                      # residual path: a_out=a_in+o -> dL/da_in += dout
        er=[];e_num=[];g_den=[]
        for t in range(Tt):
            (src,xk,kf,v,xq,qf,num,den,S)=cache[t]
            r=num/den; ert=dout[:,t]                              # dL/do_t = dout_t (o added to residual); dL/dr = Wo^T dout
            dWo+=(ert*scale).T@r; dbo+=(ert*scale).sum(0)
            err=ert@W['Wo']; er.append(err); e_num.append(err/den); g_den.append(-(err*num).sum(-1,keepdim=True)/den**2)
        zt=[];zr=torch.zeros(B,FE,device=DEV)
        for t in range(Tt): zr=zr+cache[t][2]; zt.append(zr.clone())
        P=torch.zeros(B,FE,DV,device=DEV); Q=torch.zeros(B,FE,device=DEV)
        for t in reversed(range(Tt)):
            (src,xk,kf,v,xq,qf,num,den,S)=cache[t]
            P=P+torch.einsum('bf,bd->bfd',qf,e_num[t]); Q=Q+g_den[t]*qf
            dv=torch.einsum('bf,bfd->bd',kf,P); dWv+=(dv*scale).T@a[:,t]
            din[:,t]+=(dv@W['Wv'])*1.0                            # value path -> a_in[:,t]
            dkf=torch.einsum('bfd,bd->bf',P,v)+Q; dxk=tvjp(xk,dkf); dWk+=(dxk*scale).T@src
            if t>0: din[:,t-1]+=(dxk@W['Wk'])*1.0                 # key path (shift) -> a_in[:,t-1]
            dqf=torch.einsum('bfd,bd->bf',S,e_num[t])+g_den[t]*zt[t]; dxq=tvjp(xq,dqf); dWq+=(dxq*scale).T@a[:,t]
            din[:,t]+=(dxq@W['Wq'])*1.0                           # query path -> a_in[:,t]
        return {'Wk':dWk,'Wq':dWq,'Wv':dWv,'Wo':dWo,'bo':dbo}, din

class Net(nn.Module):
    def __init__(s): super().__init__(); s.emb=nn.Embedding(VOCAB,D); s.layers=nn.ModuleList([Layer() for _ in range(NL)]); s.head=nn.Linear(D,VOCAB); nn.init.normal_(s.emb.weight,0,0.02)
    def forward(s,X):
        a=s.emb(X)
        for L in s.layers: a,_=L(a)
        return s.head(a)

def bptt_run(steps,B,lr):
    torch.manual_seed(0);np.random.seed(0); m=Net().to(DEV); opt=torch.optim.Adam(m.parameters(),lr=lr); c=[]
    for i in range(1,steps+1):
        X=torch.tensor(dm.batch(B),device=DEV); lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,VOCAB),X[:,1:T+1].reshape(-1))
        opt.zero_grad();loss.backward();opt.step()
        if i%50==0: c.append(loss.item()); print(f"  BPTT  {i} loss={loss.item():.3f}",flush=True)
    return c

def pc_run(steps,B,lr,K=8,eta=0.3):
    torch.manual_seed(0);np.random.seed(0); m=Net().to(DEV)
    prm={'emb':m.emb.weight,'hW':m.head.weight,'hb':m.head.bias}
    for li,L in enumerate(m.layers):
        for n in ('Wk','Wq','Wv','Wo'): prm[f'{li}.{n}']=getattr(L,n).weight
        prm[f'{li}.bo']=L.Wo.bias
    st={k:torch.zeros_like(p) for k,p in prm.items()}; vv={k:torch.zeros_like(p) for k,p in prm.items()}
    b1,b2,eps=0.9,0.999,1e-8; curve=[]
    for i in range(1,steps+1):
        X=torch.tensor(dm.batch(B),device=DEV); Y=X[:,1:T+1]; a0=m.emb(X[:,:T]).detach()
        with torch.no_grad():
            # forward predictions (clamped a0)
            ahat=[a0]; caches=[]
            a=a0
            for L in m.layers: o,ca=L(a); ahat.append(o); caches.append(ca); a=o
            aL=[ai.clone() for ai in ahat]                        # value nodes init = predictions
            # SETTLE hidden value nodes (only intermediate; output layer node aL[NL] settled w/ readout error)
            for _ in range(K):
                # readout error at top
                lg=aL[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
                e=p.clone(); e.view(-1,VOCAB)[torch.arange(B*T,device=DEV),Y.reshape(-1)]-=1.0
                rd=(e@m.head.weight)/(B*T)                        # dL/da_top [B,T,D]
                # recompute predictions from current lower nodes, get eps
                epsl=[None]*(NL+1); pca=[None]*NL
                a=aL[0]
                for li,L in enumerate(m.layers):
                    o,ca=L(aL[li]); pca[li]=ca; epsl[li+1]=aL[li+1]-o
                # top node update: dE/da_top = eps_top + readout
                aL[NL]=aL[NL]-eta*(epsl[NL]+rd)
                # intermediate node updates: dE/da_l = eps_l - VJP_{l+1}(eps_{l+1}) [top-down], for l=1..NL-1
                for li in range(NL-1,0,-1):
                    _,din=m.layers[li].local_vjp(aL[li],pca[li],epsl[li+1],1.0)  # dL/da_{li} from layer li+1's error
                    aL[li]=aL[li]-eta*(epsl[li]-din)
            # WEIGHT updates from settled local errors (eps) + readout
            g={k:torch.zeros_like(p) for k,p in prm.items()}
            lg=aL[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,VOCAB)[torch.arange(B*T,device=DEV),Y.reshape(-1)]-=1.0; e=e/(B*T)
            g['hW']=e.reshape(-1,VOCAB).T@aL[NL].reshape(-1,D); g['hb']=e.reshape(-1,VOCAB).sum(0)
            din0=None
            for li,L in enumerate(m.layers):
                o,ca=L(aL[li]); eps_l=aL[li+1]-o                  # local prediction error drives this layer
                pg,din=L.local_vjp(aL[li],ca,-eps_l,1.0)          # descend 0.5||eps||^2: grad via layer VJP of -eps
                for n in ('Wk','Wq','Wv','Wo'): g[f'{li}.{n}']=pg[n]
                g[f'{li}.bo']=pg['bo']
                if li==0: din0=din                                # dL/da0 (a0 is clamped = emb) -> emb grad
            gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); g['emb']=gemb
            for k in prm:
                st[k]=b1*st[k]+(1-b1)*g[k]; vv[k]=b2*vv[k]+(1-b2)*g[k]**2
                prm[k]-=lr*(st[k]/(1-b1**i))/((vv[k]/(1-b2**i)).sqrt()+eps)
        if i%50==0:
            with torch.no_grad():
                a=m.emb(X[:,:T])
                for L in m.layers: a,_=L(a)
                loss=F.cross_entropy((a@m.head.weight.T+m.head.bias).reshape(-1,VOCAB),Y.reshape(-1)).item()
            curve.append(loss); print(f"  PC    {i} loss={loss:.3f}",flush=True)
    return curve

if __name__=="__main__":
    print(f"device={DEV} {NL} layers D={D} T={T} -- multi-layer PC(depth)+eligibility(time) vs BPTT, real mix",flush=True)
    print("BPTT baseline:",flush=True); bp=bptt_run(400,32,3e-3)
    print("PC local (settling depth + eligibility time, NO global backprop):",flush=True); pc=pc_run(400,32,3e-3)
    print(f"\nfinal BPTT={bp[-1]:.3f}  PC-local={pc[-1]:.3f}  gap={pc[-1]-bp[-1]:+.3f}",flush=True)
    print(("PASS -> multi-layer pure-local (PC+eligibility) tracks BPTT" if pc[-1]-bp[-1]<0.3 else
           "PARTIAL -> PC-depth approximation gap; tune settling K/eta" if pc[-1]-bp[-1]<0.8 else
           "investigate multi-layer PC"),flush=True)
