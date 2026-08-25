"""#4 DIAGNOSTIC: is the multi-layer PC lag a SETTLING-CONVERGENCE issue (fixable) or STRUCTURAL (bug)?
Measure cosine(PC-grad, true backprop-grad) per layer as a function of settling steps K. Theory: at PC inference
equilibrium eps_l = J^T eps_{l+1} == backprop error recursion, so PC-grad -> backprop-grad as K->convergence.
If cosine -> 1 with more K -> pure convergence issue (fix: more/annealed settling, or Z-IL exact-in-depth). If cosine
plateaus < 1 -> structural error to fix. Reuses Layer/Net from pc_derisk4_multilayer.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import sys; sys.path.insert(0,"/opt/code/membrae")
import pc_derisk4_multilayer as M
DEV=M.DEV; VOCAB=M.VOCAB; T=M.T; NL=M.NL; D=M.D
torch.manual_seed(0); np.random.seed(0)

def true_grads(m,X):
    Y=X[:,1:T+1]; lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,VOCAB),Y.reshape(-1)); m.zero_grad(); loss.backward()
    g={'emb':m.emb.weight.grad.clone(),'hW':m.head.weight.grad.clone(),'hb':m.head.bias.grad.clone()}
    for li,L in enumerate(m.layers):
        for n in ('Wk','Wq','Wv','Wo'): g[f'{li}.{n}']=getattr(L,n).weight.grad.clone()
        g[f'{li}.bo']=L.Wo.bias.grad.clone()
    return g

def pc_grads(m,X,K,eta):
    Y=X[:,1:T+1]; B=X.shape[0]; a0=m.emb(X[:,:T]).detach()
    with torch.no_grad():
        ahat=[a0]; a=a0
        for L in m.layers: o,_=L(a); ahat.append(o); a=o
        aL=[ai.clone() for ai in ahat]
        for _ in range(K):
            lg=aL[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,VOCAB)[torch.arange(B*T,device=DEV),Y.reshape(-1)]-=1.0
            rd=(e@m.head.weight)/(B*T)
            epsl=[None]*(NL+1); pca=[None]*NL
            for li,L in enumerate(m.layers): o,ca=L(aL[li]); pca[li]=ca; epsl[li+1]=aL[li+1]-o
            aL[NL]=aL[NL]-eta*(epsl[NL]+rd)
            for li in range(NL-1,0,-1):
                _,din=m.layers[li].local_vjp(aL[li],pca[li],epsl[li+1],1.0); aL[li]=aL[li]-eta*(epsl[li]-din)
        # weight grads from settled state
        g={}; lg=aL[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
        e=p.clone(); e.view(-1,VOCAB)[torch.arange(B*T,device=DEV),Y.reshape(-1)]-=1.0; e=e/(B*T)
        g['hW']=e.reshape(-1,VOCAB).T@aL[NL].reshape(-1,D); g['hb']=e.reshape(-1,VOCAB).sum(0)
        din0=None
        for li,L in enumerate(m.layers):
            o,ca=L(aL[li]); eps_l=aL[li+1]-o
            pg,din=L.local_vjp(aL[li],ca,-eps_l,1.0)
            for n in ('Wk','Wq','Wv','Wo'): g[f'{li}.{n}']=pg[n]
            g[f'{li}.bo']=pg['bo']
            if li==0: din0=din
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); g['emb']=gemb
    return g

if __name__=="__main__":
    m=M.Net().to(DEV); X=torch.tensor(M.dm.batch(24),device=DEV)
    tg=true_grads(m,X)
    keys=['emb','0.Wk','0.Wv','0.Wo','1.Wk','1.Wv','1.Wo','hW']
    print(f"cosine(PC-grad, backprop-grad) vs settling K  (->1 = convergence issue; plateau<1 = structural)",flush=True)
    print("  K   eta  "+"  ".join(f"{k:>6s}" for k in keys)+"   MIN",flush=True)
    best=(-9,None)
    for eta in (0.05,0.1,0.15,0.2,0.3):
        for K in (1,2,3,4,6,8,12):
            g=pc_grads(m,X,K,eta)
            cs=[F.cosine_similarity(g[k].flatten(),tg[k].flatten(),dim=0).item() for k in keys]
            mn=min(cs)
            if mn>best[0]: best=(mn,(K,eta))
            print(f"  {K:<3d} {eta:<4} "+"  ".join(f"{c:6.3f}" for c in cs)+f"   {mn:6.3f}",flush=True)
    print(f"\nBEST: min-cosine={best[0]:.4f} at (K,eta)={best[1]}  (>0.98 = PC-depth ~= backprop, #4 fixable)",flush=True)
