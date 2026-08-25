"""PC-ALM (Augmented Lagrangian Predictive Coding) local credit for our recurrent core. Reframes settling as
constrained optimization: minimize output loss s.t. a_l = f_l(a_{l-1}) for all layers. Per-layer DUAL variables
lambda_l accumulate the constraint violation (prediction error) and converge to the EXACT backprop costate ->
credit is BALLISTIC (depth-independent magnitude), closing the backprop gap at T=2L. Fully local, no weight transport,
no global backward. This replaces naive PC settling (diffusive, O(depth), ~0.85 floor). Reuses core.Layer.local_vjp.

Notation (residual layers, l=1..L; a_0 = emb clamped; output loss on a_L):
  f_l(a) = a + o_l(a);  c_l = a_l - f_l(a_{l-1});  g_l = lambda_l + rho*c_l  (generalized error)
  primal: da_l = g_l - J_{l+1}^T g_{l+1} (hidden), da_L = dLoss/da_L + g_L
  dual:   lambda_l += rho_dual * c_l
  weight: dTheta_l = local_vjp(a_{l-1}, cache_l, dout = -g_l)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV

def pcalm_grads(m,X,T_settle=None,eta_a=0.5,rho=1.0,rho_dual=1.0):
    cfg=m.cfg; NL=cfg.NL; T=cfg.T; D=cfg.D; vocab=cfg.vocab
    T_settle=T_settle or 2*NL; Y=X[:,1:T+1]; B=X.shape[0]; ar=torch.arange(B*T,device=DEV)
    with torch.no_grad():
        a0=m.emb(X[:,:T]); a=[a0]; x=a0
        for L in m.layers: o,_=L(x); a.append(o); x=o          # a[l] init = feedforward => c_l=0 initially
        lam=[torch.zeros_like(a[l]) for l in range(NL+1)]        # duals per layer (lam[0] unused)
        for _ in range(T_settle):
            # forward each layer on current activities -> predictions, errors c_l, caches
            cach=[None]*(NL+1); c=[None]*(NL+1)
            for l in range(1,NL+1):
                o,ca=m.layers[l-1](a[l-1]); cach[l]=ca; c[l]=a[l]-o
            g=[None]+[lam[l]+rho*c[l] for l in range(1,NL+1)]
            # primal updates
            # output node
            lg=a[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,vocab)[ar,Y.reshape(-1)]-=1.0; rd=(e@m.head.weight)/(B*T)
            newa=[ai.clone() for ai in a]
            newa[NL]=a[NL]-eta_a*(rd+g[NL])
            for l in range(NL-1,0,-1):
                _,din=m.layers[l].local_vjp(a[l],cach[l+1],g[l+1])   # J_{l+1}^T g_{l+1}
                newa[l]=a[l]-eta_a*(g[l]-din)
            a=newa
            # dual ascent (accumulate violation)
            for l in range(1,NL+1): lam[l]=lam[l]+rho_dual*c[l]
        # weight grads from final generalized errors
        cach=[None]*(NL+1); c=[None]*(NL+1)
        for l in range(1,NL+1):
            o,ca=m.layers[l-1](a[l-1]); cach[l]=ca; c[l]=a[l]-o
        g=[None]+[lam[l]+rho*c[l] for l in range(1,NL+1)]
        grad={}; lg=a[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
        e=p.clone(); e.view(-1,vocab)[ar,Y.reshape(-1)]-=1.0; e=e/(B*T)
        grad['hW']=e.reshape(-1,vocab).T@a[NL].reshape(-1,D); grad['hb']=e.reshape(-1,vocab).sum(0)
        din0=None
        for l in range(1,NL+1):
            pg,din=m.layers[l-1].local_vjp(a[l-1],cach[l],-g[l])
            li=l-1
            for n in ('Wk','Wq','Wv','Wo'): grad[f'{li}.{n}']=pg[n]
            grad[f'{li}.bo']=pg['bo']
            if l==1: din0=din
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); grad['emb']=gemb
    return grad

if __name__=="__main__":
    import sys; sys.path.insert(0,"/opt/code/membrae"); from data_mix import DataMix
    T=32; dm=DataMix(ctx=T,verbose=False)
    print("PC-ALM depth test: cosine(PC-ALM grad, backprop grad) at T_settle=2*NL (target: min>=0.99, floor lifted)",flush=True)
    for NL in (2,4,6,8):
        cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T); m=C.Core(cfg).to(DEV); X=torch.tensor(dm.batch(24),device=DEV)
        Y=X[:,1:T+1]; lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),Y.reshape(-1)); m.zero_grad(); loss.backward()
        tg={}
        for li,L in enumerate(m.layers):
            for n in ('Wk','Wv','Wo'): tg[f'{li}.{n}']=getattr(L,n).weight.grad.clone()
        tg['emb']=m.emb.weight.grad.clone()
        best=(-9,None)
        for eta_a in (0.3,0.5,0.8):
            for rho in (0.5,1.0):
                g=pcalm_grads(m,X,T_settle=2*NL,eta_a=eta_a,rho=rho,rho_dual=rho)
                cs=[F.cosine_similarity(g[k].flatten(),tg[k].flatten(),dim=0).item() for k in tg]
                if min(cs)>best[0]: best=(min(cs),(eta_a,rho),np.mean(cs))
        print(f"  NL={NL:<2d} T=2L={2*NL:<3d}: BEST min-cos={best[0]:.4f} mean={best[2]:.4f} at (eta_a,rho)={best[1]}",flush=True)
