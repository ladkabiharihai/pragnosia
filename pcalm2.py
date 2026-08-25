"""PC-ALM v2: fix the worst-layer floor via DECOUPLED linearization. PC-ALM settling produces the per-layer depth-error
signal g_l (duals). Backprop computes each layer's weight gradient at the FEEDFORWARD activations; my v1 wrongly used
the drifted SETTLED activations for the eligibility VJP -> worst-layer floor. Fix: run settling to get g_l, then apply
the weight VJP at the FEEDFORWARD caches (decouple). Test cosine vs backprop at depth 4/6/8 across seeds.
Fully local, no autograd, no global backward. Reuses core.Layer.local_vjp.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV

def pcalm2_grads(m,X,T_settle=None,eta_a=0.5,rho=1.0,decouple=True):
    cfg=m.cfg; NL=cfg.NL; T=cfg.T; D=cfg.D; vocab=cfg.vocab
    T_settle=T_settle or 4*NL; Y=X[:,1:T+1]; B=X.shape[0]; ar=torch.arange(B*T,device=DEV)
    with torch.no_grad():
        # FEEDFORWARD reference: activities + caches (the true linearization point for weight grads)
        ff_a=[m.emb(X[:,:T])]; ff_ca=[None]; x=ff_a[0]
        for L in m.layers: o,ca=L(x); ff_a.append(o); ff_ca.append(ca); x=o
        a=[ai.clone() for ai in ff_a]; lam=[torch.zeros_like(ai) for ai in ff_a]
        # SETTLING: local PC-ALM dynamics on current activities -> converge duals g_l to the depth errors
        for _ in range(T_settle):
            cach=[None]*(NL+1); c=[None]*(NL+1)
            for l in range(1,NL+1): o,ca=m.layers[l-1](a[l-1]); cach[l]=ca; c[l]=a[l]-o
            g=[None]+[lam[l]+rho*c[l] for l in range(1,NL+1)]
            lg=a[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,vocab)[ar,Y.reshape(-1)]-=1.0; rd=(e@m.head.weight)/(B*T)
            newa=[ai.clone() for ai in a]; newa[NL]=a[NL]-eta_a*(rd+g[NL])
            for l in range(NL-1,0,-1):
                _,din=m.layers[l].local_vjp(a[l],cach[l+1],g[l+1]); newa[l]=a[l]-eta_a*(g[l]-din)
            a=newa
            for l in range(1,NL+1): lam[l]=lam[l]+rho*c[l]
        # final depth errors g_l (at settled point)
        c=[None]*(NL+1)
        for l in range(1,NL+1): o,_=m.layers[l-1](a[l-1]); c[l]=a[l]-o
        g=[None]+[lam[l]+rho*c[l] for l in range(1,NL+1)]
        # WEIGHT grads: DECOUPLE -> apply g_l through the FEEDFORWARD caches (correct linearization)
        aout=ff_a[NL] if decouple else a[NL]
        grad={}; lg=aout@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
        e=p.clone(); e.view(-1,vocab)[ar,Y.reshape(-1)]-=1.0; e=e/(B*T)
        grad['hW']=e.reshape(-1,vocab).T@aout.reshape(-1,D); grad['hb']=e.reshape(-1,vocab).sum(0); din0=None
        for l in range(1,NL+1):
            aprev=ff_a[l-1] if decouple else a[l-1]
            cache=ff_ca[l] if decouple else m.layers[l-1](aprev)[1]
            pg,din=m.layers[l-1].local_vjp(aprev,cache,-g[l]); li=l-1
            for n in ('Wk','Wq','Wv','Wo'): grad[f'{li}.{n}']=pg[n]
            grad[f'{li}.bo']=pg['bo']
            if l==1: din0=din
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); grad['emb']=gemb
    return grad

if __name__=="__main__":
    import sys; sys.path.insert(0,"/opt/code/membrae"); from data_mix import DataMix
    T=32; dm=DataMix(ctx=T,verbose=False)
    print("PC-ALM v2: worst-layer cosine vs backprop across 3 seeds -- does DECOUPLE lift the floor? (T=4L)",flush=True)
    for NL in (4,6,8):
        for dc,tag in [(False,'baseline(settled VJP)'),(True,'decouple(ff VJP)')]:
            mins=[];means=[]
            for seed in range(3):
                torch.manual_seed(seed); np.random.seed(seed)
                cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T); m=C.Core(cfg).to(DEV); X=torch.tensor(dm.batch(24),device=DEV)
                Y=X[:,1:T+1]; lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),Y.reshape(-1)); m.zero_grad(); loss.backward()
                tg=[m.layers[li].Wk.weight.grad.clone() for li in range(NL)]+[m.layers[li].Wv.weight.grad.clone() for li in range(NL)]
                g=pcalm2_grads(m,X,T_settle=4*NL,eta_a=0.5,rho=1.0,decouple=dc)
                cs=[F.cosine_similarity(g[f'{li}.Wk'].flatten(),tg[li].flatten(),dim=0).item() for li in range(NL)]
                cs+=[F.cosine_similarity(g[f'{li}.Wv'].flatten(),tg[NL+li].flatten(),dim=0).item() for li in range(NL)]
                mins.append(min(cs)); means.append(np.mean(cs))
            print(f"  NL={NL:<2d} {tag:22s} worst={min(mins):.3f} (per-seed {[f'{x:.2f}' for x in mins]})  mean={np.mean(means):.3f}",flush=True)
