"""PC-ALM v3: two levers against the deep-credit gap.
 (A) muPC depth-scaling: residual branch scaled by 1/sqrt(NL) (Depth-muP), published stable to 128 layers.
 (B) DUAL-STOPPING rule: track the PRIMAL STATIONARITY residual S = sum_l ||dE/da_l||^2 (intrinsic, no backprop info)
     and use the (a,lambda) snapshot at its MINIMUM. At the ALM solution the duals equal the adjoint and S->0, so
     min-S is the principled stopping point; it also fixes the T>4L dual OVERSHOOT.
Fully local, no autograd, no global backward. Reuses core.Layer.local_vjp (verified exact incl. res_scale).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV

def pcalm3_grads(m,X,T_settle=None,eta_a=0.5,rho=1.0,stop=True):
    cfg=m.cfg; NL=cfg.NL; T=cfg.T; D=cfg.D; V=cfg.vocab
    T_settle=T_settle or 4*NL; Y=X[:,1:T+1]; B=X.shape[0]; ar=torch.arange(B*T,device=DEV)
    with torch.no_grad():
        ff=[m.emb(X[:,:T])]; ffca=[None]; x=ff[0]
        for L in m.layers: o,ca=L(x); ff.append(o); ffca.append(ca); x=o
        a=[ai.clone() for ai in ff]; lam=[torch.zeros_like(ai) for ai in ff]
        best=(float('inf'),None,None,0)
        for k in range(T_settle):
            cach=[None]*(NL+1); c=[None]*(NL+1)
            for l in range(1,NL+1): o,ca=m.layers[l-1](a[l-1]); cach[l]=ca; c[l]=a[l]-o
            g=[None]+[lam[l]+rho*c[l] for l in range(1,NL+1)]
            lg=a[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,V)[ar,Y.reshape(-1)]-=1.0; rd=(e@m.head.weight)/(B*T)
            gtop=rd+g[NL]; S=gtop.pow(2).sum().item()
            hid=[None]*(NL+1)
            for l in range(NL-1,0,-1):
                _,din=m.layers[l].local_vjp(a[l],cach[l+1],g[l+1]); hid[l]=g[l]-din; S+=hid[l].pow(2).sum().item()
            if stop and S<best[0]: best=(S,[ai.clone() for ai in a],[li.clone() for li in lam],k)
            na=[ai.clone() for ai in a]; na[NL]=a[NL]-eta_a*gtop
            for l in range(NL-1,0,-1): na[l]=a[l]-eta_a*hid[l]
            a=na
            for l in range(1,NL+1): lam[l]=lam[l]+rho*c[l]
        kbest=T_settle
        if stop and best[1] is not None: a,lam,kbest=best[1],best[2],best[3]
        c=[None]*(NL+1)
        for l in range(1,NL+1): o,_=m.layers[l-1](a[l-1]); c[l]=a[l]-o
        g=[None]+[lam[l]+rho*c[l] for l in range(1,NL+1)]
        grad={}; lg=ff[NL]@m.head.weight.T+m.head.bias; p=lg.softmax(-1)
        e=p.clone(); e.view(-1,V)[ar,Y.reshape(-1)]-=1.0; e=e/(B*T)
        grad['hW']=e.reshape(-1,V).T@ff[NL].reshape(-1,D); grad['hb']=e.reshape(-1,V).sum(0); din0=None
        for l in range(1,NL+1):
            pg,din=m.layers[l-1].local_vjp(ff[l-1],ffca[l],-g[l]); li=l-1
            for n in ('Wk','Wq','Wv','Wo'): grad[f'{li}.{n}']=pg[n]
            grad[f'{li}.bo']=pg['bo']
            if l==1: din0=din
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); grad['emb']=gemb
        grad['_kbest']=kbest
    return grad

def evaluate(NL,T,dm,rs,stop,T_settle,seeds=3,B=16,eta_a=0.5):
    worst=[];means=[];ks=[]
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T,res_scale=rs); m=C.Core(cfg).to(DEV)
        X=torch.tensor(dm.batch(B),device=DEV); Y=X[:,1:T+1]
        lg=m(X[:,:T]); loss=F.cross_entropy(lg.reshape(-1,cfg.vocab),Y.reshape(-1)); m.zero_grad(); loss.backward()
        g=pcalm3_grads(m,X,T_settle=T_settle,eta_a=eta_a,stop=stop)
        cs=[F.cosine_similarity(g[f'{li}.{n}'].flatten(),getattr(m.layers[li],n).weight.grad.flatten(),dim=0).item()
            for li in range(NL) for n in ('Wk','Wq','Wv','Wo')]
        worst.append(min(cs)); means.append(float(np.mean(cs))); ks.append(g['_kbest'])
    return min(worst),float(np.mean(means)),worst,ks

if __name__=="__main__":
    import sys; sys.path.insert(0,"/opt/code/membrae"); from data_mix import DataMix
    T=32; dm=DataMix(ctx=T,verbose=False); NL=int(sys.argv[1]) if len(sys.argv)>1 else 6
    print(f"PC-ALM v3 levers @ NL={NL}, 3 seeds. baseline vs muPC vs dual-STOP vs both",flush=True)
    for Ts in (4*NL, 8*NL):
        print(f" --- T_settle={Ts} ({Ts//NL}L) ---",flush=True)
        for rs,stop,tag in [(1.0,False,'baseline'),(1.0,True,'dual-STOP'),
                            (NL**-0.5,False,'muPC'),(NL**-0.5,True,'muPC+STOP')]:
            w,mn,ws,ks=evaluate(NL,T,dm,rs,stop,Ts)
            print(f"   {tag:11s} worst={w:6.3f}  mean={mn:6.3f}  per-seed={[f'{x:.2f}' for x in ws]}  k*={ks}",flush=True)
