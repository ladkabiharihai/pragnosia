"""PC DE-RISK #0 (the load-bearing bet, VISION 0c): can PURE LOCAL Predictive Coding train a net with NO backprop,
matching backprop? Standard PC (Whittington-Bogacz / Millidge): value nodes settle to minimize local prediction
errors; weights update by a LOCAL rule (error x presynaptic activity). Backprop baseline = same architecture, same data.
Task: 2-moons (clearly nonlinear). Success = PC test-accuracy approaches backprop's. No autograd on the PC path.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"

# --- 2-moons dataset (synthetic, nonlinear, no sklearn dependency) ---
def moons(n=1000, noise=0.15):
    n2=n//2; t=np.linspace(0,np.pi,n2)
    x0=np.stack([np.cos(t), np.sin(t)],1)                      # upper moon
    x1=np.stack([1-np.cos(t), 1-np.sin(t)-0.5],1)              # lower moon shifted
    X=np.concatenate([x0,x1],0)+noise*np.random.randn(n,2)
    y=np.concatenate([np.zeros(n2),np.ones(n2)]).astype(int)
    p=np.random.permutation(n); X,y=X[p],y[p]
    X=(X-X.mean(0))/X.std(0)
    return torch.tensor(X,dtype=torch.float32,device=DEV), torch.tensor(y,dtype=torch.long,device=DEV)
Xtr,ytr=moons(1000); Xte,yte=moons(400)
SIZES=[2,64,64,2]                                              # MLP arch used by BOTH learners
def onehot(y,c=2): o=torch.zeros(len(y),c,device=DEV); o[range(len(y)),y]=1; return o

# ================= BACKPROP baseline (same arch) =================
def run_backprop(steps=400,lr=0.05):
    Ws=[torch.randn(SIZES[i],SIZES[i+1],device=DEV)*(2/SIZES[i])**0.5 for i in range(len(SIZES)-1)]
    bs=[torch.zeros(SIZES[i+1],device=DEV) for i in range(len(SIZES)-1)]
    for w in Ws+bs: w.requires_grad_(True)
    opt=torch.optim.SGD(Ws+bs,lr=lr,momentum=0.9)
    for _ in range(steps):
        h=Xtr
        for i,(w,b) in enumerate(zip(Ws,bs)):
            h=h@w+b
            if i<len(Ws)-1: h=torch.tanh(h)
        loss=torch.nn.functional.cross_entropy(h,ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h=Xte
        for i,(w,b) in enumerate(zip(Ws,bs)):
            h=h@w+b
            if i<len(Ws)-1: h=torch.tanh(h)
        acc=(h.argmax(1)==yte).float().mean().item()
    return acc,loss.item()

# ================= PURE PREDICTIVE CODING (no backprop) =================
# Bottom-up generative view: a[l] predicted from a[l-1]: pred[l]=phi(a[l-1])@W[l-1]+b. Errors e[l]=a[l]-pred[l].
# Settle hidden activities to minimize E=0.5*sum||e[l]||^2 (local). Then LOCAL weight update dW = phi(a[l-1])^T e[l].
# Input a[0] clamped to X; output a[L] clamped to target during learning (free at test). All updates LOCAL, no autograd.
def phi(x): return torch.tanh(x)
def dphi(x): return 1-torch.tanh(x)**2
def run_pc(steps=400, lr=0.05, T=20, infer_lr=0.5):
    L=len(SIZES)-1
    Ws=[torch.randn(SIZES[i],SIZES[i+1],device=DEV)*(2/SIZES[i])**0.5 for i in range(L)]
    bs=[torch.zeros(SIZES[i+1],device=DEV) for i in range(L)]
    def predict(a):                                            # pred[l+1] from a[l]
        return [phi(a[l])@Ws[l]+bs[l] for l in range(L)]
    def settle(X, target=None, T=T):
        # init activities by a forward pass
        a=[X]
        for l in range(L):
            z=phi(a[l])@Ws[l]+bs[l]; a.append(z)
        if target is not None: a[L]=target.clone()             # clamp output during learning
        a=[ai.clone() for ai in a]
        for _ in range(T):                                     # settle hidden nodes (l=1..L-1, and L if free)
            pred=predict(a)                                    # pred[l] approximates a[l+1]
            e=[a[l+1]-pred[l] for l in range(L)]               # e[l] is error at layer l+1
            for l in range(1,L+1):
                if l==L and target is not None: continue       # output clamped
                # dE/da[l] = e[l-1]  MINUS  contribution flowing from e[l] (top-down): dphi(a[l])*(e[l]@W[l]^T)
                grad=e[l-1].clone()
                if l<L: grad=grad - dphi(a[l])*(e[l]@Ws[l].T)
                a[l]=a[l]-infer_lr*grad
        return a
    for _ in range(steps):
        a=settle(Xtr, onehot(ytr))
        pred=predict(a); e=[a[l+1]-pred[l] for l in range(L)]
        for l in range(L):                                     # LOCAL weight update (no backprop): presyn phi(a[l]) x error e[l]
            Ws[l]=Ws[l]+lr*(phi(a[l]).T@e[l])/len(Xtr)
            bs[l]=bs[l]+lr*e[l].mean(0)
    # test: settle with output FREE, read a[L]
    a=settle(Xte, target=None, T=T*2)
    acc=(a[L].argmax(1)==yte).float().mean().item()
    En=sum((a[l+1]-predict(a)[l]).pow(2).mean().item() for l in range(L))
    return acc,En

if __name__=="__main__":
    print(f"device={DEV}, arch={SIZES}, task=2-moons (nonlinear)",flush=True)
    bp_acc,bp_loss=run_backprop()
    print(f"BACKPROP : test-acc={bp_acc*100:.1f}%  (train-loss {bp_loss:.3f})",flush=True)
    best=0; bestcfg=None
    for T in (20,40,80):                                       # more settling -> better local gradient approximation
        for ilr in (0.1,0.2,0.4):
            for lr,steps in ((0.05,400),(0.1,800)):
                torch.manual_seed(0); np.random.seed(0)
                acc,en=run_pc(steps=steps,lr=lr,T=T,infer_lr=ilr)
                if acc>best: best=acc; bestcfg=(T,ilr,lr,steps)
    print(f"PC (local, NO backprop) BEST: test-acc={best*100:.1f}%  cfg(T,ilr,lr,steps)={bestcfg}",flush=True)
    gap=bp_acc-best
    print(f"\nGAP backprop->PC = {gap*100:.1f} pts. {'PC MATCHES backprop -> load-bearing bet HOLDS at toy scale' if gap<0.05 else 'PC CLOSE (tuning) -> viable, refine' if gap<0.15 else 'PC LAGS -> investigate' if gap<0.25 else 'PC FAILS -> rethink'}",flush=True)
