"""FIX THE GROWTH so new capacity actually gets recruited. Ablation showed 84% of params contribute 0.59 nats
while the 16% seed carries 8.13. Diagnosis: grow_width writes ZERO into every new dim (emb, Wo, m2) while reading
them randomly, so the new dims are a RANDOM PERTURBATION of an already-fitted function -- gradient descent has no
reason to prefer them, and pushes them back toward zero.

FOUR STRATEGIES, tested head to head. Each grows the SAME trained model, trains the SAME steps, then measures how
much load the new dims actually carry (damage when zeroed).
  A ZERO      : current behaviour -- write zero, read random.
  B RANDOM    : write small random into new dims (breaks exact preservation, but gives a real signal to learn from).
  C NET2NET   : new dims COPY an existing useful dim's input weights and SPLIT its output weight, then add noise.
                Function preserved exactly, but the new unit starts as a MEANINGFUL feature it can specialise from.
  D ZERO+LR   : current init, but a 10x learning rate on the newly added parameters only.
"""
import sys,time,copy; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("fix_growth")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix
DEV=C.DEV; CH=512; B=8; PRE=1200; POST=1500; ADD=128; D0=256
dm=DataMix(ctx=CH+1,verbose=False,only='window2')
np.random.seed(7); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(6)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
    m.train(); return t/len(EV)
def train(m,cfg,steps,lr=1e-3,newp=None,newlr=None):
    if newp is None: opt=torch.optim.Adam(m.parameters(),lr=lr)
    else:
        ids={id(p) for p in newp}
        old=[p for p in m.parameters() if id(p) not in ids]
        opt=torch.optim.Adam([{'params':old,'lr':lr},{'params':newp,'lr':newlr}])
    for it in range(steps):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            l=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m
def load_contrib(m,cfg,lo,hi):
    """how much do dims [lo,hi) contribute? = damage when their WRITERS are zeroed"""
    base=evl(m,cfg); sl=slice(lo,hi); sav={}
    with torch.no_grad():
        for i,L in enumerate(m.layers):
            sav[i]=(L.Wo.weight[sl,:].clone(),L.m2.weight[sl,:].clone())
            L.Wo.weight[sl,:]=0; L.m2.weight[sl,:]=0
    d=evl(m,cfg)-base
    with torch.no_grad():
        for i,L in enumerate(m.layers):
            a,b=sav[i]; L.Wo.weight[sl,:]=a; L.m2.weight[sl,:]=b
    return base,d
# ---- strategy B: write small random into the new dims instead of zero ----
def grow_random(m,add,scale=0.02):
    D=m.cfg.D; nD=G.grow_width(m,add)
    with torch.no_grad():
        m.emb.weight[:,D:].normal_(0,scale)
        for L in m.layers:
            L.Wo.weight[D:].normal_(0,scale/ (L.Wo.weight.shape[1]**0.5))
            L.m2.weight[D:].normal_(0,scale/ (L.m2.weight.shape[1]**0.5))
    return nD
# ---- strategy C: Net2Net -- copy a useful existing dim and split its output ----
def grow_net2net(m,add,noise=0.01):
    D=m.cfg.D
    with torch.no_grad():
        # pick the `add` most ACTIVE existing dims to clone (they carry real features)
        act=m.emb.weight.float().pow(2).mean(0)
        src=torch.topk(act,add).indices
    nD=G.grow_width(m,add)
    with torch.no_grad():
        # new dim j is a copy of src[j]; split the OUTPUT weight of src[j] between the two so the sum is preserved
        m.emb.weight[:,D:]=m.emb.weight[:,src]*0.5
        m.emb.weight[:,src]*=0.5
        for L in m.layers:
            L.Wo.weight[D:]=L.Wo.weight[src]*0.5; L.Wo.weight[src]*=0.5
            L.m2.weight[D:]=L.m2.weight[src]*0.5; L.m2.weight[src]*=0.5
            # readers of the cloned pair must also be split so the downstream sum is unchanged
            for nm in ('Wk','Wq','Wv'):
                W=getattr(L,nm).weight; W[:,D:]=W[:,src]; 
            L.m1.weight[:,D:]=L.m1.weight[:,src]
            for nm in ('Wk','Wq','Wv'):
                getattr(L,nm).weight[:,D:]+=torch.randn_like(getattr(L,nm).weight[:,D:])*noise
        m.head.weight[:,D:]=m.head.weight[:,src]*0.5; m.head.weight[:,src]*=0.5
    return nD
if __name__=="__main__":
    print(f"FIXING GROWTH: pretrain D={D0} for {PRE} steps, grow +{ADD}, train {POST}, measure new-dim load\n",flush=True)
    torch.manual_seed(0); np.random.seed(0)
    cfg0=C.Config(D=D0,DIN=8,DV=D0,NL=4,T=CH,res_scale=4**-0.5)
    seed=C.Core(cfg0).to(DEV); train(seed,cfg0,PRE)
    base_loss=evl(seed,cfg0)
    sd0=copy.deepcopy(seed.state_dict())
    print(f"  seed model before growing: loss {base_loss:.4f}\n",flush=True)
    print(f"  {'strategy':>22s} {'loss after grow':>16s} {'loss after train':>17s} {'NEW-DIM LOAD':>13s}",flush=True)
    results={}
    for tag,fn,lrargs in [("A zero (current)",lambda m:G.grow_width(m,ADD),{}),
                          ("B random write",lambda m:grow_random(m,ADD),{}),
                          ("C net2net copy+split",lambda m:grow_net2net(m,ADD),{}),
                          ("D zero + 10x LR",lambda m:G.grow_width(m,ADD),{"lr10":True})]:
        torch.manual_seed(1); np.random.seed(1)
        cfg=C.Config(D=D0,DIN=8,DV=D0,NL=4,T=CH,res_scale=4**-0.5)
        m=C.Core(cfg).to(DEV); m.load_state_dict(sd0)
        before=[p for p in m.parameters()]
        fn(m); cfg=m.cfg
        just=evl(m,cfg)
        newp=None; nlr=None
        if lrargs.get("lr10"):
            # newly-added slices: approximate by taking the whole grown tensors' new rows
            newp=[]
            for L in m.layers: newp+= [L.Wo.weight,L.m2.weight]
            nlr=1e-2
        train(m,cfg,POST,lr=1e-3,newp=newp,newlr=nlr)
        after,dmg=load_contrib(m,cfg,D0,D0+ADD)
        results[tag]=(just,after,dmg)
        print(f"  {tag:>22s} {just:16.4f} {after:17.4f} {dmg:+13.4f}",flush=True)
        del m; torch.cuda.empty_cache()
    print(f"\n  (NEW-DIM LOAD = nats lost when the new dims are silenced; higher = capacity actually recruited)",flush=True)
    best=max(results.items(),key=lambda kv:kv[1][2])
    print(f"  WINNER: {best[0]}  -- new dims worth {best[1][2]:+.4f} nats, final loss {best[1][1]:.4f}",flush=True)
