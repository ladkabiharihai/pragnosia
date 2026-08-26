"""LAYERWISE LOCAL OBJECTIVES: unlock TRUE DEPTH under pure local plasticity (no backprop, no settling).
Each layer gets its OWN next-token objective read out through the shared head, and gradients are BLOCKED between
layers. So there is NO cross-layer credit to assign -- the mechanism that failed at depth is removed entirely.
Within-layer credit stays EXACT (eligibility traces, core.Layer.local_vjp, verified cosine 1.0000).
Why this should not suffer the classic greedy-layerwise failure (early layers destroying information):
  - layers are RESIDUAL (a_l = a_{l-1} + rs*o_l), so the stream carries everything forward; a layer can only ADD.
  - every layer shares the SAME target and the SAME readout, so each layer REFINES the prediction (iterative
    refinement / logit-lens), rather than solving a myopic private task.
Cost: NL forwards + NL local VJPs per step = ~backprop cost, vs 8-12x for PC settling. Big efficiency win (0b2).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV

def layerwise_grads(m,X,per_layer_heads=None):
    """Pure-local grads from per-layer local objectives. per_layer_heads: optional list of (W,b) for separate heads."""
    cfg=m.cfg; NL=cfg.NL; T=cfg.T; D=cfg.D; V=cfg.vocab
    Y=X[:,1:T+1]; B=X.shape[0]; ar=torch.arange(B*T,device=DEV)
    with torch.no_grad():
        g={}; a_in=m.emb(X[:,:T])
        dW=torch.zeros_like(m.head.weight); db=torch.zeros_like(m.head.bias); din0=None
        for l in range(NL):
            a_out,cache=m.layers[l](a_in)
            if per_layer_heads is None: Wh,bh=m.head.weight,m.head.bias
            else: Wh,bh=per_layer_heads[l]
            lg=a_out@Wh.T+bh; p=lg.softmax(-1)
            e=p.clone(); e.view(-1,V)[ar,Y.reshape(-1)]-=1.0; e=e/(B*T)      # LOCAL error of THIS layer's objective
            if per_layer_heads is None:
                dW+=e.reshape(-1,V).T@a_out.reshape(-1,D); db+=e.reshape(-1,V).sum(0)
            else:
                g[f'h{l}.W']=e.reshape(-1,V).T@a_out.reshape(-1,D); g[f'h{l}.b']=e.reshape(-1,V).sum(0)
            dout=e@Wh                                                        # dL_l / da_out  (local, no backprop)
            pg,din=m.layers[l].local_vjp(a_in,cache,dout)                    # EXACT within-layer credit
            for n in ('Wk','Wq','Wv','Wo'): g[f'{l}.{n}']=pg[n]
            g[f'{l}.bo']=pg['bo']
            if l==0: din0=din
            a_in=a_out                                                       # forward only: NO gradient crosses here
        gemb=torch.zeros_like(m.emb.weight); gemb.index_add_(0,X[:,:T].reshape(-1),din0.reshape(-1,D)); g['emb']=gemb
        g['hW']=dW; g['hb']=db
    return g

if __name__=="__main__":
    import sys,time; sys.path.insert(0,"/opt/code/membrae"); from data_mix import DataMix
    NL=int(sys.argv[1]) if len(sys.argv)>1 else 6
    T=32; B=32; LR=3e-3; STEPS=400; RS=NL**-0.5; dm=DataMix(ctx=T,verbose=False)
    def fresh():
        torch.manual_seed(0); np.random.seed(0)
        cfg=C.Config(D=96,DIN=8,DV=96,NL=NL,T=T,res_scale=RS); return cfg,C.Core(cfg).to(DEV)
    def evloss(m,cfg,nb=8):
        tot=0.
        with torch.no_grad():
            for _ in range(nb):
                X=torch.tensor(dm.batch(B),device=DEV)
                tot+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
        return tot/nb
    def run(arm):
        cfg,m=fresh(); torch.manual_seed(1); np.random.seed(1); t0=time.time()
        opt=torch.optim.Adam(m.parameters(),lr=LR) if arm=='bptt' else C.LocalAdam(m.param_dict(),lr=LR)
        for i in range(1,STEPS+1):
            X=torch.tensor(dm.batch(B),device=DEV)
            if arm=='bptt':
                loss=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
            else: opt.step(layerwise_grads(m,X))
            if i%100==0: print(f"    {arm:10s} step {i:4d} heldout={evloss(m,cfg):.4f} ({time.time()-t0:.0f}s)",flush=True)
        return evloss(m,cfg),time.time()-t0
    print(f"LAYERWISE-LOCAL vs BPTT @ NL={NL}, D=96, T={T}, B={B}, {STEPS} steps, real 290B mix",flush=True)
    bl,bt=run('bptt');  print(f"  == BPTT           final={bl:.4f} ppl={np.exp(bl):.1f} ({bt:.0f}s)",flush=True)
    ll,lt=run('layerwise'); print(f"  == LAYERWISE(loc) final={ll:.4f} ppl={np.exp(ll):.1f} ({lt:.0f}s, {lt/bt:.1f}x)",flush=True)
    print(f"\nGAP vs backprop = {ll-bl:+.4f}   (PC-settling was +1.06 at 8.2x cost)",flush=True)
    print(("PASS: layerwise pure-local ~= backprop at depth -> TRUE DEPTH UNLOCKED" if ll-bl<0.15 else
           "CLOSE" if ll-bl<0.4 else "gap remains"),flush=True)
