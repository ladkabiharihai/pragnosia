"""Is window2 (Cosmopedia + FineWeb-Edu, textbook/reasoning-heavy) actually BETTER training data than dclm (raw web)?
The research says educational data gives ~10x token efficiency at small scale, but that was transformers. Test it
HERE: train the same small model for the same steps on three mixes and compare held-out loss on a NEUTRAL eval set."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log,pbar; setup_log("mix_quality")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV; CH=1024; B=8; STEPS=1500
import data_mix as DM
def mixer(weights):
    """build a DataMix then override its sampling weights"""
    dm=DM.DataMix(ctx=CH+1,verbose=False)
    w=np.zeros(len(dm.names))
    for i,n in enumerate(dm.names): w[i]=weights.get(n,0.0)
    dm.w=w/w.sum(); return dm
def train(dm,tag):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=384,DIN=8,DV=384,NL=4,T=CH,res_scale=16**-0.5); m=C.Core(cfg).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); torch.manual_seed(1); np.random.seed(1); t0=time.time()
    for it in range(STEPS):
        X=torch.tensor(dm.batch(B),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m,cfg,time.time()-t0
def heldout(m,cfg,evalsets):
    out={}
    with torch.no_grad():
        for name,batches in evalsets.items():
            tot=0.
            for X in batches:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    tot+=F.cross_entropy(m(X[:,:CH]).reshape(-1,cfg.vocab).float(),X[:,1:CH+1].reshape(-1)).item()
            out[name]=tot/len(batches)
    return out
if __name__=="__main__":
    # NEUTRAL eval sets, fixed seed: judge each mix on BOTH kinds of data so we do not just measure self-similarity
    np.random.seed(999)
    ev={}
    for nm,wts in [("edu(window2)",{"window2":1.0}),("web(dclm)",{"dclm":1.0}),("reasoning",{"hermes":1.0,"tulu3":1.0})]:
        d=mixer(wts); ev[nm]=[torch.tensor(d.batch(4),device=DEV) for _ in range(6)]
    MIXES={
      "current (43/33 edu/web)": {"window2":.437,"dclm":.330,"finemath":.057,"opencode":.034,"openmath":.034,
                                  "hermes":.034,"tulu3":.034,"evolcode":.011,"glaive":.011,"numina":.011,"magicoder":.006},
      "edu-heavy (65/10)":       {"window2":.65,"dclm":.10,"finemath":.05,"opencode":.03,"openmath":.03,
                                  "hermes":.05,"tulu3":.05,"numina":.01,"evolcode":.01,"glaive":.01,"magicoder":.005},
      "web-heavy (20/60)":       {"window2":.20,"dclm":.60,"finemath":.05,"opencode":.03,"openmath":.03,
                                  "hermes":.03,"tulu3":.03,"numina":.01,"evolcode":.01,"glaive":.01,"magicoder":.005},
    }
    print(f"MIX QUALITY TEST: same model/steps, three mixes, judged on NEUTRAL held-out sets ({STEPS} steps)",flush=True)
    print(f"  {'mix':>26s} {'edu eval':>9s} {'web eval':>9s} {'reasoning':>10s} {'MEAN':>8s}",flush=True)
    for tag,wts in MIXES.items():
        m,cfg,dt=train(mixer(wts),tag); r=heldout(m,cfg,ev)
        mean=np.mean(list(r.values()))
        print(f"  {tag:>26s} {r['edu(window2)']:9.4f} {r['web(dclm)']:9.4f} {r['reasoning']:10.4f} {mean:8.4f}",flush=True)
        del m; torch.cuda.empty_cache()
