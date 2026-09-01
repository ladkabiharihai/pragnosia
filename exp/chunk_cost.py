"""CHUNK LENGTH: 4x cheaper attention, but does it cost quality? Our attention is O(chunk^2), and we carry state
BETWEEN chunks, so a shorter chunk computes the same function far more cheaply -- chunk 512 is 4x fewer attention
FLOPs than chunk 2048 for the same tokens.

THE CATCH: the state is DETACHED between chunks, so the chunk length is also the BACKPROP HORIZON. Shorter chunks
truncate the gradient path. The chunk size was last benchmarked when DIN=8 (FE=45); at DIN=24 (FE=325) the
quadratic term is 7.2x heavier, so the balance may have moved.

Measure throughput AND loss at matched TOKENS, then at matched WALL-CLOCK -- which is what actually matters."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("chunk_cost")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; D=256; NL=4; DIN=24; TOKENS=24_000_000   # small footprint: the trainer and the arch control own most of the card
np.random.seed(7)
_dm=DataMix(ctx=1025,verbose=False,only='window2')
EV=[torch.tensor(_dm.batch(2),device=DEV) for _ in range(12)]
def evl(m,cfg):
    m.eval(); t=0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:1024]).reshape(-1,cfg.vocab).float(),X[:,1:1025].reshape(-1)).item()
    m.train(); return t/len(EV)
def run(CH,budget_tokens=TOKENS,budget_s=None):
    """train on 32768-token documents in chunks of CH, carrying state -- the real training loop, shrunk."""
    torch.manual_seed(0); np.random.seed(0)
    B=max(1,4096//CH)                      # keep tokens/step roughly constant
    cfg=C.Config(D=D,DIN=DIN,DV=D,NL=NL,T=CH,res_scale=NL**-0.5,tie_emb=True)
    m=C.Core(cfg).to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    DOC=4096; NCH=DOC//CH
    dm=DataMix(ctx=DOC+1,verbose=False,only='window2')
    seen=0; t0=time.time()
    while (seen<budget_tokens) if budget_s is None else (time.time()-t0<budget_s):
        X=torch.tensor(dm.batch(B),device=DEV); st=None
        opt.zero_grad()
        for i in range(NCH):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg,st=m(xi,states=st,return_states=True)
                l=F.cross_entropy(lg.reshape(-1,cfg.vocab),yi.reshape(-1))/NCH
            st=[(a.detach(),b.detach()) for a,b in st]
            l.backward()
            seen+=B*CH
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    dt=time.time()-t0
    out=evl(m,cfg); del m,opt; torch.cuda.empty_cache()
    return out,seen/dt,seen
if __name__=="__main__":
    print(f"CHUNK LENGTH at DIN={DIN} (FE=325). Attention is O(chunk^2); state is detached between chunks.\n",flush=True)
    print(f"A) MATCHED TOKENS ({TOKENS/1e6:.0f}M each) -- isolates the backprop-horizon effect",flush=True)
    print(f"   {'chunk':>6s} {'loss':>8s} {'ktok/s':>8s} {'time':>7s}",flush=True)
    sp={}
    for CH in (512,1024,2048):
        l,tps,seen=run(CH); sp[CH]=tps
        print(f"   {CH:6d} {l:8.4f} {tps/1e3:8.0f} {seen/tps:6.0f}s",flush=True)
    print(f"\nB) MATCHED WALL-CLOCK (120s each) -- what a fixed GPU budget actually buys",flush=True)
    print(f"   {'chunk':>6s} {'loss':>8s} {'tokens seen':>12s}",flush=True)
    for CH in (512,1024,2048):
        l,tps,seen=run(CH,budget_s=120)
        print(f"   {CH:6d} {l:8.4f} {seen/1e6:11.1f}M",flush=True)
