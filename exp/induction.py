"""INDUCTION, MEASURED WITH ENOUGH TRIALS TO MEAN SOMETHING.

Every induction number in this project came from n=100 single-sequence forwards, which carries a standard error
of +-2.6% at p=0.07. That is not a detail: two runs of an IDENTICAL configuration (readscale arm B and the DIN=16
arm of dinsweep -- same seeds, same budget) scored 7% and 3%. I reported the ordering of arm B against arm C at
7% vs 10% as if it were a result; at that sample size it was noise.

The trials are independent and the model carries no state between them, so they BATCH exactly -- which makes
n=1000 cheaper than the old n=100 loop. Reports a Wilson 95% interval, because a normal interval is wrong near 0
and induction sits near 0 for every model here.
"""
import sys, math; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch
import core as C
DEV=C.DEV

def wilson(k,n,z=1.96):
    """95% CI for a proportion. Correct at k=0, where the normal approximation gives the useless +-0.0."""
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*max(0.,c-h),100*min(1.,c+h))

@torch.no_grad()
def induction(m,L=128,n=1000,bs=50,seed=2,vocab_lo=1000,vocab_hi=30000):
    """[A][B] ... [A] -> [B] over random tokens. Batched; identical to the per-sequence loop, just faster."""
    was=m.training; m.eval()
    rng=np.random.RandomState(seed); hit=0; done=0
    while done<n:
        b=min(bs,n-done)
        seq=rng.randint(vocab_lo,vocab_hi,(b,L))
        ab=rng.randint(vocab_lo,vocab_hi,(b,2))
        seq[:,10]=ab[:,0]; seq[:,11]=ab[:,1]; seq[:,-1]=ab[:,0]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(torch.tensor(seq,device=DEV))[:,-1].float()
        hit+=int((lg.argmax(-1).cpu().numpy()==ab[:,1]).sum()); done+=b
    if was: m.train()
    lo,hi=wilson(hit,n)
    return 100*hit/n,lo,hi,hit,n

if __name__=="__main__":
    from util import cfg_from_ckpt
    ck=sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/p124g.pt"
    N=int(sys.argv[2]) if len(sys.argv)>2 else 1000
    sd=torch.load(ck,map_location=DEV); cfg=cfg_from_ckpt(sd,T=2048,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    print(f"  {ck.split('/')[-1]}   n={N} per length, Wilson 95% interval")
    print(f"  {'L':>6s} {'induction':>10s} {'95% CI':>16s}")
    for L in (64,128,256,512):
        p,lo,hi,k,n=induction(m,L=L,n=N)
        print(f"  {L:6d} {p:9.1f}% {f'[{lo:.1f}, {hi:.1f}]':>16s}",flush=True)
