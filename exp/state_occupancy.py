"""IS THE RECURRENT STATE ACTUALLY FULL? The measurement that decides whether "use the same state better" is live.

WHY IT MATTERS. Retrieval dies at ~2000 tokens and we have ruled out training context, interference, and DIN. The
state is a fixed FE x DV matrix that every token writes into and every read superimposes over. Two possibilities:
  (a) the state is UNDER-USED -- low effective rank, most directions carrying nothing. Then better ADDRESSING
      (sparse slots, selective read) buys capacity for free, with no growth and no extra compute.
  (b) the state is SATURATED -- near-full-rank, variance spread evenly. Then no addressing scheme helps, and the
      only routes are more state or a fundamentally different memory (Sparse Delta Memory's explicit slots).
These imply completely different next experiments, so measure before designing.

METHOD: stream real text through the model, take each layer's state M (FE x DV), and look at its singular value
spectrum. Report effective rank (participation ratio of the singular values -- robust, no arbitrary cutoff), the
fraction of dimensions holding 90% of the energy, and how those change as more tokens are written.
"""
import sys, os; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("state_occupancy")
cap_gpu(float(os.environ.get("CAP","12")))
import numpy as np, torch, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
CK=sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/p124f.pt"
sd=torch.load(CK,map_location=DEV); cfg=cfg_from_ckpt(sd,T=CH,res_scale=sd['res_scale'])
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
print(f"  {CK.split('/')[-1]}: D{cfg.D}x{cfg.NL}L DIN{cfg.DIN}  FE={cfg.FE}  DV={cfg.DV}")
print(f"  state per layer is {cfg.FE} x {cfg.DV} = {cfg.FE*cfg.DV/1e6:.2f}M numbers; max possible rank {min(cfg.FE,cfg.DV)}\n")

dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70); np.random.seed(5)
POOL=[]
while len(POOL)<200000: POOL.extend(dm.batch(1)[0].tolist())

def eff_rank(S):
    """participation ratio of the singular values: (sum s)^2 / sum s^2. No arbitrary threshold."""
    s=S.double(); return float((s.sum()**2)/(s.pow(2).sum()+1e-12))

def occupancy(ntok):
    st=[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
    pv=None
    with torch.no_grad():
        for i in range(0,ntok,CH):
            x=torch.tensor([POOL[i:i+CH]],device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                _,st,pv=m(x,states=st,return_states=True,prev=pv,return_prev=True)
    out=[]
    with torch.no_grad():
        for li,(M,z) in enumerate(st):
            A=M[0].float()
            S=torch.linalg.svdvals(A)
            er=eff_rank(S)
            c=S.pow(2).cumsum(0)/S.pow(2).sum()
            n90=int((c<0.90).sum())+1
            out.append((er,n90,float(S[0]),float(S.sum())))
    del st,pv; torch.cuda.empty_cache(); return out

print(f"  {'tokens':>8s} {'layer':>6s} {'eff.rank':>9s} {'of max':>7s} {'dims for 90% energy':>20s}")
for ntok in (2048, 16384, 65536, 200704):
    o=occupancy(ntok)
    ers=[x[0] for x in o]; n90s=[x[1] for x in o]
    mx=min(cfg.FE,cfg.DV)
    print(f"  {ntok:8,} {'mean':>6s} {np.mean(ers):9.1f} {100*np.mean(ers)/mx:6.1f}% {np.mean(n90s):19.0f}",flush=True)
    if ntok==200704:
        print(f"           per-layer eff.rank: "+" ".join(f"{x[0]:.0f}" for x in o),flush=True)
        print(f"           per-layer 90%-dims: "+" ".join(f"{x[1]:d}" for x in o),flush=True)
print(f"\n  max possible rank = min(FE,DV) = {min(cfg.FE,cfg.DV)}")
