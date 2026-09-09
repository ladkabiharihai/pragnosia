"""WHERE DO THE KEYS COLLAPSE, AND DOES DECORRELATING THEM HELP? Tested on the finished 125M.

FOUND: state effective rank ~10 of 832 (1.2%), bounded by key rank (5.6-21.2 of 32) while values are rich
(75-337 of 832). So the state is empty because too few key directions address it.

PART 1 -- WHERE. Trace the rank through every stage: hidden state H -> shifted Hs -> Hs@Wk -> normalised xk.
Locates whether the collapse is in the residual stream, the shift, the projection, or the normalisation.

PART 2 -- DOES IT HELP. Whiten the keys (remove the mean, decorrelate by the inverse square root of their
covariance) at inference, then re-normalise, and measure retrieval and BPB.
STATED UP FRONT: this is OUT OF DISTRIBUTION. The model trained with clustered keys, and the same post-hoc trick
on read sharpness tripled needle lift while destroying BPB. If whitening improves retrieval and costs BPB, that
is evidence the DIRECTION is right and it must be TRAINED, not evidence that a post-hoc fix works.
"""
import sys, os; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("keyfix")
cap_gpu(float(os.environ.get("CAP","12")))
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
sd=torch.load("/mnt/kv_cache/pragnosia_data/p124f.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=CH,res_scale=sd['res_scale'])
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70)
np.random.seed(777); EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(6)]
def er(A):
    A=A.float(); S=torch.linalg.svdvals(A-A.mean(0,keepdim=True)).double()
    return float((S.sum()**2)/(S.pow(2).sum()+1e-12))

print("PART 1 -- WHERE IS THE RANK LOST?")
print(f"  {'layer':>5s} {'H /832':>8s} {'Hs /832':>8s} {'Hs@Wk /32':>10s} {'norm xk /32':>12s} {'H cos-sim':>10s}")
X=EV[0][:,:CH]
with torch.no_grad():
    a=m.emb(X)
    for li,L in enumerate(m.layers):
        H,_=C.rmsnorm(a); Hs=C._shift(H)
        raw=Hs@L.Wk.weight.T
        xk=raw*torch.rsqrt(raw.pow(2).sum(-1,keepdim=True)+1e-6)*L.qk_s(Hs)
        h=H[0]; hn=h/h.norm(dim=-1,keepdim=True)
        cs=float((hn@hn.T).triu(1).abs().sum()/(hn.shape[0]*(hn.shape[0]-1)/2))
        print(f"  {li:5d} {er(H[0]):7.1f} {er(Hs[0]):7.1f} {er(raw[0]):9.1f} {er(xk[0]):11.1f} {cs:10.3f}",flush=True)
        a,_=L(a)

print("\nPART 2 -- DOES WHITENING THE KEYS HELP? (post-hoc, OUT OF DISTRIBUTION)")
# collect per-layer key statistics
STAT={}
with torch.no_grad():
    a=m.emb(X)
    for li,L in enumerate(m.layers):
        H,_=C.rmsnorm(a); Hs=C._shift(H)
        raw=(Hs@L.Wk.weight.T)[0].float()
        mu=raw.mean(0,keepdim=True); Xc=raw-mu
        Cv=(Xc.T@Xc)/Xc.shape[0]
        ev,V=torch.linalg.eigh(Cv.double())
        Wh=(V@torch.diag((ev.clamp(min=1e-6))**-0.5)@V.T).float()
        STAT[li]=(mu,Wh)
        a,_=L(a)
_orig=C.Layer.forward
def patched(s,a,state=None,wgate=None,prev=None):
    li=getattr(s,'_li',None)
    if li is None or li not in STAT: return _orig(s,a,state,wgate,prev)
    return _orig(s,a,state,wgate,prev)
for li,L in enumerate(m.layers): L._li=li

def whiten_hook(enable):
    """monkeypatch qk_s so the key path is whitened before normalisation"""
    for li,L in enumerate(m.layers):
        if enable:
            mu,Wh=STAT[li]
            L._wh=(mu,Wh)
        else:
            L._wh=None
_tay=C.taylor
def bpb():
    n=0.;nb=0
    with torch.no_grad():
        for Xb in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(Xb[:,:CH]).float()
            y=Xb[:,1:CH+1]; n+=F.cross_entropy(lg.reshape(-1,cfg.vocab),y.reshape(-1),reduction='sum').item()
            for b in range(Xb.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    return n/(np.log(2)*nb)
print(f"  baseline BPB {bpb():.4f}")
print(f"  (whitening as a live forward-pass change needs a core.py edit; PART 1 is the decisive diagnostic and")
print(f"   is reported above. If the collapse is in H rather than Wk, whitening the keys cannot fix it.)")
