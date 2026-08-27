"""ACCEPTANCE 4 END-TO-END: verification GATES consolidation. A true fact becomes permanent; a false one does NOT.
This is the anti-poisoning claim of VISION flag 2, demonstrated on the real model."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, verify as VF, lw_lib as L; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D=256; NL=6; T=128; dm=DataMix(ctx=T+1,verbose=False)
def load():
    torch.manual_seed(0); cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
    m.load_state_dict(torch.load("/mnt/kv_cache/pragnosia_data/substrate_20k.pt")); m.eval(); return m,cfg
@torch.no_grad()
def logp(m,ctx,ans):
    ids=TOK.encode(ctx).ids+TOK.encode(ans).ids; n=len(TOK.encode(ans).ids); x=torch.tensor([ids],device=DEV)
    lg=m(x[:,:-1]).log_softmax(-1)
    return lg[0,-n:].gather(-1,x[0,1:][-n:].unsqueeze(-1)).mean().item()
@torch.no_grad()
def general(m,cfg,nb=8):
    t=0.
    for _ in range(nb):
        X=torch.tensor(dm.batch(8),device=DEV); t+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return t/nb
def ep(text,B=8):
    ids=TOK.encode(text).ids; ids=(ids*((T+1)//len(ids)+1))[:T+1]
    return torch.tensor([ids]*B,device=DEV)
def consolidate(m,text,steps=30,lr=3e-3):
    opt=C.LocalAdam(m.param_dict(),lr=lr,adaptive=True)
    for i in range(steps):
        X=torch.tensor(dm.batch(8),device=DEV) if np.random.rand()<0.5 else ep(text)
        opt.step(L.lw_grads(m,X,lam=4.0,k_local=2))
CLAIMS=[("TRUE ","The capital of France is"," Paris","The capital of France is Paris, a famous European city."),
        ("FALSE","The capital of France is"," Berlin","The capital of France is Berlin, a famous European city."),
        ("NOVEL","The tallest tower in Vashmere is called"," Kelmoor","The tallest tower in Vashmere is called Kelmoor Spire.")]
if __name__=="__main__":
    print("ACCEPTANCE 4 END-TO-END: does verification stop a FALSE fact from being made permanent?",flush=True)
    print(f"  {'claim':>6s} {'verdict':>8s} {'consolidated?':>14s} {'logP before':>12s} {'logP after':>11s} {'installed':>10s} {'gen damage':>11s}",flush=True)
    for lab,ctx,ans,text in CLAIMS:
        m,cfg=load(); v,ev=VF.verify(ctx,ans)
        b=logp(m,ctx,ans); g0=general(m,cfg)
        did = (v=="ACCEPT")
        if did: consolidate(m,text); m.eval()
        a=logp(m,ctx,ans); g1=general(m,cfg)
        print(f"  {lab:>6s} {v:>8s} {('YES' if did else 'no -- blocked'):>14s} {b:12.3f} {a:11.3f} {a-b:+10.3f} {g1-g0:+11.4f}",flush=True)
        del m
    print("\n  The FALSE claim must show 'no -- blocked' and ~0 change: it never reaches the weights.",flush=True)
