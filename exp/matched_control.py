"""THE DECISIVE TEST: is our architecture WORSE at language, or just undertrained?
Train a standard TRANSFORMER on the EXACT same data, tokenizer, context and token budget as our own run reached
at step 10000 (49M params, 2.6B tokens, fixed-eval loss 5.0420), then compare on the SAME fixed eval set.
  - transformer better  -> our recurrent architecture pays a real language-modelling tax
  - comparable/ours better -> we are simply undertrained, and the grind is the answer
Runs ALONGSIDE training (small, ~8GB) so the main run is not stopped."""
import sys,time,os; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log,pbar; setup_log("matched_control")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
DEV=C.DEV; V=32768; CH=2048; B=4; ACC=2          # effective batch 8, same as ours
TARGET_TOK=2.6e9
class Blk(nn.Module):
    def __init__(s,d,h=8):
        super().__init__(); s.h=h
        s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d,bias=False); s.o=nn.Linear(d,d,bias=False)
        s.n2=nn.LayerNorm(d); s.m1=nn.Linear(d,4*d); s.m2=nn.Linear(4*d,d)
    def forward(s,x):
        Bs,T,D=x.shape; q,k,v=s.qkv(s.n1(x)).chunk(3,-1)
        q,k,v=[t.view(Bs,T,s.h,D//s.h).transpose(1,2) for t in (q,k,v)]
        x=x+s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(Bs,T,D))
        return x+s.m2(F.gelu(s.m1(s.n2(x))))
class TLM(nn.Module):
    def __init__(s,d=512,nl=6):
        super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(CH,d)
        s.bl=nn.ModuleList([Blk(d) for _ in range(nl)]); s.nf=nn.LayerNorm(d); s.head=nn.Linear(d,V,bias=False)
        s.head.weight=s.emb.weight                              # tied, like most small LMs
        # BUG FIX: PyTorch's nn.Embedding defaults to N(0,1) -- 50x larger than the 0.02 our own Core uses. With the
        # default, the positional embedding SWAMPS the token signal and the model stalls at the unigram floor
        # (measured: dead flat at loss 7.47 from 0.29B to 0.52B tokens). Match Core's init so the comparison is fair.
        nn.init.normal_(s.emb.weight,0,0.02); nn.init.normal_(s.pos.weight,0,0.02)
    def forward(s,X):
        x=s.emb(X)+s.pos(torch.arange(X.shape[1],device=X.device))[None]
        for b in s.bl: x=b(x)
        return s.head(s.nf(x))
if __name__=="__main__":
    torch.manual_seed(0); np.random.seed(0)
    m=TLM(512,6).to(DEV)
    P=sum(p.numel() for p in m.parameters())
    print(f"TRANSFORMER CONTROL: D=512 x 6L, {P/1e6:.1f}M params (ours at the comparison point: 49.4M)",flush=True)
    print(f"  same tokenizer (32k), same corpus (window2), same ctx {CH}, target {TARGET_TOK/1e9:.1f}B tokens",flush=True)
    dm=DataMix(ctx=CH+1,verbose=False,only='window2')
    # IDENTICAL fixed eval set to evaluate.py (same seed/procedure) so the numbers are directly comparable
    np.random.seed(1234); EV=[torch.tensor(dm.batch(4),device=DEV) for _ in range(12)]
    dec=[q for n,q in m.named_parameters() if q.dim()>=2]        # weight matrices get weight decay
    nod=[q for n,q in m.named_parameters() if q.dim()<2]         # norms/biases do not (standard GPT practice)
    opt=torch.optim.AdamW([{'params':dec,'weight_decay':0.1},{'params':nod,'weight_decay':0.0}],
                          lr=6e-4,betas=(0.9,0.95))
    steps=int(TARGET_TOK/(CH*B*ACC)); warm=300
    def lr_at(i):
        if i<warm: return 6e-4*(i+1)/warm
        f=(i-warm)/max(1,steps-warm); return 6e-4*(0.1+0.9*0.5*(1+np.cos(np.pi*min(1,f))))
    @torch.no_grad()
    def ev():
        m.eval(); t=0
        for X in EV:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                t+=F.cross_entropy(m(X[:,:CH]).reshape(-1,V).float(),X[:,1:CH+1].reshape(-1)).item()
        m.train(); return t/len(EV)
    print(f"  {steps} steps of {B}x{ACC} x {CH} tokens",flush=True)
    bar=pbar(range(steps),"tf_control",total=steps); ema=None; t0=time.time()
    for it in bar:
        for g in opt.param_groups: g['lr']=lr_at(it)
        opt.zero_grad()
        for _ in range(ACC):
            X=torch.tensor(dm.batch(B),device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                l=F.cross_entropy(m(X[:,:CH]).reshape(-1,V),X[:,1:CH+1].reshape(-1))/ACC
            l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        v=l.item()*ACC; ema=v if ema is None else .995*ema+.005*v
        if (it+1)%100==0:
            tps=(it+1)*B*ACC*CH/(time.time()-t0)
            bar.set_postfix_str(f"loss={ema:.3f} ppl={np.exp(min(20,ema)):.0f} {tps/1e3:.0f}ktok/s")
        if (it+1)%2000==0 or it+1==steps:
            e=ev(); tok=(it+1)*B*ACC*CH
            bar.write(f"    [{tok/1e9:.2f}B tokens] TRANSFORMER fixed-eval loss = {e:.4f} (ppl {np.exp(e):.1f})")
            torch.save({'model':m.state_dict(),'step':it+1,'eval':e},"/mnt/kv_cache/pragnosia_data/tf_control.pt")
    print(f"\nFINAL: transformer {ev():.4f}   |   OURS at 2.6B tokens/49M = 5.0420",flush=True)
