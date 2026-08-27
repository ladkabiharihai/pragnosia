"""HIPPOCAMPUS: the persistent fast-weight episodic store `M` -- the seat of CONTINUAL LEARNING AT INFERENCE.
Key realization: our layer's retrieval is already a fast-weight associative read,
    num_t = sum_{s<=t} (qf_t . kf_s) v_s ,  den_t = sum_{s<=t} (qf_t . kf_s)
so a PERSISTENT memory is just that same state carried ACROSS sequences:
    num_t += qf_t . M ,  den_t += qf_t . zM ,   M <- decay*M + g*(kf (x) v)
No new mechanism, no backprop: a gated Hebbian write (three-factor -- the gate g IS the neuromodulator) and a decay
that makes unverified traces fade (reversibility). A write-LOG keeps the raw episode so the slow path can verify it.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C
DEV=C.DEV

class Hippocampus:
    def __init__(s,m,layer=-1,decay=0.999):
        s.m=m; s.cfg=m.cfg; s.li=layer%len(m.layers); s.decay=decay
        s.M=torch.zeros(s.cfg.FE,s.cfg.DV,device=DEV); s.z=torch.zeros(s.cfg.FE,device=DEV); s.log=[]
    def _trace(s,X):
        """run the model up to the memory layer and return that layer's write traces (kf) and values (v)."""
        with torch.no_grad():
            a=s.m.emb(X)
            for L in s.m.layers[:s.li]: a,_=L(a)
            L=s.m.layers[s.li]; H,_=C.rmsnorm(a); Hs=C._shift(H)
            kf=C.taylor(Hs@L.Wk.weight.T,s.cfg); v=H@L.Wv.weight.T
            qf=C.taylor(H@L.Wq.weight.T,s.cfg)
        return kf,v,qf
    @torch.no_grad()
    def surprise(s,X):
        """the model's OWN prediction error per position -- the intrinsic novelty/salience signal (dopamine analog)."""
        lg=s.m(X[:,:-1]); lp=lg.log_softmax(-1)
        return -lp.gather(-1,X[:,1:].unsqueeze(-1)).squeeze(-1)          # [B,T-1] nats of surprise
    @torch.no_grad()
    def write(s,X,gate=None,source=None):
        """gated Hebbian write. `gate` per position (the propose gate); default = the model's own surprise."""
        kf,v,_=s._trace(X); B,T,_=kf.shape
        if gate is None:
            sp=s.surprise(X); g=torch.cat([sp[:,:1]*0,sp],1)             # align to positions
            g=(g/ (g.mean()+1e-6)).clamp(0,4)                            # relative surprise, bounded
        else: g=torch.as_tensor(gate,device=DEV).expand(B,T).float()
        s.M=s.decay*s.M+torch.einsum('btf,btd,bt->fd',kf,v,g)/max(1,B)
        s.z=s.decay*s.z+torch.einsum('btf,bt->f',kf,g)/max(1,B)
        if source is not None: s.log.append(source)
    @torch.no_grad()
    def read_logits(s,X,use_mem=True):
        """forward pass where the memory layer ALSO reads the persistent store (immediate provisional use)."""
        a=s.m.emb(X)
        for i,L in enumerate(s.m.layers):
            if i!=s.li: a,_=L(a); continue
            cfg=s.cfg; H,nrm=C.rmsnorm(a); Hs=C._shift(H)
            kf=C.taylor(Hs@L.Wk.weight.T,cfg); qf=C.taylor(H@L.Wq.weight.T,cfg); v=H@L.Wv.weight.T
            A=torch.einsum('btf,bsf->bts',qf,kf).tril()
            den=A.sum(-1,keepdim=True); num=torch.einsum('bts,bsd->btd',A,v)
            if use_mem:
                num=num+torch.einsum('btf,fd->btd',qf,s.M); den=den+torch.einsum('btf,f->bt',qf,s.z).unsqueeze(-1)
            a=a+L.rs*((num/(den+1e-5))@L.Wo.weight.T+L.Wo.bias)
        h,_=C.rmsnorm(a); return s.m.head(h)
    @torch.no_grad()
    def answer_logprob(s,ctx_ids,ans_ids,use_mem=True):
        """logprob the model assigns to `ans_ids` following `ctx_ids` -- the retention/recall measure."""
        ids=torch.tensor([ctx_ids+ans_ids],device=DEV)
        lg=s.read_logits(ids[:,:-1],use_mem=use_mem).log_softmax(-1)
        tgt=ids[0,1:]; n=len(ans_ids)
        return lg[0,-n:].gather(-1,tgt[-n:].unsqueeze(-1)).mean().item()
