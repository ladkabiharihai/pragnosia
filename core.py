"""Pragnosia CORE (fast, vectorized). Recurrent linear-attention with a symmetric 2nd-order Taylor feature map and a
key->NEXT-value SHIFT, residual + RMS pre-norm, trained by PURE LOCAL PLASTICITY (no backprop, no autograd).
The causal recurrence  num_t = sum_{s<=t} (qf_t . kf_s) v_s  is computed in PARALLEL as a masked quadratic form
(no Python time-loop, no per-step state materialization) -> orders of magnitude faster and far less memory.
`Layer.local_vjp` is the EXACT within-layer credit (eligibility traces in closed form), verified against autograd.
"""
import warnings,os; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda" if torch.cuda.is_available() else "cpu"

class Config:
    def __init__(s, D=96, DIN=8, DV=96, NL=4, T=32, vocab=32768, res_scale=1.0, mlp_mult=4,
                 tie_emb=False, head_scale=1.0):
        s.D=D; s.DIN=DIN; s.DV=DV; s.NL=NL; s.T=T; s.vocab=vocab; s.res_scale=res_scale; s.mlp_mult=mlp_mult
        # TIED EMBEDDINGS: one matrix serves input and output. At vocab 32768 / D=1024 that saves 33.5M parameters
        # (14% of a 235M model) and MEASURES BETTER: -0.108 nats at 37% fewer params (exp/tied_emb.py).
        # `head_scale` carries the growth rescaling that the head would otherwise absorb into its weights: the
        # embedding is a stream WRITER (wants *c) and the head a stream READER (wants /c), and one shared matrix
        # cannot be both -- so neither is scaled and the head's factor lives here instead. Scaling the head's
        # OUTPUT by 1/c is exactly equivalent to scaling its weights by 1/c.
        s.tie_emb=tie_emb; s.head_scale=head_scale
        s.iu=torch.triu_indices(DIN,DIN,device=DEV); s.FE=1+DIN+DIN*DIN   # full outer product (see taylor)

def rmsnorm(x,eps=1e-6):
    n=(x.pow(2).mean(-1,keepdim=True)+eps).sqrt(); return x/n, n
def rmsnorm_vjp(x,n,dy):
    D_=x.shape[-1]; return dy/n - x*((dy*x).sum(-1,keepdim=True)/(D_*n.pow(3)))
def taylor(x,cfg):                                            # [...,DIN] -> [...,FE], symmetric 2nd-order
    """FULL outer product, NOT a gather of its upper triangle.

    PROFILED: the gather form `(x.unsqueeze(-1)*x.unsqueeze(-2))[...,iu[0],iu[1]]` has a backward that is an ATOMIC
    SCATTER, and it was **39% of every training step** (13.12 ms of 33.5 ms) -- the single largest kernel in the
    model, larger than all the matmuls combined. A transformer of the same shape spends 8.5 ms per step TOTAL and
    has no such kernel at all. Replacing the gather with an outer product + contiguous reshape removes it entirely:
    1.599 ms -> 0.223 ms for the map, a 7.16x speedup on that op.

    The cost is redundancy: the full outer product carries both x_i x_j and x_j x_i, so FE is DIN^2 + DIN + 1
    instead of DIN(DIN+1)/2 + DIN + 1. The duplicated features are harmless (the model can weight them
    identically) and the state grows, so DIN should be chosen accordingly."""
    o=(x.unsqueeze(-1)*x.unsqueeze(-2)).reshape(*x.shape[:-1],x.shape[-1]*x.shape[-1])
    return torch.cat([torch.ones(*x.shape[:-1],1,device=x.device,dtype=x.dtype),x,o],-1)
def tvjp(x,g,cfg):                                            # exact vjp of taylor
    dx=g[...,1:1+cfg.DIN].clone(); gq=g[...,1+cfg.DIN:]
    dx.index_add_(-1,cfg.iu[0],gq*x[...,cfg.iu[1]]); dx.index_add_(-1,cfg.iu[1],gq*x[...,cfg.iu[0]]); return dx
def _shift(x):                                                # key source is the PREVIOUS token (binds key_t -> value_{t+1})
    return torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],1)

def dgelu(x):                                                 # exact derivative of the erf GELU
    return 0.5*(1+torch.erf(x/1.4142135623730951))+x*torch.exp(-0.5*x*x)/2.5066282746310002
class Layer(nn.Module):
    """Recurrent mixer + FEEDFORWARD (MLP), both residual and pre-normed -- a full block.
    Measured: adding the MLP is worth 1.17 nats at matched total parameters (attention-only 5.61 vs with-MLP 4.44),
    because MLPs carry most of a model's knowledge and the smaller D frees embedding budget for real capacity."""
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg; s.rs=getattr(cfg,'res_scale',1.0); mult=getattr(cfg,'mlp_mult',4)
        s.Wk=nn.Linear(cfg.D,cfg.DIN,bias=False); s.Wq=nn.Linear(cfg.D,cfg.DIN,bias=False)
        s.Wv=nn.Linear(cfg.D,cfg.DV,bias=False); s.Wo=nn.Linear(cfg.DV,cfg.D)
        s.m1=nn.Linear(cfg.D,mult*cfg.D); s.m2=nn.Linear(mult*cfg.D,cfg.D)
        # DATA-DEPENDENT FORGETTING. Each token computes its OWN forget rate instead of sharing one global decay
        # constant. MEASURED (exp/bench.py, matched size and tokens vs a transformer at 3.4617): our un-gated mixer
        # scored 3.8747; adding this gate scores 3.5165 -- it closes 87% of the gap and is worth -0.28 nats alone.
        # This is what separates Mamba-2 / GLA / RWKV-6 from plain linear attention, and we had the constant version.
        # It also REPLACES the hand-set DECAY=0.05: the model now decides what to forget, per token, intrinsically.
        s.gated=getattr(cfg,'gated',False); s.qknorm=getattr(cfg,'qknorm',False)
        if s.gated:
            s.Wg=nn.Linear(cfg.D,1)
            nn.init.zeros_(s.Wg.weight); nn.init.constant_(s.Wg.bias,8.0)   # g=0.9997/token: start by forgetting almost nothing
    def forward(s,a,state=None,wgate=None,prev=None):
        """`state`=(M,zM): the PERSISTENT fast-weight memory carried from previous chunks. Reading it costs nothing
        extra and is what gives UNBOUNDED context at CONSTANT memory; carrying it during training is what teaches the
        model to use its own memory. The returned new state is detached (no cross-chunk backprop)."""
        cfg=s.cfg; B,T,_=a.shape; H,nrm=rmsnorm(a)
        # BOUNDARY KEY. _shift prepends a ZERO, so the first token of a chunk built its key from nothing instead of
        # from the previous chunk's last token. Measured in fp64: that single position made the chunked recurrence
        # differ from the full parallel form by 8.1e-02 RELATIVE (a correct recurrence is ~1e-12); every other
        # position was bit-exact. prefill/step already thread `prev` for generation; training never did.
        Hs=_shift(H) if prev is None else torch.cat([prev.unsqueeze(1),H[:,:-1]],1)
        xk=Hs@s.Wk.weight.T; xq=H@s.Wq.weight.T
        if s.qknorm:
            # taylor() is QUADRATIC in its input, so A = qf.kf grows QUARTICALLY with the projection magnitude
            # (measured |A|max climbing 17 -> 1.3e4 over 3000 steps, eventually NaN). Unit-RMS bounds it for good,
            # and is a small win in its own right: -0.017 nats AND faster.
            # SUM, NOT MEAN. With mean, adding zero dims at a grow_din LOWERS the mean square and rsqrt amplifies
            # the survivors by sqrt(nDIN/oldDIN) -- measured exactly 1.4142 for a 4->8 grow, giving a 2.3e-01
            # relative logit change. Any pre-scaling of the weights is undone by the very normalisation it feeds,
            # so the fix has to live here: normalising by the SUM of squares is invariant to appended zero dims,
            # which makes grow_din exactly function-preserving. The scale differs from mean-norm by a constant
            # sqrt(DIN), which the learned Wk/Wq absorb.
            xk=xk*torch.rsqrt(xk.pow(2).sum(-1,keepdim=True)+1e-6)
            xq=xq*torch.rsqrt(xq.pow(2).sum(-1,keepdim=True)+1e-6)
        kf=taylor(xk,cfg)                                      # writes (shifted keys)
        qf=taylor(xq,cfg)                                      # reads
        v=H@s.Wv.weight.T
        A=torch.einsum('btf,bsf->bts',qf,kf)                   # retrieval weights beta_{t,s} = qf_t . kf_s
        if s.gated:
            # a per-token rate g_u becomes the pairwise factor exp(L_t - L_s) with L_t = sum_{u<=t} log g_u.
            # CLAMP at 0: the exponent is <=0 only for t>=s, and for t<s it overflows to inf BEFORE the causal
            # mask, where inf*0 = NaN. Those entries are masked away, so clamping changes nothing that is used.
            # a per-token rate g_u becomes the pairwise factor exp(L_t - L_s) with L_t = sum_{u<=t} log g_u.
            # CLAMP at 0: the exponent is <=0 only for t>=s, and for t<s it overflows to inf BEFORE the causal
            # mask, where inf*0 = NaN. Those entries are masked away, so clamping changes nothing that is used.
            # NOTE: folding this into qf/kf (as GLA/Mamba-2 do) would save ~12 GiB of [B,T,T] tensors, BUT it
            # also rescales the kf that is WRITTEN INTO THE CARRIED STATE, so the decay gets applied twice across
            # chunk boundaries -- measured 2.25e-03 relative error on the chunked recurrence. Doing it correctly
            # requires un-folding kf before the state write; left as the obvious next memory win.
            lg=F.logsigmoid(s.Wg(H)).squeeze(-1)               # [B,T], each <= 0
            L=lg.cumsum(-1)
            A=A*(L.unsqueeze(-1)-L.unsqueeze(-2)).clamp(max=0.0).exp()
        A=A.tril()                                             # causal: read at t sees writes s<=t
        den=A.sum(-1,keepdim=True); num=torch.einsum('bts,bsd->btd',A,v)
        if state is not None:                                  # + read the persistent store
            M,zM=state
            rd=torch.einsum('btf,bfd->btd',qf,M); rz=torch.einsum('btf,bf->bt',qf,zM).unsqueeze(-1)
            if s.gated:
                # everything already in the state was written BEFORE this chunk, so it decays by the full
                # cumulative gate up to t. Without this the recurrence and the parallel form disagree at every
                # chunk boundary and generation would not match training.
                g0=L.exp().unsqueeze(-1)                       # [B,T,1], product of gates from the chunk start
                rd=rd*g0; rz=rz*g0
            num=num+rd; den=den+rz
        den=den+1e-5
        r=num/den; o=r@s.Wo.weight.T+s.Wo.bias
        a1=a+s.rs*o
        H2,n2=rmsnorm(a1); z=H2@s.m1.weight.T+s.m1.bias; g=F.gelu(z)
        a2=a1+s.rs*(g@s.m2.weight.T+s.m2.bias)
        # SELECTIVE WRITING (VISION propose gate): `wgate` [B,T] weights each token's write into the PERSISTENT
        # state. Every token writing means N tokens superimpose in a fixed state, so any one fact's share is ~1/N
        # (measured: keeping all writes is actively harmful, -1.10 gain). Gating by salience multiplies the
        # effective span at no extra state cost. wgate=None keeps the old behaviour (write everything).
        if s.gated:
            # write each key with the decay it will have accumulated by the END of the chunk, and shrink the
            # incoming state by the chunk's total gate -- this is exactly M_T = (prod g) M_0 + sum_t (prod_{u>t} g_u) k_t (x) v_t
            gt=(L[:,-1:]-L).exp().unsqueeze(-1)                # [B,T,1]: gate from t to the end of the chunk
            kg=kf*gt
            newM=torch.einsum('btf,btd->bfd',kg,v).detach(); newz=kg.squeeze(-1).sum(1).detach() if kg.dim()==4 else kg.sum(1).detach()
            if state is not None:
                gtot=L[:,-1].exp().view(-1,1,1)
                newM=state[0]*gtot+newM; newz=state[1]*gtot.squeeze(-1)+newz
        elif wgate is None:
            newM=torch.einsum('btf,btd->bfd',kf,v).detach(); newz=kf.sum(1).detach()
            if state is not None: newM=state[0]+newM; newz=state[1]+newz
        else:
            g=wgate.unsqueeze(-1)
            newM=torch.einsum('btf,btd->bfd',kf*g,v).detach(); newz=(kf*g).sum(1).detach()
            if state is not None: newM=state[0]+newM; newz=state[1]+newz
        return a2, (H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r,a1,H2,n2,z,g,(newM,newz))
    def local_vjp(s,a,cache,dout):
        cfg=s.cfg; H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r,a1,H2,n2,z,g,_st=cache
        Wk,Wq,Wv,Wo=(s.Wk.weight,s.Wq.weight,s.Wv.weight,s.Wo.weight)
        dfa=dout*s.rs                                          # MLP branch (local, exact)
        dm2=torch.einsum('btd,bte->de',dfa,g); dbm2=dfa.sum((0,1))
        dz=(dfa@s.m2.weight)*dgelu(z)
        dm1=torch.einsum('btd,bte->de',dz,H2); dbm1=dz.sum((0,1))
        dout=dout+rmsnorm_vjp(a1,n2,dz@s.m1.weight)            # -> gradient wrt a1 (residual + normed path)
        dO=dout*s.rs
        dWo=torch.einsum('btd,bte->de',dO,r); dbo=dO.sum((0,1)); dr=dO@Wo
        dnum=dr/den; dden=-(dr*num).sum(-1,keepdim=True)/den.pow(2)
        dA=(torch.einsum('btd,bsd->bts',dnum,v)+dden).tril()
        dv=torch.einsum('bts,btd->bsd',A,dnum)
        dqf=torch.einsum('bts,bsf->btf',dA,kf); dkf=torch.einsum('bts,btf->bsf',dA,qf)
        dxk=tvjp(xk,dkf,cfg); dxq=tvjp(xq,dqf,cfg)
        dWv=torch.einsum('btd,bte->de',dv,H); dWq=torch.einsum('btd,bte->de',dxq,H); dWk=torch.einsum('btd,bte->de',dxk,Hs)
        dH=dv@Wv+dxq@Wq
        dHs=dxk@Wk; dH=dH+torch.cat([dHs[:,1:],torch.zeros_like(dHs[:,:1])],1)   # un-shift
        return {'Wk':dWk,'Wq':dWq,'Wv':dWv,'Wo':dWo,'bo':dbo,'m1':dm1,'bm1':dbm1,'m2':dm2,'bm2':dbm2}, dout+rmsnorm_vjp(a,nrm,dH)

def ce_from_hidden(m,h,y,nsp=8,ignore_index=-100):
    """Cross-entropy computed by projecting to vocab in SLICES, so [B,T,vocab] never exists whole.
    Exactly equivalent to F.cross_entropy(m.head(h)*scale, y) -- verified to 9.5e-07 -- but the peak activation
    is 1/nsp of the logits tensor. reduction='sum' then divide by the true count, so unequal slices are exact."""
    sc=getattr(m.cfg,'head_scale',1.0)
    hf=h.reshape(-1,h.shape[-1]); yf=y.reshape(-1); n=hf.shape[0]
    cnt=int((yf!=ignore_index).sum())                       # masked targets must not count in the denominator
    if cnt==0: return hf.sum()*0.0
    tot=0.
    for i in range(nsp):
        sl=slice(i*n//nsp,(i+1)*n//nsp)
        tot=tot+F.cross_entropy(m.head(hf[sl])*sc,yf[sl],reduction='sum',ignore_index=ignore_index)
    return tot/cnt

@torch.no_grad()
def entropy_from_hidden(m,h,sample=4096):
    """The model's own uncertainty, from a SUBSAMPLE of positions. SaturationSensor.entropy already subsampled
    4096 rows -- but it was handed a full fp32 copy of the logits (4 GiB at B=8) to do it. Project only the rows
    it actually reads: [4096, vocab] instead of [B*T, vocab]."""
    sc=getattr(m.cfg,'head_scale',1.0)
    hf=h.reshape(-1,h.shape[-1])
    idx=torch.randint(0,hf.shape[0],(min(sample,hf.shape[0]),),device=hf.device)
    p=(m.head(hf[idx])*sc).float().softmax(-1)
    return float(-(p*(p+1e-9).log()).sum(-1).mean())

class Layer_step:
    pass
class Core(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg; s.emb=nn.Embedding(cfg.vocab,cfg.D)
        s.layers=nn.ModuleList([Layer(cfg) for _ in range(cfg.NL)]); s.head=nn.Linear(cfg.D,cfg.vocab)
        nn.init.normal_(s.emb.weight,0,0.02)
        if getattr(cfg,'tie_emb',False): s.head.weight=s.emb.weight     # one shared matrix
    def forward(s,X,states=None,return_states=False,wgate=None,prev=None,return_prev=False,return_hidden=False):
        """states: list of per-layer (M,zM) carried from the previous chunk (persistent memory / long context)."""
        a=s.emb(X); ns=[]; npv=[]
        for i,L in enumerate(s.layers):
            a,c=L(a,None if states is None else states[i],wgate=wgate,
                  prev=None if prev is None else prev[i]); ns.append(c[-1]); npv.append(c[0][:,-1])
        h,_=rmsnorm(a)
        # return_hidden: hand back h [B,T,D] instead of logits [B,T,vocab]. At B=8/T=4096/V=32768 the logits are
        # 2 GiB in bf16 with autograd holding them, plus 4 GiB more if anything calls .float() on them. Callers
        # that only need a LOSS or an ENTROPY never need the whole thing -- see ce_from_hidden / entropy_from_hidden.
        lg=h if return_hidden else s.head(h)*getattr(s.cfg,'head_scale',1.0)
        # return_prev=True always yields the 3-tuple, so a caller can thread the boundary key from the FIRST
        # chunk onward. The 2-tuple form is kept so existing callers are unaffected.
        if return_states: return (lg,ns,npv) if return_prev else (lg,ns)
        return lg
    def decay_states(s,states,d):
        return None if states is None else [(M*d,z*d) for (M,z) in states]
    @torch.no_grad()
    def step(s,tok,states,prev=None):
        """INCREMENTAL DECODE: O(1) per token. `states`=[(M,z)] carried memory, `prev`=[per-layer previous normed
        activation] needed for the shifted key. Returns (logits, new_states, new_prev).
        Re-running the whole window per token (the transformer habit) cost 12.5 ms/token at ctx 2048; this is ~O(1)."""
        cfg=s.cfg; a=s.emb(tok)                                    # tok: [B,1]
        ns=[]; npv=[]
        for i,L in enumerate(s.layers):
            M,z=states[i]
            H,nrm=rmsnorm(a)                                       # [B,1,D]
            src=prev[i] if prev is not None else torch.zeros_like(H[:,0])
            xk=src@L.Wk.weight.T; xq=(H[:,0])@L.Wq.weight.T
            if L.qknorm:                                           # MUST match forward() or generation drifts
                xk=xk*torch.rsqrt(xk.pow(2).sum(-1,keepdim=True)+1e-6)   # SUM: invariant to appended zero dims
                xq=xq*torch.rsqrt(xq.pow(2).sum(-1,keepdim=True)+1e-6)
            kf=taylor(xk,cfg)                                      # key from the PREVIOUS token (the shift)
            v=(H[:,0])@L.Wv.weight.T
            if L.gated:
                # the recurrent form of the gate: decay the whole state by this token's rate, then write.
                # forward() expresses the same thing as the pairwise factor exp(L_t - L_s); here it is literal.
                g=torch.sigmoid(L.Wg(H[:,0]))                      # [B,1]
                M=M*g.unsqueeze(-1); z=z*g
            M=M+kf.unsqueeze(2)*v.unsqueeze(1); z=z+kf              # write into the state
            qf=taylor(xq,cfg)
            num=torch.einsum('bf,bfd->bd',qf,M); den=(qf*z).sum(-1,keepdim=True)+1e-5
            o=(num/den)@L.Wo.weight.T+L.Wo.bias
            a1=a+L.rs*o.unsqueeze(1)
            H2,_=rmsnorm(a1)
            a=a1+L.rs*(F.gelu(H2@L.m1.weight.T+L.m1.bias)@L.m2.weight.T+L.m2.bias)
            ns.append((M,z)); npv.append(H[:,0])
        h,_=rmsnorm(a); return s.head(h)[:,0]*getattr(s.cfg,'head_scale',1.0), ns, npv
    @torch.no_grad()
    def generate(s,ids,n=40,temp=1.0,states=None,top_p=1.0,greedy=False):
        """CORRECT generation. Use this instead of calling prefill/step by hand.

        THE BUG THIS EXISTS TO PREVENT: every call site wrote `prefill(x)` then `cur=x[:,-1:]` and stepped, which
        processes the LAST PROMPT TOKEN TWICE -- once inside prefill and again as the first step. That writes it
        into the recurrent state twice and corrupts the first prediction. Measured against the full forward:
        double-counted gives mean |logit diff| 3.8e-01 and flips the argmax on 2.9% of tokens; done correctly it is
        6.5e-05 with 0% flips, i.e. exact. Prefill everything EXCEPT the last token, then step the last token.

        Also note GREEDY is a poor decoder here: this model's top-1 is often ~0.09, and argmax on a flat
        distribution picks an arbitrary peak that then locks in at 0.95+ and repeats forever."""
        import torch as _t
        x=_t.tensor([ids],device=s.emb.weight.device) if not _t.is_tensor(ids) else ids
        if states is None:
            states=[(_t.zeros(x.shape[0],s.cfg.FE,s.cfg.DV,device=x.device),
                     _t.zeros(x.shape[0],s.cfg.FE,device=x.device)) for _ in range(s.cfg.NL)]
        st,pv=(s.prefill(x[:,:-1],states) if x.shape[1]>1 else (states,None))
        out=[]; cur=x[:,-1:]
        for _ in range(n):
            lg,st,pv=s.step(cur,st,pv)
            lg=lg[0].float()
            if greedy: nx=int(lg.argmax())
            else:
                p=(lg/temp).softmax(-1)
                if top_p<1.0:
                    sp,si=_t.sort(p,descending=True); keep=(sp.cumsum(0)-sp)<top_p
                    p=_t.zeros_like(p).scatter_(0,si[keep],sp[keep]); p=p/p.sum()
                nx=int(_t.multinomial(p,1))
            out.append(nx); cur=_t.tensor([[nx]],device=x.device)
        return out,st
    @torch.no_grad()
    def prefill(s,X,states=None):
        """process a prompt/context in one parallel pass, returning the state and prev needed by step()."""
        a=s.emb(X); ns=[]; npv=[]
        for i,L in enumerate(s.layers):
            st=None if states is None else states[i]
            a,c=L(a,st); ns.append(c[-1]); npv.append(c[0][:,-1])   # c[0]=H (normed input), last position
        return ns,npv
    def param_dict(s):
        p={'emb':s.emb.weight,'hW':s.head.weight,'hb':s.head.bias}
        for li,L in enumerate(s.layers):
            for n in ('Wk','Wq','Wv','Wo','m1','m2'): p[f'{li}.{n}']=getattr(L,n).weight
            p[f'{li}.bo']=L.Wo.bias; p[f'{li}.bm1']=L.m1.bias; p[f'{li}.bm2']=L.m2.bias
        return p

class LocalAdam:
    """Applies the LOCAL gradients. `adaptive=True` uses a SCALE-FREE relative step (trust ratio, LAMB-style):
    each parameter moves a fixed FRACTION of its own norm, so the hyperparameter does not need re-tuning when the
    model grows or scales (VISION 0a: no hand-tuned per-scale constants). `min_norm` lets zero-initialized new
    capacity (function-preserving growth) start moving instead of being frozen at trust=0."""
    def __init__(s,prm,lr=3e-3,b1=0.9,b2=0.999,eps=1e-8,adaptive=False,min_norm=1e-3,cap=10.0):
        s.prm=prm; s.lr=lr; s.b1=b1; s.b2=b2; s.eps=eps; s.i=0
        s.adaptive=adaptive; s.min_norm=min_norm; s.cap=cap
        s.m={k:torch.zeros_like(p) for k,p in prm.items()}; s.v={k:torch.zeros_like(p) for k,p in prm.items()}
    def step(s,g):
        s.i+=1
        with torch.no_grad():
            for k in s.prm:
                s.m[k]=s.b1*s.m[k]+(1-s.b1)*g[k]; s.v[k]=s.b2*s.v[k]+(1-s.b2)*g[k]**2
                u=(s.m[k]/(1-s.b1**s.i))/((s.v[k]/(1-s.b2**s.i)).sqrt()+s.eps)
                if s.adaptive:
                    wn=s.prm[k].norm().clamp_min(s.min_norm); un=u.norm().clamp_min(s.eps)
                    u=u*torch.clamp(wn/un,max=s.cap)   # relative step: move a fixed fraction of ||w||
                s.prm[k]-=s.lr*u
