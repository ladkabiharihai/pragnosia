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
def _sub_mix(qf,kf,v,L,state,b,xq=None,xk=None,wgate=None):
    """SUB-CHUNKED LINEAR ATTENTION (the GLA/Mamba-2 formulation), mathematically identical to the full quadratic
    form but O(T x b x FE) instead of O(T^2 x FE), and it never materialises [B,T,T].

    Split the chunk into blocks of b. Carry a running (M,z) BETWEEN blocks and use the quadratic form only WITHIN
    one. A read at t of a key at s needs exp(L_t - L_s); splitting at a block boundary p gives
        exp(L_t - L_s) = exp(L_t - L_p) * exp(L_p - L_s)
    so the left factor is computed locally and the right one is already folded into what M holds. Every exponent
    is then RELATIVE TO A BLOCK BOUNDARY and therefore <= 0 -- which is exactly why this is numerically safe where
    folding the gate into qf/kf globally was not (that produced exp(L_t - L_T) >= 1, which overflows for a
    strongly-forgetting gate; measured as the 2.25e-03 error that killed the earlier attempt).

    The running M is NOT detached inside the loop: gradients must flow from a later block back through M to an
    earlier one, exactly as they do through the full A. Only the returned state is detached.
    """
    B,T,FE=qf.shape; DV=v.shape[-1]; nb=T//b
    qb=qf.view(B,nb,b,FE); kb=kf.view(B,nb,b,FE); vb=v.view(B,nb,b,DV)
    # THE SAME IDENTITY AS _full_mix: within a block, A = qf.kf = 1 + (xq.xk) + (xq.xk)^2, so the pairwise term
    # never needs the FE-dim features. I optimised _full_mix and left this path on the slow form -- which is the
    # path that matters MORE, because it is the one with linear memory. The features are still used for the
    # cross-block state read/write, which is linear in T.
    DIN=None if xq is None else xq.shape[-1]
    if xq is not None:
        xqb=xq.view(B,nb,b,DIN); xkb=xk.view(B,nb,b,DIN)
    if wgate is not None: wgb=wgate.view(B,nb,b)
    if L is not None:
        Lb=L.view(B,nb,b)
        last=Lb[:,:,-1]
        Lstart=torch.cat([torch.zeros(B,1,device=L.device,dtype=L.dtype),last[:,:-1]],1)
        Lc=Lb-Lstart.unsqueeze(-1)                      # <= 0 within every block
    if state is None:
        M=qf.new_zeros(B,FE,DV); z=qf.new_zeros(B,FE)
    else:
        M,z=state
    ns=[]; ds=[]
    for i in range(nb):
        q=qb[:,i]; k=kb[:,i]; vv=vb[:,i]
        if xq is not None:
            c=torch.einsum('btd,bsd->bts',xqb[:,i],xkb[:,i]); A=1+c+c*c
        else:
            A=torch.einsum('btf,bsf->bts',q,k)
        if wgate is not None: A=A*wgb[:,i].unsqueeze(-2)   # selective write, same weight as the state write
        if L is not None:
            lc=Lc[:,i]
            A=A*(lc.unsqueeze(-1)-lc.unsqueeze(-2)).clamp(max=0.0).exp()
        A=A.tril()
        num=torch.einsum('bts,bsd->btd',A,vv); den=A.sum(-1,keepdim=True)
        rd=torch.einsum('btf,bfd->btd',q,M); rz=torch.einsum('btf,bf->bt',q,z).unsqueeze(-1)
        if L is not None:
            g0=lc.exp().unsqueeze(-1); rd=rd*g0; rz=rz*g0
        ns.append(num+rd); ds.append(den+rz)
        if L is not None:
            ge=(lc[:,-1:]-lc).exp().unsqueeze(-1)       # decay each key to the END of this block
            kg=k*ge
            if wgate is not None: kg=kg*wgb[:,i].unsqueeze(-1)
            tot=lc[:,-1].exp().view(B,1,1)
            M=M*tot+torch.einsum('btf,btd->bfd',kg,vv); z=z*tot.squeeze(-1)+kg.sum(1)
        else:
            kk=k if wgate is None else k*wgb[:,i].unsqueeze(-1)
            M=M+torch.einsum('btf,btd->bfd',kk,vv); z=z+kk.sum(1)
    return torch.cat(ns,1),torch.cat(ds,1),(M.detach(),z.detach())


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
        # READ SHARPNESS. taylor() of a UNIT-NORM vector makes the retrieval weight a polynomial in the cosine
        # alone: A = qf.kf = 1 + c + c^2 with c = cos(q,k) in [-1,1]. So a PERFECT key match scores 3 and an
        # orthogonal one scores 1 -- the selectivity of this kernel is HARD-CAPPED AT 3:1, no matter what Wq/Wk
        # learn. At L=256 the right key therefore gets 3/(3+255) = 1.2% of the read and is drowned by the other
        # 255. That is a closed-form explanation for induction sitting at exactly 0%, and it is a property of
        # qk-norm, not of gating: softmax attention has no such ceiling because exp() is unbounded.
        # Scaling the normalised vectors by s gives A = 1 + s^2 c + s^4 c^2, ratio (1+s^2+s^4):1 -- s=3 is 91:1.
        # LEARNED PER LAYER, so the model chooses its own sharpness and can keep some layers soft for generation
        # and make others sharp for retrieval, which is how transformers get specialised induction heads.
        # Stability is unaffected: the DIRECTION is still normalised, so |A| is bounded by 1+s^2+s^4 with s a
        # single learned scalar, and the sum-of-squares norm keeps grow_din exactly function-preserving.
        # PER-TOKEN sharpness goes one step further, and is the only version that can resolve the tension rather
        # than trade it: a sharp read is what finds a needle and is ALSO what turns generation into an attractor,
        # so a single per-layer value has to compromise. Letting each token set its own lets the model sharpen on
        # the tokens that are doing retrieval and stay soft on the ones writing prose. Same clamp, so |A| is
        # bounded identically; init at zero weights so it starts at exactly s=1 like every other arm.
        # SELECTIVE WRITING, the vision's propose gate. `wgate` has existed since the start but is only consulted
        # in the UNGATED branch, so with gating on -- every shipped configuration -- it has never once run.
        # WHY IT IS THE HORIZON LEVER. The state is fixed size and EVERY token writes, so one fact's share after
        # N tokens is ~1/N. The learned forget half-life came out at 1847 and 1929 tokens in the two horizon arms
        # and retrieval dies at ~2000 in both -- the gate is not short-sighted, it is trading horizon against
        # interference, and holding memories longer would only pile up more of them. Writing selectively attacks
        # the interference instead, which is the term that forces the trade.
        # Init bias +4 (sigmoid 0.982, near write-everything) so this starts as the current model does.
        s.sel=getattr(cfg,'sel_write',False)
        if s.sel:
            s.Wsal=nn.Linear(cfg.D,1); nn.init.zeros_(s.Wsal.weight); nn.init.constant_(s.Wsal.bias,4.0)
        # KEY/QUERY CENTERING. Measured on the finished 125M: hidden states are extremely anisotropic (mean
        # pairwise cosine 0.976 in early layers), Wk inherits it, and the keys end up clustered around a MEAN
        # direction -- uncentered rank ~6 of 32 against ~20 centered. So |cos(q,k)| averages 0.30 instead of ~0,
        # and the 3:1 algebraic ceiling binds at 1.45x instead of near 3.00x: the model runs at 48% of its own
        # capped ceiling. Removing the mean measured 1.35x more selectivity (1.45 -> 1.97) for zero parameters
        # and zero FLOPs. A RUNNING mean (not a batch mean) so forward and step compute the same function.
        s.qk_center=getattr(cfg,'qk_center',False)
        if s.qk_center:
            s.register_buffer('k_mu',torch.zeros(cfg.DIN))
            s.register_buffer('q_mu',torch.zeros(cfg.DIN))
        s.qk_learn=getattr(cfg,'qk_learn_scale',False); s.qk_tok=getattr(cfg,'qk_token_scale',False)
        if s.qk_tok:
            s.Wqs=nn.Linear(cfg.D,1); nn.init.zeros_(s.Wqs.weight); nn.init.zeros_(s.Wqs.bias)
        elif s.qk_learn: s.qk_ls=nn.Parameter(torch.zeros(1))
    def qk_s(s,H=None):
        cl=getattr(s.cfg,'qk_clamp',2.0)                   # s in [exp(-cl), exp(cl)]; bounds |A| at 1+s^2+s^4
        if s.qk_tok:
            if H is None: return 1.0
            return s.Wqs(H).clamp(-cl,cl).exp()             # [B,T,1], broadcasts over DIN
        if s.qk_learn: return s.qk_ls.clamp(-cl,cl).exp()
        return getattr(s.cfg,'qk_scale',1.0)
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
        if s.qk_center:
            if s.training:
                with torch.no_grad():
                    s.k_mu.mul_(0.99).add_(xk.detach().float().mean((0,1)),alpha=0.01)
                    s.q_mu.mul_(0.99).add_(xq.detach().float().mean((0,1)),alpha=0.01)
            xk=xk-s.k_mu.to(xk.dtype); xq=xq-s.q_mu.to(xq.dtype)
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
            xk=xk*torch.rsqrt(xk.pow(2).sum(-1,keepdim=True)+1e-6)*s.qk_s(Hs)
            xq=xq*torch.rsqrt(xq.pow(2).sum(-1,keepdim=True)+1e-6)*s.qk_s(H)
        kf=taylor(xk,cfg)                                      # writes (shifted keys)
        qf=taylor(xq,cfg)                                      # reads
        v=H@s.Wv.weight.T
        # gate cumulant, needed by BOTH paths
        L=F.logsigmoid(s.Wg(H)).squeeze(-1).cumsum(-1) if s.gated else None
        if s.sel and wgate is None: wgate=torch.sigmoid(s.Wsal(H)).squeeze(-1)   # [B,T] write strength per token
        sub=getattr(cfg,'subchunk',0)
        if sub and T%sub==0:
            # SUB-CHUNKED PATH: same function, O(T x sub x FE) instead of O(T^2 x FE), and no [B,T,T] tensor.
            num,den,(newM,newz)=_sub_mix(qf,kf,v,L,state,sub,xq,xk,wgate)
            A=None
        else:
            A,num,den,newM,newz=s._full_mix(qf,kf,v,L,state,wgate,xq,xk)
        den=den+1e-5
        r=num/den; o=r@s.Wo.weight.T+s.Wo.bias
        a1=a+s.rs*o
        H2,n2=rmsnorm(a1); z=H2@s.m1.weight.T+s.m1.bias; g=F.gelu(z)
        a2=a1+s.rs*(g@s.m2.weight.T+s.m2.bias)
        return a2, (H,nrm,Hs,xk,kf,xq,qf,v,A,num,den,r,a1,H2,n2,z,g,(newM,newz))

    def _full_mix(s,qf,kf,v,L,state,wgate,xq=None,xk=None):
        """the original quadratic form, kept intact as the reference path and the fallback"""
        cfg=s.cfg
        if xq is not None and getattr(cfg,'fast_read',True):
            # IDENTITY, NOT AN APPROXIMATION. taylor(x) = [1, x, x(x)x], so for any two DIN-vectors
            #     qf . kf = 1 + (xq.xk) + (xq.xk)^2
            # The pairwise retrieval weight is a DIN-dim dot product and two flops. Expanding to FE = 1+DIN+DIN^2
            # dims first and then contracting over FE does the SAME arithmetic DIN times over -- and it does it
            # inside the only term that is quadratic in T, so it is exactly where the waste multiplies.
            # The features are still built, but only for the STATE (read qf.M, write kf(x)v), which is LINEAR in T.
            # Verified bit-comparable against the feature form; the win grows with DIN, which matters because
            # selectivity saturates at DIN:1 and DIN is the next ceiling to lift.
            # NUMERICS, measured not assumed. In bf16 this is very slightly LESS accurate than the feature form
            # (1.20e-03 vs 1.03e-03 against an fp32 reference, = +1.7e-04 BPB on the 124M), because the FE-dim
            # contraction it replaces accumulates inside the tensor core at fp32 and rounds once, whereas here c
            # is rounded to bf16 before being squared. Promoting the polynomial to fp32 does NOT fix that -- c is
            # already rounded by then -- and it costs half the speed (1.24x -> 1.07x), so it was reverted. The
            # residual is 4x smaller than either form's own distance from fp32, and far below any threshold that
            # decides anything here.
            c=torch.einsum('btd,bsd->bts',xq,xk); A=1+c+c*c
        else:
            A=torch.einsum('btf,bsf->bts',qf,kf)               # retrieval weights beta_{t,s} = qf_t . kf_s
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
            A=A*(L.unsqueeze(-1)-L.unsqueeze(-2)).clamp(max=0.0).exp()
        if wgate is not None: A=A*wgate.unsqueeze(-2)          # ... and the same weight on the WITHIN-chunk read,
        A=A.tril()                                             # or the parallel form and the recurrence disagree
                                                               # causal: read at t sees writes s<=t
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
        # SELECTIVE WRITING (VISION propose gate): `wgate` [B,T] weights each token's write into the PERSISTENT
        # state. Every token writing means N tokens superimpose in a fixed state, so any one fact's share is ~1/N
        # (measured: keeping all writes is actively harmful, -1.10 gain). Gating by salience multiplies the
        # effective span at no extra state cost. wgate=None keeps the old behaviour (write everything).
        if s.gated:
            # write each key with the decay it will have accumulated by the END of the chunk, and shrink the
            # incoming state by the chunk's total gate -- this is exactly M_T = (prod g) M_0 + sum_t (prod_{u>t} g_u) k_t (x) v_t
            gt=(L[:,-1:]-L).exp().unsqueeze(-1)                # [B,T,1]: gate from t to the end of the chunk
            kg=kf*gt
            if wgate is not None: kg=kg*wgate.unsqueeze(-1)    # selective write into the PERSISTENT state
            newM=torch.einsum('btf,btd->bfd',kg,v).detach(); newz=kg.squeeze(-1).sum(1).detach() if kg.dim()==4 else kg.sum(1).detach()
            if state is not None:
                gtot=L[:,-1].exp().view(-1,1,1)
                newM=state[0]*gtot+newM; newz=state[1]*gtot.squeeze(-1)+newz
        elif wgate is None:
            newM=torch.einsum('btf,btd->bfd',kf,v).detach(); newz=kf.sum(1).detach()
            if state is not None: newM=state[0]+newM; newz=state[1]+newz
        else:
            gw=wgate.unsqueeze(-1)
            newM=torch.einsum('btf,btd->bfd',kf*gw,v).detach(); newz=(kf*gw).sum(1).detach()
            if state is not None: newM=state[0]+newM; newz=state[1]+newz
        return A,num,den,newM,newz
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
    def step(s,tok,states,prev=None,write_scale=1.0):
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
            if L.qk_center:                                        # frozen running mean, same as forward()
                xk=xk-L.k_mu.to(xk.dtype); xq=xq-L.q_mu.to(xq.dtype)
            if L.qknorm:                                           # MUST match forward() or generation drifts
                xk=xk*torch.rsqrt(xk.pow(2).sum(-1,keepdim=True)+1e-6)*L.qk_s(src)   # SUM: invariant to appended zero dims
                xq=xq*torch.rsqrt(xq.pow(2).sum(-1,keepdim=True)+1e-6)*L.qk_s(H[:,0])
            kf=taylor(xk,cfg)                                      # key from the PREVIOUS token (the shift)
            v=(H[:,0])@L.Wv.weight.T
            if L.gated:
                # the recurrent form of the gate: decay the whole state by this token's rate, then write.
                # forward() expresses the same thing as the pairwise factor exp(L_t - L_s); here it is literal.
                g=torch.sigmoid(L.Wg(H[:,0]))                      # [B,1]
                M=M*g.unsqueeze(-1); z=z*g
            # write_scale weights what GENERATION writes back into the state. During training the model only
            # ever writes REAL text; at generation it writes its own output, and with a learned forget half-life
            # of ~2000 tokens those writes accumulate almost undecayed until the read is dominated by whatever it
            # just produced -- a positive feedback loop. Measured: unchanged (1.0) collapses to 4gram-rep 0.79 by
            # token 1000, while a frozen state (0.0) holds 0.00 for the full 1200. Default 1.0 = old behaviour.
            ws=write_scale*(torch.sigmoid(L.Wsal(H[:,0])) if L.sel else 1.0)     # selective write, matching forward()
            M=M+ws.unsqueeze(-1)*kf.unsqueeze(2)*v.unsqueeze(1) if L.sel else M+write_scale*kf.unsqueeze(2)*v.unsqueeze(1)
            z=z+ws*kf if L.sel else z+write_scale*kf                             # write into the state
            qf=taylor(xq,cfg)
            num=torch.einsum('bf,bfd->bd',qf,M); den=(qf*z).sum(-1,keepdim=True)+1e-5
            o=(num/den)@L.Wo.weight.T+L.Wo.bias
            a1=a+L.rs*o.unsqueeze(1)
            H2,_=rmsnorm(a1)
            a=a1+L.rs*(F.gelu(H2@L.m1.weight.T+L.m1.bias)@L.m2.weight.T+L.m2.bias)
            ns.append((M,z)); npv.append(H[:,0])
        h,_=rmsnorm(a); return s.head(h)[:,0]*getattr(s.cfg,'head_scale',1.0), ns, npv
    @torch.no_grad()
    def generate(s,ids,n=40,temp=1.0,states=None,top_p=1.0,greedy=False,write_scale=0.25,
                 refresh_every=1000,refresh_blend=0.5):
        """write_scale controls how much of its OWN output the model writes back into the recurrent state.
REFRESH (refresh_every / refresh_blend) IS WHAT MAKES LONG GENERATION HOLD. Generated tokens
        accumulate in a fixed-size state; because the model's own output is far more self-similar than real text,
        it eventually dominates the read, becomes an attractor, and the text loops. Every `refresh_every` tokens
        the state is blended back toward the PROMPT's state, so recent output still informs generation but the
        model's own voice never accumulates enough to take over. Measured over 20,000 generated tokens:
            constant write_scale 0.25   4-gram rep 0.02 -> 0.20 -> 0.34 -> 0.42 by 4k, then loops
            refresh 1000 / blend 0.5    rep 0.04, 0.07, 0.05, 0.11, 0.10 across 0-20k   (real text 0.05)
        and the text is still on topic 19,000 tokens after the prompt. Swept: period 250/500/1000 and blend
        0.25/0.5/0.75; 1000 and 0.5 won. Set refresh_every=0 to disable.

        DEFAULT CHANGED 0.0 -> 0.25. The 0.0 default was measured on a model from the OLD trainer, where ANY
        self-writing collapsed into word loops. On the clean run it is the opposite: 0.0 means the model cannot
        SEE its own output -- each step has only the current token plus a frozen prompt state -- so once it emits a
        newline, "current token = newline" predicts another, and it loops by construction. Measured at 65M on the
        clean run, 6 documents:
            write_scale 0.00 -> 4gram-rep 0.24, longest newline run 28.3   (real text: 0.02)
            write_scale 0.25 -> 4gram-rep 0.03, longest newline run  0.0   <- essentially real-text parity
            write_scale 0.50 -> 4gram-rep 0.15, longest newline run  8.7
            write_scale 1.00 -> 4gram-rep 0.14, longest newline run  5.7
        The optimum is model-dependent, so re-measure it rather than assuming; this default is the measured best
        on the current architecture.

        Earlier measurement, 12 generations of 1200 tokens each (54M, OLD trainer), real text at 4gram-rep 0.14:
            write_scale 1.00 -> rep 0.44 +-0.27, tail 0.59   ('century century century century ...')
            write_scale 0.50 -> rep 0.42 +-0.22, tail 0.61
            write_scale 0.25 -> rep 0.33 +-0.24, tail 0.51
            write_scale 0.10 -> rep 0.23 +-0.22, tail 0.40
            write_scale 0.00 -> rep 0.03 +-0.05, tail 0.01   <- stable, and 5x LESS variance
        It is effectively binary: any self-writing at all eventually dominates the read, because the model's own
        output is far more self-similar than real text and its keys align. The full PRIMED state is untouched, so
        everything the model READ is still available -- only its own output is withheld.
        Confirmed STRUCTURAL, not a capacity problem: a 202M model on 5.4B tokens degenerates identically, and so
        do UNGATED models. Training-side repair (scheduled sampling) was tried and made it worse; see PROGRESS.md."""
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
        _base=[(a.clone(),z.clone()) for a,z in st]      # the PROMPT's state, the anchor refresh returns to
        out=[]; cur=x[:,-1:]
        for i in range(n):
            if refresh_every and i and i%refresh_every==0:
                b=refresh_blend
                st=[((1-b)*a+b*a0,(1-b)*z+b*z0) for (a,z),(a0,z0) in zip(st,_base)]
            lg,st,pv=s.step(cur,st,pv,write_scale=write_scale)
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
