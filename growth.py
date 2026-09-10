"""GROWTH: function-preserving capacity growth for the Pragnosia core (grow from infant, monotonic = never hurts).
Axes: DEPTH (+residual layer, Wo=0 -> identity), STATE/DV (+value dims, new Wo cols=0), WIDTH/D (+residual-stream
dims, with the RMS-norm rescaling that keeps the function EXACTLY unchanged), VOCAB (+token rows).
Every op is verified numerically (max|dOut| ~ 0) by the self-test. New capacity starts INERT but receives gradient,
so it can only add. Read-side weights get small random init (symmetry breaking); write-side starts at zero."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
import core as C
DEV=C.DEV

def _new_lin(inf,outf,dev,zero=False,scale=0.02):
    w=torch.zeros(outf,inf,device=dev) if zero else torch.randn(outf,inf,device=dev)*scale
    return w

def grow_depth(m,n=1):
    """+n residual layers at the top, each initialized to IDENTITY (Wo=0) -> function exactly preserved."""
    cfg=m.cfg
    for _ in range(n):
        L=C.Layer(cfg).to(DEV)
        with torch.no_grad():                                   # BOTH residual branches must start at zero for identity
            L.Wo.weight.zero_(); L.Wo.bias.zero_(); L.m2.weight.zero_(); L.m2.bias.zero_()
        m.layers.append(L)
    cfg.NL=len(m.layers); return cfg.NL

def grow_state(m,add):
    """+add value/state dims (DV): Wv gains rows (random), Wo gains cols (ZERO) -> output unchanged."""
    cfg=m.cfg
    for L in m.layers:
        with torch.no_grad():
            nv=nn.Linear(cfg.D,cfg.DV+add,bias=False).to(DEV); nv.weight[:cfg.DV]=L.Wv.weight
            nv.weight[cfg.DV:]=_new_lin(cfg.D,add,DEV)
            no=nn.Linear(cfg.DV+add,cfg.D).to(DEV); no.weight[:,:cfg.DV]=L.Wo.weight
            no.weight[:,cfg.DV:]=0.0; no.bias.copy_(L.Wo.bias)                 # new state writes nothing yet
            L.Wv,L.Wo=nv,no
    cfg.DV=cfg.DV+add; return cfg.DV

def grow_width(m,add):
    """+add residual-stream dims (D). RMS-norm makes this subtle: adding zero dims changes mean(x^2), so rescale by
    c=sqrt((D+add)/D) on everything that WRITES the stream and 1/c on everything that READS it. Exactly preserving."""
    cfg=m.cfg; D=cfg.D; nD=D+add; c=(nD/D)**0.5
    cw=1.0 if getattr(cfg,'tie_emb',False) else c    # writer scale: 1 when tied (stream magnitude is unchanged)
    with torch.no_grad():
        ne=nn.Embedding(cfg.vocab,nD).to(DEV)
        # when tied, the shared matrix must not be scaled (the head needs /c, the embedding *c); the head's factor
        # is carried on cfg.head_scale instead, and the stream keeps its magnitude so readers still take /c.
        ne.weight[:,:D]=m.emb.weight if getattr(cfg,'tie_emb',False) else m.emb.weight*c
        ne.weight[:,D:]=0.0
        m.emb=ne
        nDV=cfg.DV+add                                                         # the VALUE/state dim must widen too, or
        for L in m.layers:                                                      # the recurrent memory capacity freezes
            for nm in ('Wk','Wq'):                                             # keys/queries: output dim is DIN (fixed)
                old=getattr(L,nm); outf=old.weight.shape[0]
                new=nn.Linear(nD,outf,bias=False).to(DEV)
                new.weight[:,:D]=old.weight/c; new.weight[:,D:]=_new_lin(add,outf,DEV)
                setattr(L,nm,new)
            nv=nn.Linear(nD,nDV,bias=False).to(DEV)                            # Wv: wider input AND wider value space
            nv.weight[:cfg.DV,:D]=L.Wv.weight/c; nv.weight[:cfg.DV,D:]=_new_lin(add,cfg.DV,DEV)
            nv.weight[cfg.DV:]=_new_lin(nD,add,DEV)                            # new value dims (learn; written nowhere yet)
            L.Wv=nv
            no=nn.Linear(nDV,nD).to(DEV)                                       # WRITES the stream -> *c, new dims 0
            no.weight[:D,:cfg.DV]=L.Wo.weight*cw; no.weight[:D,cfg.DV:]=0.0      # new value dims contribute nothing yet
            no.weight[D:]=0.0
            no.bias[:D]=L.Wo.bias*cw; no.bias[D:]=0.0
            L.Wo=no
            # MLP: m1 READS the stream, m2 WRITES it. The HIDDEN dim must grow with D or the MLP ratio collapses
            # (4x -> 1.5x by the target) and the measured 1.17-nat MLP advantage erodes as the model grows.
            if getattr(L,'gated',False):
                # THE GATE PROJECTION MUST GROW TOO. Wg reads the residual stream, so it is a READER and takes /c
                # like Wk/Wq. Omitting this crashed run 4 at growth #1: D went 384 -> 640 while Wg stayed [1,384]
                # ("a and b must have same reduction dim, but got [16384, 640] X [384, 1]").
                ng=nn.Linear(nD,L.Wg.weight.shape[0]).to(DEV)
                ng.weight[:,:D]=L.Wg.weight/c; ng.weight[:,D:]=0.0    # new dims contribute nothing yet
                ng.bias.copy_(L.Wg.bias)                              # keep the learned forget rate exactly
                L.Wg=ng
            if getattr(L,'sel',False):
                # SELECTIVE WRITE reads the residual stream, so it is a READER: /c with new dims at zero. Missing
                # exactly this for Wg is what crashed run 4 at its first growth.
                nw=nn.Linear(nD,1).to(DEV)
                nw.weight[:,:D]=L.Wsal.weight/c; nw.weight[:,D:]=0.0
                nw.bias.copy_(L.Wsal.bias)                            # keep the learned write policy exactly
                L.Wsal=nw
            if getattr(L,'qk_tok',False):
                # READ SHARPNESS reads the residual stream too, so it is a READER and takes /c with new dims at
                # zero -- the same shape of bug that crashed run 4 when Wg was missed here.
                nq=nn.Linear(nD,1).to(DEV)
                nq.weight[:,:D]=L.Wqs.weight/c; nq.weight[:,D:]=0.0
                nq.bias.copy_(L.Wqs.bias)                             # keep the learned sharpness exactly
                L.Wqs=nq
            hid=L.m1.weight.shape[0]; mult=getattr(cfg,'mlp_mult',4); nhid=mult*nD; addh=nhid-hid
            n1=nn.Linear(nD,nhid).to(DEV)
            n1.weight[:hid,:D]=L.m1.weight/c; n1.weight[:hid,D:]=_new_lin(add,hid,DEV)
            if addh>0: n1.weight[hid:]=_new_lin(nD,addh,DEV)                   # new hidden units: random (they learn)
            n1.bias[:hid]=L.m1.bias; n1.bias[hid:]=0.0
            n2=nn.Linear(nhid,nD).to(DEV)
            n2.weight[:D,:hid]=L.m2.weight*cw; n2.weight[:D,hid:]=0.0           # new hidden units write NOTHING yet
            n2.weight[D:]=0.0                                                   # new stream dims are not written yet
            n2.bias[:D]=L.m2.bias*cw; n2.bias[D:]=0.0
            L.m1,L.m2=n1,n2
        if getattr(cfg,'tie_emb',False):
            # TIED: the head shares the embedding matrix, so it cannot carry its own /c. Neither matrix is scaled
            # (see the embedding branch above) and the head's factor accumulates on cfg.head_scale instead.
            nh=nn.Linear(nD,cfg.vocab).to(DEV)
            # head_scale multiplies the head's ENTIRE output, bias included -- but only the WEIGHTS should take
            # the 1/c (the normed stream grew by c; the bias did not). So pre-multiply the bias by c to cancel it:
            #   (W.(c.h) + c.b)/c == W.h + b   exactly.
            # Without this the bias silently becomes b/c, which measured as a 1.23e-02 logit error after growth.
            nh.bias.copy_(m.head.bias*c)
            nh.weight=m.emb.weight                                             # re-tie: growth silently untied it
            m.head=nh; cfg.head_scale=getattr(cfg,'head_scale',1.0)/c
        else:
            nh=nn.Linear(nD,cfg.vocab).to(DEV)                                 # READS the normed stream -> /c
            nh.weight[:,:D]=m.head.weight/c; nh.weight[:,D:]=0.0; nh.bias.copy_(m.head.bias)
            m.head=nh
    cfg.D=nD; cfg.DV=nDV
    for L in m.layers: L.cfg=cfg; L.rs=cfg.res_scale
    return nD

def grow_din(m,add):
    """GROW THE RETRIEVAL SPACE. DIN is the width of the key/query projection, and the Taylor map turns it into
    FE = 1 + DIN + DIN(DIN+1)/2 features through which EVERY key-query association in the model must pass.

    THE DEFECT THIS FIXES: DIN was fixed at 8 from the infant onward, and grow_width deliberately kept it there
    ("keys/queries: output dim is DIN (fixed)"). So the model grew 2.7x in width, 384 -> 1024, while its retrieval
    space never grew at all: still 45 features against 1024 residual dims. A transformer at D=1024 has 1024
    dimensions of attention. MEASURED (exp/din_scaling.py): raising DIN 8 -> 32 is worth 0.133 nats at D=384 and
    0.196 nats at D=768 -- the penalty GROWS with width, which is exactly the regime our 235M model is in.

    FUNCTION-PRESERVING: new key/query dims are initialised to ZERO, so their linear Taylor features are zero and
    every cross term with them is zero. The added features contribute nothing until trained, and the existing
    retrieval is bit-for-bit unchanged. DIN costs almost no parameters (Wk,Wq are D x DIN); it costs state size
    (FE x DV) and attention compute."""
    cfg=m.cfg; D=cfg.D; oldDIN=cfg.DIN; nDIN=oldDIN+add
    with torch.no_grad():
        for L in m.layers:
            for nm in ('Wk','Wq'):
                old=getattr(L,nm)
                new=nn.Linear(D,nDIN,bias=False).to(DEV)
                new.weight[:oldDIN]=old.weight; new.weight[oldDIN:]=0.0     # new dims inert
                # no rescaling needed: core's qk-norm normalises by the SUM of squares, which is invariant to
                # appended zero dims. Pre-scaling the weights here was useless anyway -- the normalisation it
                # feeds divides the scaling straight back out.
                setattr(L,nm,new)
            # centering buffers are DIN-dim and must grow with it. Zero for the new dims is exactly right: a
            # dimension whose projection is zero has mean zero, so subtracting zero leaves it inert.
            for bn in ('k_mu','q_mu'):
                if hasattr(L,bn):
                    ob=getattr(L,bn); nb=torch.zeros(nDIN,device=ob.device,dtype=ob.dtype)
                    nb[:oldDIN]=ob; L.register_buffer(bn,nb)
    cfg.DIN=nDIN
    # FE MUST MATCH taylor(). taylor() now builds the FULL outer product (DIN*DIN terms), not the upper triangle,
    # so FE = 1 + DIN + DIN^2. This line still used the triangular count and silently disagreed with the model.
    cfg.iu=torch.triu_indices(nDIN,nDIN,device=DEV); cfg.FE=1+nDIN+nDIN*nDIN
    for L in m.layers: L.cfg=cfg
    return nDIN
def grow_vocab(m,new_vocab):
    """+new token rows in emb and head. Old tokens' behaviour is unchanged (function preserved on the old vocab)."""
    cfg=m.cfg; V=cfg.vocab
    with torch.no_grad():
        ne=nn.Embedding(new_vocab,cfg.D).to(DEV); ne.weight[:V]=m.emb.weight
        nn.init.normal_(ne.weight[V:],0,0.02); m.emb=ne
        nh=nn.Linear(cfg.D,new_vocab).to(DEV); nh.weight[:V]=m.head.weight; nh.bias[:V]=m.head.bias
        nh.weight[V:]=0.0; nh.bias[V:]=-30.0                                   # new tokens start unpredicted
        m.head=nh
    cfg.vocab=new_vocab; return new_vocab

if __name__=="__main__":
    import sys; sys.path.insert(0,"/opt/code/membrae")
    torch.manual_seed(0)
    print("FUNCTION-PRESERVATION CHECK (max|dOut| should be ~0 for every growth axis):",flush=True)
    for tag,fn in [("depth  +2",lambda m:grow_depth(m,2)),
                   ("state  +64 (DV)",lambda m:grow_state(m,64)),
                   ("width  +64 (D)",lambda m:grow_width(m,64)),
                   ("vocab  ->33000",lambda m:grow_vocab(m,33000))]:
        cfg=C.Config(D=128,DIN=8,DV=128,NL=4,T=24,vocab=32768,res_scale=0.5); m=C.Core(cfg).to(DEV).eval()
        X=torch.randint(0,32768,(3,24),device=DEV)
        with torch.no_grad(): y0=m(X)[...,:32768].clone()
        new=fn(m)
        with torch.no_grad(): y1=m(X)[...,:32768]
        d=(y0-y1).abs().max().item()
        print(f"  {tag:18s} -> {new}   max|dOut|={d:.2e}  {'PRESERVED' if d<1e-3 else 'BROKEN'}",flush=True)

class SaturationSensor:
    """INTRINSIC growth trigger (VISION 0a: the model decides WHEN to grow, not a hand-written schedule).
    Signal = the model's OWN predictive entropy (label-free: it needs no targets, it is the model's uncertainty
    about its own outputs). Criterion is NOISE-RELATIVE, not a magic threshold: fit the slope of that entropy over a
    window and compare it to the slope's own standard error. If the entropy is no longer falling faster than its own
    noise, the model has extracted what it can at this capacity -- it is saturated, and it asks to grow."""
    def __init__(s,window=40,t_stat=1.0,min_gap=2000,floor_tokens=0,ema=0.02,consec=3,block=50):
        # window/ema matter: with a short window on the RAW signal the test shows the sensor fires on NOISE
        # (an improving-but-noisy trace looks saturated), which turns "intrinsic growth" into a hidden schedule.
        # Smoothing the signal and lengthening the window makes it discriminate correctly.
        # BLOCK MEANS, NOT AN EMA. The signal was smoothed with an exponential moving average and then fed to
        # a least-squares t-test that assumes independent samples. An EMA makes consecutive samples strongly
        # autocorrelated, so the residuals understate the true noise, the standard error comes out far too
        # small, and |t| is inflated. Measured live on the flagship the statistic swung between -20 and +24 on
        # a model that was simply training normally, and the only moments it read "flat" were the instants the
        # swing crossed zero. Averaging disjoint blocks of raw entropy reduces noise WITHOUT manufacturing
        # autocorrelation, so the t-test's own assumption is roughly satisfied and |t|<=1 means what it says.
        s.block=block; s.buf=[]
        s.w=window; s.t=t_stat; s.min_gap=min_gap; s.hist=[]; s.last=0; s.floor=floor_tokens
        s.ema_a=ema; s.ema=None; s.consec=consec; s.nsat=0
        s.n_since=0     # blocks since the criterion was last EVALUATED, so consecutive tests use disjoint data
    @staticmethod
    @torch.no_grad()
    def entropy(logits,sample=4096):
        """the model's own uncertainty (subsampled: a full float softmax over 32k vocab is wasteful)."""
        f=logits.reshape(-1,logits.shape[-1])
        idx=torch.randint(0,f.shape[0],(min(sample,f.shape[0]),),device=f.device)
        p=f[idx].float().softmax(-1)
        return float(-(p*(p+1e-9).log()).sum(-1).mean())
    def verdict(s):
        """The sensor's OWN opinion, independent of any external floor: (saturated?, t-statistic).
        Exposed separately because growth firing on the FIRST step a floor permits it means the FLOOR decided,
        not the sensor -- and that is the difference between intrinsic growth and a hidden schedule. Measured:
        growth #1 fired at step 46,947 against a floor of exactly 46,947, so the sensor was rubber-stamping.
        This makes that visible instead of inferable."""
        if len(s.hist)<s.w: return None,None
        y=np.asarray(s.hist,dtype=np.float64); x=np.arange(len(y),dtype=np.float64)
        n=len(y); sx=x-x.mean(); slope=float((sx*(y-y.mean())).sum()/(sx**2).sum())
        resid=y-(y.mean()+slope*sx); se=float(np.sqrt((resid**2).sum()/max(1,n-2)/(sx**2).sum()))
        if se<=0: return None,None
        t=(-slope)/se
        # SATURATED MEANS FLAT, NOT MERELY 'NOT FALLING'. The old test was `t <= t_stat`, which classifies a
        # RISING entropy as saturated -- and entropy always rises right after a growth, because new capacity
        # perturbs the model. Measured: it fired at 0.9 and 0.1 tok/param with t=-15.44, i.e. entropy climbing
        # steeply, which is disruption and the opposite of convergence. Requiring |t| <= t_stat means the slope
        # must be indistinguishable from zero: falling (t>0) is still improving, rising (t<0) is still settling,
        # and only flat is saturated.
        return (abs(t) <= s.t), t

    def observe(s,ent,step,tokens_seen=0):
        s.buf.append(float(ent))
        if len(s.buf)<s.block: return False                # accumulate a block of raw entropy first
        s.hist.append(float(np.mean(s.buf))); s.buf=[]      # one nearly-independent sample per block
        if len(s.hist)>s.w: s.hist.pop(0)
        s.n_since+=1
        if len(s.hist)<s.w or step-s.last<s.min_gap or tokens_seen<s.floor: return False
        # EVALUATE ON DISJOINT WINDOWS. observe() runs every step and the window slides by one, so consecutive
        # calls share 399 of 400 samples. Counting `consec` consecutive PASSES therefore meant three consecutive
        # STEPS, which is an instant, not sustained flatness -- and measured live it fired repeatedly at 13.5
        # tok/param while the t-statistic was swinging between -20 and +24, i.e. on the moment the swing crossed
        # zero. Only testing once per full window makes `consec` mean what the docstring says: three independent
        # windows, 1200 steps of genuinely flat entropy.
        if s.n_since<s.w: return False
        s.n_since=0
        y=np.asarray(s.hist,dtype=np.float64); x=np.arange(len(y),dtype=np.float64)
        n=len(y); sx=x-x.mean(); slope=float((sx*(y-y.mean())).sum()/(sx**2).sum())
        resid=y-(y.mean()+slope*sx); se=float(np.sqrt((resid**2).sum()/max(1,n-2)/(sx**2).sum()))
        if se<=0: return False
        still_improving = (-slope)/se > s.t                      # is entropy falling faster than its own noise?
        # same criterion as verdict(): flat, not merely not-falling
        t=(-slope)/se
        if abs(t) <= s.t:
            s.nsat += 1
            # REQUIRE SUSTAINED FLATNESS. One window can look flat while the model is mid-transient; asking for
            # `consec` consecutive flat windows is what separates converged from momentarily quiet.
            if s.nsat >= s.consec:
                # DO NOT clear state here. Clearing on fire meant a sensor that saturated while an external
                # floor was still closed threw its evidence away and had to re-earn 6000 steps of flatness,
                # so by the time the floor opened the sensor was mid-rebuild and the floor decided by default.
                # Saturation now PERSISTS until the caller acts on it and calls reset() after the growth.
                s.last=step; return True                          # SATURATED -> the model asks to grow
        else:
            s.nsat = 0
        return False

    def reset(s):
        """Call after a growth. The window would otherwise straddle the event and mix two different models."""
        s.hist=[]; s.ema=None; s.nsat=0


def snapshot_opt_state(m, opt):
    """Capture Adam's moments keyed by PARAMETER NAME, before a growth replaces the tensors.

    WHY THIS EXISTS. Growth is function preserving for the WEIGHTS: new parameters appear and the model computes
    the same function, verified to 2.3e-07. It has never been preserving for the OPTIMIZER. Every growth in
    every run so far rebuilt Adam from scratch, which zeroes exp_avg and exp_avg_sq for every parameter,
    including the ones that did not change. Adam then divides by sqrt(v) with v starting at zero, so the first
    steps after a growth take near-maximal-size steps on a model that was just enlarged. The weights survive
    the growth and the optimizer's entire estimate of the loss landscape does not.

    Measured consequence in the 125M-to-405M run: BPB rose across the whole run rather than falling, from 0.9329
    at step 260,000 to 1.0183 at step 672,000, with the steepest climbs immediately after each growth event.
    That is not the only cause (the learning rate was also held at 6e-4 from 125M all the way to 405M) but it is
    a real one and it is fixable exactly.
    """
    byname = {}
    pid2name = {id(p): n for n, p in m.named_parameters()}
    for group in opt.param_groups:
        for p in group['params']:
            st = opt.state.get(p)
            if st is None or 'exp_avg' not in st: continue
            n = pid2name.get(id(p))
            if n is None: continue
            byname[n] = (st['exp_avg'].detach().clone(),
                         st['exp_avg_sq'].detach().clone(),
                         st.get('step', None))
    return byname


def restore_opt_state(m, opt, byname):
    """Copy the snapshotted moments into the grown tensors, zero-padding the new dimensions.

    A grown parameter keeps its old values in the leading slice (that is what makes growth function preserving),
    so the old moments belong in exactly the same slice. New rows and columns start at zero, which is the honest
    initial state for a direction the optimizer has never seen. Returns how many tensors were carried, so the
    caller can log it rather than assume it worked."""
    carried = 0
    for n, p in m.named_parameters():
        old = byname.get(n)
        if old is None: continue
        ea, eas, step = old
        if ea.dim() != p.dim(): continue                     # shape class changed: no honest mapping
        if any(o > s for o, s in zip(ea.shape, p.shape)): continue   # shrank: not a growth we understand
        st = opt.state[p]                                    # opt.state is a defaultdict: this creates {}
        new_ea = torch.zeros_like(p); new_eas = torch.zeros_like(p)
        sl = tuple(slice(0, s) for s in ea.shape)
        new_ea[sl] = ea.to(new_ea.dtype); new_eas[sl] = eas.to(new_eas.dtype)
        st['exp_avg'] = new_ea; st['exp_avg_sq'] = new_eas
        # 'step' must ALWAYS be populated. Adam only runs its own initialisation when the state dict is empty,
        # so a state carrying moments but no step is not re-initialised and raises on the next step. Fused Adam
        # additionally wants step as a device tensor rather than a Python number.
        st['step'] = (step.detach().clone() if torch.is_tensor(step)
                      else torch.tensor(float(step or 0.0), device=p.device))
        carried += 1
    return carried
