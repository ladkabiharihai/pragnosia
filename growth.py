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
            nh=nn.Linear(nD,cfg.vocab).to(DEV); nh.bias.copy_(m.head.bias)
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
    cfg=m.cfg; D=cfg.D; nDIN=cfg.DIN+add
    with torch.no_grad():
        for L in m.layers:
            for nm in ('Wk','Wq'):
                old=getattr(L,nm)
                new=nn.Linear(D,nDIN,bias=False).to(DEV)
                new.weight[:cfg.DIN]=old.weight; new.weight[cfg.DIN:]=0.0   # new dims inert -> exactly preserving
                setattr(L,nm,new)
    cfg.DIN=nDIN
    cfg.iu=torch.triu_indices(nDIN,nDIN,device=DEV); cfg.FE=1+nDIN+cfg.iu.shape[1]
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
    def __init__(s,window=400,t_stat=1.0,min_gap=2000,floor_tokens=0,ema=0.02):
        # window/ema matter: with a short window on the RAW signal the test shows the sensor fires on NOISE
        # (an improving-but-noisy trace looks saturated), which turns "intrinsic growth" into a hidden schedule.
        # Smoothing the signal and lengthening the window makes it discriminate correctly.
        s.w=window; s.t=t_stat; s.min_gap=min_gap; s.hist=[]; s.last=0; s.floor=floor_tokens
        s.ema_a=ema; s.ema=None
    @staticmethod
    @torch.no_grad()
    def entropy(logits,sample=4096):
        """the model's own uncertainty (subsampled: a full float softmax over 32k vocab is wasteful)."""
        f=logits.reshape(-1,logits.shape[-1])
        idx=torch.randint(0,f.shape[0],(min(sample,f.shape[0]),),device=f.device)
        p=f[idx].float().softmax(-1)
        return float(-(p*(p+1e-9).log()).sum(-1).mean())
    def observe(s,ent,step,tokens_seen=0):
        s.ema=ent if s.ema is None else (1-s.ema_a)*s.ema+s.ema_a*ent      # smooth first: raw entropy is too noisy
        s.hist.append(s.ema)
        if len(s.hist)>s.w: s.hist.pop(0)
        if len(s.hist)<s.w or step-s.last<s.min_gap or tokens_seen<s.floor: return False
        y=np.asarray(s.hist,dtype=np.float64); x=np.arange(len(y),dtype=np.float64)
        n=len(y); sx=x-x.mean(); slope=float((sx*(y-y.mean())).sum()/(sx**2).sum())
        resid=y-(y.mean()+slope*sx); se=float(np.sqrt((resid**2).sum()/max(1,n-2)/(sx**2).sum()))
        if se<=0: return False
        still_improving = (-slope)/se > s.t                      # is entropy falling faster than its own noise?
        if not still_improving:
            s.last=step; s.hist=[]; return True                  # SATURATED -> the model asks to grow
        return False
