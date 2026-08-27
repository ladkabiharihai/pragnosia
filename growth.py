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
    with torch.no_grad():
        ne=nn.Embedding(cfg.vocab,nD).to(DEV); ne.weight[:,:D]=m.emb.weight*c; ne.weight[:,D:]=0.0
        m.emb=ne
        for L in m.layers:
            for nm in ('Wk','Wq','Wv'):                                        # READ the (normed) stream -> /c, new cols random
                old=getattr(L,nm); outf=old.weight.shape[0]
                new=nn.Linear(nD,outf,bias=False).to(DEV)
                new.weight[:,:D]=old.weight/c; new.weight[:,D:]=_new_lin(add,outf,DEV)
                setattr(L,nm,new)
            no=nn.Linear(L.Wo.weight.shape[1],nD).to(DEV)                      # WRITES the stream -> *c, new dims 0
            no.weight[:D]=L.Wo.weight*c; no.weight[D:]=0.0
            no.bias[:D]=L.Wo.bias*c; no.bias[D:]=0.0
            L.Wo=no
            hid=L.m1.weight.shape[0]                                            # MLP: m1 READS the stream, m2 WRITES it
            n1=nn.Linear(nD,hid).to(DEV)
            n1.weight[:,:D]=L.m1.weight/c; n1.weight[:,D:]=_new_lin(add,hid,DEV); n1.bias.copy_(L.m1.bias)
            n2=nn.Linear(hid,nD).to(DEV)
            n2.weight[:D]=L.m2.weight*c; n2.weight[D:]=0.0
            n2.bias[:D]=L.m2.bias*c; n2.bias[D:]=0.0
            L.m1,L.m2=n1,n2
        nh=nn.Linear(nD,cfg.vocab).to(DEV)                                     # READS the normed stream -> /c
        nh.weight[:,:D]=m.head.weight/c; nh.weight[:,D:]=0.0; nh.bias.copy_(m.head.bias)
        m.head=nh
    cfg.D=nD
    for L in m.layers: L.cfg=cfg
    return nD

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
    def __init__(s,window=200,t_stat=1.0,min_gap=400,floor_tokens=0):
        s.w=window; s.t=t_stat; s.min_gap=min_gap; s.hist=[]; s.last=0; s.floor=floor_tokens
    @staticmethod
    @torch.no_grad()
    def entropy(logits,sample=4096):
        """the model's own uncertainty (subsampled: a full float softmax over 32k vocab is wasteful)."""
        f=logits.reshape(-1,logits.shape[-1])
        idx=torch.randint(0,f.shape[0],(min(sample,f.shape[0]),),device=f.device)
        p=f[idx].float().softmax(-1)
        return float(-(p*(p+1e-9).log()).sum(-1).mean())
    def observe(s,ent,step,tokens_seen=0):
        s.hist.append(ent)
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
