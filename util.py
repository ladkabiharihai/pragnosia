"""Shared utilities: standard logging location and a tqdm progress bar for every training loop.
All logs live in logs/ (never scattered in the workspace root). Usage:
    from util import LOGS, pbar
    for i in pbar(range(STEPS), "substrate", loss=lambda: cur):   ...
"""
import os,sys,time
from tqdm import tqdm
ROOT=os.path.dirname(os.path.abspath(__file__)); LOGS=os.path.join(ROOT,"logs")
os.makedirs(LOGS,exist_ok=True)
def logpath(name): return os.path.join(LOGS,f"{name}.log")
class _Tee:
    def __init__(s,*f): s.f=f
    def write(s,x):
        for h in s.f:
            h.write(x); h.flush()
    def flush(s):
        for h in s.f: h.flush()
def setup_log(name=None):
    """Every run writes to logs/<name>.log AND stdout, so any run can be tailed live:
         tail -f logs/<name>.log
    Called automatically by scripts; name defaults to the script filename."""
    import __main__
    if name is None:
        name=os.path.splitext(os.path.basename(getattr(__main__,"__file__","run")))[0]
    fh=open(logpath(name),"a",buffering=1)
    fh.write(f"\n===== {name} started {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    sys.stdout=_Tee(sys.__stdout__,fh); sys.stderr=_Tee(sys.__stderr__,fh)
    return logpath(name)
class _LinePbar:
    """Progress as CLEAN LINES (no carriage returns) -- readable with `tail -f` on a log file, which a tqdm bar is
    not. Same interface as tqdm (iterate, .set_postfix_str, .write)."""
    def __init__(s,it,desc,total,every):
        s.it=it; s.desc=desc; s.every=every; s.post=""
        s.total=total if total is not None else (len(it) if hasattr(it,"__len__") else None)
        s.t0=time.time(); s.last=0.0; s.n=0
    def set_postfix_str(s,x): s.post=x
    def write(s,x): print(x,flush=True)
    def __iter__(s):
        for v in s.it:
            yield v
            s.n+=1; now=time.time()
            if now-s.last>=s.every or (s.total and s.n==s.total):
                s.last=now; el=now-s.t0; rate=s.n/max(el,1e-9)
                if s.total:
                    rem=(s.total-s.n)/max(rate,1e-9)
                    print(f"[{s.desc}] {s.n}/{s.total} ({100*s.n/s.total:4.1f}%) elapsed {el/60:5.1f}m "
                          f"eta {rem/60:5.1f}m {rate:6.1f} it/s  {s.post}",flush=True)
                else:
                    print(f"[{s.desc}] {s.n} elapsed {el/60:5.1f}m {rate:6.1f} it/s  {s.post}",flush=True)
def pbar(it,desc="train",total=None,every=10.0):
    """Progress bar. Interactive terminal -> tqdm bar; writing to a log file -> clean periodic lines."""
    istty=getattr(sys.__stdout__,"isatty",lambda:False)()
    if istty:
        return tqdm(it,desc=desc,total=total,file=sys.__stdout__,ncols=110,mininterval=every,smoothing=0.1,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    return _LinePbar(it,desc,total,every)

def cfg_from_ckpt(sd,T=2048,res_scale=16**-0.5):
    """Build a Config that MATCHES a checkpoint, inferring anything the checkpoint did not record.
    DIN is read from the Wk weight shape when absent: DIN grows now (grow_din), and a checkpoint written before
    that change records only D/DV/NL, so a loader that assumed DIN=8 would fail to load a grown model. This is the
    same class of bug as DV not being recorded, caught the same way -- by reloading a grown checkpoint."""
    import core as _C
    w=sd['model']
    DIN=sd.get('DIN') or int(w['layers.0.Wk.weight'].shape[0])
    DV=sd.get('DV') or int(w['layers.0.Wv.weight'].shape[0])
    NL=sd.get('NL') or len([k for k in w if k.endswith('.Wk.weight')])
    cfg=_C.Config(D=sd['D'],DIN=DIN,DV=DV,NL=NL,T=T,res_scale=res_scale)
    cfg.tie_emb=sd.get('tie_emb',False); cfg.head_scale=sd.get('head_scale',1.0)
    # gated/qknorm MUST be restored too: without them Core builds no Wg and load_state_dict fails on the
    # unexpected layers.N.Wg keys. Fall back to INFERRING from the weights so checkpoints written before these
    # flags were persisted still load.
    cfg.gated=sd.get('gated', any(k.endswith('Wg.weight') for k in sd['model']))
    cfg.qknorm=sd.get('qknorm', cfg.gated)
    return cfg
