"""Multi-corpus 32K data mixer: draws training windows from ALL available *_32k.bin (~290B tokens), weighted so the
thin high-value bins (math/code/instruction) are not drowned by the 170B web bulk. Weights are a blend of sqrt(size)
(so big corpora dominate but not overwhelmingly) and a hand floor for faculty bins. One place that owns 'use all data'.
"""
import os, numpy as np, glob, re
DATADIR="/mnt/kv_cache/pragnosia_data"
# faculty floors (min sampling share) so small but important corpora are always represented
# Floors keep every faculty present. Values reduced for the intrinsically REPETITIVE corpora: measured 4-gram
# repetition is 0.20 (finemath/openmath), 0.26 (numina), 0.15 (opencode) vs 0.05 for web text, and at 25% of batches
# they pushed the model's own generations to 0.14 repetition. Math/code now ~13% instead of ~25%.
FLOOR={"finemath":0.04,"openmath":0.03,"opencode":0.03,"recall":0.05,"tulu3":0.03,"numina":0.01,
       "hermes":0.03,"evolcode":0.01,"glaive":0.01,"magicoder":0.005}
# CAP the intrinsically repetitive corpora. Lowering their FLOORS did nothing (24.8% -> 24.7%) because the sqrt(size)
# base weight already exceeded the floors -- the floors were never the binding constraint. An explicit cap is.
CAP={"finemath":0.05,"openmath":0.03,"numina":0.01,"opencode":0.03,"evolcode":0.01,"magicoder":0.005}
class DataMix:
    def __init__(s, ctx=64, min_tok=2_000_000, exclude=("recall",), verbose=True):
        s.ctx=ctx; s.bins=[]; s.names=[]; sizes=[]
        for f in sorted(glob.glob(f"{DATADIR}/*_32k.bin")):
            nm=re.sub(r'_32k\.bin$','',os.path.basename(f))
            if any(x in nm for x in exclude): continue                 # recall handled separately (answer-masked)
            m=np.memmap(f,dtype=np.uint16,mode="r")
            if len(m)<max(min_tok,ctx+2): continue
            s.bins.append(m); s.names.append(nm); sizes.append(len(m))
        sizes=np.array(sizes,dtype=np.float64)
        w=np.sqrt(sizes); w=w/w.sum()                                  # sqrt-size base weight
        for i,nm in enumerate(s.names):                                # apply faculty floors, then caps
            for k,fl in FLOOR.items():
                if k in nm: w[i]=max(w[i],fl)
            for k,cp in CAP.items():
                if k in nm: w[i]=min(w[i],cp)
        s.w=w/w.sum(); s.sizes=sizes
        if verbose:
            print(f"[DataMix] {len(s.names)} corpora, {sizes.sum()/1e9:.0f}B tok total, ctx={ctx}",flush=True)
            for nm,ww,sz in sorted(zip(s.names,s.w,sizes),key=lambda x:-x[1]):
                print(f"    {nm:12s} {sz/1e9:6.1f}B  sample-w={ww*100:4.1f}%",flush=True)
    def batch(s, B):
        C=s.ctx; ci=np.random.choice(len(s.bins),size=B,p=s.w); X=np.empty((B,C+1),dtype=np.int64)
        for b,c in enumerate(ci):
            m=s.bins[c]; i=np.random.randint(0,len(m)-C-1); X[b]=m[i:i+C+1]
        return X                                                       # [B,C+1] token ids
if __name__=="__main__":
    dm=DataMix(ctx=64); xb=dm.batch(8)
    print("batch shape",xb.shape,"min/max id",xb.min(),xb.max(),flush=True)
