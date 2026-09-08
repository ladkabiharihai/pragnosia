"""VERIFICATION (VISION 7d): the gate deciding what may become permanent knowledge. Anti-poisoning.
v2 -- real EVIDENCE RETRIEVAL instead of crude co-occurrence:
  find the claim's CONTEXT PATTERN in the trusted corpus and look at what actually FOLLOWS it.
  That distribution IS the evidence, and it separates the three cases the vision needs:
    - claim is what the corpus says            -> ACCEPT (consolidate)
    - corpus says something ELSE, consistently -> REJECT (contradicted; this is the anti-poisoning case)
    - the pattern does not occur at all        -> HOLD  (novel/unknown: keep in fast memory, do NOT make permanent)
Crucially this distinguishes "contradicted" from "never heard of it", which a prior-confidence check cannot."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
import core as C
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
_A=None
def corpus(n=600_000_000):
    global _A
    if _A is None:
        _A=np.asarray(np.memmap("/mnt/kv_cache/pragnosia_data/window2_32k.bin",dtype=np.uint16,mode="r")[:n])
    return _A
_FREQ=None
def tok_freq():
    """token frequency table (from a corpus sample) -- used to require that an evidence key contains real CONTENT,
    not just function words. A key like 'is called' matches thousands of unrelated sentences and is worthless."""
    global _FREQ
    if _FREQ is None:
        A=corpus()[:50_000_000]; _FREQ=np.bincount(A,minlength=32768).astype(np.float64)/len(A)
    return _FREQ
def specific(pat,max_freq=2e-3):
    """does this key contain at least one content token (rare enough to identify the subject)?"""
    f=tok_freq(); return len(pat)>0 and float(min(f[t] for t in pat))<max_freq
def find_pattern(pat,max_hits=4000):
    """positions where the token sequence `pat` occurs in the corpus."""
    A=corpus()
    if not pat: return np.empty(0,dtype=np.int64)
    pos=np.flatnonzero(A[:len(A)-len(pat)]==pat[0])
    for j,t in enumerate(pat[1:],1):
        if len(pos)==0: break
        pos=pos[A[pos+j]==t]
        if len(pos)>max_hits*8: pos=pos[:max_hits*8]
    return pos[:max_hits]
def _ev_for(ct,at,key_tokens,topk):
    pat=ct[-key_tokens:] if len(ct)>=key_tokens else ct
    pos=find_pattern(pat); A=corpus()
    if len(pos)==0: return dict(n=0,claim_share=0.0,top=[],claim_rank=None)
    nxt=A[pos+len(pat)]
    vals,cnts=np.unique(nxt,return_counts=True); order=np.argsort(-cnts)
    vals,cnts=vals[order],cnts[order]; tot=cnts.sum()
    a0=at[0]; share=float(cnts[vals==a0].sum())/tot if (vals==a0).any() else 0.0
    rank=int(np.flatnonzero(vals==a0)[0]) if (vals==a0).any() else None
    top=[(TOK.decode([int(v)]),int(c),float(c)/tot) for v,c in zip(vals[:topk],cnts[:topk])]
    return dict(n=int(len(pos)),claim_share=share,top=top,claim_rank=rank,key=key_tokens)
def evidence(ctx,ans,topk=6,min_n=20):
    """Try several key lengths; keep the most INFORMATIVE evidence (enough hits AND a decisive top answer).
    A long key is precise but rare; a short key is common but noisy -- so search the range instead of guessing."""
    ct=TOK.encode(ctx).ids; at=TOK.encode(ans).ids
    best=None
    for k in (5,4,3):
        pat=ct[-k:] if len(ct)>=k else ct
        if not specific(pat): continue                       # reject subject-free keys ("is called", "one of the")
        e=_ev_for(ct,at,k,topk)
        if e['n']<min_n: continue
        # informative = the corpus gives a decisive continuation, or it contains our claim
        score=(e['claim_share'] if e['claim_share']>0 else 0)+(e['top'][0][2] if e['top'] else 0)
        if best is None or score>best[0]: best=(score,e)
    return best[1] if best else _ev_for(ct,at,4,topk)
def verify(ctx,ans,min_evidence=20,accept_share=0.10,accept_rank=3,contradict_share=0.25):
    """ACCEPT / REJECT / HOLD from retrieved evidence alone (mechanical judge, as VISION 7d specifies for the young model)."""
    ev=evidence(ctx,ans)
    if ev['n']<min_evidence: return "HOLD",ev                       # no evidence either way -> never make it permanent
    if ev['claim_share']>=accept_share or (ev['claim_rank'] is not None and ev['claim_rank']<accept_rank):
        return "ACCEPT",ev
    if ev['top'] and ev['top'][0][2]>=contradict_share: return "REJECT",ev   # corpus says something else, consistently
    return "HOLD",ev
