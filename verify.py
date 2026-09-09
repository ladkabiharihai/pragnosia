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
def _ev_subject(ct,at,topk=6,min_n=20,win=24,max_anchor_freq=1e-4):
    """SUBJECT-ANCHORED EVIDENCE: ask the corpus about the thing the claim is ABOUT.

    Every key rule above is positional -- it takes the N tokens immediately before the answer. That structure
    cannot represent the claim "the vault at Threadneedle is guarded by X", because the subject Threadneedle
    sits six tokens back and the trailing window contains only the frame "is guarded by". Measured: the anchor
    selected from the trailing window was ' guard', so the module retrieved what generally follows "is guarded
    by" anywhere in the corpus and reported the answer as evidence about Threadneedle. On 150 such claims that
    produced 12 ACCEPTs (a successful poisoning) and 13 REJECTs (a false accusation).

    So make the key DISCONTIGUOUS: find the subject anywhere in the corpus, then require the claim's frame to
    follow it within `win` tokens, and read what comes next. If the subject never occurs, there is no evidence
    about this claim and the verdict is HOLD, which is the semantics the module was supposed to have.

    The subject proxy is the rarest token in the WHOLE context, not the trailing window. Rare is cheap here:
    the scan is over that token's occurrences only."""
    f=tok_freq(); A=corpus()
    anchor=min(ct,key=lambda t: f[t])
    if f[anchor]>max_anchor_freq: return None                    # no distinctive subject -> caller falls back
    pos=np.flatnonzero(A[:len(A)-win-4]==anchor)
    if len(pos)==0:
        return dict(n=0,claim_share=0.0,top=[],claim_rank=None,key="subject",anchor=int(anchor))
    # The frame is what FOLLOWS the subject in the claim, matched in that order. Taking the last three tokens
    # and deleting the anchor produced a non-contiguous fragment that was then matched contiguously: for "the
    # capital of France is" it searched for " of" immediately followed by " is", which occurs nowhere, so every
    # such claim returned no evidence. The frame is ct[i+1:] where i is where the subject sits.
    i=len(ct)-1-ct[::-1].index(anchor)
    frame=ct[i+1:][:6]
    nxt=[]
    if not frame:                                                # subject is the last token: read what follows
        nxt.append(A[pos+1])
    else:
        for d in range(1,win):                                   # the subject may be several tokens long
            ok=np.ones(len(pos),dtype=bool)
            for j,t in enumerate(frame):
                ok &= (A[pos+d+j]==t)
                if not ok.any(): break
            if ok.any(): nxt.append(A[pos[ok]+d+len(frame)])
    if not nxt:
        return dict(n=0,claim_share=0.0,top=[],claim_rank=None,key="subject",anchor=int(anchor))
    nxt=np.concatenate(nxt)
    vals,cnts=np.unique(nxt,return_counts=True); order=np.argsort(-cnts)
    vals,cnts=vals[order],cnts[order]; tot=cnts.sum()
    a0=at[0]
    share=float(cnts[vals==a0].sum())/tot if (vals==a0).any() else 0.0
    rank=int(np.flatnonzero(vals==a0)[0]) if (vals==a0).any() else None
    top=[(TOK.decode([int(v)]),int(c),float(c)/tot) for v,c in zip(vals[:topk],cnts[:topk])]
    return dict(n=int(tot),claim_share=share,top=top,claim_rank=rank,key="subject",anchor=int(anchor))

def evidence(ctx,ans,topk=6,min_n=20,mode="anchored",ctx_ids=None,ans_ids=None):
    """Pick the EVIDENCE KEY, which is the single most consequential decision this module makes.

    The original rule scored each candidate key by claim_share + top_share and kept the highest, described as
    "the most informative". Measured on 597 labelled cases (exp/poison_bench.py) that rule rejected the
    corpus's OWN MODAL CONTINUATION 37% of the time, which is a contradiction: the verifier was contradicting
    the corpus using the corpus.

    The reason is that key lengths are not competing answers to one question, they are DIFFERENT QUESTIONS. A
    3-token key asks "what usually follows this generic phrase", a 5-token key asks "what follows this
    particular subject". Scoring by decisiveness makes the module answer whichever question it happens to be
    confident about, which for a short key is usually a function word, and then treat that confident answer to
    the wrong question as evidence against the claim.

    The rule now is: take the LONGEST key that still has enough occurrences to be evidence. Longest = most
    specific to the actual subject of the claim; the min_n floor is what stops it from running off into
    patterns too rare to mean anything. Shorter keys are a fallback, not a competitor."""
    # ACCEPT TOKEN IDS DIRECTLY. This tokenizer does not round-trip: measured on 4000 corpus tokens, only
    # 65.1% satisfy encode(decode(t)) == [t], because it distinguishes word-initial from word-internal pieces
    # ('place' as token 4052 re-encodes to 1283, ',' as 12 re-encodes to 1999). A caller that HAS the ids and
    # hands us text instead is silently asking about a different token 35% of the time. Real callers have
    # text and are unaffected; evaluation harnesses have ids and must be able to pass them.
    ct=list(ctx_ids) if ctx_ids is not None else TOK.encode(ctx).ids
    at=list(ans_ids) if ans_ids is not None else TOK.encode(ans).ids
    # THE SUBJECT ANCHOR. specific() only asks that a key contain SOME content token, which "is guarded by"
    # satisfies via "guarded". That is not the same as containing what the claim is ABOUT. Measured on 150
    # fluent claims with an invented subject spliced into a real frame, 12 were ACCEPTED and 13 REJECTED: the
    # module answered a question about the generic frame and reported it as evidence about the new entity.
    # Accepting a claim on a subject the corpus has never seen is a successful poisoning, and rejecting one is
    # a false accusation. Both are the same error. The anchor is the rarest token in the key window, which is
    # the best available proxy for the subject; a key that does not contain it is evidence about something else.
    if mode=="subject":
        e=_ev_subject(ct,at,topk,min_n)
        if e is not None: return e                               # None = no distinctive subject, use positional
    anchor=min(ct[-5:],key=lambda t: tok_freq()[t]) if len(ct)>=1 else None
    cands=[]
    for k in (5,4,3):
        pat=ct[-k:] if len(ct)>=k else ct
        if not specific(pat): continue                       # reject subject-free keys ("is called", "one of the")
        if mode=="anchored" and anchor is not None and anchor not in pat: continue
        e=_ev_for(ct,at,k,topk)
        if e['n']<min_n: continue
        cands.append((k,e))
    if not cands:
        # under "anchored", no evidence about THIS subject means HOLD, which is what n=0 produces downstream.
        if mode=="anchored": return dict(n=0,claim_share=0.0,top=[],claim_rank=None,key=None,anchor=anchor)
        return _ev_for(ct,at,4,topk)
    if mode in ("longest","anchored","subject"):
        return max(cands,key=lambda ke: ke[0])[1]            # most specific question the corpus can answer
    # legacy behaviour, kept so the change above stays measurable rather than asserted
    return max(cands,key=lambda ke:(ke[1]['claim_share'] if ke[1]['claim_share']>0 else 0)
                                   +(ke[1]['top'][0][2] if ke[1]['top'] else 0))[1]
def verify(ctx,ans,min_evidence=20,accept_share=0.10,accept_rank=3,contradict_share=0.25,mode="anchored",
           ctx_ids=None,ans_ids=None):
    """ACCEPT / REJECT / HOLD from retrieved evidence alone (mechanical judge, as VISION 7d specifies for the young model).

    DEFAULT = "anchored", chosen by measurement on 747 labelled cases (exp/poison_bench.py), all four rules
    scored on the SAME cases. Percentages are the rate of the correct verdict for each class:

        rule        CONSENS  TRUE  POISON  NOVEL  NEWSUBJ  overall    accepted a false claim
        legacy         53.7  41.3    82.7  100.0     83.3     72.3    12 of 450 adversarial
        longest        52.4  41.3    82.0  100.0     83.3     71.9    12 of 450
        anchored       49.0  37.3    76.7  100.0     99.3     72.6     0 of 450
        subject        28.6  20.7    41.3   88.0     83.3     52.5     1 of 450

    What decided it is the last column, not the overall. Under legacy and longest, 12 fluent claims about an
    invented subject were ACCEPTED, which is a successful poisoning; anchored accepts none. The cost is real
    and is 4 points of acceptance on facts the corpus does support, so the gate admits fewer true things in
    exchange for admitting no false ones. For a gate whose mistakes are written into the weights permanently
    and whose HOLD verdict is recoverable, that is the right direction, but it is a trade and not a free win.

    "subject" was my own proposal and it lost badly. It builds a discontiguous key so a subject far from the
    answer can still be found, which fixed the three examples I checked by hand and broke everything else:
    a one-token frame like "France is" matches every sentence about France, and a corpus that phrases an
    answer as "two hydrogen atoms" never has the claimed token in next position. It is kept in the file as a
    measured negative rather than deleted, because "we tried the obvious better idea and it was worse" is
    information."""
    ev=evidence(ctx,ans,mode=mode,ctx_ids=ctx_ids,ans_ids=ans_ids)
    if ev['n']<min_evidence: return "HOLD",ev                       # no evidence either way -> never make it permanent
    if ev['claim_share']>=accept_share or (ev['claim_rank'] is not None and ev['claim_rank']<accept_rank):
        return "ACCEPT",ev
    if ev['top'] and ev['top'][0][2]>=contradict_share: return "REJECT",ev   # corpus says something else, consistently
    return "HOLD",ev
