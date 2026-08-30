"""THE PROPOSER WAS ASKING THE WRONG QUESTION.

Both versions took the continuation FROM THE DOCUMENT and asked the corpus to confirm it. But candidates are chosen
by SURPRISE -- precisely the tokens the corpus would not predict -- so verification mostly says REJECT. The two
criteria were fighting: surprise selects novelty, verification accepts consensus.

What is actually worth learning is the intersection the old design could never hit:
    the model is IGNORANT (high surprise)  AND  the world AGREES (strong corpus consensus)

So: find a checkable key, ask the CORPUS what follows it, and if the corpus is decisive while the model is not,
propose the CORPUS's answer. The claim is then true by construction, and what gets consolidated is a real regularity
the individual did not know -- not whatever this one document happened to say.

This also makes the loop honest about its source: the corpus is the verified authority (VISION 7d), so the fact
taught is the corpus's, and the document merely points at where to look.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("proposer2")
import numpy as np, torch
import core as C
from data_mix import DataMix
import verify as V
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
SUB = "/mnt/kv_cache/pragnosia_data/life/body.pt"


def load():
    sd = torch.load(SUB, map_location=DEV)
    cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=1024, res_scale=16 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m, cfg


@torch.no_grad()
def surprise(m, ids):
    X = torch.tensor([ids], device=DEV)
    lg = m(X[:, :-1]).float().log_softmax(-1)
    return -lg[0].gather(-1, X[0, 1:].unsqueeze(-1)).squeeze(-1)


_A = None
def small_corpus(n=120_000_000):
    global _A
    if _A is None:
        _A = np.asarray(np.memmap("/mnt/kv_cache/pragnosia_data/window2_32k.bin",
                                  dtype=np.uint16, mode="r")[:n])
    return _A


def corpus_consensus(pat, cap=3000):
    """What does the corpus say follows this key, and how decisively? Returns (top_token, share, n_hits)."""
    A = small_corpus()
    if not pat: return None, 0.0, 0
    pos = np.flatnonzero(A[:len(A) - len(pat)] == pat[0])
    for j, t in enumerate(pat[1:], 1):
        if len(pos) == 0: return None, 0.0, 0
        pos = pos[A[pos + j] == t]
        if len(pos) > cap * 4: pos = pos[:cap * 4]
    if len(pos) < 8: return None, 0.0, len(pos)
    nxt = A[pos + len(pat)]
    vals, cnts = np.unique(nxt, return_counts=True)
    i = int(np.argmax(cnts))
    return int(vals[i]), float(cnts[i]) / cnts.sum(), int(len(pos))


@torch.no_grad()
def propose(m, ids, sp, top_k=3, min_share=0.25, min_hits=20, scan=260):
    """IGNORANT-BUT-AGREED: a checkable key where the corpus is decisive and the model is not."""
    order = torch.argsort(sp, descending=True).tolist()
    out = []; seen = set()
    X = torch.tensor([ids], device=DEV)
    lg = m(X[:, :-1]).float().log_softmax(-1)[0]
    for pos in order[:scan]:
        if len(out) >= top_k: break
        a = pos + 1
        if a < 6 or a >= len(ids): continue
        if any(abs(a - s) < 8 for s in seen): continue
        key = ids[a - 4:a]
        if not V.specific(key): continue
        tok, share, n = corpus_consensus(key)
        if tok is None or share < min_share or n < min_hits: continue
        model_lp = float(lg[a - 1, tok])                       # does the model already predict the corpus answer?
        if model_lp > -2.0: continue                            # it already knows this -- nothing to learn
        out.append({'ctx': TOK.decode(ids[max(0, a - 24):a]).strip(),
                    'ans': TOK.decode([tok]),
                    'corpus_share': share, 'hits': n, 'model_logp': model_lp})
        seen.add(a)
    return out


if __name__ == "__main__":
    m, cfg = load()
    np.random.seed(4); dm = DataMix(ctx=1025, verbose=False, only='window2')
    passages = [dm.batch(1)[0][:1024].tolist() for _ in range(12)]
    print(f"IGNORANT-BUT-AGREED PROPOSER over {len(passages)} passages\n", flush=True)
    acc = rej = hold = 0; n = 0; t0 = time.time(); ex = []
    for ids in passages:
        sp = surprise(m, ids)
        for cl in propose(m, ids, sp):
            n += 1
            try:
                verdict, ev = V.verify(cl['ctx'], cl['ans'])
            except Exception:
                verdict, ev = "HOLD", {'n': 0}
            if verdict == "ACCEPT":
                acc += 1
                if len(ex) < 5:
                    ex.append(f"{cl['ctx'][-40:]!r} -> {cl['ans']!r}  "
                              f"corpus {cl['corpus_share']*100:.0f}% of {cl['hits']} | model logP {cl['model_logp']:.1f}")
            elif verdict == "REJECT": rej += 1
            else: hold += 1
    print(f"  proposed {n}  ACCEPT {acc}  REJECT {rej}  HOLD {hold}   "
          f"accept-rate {100*acc/max(1,n):.1f}%   {time.time()-t0:.0f}s", flush=True)
    print(f"\n  previous designs: old (document's answer) 2.9%, checkable-key (document's answer) 0.0%\n", flush=True)
    for e in ex: print(f"    ACCEPTED: {e}", flush=True)
