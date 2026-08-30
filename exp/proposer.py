"""THE PROPOSER GENERATES UNCHECKABLE CLAIMS. Twelve proposals across two live runs, zero reached ACCEPT.

Reading verify.py explains why. Evidence works by taking the LAST 3-5 tokens of the context as a search key, and it
needs that key to (a) contain a CONTENT token, not just function words, and (b) occur at least 20 times in the
corpus. live.py's propose() slices an arbitrary 40-token window around a surprising position, so its key is usually
either function words ("one of the") or a phrase that appears once and can never be evidenced.

A claim you cannot check is not a useful proposal. So the proposer should look for surprise AT a checkable key --
the intrinsic salience signal is unchanged, it is just aimed at spans the individual can actually settle.

Compares ACCEPT/REJECT/HOLD rates for the old and new proposer over the same passages.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("proposer")
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


def propose_old(ids, sp, top_k=3, span=12):
    """what live.py does now: arbitrary window around the most surprising span."""
    if len(sp) < span * 2: return []
    w = sp.unfold(0, span, span // 2).mean(1)
    idx = torch.topk(w, min(top_k, len(w))).indices.tolist()
    out = []
    for i in idx:
        a = i * (span // 2)
        ctx = TOK.decode(ids[max(0, a - 40):a + span])
        ans = TOK.decode(ids[a + span:a + span + 4])
        if len(ctx.strip()) > 20 and len(ans.strip()) > 2:
            out.append({'ctx': ctx.strip(), 'ans': ans.strip()})
    return out


# ---- the corpus index used to pre-check that a key is findable ----
_A = None
def small_corpus(n=120_000_000):
    global _A
    if _A is None:
        _A = np.asarray(np.memmap("/mnt/kv_cache/pragnosia_data/window2_32k.bin",
                                  dtype=np.uint16, mode="r")[:n])
    return _A


def key_hits(pat, cap=400):
    """how many times does this token sequence occur? (small corpus slice, just enough to decide findability)"""
    A = small_corpus()
    if not pat: return 0
    pos = np.flatnonzero(A[:len(A) - len(pat)] == pat[0])
    for j, t in enumerate(pat[1:], 1):
        if len(pos) == 0: return 0
        pos = pos[A[pos + j] == t]
        if len(pos) > cap: return len(pos)
    return len(pos)


def propose_new(ids, sp, top_k=3, want=24, min_hits=8):
    """Surprise AT a CHECKABLE key. Same intrinsic salience signal, aimed where it can be settled:
    the key must contain a content token (verify.specific) and must actually recur in the corpus."""
    order = torch.argsort(sp, descending=True).tolist()
    out = []; seen = set()
    for pos in order[:want * 8]:
        if len(out) >= top_k: break
        a = pos + 1                                   # the surprising token itself is the answer
        if a < 6 or a >= len(ids): continue
        if any(abs(a - s) < 8 for s in seen): continue
        key = ids[a - 4:a]
        if not V.specific(key): continue              # reject function-word keys
        if key_hits(key) < min_hits: continue         # reject keys the corpus cannot evidence
        ctx = TOK.decode(ids[max(0, a - 24):a])
        ans = TOK.decode([ids[a]])
        if len(ctx.strip()) < 10 or not ans.strip(): continue
        out.append({'ctx': ctx.strip(), 'ans': ans, 'surprise': float(sp[pos])})
        seen.add(a)
    return out


def run(name, fn, m, passages):
    acc = rej = hold = 0; n_prop = 0; t0 = time.time()
    examples = []
    for ids in passages:
        sp = surprise(m, ids)
        for cl in fn(ids, sp):
            n_prop += 1
            try:
                verdict, ev = V.verify(cl['ctx'], cl['ans'])
            except Exception:
                verdict, ev = "HOLD", {'n': 0}
            if verdict == "ACCEPT":
                acc += 1
                if len(examples) < 3:
                    examples.append(f"{cl['ctx'][-46:]!r} -> {cl['ans']!r}  (n={ev.get('n',0)})")
            elif verdict == "REJECT": rej += 1
            else: hold += 1
    print(f"  {name:22s} proposed {n_prop:3d}  ACCEPT {acc:3d}  REJECT {rej:3d}  HOLD {hold:3d}   "
          f"accept-rate {100*acc/max(1,n_prop):5.1f}%   {time.time()-t0:5.1f}s", flush=True)
    for e in examples: print(f"      accepted: {e}", flush=True)
    return acc, n_prop


if __name__ == "__main__":
    m, cfg = load()
    np.random.seed(4); dm = DataMix(ctx=1025, verbose=False, only='window2')
    passages = [dm.batch(1)[0][:1024].tolist() for _ in range(12)]
    print(f"PROPOSER COMPARISON over {len(passages)} passages of 1024 tokens\n", flush=True)
    run("old (arbitrary span)", propose_old, m, passages)
    run("new (checkable key)", propose_new, m, passages)
