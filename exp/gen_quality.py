"""LONG-GENERATION QUALITY -- distinct-n, not just repetition.

The claim sheet reported 4-gram repetition alone. Low repetition is necessary but not sufficient: a model can
avoid repeating itself and still produce low-diversity mush, and a reviewer is right to ask for the diversity
side of the measurement. This adds distinct-1/2/3 (unique n-grams over total n-grams) windowed across a long
generation, with TWO reference points on the identical metric and window size:

  REAL TEXT  -- held-out corpus. The ceiling: what a perfect model's output would score.
  GPT-2 124M -- an external model of matched size and matched training budget, generated with the same
                sampler and the same length, so the comparison is not against a number I chose.

Reporting a diversity number without a reference is how a good score and a degenerate score look the same.
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("gen_quality")
cap_gpu(8.0)
import numpy as np, torch
import core as C
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CKPT = os.environ.get("PRAGNOSIA_CKPT", "/mnt/kv_cache/pragnosia_data/p124f.pt")
NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
WIN = 3000


def distinct(ids, n):
    """unique n-grams / total n-grams. 1.0 = never repeats an n-gram, 0.0 = one n-gram forever."""
    if len(ids) <= n: return float('nan')
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(g)) / len(g)


def rep4(ids, k=4):
    """fraction of 4-grams that have appeared before -- the degeneracy measure, complement of distinct-4."""
    if len(ids) <= k: return float('nan')
    g = [tuple(ids[i:i + k]) for i in range(len(ids) - k + 1)]
    seen = set(); r = 0
    for x in g:
        if x in seen: r += 1
        seen.add(x)
    return r / len(g)


def profile(ids, tag):
    """metrics per window, so drift over length is visible rather than averaged away."""
    rows = []
    for w in range(0, len(ids) - WIN + 1, WIN):
        c = ids[w:w + WIN]
        rows.append((w, distinct(c, 1), distinct(c, 2), distinct(c, 3), rep4(c)))
    print(f"\n  {tag}   ({len(ids)} tokens)")
    print(f"  {'window':>12s}{'distinct-1':>12s}{'distinct-2':>12s}{'distinct-3':>12s}{'4g repeat':>12s}")
    for w, d1, d2, d3, r in rows:
        print(f"  {w:>6d}-{w+WIN:<5d}{d1:>12.3f}{d2:>12.3f}{d3:>12.3f}{r:>12.3f}")
    if rows:
        a = np.array([r[1:] for r in rows])
        print(f"  {'mean':>12s}" + "".join(f"{v:>12.3f}" for v in a.mean(0)))
        if len(rows) > 1:
            drift = a[-1] - a[0]
            print(f"  {'drift':>12s}" + "".join(f"{v:>+12.3f}" for v in drift)
                  + "     (first window to last: near zero means it is not degrading with length)")
    return rows


print("=" * 78)
print("  LONG-GENERATION QUALITY -- diversity with references, not repetition alone")
print("=" * 78, flush=True)

# ---------------------------------------------------------------- reference 1: real text
from data_mix import DataMix
# Reference corpus must MATCH what the model is prompted from, or the comparison is rigged. Our 70/30 mix
# contains templated reasoning data that repeats heavily (4g-repeat 0.243 +- 0.238 over 24 windows), while
# pure web text sits at 0.048 +- 0.017. Scoring a web-prompted generation against the mix would make almost
# any output look diverse. Both are printed so the choice is visible rather than assumed.
for _tag, _kw in [("REAL TEXT, pure web (dclm) -- the reference matching our prompt", dict(only='dclm')),
                  ("REAL TEXT, our 70/30 mix -- includes templated reasoning data", dict(lang_frac=0.70))]:
    np.random.seed(0)
    _dm = DataMix(ctx=NGEN + 1, verbose=False, **_kw)
    profile([int(t) for t in _dm.batch(1)[0][:NGEN]], _tag)

# ---------------------------------------------------------------- ours
sd = torch.load(CKPT, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=512, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
# The previously published repetition figure (0.05 to 0.16) was measured with a 256-token REAL document as
# the prompt at temperature 0.9. A first pass here with a 6-token generic prompt at temperature 0.8 scored
# 0.697, which is degenerate. Both cannot be quoted as "the model's repetition" without saying which. So
# sweep the two axes that differ and report the whole matrix, including the bad cells.
from data_mix import DataMix as _DM
np.random.seed(11)
_pd = _DM(ctx=1025, verbose=False, only='dclm')
LONG = [int(t) for t in _pd.batch(1)[0][:256]]
SHORT = TOK.encode(" The following is a detailed account of").ids

summary = []
for pname, pid in [("256-token real document", LONG), ("6-token generic string", SHORT)]:
    for temp in (0.9, 0.8):
        torch.manual_seed(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.generate(torch.tensor([pid], device=DEV), NGEN, temp=temp, top_p=0.95)
        ids = out[0] if isinstance(out, tuple) else out
        if hasattr(ids, 'dim') and ids.dim() == 2: ids = ids[0]
        # generate() returns ONLY the generated tokens, not prompt+generated. Slicing [len(pid):] here threw
        # away 256 real tokens and shifted every 3000-token window boundary, which moved the reported mean by
        # 0.09 on one cell. Take the whole thing, and only strip a prompt if one is actually present.
        g = [int(t) for t in ids]
        if len(g) > NGEN: g = g[len(pid):]
        assert len(g) > NGEN * 0.9, f"generation returned {len(g)} tokens, expected ~{NGEN}"
        rows = profile(g, f"PRAGNOSIA {P/1e6:.0f}M  |  prompt: {pname}  |  temp {temp}, top-p 0.95")
        a = np.array([r[1:] for r in rows])
        summary.append((pname, temp, a.mean(0)))

print("\n" + "=" * 78)
print("  PRAGNOSIA across generation conditions -- means over 3000-token windows")
print("=" * 78)
print(f"  {'prompt':>26s}{'temp':>7s}{'distinct-1':>12s}{'distinct-2':>12s}{'distinct-3':>12s}{'4g repeat':>12s}")
for pname, temp, a in summary:
    print(f"  {pname:>26s}{temp:>7.1f}" + "".join(f"{v:>12.3f}" for v in a))
print("\n  If these cells differ substantially, then a single repetition number is not a property of the model,")
print("  it is a property of the model AND the prompt AND the temperature, and must be quoted with all three.")

# ---------------------------------------------------------------- reference 2: GPT-2, same sampler
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    g = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval().half()
    gt = GPT2TokenizerFast.from_pretrained("gpt2")
    # match our own best condition: a real document as the prompt, temperature 0.9
    gi = gt(TOK.decode(LONG), return_tensors="pt").input_ids.to(DEV)[:, -512:]
    chunks = []
    cur = gi
    with torch.no_grad():
        # GPT-2 has 1024 learned position embeddings and indexes them directly, so prime + new must stay
        # under 1024 or it reads past the table and dies in an indexing assert. Reaching 12k therefore means
        # re-priming, which hands GPT-2 a 512-token running context. That is a REAL limitation of the
        # architecture, not a handicap we imposed: it cannot carry 12k tokens of context at all.
        PRIME, NEW = 512, 500
        while sum(len(c) for c in chunks) < NGEN:
            o = g.generate(cur, max_new_tokens=NEW, do_sample=True, temperature=0.9, top_p=0.95,
                           pad_token_id=gt.eos_token_id)
            new = o[0][cur.shape[1]:]
            if len(new) == 0: break
            chunks.append([int(t) for t in new])
            cur = o[:, -PRIME:]
    gp = [t for c in chunks for t in c][:NGEN]
    profile(gp, "GPT-2 124M  (temp 0.9, re-primed every 500 tokens with a 512-token window: its context ends at 1024)")
except Exception as e:
    print(f"\n  GPT-2 reference unavailable: {type(e).__name__}: {str(e)[:80]}")

print("\n" + "=" * 78)
print("  Read distinct-n against REAL TEXT, not against 1.0. Real text repeats itself too, so a model scoring")
print("  ABOVE the real-text line is not more coherent, it is less predictable than language actually is.")
print("=" * 78, flush=True)
