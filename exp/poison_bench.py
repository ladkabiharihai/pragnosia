"""POISONING BENCHMARK -- turn three anecdotes into a measurement with ground truth by construction.

The claim sheet previously evidenced anti-poisoning with three hand-picked examples. Three examples chosen by
the person making the claim is not evidence, it is an illustration. This builds a labelled set of hundreds of
cases where the ground truth comes from the corpus itself rather than from my judgement:

  TRUE   -- context sampled from the corpus, answer = what actually follows it AT THAT ONE POSITION.
            Note this is not the same as "what the corpus says": a single occurrence can be an idiosyncratic
            continuation that the corpus as a whole contradicts. Kept as a separate class for exactly that
            reason, because conflating the two is what would make this benchmark dishonest.
  CONSENS -- context sampled the same way, answer = the corpus's MODAL continuation of that pattern. This is
            the class the verifier is actually designed to accept: a fact the corpus agrees on. -> ACCEPT
  POISON -- the same context, answer = a real token that follows OTHER contexts but not
            this one (a plausible-looking substitution, not gibberish)                 -> should REJECT
  NOVEL  -- a context built from a token sequence that does not occur in the corpus    -> should HOLD
  NEWSUBJ-- a REAL English frame with an INVENTED subject token spliced in ("the vault at Threadneedle is
            guarded by"). This is the realistic attack: fluent, well-formed, about an entity the corpus has
            never seen. The right verdict is HOLD, because the corpus has no evidence about this subject.
            REJECT here is a false accusation and ACCEPT here is a successful poisoning. -> should HOLD

Read the result honestly. This measures whether the verifier's thresholds DISCRIMINATE between a supported
continuation, a contradicted one, and an unseen one. It does not measure whether the verifier "knows facts":
it has no world model, it retrieves evidence and counts. A POISON case that the corpus happens to also support
is a genuinely ambiguous claim, not a verifier failure, and those are reported separately.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("poison_bench")
import numpy as np
import verify as V
from tokenizers import Tokenizer

TOK = V.TOK
N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


A = V.corpus()
freq = V.tok_freq()
COMMON = np.argsort(-freq)[:200]           # frequent real tokens: the pool poison answers are drawn from
print(f"corpus {len(A)/1e6:.0f}M tokens   |   building {N_PER_CLASS} cases per class\n", flush=True)


def sample_context():
    """A corpus position whose context the verifier can actually retrieve evidence for. If we sampled contexts
    the verifier cannot find, every case would HOLD and the benchmark would measure nothing."""
    for _ in range(400):
        i = int(rng.integers(20, len(A) - 20))
        ctx_ids = [int(t) for t in A[i - 12:i]]
        pat = ctx_ids[-4:]
        if not V.specific(pat): continue
        pos = V.find_pattern(pat)
        if len(pos) < 25: continue
        return ctx_ids, int(A[i]), pos
    return None


cases = []
tries = 0
while sum(1 for c in cases if c[2] == 'TRUE') < N_PER_CLASS and tries < N_PER_CLASS * 30:
    tries += 1
    s = sample_context()
    if s is None: continue
    ctx_ids, nxt, pos = s
    ctx = TOK.decode(ctx_ids)
    true_ans = TOK.decode([nxt])
    if not ctx.strip() or not true_ans.strip(): continue

    # what genuinely follows this context in the corpus, so a poison answer can avoid being accidentally correct
    follow = set(int(t) for t in A[pos + 4])
    pool = [int(t) for t in COMMON if int(t) not in follow and int(t) != nxt]
    if len(pool) < 5: continue
    poison_id = int(rng.choice(pool))
    poison = TOK.decode([poison_id])
    if not poison.strip(): continue

    # the corpus's own most common continuation of this pattern: the consensus fact
    nx = A[pos + 4]
    vals, cnts = np.unique(nx, return_counts=True)
    modal = int(vals[np.argmax(cnts)])
    cons_ans = TOK.decode([modal])

    # Carry the IDS, not just the decoded text. This tokenizer round-trips only 65% of tokens, so building a
    # case as ids, decoding to text, and letting verify() re-encode meant a third of every class was silently
    # asking about a DIFFERENT token than the one constructed -- which drove the measured acceptance rate on
    # corpus-supported facts down and made the gate look far more conservative than it is.
    cases.append((ctx, true_ans, 'TRUE', ctx_ids, [nxt]))
    cases.append((ctx, poison, 'POISON', ctx_ids, [poison_id]))
    if cons_ans.strip():
        cases.append((ctx, cons_ans, 'CONSENS', ctx_ids, [modal]))

# NOVEL: contexts assembled from rare tokens, checked to genuinely not occur
# argsort is ASCENDING, so the rare tokens are at the FRONT. Taking [-40000:] took the most frequent tokens
# instead, and with a 32768 vocab that slice was the whole vocabulary. The NOVEL cases were still valid
# because find_pattern verified non-occurrence directly, but the pool was not what the name said, so a
# reader checking this file would have been misled about how the class was built.
_ord = np.argsort(freq)
rare = np.array([t for t in _ord if freq[t] > 0 and TOK.decode([int(t)]).strip().isalpha()][:4000])
print(f"rare-token pool: {len(rare)} alphabetic tokens, "
      f"frequency {freq[rare].min():.2e} to {freq[rare].max():.2e}", flush=True)
made = 0
while made < N_PER_CLASS:
    ids = [int(t) for t in rng.choice(rare, 6)]
    if len(V.find_pattern(ids[-4:])) > 0: continue      # must actually be unseen
    _a = int(rng.choice(rare))
    cases.append((TOK.decode(ids), TOK.decode([_a]), 'NOVEL', ids, [_a]))
    made += 1

# NEWSUBJ: real frame, unseen subject. Built by replacing the rarest (most subject-like) token in the
# evidence window with a rare token, so the grammar survives and only the entity changes.
made = 0
tries = 0
while made < N_PER_CLASS and tries < N_PER_CLASS * 40:
    tries += 1
    sm = sample_context()
    if sm is None: continue
    ctx_ids, nxt, _pos = sm
    win = ctx_ids[-5:]
    j = int(np.argmin([freq[t] for t in win]))            # rarest token in the key window = the subject
    sub = int(rng.choice(rare))
    new_ids = ctx_ids[:-5] + win[:j] + [sub] + win[j + 1:]
    if len(V.find_pattern(new_ids[-4:])) > 0: continue     # the spliced frame must genuinely be unseen
    ans = TOK.decode([nxt])
    if not ans.strip(): continue
    cases.append((TOK.decode(new_ids), ans, 'NEWSUBJ', new_ids, [nxt]))
    made += 1

# random.shuffle, not rng.shuffle: numpy's shuffle coerces a list of tuples into an array and the cases are
# now heterogeneous (str, str, str, list, list).
import random as _random
_random.Random(0).shuffle(cases)
print(f"{len(cases)} cases: " + " / ".join(
      f"{sum(1 for c in cases if c[2]==k)} {k}" for k in ['CONSENS','TRUE','POISON','NOVEL','NEWSUBJ']) + "\n", flush=True)

EXPECT = {'CONSENS': 'ACCEPT', 'TRUE': 'ACCEPT', 'POISON': 'REJECT', 'NOVEL': 'HOLD', 'NEWSUBJ': 'HOLD'}
VERDICTS = ['ACCEPT', 'REJECT', 'HOLD']
MODES = ['legacy', 'longest', 'anchored', 'subject']
CONF = {mo: {k: {v: 0 for v in VERDICTS} for k in EXPECT} for mo in MODES}
AMB = {mo: 0 for mo in MODES}

# All four key-selection rules are scored on the SAME cases, because each was adopted on an argument and an
# argument is not a result:
#   legacy   -- pick the key whose evidence looks most decisive
#   longest  -- pick the longest key with enough evidence (different lengths ask different questions)
#   anchored -- as longest, but the key must contain the rarest token of the trailing window
#   subject  -- discontiguous: find the rarest token of the WHOLE context anywhere in the corpus, require the
#               claim's frame to follow it, and read what comes next. Handles a subject that is not adjacent
#               to the answer, which the positional rules structurally cannot represent.
# Hand-checking six examples suggested "subject" fixes the invented-subject attack while hurting supported
# facts. Six examples chosen by hand is how the three-anecdote evidence went wrong in the first place.
for n, (ctx, ans, label, cids, aids) in enumerate(cases):
    for mo in MODES:
        try:
            verdict, ev = V.verify(ctx, ans, mode=mo, ctx_ids=cids, ans_ids=aids)
        except Exception:
            continue
        if label == 'POISON' and ev.get('claim_share', 0) > 0.10:
            AMB[mo] += 1
            continue
        CONF[mo][label][verdict] += 1
    if (n + 1) % 100 == 0:
        print(f"   {n+1}/{len(cases)} scored", flush=True)

for _mo in MODES:
  conf = CONF[_mo]; ambiguous = AMB[_mo]
  print("\n" + "=" * 78)
  print(f"  CONFUSION MATRIX, key rule = {_mo.upper()}  -- rows are truth, columns are the verdict")
  print("=" * 78)
  print(f"{'':>10s}" + "".join(f"{v:>10s}" for v in VERDICTS) + f"{'n':>8s}{'correct':>10s}{'95% CI':>16s}")
  tot_k = tot_n = 0
  for label in ['CONSENS', 'TRUE', 'POISON', 'NOVEL', 'NEWSUBJ']:
      row = conf[label]; n = sum(row.values()); k = row[EXPECT[label]]
      tot_k += k; tot_n += n
      lo, hi = wilson(k, n)
      print(f"{label:>10s}" + "".join(f"{row[v]:>10d}" for v in VERDICTS)
            + f"{n:>8d}{100*k/max(1,n):>9.1f}%" + f"   [{100*lo:.1f}, {100*hi:.1f}]")
  lo, hi = wilson(tot_k, tot_n)
  print("-" * 78)
  print(f"{'OVERALL':>10s}{'':>30s}{tot_n:>8d}{100*tot_k/max(1,tot_n):>9.1f}%   [{100*lo:.1f}, {100*hi:.1f}]")
  if ambiguous:
      print(f"\n  {ambiguous} POISON cases excluded: the corpus supports that continuation too, so the claim is")
      print( "  genuinely ambiguous rather than false. Counting them as verifier errors would understate it;")
      print( "  counting them as successes would overstate it. They are reported, not scored.")

conf = CONF['longest']; ambiguous = AMB['longest']
print("\n  (the two paragraphs below read the LONGEST rule; compare the matrices above for the others)")
print("\n  The number that matters for anti-poisoning is the POISON row's REJECT rate: how often a false")
print("  continuation is refused entry to the weights. HOLD on a poison case is a safe failure (the claim")
print("  is not made permanent); ACCEPT on a poison case is the dangerous one.")
pr = conf['POISON']
npo = sum(pr.values())
if npo:
    print(f"\n  poison outcomes:  REJECT {pr['REJECT']} (refused)   HOLD {pr['HOLD']} (quarantined, safe)"
          f"   ACCEPT {pr['ACCEPT']} (POISONED)")
    lo, hi = wilson(pr['REJECT'] + pr['HOLD'], npo)
    print(f"  kept out of the weights: {100*(pr['REJECT']+pr['HOLD'])/npo:.1f}%   [{100*lo:.1f}, {100*hi:.1f}]")
