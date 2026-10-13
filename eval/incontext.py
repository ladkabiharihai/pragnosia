"""VISION GOAL 7, THE WAY THIS ARCHITECTURE IS BUILT FOR: LEARNING BY READING, NOT BY TRAINING.

Continual learning by fine-tuning failed for both models (eval/continual.py): neither kept what it knew, and the
carrier did not even learn the new facts. But a model with a fixed-size state and no context limit has another
way to learn: read the new facts, and answer from its memory. The weights never change, so nothing already
known can be forgotten -- by construction, not by a regulariser.

The same 120 invented facts as eval/continual.py, one sentence each, placed in the context; then a question in
a phrasing that was not in the context. Four-way, chance 25%. Measured at growing numbers of facts, and for
the largest set by how far back the fact sits. GPT-2 is measured wherever the facts fit in its 1,024 positions,
and marked where they do not.
usage: eval/incontext.py [checkpoint]"""
import sys, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import numpy as np, torch
from src.bench import Ours, GPT2

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
rng = random.Random(1234)
SYL = ["vel", "dra", "osk", "arth", "mir", "tan", "quel", "lin", "zor", "va", "bre", "nth", "kal", "iso", "rum",
       "pel", "dor", "gha", "sen", "tri", "wex", "ulm", "fae", "rok", "yst", "cor", "nib", "sha", "lom", "dex"]
seen = set()
def fresh(n):
    while True:
        w = "".join(rng.choice(SYL) for _ in range(n)).capitalize()
        if w not in seen and len(w) >= 5: seen.add(w); return w
countries = [fresh(3) for _ in range(60)]; capitals = [fresh(2) for _ in range(60)]
machines = [fresh(2) for _ in range(60)]; people = [f"{fresh(2)} {fresh(2)}" for _ in range(60)]
FACTS = []                                   # (sentence placed in context, question, answer, distractors)
for i in range(60):
    FACTS.append((f"The capital of {countries[i]} is {capitals[i]}.",
                  f"Question: which city is the capital of {countries[i]}? Answer:", f" {capitals[i]}",
                  [f" {o}" for o in rng.sample([x for x in capitals if x != capitals[i]], 3)]))
    FACTS.append((f"The {machines[i]} engine was invented by {people[i]}.",
                  f"Question: who invented the {machines[i]} engine? Answer:", f" {people[i]}",
                  [f" {o}" for o in rng.sample([x for x in people if x != people[i]], 3)]))
# INCONTEXT_KINDS=new (2026-10-12): two KINDS OF FACT no training builder writes (eval/build_recall.py: capital, engine, birthplace,
# dog owner, code, colour) and no self-improvement round labels (eval/selfimprove.py: capital, engine, birthplace, code): rivers
# flowing into seas, guild leaders. Asks whether the self-improved lookup carries to kinds it never saw, not only to new names.
import os
KEY_ = "capital"
if os.environ.get("INCONTEXT_KINDS") == "new":
    rivers = [fresh(3) for _ in range(60)]; seas = [fresh(2) for _ in range(60)]
    guilds = [fresh(2) for _ in range(60)]; leaders = [f"{fresh(2)} {fresh(2)}" for _ in range(60)]
    FACTS, KEY_ = [], "river"
    for i in range(60):
        FACTS.append((f"The {rivers[i]} river flows into the {seas[i]} Sea.",
                      f"Question: into which sea does the {rivers[i]} river flow? Answer:", f" {seas[i]}",
                      [f" {o}" for o in rng.sample([x for x in seas if x != seas[i]], 3)]))
        FACTS.append((f"{leaders[i]} leads the {guilds[i]} guild.",
                      f"Question: who is the leader of the {guilds[i]} guild? Answer:", f" {leaders[i]}",
                      [f" {o}" for o in rng.sample([x for x in leaders if x != leaders[i]], 3)]))
order = list(range(len(FACTS))); random.Random(9).shuffle(order)
# HARD mode (argv[2] == "hard"): every wrong option is ANOTHER ANSWER FROM THE SAME CONTEXT. Measured 2026-09-24:
# in the default mode the wrong options were drawn from all 60 names, so with 10 facts in context most of them
# had never appeared and "pick the name that was in the text" scored well without any lookup. Here only lookup
# scores: every option was in the text.
HARD = len(sys.argv) > 2 and sys.argv[2] == "hard"
def hard(sub):
    out = []
    for f in sub:
        pool = [g[2] for g in sub if g is not f and (KEY_ in g[1]) == (KEY_ in f[1])]
        out.append((f[0], f[1], f[2], rng.sample(pool, 3) if len(pool) >= 3 else f[3]))
    return out


def ask(model, enc, context_ids, q, ans, wrongs, max_ctx=None):
    lps = []
    for c in [ans] + wrongs:
        ids = context_ids + enc(" " + q) + enc(c)
        if max_ctx and len(ids) > max_ctx: return None
        lg = model.logits_last(ids, 1).float().log_softmax(-1)
        nc = min(len(enc(c)), lg.shape[1] - 1)
        y = torch.tensor(ids[-nc:], device=lg.device)
        lps.append(float(lg[0, -nc - 1:-1].gather(-1, y.unsqueeze(-1)).sum()))
    return int(np.argmax(lps)) == 0


import os
models = [("Carrier", Ours(CK, "/mnt/kv_cache/tok_v8192.json"), None)] + \
         ([] if os.environ.get("INCONTEXT_GPT2") == "0" else [("GPT-2 124M", GPT2(), 1024)])
if len(sys.argv) > 2 and sys.argv[2] == "needle":
    # IN-DISTRIBUTION NEEDLES, BY DISTANCE (2026-10-02): the passkey test is out of distribution on purpose (repeated
    # filler, its own wording). This asks the trained kind instead -- real prose filler, eval/build_recall.py's needle
    # phrasings -- with ONE needle placed exactly D tokens before the question, so "can it retrieve at all past the
    # 512-token attention window" is separated from "does it generalise to the passkey format". Greedy, exact match.
    import numpy as np
    src_ = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
    NEED_ = [("The secret code is {V}.", "What is the secret code?"),
             ("The combination for the {N} lock is {V}.", "What is the combination for the {N} lock?"),
             ("{N}'s favourite number is {V}.", "What is {N}'s favourite number?"),
             ("The {N} team meets in room {V}.", "Which room does the {N} team meet in?")]
    NM = ["Alice", "Omar", "Priya", "Lukas", "Mei", "Tomas", "Aisha", "Diego"]
    lab, m, _ = models[0]; r_ = random.Random(4242); rn_ = np.random.default_rng(4242)
    DIST = [int(v) for v in os.environ.get("NEEDLE_DIST", "64,256,1024,4096,16384").split(",")]
    N_ = int(os.environ.get("NEEDLE_N", "24"))
    print(f"IN-DISTRIBUTION NEEDLE, one needle D tokens before the question, real prose filler, greedy exact; {CK}")
    for D in DIST:
        ok = 0; shown = False
        for t in range(N_):
            j = r_.randrange(len(NEED_)); n_ = r_.choice(NM); v_ = str(r_.randint(100, 999999))
            st = m.encode(" " + NEED_[j][0].format(N=n_, V=v_) + " ")
            q = m.encode("\n\nQuestion: " + NEED_[j][1].format(N=n_) + " Answer:")
            al = int(os.environ.get("NEEDLE_ALIGN", "0"))
            # at least one alignment unit of leading filler, so trimming to a multiple of it never cuts the needle off
            pre = max(int(os.environ.get("NEEDLE_PRE", "1024")), al)
            o = int(rn_.integers(0, len(src_) - 3 * (D + pre)))
            f = np.asarray(src_[o:o + 3 * (D + pre)]); f = f[f != 8191].tolist()
            ids = f[:pre] + st + f[pre:pre + D] + q
            # NEEDLE_ALIGN=512: trim the leading filler so the prompt is a multiple of 512 tokens. A prefill of such a
            # length runs the windowed attention BLOCK-LOCAL, exactly as trained, so nothing past the block can be seen
            # by attention and a hit at D >= 512 can only come from the carrier. Any other length runs it as FULL
            # causal attention over the window, which training never did (found 2026-10-02).
            if al: ids = ids[len(ids) % al:]
            out = m.decode(m.generate(ids, 8, temp=0.0))
            hit = v_ in out.replace(" ", "")[:12]; ok += hit
            if not shown and not hit: print(f"   D={D} miss: wanted {v_}, wrote {out!r}"); shown = True
        print(f"  distance {D:>6d} tokens: {100*ok/N_:5.1f}% ({ok}/{N_})", flush=True)
    sys.exit(0)
if len(sys.argv) > 2 and sys.argv[2] == "passkey":
    # PASSKEY RETRIEVAL (Mohtashami & Jaggi 2023), the standard long-context check and NOTHING like our training
    # data (eval/build_recall.py writes facts and questions; this writes filler and one number): a five-digit key in
    # repeated filler at a chosen depth of a context of chosen length, then "What is the pass key? The pass key is",
    # greedy, scored on the exact five digits. GPT-2 wherever it fits in 1,024 positions.
    FILL = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
    LENS = [int(v) for v in os.environ.get("PASSKEY_LENS", "512,2048,8192,32768").split(",")]
    DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0); REPS = int(os.environ.get("PASSKEY_REPS", "4"))
    print(f"PASSKEY: five digits in filler, greedy exact match, {len(DEPTHS)} depths x {REPS} keys per length; {CK}")
    print(f"{'tokens':<10s}" + "".join(f"{lab:>16s}" for lab, _, _ in models) + "   (ours by depth 0 / 25 / 50 / 75 / 100%)")
    r_ = random.Random(77)
    for L in LENS:
        row = f"{L:<10d}"; bydepth = []
        for lab, m, cap in models:
            enc = m.encode if hasattr(m, "encode") else m.tok.encode
            dec = m.decode if hasattr(m, "decode") else m.tok.decode
            fill = enc(FILL); k = tot = 0; fits = True; dk = []
            for dpt in DEPTHS:
                kd = 0
                for rep in range(REPS):
                    key = str(r_.randint(10000, 99999))
                    needle = enc(f" The pass key is {key}. Remember it. {key} is the pass key. ")
                    q = enc(" What is the pass key? The pass key is")
                    n_fill = max(0, L - len(needle) - len(q))
                    body = (fill * (n_fill // len(fill) + 1))[:n_fill]
                    at = int(dpt * n_fill)
                    ids = body[:at] + needle + body[at:] + q
                    if cap and len(ids) > cap: fits = False; break
                    out = dec(m.generate(ids, 8, temp=0.0)) if hasattr(m, "generate") else ""
                    ok = key in out.replace(" ", "")[:12]
                    if not ok and rep == 0 and lab == "Carrier" and dpt in (0.5, 1.0): print(f"   L={L} depth {dpt}: wanted {key}, wrote {out!r}")
                    k += ok; kd += ok; tot += 1
                if not fits: break
                dk.append(kd)
            row += f"{(f'{100*k/tot:.0f}% ({k}/{tot})' if fits else 'does not fit'):>16s}"
            if lab == "Carrier": bydepth = dk
        print(row + "   " + " / ".join(f"{d_}/{REPS}" for d_ in bydepth), flush=True)
    sys.exit(0)
if len(sys.argv) > 2 and sys.argv[2] == "powered":
    # POWERED (2026-10-01): the rows above ask n questions about ONE context of n facts, so "10 facts" is 10
    # questions and 20% vs 40% is 2/10 vs 4/10 -- noise. Here every n gets TRIALS fresh contexts (a new random
    # subset, a new order), up to 10 questions each, hard distractors, and a 95% interval on every number.
    TRIALS = int(os.environ.get("INCONTEXT_TRIALS", "50"))
    def ci(k, n): p_ = k / n; return f"{100 * p_:5.1f}% +-{196 * (p_ * (1 - p_) / n) ** 0.5:4.1f}"
    print(f"POWERED, HARD: {TRIALS} contexts per row, up to 10 questions each; chance 25%; {CK}")
    print(f"{'facts in context':<18s}" + "".join(f"{lab:>30s}" for lab, _, _ in models))
    pos_k, pos_n = np.zeros(4), np.zeros(4)
    for n in (10, 30, 60, 120):
        row = f"{n:<18d}"
        for lab, m, cap in models:
            enc = m.encode if hasattr(m, "encode") else m.tok.encode
            r_ = random.Random(1000 + n); k = tot = 0; fits = True
            for t in range(TRIALS):
                sub = hard(r_.sample(FACTS, n))
                ctx = enc(" ".join(f[0] for f in sub))
                for qi in r_.sample(range(n), min(10, n)):
                    f = sub[qi]; res = ask(m, enc, ctx, f[1], f[2], f[3], cap)
                    if res is None: fits = False; break
                    k += res; tot += 1
                    if lab == "Carrier" and n == 120: pos_k[qi * 4 // n] += res; pos_n[qi * 4 // n] += 1
                if not fits: break
            row += f"{ci(k, tot) + f' (n={tot})':>30s}" if fits else f"{'does not fit':>30s}"
        print(row, flush=True)
    print("Carrier at 120 facts, by where the fact sits: " + "   ".join(
        f"{['earliest', 'early-mid', 'late-mid', 'latest'][j]} {ci(pos_k[j], max(1, pos_n[j]))}" for j in range(4)))
    sys.exit(0)
print("Facts are read, not trained on: the weights never change, so nothing known can be forgotten."
      + ("  HARD: every wrong option is another answer from the same context." if HARD else "") + "\n")
print(f"{'facts in context':<18s}" + "".join(f"{lab:>22s}" for lab, _, _ in models))
for n in (10, 30, 60, 120):
    sub = [FACTS[i] for i in order[:n]]
    if HARD: sub = hard(sub)
    row = f"{n:<18d}"
    for lab, m, cap in models:
        enc = m.encode if hasattr(m, "encode") else m.tok.encode
        ctx = enc(" ".join(f[0] for f in sub))
        res = [ask(m, enc, ctx, f[1], f[2], f[3], cap) for f in sub]
        if any(r is None for r in res):
            row += f"{'does not fit (' + str(len(ctx)) + ' tok)':>22s}"
        else:
            row += f"{100 * np.mean(res):>13.1f}% ({len(ctx):>4d} tok)"
    print(row, flush=True)
print("\nNo facts in context at all (what each model guesses cold): ", end="")
for lab, m, cap in models:
    enc = m.encode if hasattr(m, "encode") else m.tok.encode
    print(f"{lab} {100 * np.mean([ask(m, enc, [], f[1], f[2], f[3], cap) for f in FACTS]):.1f}%   ", end="")
print("(chance 25%)")
lab, m, _ = models[0]
sub = [FACTS[i] for i in order]
if HARD: sub = hard(sub)
ctx = m.encode(" ".join(f[0] for f in sub))
res = np.array([ask(m, m.encode, ctx, f[1], f[2], f[3]) for f in sub])
print(f"\nCarrier with all 120 facts in context ({len(ctx)} tokens), accuracy by how far back the fact sits:")
for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120)):
    print(f"   facts {lo + 1:>3d}-{hi:<3d} (the {'most recent' if lo == 90 else 'earliest' if lo == 0 else 'middle'} quarter): "
          f"{100 * res[lo:hi].mean():.1f}%")
