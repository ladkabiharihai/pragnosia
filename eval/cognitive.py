"""GOAL 2, A COGNITIVE MODEL, MEASURED AGAINST HUMANS (2026-10-13; user: "compare it against human not other llm"). Classic
tasks of cognitive psychology with published human results, every item a FRESH variant (new names, numbers, letters, words) so
no famous wording can have been memorised. Two readings per task: the LEVEL (accuracy against the human figure) and the PATTERN
(does it err where humans err: primacy / recency, belief bias, matching bias, the intuitive CRT answer, the conjunction fallacy).
v2: options labelled (a)-(d) and scored over every ordering; yes/no tasks read threshold-free (AUC); span by the standard rule; recall tasks generated
greedily. Human figures, from the papers named beside each task (to be checked against the sources before any report):
  digit span          forward ~7 (Miller 1956), backward ~5 (adult norms)
  free recall         15 words, immediate: U-shaped serial position curve, primacy and recency (Murdock 1962)
  2-back              ~90% correct (letters; adult n-back norms, approximate)
  Wason selection     abstract rule ~10% correct, matching error "P and Q" common (Wason 1968); social rule ~75% (Griggs & Cox 1982)
  belief bias         conclusions accepted: valid-believable 89%, valid-unbelievable 56%, invalid-believable 71%,
                      invalid-unbelievable 10% (Evans, Barston & Pollard 1983)
  CRT                 ~41% of items right (mean 1.24 of 3), most errors the intuitive answer (Frederick 2005)
  conjunction         ~85% judge "A and B" more probable than "A" (Tversky & Kahneman 1983)
  false belief        adults ~100% (Baron-Cohen, Leslie & Frith 1985: typically developing 4-year-olds 85%)
usage: eval/cognitive.py <ckpt>=<label> ...      N=<items per task cell, default 40>"""
import os, sys, re, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(6.0)
import numpy as np
from src.bench import Ours

N = int(os.environ.get("N", "40")); R = random.Random(20261013)
NAMES = ["Asha", "Bruno", "Chen", "Dalia", "Emeka", "Freya", "Goran", "Hana", "Ilya", "Jonas", "Kemi", "Lior", "Mira", "Nadim",
         "Oona", "Pavel", "Rhea", "Soren", "Tala", "Umar", "Vera", "Wen", "Yusuf", "Zora"]
WORDS = ["apple", "river", "candle", "horse", "window", "garden", "pencil", "mountain", "bottle", "carpet", "lemon", "tiger",
         "button", "castle", "ladder", "violin", "basket", "rocket", "mirror", "pillow", "wagon", "thunder", "needle", "island",
         "feather", "hammer", "orange", "rabbit", "kettle", "anchor", "blanket", "cherry", "dragon", "engine", "forest", "glove",
         "helmet", "jacket", "lantern", "marble", "napkin", "ocean", "parrot", "quilt", "saddle", "teapot", "umbrella", "valley"]
PAIRS = [("dogs", "animals"), ("roses", "flowers"), ("oaks", "trees"), ("salmon", "fish"), ("hammers", "tools"),
         ("sparrows", "birds"), ("cars", "vehicles"), ("apples", "fruits"), ("violins", "instruments"), ("shirts", "clothes"),
         ("carrots", "vegetables"), ("diamonds", "stones"), ("novels", "books"), ("cats", "mammals"), ("tulips", "plants"),
         ("trucks", "machines"), ("eagles", "animals"), ("sofas", "furniture"), ("lakes", "bodies of water"), ("pianos", "objects")]
NONCE = ["wampets", "dorbels", "fesks", "glimmers", "tavons", "brulks", "zinnets", "molps", "krests", "vandles"]


def mc(M, prompt, options):
    # PMI SCORING (Holtzman et al. 2021): each option's log-probability after the task, minus its log-probability after the bare
    # answer cue ("Answer:" ... to the end of the prompt). Raw or length-normalised scores carried the options' own priors: in the
    # smoke test the base model said "yes" to every syllogism and chose the longest Wason option every time.
    cue = prompt[prompt.rfind("Answer:"):]
    pre, pre0 = M.prefix(M.encode(prompt)), M.prefix(M.encode(cue))
    return int(np.argmax([M.continuation_lp(pre, M.encode(o)) - M.continuation_lp(pre0, M.encode(o)) for o in options]))


def mc_labelled(M, stem, options):
    # v2 (2026-10-13): the options as (a)-(d), every rotation scored and each option's label score summed over the positions it
    # took -- v1 scored the option texts, and PMI rewarded the option that copied the prompt (Wason's "all four cards"), while a
    # fixed order left exactly-50% position choices on two-option tasks. Returns the option index chosen.
    k = len(options); tot = np.zeros(k)
    for r in range(k):
        order = options[r:] + options[:r]
        p = stem + "\n" + "\n".join(f"({'abcd'[i]}) {o}" for i, o in enumerate(order)) + "\nAnswer: ("
        cue = "Answer: ("; pre, pre0 = M.prefix(M.encode(p)), M.prefix(M.encode(cue))
        for i in range(k):
            lab = M.encode(f"{'abcd'[i]})")
            tot[(i + r) % k] += M.continuation_lp(pre, lab) - M.continuation_lp(pre0, lab)
    return int(np.argmax(tot))


def margin(M, prompt):
    # yes-minus-no PMI margin: a threshold-free reading of a yes/no task (v1's constant "No" hid any sensitivity)
    cue = prompt[prompt.rfind("Answer:"):]; pre, pre0 = M.prefix(M.encode(prompt)), M.prefix(M.encode(cue))
    sc = [M.continuation_lp(pre, M.encode(o)) - M.continuation_lp(pre0, M.encode(o)) for o in (" Yes", " No")]
    return sc[0] - sc[1]


def auc(pos, neg):
    # P(a positive item's margin > a negative item's): 0.5 = no sensitivity, 1.0 = perfect, whatever the response bias
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def gen(M, prompt, n):
    out = M.generate(M.encode(prompt), n, temp=0.0, stop=8191)
    if 8191 in out: out = out[:out.index(8191)]
    return M.decode(out)


def digit_span(M, backward):
    # ZERO-SHOT (the smoke test: with two worked examples the model repeated the LAST EXAMPLE's digits for lists under 11,
    # copying whatever followed the previous "Repeat:"); the cue restates the list's own phrase, so forward recall is a copy
    # of the list and backward recall cannot be one
    cue = "In reverse order, the digits are" if backward else "Repeated back, the digits are"
    acc = {}
    for L in range(3, 16):
        ok = 0
        for _ in range(N // 4):
            s = [str(R.randrange(10)) for _ in range(L)]
            # BRACKETS: "are 9 7 3" -> "7 3": the first digit after a word tokenises differently from the rest, and the copy started
            # one late; inside brackets the first digit has the same boundary in the list and in the answer
            out = re.findall(r"\d", gen(M, f"The digits are [{' '.join(s)}]. {cue} [", 2 * L + 2).split("]")[0])
            ok += out == (s[::-1] if backward else s)
        acc[L] = ok / (N // 4)
    span = 2                                            # the standard rule: the longest length passed with every shorter one passed
    for L in sorted(acc):
        if acc[L] < 0.5: break
        span = L
    return span, acc


def free_recall(M):
    pos = np.zeros(15); n = 0
    for _ in range(N // 2):
        ws = R.sample(WORDS, 15)
        out = gen(M, "Study this list of words: " + ", ".join(ws) + ".\nNow write down every word you remember from the list:", 80).lower()
        got = set(re.findall(r"[a-z]+", out))
        pos += np.array([w in got for w in ws]); n += 1
    return pos / n


def two_back(M):
    ms = {True: [], False: []}
    for i in range(N * 2):
        seq = [R.choice("BCDFG") for _ in range(R.randint(6, 14))]
        match = i % 2 == 0
        seq[-1] = seq[-3] if match else R.choice([c for c in "BCDFG" if c != seq[-3]])
        p = ("Letters were shown one at a time: " + " ".join(seq) + ".\nQuestion: Is the last letter the same as the letter two "
             "positions before it? Answer:")
        ms[match].append(margin(M, p))
    return auc(ms[True], ms[False])


def wason(M, social):
    pick = np.zeros(4)
    for _ in range(N):
        if social:
            act, alt, need, lack = R.choice([("entering the lab", "staying outside", "wearing goggles", "not wearing goggles"),
                ("driving a car", "riding a bicycle", "holding a licence", "holding no licence"),
                ("boarding the ferry", "waiting on the dock", "holding a ticket", "holding no ticket"),
                ("drinking beer", "drinking juice", "over 18", "under 18"),
                ("using the pool", "sitting in the cafe", "wearing a cap", "not wearing a cap")])
            p = (f"Four people are each described by a card: one side says what the person is doing, the other side says whether "
                 f"they are {need} or {lack}. The rule: if a person is {act}, then they must be {need}. The visible sides are: "
                 f"{act}, {alt}, {need}, {lack}. Which cards must be turned over to check whether anyone is breaking the rule?\nAnswer: the cards")
            opts = [f" {act} and {lack}", f" {act} and {need}", f" {act} only", f" {act}, {alt}, {need} and {lack}"]
        else:
            v, c = R.choice("AEIOU"), R.choice("BDFGKMPRT"); e, o = R.choice("2468"), R.choice("3579")
            p = (f"Four cards each have a letter on one side and a number on the other side. The rule: if a card has a vowel on one "
                 f"side, then it has an even number on the other side. The visible faces are: {v}, {c}, {e}, {o}. Which cards must be "
                 f"turned over to check whether the rule is broken?\nAnswer: the cards")
            opts = [f" {v} and {o}", f" {v} and {e}", f" {v} only", f" {v}, {c}, {e} and {o}"]
        pick[mc_labelled(M, p.rsplit("\nAnswer:", 1)[0], [o.strip() for o in opts])] += 1
    return pick / N                     # [right (P, not-Q), matching (P, Q), P only, all]


def belief_bias(M):
    mg = {}
    for valid in (True, False):
        for believable in (True, False):
            mg[(valid, believable)] = []
            for _ in range(N):
                sub, sup = R.choice(PAIRS); m = R.choice(NONCE)
                x, y = (sub, sup) if believable else (sup, sub)          # conclusion "all x are y"
                prem = f"All {x} are {m}. All {m} are {y}." if valid else f"All {x} are {m}. All {y} are {m}."
                p = (f"Premises: {prem}\nConclusion: All {x} are {y}.\nQuestion: Does the conclusion follow logically from the "
                     f"premises, whether or not it is true? Answer:")
                mg[(valid, believable)].append(margin(M, p))
    allm = np.concatenate([np.array(v) for v in mg.values()]); thr = np.median(allm)
    acc = {k: float((np.array(v) > thr).mean()) for k, v in mg.items()}     # acceptance with the threshold at its own median
    logic = auc(mg[(True, True)] + mg[(True, False)], mg[(False, True)] + mg[(False, False)])
    belief = auc(mg[(True, True)] + mg[(False, True)], mg[(True, False)] + mg[(False, False)])
    return acc, logic, belief


def crt(M):
    right = intuitive = 0; n = 0
    for _ in range(N):
        k = R.randrange(3)
        if k == 0:
            x = R.randint(3, 20) * 5; d = R.choice([100, 200, 50]); tot = d + 2 * x; a, b = R.choice([("bat", "ball"), ("pen", "cap"), ("book", "bookmark")])
            p = f"A {a} and a {b} cost {tot} cents in total. The {a} costs {d} cents more than the {b}. How many cents does the {b} cost?\nAnswer:"
            ans, intu = x, tot - d
        elif k == 1:
            n_, m_ = R.choice([5, 4, 6, 3]), R.choice([100, 50, 80, 200])
            p = f"If {n_} machines take {n_} minutes to make {n_} widgets, how many minutes would {m_} machines take to make {m_} widgets?\nAnswer:"
            ans, intu = n_, m_
        else:
            D = R.choice([48, 40, 60, 30, 24])
            p = f"In a lake there is a patch of lily pads. Every day the patch doubles in size. If it takes {D} days for the patch to cover the whole lake, how many days does it take to cover half of the lake?\nAnswer:"
            ans, intu = D - 1, D // 2
        others = [v for v in {ans + 7, intu * 2, ans * 3, intu + 11} if v not in (ans, intu)][:2]
        opts = [ans, intu] + others
        j = mc_labelled(M, p.rsplit("\nAnswer:", 1)[0], [str(v) for v in opts])
        right += j == 0; intuitive += j == 1; n += 1
    return right / n, intuitive / n


def conjunction(M):
    traits = [("deeply concerned with discrimination and social justice, and took part in demonstrations", "a bank teller", "active in the feminist movement"),
              ("fascinated by computers since childhood and spends evenings writing software", "a schoolteacher", "a programmer in their free time"),
              ("a passionate climber who spends every weekend in the mountains", "an accountant", "a member of an alpine club"),
              ("devoted to animals and volunteers at a shelter", "a librarian", "a vegetarian"),
              ("obsessed with music, playing in two bands since school", "an insurance clerk", "a part-time guitar teacher")]
    conj = 0
    for _ in range(N):
        nm = R.choice(NAMES); desc, a, b = R.choice(traits); age = R.randint(28, 40)
        o1, o2 = f"{nm} is {a}.", f"{nm} is {a} and is {b}."
        stem = f"{nm} is {age} years old, single, outspoken and very bright, {desc}.\nWhich is more probable?"
        conj += mc_labelled(M, stem, [o1, o2]) == 1
    return conj / N


def false_belief(M):
    out = {}
    for seen in (False, True):
        ok = 0
        for _ in range(N):
            a, b = R.sample(NAMES, 2); obj = R.choice(["ball", "key", "coin", "marble", "ring", "pen"])
            l1, l2 = R.sample(["basket", "box", "drawer", "cupboard", "bag", "jar"], 2)
            mid = (f"{a} stays in the room and watches {b} move the {obj} to the {l2}." if seen else
                   f"{a} leaves the room. While {a} is away, {b} moves the {obj} to the {l2}. Then {a} comes back.")
            stem = f"{a} puts the {obj} in the {l1}. {mid}\nQuestion: Where will {a} look for the {obj}?"
            ok += mc_labelled(M, stem, [f"in the {l1}", f"in the {l2}"]) == (1 if seen else 0)
        out[seen] = ok / N
    return out


for spec in sys.argv[1:]:
    ck, lab = spec.split("=") if "=" in spec else (spec, os.path.basename(spec))
    M = Ours(ck, "/mnt/kv_cache/tok_v8192.json"); R.seed(20261013)        # every model sees the same items
    print(f"\n######## {lab} ({ck}); N={N} per cell", flush=True)
    sf, af = digit_span(M, False); sb, ab = digit_span(M, True)
    print(f"DIGIT SPAN       forward {sf}, backward {sb}   (humans ~7 / ~5)   by length F " +
          " ".join(f"{L}:{100*a:.0f}" for L, a in af.items()) + "  B " + " ".join(f"{L}:{100*a:.0f}" for L, a in ab.items()), flush=True)
    fr = free_recall(M)
    print(f"FREE RECALL      by position 1-15: " + " ".join(f"{100*v:.0f}" for v in fr) +
          f"   first3 {100*fr[:3].mean():.0f} / middle {100*fr[5:10].mean():.0f} / last3 {100*fr[-3:].mean():.0f}   (humans U-shaped)", flush=True)
    print(f"2-BACK           sensitivity (AUC, match vs non-match) {two_back(M):.2f}   (humans ~90% right; no sensitivity 0.50)", flush=True)
    wa, ws = wason(M, False), wason(M, True)
    print(f"WASON abstract   right {100*wa[0]:.0f}%, matching (P and Q) {100*wa[1]:.0f}%, P only {100*wa[2]:.0f}%, all {100*wa[3]:.0f}%   (humans ~10% right)", flush=True)
    print(f"WASON social     right {100*ws[0]:.0f}%, matching {100*ws[1]:.0f}%, P only {100*ws[2]:.0f}%, all {100*ws[3]:.0f}%   (humans ~75% right)", flush=True)
    bb, lg, bl = belief_bias(M)
    print(f"BELIEF BIAS      accepted at its own median: valid-believable {100*bb[(True,True)]:.0f}, valid-unbelievable {100*bb[(True,False)]:.0f}, "
          f"invalid-believable {100*bb[(False,True)]:.0f}, invalid-unbelievable {100*bb[(False,False)]:.0f}   (humans 89 / 56 / 71 / 10); "
          f"sensitivity to logic {lg:.2f}, to belief {bl:.2f} (AUC; none 0.50)", flush=True)
    cr, ci = crt(M)
    print(f"CRT              right {100*cr:.0f}%, intuitive error {100*ci:.0f}%   (humans ~41% right, most errors intuitive; chance 25)", flush=True)
    print(f"CONJUNCTION      'A and B' judged more probable: {100*conjunction(M):.0f}%   (humans ~85; chance 50)", flush=True)
    fb = false_belief(M)
    print(f"FALSE BELIEF     false-belief stories {100*fb[False]:.0f}% right, true-belief controls {100*fb[True]:.0f}%   (adults ~100 / ~100; chance 50)", flush=True)
