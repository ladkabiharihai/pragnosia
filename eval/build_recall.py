"""TRAINING DATA FOR IN-CONTEXT LOOKUP (vision goal 9). Measured 2026-09-24: with 120 invented facts in context and
every wrong option also in the context, the model is at chance even for the facts inside its attention window, so
it cannot yet look a fact up among many; it only notices which names appeared. Gated delta-rule models learn this
lookup well once they see it (Arora et al. 2023, "Zoology"); ordinary text almost never asks for it.

Each document lists 10-150 invented facts of six kinds in varied wording, then asks about a random subset in a
DIFFERENT wording, answers given. Documents run from a few hundred to several thousand tokens, so most answers sit
beyond the 512-token attention window and the carrier's state has to carry them. The syllables differ from the
ones eval/incontext.py builds its names from, so no test fact can occur here.
usage: eval/build_recall.py <out_dir> <million_tokens> [needle]"""
import os, sys, random
import numpy as np
from tokenizers import Tokenizer

OUT, MTOK = sys.argv[1], float(sys.argv[2])
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
EOD = 8191
SYL = ["bax", "cun", "fim", "gor", "hep", "jal", "kiv", "lup", "mox", "nar", "pim", "rud", "sov", "tek", "wun",
       "yal", "zep", "bri", "clo", "fen", "glo", "hax", "jor", "kel", "lib", "mun", "nop", "qui", "ret", "sil"]
KINDS = [  # (fact templates, question templates) over (X, Y)
    (["The capital of {X} is {Y}.", "{Y} is the capital of {X}.", "{X} has its capital at {Y}."],
     ["What is the capital of {X}?", "Which city is the capital of {X}?", "Name the capital of {X}."]),
    (["The {X} engine was invented by {Y}.", "{Y} invented the {X} engine.", "The inventor of the {X} engine was {Y}."],
     ["Who invented the {X} engine?", "Which person invented the {X} engine?", "Name the inventor of the {X} engine."]),
    (["{X} was born in {Y}.", "The birthplace of {X} is {Y}.", "{X} comes from {Y}."],
     ["Where was {X} born?", "What is the birthplace of {X}?", "Which town is {X} from?"]),
    (["The dog called {X} belongs to {Y}.", "{Y} owns a dog named {X}.", "{X} is the dog of {Y}."],
     ["Who owns the dog called {X}?", "Whose dog is {X}?", "Who is the owner of {X}?"]),
    (["The code for {X} is {Y}.", "{X} has the code {Y}.", "Use code {Y} for {X}."],
     ["What is the code for {X}?", "Which code belongs to {X}?", "Give the code of {X}."]),
    (["{X} is coloured {Y}.", "The colour of {X} is {Y}.", "{X} is painted {Y}."],
     ["What colour is {X}?", "Which colour is {X} painted?", "Name the colour of {X}."]),
]
COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "grey", "brown", "pink", "silver"]
QA = [("Question: {q} Answer: {a}"), ("Q: {q}\nA: {a}"), ("{q} {a}.")]
r = random.Random(20260924)


def name(n): return "".join(r.choice(SYL) for _ in range(n)).capitalize()


def value(k):
    if k == 1: return f"{name(2)} {name(2)}"
    if k == 4: return str(r.randint(100, 9999))
    if k == 5: return r.choice(COLOURS)
    return name(2)


def document():
    n = r.choice([10, 20, 40, 80, 150])
    facts, used = [], set()
    while len(facts) < n:
        k = r.randrange(len(KINDS)); x = name(r.choice([2, 3]))
        if x in used: continue
        used.add(x); facts.append((k, x, value(k)))
    lines = [r.choice(KINDS[k][0]).format(X=x, Y=y) for k, x, y in facts]
    body = " ".join(lines)
    asked = r.sample(facts, min(len(facts), r.choice([5, 10, 20])))
    qa = "\n".join(r.choice(QA).format(q=r.choice(KINDS[k][1]).format(X=x), a=y) for k, x, y in asked)
    return body + "\n\n" + qa


os.makedirs(OUT, exist_ok=True)
target = int(MTOK * 1e6); out = []; total = 0
if len(sys.argv) > 3 and sys.argv[3] == "needle":
    # NEEDLES IN REAL TEXT (2026-10-01). Passkey retrieval measured 55% at 512 tokens and 0% from 2,048 on: the
    # carrier keeps nothing exact past the attention window, and nothing in training asks it to. Each document is REAL
    # prose (random spans of Wikipedia / FineWeb-Edu, end-of-document tokens removed) of 256-16,384 tokens with 1-4
    # needles (a number or a word bound to a name) at random depths, then questions about some of them, answers given.
    # Kept OUT of the passkey test's distribution on purpose: the test uses repeated filler ("The grass is green...")
    # and the words "pass key"; neither occurs here. Writes needle.bin.
    import numpy as np
    src = [np.memmap(f"/mnt/kv_cache/tokfull_v8192/{k}.bin", dtype=np.uint16, mode="r") for k in ("wikipedia", "fineweb-edu")]
    NAMES = ["Alice", "Omar", "Priya", "Lukas", "Mei", "Tomas", "Aisha", "Diego", "Hana", "Ivan", "Zara", "Kofi",
             "Nadia", "Pavel", "Sofia", "Ravi", "Elena", "Jonas", "Leila", "Marco", "orange", "harbour", "north",
             "copper", "willow", "falcon", "maple", "granite", "river", "summit"]
    WORDS = ["lantern", "pepper", "violet", "anchor", "thunder", "meadow", "cobalt", "saffron", "glacier", "ember",
             "quartz", "juniper", "velvet", "comet", "tundra", "marble", "nectar", "orchid", "basalt", "cinder"]
    NEED = [  # (statement, question) over N (a name) and V (the value)
        ("The secret code is {V}.", "What is the secret code?"),
        ("Remember this number: {V}.", "What number were you asked to remember?"),
        ("The combination for the {N} lock is {V}.", "What is the combination for the {N} lock?"),
        ("{N}'s favourite number is {V}.", "What is {N}'s favourite number?"),
        ("The {N} team meets in room {V}.", "Which room does the {N} team meet in?"),
        ("The password for the {N} account is {V}.", "What is the password for the {N} account?"),
        ("The {N} shipment carries tracking number {V}.", "What is the tracking number of the {N} shipment?"),
        ("Write down the word {V}; you will need it later.", "Which word were you told to write down?")]
    rn = np.random.default_rng(20261001)
    def value(j):
        if j == 5 or j == 7: return r.choice(WORDS) + (str(r.randint(10, 99)) if j == 5 else "")
        return str(r.randint(100, 999999))
    while total < target:
        # NEEDLE_MIN_DIST (2026-10-03): every needle at least this many tokens before the end, documents 2k-8k tokens
        # (inside one training window): past a 1,024-token attention window, only the carrier can answer
        MD = int(os.environ.get("NEEDLE_MIN_DIST", "0"))
        L = int(np.exp(rn.uniform(np.log(2048 if MD else 256), np.log(8192 if MD else 16384))))
        k = r.choice([1, 1, 2, 3, 4]); used_t = set(); facts = []
        while len(facts) < k:
            j = r.randrange(len(NEED))
            if j in (0, 1, 7) and j in used_t: continue          # nameless templates at most once per document
            used_t.add(j); facts.append((j, r.choice(NAMES), value(j)))
        stmts = [tok.encode(" " + NEED[j][0].format(N=n, V=v) + " ").ids for j, n, v in facts]
        asked = r.sample(facts, r.randint(1, k))
        qa = tok.encode("\n\n" + "\n".join(r.choice(QA).format(q=NEED[j][1].format(N=n), a=v) for j, n, v in asked)).ids
        n_fill = max(64, L - sum(map(len, stmts)) - len(qa))
        sb = src[r.randrange(2)]; o = int(rn.integers(0, len(sb) - 2 * n_fill))
        fill = np.asarray(sb[o:o + 2 * n_fill]); fill = fill[fill != EOD][:n_fill].tolist()
        cuts = sorted(r.randrange(max(1, len(fill) + 1 - MD)) for _ in stmts)
        doc, prev = [], 0
        for c, st in zip(cuts, stmts): doc += fill[prev:c] + st; prev = c
        doc += fill[prev:] + qa + [EOD]
        out.append(np.array(doc, dtype=np.uint16)); total += len(doc)
        if len(out) % 2000 == 0: print(f"  {total/1e6:.1f}M tokens", flush=True)
    arr = np.concatenate(out)[:target]; arr.tofile(f"{OUT}/needle.bin")
    print(f"done: {len(arr):,} tokens -> {OUT}/needle.bin; {len(out):,} documents, mean {len(arr)/len(out):.0f} tokens")
    sys.exit(0)
while total < target:
    docs = [document() for _ in range(2000)]
    for e in tok.encode_batch(docs):
        out.append(np.array(e.ids + [EOD], dtype=np.uint16)); total += len(e.ids) + 1
    print(f"  {total/1e6:.1f}M tokens", flush=True)
arr = np.concatenate(out)[:target]
arr.tofile(f"{OUT}/recall.bin")
print(f"done: {len(arr):,} tokens -> {OUT}/recall.bin; mean document {len(arr)/sum(1 for _ in out):.0f} tokens")
