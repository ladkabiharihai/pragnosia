"""TRAINING DATA FOR IN-CONTEXT LOOKUP (vision goal 9). Measured 2026-09-24: with 120 invented facts in context and
every wrong option also in the context, the model is at chance even for the facts inside its attention window, so
it cannot yet look a fact up among many; it only notices which names appeared. Gated delta-rule models learn this
lookup well once they see it (Arora et al. 2023, "Zoology"); ordinary text almost never asks for it.

Each document lists 10-150 invented facts of six kinds in varied wording, then asks about a random subset in a
DIFFERENT wording, answers given. Documents run from a few hundred to several thousand tokens, so most answers sit
beyond the 512-token attention window and the carrier's state has to carry them. The syllables differ from the
ones eval/incontext.py builds its names from, so no test fact can occur here.
usage: eval/build_recall.py <out_dir> <million_tokens>"""
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
while total < target:
    docs = [document() for _ in range(2000)]
    for e in tok.encode_batch(docs):
        out.append(np.array(e.ids + [EOD], dtype=np.uint16)); total += len(e.ids) + 1
    print(f"  {total/1e6:.1f}M tokens", flush=True)
arr = np.concatenate(out)[:target]
arr.tofile(f"{OUT}/recall.bin")
print(f"done: {len(arr):,} tokens -> {OUT}/recall.bin; mean document {len(arr)/sum(1 for _ in out):.0f} tokens")
