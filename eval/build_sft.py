"""Question-answer training data, built from datasets already on disk, in our tokenizer.

Every example becomes   "User: <question>\\nAssistant: <answer>" + end-of-document
with the loss applied to the ANSWER tokens only, so the model learns to respond rather than to imitate
questions. Examples are packed end to end into 1,024-token rows (the end-of-document token separates them,
exactly as documents were separated in pretraining) with a parallel mask row.

Sources, all local (no download): Alpaca, Dolly-15k, OpenAssistant (English, top-ranked replies only),
CAMEL biology/chemistry/physics, SciQ train (answer + its supporting passage), ARC train, CommonsenseQA train.
Held out and never written here: SciQ test, ARC-Easy test, ARC-Challenge test -- eval/qa.py asks those.
usage: eval/build_sft.py [out_dir]"""
import os, sys, random, glob
import numpy as np
import datasets
from tokenizers import Tokenizer

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/sft_v8192"
BASE = "/mnt/kv_cache/hf_home/datasets"
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
EOD, MAXEX = 8191, 900
ROW = int(os.environ.get("SFT_ROW", "1025"))   # 1,024 inputs + 1 target: attention needs a multiple of its 512 window
os.makedirs(OUT, exist_ok=True)


def shards(name, split_hint=None):
    fs = sorted(f for f in glob.glob(f"{BASE}/{name}/**/*.arrow", recursive=True)
                if split_hint is None or split_hint in os.path.basename(f))
    return [datasets.Dataset.from_file(f) for f in fs]


pairs = []                                                   # (source, prompt, answer)
for ds in shards("tatsu-lab___alpaca"):
    for r in ds:
        q = r["instruction"] + (("\n" + r["input"]) if r["input"] else "")
        pairs.append(("alpaca", q, r["output"]))
for ds in shards("databricks___databricks-dolly-15k"):
    for r in ds:
        q = r["instruction"] + (("\n\nContext: " + r["context"]) if r["context"] else "")
        pairs.append(("dolly", q, r["response"]))
msgs = {}
for ds in shards("OpenAssistant___oasst1"):
    for r in ds:
        msgs[r["message_id"]] = r
for m in msgs.values():
    if m["role"] == "assistant" and m["lang"] == "en" and m["rank"] == 0 and m["parent_id"] in msgs:
        p = msgs[m["parent_id"]]
        if p["role"] == "prompter" and p["lang"] == "en":
            pairs.append(("oasst", p["text"], m["text"]))
for sub in ("biology", "chemistry", "physics"):
    for ds in shards(f"camel-ai___{sub}"):
        for r in ds:
            pairs.append((f"camel-{sub}", r["message_1"], r["message_2"]))
for ds in shards("sciq", "train"):
    for r in ds:
        a = r["correct_answer"].strip()
        a = a[0].upper() + a[1:] + "."
        if r["support"].strip(): a += " " + r["support"].strip()
        pairs.append(("sciq", r["question"], a))
for ds in shards("allenai___ai2_arc", "train"):
    for r in ds:
        lab = r["choices"]["label"]; txt = r["choices"]["text"]
        if r["answerKey"] in lab:
            pairs.append(("arc", r["question"], txt[lab.index(r["answerKey"])].strip().rstrip(".") + "."))
for ds in shards("tau___commonsense_qa", "train"):
    for r in ds:
        lab = r["choices"]["label"]; txt = r["choices"]["text"]
        if r["answerKey"] in lab:
            pairs.append(("csqa", r["question"], txt[lab.index(r["answerKey"])].strip() + "."))

# SFT_MIX=factual: the same sources tilted toward SHORT, FACTUAL answers. The first fine-tune taught the form
# of an answer but got half of eight plain facts wrong, and 60k of its 160k examples were long CAMEL
# explanations. This keeps Dolly's question-answering categories only, Alpaca and OpenAssistant answers that
# are short, SciQ's answer with one sentence of support, and a sample of CAMEL.
if os.environ.get("SFT_MIX") == "factual":
    dolly_cat = {}
    for ds in shards("databricks___databricks-dolly-15k"):
        for r in ds: dolly_cat[r["instruction"]] = r["category"]
    rng_f = random.Random(1)
    out = []
    for src, q, a in pairs:
        nw = len(a.split())
        if src == "alpaca" and nw > 60: continue
        if src == "oasst" and nw > 120: continue
        if src == "dolly" and dolly_cat.get(q.split("\n\nContext:")[0]) not in ("open_qa", "closed_qa", "general_qa"): continue
        if src == "sciq":
            first = a.split(". ")
            a = ". ".join(first[:2]).rstrip(".") + "."
        out.append((src, q, a))
    camel_idx = [i for i, x in enumerate(out) if x[0].startswith("camel")]
    drop = set(rng_f.sample(camel_idx, max(0, len(camel_idx) - 15000)))
    pairs = [x for i, x in enumerate(out) if i not in drop]

counts = {}
for s, _, _ in pairs: counts[s] = counts.get(s, 0) + 1
print("examples by source:", ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())), f"-- {len(pairs):,} total")

random.Random(0).shuffle(pairs)
nval = 2000
for split, part in (("val", pairs[:nval]), ("train", pairs[nval:])):
    toks, mask = [], []
    kept = dropped = 0
    for _, q, a in part:
        pi = tok.encode(f"User: {q.strip()}\nAssistant:").ids
        ai = tok.encode(" " + a.strip()).ids + [EOD]
        if len(pi) + len(ai) > MAXEX or not a.strip():
            dropped += 1; continue
        toks += pi + ai; mask += [0] * len(pi) + [1] * len(ai); kept += 1
    n = len(toks) // ROW * ROW
    np.array(toks[:n], dtype=np.uint16).tofile(f"{OUT}/{split}.bin")
    np.array(mask[:n], dtype=np.uint8).tofile(f"{OUT}/{split}.mask")
    print(f"{split}: {kept:,} examples kept, {dropped:,} over {MAXEX} tokens dropped, {n:,} tokens "
          f"({n // ROW:,} rows of {ROW}), {100 * np.mean(mask[:n]):.1f}% of tokens carry loss")
