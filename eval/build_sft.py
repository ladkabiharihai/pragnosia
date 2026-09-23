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
