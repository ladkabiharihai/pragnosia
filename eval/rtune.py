"""ABSTENTION FROM THE MODEL'S OWN MISTAKES (vision goals 4 and 5), R-Tuning (Zhang et al. 2023).

The model answers training questions it can be checked on; where it was right the target is the right answer,
where it was wrong the target becomes "I don't know." Fine-tuned on that, it learns to decline exactly the kind
of question it gets wrong. Nothing outside the weights decides: no threshold, no second model, no retrieval.
Only TRAINING splits are asked here (SciQ train, ARC train); eval/qa.py asks the held-out test splits.

Writes a JSON-lines file of (question, target) pairs that eval/build_sft.py adds through SFT_EXTRA.
usage: eval/rtune.py <ckpt> <out.jsonl>"""
import sys, os, glob, json, re, time, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import numpy as np, torch, datasets
from src.bench import Ours

IDK = "I don't know."
BASE = "/mnt/kv_cache/hf_home/datasets"


def shards(name, hint):
    fs = sorted(f for f in glob.glob(f"{BASE}/{name}/**/*.arrow", recursive=True) if hint in os.path.basename(f))
    return [r for f in fs for r in datasets.Dataset.from_file(f)]


def first_sentence(t):                      # the same rule eval/qa.py scores by
    t = t.split("\nUser:")[0].strip()
    m = re.match(r"(.+?[.!?])(\s|$)", t, re.S)
    return (m.group(1) if m else t)[:300]


items = []                                  # (question, gold short answer)
for r in shards("sciq", "train"):
    a = r["correct_answer"].strip()
    items.append((r["question"], a))
for r in shards("allenai___ai2_arc", "train"):
    lab, txt = r["choices"]["label"], r["choices"]["text"]
    if r["answerKey"] in lab:
        items.append((r["question"], txt[lab.index(r["answerKey"])].strip().rstrip(".")))

o = Ours(sys.argv[1], "/mnt/kv_cache/tok_v8192.json")
out = open(sys.argv[2], "w")
right = 0; t0 = time.time()
for i, (q, a) in enumerate(items):
    ids = o.generate(o.encode(f"User: {q}\nAssistant:"), 32, temp=0.0, seed=0, stop=8191)
    if 8191 in ids: ids = ids[:ids.index(8191)]
    ok = a.lower() in first_sentence(o.decode(ids)).lower()
    right += ok
    target = (a[0].upper() + a[1:] + ".") if ok else IDK
    out.write(json.dumps({"q": q, "a": target, "model_right": bool(ok)}) + "\n")
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(items)}  model right on {100*right/(i+1):.1f}%  {(i+1)/(time.time()-t0):.1f} q/s", flush=True)
out.close()
print(f"done: {len(items):,} training questions, model right on {right:,} ({100*right/len(items):.1f}%), "
      f"{len(items)-right:,} relabelled \"{IDK}\"", flush=True)
