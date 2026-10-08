"""THE FINAL PHASE'S EXTRA SOURCES (2026-10-06; plan approved by the user: the flagship's last days on high-quality and
instruction data, TRAIN SPLITS ONLY, the mix chosen by the model). Written as separate corpora so the model can weight them
separately (train.py --mix-auto): every example is plain text in the template the Q&A evaluation asks with,
"User: <question>\\nAssistant: <answer>" + end-of-document, the loss on every token (pretraining, not masked fine-tuning).
  instr_chat    Alpaca, Dolly-15k, OpenAssistant, CAMEL biology/chemistry/physics (as eval/build_sft.py builds them)
  instr_qa      SciQ train (answer + support), ARC train, CommonsenseQA train, OpenBookQA train
  instr_math    Orca-Math 200k, MetaMathQA, GSM8K train
  instr_sense   HellaSwag train and PIQA train as plain text (context + the right ending / goal + the right solution)
Held out and never written: every test and validation split eval/ asks (SciQ, ARC, OpenBookQA, HellaSwag, PIQA, GSM8K).
usage: eval/build_instr.py <out_dir>"""
import os, sys, random, glob
import numpy as np, datasets
from tokenizers import Tokenizer
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
src = open("/opt/code/membrae/eval/build_sft.py").read()
g = {"__name__": "build_sft_pairs"}; sys_argv = sys.argv; sys.argv = ["build_sft.py", "/tmp/claude-1000/sft_scratch_unused"]
exec(src[:src.index("# SFT_MIX=factual")], g)                     # the same pairs eval/build_sft.py makes
sys.argv = sys_argv
pairs, shards, tok, EOD = g["pairs"], g["shards"], g["tok"], 8191
fam = {"alpaca": "chat", "dolly": "chat", "oasst": "chat", "sciq": "qa", "arc": "qa", "csqa": "qa"}
out = {"chat": [], "qa": [], "math": [], "sense": []}
for s, q, a in pairs:
    out[fam.get(s, "chat" if s.startswith("camel") else "qa")].append(f"User: {q.strip()}\nAssistant: {a.strip()}")
for ds in shards("openbookqa", "train"):
    for r in ds:
        lab, txt = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] in lab: out["qa"].append(f"User: {r['question_stem'].strip()}\nAssistant: {txt[lab.index(r['answerKey'])].strip()}.")
for ds in shards("microsoft___orca-math-word-problems-200k"):
    for r in ds: out["math"].append(f"User: {r['question'].strip()}\nAssistant: {r['answer'].strip()}")
for ds in shards("meta-math___meta_math_qa"):
    for r in ds: out["math"].append(f"User: {r['query'].strip()}\nAssistant: {r['response'].strip()}")
for ds in shards("openai___gsm8k", "train"):
    for r in ds:
        sol, ans = r["answer"].split("####"); out["math"].append(f"User: {r['question'].strip()}\nAssistant: {sol.strip()} The answer is {ans.strip()}.")
for ds in shards("Rowan___hellaswag", "train"):
    for r in ds: out["sense"].append(f"{r['activity_label']}: {r['ctx'].strip()} {r['endings'][int(r['label'])].strip()}")
for ds in shards("piqa", "train"):
    for r in ds: out["sense"].append(f"{r['goal'].strip()} {(r['sol1'], r['sol2'])[int(r['label'])].strip()}")
for k, docs in out.items():
    random.Random(0).shuffle(docs)
    ids = []
    for i in range(0, len(docs), 2000):
        for e in tok.encode_batch(docs[i:i + 2000]): ids += e.ids + [EOD]
    np.array(ids, dtype=np.uint16).tofile(f"{OUT}/instr_{k}.bin")
    print(f"instr_{k}: {len(docs):,} examples, {len(ids):,} tokens", flush=True)
