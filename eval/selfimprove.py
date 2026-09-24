"""SELF-IMPROVEMENT FROM THE MODEL'S OWN AGREEMENT (vision goal 3). Huang et al. 2022, "Large Language Models Can
Self-Improve": no labels are used to train. The model answers each unlabeled question K times with sampling; where
a majority of its own answers agree, that majority answer becomes a training target, its self-consistency being the
confidence. Answers of "I don't know" never count as a label. The gold answer is read ONLY afterwards, to report how
often the self-labels were right; it never reaches the training file.

Writes a JSON-lines file of (question, target) pairs that eval/build_sft.py adds through SFT_EXTRA.
usage: eval/selfimprove.py <ckpt> <out.jsonl> [n_questions]"""
import sys, os, glob, json, re, time, random, warnings
from collections import Counter
warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import datasets
from src.bench import Ours

K, TEMP, MAJ = 5, 0.7, 3                    # 5 samples; a label needs 3 of 5 to agree
BASE = "/mnt/kv_cache/hf_home/datasets"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000


def shards(name, hint):
    fs = sorted(f for f in glob.glob(f"{BASE}/{name}/**/*.arrow", recursive=True) if hint in os.path.basename(f))
    return [r for f in fs for r in datasets.Dataset.from_file(f)]


def first_sentence(t):
    t = t.split("\nUser:")[0].strip()
    m = re.match(r"(.+?[.!?])(\s|$)", t, re.S)
    return (m.group(1) if m else t)[:300].strip()


def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


items = [(r["question"], r["correct_answer"]) for r in shards("sciq", "train")]
for r in shards("allenai___ai2_arc", "train"):
    lab, txt = r["choices"]["label"], r["choices"]["text"]
    if r["answerKey"] in lab: items.append((r["question"], txt[lab.index(r["answerKey"])]))
random.Random(0).shuffle(items); items = items[:N]

o = Ours(sys.argv[1], "/mnt/kv_cache/tok_v8192.json")
out = open(sys.argv[2], "w")
kept = right_kept = 0; t0 = time.time()
for i, (q, gold) in enumerate(items):
    p = o.encode(f"User: {q}\nAssistant:")
    ans = []
    for s in range(K):
        ids = o.generate(p, 32, temp=TEMP, seed=1000 * i + s, stop=8191)
        if 8191 in ids: ids = ids[:ids.index(8191)]
        ans.append(first_sentence(o.decode(ids)))
    votes = Counter(norm(a) for a in ans if a and "don't know" not in a.lower())
    if votes:
        top, n = votes.most_common(1)[0]
        if n >= MAJ and top:
            target = next(a for a in ans if norm(a) == top)
            ok = norm(gold) in top                        # diagnostic only: never written to the training file
            kept += 1; right_kept += ok
            out.write(json.dumps({"q": q, "a": target, "model_right": True, "agree": n}) + "\n")
    if (i + 1) % 250 == 0:
        print(f"  {i+1}/{len(items)}  self-labelled {kept} ({100*kept/(i+1):.0f}%), of which right {100*right_kept/max(1,kept):.0f}%"
              f"  {(i+1)/(time.time()-t0):.1f} q/s", flush=True)
out.close()
print(f"done: {len(items):,} unlabeled questions, {kept:,} self-labelled by >= {MAJ}/{K} agreement; "
      f"checked afterwards against the gold answers, {100*right_kept/max(1,kept):.1f}% of those labels are right", flush=True)
