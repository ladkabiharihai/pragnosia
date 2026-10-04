"""SELF-IMPROVEMENT FROM THE MODEL'S OWN AGREEMENT (vision goal 3). Huang et al. 2022, "Large Language Models Can
Self-Improve": no labels are used to train. The model answers each unlabeled question K times with sampling; where
a majority of its own answers agree, that majority answer becomes a training target, its self-consistency being the
confidence. Answers of "I don't know" never count as a label. The gold answer is read ONLY afterwards, to report how
often the self-labels were right; it never reaches the training file.

Writes a JSON-lines file of (question, target) pairs that eval/build_sft.py adds through SFT_EXTRA.
usage: eval/selfimprove.py <ckpt> <out.jsonl> [n_questions]   |   eval/selfimprove.py lookup <ckpt> <out_dir> [n_docs]"""
import sys, os, glob, json, re, time, random, warnings
from collections import Counter
warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import datasets
from src.bench import Ours

K, TEMP, MAJ = 5, 0.7, 3                    # 5 samples; a label needs 3 of 5 to agree

# LOOKUP MODE (2026-10-03): eval/selfimprove.py lookup <ckpt> <out_dir> [n_docs]
# Factual Q&A failed the gate (labels 19-38% right). In-context lookup is a skill the model has: 56-59% at 10-120 facts
# with every distractor from the same context, 100% for a single needle. So: invented documents of facts (syllables used
# by NO training builder and NO test), questions about them, the model answers each question under 3 orderings of
# the facts, and the answer at least 2 orderings agree on becomes the label. Gold is read only afterwards, to report how
# often the labels are right (the gate); it never reaches the training file (<out_dir>/selflabel.bin, documents of
# facts + the self-labelled questions, in the lookup training format).
if len(sys.argv) > 1 and sys.argv[1] == "lookup":
    import numpy as np
    from tokenizers import Tokenizer
    CK, OUT = sys.argv[2], sys.argv[3]; ND = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    os.makedirs(OUT, exist_ok=True)
    tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
    SY = ["zu", "qel", "vori", "pax", "dyn", "ilo", "brun", "sef", "omb", "tir", "ganu", "hesk", "wol", "yvi", "crad",
          "ulm", "fesh", "nok", "ari", "plen", "duv", "skir", "emo", "rath", "jin", "obel", "trem", "kasu", "vin", "lod"]
    # a fresh set of documents per round (round 1 wrote /mnt/kv_cache/tokself with seed 20261003; a later round must not
    # self-label the documents the model has already trained on)
    import zlib
    rr = random.Random(20261003 if OUT.rstrip("/").endswith("tokself") else 20261003 + zlib.crc32(OUT.encode()))
    def nm(n): return "".join(rr.choice(SY) for _ in range(n)).capitalize()
    KINDS_ = [("The capital of {X} is {Y}.", "What is the capital of {X}?"),
              ("The {X} engine was invented by {Y}.", "Who invented the {X} engine?"),
              ("{X} was born in {Y}.", "Where was {X} born?"),
              ("The code for {X} is {Y}.", "What is the code for {X}?")]
    def val(k): return f"{nm(2)} {nm(2)}" if k == 1 else str(rr.randint(100, 9999)) if k == 3 else nm(2)
    o = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
    def answer(facts, q):
        ids = o.encode(" ".join(facts) + "\n\nQuestion: " + q + " Answer:")
        t = o.decode(o.generate(ids, 10, temp=0.0, stop=8191))
        return t.strip().split("\n")[0].strip().rstrip(".").strip()
    docs, n_lab = [], 0; right = asked = 0; t0 = time.time()
    for d in range(ND):
        n = rr.choice([20, 40, 80]); facts, used = [], set()
        while len(facts) < n:
            k = rr.randrange(len(KINDS_)); x = nm(rr.choice([2, 3]))
            if x in used: continue
            used.add(x); facts.append((k, x, val(k)))
        sent = [KINDS_[k][0].format(X=x, Y=y) for k, x, y in facts]
        qa = []
        for k, x, y in rr.sample(facts, 8):
            q = KINDS_[k][1].format(X=x); ans = []
            for _ in range(3):
                order = sent[:]; rr.shuffle(order); ans.append(answer(order, q))
            top, c = Counter(ans).most_common(1)[0]
            asked += 1
            if c >= 2 and top:
                qa.append(f"Question: {q} Answer: {top}"); n_lab += 1; right += (top == y)   # gold: diagnostic only
        if qa:
            order = sent[:]; rr.shuffle(order); docs.append(" ".join(order) + "\n\n" + "\n".join(qa))
        if (d + 1) % 100 == 0:
            print(f"  {d+1}/{ND} documents: {n_lab}/{asked} questions self-labelled ({100*n_lab/asked:.0f}%), "
                  f"labels right {100*right/max(1,n_lab):.1f}%  ({(d+1)/(time.time()-t0):.2f} doc/s)", flush=True)
    arr = np.concatenate([np.array(e.ids + [8191], dtype=np.uint16) for e in tk.encode_batch(docs)])
    arr.tofile(f"{OUT}/selflabel.bin")
    print(f"done: {len(docs):,} documents, {n_lab:,} of {asked:,} questions self-labelled by >= 2 of 3 orderings; "
          f"checked afterwards against gold, {100*right/max(1,n_lab):.1f}% right; {len(arr):,} tokens -> {OUT}/selflabel.bin")
    sys.exit(0)
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
