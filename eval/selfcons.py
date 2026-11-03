"""KNOWS WHAT IT KNOWS, WITHOUT LABELS: abstain when the model disagrees with itself (2026-10-06, goals 4-5).
Training the model to say "I don't know" on its own mistakes lowered wrong answers but also right ones (SciQ right 8.4 ->
3.5%). Here nothing is trained and no threshold is set by hand: for every question the fine-tuned model answers greedily
and also samples K answers at temperature 1 (its own distribution); it ANSWERS only if at least half of the samples
share an answer word with the greedy answer (a word of the answer's first sentence that is not in the question and is not
a stop word), and otherwise DECLINES. Scored exactly as eval/qa.py scores: right / wrong / declined, and accuracy when it
answers. usage: eval/selfcons.py <fine-tuned ckpt> [K]"""
import os, sys, re, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
from src.common import cap_gpu
cap_gpu(6.0)
import numpy as np, torch, datasets
from src.bench import Ours
src = open("/opt/code/membrae/eval/qa.py").read()
exec(src[src.index("EVERYDAY = ["):src.index("def arrow(")])          # the same 48 everyday questions as eval/qa.py
SCIQ = datasets.Dataset.from_file(glob.glob("/mnt/kv_cache/hf_home/datasets/sciq/**/sciq-test.arrow", recursive=True)[0])
CK = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
o = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
STOP = set("the a an of in on at to is are was were be been it its this that these those and or for with by as from what which "
           "who whom how why when where does do did has have had can could would should will i you he she they we not no".split())
def first_sentence(t):
    t = t.split("\nUser:")[0].strip(); m = re.match(r"(.+?[.!?])(\s|$)", t, re.S)
    return (m.group(1) if m else t)[:300]
def gen(p, temp, seed, n):
    out = o.generate(o.encode(f"User: {p}\nAssistant:"), n, temp=temp, seed=seed, stop=8191)
    if 8191 in out: out = out[:out.index(8191)]
    return first_sentence(o.decode(out)).lower()
def words(t): return set(re.findall(r"[a-z0-9]+", t))
def key(ans, q): return words(ans) - words(q.lower()) - STOP
def declined(a): return any(k in a for k in ("i don't know", "i do not know", "i'm not sure", "i am not sure"))
def run(name, items, n):
    rows = []
    for q, acc in items:
        g = gen(q, 0.0, 0, n); kg = key(g, q)
        agree = [bool(kg & key(gen(q, 1.0, s, n), q)) for s in range(1, K + 1)]
        cons = float(np.mean(agree)) if kg else 0.0
        rows.append((any(v in g for v in acc), declined(g), cons))
    r = np.array([x[0] for x in rows]); d = np.array([x[1] for x in rows]); c = np.array([x[2] for x in rows])
    for lab, ans_mask in (("greedy, always answers", ~d), (f"self-consistent (>= half of {K} samples agree)", ~d & (c >= 0.5))):
        right = (r & ans_mask).mean(); wrong = (~r & ans_mask).mean(); dec = 1 - ans_mask.mean()
        print(f"  {name:<9s} {lab:<44s} right {100 * right:5.1f}%  WRONG {100 * wrong:5.1f}%  declined {100 * dec:5.1f}%  "
              f"-> accuracy when it answers {100 * right / max(1e-9, right + wrong):5.1f}%", flush=True)
    print(f"  {name:<9s} consistency when right {c[r].mean():.2f}, when wrong {c[~r].mean():.2f}", flush=True)
    # AUROC (2026-10-15, goal 4 at scale): does self-agreement separate right answers from wrong ones (0.5 = no; the option-softmax
    # confidence of eval/calibration.py reached only ~0.6)
    if r.any() and (~r).any():
        pos, neg = c[r], c[~r]
        au = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
        print(f"  {name:<9s} self-agreement AUROC (right vs wrong) {au:.3f}  (n right {int(r.sum())}, wrong {int((~r).sum())})", flush=True)
print(f"SELF-CONSISTENCY ABSTENTION, no labels, no threshold set by hand: {CK}, K={K} samples at temperature 1")
run("EVERYDAY", EVERYDAY, 48)
run("SCIQ", [(x["question"], [x["correct_answer"].lower().strip()]) for x in list(SCIQ)[:1000]], 32)
