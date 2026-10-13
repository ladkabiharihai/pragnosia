"""OPEN-BOOK SCIENCE QA AT A DISTANCE (2026-10-06): reading, not remembering.
SciQ test questions come with the paragraph that answers them. Every model gets the SAME long context: real science
paragraphs (SciQ train supports) with the question's own paragraph placed D tokens before the question, then
"Question: ... Answer:" and SciQ's four options, scored by summed log-probability (chance 25%). Closed book (no context)
is the baseline: what each model already knows. A model sees as much of the context as it can: GPT-2 its last 1,024
tokens, SmolLM2 its last 8,192, ours all of it (window by window, the state carried; with recall, every key kept).
usage: eval/openbook.py <ckpt>[=label][+recall] ...      N=<questions per cell, default 150>"""
import os, sys, glob, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(float(os.environ.get("OPENBOOK_CAP", "6.0")))  # OPENBOOK_CAP (2026-10-12): the self-improved row hit 6 GiB at 32k
import numpy as np, torch, datasets
from src.bench import Ours
N = int(os.environ.get("N", "150")); DIST = [int(d) for d in os.environ.get("DIST", "0,2048,8192,32768").split(",")]
BASE = "/mnt/kv_cache/hf_home/datasets/sciq"
test = datasets.Dataset.from_file(glob.glob(f"{BASE}/**/sciq-test.arrow", recursive=True)[0])
train = datasets.Dataset.from_file(glob.glob(f"{BASE}/**/sciq-train.arrow", recursive=True)[0])
Q = [r for r in test if len(r["support"]) > 50]
random.Random(0).shuffle(Q); Q = Q[:N]
FILL = [r["support"].strip() for r in train if len(r["support"]) > 50]

class HF:
    def __init__(self, name, lab):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache")
        # memory-efficient attention: SmolLM2's default at 8,192 tokens held a full score matrix and passed a 6 GiB cap
        self.m = AutoModelForCausalLM.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache", torch_dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").cuda().eval()
        self.lab = lab; self.max_ctx = getattr(self.m.config, "max_position_embeddings", 1024)
    def opt_lp(self, ctx, opt):
        a = self.tok.encode(ctx); b = self.tok.encode(" " + opt); ids = (a + b)[-self.max_ctx:]
        with torch.no_grad():
            lp = self.m(torch.tensor([ids], device="cuda")).logits[0].float().log_softmax(-1)
        y = torch.tensor(ids[-len(b):], device="cuda")
        return float(lp[-len(b) - 1:-1].gather(-1, y.unsqueeze(-1)).sum())

class OursM:
    def __init__(self, spec):
        rec = spec.endswith("+recall"); spec = spec.replace("+recall", "")
        ck, lab = (spec.split("=") + [None])[:2]
        self.o = Ours(ck, "/mnt/kv_cache/tok_v8192.json")
        # THE FUSED SCAN for reading (the python chunk scan is ~10-30x slower; same picks, eval/qa.py)
        import src.assoc as A_; A_.FUSED_SCAN = True
        for mx_ in self.o.m.mix:
            if hasattr(mx_, "need_sent"): mx_.need_sent = False
        if rec:
            from src.recall import enable; enable(self.o.m, 16)
        if os.environ.get("RECALL_K") and getattr(self.o.m, "recall_k", 0):    # a different k at inference (trained at 16)
            self.o.m.set_recall(int(os.environ["RECALL_K"]), span=0); lab = (lab or ck) + f" k={os.environ['RECALL_K']}"
        self.lab = (lab or os.path.basename(os.path.dirname(ck)) or ck) + (" +recall" if rec else "") + \
                   (" (recall-trained)" if getattr(self.o.m, "recall_k", 0) and not rec else "")
    def opt_lp(self, ctx, opt):
        # the context is read ONCE per question (its state cached) and each option scored as a continuation of it
        if getattr(self, "_ck", None) != ctx:
            self._ck = ctx; self._pre = self.o.prefix(self.o.encode(ctx))
        return self.o.continuation_lp(self._pre, self.o.encode(" " + opt))

from tokenizers import Tokenizer
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
def ntok(t): return len(tk.encode(t).ids)
rng = random.Random(1)
def filler(n_tokens):
    out, n = [], 0
    while n < n_tokens:
        p = rng.choice(FILL); out.append(p); n += ntok(p) + 2
    return "\n\n".join(out)
# the contexts are built once and shared by every model (same filler, same placement)
CTX = {}
for D in DIST:
    for i, r in enumerate(Q):
        pre = filler(256)
        CTX[(D, i)] = pre + "\n\n" + r["support"].strip() + ("\n\n" + filler(D) if D else "") + \
                      f"\n\nQuestion: {r['question']}\nAnswer:"
def options(r): return [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]

HFS = [("gpt2", "GPT-2 124M"), ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M")]
if os.environ.get("ONLY_HF"): HFS = [h for h in HFS if os.environ["ONLY_HF"] in h[0]]
if os.environ.get("OPENBOOK_HF") == "0": HFS = []        # (2026-10-12) the baselines' rows are measured already; ours only
models = [OursM(s) for s in sys.argv[1:]] + [HF(n, l) for n, l in HFS]
print(f"OPEN-BOOK SciQ: {N} test questions per cell, the answering paragraph D tokens (ours) before the question among "
      f"real science paragraphs; 4-way, chance 25%")
print(f"{'model':<34s}{'closed book':>12s}" + "".join(f"{'D=' + format(d, ','):>12s}" for d in DIST))
for M in models:
    row = []
    ok = sum(int(np.argmax([M.opt_lp(f"Question: {r['question']}\nAnswer:", o) for o in options(r)])) == 0 for r in Q)
    row.append(100 * ok / len(Q))
    for D in DIST:
        ok = sum(int(np.argmax([M.opt_lp(CTX[(D, i)], o) for o in options(r)])) == 0 for i, r in enumerate(Q))
        row.append(100 * ok / len(Q))
    print(f"{M.lab:<34s}" + "".join(f"{v:>11.1f}%" for v in row), flush=True)
    del M; torch.cuda.empty_cache()
print(f"(95% interval about +-{196 * (0.25 ** 0.5) / N ** 0.5:.0f} points per cell; GPT-2 sees the last 1,024 tokens, SmolLM2 the last 8,192)")
