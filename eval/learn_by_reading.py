"""GOAL 7, LEARNING WITHOUT FORGETTING -- BY READING (2026-10-07). Fine-tuning new facts into the weights damaged general text
at every step count tried (eval/continual.py curve: no point learns without forgetting). This architecture's other way to
learn: read the facts once, keep going, answer later. The weights never change, so nothing already known can be
forgotten -- by construction. The question is how long the reading LASTS: N invented facts (eval/incontext.py's generator:
invented countries, capitals, machines, inventors), then D tokens of unrelated real prose, then a question about one of
them, four-way with every wrong option another answer from the same facts (hard). Read once, scored by log-probability.
usage: eval/learn_by_reading.py <ckpt>[=label] ...        N_FACTS=30  N_Q=150  DIST=0,2048,8192,32768"""
import os, sys, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(6.0)
import numpy as np, torch
from src.bench import Ours
import src.assoc as A
NF = int(os.environ.get("N_FACTS", "30")); NQ = int(os.environ.get("N_Q", "150"))
DIST = [int(d) for d in os.environ.get("DIST", "0,2048,8192,32768").split(",")]
rng = random.Random(1234)
SYL = ["vel", "dra", "osk", "arth", "mir", "tan", "quel", "lin", "zor", "va", "bre", "nth", "kal", "iso", "rum",
       "pel", "dor", "gha", "sen", "tri", "wex", "ulm", "fae", "rok", "yst", "cor", "nib", "sha", "lom", "dex"]
seen = set()
def fresh(n):
    while True:
        w = "".join(rng.choice(SYL) for _ in range(n)).capitalize()
        if w not in seen and len(w) >= 5: seen.add(w); return w
wiki = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
# each question gets its own fresh set of NF facts (half capitals, half inventors) and its own filler, shared by every model
ITEMS = []
for qi in range(NQ):
    seen.clear()                    # names unique WITHIN a question; two-syllable names are only 900, and a global
                                    # uniqueness rule looped forever once they ran out (2026-10-07: both first runs hung)
    cs = [(fresh(3), fresh(2)) for _ in range(NF // 2)]; ms = [(fresh(2), f"{fresh(2)} {fresh(2)}") for _ in range(NF - NF // 2)]
    facts = [f"The capital of {c} is {y}." for c, y in cs] + [f"The {mm} engine was invented by {p}." for mm, p in ms]
    rng.shuffle(facts)
    if qi % 2 == 0:
        c, y = rng.choice(cs); q = f"Question: which city is the capital of {c}? Answer:"; ans = y
        wrong = rng.sample([yy for cc, yy in cs if yy != y], 3)
    else:
        mm, p = rng.choice(ms); q = f"Question: who invented the {mm} engine? Answer:"; ans = p
        wrong = rng.sample([pp for m2, pp in ms if pp != p], 3)
    o = int(np.random.default_rng(qi).integers(0, len(wiki) - 200_000))
    ITEMS.append((" ".join(facts), q, ans, wrong, o))

print(f"LEARNING BY READING: {NF} invented facts read once, then D tokens of unrelated prose, then a question (4-way, every "
      f"wrong option another answer from the same facts; chance 25%); {NQ} questions per cell")
print(f"{'model':<30s}" + "".join(f"{'D=' + format(d, ','):>12s}" for d in DIST))
class HF:
    """a Hugging Face model for comparison (spec hf:<name>=<label>): the same facts, filler (our tokens decoded to text) and
    question; it sees the last max_position_embeddings tokens, so past its window the facts are simply gone"""
    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache")
        self.m = AutoModelForCausalLM.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache", torch_dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").cuda().eval()
        self.max_ctx = getattr(self.m.config, "max_position_embeddings", 1024)
    def lps(self, text, opts):
        a = self.tok.encode(text); out = []
        for c in opts:
            b = self.tok.encode(" " + c); ids = (a + b)[-self.max_ctx:]
            with torch.no_grad(): lp = self.m(torch.tensor([ids], device="cuda")).logits[0].float().log_softmax(-1)
            out.append(float(lp[-len(b) - 1:-1].gather(-1, torch.tensor(ids[-len(b):], device="cuda").unsqueeze(-1)).sum()))
        return out
from tokenizers import Tokenizer as _T
_tk = _T.from_file("/mnt/kv_cache/tok_v8192.json")
for spec in sys.argv[1:]:
    ck, lab = (spec.split("=") + [None])[:2]
    if ck.startswith("hf:"):
        h = HF(ck[3:]); print(f"{(lab or ck):<30s}", end="", flush=True)
        for D in DIST:
            ok = 0
            for facts, q, ans, wrong, off in ITEMS:
                f = np.asarray(wiki[off:off + 3 * D + 10]); f = f[f != 8191][:D].tolist()
                ok += int(np.argmax(h.lps(facts + "\n\n" + (_tk.decode(f) if D else "") + "\n\n" + q, [ans] + wrong))) == 0
            print(f"{100 * ok / len(ITEMS):>11.1f}%", end="", flush=True)
        print(flush=True); del h; torch.cuda.empty_cache(); continue
    o_ = Ours(ck, "/mnt/kv_cache/tok_v8192.json"); A.FUSED_SCAN = True
    if os.environ.get("RECALL_K") and getattr(o_.m, "recall_k", 0):
        o_.m.set_recall(int(os.environ["RECALL_K"]), span=0); lab = (lab or ck) + f" k={os.environ['RECALL_K']}"
    for mx in o_.m.mix:
        if hasattr(mx, "need_sent"): mx.need_sent = False
    row = []
    print(f"{(lab or ck):<30s}", end="", flush=True)
    for D in DIST:
        ok = 0
        for facts, q, ans, wrong, off in ITEMS:
            f = np.asarray(wiki[off:off + 3 * D + 10]); f = f[f != 8191][:D].tolist()
            ids = o_.encode(facts + "\n\n") + f + o_.encode("\n\n" + q)
            pre = o_.prefix(ids)
            lps = [o_.continuation_lp(pre, o_.encode(" " + c)) for c in [ans] + wrong]
            ok += int(np.argmax(lps)) == 0
        row.append(100 * ok / len(ITEMS)); print(f"{row[-1]:>11.1f}%", end="", flush=True)
    print(flush=True)
    del o_; torch.cuda.empty_cache()
print(f"(95% interval about +-{196 * 0.25 ** 0.5 / NQ ** 0.5:.0f} points per cell)")
