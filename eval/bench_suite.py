"""STANDARD REASONING / KNOWLEDGE / ARITHMETIC BENCHMARKS, our base model against GPT-2 124M and SmolLM2-135M, every model
scored the same way (2026-10-05; user: "what's the generation quality, factual, reasoning and arithmetic capability").
Zero-shot, local data, the usual protocol of the lm-evaluation-harness family:
  HellaSwag (validation)   pick the ending with the highest log-likelihood per character (acc_norm)
  PIQA (validation)        the same, two solutions
  ARC-Easy / ARC-Challenge (test), OpenBookQA (test)   "Question: ... Answer:" + each choice, acc_norm
  LAMBADA (OpenAI test)    the last word, greedy, exact (accuracy)
  GSM8K (test)             8 worked examples from the TRAIN split as the prompt, greedy, the final number exact
  facts (48), arithmetic (60, four-way, distractors are real model errors), reasoning (400, generated, four-way): src/facts.py
  arithmetic exact (60): the answer generated, first integer exact
usage: eval/bench_suite.py <our ckpt>[=label] [...]     N=<items per benchmark, default 1000>"""
import os, sys, glob, re, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
from src.common import cap_gpu
cap_gpu(14.0)
import numpy as np, torch, torch.nn.functional as F, datasets
from src.bench import Ours
from src.facts import FACTS, arithmetic_items, reasoning_items_large
N = int(os.environ.get("N", "1000")); BASE = "/mnt/kv_cache/hf_home/datasets"
def arrow(pat):
    d, f_ = pat.split("/*/")                                # <dataset>/<config>/<version>/<hash>/<file>: search below
    fs = sorted(glob.glob(f"{BASE}/{d}/**/{f_}", recursive=True))
    assert fs, f"no files for {pat}"
    return [r for f in fs for r in datasets.Dataset.from_file(f)]

class HF:
    def __init__(self, name, lab):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache")
        self.m = AutoModelForCausalLM.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache", torch_dtype=torch.bfloat16).cuda().eval()
        self.lab = lab; self.max_ctx = getattr(self.m.config, "max_position_embeddings", 1024)
    def encode(self, t): return self.tok.encode(t)
    def decode(self, ids): return self.tok.decode(ids)
    @torch.no_grad()
    def logprobs(self, ids):
        ids = ids[-self.max_ctx:]
        return self.m(torch.tensor([ids], device="cuda")).logits[0].float().log_softmax(-1)
    @torch.no_grad()
    def generate(self, ids, n):
        out = self.m.generate(torch.tensor([ids[-(self.max_ctx - n):]], device="cuda"), max_new_tokens=n, do_sample=False,
                              pad_token_id=self.tok.eos_token_id)
        return out[0, len(ids[-(self.max_ctx - n):]):].tolist()

class OursM:
    def __init__(self, ck, lab):
        self.o = Ours(ck, "/mnt/kv_cache/tok_v8192.json"); self.lab = lab
    def encode(self, t): return self.o.encode(t)
    def decode(self, ids): return self.o.decode(ids)
    @torch.no_grad()
    def logprobs(self, ids):
        m = self.o.m; x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.linear(m.trunk(x), m.e.weight)[0].float().log_softmax(-1)
    def generate(self, ids, n): return self.o.generate(ids, n, temp=0.0)

def cont_lp(M, ctx, cont):
    """log p(cont | ctx), summed over the continuation's tokens"""
    a = M.encode(ctx); b = M.encode(ctx + cont)[len(a):] if M.encode(ctx + cont)[:len(a)] == a else M.encode(cont)
    ids = a + b; lp = M.logprobs(ids)
    y = torch.tensor(ids[len(a):], device="cuda")
    return float(lp[len(a) - 1:len(ids) - 1].gather(-1, y.unsqueeze(-1)).sum())

def mc(M, items, norm=True):
    """items: (context, [choices], gold index) -> acc_norm (log-likelihood per character), or summed (norm=False, the
    scorecard's rule for the short-answer probes)"""
    right = 0
    for ctx, ch, g in items:
        sc = [cont_lp(M, ctx, c) / (max(1, len(c)) if norm else 1) for c in ch]
        right += int(np.argmax(sc)) == g
    return 100 * right / len(items)

def first_int(t):
    m_ = re.search(r"-?\d[\d,]*", t); return m_.group(0).replace(",", "") if m_ else None

def suite():
    rng = np.random.default_rng(0)
    def take(rows): return [rows[i] for i in rng.permutation(len(rows))[:N]] if len(rows) > N else rows
    hs = take(arrow("Rowan___hellaswag/default/*/hellaswag-validation.arrow"))
    pq = take(arrow("piqa/default/*/piqa-validation.arrow"))
    ae = take(arrow("allenai___ai2_arc/ARC-Easy/*/ai2_arc-test.arrow"))
    ac = take(arrow("allenai___ai2_arc/ARC-Challenge/*/ai2_arc-test.arrow"))
    ob = take(arrow("openbookqa/main/*/openbookqa-test.arrow"))
    lb = take(arrow("EleutherAI___lambada_openai/en/*/lambada_openai-test.arrow"))
    gtest = arrow("openai___gsm8k/main/*/gsm8k-test.arrow"); gtrain = arrow("openai___gsm8k/main/*/gsm8k-train.arrow")
    gs = [gtest[i] for i in rng.permutation(len(gtest))[:min(N, 250)]]
    def pre(t): return re.sub(r"\[.*?\]", "", t.replace(" [title]", ". ")).strip()
    B = {
        "HellaSwag": [(pre(r["activity_label"] + ": " + r["ctx"]), [" " + pre(e) for e in r["endings"]], int(r["label"])) for r in hs],
        "PIQA": [(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"])) for r in pq],
        "ARC-Easy": [(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                      r["choices"]["label"].index(r["answerKey"])) for r in ae if r["answerKey"] in r["choices"]["label"]],
        "ARC-Challenge": [(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                           r["choices"]["label"].index(r["answerKey"])) for r in ac if r["answerKey"] in r["choices"]["label"]],
        "OpenBookQA": [(f"Question: {r['question_stem']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                        r["choices"]["label"].index(r["answerKey"])) for r in ob],
    }
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer'].split('####')[0].strip()} The answer is "
                    f"{r['answer'].split('####')[1].strip()}.\n\n" for r in gtrain[:8])
    return B, lb, gs, shots

B, LB, GS, SHOTS = suite()
models = [OursM(a.split("=")[0], a.split("=")[1] if "=" in a else os.path.basename(a)) for a in sys.argv[1:]] + \
         [HF("gpt2", "GPT-2 124M"), HF("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M")]
rows = {}
for M in models:
    r = {}
    for name, items in B.items():
        r[name] = mc(M, items); print(f"  [{M.lab}] {name} {r[name]:.1f}", flush=True)
    ok = 0
    for row in LB:
        t = row["text"]; ctx, last = t.rsplit(" ", 1)
        out = M.decode(M.generate(M.encode(ctx), 6)).strip()
        ok += out.split(" ")[0].strip(".,;:!?\"'") == last.strip(".,;:!?\"'")
    r["LAMBADA"] = 100 * ok / len(LB); print(f"  [{M.lab}] LAMBADA {r['LAMBADA']:.1f}", flush=True)
    ok = 0
    for row in GS:
        out = M.decode(M.generate(M.encode(SHOTS + f"Question: {row['question']}\nAnswer:"), 160))
        blk = out.split("\n\n")[0]; m_ = re.search(r"answer is\s*\$?(-?[\d,]+)", blk)
        nums = re.findall(r"-?\d[\d,]*", blk)
        pred = m_.group(1).replace(",", "") if m_ else (nums[-1].replace(",", "") if nums else None)
        ok += pred == row["answer"].split("####")[1].strip().replace(",", "")
    r["GSM8K"] = 100 * ok / len(GS); print(f"  [{M.lab}] GSM8K {r['GSM8K']:.1f} (n={len(GS)})", flush=True)
    r["facts (48)"] = mc(M, [(q, [ans] + list(wr), 0) for q, ans, wr in FACTS], norm=False)
    ar = arithmetic_items(60)
    r["arithmetic 4-way"] = mc(M, [(s_, [a_] + list(wr), 0) for s_, a_, wr in ar], norm=False)
    ex = 0
    for s_, a_, _ in ar:
        d = first_int(M.decode(M.generate(M.encode(s_), 6))); ex += d == a_.strip()
    r["arithmetic exact"] = 100 * ex / len(ar)
    re_ = reasoning_items_large(400)
    r["reasoning (400)"] = mc(M, [(it[0], [it[1]] + list(it[2]), 0) for it in re_], norm=False)
    print(f"  [{M.lab}] facts {r['facts (48)']:.1f}  arithmetic {r['arithmetic 4-way']:.1f} / exact {r['arithmetic exact']:.1f}  "
          f"reasoning {r['reasoning (400)']:.1f}", flush=True)
    rows[M.lab] = r
    del M; torch.cuda.empty_cache()
names = list(next(iter(rows.values())).keys())
print("\n" + f"{'benchmark':<20s}" + "".join(f"{l:>16s}" for l in rows))
for n in names: print(f"{n:<20s}" + "".join(f"{rows[l][n]:>15.1f}%" for l in rows))
print("chance: HellaSwag 25, PIQA 50, ARC 25, OpenBookQA 25, facts/arithmetic/reasoning 25; LAMBADA, GSM8K, exact ~0")
