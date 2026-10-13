"""FINE-TUNE SET, SECOND GENERATION (2026-10-13; data from eval/get_sft_data.py). Six kinds, each with a token budget:
  chat        smol-smoltalk (minus its self-oss-instruct copy), tulu-3-sft-mixture (a sample)
  arithmetic  orca-math word problems, MetaMathQA, GSM8K TRAIN, and generated plain arithmetic in chat form (the base answers
              plain sums 8/8 and the old fine-tune took that to 3/8: the new set must keep it)
  coding      Magicoder-OSS-Instruct, self-oss-instruct-sc2 (executed solutions)
  thinking    OpenThoughts-114k: "<think> ... </think>" then the solution, examples that fit a 4,096-token row
  cognition   Psych-101 human choices (loss on the choices only), minus the experiments the goal-2 battery uses (digit span,
              n-back) and minus a held-out tenth of experiments for testing prediction of human behaviour
Every example is "User: ...\\nAssistant: ..." (multi-turn kept; a system message as "System: ..."), loss on the assistant's tokens
only, packed into 4,097-token rows (eval/selfimprove.py's format: train.bin / train.mask / val.bin / val.mask).
DECONTAMINATION: an example is dropped if any 10-word span of it (stride 4) occurs in an evaluation set we report (SciQ test,
ARC-Easy / Challenge test, HellaSwag validation, PIQA validation, OpenBookQA test, LAMBADA test, GSM8K test), or if it names one
of the goal-2 battery's classic puzzles (bat and ball, lily pads, widgets, Linda / bank teller, Wason, Sally-Anne, false
belief, syllogisms, digit span, n-back, free recall, conjunction fallacy, cognitive reflection).
usage: eval/build_sft2.py <out_dir>        BUDGET_SCALE=<x> multiplies every budget (default 1)"""
import os, sys, re, glob, random, json, time
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
import numpy as np, datasets
from tokenizers import Tokenizer

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
B = "/mnt/kv_cache/hf_home/datasets"; tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json"); EOD = 8191; ROW = 4097
SC = float(os.environ.get("BUDGET_SCALE", "1")); r = random.Random(20261013)
BUDGET = {k: int(v * 1e6 * SC) for k, v in dict(chat=110, tulu=40, orca=35, metamath=50, gsm8k=5, arith=8, magicoder=30,
                                                   selfoss=25, thinking=60, psych=25).items()}
KIND = dict(chat="chat", tulu="chat", orca="arithmetic", metamath="arithmetic", gsm8k="arithmetic", arith="arithmetic",
            magicoder="coding", selfoss="coding", thinking="thinking", psych="cognition")

# ---------------------------------------------------------------- decontamination
def words(t): return re.findall(r"[a-z0-9]+", t.lower())
def arrow(pat):
    d, f_ = pat.split("/*/"); fs = sorted(glob.glob(f"{B}/{d}/**/{f_}", recursive=True)); assert fs, pat
    return [x for f in fs for x in datasets.Dataset.from_file(f)]
EV = []
EV += [x["question"] + " " + x["correct_answer"] + " " + x.get("support", "") for x in arrow("sciq/default/*/sciq-test.arrow")]
for c in ("ARC-Easy", "ARC-Challenge"): EV += [x["question"] for x in arrow(f"allenai___ai2_arc/{c}/*/ai2_arc-test.arrow")]
EV += [x["ctx"] + " " + " ".join(x["endings"]) for x in arrow("Rowan___hellaswag/default/*/hellaswag-validation.arrow")]
EV += [x["goal"] + " " + x["sol1"] + " " + x["sol2"] for x in arrow("piqa/default/*/piqa-validation.arrow")]
EV += [x["question_stem"] for x in arrow("openbookqa/main/*/openbookqa-test.arrow")]
EV += [x["text"] for x in arrow("EleutherAI___lambada_openai/en/*/lambada_openai-test.arrow")]
EV += [x["question"] for x in arrow("openai___gsm8k/main/*/gsm8k-test.arrow")]
NG = 10; GRAMS = set()
for t in EV:
    w = words(t)
    for i in range(len(w) - NG + 1): GRAMS.add(" ".join(w[i:i + NG]))
BAN = re.compile(r"bat and (a )?ball|lily ?pads?|\bwidgets?\b|\blinda\b|bank teller|\bwason\b|selection task|\bsally\b|false[- ]belief|"
                 r"syllogis|digit span|n-back|2-back|two-back|serial position|free recall|conjunction fallacy|cognitive reflection", re.I)
print(f"decontamination: {len(EV):,} evaluation texts, {len(GRAMS):,} {NG}-word spans", flush=True)
def clean(text):
    if BAN.search(text): return False
    w = words(text)
    return not any(" ".join(w[i:i + NG]) in GRAMS for i in range(0, max(1, len(w) - NG + 1), 4))

# ---------------------------------------------------------------- sources -> [(text, loss?)]
def conv(msgs):                       # [{"role"/"from", "content"/"value"}] -> segments
    seg = []
    for m in msgs:
        role = m.get("role", m.get("from")); txt = (m.get("content") or m.get("value") or "").strip()
        if role in ("system",): seg.append((f"System: {txt}\n", False))
        elif role in ("user", "human"): seg.append((f"User: {txt}\n", False))
        else: seg += [("Assistant:", False), (f" {txt}\n", True)]
    return seg
def qa(q, a): return [(f"User: {q.strip()}\n", False), ("Assistant:", False), (f" {a.strip()}\n", True)]
def ds(repo, cfg=None, split="train"): return datasets.load_dataset(repo, cfg, cache_dir=B)[split]

def gen_chat():
    for x in ds("HuggingFaceTB/smol-smoltalk"):
        if x["source"] != "self-oss-instruct": yield conv(x["messages"])
def gen_tulu():
    d = ds("allenai/tulu-3-sft-mixture"); idx = list(range(len(d))); r.shuffle(idx)
    for i in idx: yield conv(d[i]["messages"])
def gen_orca():
    for x in ds("microsoft/orca-math-word-problems-200k"): yield qa(x["question"], x["answer"])
def gen_metamath():
    for x in ds("meta-math/MetaMathQA"): yield qa(x["query"], x["response"])
def gen_gsm8k():
    for x in ds("openai/gsm8k", "main"):
        work, ans = x["answer"].split("####"); work = re.sub(r"<<[^>]*>>", "", work).strip()
        yield qa(x["question"], f"{work}\nThe answer is {ans.strip()}.")
def gen_arith():                      # plain arithmetic in chat form (the skill the old fine-tune erased)
    W = {"+": ["plus", "added to"], "-": ["minus"], "*": ["times", "multiplied by"], "/": ["divided by"]}
    while True:
        op = r.choice("+-*/"); a = r.randint(0, 999 if op in "+-" else 99); b = r.randint(0, 999 if op in "+-" else 12)
        if op == "-" and b > a: a, b = b, a
        if op == "/": b = max(b, 1); a = a * b
        c = {"+": a + b, "-": a - b, "*": a * b, "/": a // max(b, 1)}[op]; w = r.choice(W[op]); s = {"*": "x"}.get(op, op)
        q = r.choice([f"What is {a} {w} {b}?", f"Calculate {a} {s} {b}.", f"{a} {s} {b} = ?", f"How much is {a} {w} {b}?"])
        yield qa(q, r.choice([f"{a} {w} {b} is {c}.", f"{a} {s} {b} = {c}", f"{c}", f"The answer is {c}."]))
def gen_magicoder():
    for x in ds("ise-uiuc/Magicoder-OSS-Instruct-75K"): yield qa(x["problem"], x["solution"])
def gen_selfoss():
    for x in ds("bigcode/self-oss-instruct-sc2-exec-filter-50k"): yield qa(x["prompt"], x["response"])
def gen_thinking():
    for x in ds("open-thoughts/OpenThoughts-114k"):
        c = x["conversations"]; a = c[1]["value"]
        a = a.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
        a = a.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()
        yield qa(c[0]["value"], a)
PSY_TEST = set()
def gen_psych():                      # loss on the human choices (between << and >>) only
    d = ds("marcelbinz/Psych-101"); exps = sorted(set(d["experiment"]))
    keep = [e for e in exps if "digitspan" not in e and "nback" not in e]
    PSY_TEST.update(r.sample(keep, max(1, len(keep) // 10)))
    for x in d:
        if x["experiment"] not in keep or x["experiment"] in PSY_TEST: continue
        seg = []
        for part in re.split(r"(<<.*?>>)", x["text"]):
            if part.startswith("<<"): seg += [("<<", False), (part[2:-2], True), (">>", False)]
            elif part: seg.append((part, False))
        yield seg
GEN = dict(chat=gen_chat, tulu=gen_tulu, orca=gen_orca, metamath=gen_metamath, gsm8k=gen_gsm8k, arith=gen_arith,
           magicoder=gen_magicoder, selfoss=gen_selfoss, thinking=gen_thinking, psych=gen_psych)

# ---------------------------------------------------------------- tokenize, budget, pack
# rows stream to disk (~390M tokens as Python lists would need ~14 GB of RAM); shuffled afterwards through a memory map
FT, FM = open(f"{OUT}/_rows.bin", "wb"), open(f"{OUT}/_rows.mask", "wb"); NROWS = 0; report = {}
cur_t, cur_m = [], []
def flush():
    global cur_t, cur_m, NROWS
    if cur_t:
        pad = ROW - len(cur_t)
        np.array(cur_t + [EOD] * pad, dtype=np.uint16).tofile(FT); np.array(cur_m + [0] * pad, dtype=np.uint8).tofile(FM); NROWS += 1
    cur_t, cur_m = [], []
for name, g in GEN.items():
    t0 = time.time(); used = n_ok = n_dirty = n_long = 0; loss_tok = 0
    for seg in g():
        if used >= BUDGET[name]: break
        text = "".join(s for s, _ in seg)
        if name != "arith" and not clean(text): n_dirty += 1; continue
        ids, msk = [], []
        for s, l in seg:
            e = tok.encode(s).ids; ids += e; msk += [int(l)] * len(e)
        ids.append(EOD); msk.append(0)
        if len(ids) > ROW - 1:
            if name != "psych": n_long += 1; continue
            ids, msk = ids[:ROW - 1], msk[:ROW - 1]          # a long experiment: its first 4,096 tokens (instructions + early trials)
        if len(cur_t) + len(ids) > ROW: flush()
        cur_t += ids; cur_m += msk; used += len(ids); loss_tok += sum(msk); n_ok += 1
    flush()
    report[name] = dict(kind=KIND[name], examples=n_ok, tokens=used, loss_tokens=loss_tok, dropped_contaminated=n_dirty, dropped_too_long=n_long)
    print(f"  {name:10s} {KIND[name]:10s} {n_ok:>8,} examples {used/1e6:6.1f}M tokens ({100*loss_tok/max(1,used):.0f}% loss)  "
          f"dropped: {n_dirty:,} contaminated, {n_long:,} too long   {time.time()-t0:.0f}s", flush=True)
FT.close(); FM.close()
T = np.memmap(f"{OUT}/_rows.bin", dtype=np.uint16, mode="r").reshape(NROWS, ROW); M = np.memmap(f"{OUT}/_rows.mask", dtype=np.uint8, mode="r").reshape(NROWS, ROW)
idx = np.random.default_rng(20261013).permutation(NROWS); nv = max(1, NROWS // 50); share = 0.0
for nm, sel in (("val", idx[:nv]), ("train", idx[nv:])):
    with open(f"{OUT}/{nm}.bin", "wb") as ft, open(f"{OUT}/{nm}.mask", "wb") as fm:
        for k in range(0, len(sel), 2048):
            j = np.sort(sel[k:k + 2048]); r.shuffle(j := list(j)); T[j].tofile(ft); M[j].tofile(fm); share += float(M[j].sum())
os.remove(f"{OUT}/_rows.bin"); os.remove(f"{OUT}/_rows.mask")
report["_total"] = dict(rows=NROWS, train_rows=NROWS - nv, val_rows=nv, tokens=NROWS * ROW, loss_share=share / (NROWS * ROW),
                        psych101_held_out_experiments=sorted(PSY_TEST))
json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
print(f"done: {NROWS:,} rows of {ROW} ({NROWS*ROW/1e6:.0f}M tokens, {100*share/(NROWS*ROW):.1f}% carry loss) -> {OUT}; held-out Psych-101 experiments: {sorted(PSY_TEST)}", flush=True)
