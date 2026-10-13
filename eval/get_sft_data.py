"""FINE-TUNE DATA, SECOND GENERATION (2026-10-13; user: "first get better finetune data for arithmetic, coding, chat, cognition,
reasoning and thinking"). The first fine-tune data (Alpaca, Dolly, OpenAssistant, CAMEL, SciQ / ARC / CommonsenseQA train; 13.5M
tokens) taught a short-answer style and erased what the base could do (plain arithmetic 8/8 -> 3/8, digit copying 90 -> 10-30%).
Downloaded here into the local datasets cache, permissive licences only, every source's test split left alone:
  chat        HuggingFaceTB/smol-smoltalk (Apache-2.0; SmolLM2's SFT mix for 135M-360M), allenai/tulu-3-sft-mixture (ODC-BY)
  arithmetic  microsoft/orca-math-word-problems-200k (MIT), meta-math/MetaMathQA (MIT), openai/gsm8k TRAIN only (MIT)
  coding      ise-uiuc/Magicoder-OSS-Instruct-75K (MIT), bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-BY)
  thinking    open-thoughts/OpenThoughts-114k (Apache-2.0; full reasoning traces)
  cognition   marcelbinz/Psych-101 (Apache-2.0; human trial-by-trial choices, 160 experiments)
Decontamination against our evaluations (benchmark test sets and the goal-2 battery's classic puzzles) happens when the
fine-tune set is BUILT, not here. usage: eval/get_sft_data.py"""
import os, time
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
import datasets
BASE = "/mnt/kv_cache/hf_home/datasets"
SETS = [("HuggingFaceTB/smol-smoltalk", None), ("microsoft/orca-math-word-problems-200k", None), ("meta-math/MetaMathQA", None),
        ("openai/gsm8k", "main"), ("ise-uiuc/Magicoder-OSS-Instruct-75K", None), ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None),
        ("allenai/tulu-3-sft-mixture", None), ("marcelbinz/Psych-101", None), ("open-thoughts/OpenThoughts-114k", None)]
for repo, cfg in SETS:
    t0 = time.time()
    try:
        d = datasets.load_dataset(repo, cfg, cache_dir=BASE)
        print(f"OK   {repo:48s} " + ", ".join(f"{k} {len(v):,}" for k, v in d.items()) + f"   cols {list(next(iter(d.values())).features)[:6]}  {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {repo:48s} {type(e).__name__}: {str(e)[:160]}", flush=True)
print("SFT_DATA_DONE", flush=True)
