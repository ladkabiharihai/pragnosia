"""Data acquisition and inspection.

Sampling comes before downloading. The mix percentages proposed on 2026-09-10 rest on claims about document
LENGTH ("papers are 5-20k tokens", "web is mostly under 1k"), and those were recalled, not measured. Since
context rot is a first-class goal, the long-document budget is the load-bearing part of the mix, so it gets
verified before anything is pulled in bulk.

Streaming is used deliberately: it reads only the records consumed, so a distribution can be measured from a
few thousand documents without downloading a terabyte.

NOTE ON UNITS. There is no tokenizer yet, so lengths are measured in BYTES and converted with a stated
approximation (~4 bytes/token for English prose, ~3.5 for code). Every token figure this produces is an
estimate and is labelled as one. The byte figures are exact.
"""
from __future__ import annotations
import json
import statistics
import sys
from dataclasses import dataclass, field

BYTES_PER_TOKEN = {"prose": 4.0, "code": 3.5}


@dataclass
class Source:
    key: str
    repo: str
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    kind: str = "prose"
    note: str = ""


# The candidate mix. Gutenberg is deliberately absent: the user has observed it degrading the model, which is
# consistent with it being pre-1928 public domain text with an archaic register and OCR artifacts. Removing it
# costs the longest-document source in the mix, so the long-context budget now rests on papers and code, and
# that is exactly what this sampling has to test.
SOURCES = [
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-10BT",
           note="web backbone, classifier-filtered for educational value"),
    Source("dclm", "mlfoundations/dclm-baseline-1.0-parquet",
           note="second web source with a different filter"),
    Source("pes2o", "allenai/peS2o", "v2",
           note="LONG: scientific papers, the primary long-document source now"),
    Source("starcoder", "bigcode/starcoderdata", "python", text_field="content", kind="code",
           note="LONG when concatenated by repository rather than by file"),
    Source("finemath", "HuggingFaceTB/finemath", "finemath-4plus",
           note="math and reasoning density"),
    Source("cosmopedia", "HuggingFaceTB/cosmopedia", "web_samples_v2",
           note="synthetic textbook, measurably helps small models"),
    Source("openhermes", "teknium/OpenHermes-2.5", text_field="conversations",
           note="instruction, for a later stage and not for pretraining"),
]

THRESHOLDS = (2_000, 4_000, 8_000, 16_000, 32_000)   # in TOKENS (estimated from bytes)


def _text_of(rec, field: str) -> str:
    v = rec.get(field)
    if isinstance(v, str):
        return v
    if isinstance(v, list):                                   # chat-style records
        return "\n".join(str(t.get("value", t)) if isinstance(t, dict) else str(t) for t in v)
    return "" if v is None else str(v)


def sample(src: Source, n: int = 2000, timeout_note: str = "") -> dict:
    """Stream n documents and measure the byte-length distribution. Returns a plain dict so it can be
    journalled and compared later without re-running."""
    from datasets import load_dataset
    args = (src.repo, src.config) if src.config else (src.repo,)
    ds = load_dataset(*args, split=src.split, streaming=True)
    lens, first = [], None
    for i, rec in enumerate(ds):
        if i >= n:
            break
        t = _text_of(rec, src.text_field)
        if first is None and t:
            first = t[:300]
        lens.append(len(t.encode("utf-8")))
    if not lens:
        return {"key": src.key, "error": "no records read"}
    bpt = BYTES_PER_TOKEN[src.kind]
    toks = sorted(x / bpt for x in lens)
    q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))]
    return {
        "key": src.key, "repo": src.repo, "n": len(toks), "kind": src.kind,
        "median_tok": q(0.50), "p90_tok": q(0.90), "p99_tok": q(0.99), "max_tok": toks[-1],
        "mean_tok": statistics.fmean(toks),
        "frac_over": {str(t): sum(1 for x in toks if x >= t) / len(toks) for t in THRESHOLDS},
        "sample": first, "note": src.note,
    }


# ------------------------------------------------------------------------------------------- acquisition
DL_ROOT = "/mnt/kv_cache/hf_dl"

# Measured 2026-09-10, not guessed: tokens per byte of PARQUET, from one shard of each family.
# fineweb-edu: 2153 MB parquet -> 869.8M estimated tokens = 0.40. json.gz compresses differently, hence 0.28.
PLAN_200B = [
    # key,            repo,                                     prefix,                  tok/byte, target tokens
    ("fineweb-edu",   "HuggingFaceFW/fineweb-edu",              "sample/100BT",          0.40, 64.0e9),
    ("olmo-arxiv",    "allenai/olmo-mix-1124",                  "data/arxiv/",           0.28, None),   # take all
    ("algebraic",     "allenai/olmo-mix-1124",                  "data/algebraic-stack/", 0.28, None),   # take all
    ("wikipedia",     "wikimedia/wikipedia",                    "20231101.en/",          0.40, None),   # take all
    ("finemath",      "HuggingFaceTB/finemath",                 "finemath-4plus/",       0.40, None),   # take all
    ("open-web-math", "open-web-math/open-web-math",            "data/",                 0.40, 6.0e9),
    ("starcoder",     "bigcode/starcoderdata",                  ".parquet",              0.45, 26.0e9),
    ("dclm",          "mlfoundations/dclm-baseline-1.0-parquet", ".parquet",             0.40, 82.8e9),
]


def plan_files(repo: str, prefix: str, ratio: float, target: float | None):
    """Choose shards up to a token target. `target=None` means take everything the repo has, which is what the
    four small sources need: arXiv, algebraic-stack, Wikipedia and FineMath together hold only 21B tokens, so
    a percentage target larger than the corpus is not a plan, it is arithmetic that has not been checked."""
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo, repo_type="dataset", files_metadata=True)
    fs = [f for f in info.siblings
          if prefix in f.rfilename and f.rfilename.endswith((".parquet", ".json.gz"))]
    fs.sort(key=lambda f: f.rfilename)
    out, tok, byt = [], 0.0, 0
    for f in fs:
        if target is not None and tok >= target:
            break
        out.append(f.rfilename); tok += (f.size or 0) * ratio; byt += (f.size or 0)
    return out, tok, byt


# ------------------------------------------------------------------------------------------- tokenization
# Achieved mix, as actually downloaded on 2026-09-10. Four sources were capped by their true size rather than
# by choice: arXiv, algebraic-stack, Wikipedia and FineMath together hold 21B tokens, so their share is what
# exists, not what was asked for.
MIX = {"dclm": 0.413, "fineweb-edu": 0.322, "starcoderdata": 0.130, "finemath": 0.036,
       "open-web-math": 0.030, "arxiv": 0.030, "wikipedia": 0.023, "algebraic-stack": 0.015}
TEXT_FIELD = {"starcoderdata": "content"}


def shards(key: str) -> list[str]:
    """Files on disk for one source. arXiv and algebraic-stack are json.gz; the rest are parquet."""
    import glob
    pats = {"dclm": "*dclm*/**/*.parquet", "fineweb-edu": "*fineweb-edu*/**/*.parquet",
            "starcoderdata": "*starcoderdata*/**/*.parquet", "finemath": "*finemath*/**/*.parquet",
            "open-web-math": "*open-web-math*/**/*.parquet", "wikipedia": "*wikipedia*/**/*.parquet",
            "arxiv": "*olmo-mix*/data/arxiv/**/*.json.gz",
            "algebraic-stack": "*olmo-mix*/data/algebraic-stack/**/*.json.gz"}
    return sorted(glob.glob(f"{DL_ROOT}/{pats[key]}", recursive=True))


def texts(path: str, field: str = "text"):
    """Yield documents from a parquet or json.gz shard without loading the whole file where avoidable."""
    if path.endswith(".json.gz"):
        import gzip, json as _json
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                try:
                    t = _json.loads(line).get(field) or _json.loads(line).get("text")
                except Exception:
                    continue
                if t:
                    yield t
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=[field]):
            for x in batch.column(field):
                t = x.as_py()
                if t:
                    yield t
