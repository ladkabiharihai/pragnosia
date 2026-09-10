# Progress

Append only. Every entry carries a UTC date and time. Failures, dead ends and mistakes are recorded here in
the same way as results, because a log that only records wins cannot be used to make decisions.

`src/common.py:journal()` appends here automatically at the end of every run.

## 2026-09-10

- `2026-09-10 09:41 UTC` **Clean slate.** Deleted 176 files: all code, experiments and documents from the
  previous effort. Recoverable from git at `9139b56`, which is also on origin. VISION.md rewritten to the eight
  goals alone, with no architecture, roadmap or claims in it.
- `2026-09-10 09:44 UTC` Mistake during the wipe, recorded because it nearly cost the repository: deleting
  `.gitignore` caused `git add -A` to start staging 26 GB of training bins into git. It hung and left an
  `index.lock`. Killed the git process by verified PID, cleared the lock, restored ignore rules. Repository is
  clean, not bloated.
- `2026-09-10 09:51 UTC` **Environment verified by measurement**, not assumption: H100 NVL 93.6 GiB shared with
  production (~53 GB held by vLLM, kokoro, chatterbox, whisper), python 3.10.20, torch 2.11.0+cu130, bf16
  available. Recorded in CONSTRAINTS.md.
- `2026-09-10 09:51 UTC` **Correction to a stated premise.** It was expected that raw text for re-tokenising
  was on the 3 TB disk. It was not: one `.jsonl`, zero parquet or arrow, and about 1 GB of HuggingFace dataset
  cache. The 290.2B tokens existed only as uint16 ids. Also established that those ids were trainable without a
  tokenizer, and that anything requiring text (bits per byte, decoding, chat) was not.
- `2026-09-10 09:53 UTC` **Structure created**: CONSTRAINTS.md, PROGRESS.md, README.md, `src/common.py`,
  `tests/test_common.py`. Five files. Guards for production health, GPU capping, PID identification,
  determinism and logging are enforced in code rather than left to discipline.
- `2026-09-10 09:53 UTC` `tests/test_common.py` passes, 15 of 15 checks, including that `gpu_processes()`
  identifies processes by verified pid and command line rather than by name pattern.
- `2026-09-10 09:55 UTC` **All project data deleted at the user's instruction**: `/mnt/kv_cache/pragnosia_data`
  (584 GB: 12 tokenized corpora totalling 290.2B tokens, and 46 checkpoints including `p124f.pt`, the only
  model that carried verified claims) plus a stray 1.2 GB checkpoint. Not recoverable. Disk went from 23% to 6%
  used, 3.1 TB free. Production verified healthy immediately after.
- `2026-09-10 09:55 UTC` **Deliberately not deleted**, with reasons: `lmcache_disk` (153 GB) is production's
  vLLM KV-cache offload and deleting it risks the one thing that must not break; `hf_home` and `hf_cache`
  (17 GB) hold dataset caches and baseline models, so removing them saves 17 GB of 3.5 TB and makes the
  re-download slower.
- `2026-09-10 09:56 UTC` CONSTRAINTS.md and the tests updated to the true state: no data, no tokenizer, no
  checkpoints. The test now asserts that what the code reports matches the disk, rather than assuming data is
  present.

**Current state: the repository is five files and a set of rules. There is no data, no tokenizer, no
checkpoint, and no architecture. Nothing is running. Nothing proceeds without approval.**
- `2026-09-10 10:04 UTC` **Project venv created** at `.venv`: python 3.12.3, torch 2.14.0+cu130, numpy 2.5.3,
  tokenizers 0.23.2. Verified rather than assumed: CUDA available, bf16 supported, device reported as NVIDIA
  H100 NVL, a matmul executed on the device. 5.4 GB, on `/` which has 266 GB free. `parakeet_env` belongs to
  production tooling and is no longer used or modified by this project. `src/common.py`, CONSTRAINTS.md and
  README.md all switched to the new interpreter; tests re-run against it, 13 of 13 pass.
- `2026-09-10 10:04 UTC` **VISION.md gained a ninth goal: solve context rot.** Stated as a goal only, with no
  proposed mechanism, consistent with the other eight.
- `2026-09-10 10:04 UTC` Standing instruction recorded: **all research restarts from scratch, and nothing is
  downloaded, built or run until explicitly asked.** The previous data mix is recorded in CONSTRAINTS.md as a
  historical record of what was used, not as a plan.

**Waiting on instruction. Nothing is running. No data will be downloaded until asked.**
- `2026-09-10 10:44:33 UTC` **sample-sources** (1.3 min) ok
  stream 1500 docs per source to measure document length distributions before committing to a bulk download
  (log `logs/20260910-104316-sample-sources.log`, seed 0)
- `2026-09-10 10:46:52 UTC` **Source sampling done.** Streamed 300-4000 docs per candidate to measure document length before any bulk download.
  Web confirmed short: FineWeb-Edu median 693 tok, DCLM 697. Cosmopedia capped at 2841 max, so synthetic textbooks contribute nothing to long context. Long-document sources that WORK ungated: common-pile/arxiv_papers (median 8420, 52.3% over 8k, max 418806), neuralwork/arxiver (median 9022, 58.8% over 8k), wikimedia/wikipedia (median 5198, 31% over 8k). FAILED: peS2o, dolma, proof-pile-2, scientific_papers, github-code-clean all use loading scripts that datasets 5.0.1 removed; starcoderdata, the-stack-smol, tiny-codes are gated and need an HF token. Both long-doc sources in the original proposal were therefore unusable as specified. Key measurement: in web data 1.3-1.6% of documents exceed 8k tokens but they hold 14.8-16.8% of all tokens, so the web long tail is itself a usable long-context source. Gutenberg dropped at the user's instruction (observed to degrade the model).
- `2026-09-10 10:53:19 UTC` **Dataset survey with HF auth.** 30+ candidates streamed and measured; two earlier claims corrected.
  CORRECTION 1: peS2o is NOT a long-document source. Median 318 tokens in both olmo-mix and dolmino configs, i.e. abstracts rather than full text. It was a load-bearing part of the first proposed mix and is now dropped. CORRECTION 2: an olmo-mix median of 12,718 tokens was a sampling artifact. Streaming reads files in order, so sampling a multi-subset default config measures whichever subset comes first; that number belonged to algebraic-stack, not the mix. Measured long-document sources that work: olmo-mix/arxiv (median 11751, 78.8% over 8k), olmo-mix/algebraic-stack (12718, 76.0%), common-pile/arxiv_papers (8420, 52.3%, max 418806), govreport (11571, 67.2%), wikimedia/wikipedia (5874, 38.0%). Note wikimedia/wikipedia median 5874 vs olmo-mix/wiki median 389: same source, different chunking, so the version chosen matters more than the source. Gating solved by the token: bigcode/starcoderdata now reachable (median 952, max 224220). Still unreachable, all loading-script based: pile-of-law, dolma, proof-pile-2, scientific_papers, github-code-clean. the-stack-v2 stores pointers not content and is unusable directly.
- `2026-09-10 11:11:38 UTC` **download-200B** (10.4 min) ok
  505 GB / 200.8B tokens across 1250 shards, 12 parallel workers. Raw only: nothing is tokenized.
  (log `logs/20260910-110114-download-200B.log`, seed 0)
