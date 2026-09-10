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
- `2026-09-10 11:44:18 UTC` **tokenize-1B-v8192** (2.7 min) ok
  1.0B tokens at vocab 8192, mix proportions as downloaded; uint16 bins, one per source
  (log `logs/20260910-114134-tokenize-1B-v8192.log`, seed 0)
- `2026-09-10 11:45:25 UTC` **tokenize-1B-longdocs** (0.1 min) ok
  arxiv and algebraic-stack, missed by a glob bug on the first pass
  (log `logs/20260910-114516-tokenize-1B-longdocs.log`, seed 0)
- `2026-09-10 11:50:49 UTC` **optim-ab** (4.8 min) ok
  Adam vs Fabric, 1500 steps, 16x1024 tokens, testbed transformer D512 L8 V8192
  (log `logs/20260910-114602-optim-ab.log`, seed 0)
- `2026-09-10 11:57:52 UTC` **optim-ab** (5.5 min) ok
  Adam vs Fabric, 1500 steps, 16x1024 tokens, testbed transformer D512 L8 V8192
  (log `logs/20260910-115224-optim-ab.log`, seed 0)
- `2026-09-10 12:02:44 UTC` **optim-ab** (4.1 min) ok
  Adam vs Fabric, 1500 steps, 16x1024 tokens, testbed transformer D512 L8 V8192
  (log `logs/20260910-115838-optim-ab.log`, seed 0)
- `2026-09-10 12:03:07 UTC` **1B-token slice tokenized and the optimizer tested against Adam.**
  Tokenized 1.01B tokens at vocab 8192 in the achieved mix proportions, uint16 bins per source at /mnt/kv_cache/tok1b_v8192. Long documents confirmed present: arXiv mean 21976 tokens/doc, algebraic-stack 17788. A glob bug initially skipped both long sources with NO FILES while the run reported success, because the HF cache inserts snapshots/<hash>/ into the path; fixed and re-run. OPTIMIZER ABLATION on a plain transformer testbed (D512 L8 V8192, 1500 steps, 16x1024, seed 0, 150-step warmup for both, LR swept until each optimum was INTERIOR): best Adam 5.8111 at lr 3e-3 with 8.00 B/param of state; best Fabric 4.9265 at lr 6e-3 with 2.02 B/param. Fabric is 0.885 nats better on 4x less optimizer state. Two earlier versions of this comparison were unfair to Adam and were discarded: the first gave neither optimizer warmup, which Adam needs and the trust ratio supplies implicitly; the second left Adam best at the sweep boundary, which measures the sweep rather than the optimizer. Caveats: single seed, 1500 steps, one testbed architecture, no weight decay.
- `2026-09-10 12:11:19 UTC` **growth-ab** (0.4 min) FAILED: RuntimeError('The size of tensor a (384) must match the size of tensor b (512) at non-singleton dimension 0')
  width growth 384->512 at step 400 of 1200; optimizer treatment is the only difference between arms
  (log `logs/20260910-121055-growth-ab.log`, seed 0)
- `2026-09-10 12:15:21 UTC` **growth-ab** (2.1 min) ok
  width growth 384->512 at step 400 of 1200; optimizer treatment is the only difference between arms
  (log `logs/20260910-121312-growth-ab.log`, seed 0)
- `2026-09-10 12:19:54 UTC` **growth-ab** (2.0 min) ok
  width growth 384->512 at step 400 of 1200; optimizer treatment is the only difference between arms
  (log `logs/20260910-121753-growth-ab.log`, seed 0)
- `2026-09-10 12:24:11 UTC` **growth-ab** (1.9 min) ok
  width growth 384->512 at step 400 of 1200; optimizer treatment is the only difference between arms
  (log `logs/20260910-122220-growth-ab.log`, seed 0)
- `2026-09-10 12:27:11 UTC` **Growth ablation: carrying optimizer state across a growth is worth about 1 nat. Intrinsic LR failed three times.**
  Width growth 384->512 mid-run, function preserving, optimizer treatment the only difference between arms. Across three repeats: adam-reset takes a growth shock of +0.894, +1.014, +1.039 in loss, while Fabric carrying its state takes -0.075 to -0.124, i.e. it improves through the growth rather than being knocked back. Final loss 6.13-6.19 for adam-reset against 5.29-5.33 for Fabric. Growth-aware median fill beat zero fill in all three repeats but by a small margin: -0.0165, -0.0299, -0.0399 final loss. Carrying state AT ALL is the large effect; how the new slots are filled is a small one. Also found: carrying state without resizing it does not degrade, it raises, because the factored state is still the old width, so zero-filling new slots is a deliberate act rather than a default. INTRINSIC LR NOT SOLVED. Three designs, three NaNs, three distinct causes: (1) multiplicative lr *= (1+0.02*cos) compounds to 1.02^1200 and ratchets to a bound with no restoring force, pinned at the floor; (2) bounded SNR |EMA(u)|^2/EMA(|u|^2) sits near 1 for momentum of an already normalised update, so it has no dynamic range and pinned at the ceiling; (3) Adadelta-style unit matching collapsed onto the scalar rate is a self-referential fixed point, pinned at the floor. The flag defaults off and the failures are kept in the code.
- `2026-09-10 13:09:50 UTC` **alloc** (10.3 min) ok
  matched params and depth; d traded against mlp_mult
  (log `logs/20260910-125933-alloc.log`, seed 0)
- `2026-09-10 13:11:58 UTC` **Built our own spin carrier and tested it. Three implementation bugs found; allocation hypothesis supported at 45M.**
  src/spin.py is a diagonal-complex linear recurrent carrier written here, not taken from the HF repo. tests/test_spin.py, 17 checks: the chunked parallel scan matches the sequential definition to 1e-14 in double, causality is exact (0.00e+00 change before an edited position), stable at T=8192 with |lambda|<1, an impulse at t=10 still shows 1.03e1 response at t=100, gradients reach nu and theta, and output scale is independent of channel timescale. THREE BUGS, all of which would have been misread as architecture results if the runs had merely been worse instead of NaN: (1) missing gamma=sqrt(1-|lambda|^2) input normalisation, so channels with |lambda| near 1 accumulated the whole sequence: initial loss 301 against a random baseline of 9.0; (2) missing standard LM init, since nn.Embedding defaults to N(0,1) and tied to the output gives logits of std ~sqrt(d): initial loss 515; (3) the trust ratio applied to spectral parameters, which drove |lambda| from 0.999026 to 0.999758 in 30 steps and NaN at step 34, because ||nu_raw|| is about 104 so a 6e-3 trust fraction moves the whole spectrum by 0.6 per step. nu and theta ARE the timescales and must take small absolute steps, not relative ones; a nu floor bounds |lambda|<=0.999 where gamma has a divergent derivative. ALLOCATION TEST, 45M matched params and depth, d traded against mlp_mult, 1500 steps, seed 0: spin d576/mlp4 (carrier 17.8%) 4.8497; spin d312/mlp16 (5.5%) 4.9206; spin d240/mlp27 (3.4%) 4.9430; attention d528/mlp4 5.0803. Loss rises monotonically as carrier share falls, and the previous lineage's own mlp_mult=27 allocation is the worst spin arm. Spin beats attention by 0.2306 nats at matched parameters. CAVEAT: 45M is exactly the scale where the previous work already found spin winning (21% at 37M). This supports the dilution mechanism; it does NOT test whether the win survives to 284M with the allocation held at mlp_mult=4, which is the experiment that has never been run.
