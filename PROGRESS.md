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
- `2026-09-10 13:15:50 UTC` **Correction to my own dilution claim.** It explains the 1B lineage, not the 284M tie.
  I argued the spin advantage faded because the carrier was diluted. Checking the numbers against the claim: the 284M crux, where the TIE was measured, had carrier share 14.9%, which is close to the 17.8% of the arm that WON in our 45M allocation test. So dilution does not explain that tie. It explains the 508M/706M/1022M lineage (10.2%, 7.4%, 6.3%) but not the crux comparison. The honest reading of the combined evidence is that our 45M result (spin ahead by 0.231 nats, 21% ppl) REPRODUCES their 37M result (21%), and their fade from 37M to 284M at a reasonable carrier share stands as a genuine scale effect rather than an allocation artifact. Running 284M at mlp_mult=4 would therefore most likely reproduce the tie, and is not the highest-value use of GPU.
- `2026-09-10 13:44:52 UTC` **calib** (0.1 min) ok
  is uncertainty ABSENT from the representation or merely UNEXPRESSED? /mnt/kv_cache/run1/probe_snap.pt
  (log `logs/20260910-134445-calib.log`, seed 0)
- `2026-09-10 13:46:01 UTC` **calib** (0.3 min) ok
  is uncertainty ABSENT from the representation or merely UNEXPRESSED? /mnt/kv_cache/run1/probe_snap.pt
  (log `logs/20260910-134543-calib.log`, seed 0)
- `2026-09-10 13:46:59 UTC` **calib** (0.5 min) ok
  is uncertainty ABSENT from the representation or merely UNEXPRESSED? /mnt/kv_cache/run1/probe_snap.pt
  (log `logs/20260910-134627-calib.log`, seed 0)
- `2026-09-10 13:47:49 UTC` **goal5** (0.1 min) ok
  is confidence higher on fabricated subjects than real ones, as on every previous model?
  (log `logs/20260910-134745-goal5.log`, seed 0)
- `2026-09-10 13:48:23 UTC` **First checkpoint probed. Next-token calibration is fine; the goal-5 test is not yet answerable.**
  Checkpoint at step 10000 of 120000, 77.6M params, held-out BPB 1.7476. INCREMENTAL CALIBRATION TEST, after three methodological corrections: (1) a probe on the final hidden state lost to the output confidence 0.697 vs 0.807, which answers nothing, because the output head IS a trained linear map on that same state and a logistic regression on ten thousand examples cannot be expected to beat it; (2) the incremental version, confidence+entropy versus confidence+entropy+hidden, went NEGATIVE at every layer (-0.045 to -0.055), which is overfitting: 770 features, ten thousand examples, no penalty; (3) with ridge regularisation and the penalty chosen on a validation split, reported on a disjoint test split over 61k positions, the incremental gain is -0.012 to -0.018, i.e. zero. CONCLUSION for next-token prediction: output confidence predicts correctness at AUROC 0.805 and the representation adds nothing. That is the EASY task and not where the failure lives. GOAL-5 TEST (confidence on real versus fabricated subjects, matched candidate sets): real 0.5462, fabricated 0.5378, AUROC 0.535, NOT inverted, unlike every previous model. But 0.535 is barely above chance and the model is at 1.3 tok/param, so the honest reading is that it knows nothing about either and the test is not yet answerable. The inversion on previous models appeared once they HAD knowledge, so this must be tracked as training proceeds rather than declared now.
- `2026-09-10 13:52:53 UTC` **probe-suite** (0.0 min) ok
  1 checkpoints: BPB, goal-5 calibration, goal-9 retrieval
  (log `logs/20260910-135252-probe-suite.log`, seed 0)
- `2026-09-10 13:53:09 UTC` **probe-suite** (0.1 min) ok
  1 checkpoints: BPB, goal-5 calibration, goal-9 retrieval
  (log `logs/20260910-135302-probe-suite.log`, seed 0)
- `2026-09-10 13:53:35 UTC` **train** (4.2 min) ok
  d768 L12 mlp4 ctx512 B16 120000 steps -> /mnt/kv_cache/run1
  (log `logs/20260910-134920-train.log`, seed 0)
- `2026-09-10 13:53:36 UTC` **Probe suite built: BPB, goal-5 calibration and goal-9 retrieval measured across the training trajectory.**
  Checkpoints are now preserved rather than overwritten, so the two questions that can only be answered over TIME can be asked. (1) Does calibration invert as knowledge accumulates? Every previous model in this project ended inverted, more confident about fabricated subjects than real ones, and none was measured on the way there. (2) Does retrieval reach past the training window, or track it? Both previous architectures lost retrieval at roughly their training window: spin at 256 trained on 256, the linear-recurrent core near 2000 trained on 2048. This model trains at 512 and its carrier holds per-channel timescales whose slowest half-life is about 693 tokens, so the question is real. Baseline at step 10000 (1.3 tok/param): BPB 1.7548, confidence real 0.5462 vs fabricated 0.5378 (not inverted), needle lift +0.244 at d=0 and 0.000 from d=256 outward. For scale, the previous 125M had +6.020 at d=0, so this model barely retrieves in context yet, which is what 1.3 tok/param looks like.
- `2026-09-11 06:09:03 UTC` **train** (0.4 min) ok
  d768 L12 mlp4 ctx512 B16 120000 steps -> /mnt/kv_cache/run1
  (log `logs/20260911-060838-train.log`, seed 0)
- `2026-09-11 06:20:48 UTC` **The NaN was in my scan, and the correct fix is also faster.** Clean run relaunched.
  Reproduced and located rather than guessed: the chunked scan computed lambda^(-s)=exp(nu*s), nu is learned with no upper bound, training pushed nu_max 0.345->0.382 in a thousand steps taking lambda^-128 from 1.5e19 to 1.7e21, and bf16 cannot hold a cumsum spanning twenty orders of magnitude. NaN at ~step 14,500 twice on different data. Capping nu would be a hand-set bound on what the model may learn (goal 8), so the inverse was removed instead: the recurrence is a causal convolution with kernel lambda^t whose every entry has magnitude <=1, evaluated by FFT. 21/21 tests: 7.5e-15 vs the definition in fp64, 5.6e-6 in fp32, finite and correct at nu=3.05 where the old scan would compute 1e169. And faster: 247 ktok/s compiled vs 154. Then a second NaN at step 2: torch.compile produces non-finite gradients through complex FFT backward while forward is identical and eager is correct. The scan is excluded from compile, the rest of the model still compiles, speed unchanged. Also fixed from the user's catch: --resume now restores optimizer state; it previously loaded weights only. Process failures logged: a buffered diagnostic lost with the session, and pkill -f self-matching my own shell for the fourth time. Clean run from step 0 at 06:12 UTC, 77.6M, 120k steps, ~105 min.
- `2026-09-11 06:49:52 UTC` **probe-suite** (0.4 min) ok
  6 checkpoints: BPB, goal-5 calibration, goal-9 retrieval
  (log `logs/20260911-064929-probe-suite.log`, seed 0)
- `2026-09-11 06:51:09 UTC` **needle-carried** (0.1 min) ok
  retrieval THROUGH the carried state, past the 512 window: /mnt/kv_cache/run1/snaps/step_030000.pt
  (log `logs/20260911-065104-needle-carried.log`, seed 0)
- `2026-09-11 06:51:43 UTC` **First trajectory: calibration trending the right way, retrieval decays fast, and a bug in my needle test produced clean-looking zeros.**
  Clean run passed step 32,800 with 10 spikes absorbed; the FFT scan holds where both previous runs died. Six checkpoints probed. CALIBRATION (goal 5): confidence on real subjects exceeds fabricated at 5 of 6 checkpoints and the gap is WIDENING, 0.019, -0.002, 0.013, 0.030, 0.019, 0.035 from step 5k to 30k. Every previous model in this project ended inverted; this one is moving the other way as it learns. Single seed, n=16 pairs, gap 0.035: a trend, not a result. RETRIEVAL (goal 9): the first needle test truncated the input to the last ctx tokens, so at d>=495 the FACT was cut out and the exactly-0.000 lifts meant absent, not forgotten. Fixed: LM.forward_carried threads the carrier state across windows (verified equal to the direct path within one window, max diff 0.00e+00), and the test errors rather than silently zeroing. Honest profile at step 30k: +0.360 at d=0, +0.091 at 64, +0.036 at 128, +0.009 at 256, 0.000 from 400 outward including carried distances to 2048. Half-life about 50 tokens. The carrier slowest channel has a 693-token half-life, so the capacity exists and the model has not learned to route retrieval through it; whether it does is what the rest of the trajectory will show. For scale, d=0 is 0.36 here against 6.0 on the finished 125M at 73 tok/param.
- `2026-09-11 07:23:34 UTC` **probe-suite** (1.0 min) ok
  14 checkpoints: BPB, goal-5 calibration, goal-9 retrieval
  (log `logs/20260911-072237-probe-suite.log`, seed 0)
- `2026-09-11 07:24:08 UTC` **Fourteen checkpoints to step 70k. Correction: the calibration gap is not widening.**
  BPB 1.8343 (5k) -> 1.4051 (70k), steady, LR decaying, no spikes since the first ten. CALIBRATION: I previously reported the real-vs-fabricated confidence gap as widening. That held through 30k (0.019 -> 0.035) and does not hold after: 0.014, 0.022, 0.051, 0.014, 0.000, 0.026 from 35k to 70k. Inverted at 3 of 14 checkpoints (10k, 35k, 50k), each by under 0.012. Honest reading: not inverted on average, gap about 0.02, noisy, no trend. Still the opposite of every previous model, which all ended inverted, but the claim of improvement with training is withdrawn. Absolute confidence on both classes is falling, real 0.606->0.513, fabricated 0.587->0.487. RETRIEVAL, now measured through the carried state past the window: d=0 grew 0.080 -> 0.612 (peak 60k) -> 0.376; d=256 grew about 3x, 0.014 -> 0.040 (65k); d=512 is small but nonzero at 0.005-0.010, the first evidence the slow channels carry anything. d=1024 remains noise.
- `2026-09-11 07:25:05 UTC` **vs-gpt2** (0.1 min) ok
  same bytes, same 512 context: step_070000.pt
  (log `logs/20260911-072458-vs-gpt2.log`, seed 0)
- `2026-09-11 08:08:28 UTC` **train** (109.6 min) ok
  d768 L12 mlp4 ctx512 B16 120000 steps -> /mnt/kv_cache/run1
  (log `logs/20260911-061853-train.log`, seed 0)
- `2026-09-11 08:09:26 UTC` Spin run complete: 120,000 steps, 0.98B tokens, BPB 1.2984, 57 spike-skips, no NaN.
  77.6M spin (d768 L12), 109.6 min clean, final loss 3.1244 (ppl 22.7). The FFT scan held for the whole run where both earlier runs died at ~14,500. Guard fired 57 times in 120k steps (0.05%). Matched attention baseline launched at 08:08 UTC (d704, 77.6M, identical data, schedule, optimiser and seed) -> /mnt/kv_cache/run2, with a snapshotter every 5k steps so the two trajectories can be compared checkpoint by checkpoint. Probe suite over the 23 spin checkpoints and the GPT-2 anchor on the final are running co-scheduled under an 8 GiB cap; process failure noted: pgrep -f matched my own shell for the fifth time, PIDs are now taken by executable name (ps comm==python) and verified before being recorded.
- `2026-09-11 08:10:52 UTC` **probe-suite** (1.5 min) ok
  /mnt/kv_cache/run1: 24 checkpoints: BPB, goal-5 calibration, goal-9 retrieval
  (log `logs/20260911-080920-probe-suite.log`, seed 0)
- `2026-09-11 08:11:01 UTC` **vs-gpt2** (0.1 min) ok
  same bytes, same 512 context: ckpt.pt
  (log `logs/20260911-081053-vs-gpt2.log`, seed 0)
- `2026-09-11 08:16:54 UTC` Spin trajectory over 24 checkpoints: calibration gap positive and growing, retrieval reaches 256 and not much further; anchor 0.097 BPB behind GPT-2 at 1/9 of its tokens.
  BPB 1.8343 (5k) -> 1.3105 (120k) on the probe sample, 1.2984 on the trainer's larger sample; GPT-2 124M on the same bytes at the same 512 context: 1.2230, so 0.097 behind with 0.98B tokens against its ~9B and 77.6M params against 124M. GOAL 5: confidence on real subjects exceeds fabricated at 21 of 24 checkpoints (inverted at 10k, 35k, 50k by <0.012). The gap is small (mean +0.022, final +0.056) and its slope over training is positive with a bootstrap 95% CI that excludes zero, [+5e-5, +3.9e-4] per 1k steps; first-half mean +0.017, second-half +0.026. Earlier I withdrew a 'widening' claim made on 6 points; on 24 the trend is there, and still on n=48 facts, one seed. GOAL 9: needle lift at d=0 0.08 -> 0.59, at d=256 0.015 -> 0.056 (monotone, second-half mean 2.7x first-half), at d=512 through the carried state 0.002 -> 0.007, at d=1024 ~0. Retrieval extends with training but its half-life is still well under the 512 window; the slow channels (693-token half-life) are not yet carrying facts. ENGINEERING: the scan's convolution is now a custom autograd Function with a hand-derived backward (two correlations by the same FFT), checked against autograd in fp64 to 7e-15, gradcheck at edge shapes, and the whole carrier's spectral gradients against the sequential definition to 2e-15. That removes the compile exclusion: full model compiles, compiled gradient equals eager to 6e-6 in fp32 (the earlier NaN failure mode is gone; bf16 differs by up to 5% on a scalar gate, which is rounding, not error). Speed under the same contention as the earlier table: T=512 68.3 -> 55.5 ms (1.76x -> 1.50x attention), T=8192 95.0 -> 68.2 ms (parity, 0.99x); memory 4.83 -> 4.19 GiB at 512, 6.69 -> 5.22 at 8192. Remaining growth is the per-layer T x D kernel intermediates saved for backward, ~1 GiB at 8k; fixable by recomputing the kernel in backward. 32/32 spin tests. OBSERVATION: the trainer runs spin at 150 ktok/s and attention at 154, near identical, while the model step differs 1.5x, so the trainer step is dominated by something other than the model (optimizer or data); profile when the card is clear. Attention baseline at 4,200/120,000, eta ~100 min.
- `2026-09-11 08:29:20 UTC` **train** (0.3 min) ok
  d256 L4 mlp4 ctx512 B8 400 steps -> /tmp/claude-1000/-opt-code-membrae/ab0dd831-724a-4bd0-be9e-dd7e1ea6f491/scratchpad/smoke
  (log `logs/20260911-082900-train.log`, seed 0)
- `2026-09-11 08:29:38 UTC` **train** (0.2 min) ok
  d256 L4 mlp4 ctx512 B8 600 steps -> /tmp/claude-1000/-opt-code-membrae/ab0dd831-724a-4bd0-be9e-dd7e1ea6f491/scratchpad/smoke
  (log `logs/20260911-082928-train.log`, seed 0)
- `2026-09-11 08:30:18 UTC` Trainer step decomposed: the optimizer was 20 ms of CPU per step. Fused and graph-replayed now; and the clean-card speed gap is 2.67x, not the 1.76x I reported.
  Instantaneous rates from the two runs' own logs, undisturbed stretches: spin 52.0 ms/step (158 ktok/s), attention 19.5 ms/step (420 ktok/s). My earlier speed table was measured while a trainer shared the card, which added a near-constant overhead to both and compressed the ratio to 1.76x; the real gap at T=512 was 2.67x. Correcting the record. Profiled a step (still under contention): fwd+bwd 55 ms, optimizer 22 ms of which 19.5 ms is CPU issuing ~4,500 small kernels for 160 tensors, data 0.2 ms. A CUDA-graph replay of the eager step was bit-identical and 16 ms, so the GPU work itself was ~28 unfused elementwise kernels per tensor. Fabric now has one _update function as the single source of truth for the math, run eager or through torch.compile(dynamic=True) per tensor (fused=True), and optionally captured once as a CUDA graph and replayed (cuda_graph=True), with lr as a device scalar filled outside the capture (a fill_ inside it baked the capture-time rate; caught by the test), addresses checked before every replay and recaptured if any moved, and re-warm-up after load_state_dict or grew so no JIT happens inside a capture. Tests: fused == eager to 8.8e-5 with bf16 momentum and 1e-5 without; graph == fused bit for bit; moved grad address recaptured and exact; reload exact. 21/21. Trainer-shaped loop under contention: 83.9 -> 74.3 ms/step; clean numbers when the card frees. Found on the way: torch's load_state_dict casts state to the parameter dtype, so the bf16 momentum became fp32 on every resume (2x state, different rounding); fixed and tested. run1 never resumed, so it is unaffected. Trainer defaults --fused 1 --graph 1, zeroing grads in place; smoke run 400 steps + resume 200 clean, 0 recaptures.
- `2026-09-11 08:35:23 UTC` CORRECTION: the 2.67x speed gap I recorded an hour ago was measured against a frozen model. Clean-card numbers: fwd+bwd 1.39x, full step 1.21x. The attention control had diverged; relaunched with QK-norm.
  The attention run's 19.5 ms/step from step 15,400 on was a model whose every step the spike guard was skipping: it diverged at ~11,000 (gradient-norm median 1.15 -> 5.0, single spikes to 1,442), BPB froze at 1.8251 for ten consecutive evals, and 27,800 of the next 27,800 steps were skipped while the log printed a healthy rate. I read the rate and did not read the skips. Killed at 43k by verified PID. Fixed two things: the trainer now stops when 250 of the last 500 steps are skipped and says DIVERGED, and the attention control has QK-norm (RMS-normalised q and k per head, +2k params), the standard stabiliser, because the control must be the strongest attention I can field. Relaunched at the same lr 6e-3 with the fused, graphed optimizer -> /mnt/kv_cache/run2. CLEAN-CARD SPEED (prod idle, 0% util, nothing else running), both 77.6M, 8192 tokens/step: fwd+bwd spin 26.6 ms vs attention 19.2 at T=512 (1.39x), 28.5 vs 23.0 at 2048 (1.24x), 32.2 vs 34.4 at 8192 (0.94x, spin ahead). Full trainer step at T=512: spin 48.0 ms eager optimizer -> 39.4 fused -> 34.5 graph (171 -> 238 ktok/s, +39%); attention 51.7 -> 38.2 -> 28.4 (158 -> 289 ktok/s, +83%). So with the new optimizer path spin trains 1.21x slower than attention at 512 tokens and faster beyond ~6k. The spin run1 was trained at 158 ktok/s; the same model now trains at 238.
