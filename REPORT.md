# Membrae — status against the vision

**Revised 2026-10-26 (IST afternoon).** The base model is **257M parameters** (it widened itself from 168M at step 898,001 by its own trial),
~62B training tokens; the best checkpoint is the general-text branch's end, step 947,000 (`/mnt/kv_cache/mixtest/GENERAL/ckpt.pt`). Every number
comes from a script in `eval/` with its output file named; the day-by-day record, failures and retractions included, is `PROGRESS.md`.
**Verdict in one line:** goal 6 is now met in principle at billions of parameters (inference to 15B and training to 4.9B inside a 4 GB card,
host memory and speed still open); goals 5, 7 and 9 are largely met, 3, 4, 8 and 10 partly, 2 is measured and not met, 1 is not met.
Goal 1, as the user defines it — surpass *any* LLM — is not within reach of this hardware.

**Correction (2026-10-26): every bits-per-byte number reported before today read the model with a different attention kernel than the one it
was trained with, and was ~1.2% pessimistic.** The trained attention is nearly hard (one layer's scores ~30,000; ties decided by the last
bits of the arithmetic), so the exact-fp32 reading path and the training kernel compute measurably different functions
(`eval/swa_check2.txt`, `eval/attn_scores.txt`). Read with the training kernel the 947k model scores **0.9333** on the raw held-out text,
not 0.9446 (`eval/bpb_flexread.txt`). From now on every reading uses the training kernel (`MEMBRAE_SWA_READ=flex`). The fragility itself is
a defect of the recipe (unbounded QK-norm gains) still to fix.

## The model in one paragraph

A language model whose layers are mostly a **carrier** (a gated delta-rule recurrent memory: one fixed-size state per layer, so its
memory does not grow with context), with one layer in four a **sliding-window attention** (2,048 tokens, a reach the model chose in
its own trials). Each attention query also **recalls** its 16 best-matching keys from everything older than the window (at inference
the store can live in host memory). It is trained with **no hand-set learning rate or schedule** (schedule-free Prodigy), **grows
itself** — 46M → 98M → 168M → 257M — when its own time-priced trials say growth pays on held-out text, and **lowers its own step size when a
trial beyond its own measurement noise says so**. Growth stays on in every run. Its weights can live in host memory and stream through
a small card one layer at a time, for inference and for training (`src/stream.py`, `src/stream_train.py`).

## Goal by goal

| # | Goal | Verdict | Evidence |
|---|---|---|---|
| 1 | Surpass the LLMs | **No.** Ahead of GPT-2 and Pythia; behind SmolLM2. | **Raw held-out text** (1,000 documents from five sources, bits/byte, lower is better; `eval/ref_bpb_v1.txt`, `eval/bpb_flexread.txt`): ours **0.9333** (training kernel; 0.9446 the old way), SmolLM2-135M (2T tokens) **0.8797**, GPT-2-124M 1.1023, Pythia-160M 1.2038. Ours is better than SmolLM2 on Wikipedia, behind on math, arXiv and educational text. Benchmarks at 760k (table below) not yet re-measured. |
| 2 | A cognitive model — **measured against humans** | **Not met** | Digit span forward 15+ (humans ~7); free recall U-shaped like humans; 2-back, Wason, syllogisms, CRT, conjunction, false belief **at chance** — instantly, after thinking, or after practice; fine-tunes on practice did not transfer (`eval/cognitive_think_v1.txt`, `eval/sft8_rehearse.txt`). A capacity limit at this size. Teacher-written reasoning text being collected at night (`eval/teacher_gen.py`). |
| 3 | Self-improving | **Yes on lookup; not on general knowledge** | Own labels lift hard lookup 67 → 97%; kinds never written anywhere: no gain. |
| 4 | Knows what it knows | **Partly — moderate** | Agreement among 6 of its own samples AUROC **0.68 / 0.70**; option-softmax 0.599 (SmolLM2 0.597); a hidden-state read-out 0.53–0.66, below SmolLM2's 0.62–0.81 (`eval/know_probe_v1.txt`). |
| 5 | Abstains | **Yes, no labels, no tuned threshold** | Answers only when ≥ half of 6 samples agree: everyday right-when-answering 60.4% → 76.5%, but it declines 65%. |
| 6 | Consumer GPU (4–8 GB): trains AND infers, at any parameter count | **Met in principle at billions; host memory and speed open** | **Inference** (`eval/stream_infer_v2-v5.txt`, 4 GB card cap, CUDA context included): **15.2B at 4 bits in 2.7 GB** (8.5 GB host), 8.6B at 8 bits in 2.5 GB; 8-bit costs 0.05% bits/byte, 4-bit (NF4, chosen by measurement) 3.8%. **Training** (`eval/stream_train_size6.txt`): **4.9B in 3.4 GB** (54 GB host), 2.8B in 2.5 GB (31 GB host), 1B in 2.1 GB (11 GB host); streamed training is the ordinary training — after one update 230 of 231 weights bit-identical (`eval/stream_train_step1_fused.txt`). Caveats: the big models are grown copies of the 257M (capacity, not knowledge); host RAM ~11 bytes/weight; speeds not yet measured on an idle card; the main trainer does not stream yet. |
| 7 | Continuous growth and learning | **Largely** | Grew 46M → 257M by its own trials. Episodes of new facts with self-tested rehearsal keep 83 / 70 / 73 / 90% vs 0 / 23 / 47 / 80% without (`eval/continual3_v2.txt`). General knowledge still erodes slowly. |
| 8 | Everything intrinsic | **Partly** | Step size (and its halving, by trial beyond the model's own noise), growth, attention reach, stopping, rehearsal, abstention are its own. Still hand-set: the data mix (a model-chosen mix degraded the model in October). |
| 9 | Solve context rot | **Largely** | Exact retrieval in real prose 100% from 4k to 1,048,576 tokens; reading comprehension falls 12 points from 0 to 32k. Decoding with the far store in host memory is slow at long context (~140 ms/token at 73k): a per-token index on the card is needed. |
| 10 | Intrinsic thought while idle | **Partly** | Idle consolidation: after a session of new facts, the model rehearses them on its own while idle, stopping when its own flashcards stop improving — new facts 23.3% → **76.7%** (in-context bound 78.3%), old facts held 79.2% (`eval/idle_consolidate_v1.txt`). Free-running idle "thought" does not help yet. |

## Head to head

| Test | Ours | GPT-2 124M | SmolLM2-135M | Pythia-160M (40B / 300B) |
|---|---|---|---|---|
| HellaSwag / PIQA / ARC-E / ARC-C / OBQA (base, 760k, N=500) | 31.8 / 63.2 / 45.8 / 26.0 / 26.6 | 32.1 / 62.7 / 39.8 / 22.8 / 25.6 | 43.5 / 69.1 / 58.8 / 28.7 / 32.4 | 30.4 / 62.4 / 39.4 / 23.0 / 25.6 — 28.8 / 60.3 / 36.4 / 23.0 / 26.4 |
| Facts / arithmetic 4-way / reasoning 4-way (base) | 77.1 / 48.3 / 48.8 | 79.2 / 35.0 / 61.2 | 91.7 / 36.7 / 53.2 | 66.7 / 16.7 / 39.5 — 85.4 / 73.3 / 76.8 |
| Everyday Q&A, 38 questions not in our fine-tune data | 60.5 | 15.8 | 60.5 | — |
| Confidence AUROC, 5,030 four-way items (option softmax) | 0.599 | 0.529 | 0.597 | — |
| Hidden-state "knowing" read-out, AUROC (ARC-E / SciQ) | 0.628 / 0.659 | 0.585 / 0.671 | 0.717 / 0.808 | — |
| Needle in real prose at 32k / 256k / 1M tokens | 100 / 100 / 100 | cannot see past 1,024 | cannot see past 8,192 | cannot see past 2,048 |

Absolute numbers are this suite's protocol, not leaderboard numbers; compare within the table.

## What this week found (details and every slip in PROGRESS.md)

- **The drift past 937k was the repeated generated data:** 10,000 steps with the generated corpora at 32% (seen ~9×) 0.9465, general text
  only 0.9446, start 0.9455 — worse on every source (`eval/mix_test_v1.txt`). General text alone gains only ~0.001 per 10,000 steps: the
  model is near what this recipe gets at 257M.
- **Cosmopedia (36B tokens of synthetic textbooks) hurts the web-text evaluation:** with it at 32% the eval rose 0.9443 → 0.9474 in 8,000 steps
  (`eval/anneal_main3.txt`); stopped by hand.
- **The 16,384-word vocabulary lost** its trial to the 8,192 one (`eval/vocab_trial.txt`).
- **Billions on a 4 GB card** (goal 6, above), and on the way: CUDA local memory reserved by a tuning kernel (+5.8 GB, outside PyTorch's
  accounting) — released after every step, and absent once kernel tunings are cached; recall's buffers that grew with depth.
- **The evaluation correction** (top of this page).

## Retractions and failures still standing

- Every bits/byte reported before 2026-10-26 used the reading kernel, not the training one: ~1.2% pessimistic (corrected above).
- My projection of ~16M teacher tokens a night: the first night wrote ~2.4M (the teacher and our trainer time-sliced the card; they now take turns).
- The card sat idle three times through my failure to queue a successor: ~35 h (Oct 17–19), ~3.7 days (Oct 22–25), 5.4 h (Oct 26).
- Battery v1–v3 labelled-choice numbers (withdrawn 2026-10-13).
- The first intrinsic learning rate (hypergradient) and a model-chosen data mix both degraded the model.
- Self-improvement does not carry to kinds of knowledge the model never saw, nor to general knowledge.

## Next (in order)

1. Main run 4: general text from 947k, read with the training kernel, growth on (its trial is due at 962,799); first, speeds of streamed
   inference and training with the card to itself.
2. Goal 6: the main trainer in streamed mode (growth trials included), leaner host memory or NVMe, decode at long context.
3. The attention fragility: a bounded attention that keeps the lookup heads (a soft-capped layer 3 saturates).
4. Goal 10: train idle thought by how much it improves the prediction of what comes next; goal 4: a stronger one-pass self-agreement.

## Reproduce

- Step-size diagnosis: `eval/sf_z_check.py <ckpt>`; restarted seed `eval/sf_seed.py <ckpt> <out> <d>`; branches `eval/sf_branches.sh`;
  the continuation `eval/anneal_run.py <seed> <d> <end>`
- Goal 4: `eval/calibration.py`, `eval/selfcons.py`, `eval/know_probe.py`; goal 7: `eval/continual2.py`, `eval/continual3.py`,
  `python -m src.sft ... --rehearse <dirs>`; goal 2: `eval/cognitive.py`, `MODES=instant,think,practice eval/cognitive_think.py`
- Equal-data comparison: `BENCH_HF=0 BENCH_EXTRA="EleutherAI/pythia-160m@step19000=label;..." eval/bench_suite.py <ckpt>=label`
- Training in 8 GB: `CTX=2048 NGRP=2 eval/consumer_demo.sh <ckpt> <steps>`; needles: `RECALL=16 NEEDLE_DIST=... eval/recall_test.py <ckpt> needle`
- Raw held-out text: `MEMBRAE_SWA_READ=flex eval/bpb_text.py <ckpt>=label` (`HELD=/mnt/kv_cache/heldout_cosmo.json` for Cosmopedia)
- Goal 6: `SIZES=1B,4B,8B,15B TAIL=0 STATE_HOST=1 eval/stream_infer.py`; `FUSE=1 COMPACT=1 TRITON_CACHE_AUTOTUNING=1 PHASE=size eval/stream_train_test.py`
  (and `PHASE=step1` for the bit-level check against the ordinary trainer)
