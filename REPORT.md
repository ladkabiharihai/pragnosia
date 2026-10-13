# Membrae — status against the vision

**Revised 2026-10-13 (IST afternoon).** The current model is the **many-kinds run's 604k checkpoint** (168M parameters,
~39.6B training tokens) and its final pipeline **v3** (finished 2026-10-13 09:57 IST). Every number comes from a script in
`eval/` with its output file named; the day-by-day record, failures and retractions included, is `PROGRESS.md`.
**Verdict in one line:** goals 5 and 9 are largely met, 3, 4, 6 and 7 partly, 2 is measured and far from human, 1 and 8
are not met. Goal 1, as the user defines it — surpass *any* LLM — is not within reach of this hardware this month.

## The model in one paragraph

A language model whose layers are mostly a **carrier** (a gated delta-rule recurrent memory: one fixed-size state per
layer, so its memory does not grow with context), with one layer in four a **sliding-window attention** (2,048 tokens, a
reach the model chose in its own trials). Each attention query also **recalls** its 16 best-matching keys from everything
older than the window (`src/model.py` `Attn._recall`; at inference the store can live in host memory, `FarStore`). It is
trained with **no hand-set learning rate or schedule** (schedule-free Prodigy) and **grows itself** — 46M → 98M → 168M —
when its own time-priced trials say growth pays on held-out text. Growth stays on in every run.

## Goal by goal

| # | Goal | Verdict | Evidence |
|---|---|---|---|
| 1 | Surpass the LLMs | **No.** Beats a standard transformer at equal data; GPT-2 on most benchmarks; not SmolLM2. | **Equal tokens** (`eval/bench_pythia.txt`, base models, N=1000): at ~40B tokens ours leads Pythia-160M on 7 of 10 (HellaSwag 31.8 vs 30.4, ARC-E 44.1 vs 39.4, ARC-C 24.9 vs 23.0, OBQA 28.4 vs 25.6, LAMBADA 19.1 vs 13.0, facts 77.1 vs 66.7, arithmetic 30.0 vs 16.7), trails on PIQA (62.1 vs 62.4), GSM8K (0.8 vs 2.4) and reasoning (37.2 vs 39.5); it even leads Pythia's final 300B checkpoint on the six standard benchmarks. Caveat: equal tokens, not equal data (the Pile vs our curated web text plus targeted synthetic data). **Against GPT-2 / SmolLM2** (self-improved v3): ahead of GPT-2 on 7 of 10 benchmarks, behind SmolLM2 by 9–17 points on the knowledge-heavy ones (HellaSwag 32.6 vs 43.5, ARC-E 44.8 vs 58.8). Everyday Q&A 58.3 vs 16.7 / 62.5. **Long memory:** exact retrieval to a million tokens, which neither baseline can see. |
| 2 | A cognitive model — **measured against humans** (user, 2026-10-13) | **Far from human; a clear profile** | `eval/cognitive.py`, fresh variants of classic tasks, human figures from the cited papers (v1, base / fine-tuned): digit span forward **15+** (humans ~7), backward **<3** (~5); free recall first-middle-last 62-51-48 / 98-77-77 (humans U-shaped: primacy and recency — ours has primacy, no recency); 2-back at chance (~90); CRT right 2% / 0% with the **intuitive error 55% / 48%** (humans ~41% right, errors mostly intuitive); false belief 35% / 40% (adults ~100; below chance = it names where the object *is*, the young child's error). Storage superhuman, manipulation absent, reasoning at chance. v1 had four test flaws of mine (Wason, two-option and yes/no scoring, span rule); v2 with them fixed is running. |
| 3 | Self-improving | **Yes on lookup, including kinds it learned elsewhere; no gain on kinds it never saw** | It labels its own questions (answer agreed by ≥2 of 3 orderings; checked against gold only afterwards) and trains on them. v2 (two fact kinds): 67 → **97%** on the hard lookup test. v3 (381 kinds, labels 89–97% right): capitals / engines — kinds absent from its labels — 86.4 → **97.0%** at 120 facts; two kinds never written anywhere (rivers, guilds): 57–60% → 47–58%, no gain. Not shown for general knowledge. |
| 4 | Knows what it knows | **Partly; one result to confirm** | Calibration on 48 facts: separation **+0.225 vs GPT-2's +0.186** (v3) — better, but on a small set, to be re-measured on thousands of questions. Its samples agree more when it is right: 0.51 vs 0.17 (everyday). |
| 5 | Abstains on what it does not know | **Yes, no labels, no tuned threshold** | Answers only when ≥ half of 6 samples agree (`eval/selfcons.py`): everyday right-when-answering **58.3% → 90.0%** (declines 58%), SciQ 9.8% → 25.7%. It declines too often; that is the open part. |
| 6 | Consumer GPU (4–8 GB): trains AND infers | **Inference: yes. Training: yes on 8 GB at 168M; 4 GB and "any size" not yet** | **Training under an 8 GB card's memory** (`eval/consumer_demo.sh`, cap 7.5 GiB, the real recipe resumed with its optimizer state): window 2,048 x 32 → **6.37 GB peak for the whole process** in plain training, window 4,096 x 16 → 7.15 GB (the 8,192 window needs ~9.6); **with the growth trial's held-out measurements on, the process reached 8.6 GB** (PyTorch filled a 7.5 GiB cap, plus ~0.9 GB of CUDA context) — being re-measured at a 6.9 GiB cap. The cap bounds memory, not speed. **Inference:** a million tokens of context with retrieval in 2.96 GiB allocated (4.10 GB process with its CUDA context — just over 4 GB) with the recall store in host memory. Open: a growth trial under the cap (running), training on 4 GB, and "any parameter count" (weights streamed from host memory). |
| 7 | Continuous growth and learning | **Growth: yes. A skill into the weights without forgetting: yes (with replay). Facts into the weights: not shown** | Grew 46M → 98M → 168M on its own trials; training continues (next trial at 649k). Lookup over 381 kinds fine-tuned **with 1:1 replay of general text: bits/byte 1.0143 → 1.0171 (+0.3%)** vs **+7.8%** without replay, the gain kept. Three self-improvement rounds moved no benchmark beyond noise. The old result — 120 new facts into the weights forget old ones — has not been retried with this replay. |
| 8 | Everything intrinsic | **No (mostly intrinsic training; hand-set data)** | No learning rate or schedule anywhere; size, reach and growth timing by the model's own trials; decoder repetition strength matched to real text; abstention by self-agreement. **Still hand-set:** every data mix (pretraining phases, fine-tune mixtures, the 1:1 replay ratio). A model-chosen mix failed three ways (2026-10-06/07). |
| 9 | Solve context rot | **Largely** | **Exact retrieval** in real prose: **100% at every distance from 4k to 1,048,576 tokens** (590k). **Real-text open-book** SciQ (v3): closed book 57.3, paragraph 0 / 2k / 8k / 32k back **87.3 / 84.7 / 81.3 / 75.3**. **Reading invented facts** 100 / 91 / 87 at 0 / 8k / 32k. **Prediction:** bits/byte 1.118 at 512 tokens → 1.034 at 8k. Limits: real-text reading still falls 12 points from 0 to 32k; lookup on kinds of fact never trained on reaches ~60%, not ~95%. |

## Head to head

| Test | Ours | GPT-2 124M | SmolLM2-135M | Pythia-160M (40B / 300B) |
|---|---|---|---|---|
| HellaSwag / PIQA / ARC-E / ARC-C / OBQA (base, zero-shot) | 31.8 / 62.1 / 44.1 / 24.9 / 28.4 | 32.1 / 62.7 / 39.8 / 22.8 / 25.6 | 43.5 / 69.1 / 58.8 / 28.7 / 32.4 | 30.4 / 62.4 / 39.4 / 23.0 / 25.6 — 28.8 / 60.3 / 36.4 / 23.0 / 26.4 |
| LAMBADA / GSM8K | 19.1 / 0.8 | 14.1 / 0.8 | 23.3 / 0.8 | 13.0 / 2.4 — 8.0 / 0.4 |
| Facts / arithmetic / reasoning (4-way) | 77.1 / 30.0 / 37.2 | 79.2 / 35.0 / 61.2 | 91.7 / 36.7 / 53.2 | 66.7 / 16.7 / 39.5 — 85.4 / 73.3 / 76.8 |
| Everyday Q&A / ARC (fine-tuned v3, greedy) | 58.3 / 43.7 | 16.7 / 33.8 | 62.5 / 48.4 | — |
| Needle in real prose at 32k / 256k / 1M tokens (590k) | 100 / 100 / 100 | cannot see past 1,024 | cannot see past 8,192 | cannot see past 2,048 |
| Open-book SciQ, paragraph 32k tokens back | 75.3 (closed 57.3) | 42.7 | 72.0 (closed 72.7) | — |

Absolute numbers are this suite's protocol, not leaderboard (lm-evaluation-harness) numbers; compare within the table.

## What this week found and fixed (details and every slip in PROGRESS.md)

- **Lookup was learned as six templates, not as a skill.** On two kinds of fact no data ever contained, the base and every
  self-improved model were at chance, even with 10 facts in plain view. Pretraining with 381 kinds at 12% of tokens did not
  move it in 8,000 steps (the answers are ~2% of a document's loss); a fine-tune with the loss on answers over 381 kinds did:
  **chance → ~60%** on the never-seen kinds.
- **Fine-tunes forget unless general text is replayed.** The lookup fine-tune alone: +7.8% bits/byte and chat answers
  broken ("capital of Japan?" → "96"); interleaved 1:1 with general text: +0.3%, gain kept. One of two identical replay
  runs diverged to NaN (cause not found); every fine-tuned checkpoint is now checked for non-finite weights.
- **The old fine-tune data erased base skills.** Plain arithmetic 8/8 → 3/8 and digit copying 90% → 10–30% after the
  factual fine-tune (Alpaca / Dolly / OpenAssistant / CAMEL, no replay). New data fetched and being built
  (`eval/get_sft_data.py`, `eval/build_sft2.py`: chat, arithmetic, coding, reasoning traces, human-choice data; decontaminated
  against every evaluation set and the goal-2 battery), to be trained with replay.
- **Training in 8 GB:** a shorter window alone rebuilt the position table at the window size and the optimizer state no
  longer loaded; `--pos-table 8192` keeps it.
- **Token count corrected:** the current model has seen ~39.6B tokens (604,000 x 65,536), not ~25B (that was the 382k base).

## Retractions and failures still standing

- The first intrinsic learning rate (hypergradient) beat the tuned schedule over 6,000 steps and collapsed over 120,000.
- The data mix chosen by the model: three rules tried, the last degraded the model.
- Self-improvement does not yet carry to kinds of knowledge the model never saw, nor to general knowledge.
- Real-text reading past ~32k and reasoning of any kind (goal 2) are weak.

## Next (in order)

1. Fine-tune on the new data **with replay**; check that arithmetic and copying survive and Q&A / reasoning improve.
2. Battery v2 against humans; the growth trial under the 8 GB cap; training on 4 GB; then weights streamed from host memory.
3. Training data for the faculties the battery found missing (manipulation, reasoning, perspective), the battery's own task
   types held out; calibration on thousands of questions; facts into the weights with replay.
4. Instruction data folded into the continuous training, so no separate fine-tune stage is needed (goals 7, 8).
5. Proposed: a vocabulary that grows as a model-chosen growth option (unfamiliar scripts and symbols cost 3–4x the tokens today).

## Reproduce

- Pipeline: `LOOK=1 LOOKCK=<lookup ckpt> SI_KINDS=many PCAP=8 OPENBOOK_HF=0 eval/final_pipeline2.sh <base ckpt> <tag>` →
  `eval/final_pipeline_<tag>.txt`, `eval/si_rounds_<tag>.txt`, `eval/scorecard_final_<tag>.txt`
- Equal-data comparison: `BENCH_HF=0 BENCH_EXTRA="EleutherAI/pythia-160m@step19000=label;..." eval/bench_suite.py <ckpt>=label`
- Never-seen kinds: `INCONTEXT_KINDS=new eval/incontext.py <ckpt> powered`; lookup over many kinds and its replay test:
  `eval/build_recall.py` (`RECALL_KINDS=many`), `eval/kinds.py`, `eval/look_forget.sh`, `eval/lookgold_test.sh`
- Training in 8 GB: `CTX=2048 NGRP=2 eval/consumer_demo.sh <ckpt> <steps>` (cap 7.5 GiB)
- Human battery: `eval/cognitive.py <ckpt>=label`
- Needles: `RECALL=16 NEEDLE_DIST=... eval/recall_test.py <ckpt> needle` (`RECALL_OFFLOAD=1`: store in host memory)
