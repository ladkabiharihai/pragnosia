# Membrae — status against the vision

**Revised 2026-10-03.** The current model is **run82** (97.7M; `logs/train82.out`, `eval/scorecard_r82.txt`). Every
number comes from a script in `eval/` and a log in `logs/`; the day-by-day record, failures and retractions included, is
`PROGRESS.md`.

**Reproduce everything:** `eval/scorecard.sh <base ckpt> <fine-tuned ckpt> <abstaining ckpt>` writes `eval/scorecard.txt`
(each section is a separate script; they run one at a time, because jobs sharing the card measure contention).

## The model in one paragraph

A language model whose layers are mostly a **carrier** (a gated delta-rule recurrent memory, one fixed-size state per
layer, the same operator as flash-linear-attention's `chunk_gated_delta_rule`), with one layer in four a **sliding-window
attention**. Its memory per layer does not grow with context. It is trained with **no hand-set learning rate or
schedule** (schedule-free Prodigy, fused into a CUDA-graphed optimizer step), and it **decides its own size and reach**:
at each doubling of its current stage it trials growing — wider, deeper, or a longer attention window — against staying,
on the same batches, judged on long held-out streams read the way the model is used (window by window, state carried),
and keeps an option only if it turns compute into held-out loss faster.

## Goal by goal

| # | Goal | Verdict | Evidence (run82 unless stated) |
|---|---|---|---|
| 9 | Solve context rot | **Prediction: won — flat from 8k to 1M tokens. Exact retrieval: within the attention window (~1k).** | Bits per byte fall as context grows (1.159 at 512 → 1.089 at 8,192; GPT-2 1.217 / 1.225 and cannot pass 1,024), and then stay flat: the same passages after 32k vs 8k of context −0.0006 ± 0.0010 (24 passages), 131k the same, one million tokens 1.1424 → 1.1426 in 2.89 GiB. Exact retrieval reaches as far as the model's chosen attention window: needles 100% at 256 tokens back, 17% at 1,024. With an 8,192 window (run80) retrieval reached ~7k (needles 96% at 4,096) but prediction got worse past 8k (+0.034): the model's own trials, judged on long streams, chose 1,024. In-context lookup with every distractor from the same context: 60 / 59 / 59% at 10 / 30 / 60 facts vs GPT-2 40 / 37 / 37%. |
| 6 | Consumer GPU / inference within 4 GB | **Won** | Decode 404 tok/s at 1.15 GiB vs GPT-2's 248 tok/s at 1.20 GiB; one million tokens of context in 2.89 GiB. |
| 8 | Everything intrinsic | **Achieved for rate, size and reach** | Rate: schedule-free Prodigy at published defaults (beat a tuned cosine schedule over 30,000 steps; the no-rate fine-tune matched a hand-tuned recipe). Size: grew 46M → 98M on its own trial at ~10B tokens. Reach: chose a 1,024-token attention window over six trials of its own (one grow, five stays). Not yet intrinsic: data-mix fractions. |
| 7 | Continuous growth and learning | **Growth: works and pays. Learning: learns fast, still forgets some.** | Each growth pays on held-out data (46M stalled at 1.137; 98M reached 1.114). Continual learning with no rate: new facts 25% → 93%, old facts 75% → 65%. |
| 3 | Self-improving | **Demonstrated on in-context lookup (one round); a second round added nothing** | The model labels invented lookup questions itself (the answer at least 2 of 3 orderings of the facts agree on; 79% right, checked against gold only afterwards), trains on its own labels, and improves on a held-out test with other names and wording: 54.8 / 53.8 / 58.2% → **72.0 / 77.8 / 73.0%** at 10 / 30 / 60 facts (± 4), while plain-text training from the same start stays at 54.8 / 53.6 / 59.0%. A second round (the improved model's labels, 85% right) gave no gain over plain continued training: the loop does not compound on this task. Factual Q&A remains below the gate (labels mostly wrong). |
| 1 | Surpass the LLMs | **Ahead of GPT-2 124M on nearly every axis; behind SmolLM2-135M on knowledge** | Ahead of GPT-2: context rot and length, inference speed and memory, lookup, everyday Q&A 25.0 vs 16.7 (29.2 after abstention training), SciQ 6.8 vs 2.2, ARC 38.7 vs 33.8, calibration +0.214 vs +0.186. Level: short generation quality (matches real text's statistics). Behind: facts 27 vs 33 / 48, induction 97.0 vs 98.5, long generation (loops after ~4k tokens; run80 stayed coherent to 24k). SmolLM2 (2T training tokens to our ~16B) leads Q&A: 62.5 / 12.7 / 48.4. |
| 4-5 | Knows what it knows / abstains | **Partial** | Calibration ahead of GPT-2 (+0.214 vs +0.186). Trained on its own mistakes: everyday right 25.0 → 29.2% with wrong 75 → 69%; SciQ wrong 93 → 50% with 47% declined, but SciQ right also falls 6.8 → 2.6%. |
| 2 | A cognitive model | Profile of the measures above | Memory (9), learning without forgetting (7), metacognition (4-5), self-improvement (3), self-organised growth, rate and reach (7-8). |

## How the model decides its own size and reach

At every doubling of the current stage's length it pauses and **trials** each growth option — widen by one step, double
the depth with new layers whose outputs start at zero, or double the attention window — against **not growing**: each
branch trains the same number of steps from the same snapshot on the same batches. Since 2026-10-03 the branches are
scored on eight 32k-token streams from the training mix, read window by window with every state carried (how the model
is used), so an option that helps inside a window but hurts past it pays for both. Validated: it says STAY before the
point where a bigger model is compute-optimal (checked against a 118M control) and GROW after it; judged on 8k windows it
grew its attention to 8,192 and paid for it past 8k, judged on long streams it stopped at 1,024 and is flat to 1M.

## Retractions and failures (all in PROGRESS.md)

- The first intrinsic learning rate (hypergradient) beat the tuned schedule over 6,000 steps and **collapsed** over
  120,000: withdrawn.
- "More context past 8k makes prediction worse" for the block-local models (+0.053) was a **bug**: attention output zero
  for 7,680 of 8,192 tokens of every window after the first. Fixed and re-measured flat to 131k.
- run80's first scorecard measured generation and Q&A through a broken decoder (padding attended as keys); re-measured.
- The in-context lookup test was **underpowered** (10 questions per row). Replaced by 500 questions per row with 95%
  intervals, plus the standard passkey test.
- The windowed attention computed **three different functions** — block-local in training, sliding at decode, full at
  prefill of most lengths. Fixed with one sliding window everywhere.
- The skip guard and the schedule-free optimizer interacted badly: a "skipped" step still moved the weights; fixed.
- Carried-state training (training window by window with the state carried) diverged twice and, when stable, improved
  long documents at a cost in general text; not used in the current model.
- Several hypotheses tested and refuted on 2026-10-01/02 (the fine-tune data, the pretraining mix, the fine-tune optimizer
  as causes of the Q&A gap; a short convolution added late was inconclusive).

## Remaining work before the deadline (2026-10-14)

1. run83: run82 trained ~6.5B tokens longer at 98M for facts and knowledge (running).
2. Long generation past ~8k tokens (stochastic loops); retrieval past the attention window (one long layer and a far-needle curriculum both failed).
3. Abstention that keeps right answers.
