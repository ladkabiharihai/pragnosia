# Membrae — status against the vision

**Draft of 2026-09-30.** Numbers marked **[run70]** will be filled from the long intrinsic run now training
(`logs/train_grow70.out`). Every number below comes from a script in `eval/` and a log in `logs/`; the day-by-day
record, failures and retractions included, is `PROGRESS.md`.

**Reproduce everything:** `eval/scorecard.sh <base ckpt> <fine-tuned ckpt> <abstaining ckpt>` writes `eval/scorecard.txt`
(each section is a separate script; they run one at a time, because jobs sharing the card measure contention).

## The model in one paragraph

A language model whose layers are mostly a **carrier** (a gated delta-rule recurrent memory, one fixed-size state per
layer, mathematically the same operator as flash-linear-attention's `chunk_gated_delta_rule`), with one layer in four a
sliding-window attention (512 tokens). Its memory does not grow with context. It is trained with **no hand-set learning
rate or schedule** (schedule-free Prodigy, fused into our CUDA-graphed optimizer), and it **decides its own size**: it
starts small and grows (wider or deeper, exactly preserving what it has learned) only when a trial on its own data
shows that growing turns compute into lower loss faster than staying.

## Goal by goal

| # | Goal | Verdict | Evidence |
|---|---|---|---|
| 9 | Solve context rot | **Won** (prediction); **partial** (lookup) | Held-out bits/byte keep *improving* as context grows (512→8,192: 1.214→1.167 on the 46M model; GPT-2: 1.217/1.225, cannot pass 1,024). One million tokens of context in **1.13 GiB**, flat from 131k. Looking up one fact among many in context: 50% vs GPT-2 40% at 10 facts, near chance at 30+ (the hard test: every wrong option also appears in the context). |
| 6 | Consumer GPU / inference within 4 GB | **Won** for inference; training fits 24 GB at our sizes | 452 vs 250 tok/s decoding, 931 vs 990 MiB (46M vs GPT-2 124M); 1M-token context 1.13 GiB. |
| 8 | Everything intrinsic | **Achieved** for rate and size | Rate: schedule-free Prodigy at its published defaults beat the hand-tuned cosine schedule over 30,000 steps (1.2572 vs 1.2720; our fused port 1.2611). Size: decided by trials (below). The first intrinsic rate we built (a hypergradient) *collapsed* over a long run and was withdrawn. |
| 7 | Continuous growth and learning | **Growth: works and is model-decided; learning: improved** | Growth is exact (loss unchanged to 4 decimals at every event). After the point where a bigger model is worth having (~3B tokens for 46M), widening overtakes staying within ~5,000 steps. Continual learning: new facts learned 22%→78-88% with self-rehearsal keeping old facts at 67% vs 54%. **[run70]** growth decisions over a 29.5B-token run. |
| 4-5 | Knows what it knows / abstains | **Working** | Learned from its own mistakes (R-Tuning, balanced): everyday facts right 45.8 / wrong 41.7 / declined 12.5% vs 47.9 / 52.1 / 0 before; SciQ wrong 92%→29%. Calibration separation +0.21-0.24 vs GPT-2 +0.19. |
| 1 | Surpass the LLMs | **Partial** | Ahead of GPT-2 124M on context rot, inference, calibration, facts (33-35/48 vs 33/48), induction (97.8-99.5 vs 98.5), everyday Q&A (47.9% vs 16.7%); generation quality now level with GPT-2. **Behind SmolLM2-135M on Q&A (62.5%)**, which trained on ~2T tokens to our ≤12B. **[run70]** |
| 3 | Self-improving | **Blocked, measured** | Self-labelling by majority vote needs a model right more often than wrong; ours were 19-38% right on the candidate question sets, so training on them would teach errors. Re-tested on the run70 model. |
| 2 | A cognitive model | Profile of the measures above | Memory (9), learning without forgetting (7), metacognition (4-5), self-organised growth and rate (7-8). |

## How the model decides its own size

At every doubling of the current stage's length it pauses and **trials** each option of its next growth step (widen by
one step, or double the depth with new layers whose outputs start at zero) against **not growing**: each branch trains
the same number of steps from the same snapshot on the same batches. It keeps the option only if, over the second half of
its trial, it turns compute into held-out loss faster than staying, and is not behind. Validated: it says STAY before
the crossover where the 118M control shows staying small is compute-optimal, and widening is the option that pays
after it. Three biases had to be removed to make the trial fair (the optimizer's step size restarting, its weight
average hiding new capacity — fixed with a clock per weight — and a sign error in the decision), all documented.

## Retractions and failures (all in PROGRESS.md)

- The first intrinsic learning rate (hypergradient) beat the tuned schedule over 6,000 steps and **collapsed** over
  120,000 (rate 3e-3 → 1.5e-6, learning stalled): withdrawn.
- The 12B-token growth run on a fixed schedule produced a *worse* model than its 118M predecessor (it reached full size
  too late, at a decayed rate).
- Several growth tests were invalid and re-run: layers frozen by a compounding warm-up under the new optimizer; step size
  restarting when a growth fell on the first step after a resume; clocks created after the growth.
- Copying trained layers to deepen (stacking) diverged after the crossover.
- The in-context recall test overstated recall (wrong options not in the context); re-measured.

## Remaining work before the deadline (2026-10-14)

1. **[run70]** finishes; recall training, fine-tune, abstention, full scorecard on it.
2. Self-improvement attempt on the run70 model, gated on its own answers being mostly right.
3. Training speed at short context (attention is faster below ~4k tokens: 0.72x at 512; we are faster above: 1.37x at 16k).
