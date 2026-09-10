# Vision scorecard: the eight goals, measured

Every line is a measurement on a shipped checkpoint, not an intention. Where a goal has never been measured,
that is stated. Where it is not met, that is stated first.

| # | goal | status | the number that decides it |
|---|---|---|---|
| 1 | Surpass the LLM models | **NOT MET** | beats GPT-2 124M at matched tokens (1.0798 vs 1.0935) and Pythia-160M on 33x fewer tokens; loses to SmolLM2-135M (0.8854) and Qwen2.5-494M (0.8011) |
| 2 | Cognitive model | **PARTIAL** | brain-region routing: no quality gain at matched compute (1.2819 vs 1.2737), not adopted. Layer specialisation IS emergent: read sharpness spans 1.3 to 7.4, write strength spans 0.01 to 0.80 |
| 3 | Self improving | **MECHANISM YES, DECISION NO** | growth function preserving to 4e-06 (fp32); all three growth events logged DECIDED BY: FLOOR. The sensor has never once decided |
| 4 | Knows what it knows | **NOT MET** (measured 2026-09-10) | AUROC of confidence predicting its own correctness: 0.681 at 125M, **0.402 at 405M**, i.e. below chance |
| 5 | Abstains on what it doesn't | **NOT MET, INVERTED** (measured 2026-09-10) | confidence on fabricated subjects EXCEEDS confidence on real ones in all three models (0.689 vs 0.642; 0.673 vs 0.634; 0.617 vs 0.536) |
| 6 | Consumer GPU, any params, 4GB inference | **INFERENCE PARTIAL, TRAINING NOT MET** | 125M inference peaks at 2.24 GiB including a 65k-token read; 405M at 7.15 GiB; training runs used 24 to 38 GiB |
| 7 | Continuous growth and continuous learning | **LEARNING YES, GROWTH PARTIAL** | online: +4.50 nats acquisition, 89% retention across three more facts, +0.005 BPB forgetting, 8s for four facts. Growth: see 3 |
| 8 | Intrinsic, not scaffolding | **WEAKEST** | growth timing is a hand-set 22 tok/param floor and 40 ceiling; verification is external corpus retrieval; generation stability is a hand-tuned 0.35 threshold and a 12-token run ban; the LR rule is a hand-set muP reference |

## The connection that matters

Goal 8 fails **because** goals 4 and 5 fail, and this is not a coincidence.

`verify.py` decides what may enter the weights by retrieving evidence from a trusted corpus. That is external
scaffolding, and it exists because the model cannot judge its own knowledge: measured, its confidence does not
predict its correctness (AUROC 0.681, and 0.402 at 405M), and it is *more* confident about things that do not
exist than about things that do. A model that could self-assess would not need a corpus to gate its own
learning; the gate is compensation for a missing intrinsic capability, and that capability is now measured and
absent.

The same pattern runs through the rest of the scaffolding. The 22 tok/param growth floor exists because the
saturation sensor cannot be trusted to fire. The 0.35 repetition threshold exists because the model cannot tell
it has started looping. Each hand-set constant stands in for a self-monitoring ability the model does not have.

**So the highest-value target is not more scale or more tokens. It is calibration.** If confidence predicted
correctness, goal 5 follows almost immediately (abstain below a threshold), goal 8 gets much closer (the gate
becomes intrinsic), and goal 4 is met by definition. It is also the cheapest of the eight to attack: it is a
property of the output distribution, testable in minutes on checkpoints we already have, and there is a
literature of methods that do not require retraining from scratch.

## What IS solid, and should be said plainly

- Constant memory and constant per-token latency at 1,048,576 tokens, 2.74 GiB, verified live.
- Beats GPT-2 124M at matched parameters and matched tokens on a protocol built to be fair to GPT-2.
- Online learning that retains and refuses: 0 of 450 adversarial poisoning cases accepted.
- Function-preserving growth, verified in float32 per operation.
- Retrieval range scales with state capacity: 2.1x at t=5.09 for 2.63x the state.

Seven of these reproduce in one command via `harness.py`. That is the provable part, and it is not nothing:
it is an efficient small model with properties that do not exist together elsewhere. What it is not, yet, is a
model that knows what it knows.
