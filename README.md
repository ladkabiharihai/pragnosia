# Pragnosia

A research effort with nine goals, stated in [VISION.md](VISION.md). The vision file deliberately proposes no
architecture; what is being tested and why is recorded, dated, in [PROGRESS.md](PROGRESS.md).

## Where things are

| file | what it is |
|---|---|
| [VISION.md](VISION.md) | the nine goals. The work is accountable to these and nothing else. |
| [CONSTRAINTS.md](CONSTRAINTS.md) | what hardware, data and time we actually have, measured; and the rules for how work is done. Read before proposing anything. |
| [PROGRESS.md](PROGRESS.md) | dated log of everything that happens, including failures and corrections. Append only. |
| `src/common.py` | shared infrastructure: production safety, GPU capping, pid identification, determinism, the `Run` logger that also journals, corpus access, bits-per-byte evaluation. |
| `src/corpus.py` | the eight source corpora, their measured sizes, the mix, and streaming access to the raw text. |
| `src/spin.py` | the carrier: a diagonal-complex linear recurrence, `h_t = lambda h_{t-1} + gamma u_t`, evaluated as a causal convolution by FFT with a hand-written backward; optional input/output gates (`gspin`) and per-token forgetting (`fspin`). Our own implementation. |
| `src/assoc.py` | the associative carrier (`aspin`): a per-head memory of key-value outer products with input-dependent forgetting and learned rotation frequencies, read by content, constant in size, carried across windows. Built after the diagonal carrier was measured unable to retrieve by content; mathematically in the linear-attention family, and says so. |
| `src/scan_kernel.py` | one Triton kernel per direction for the recurrence with a per-token `lambda_t`: the literal sequential form, so nothing can overflow. What `fspin` trains through, and the inference form for every variant. |
| `src/model.py` | the language model around the carrier (`LM`), the attention control (`Attn`, baseline only), `forward_carried` for state across windows. |
| `src/optim.py` | `Fabric`: factored second moment, trust-ratio steps, growth-aware state, fused and CUDA-graph replayed. |
| `src/train.py` | the trainer: cosine schedule, spike guard, divergence abort, verified checkpoints, resume with optimizer state, BPB eval. |
| `src/calib.py` | the goal probes: calibration (goal 4), real-vs-fabricated confidence (goal 5), needle retrieval through the carried state (goal 9), BPB against GPT-2. |
| `src/facts.py` | 48 factual multiple-choice items used by the goal-5 probe. |
| `tests/` | `test_common.py` (13), `test_optim.py` (21), `test_spin.py` (45), `test_model.py` (13, growth), `test_assoc.py` (15). All must pass before a run. |
| `logs/` | one file per run, not tracked. |
| `.venv/` | this project's environment, python 3.12.3 + torch 2.14.0+cu130. Not tracked. Do not use `parakeet_env`. |

Fourteen source files. A fifteenth needs a reason.

## Current state, 2026-09-11

**Data:** 200.8B raw tokens downloaded, 1.01B tokenized at vocab 8192 (see CONSTRAINTS.md for paths and mix).

**Architecture under test:** the spin carrier (no attention, no Mamba, no hybrid, by instruction). A 77.6M
model trained on the 1B slice reaches BPB 1.3105 (GPT-2 124M on the same bytes: 1.2230, with 9x the tokens).
The matched attention control is training now (`/mnt/kv_cache/run2`).

**What the trajectory probes say so far** (24 checkpoints of the spin run): confidence on real subjects
exceeds fabricated at 21 of 24, with a small positive and growing gap (single seed, n=48); retrieval reaches
256 tokens and barely past the 512 window through the carried state. Both are hypotheses, not results.

**Known costs:** at 512-token windows spin trains 1.21x slower per step than attention (1.39x on forward and
backward alone), reaching parity around 6k tokens. At inference its state is 74 KB and constant where a KV
cache at 8k context is 277 MB and growing.

## Running things

```bash
.venv/bin/python tests/test_common.py && .venv/bin/python tests/test_optim.py && .venv/bin/python tests/test_spin.py

# train (defaults: d768 L12 spin, 120k steps of 16x512, fused graphed optimizer, snapshots every 5k)
.venv/bin/python -m src.train --steps 120000 --out /mnt/kv_cache/runN
.venv/bin/python -m src.train --mixer attn --d 704 --out /mnt/kv_cache/runN     # matched control
.venv/bin/python -m src.train --resume /mnt/kv_cache/runN/ckpt.pt ...            # restores optimizer state too
```

Every run uses `src/common.py:Run`, which writes one log file and appends a dated entry to PROGRESS.md on
completion, including on failure. Long jobs are launched with `setsid nohup ... &` and their PID recorded
from `ps` by executable name, never from `pgrep -f`, which has matched the launching shell five times.

## How work is done here

The full list is in CONSTRAINTS.md. The three that matter most:

- **Nothing runs without approval,** unless the decision has been explicitly delegated.
- **Prove it before building on it.** A trend without an error bar is a hypothesis; a single seed is a
  hypothesis. Say which it is.
- **Full disclosure.** Failures, wrong numbers and mistaken premises are reported immediately, along with what
  they invalidate. PROGRESS.md carries the corrections next to the claims they correct.
