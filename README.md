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
| `src/bench.py` | THE SCORECARD: every axis the vision is accountable on, measured in one place against GPT-2 124M. Nothing scales until it says to. |
| `src/facts.py` | 48 factual multiple-choice items used by the goal-5 probe. |
| `tests/` | `test_common.py` (13), `test_optim.py` (21), `test_spin.py` (45), `test_model.py` (16), `test_assoc.py` (25). All must pass before a run. |
| `logs/` | one file per run, not tracked. |
| `.venv/` | this project's environment, python 3.12.3 + torch 2.14.0+cu130. Not tracked. Do not use `parakeet_env`. |

Fifteen source files. A sixteenth needs a reason.

## Current state, 2026-09-15

**Data:** 200.8B raw tokens downloaded; 1.01B tokenized at vocab 8192, 10B at vocab 16384, 1.2B of long
documents at vocab 8192 (paths and mix in CONSTRAINTS.md).

**Architecture:** the associative carrier (`src/assoc.py`), reached by measurement. The diagonal carrier and
its gated variants could not retrieve by content (needle lift +0.05 to +0.9 against attention's +4.8) and lost
0.07 to 0.14 bits/byte to it; the associative carrier matches attention's in-window retrieval and beats it on
web-text compression at a 512-token window (1.0994 against 1.1165 at 78M; GPT-2 124M is 1.1061 on the same
bytes with nine times the tokens). Mathematically it is in the linear-attention family and the file says so.

**Against the gate in VISION.md, at 78M, measured by `src/bench.py`:**

| axis | result |
|---|---|
| compression, web text, 512 window | win: 1.0994 vs attention 1.1165, GPT-2 1.1061 |
| compression, long documents, same 8192 window for both | loss: 0.9232 vs attention 0.8301 |
| context rot | mechanism found: usable context = training window; 512-trained rots +0.108, 8192-trained does not. Attention given the same window also does not, and improves 2.4x more |
| facts (48, four-way) | loss: 22/48 for every 78M model, GPT-2 26/48 |
| coherence (greedy) | tie: everything loops |
| training throughput, ctx 2048 | 273 vs 222 ktok/s (carrier faster); a Python chunk loop starves the GPU at small batch |
| inference memory | win: 2 MiB constant vs a KV cache of 64 MiB at 1k, 8 GB at 128k |
| inference speed | loss: 12.7 ms/token vs 8.1 (attention) and 3.9 (GPT-2), kernel quality |
| growth mid-run | works; 0.038 behind from-scratch at equal steps |

**Verdict:** "surpass the LLMs" is not met. What is proven: a fixed 2 MiB memory gets within 0.09 bits/byte of
an unbounded 1.1 GB cache on the text that favours the cache most, beats the same attention on ordinary web
text, and works at any context length in constant memory. That is a memory trade, not a win on the LLMs' metric.

**What would move it:** the delta write rule as a kernel (implemented and exact in `src/assoc.py`, unrun for
memory); a Triton kernel for the chunked scan; and more tokens, which is the only thing that fixes facts.

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
from `ps` by executable name, never from `pgrep -f` or `pkill -f`, which have matched the launching shell six times.

## How work is done here

The full list is in CONSTRAINTS.md. The three that matter most:

- **Nothing runs without approval,** unless the decision has been explicitly delegated.
- **Prove it before building on it.** A trend without an error bar is a hypothesis; a single seed is a
  hypothesis. Say which it is.
- **Full disclosure.** Failures, wrong numbers and mistaken premises are reported immediately, along with what
  they invalidate. PROGRESS.md carries the corrections next to the claims they correct.
