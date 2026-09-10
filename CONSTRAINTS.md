# Hard constraints

Facts and rules. Everything here was measured or agreed, not assumed. Verified 2026-09-10 09:51 UTC.
If something in this file is wrong, fix the file first and say so, before doing anything that relies on it.

## Hardware, and what we do not own

One NVIDIA H100 NVL, 95,830 MiB total (93.6 GiB). Driver 595.58.03.
**It is shared with production, and production must not go down.**

| holder | memory | note |
|---|---|---|
| vLLM engine (:8000) | 39.4 GB | serving; health check is `curl localhost:8000/v1/models` |
| kokoro voice, 4 workers | 7.2 GB | |
| chatterbox | 3.6 GB | |
| whisper | 2.7 GB | |
| **production total** | **~53 GB** | **never touch** |
| **available to us** | **~40 GiB** | and only if production stays at its current load |

Measured 2026-09-10: co-running any GPU job alongside a training run costs the trainer about 40% of its
throughput (175 ktok/s alone, 73 ktok/s while sharing), even when the second job is hard-capped. Memory caps
protect against OOM. They do nothing about contention.

GPU access is time limited. Treat throughput as a deliverable, not a detail.

## Software

This project has **its own virtual environment** at `.venv`, created 2026-09-10. Use it and nothing else.
`/opt/code/parakeet_env` belongs to production tooling and is not to be used or modified.

| | |
|---|---|
| interpreter | `/opt/code/membrae/.venv/bin/python` |
| python | 3.12.3 |
| torch | 2.14.0+cu130, CUDA 13.0 |
| numpy | 2.5.3 |
| tokenizers | 0.23.2 |
| verified | bf16 supported, device visible as NVIDIA H100 NVL, matmul on device executed |

The venv is 5.4 GB and sits on `/`, which has 266 GB free. It is not tracked.

## Storage

- `/mnt/kv_cache` 3.5 TB, 2.6 TB free. Everything large lives here.
- `/` 495 GB, 266 GB free.

## Data: what we actually have

**Nothing. As of 2026-09-10 09:55 UTC there is no training data, no tokenizer and no checkpoint on this
machine.** All of it was deleted deliberately to start clean. Stated plainly so that no plan is built on
something that is not here.

| what | state |
|---|---|
| tokenizer | gone. There is no 32k vocab file anywhere on disk. |
| tokenized corpora | gone. 12 streams, 580 GB, 290.2B tokens, deleted. |
| raw text | never on this disk. The corpora were tokenized elsewhere. |
| checkpoints | gone. 46 files, 29 GB, including the only model with verified claims. |

Everything therefore has to be re-acquired before any training can happen:

1. Download the source corpora. The previous mix was window2 (169.7B), dclm (96.8B), finemath (10.0B),
   opencode (5.9B), openmath (5.6B), tulu3 (0.9B), numina (0.5B), hermes (0.4B), recall (0.3B), and small
   code sets, for 290.2B tokens at a 32768 vocab. That mix is a record of what was used, not a recommendation.
2. Build a tokenizer, then tokenize. Both are decisions to make deliberately rather than to inherit.

**Kept, and not ours to delete:** `/mnt/kv_cache/lmcache_disk` (153 GB) is production's vLLM KV-cache offload
for gpt-oss-20b. **Kept, and useful:** `/mnt/kv_cache/hf_home` and `hf_cache` (17 GB) hold HuggingFace dataset
caches (fineweb-edu, smollm-corpus, NuminaMath) and baseline models (Pythia-160M, SmolLM2-135M/360M,
Qwen2.5-0.5B). Deleting those would save 17 GB of a 3.5 TB disk and make re-downloading slower.

Disk after the clean: 3.1 TB free of 3.5 TB.

## Working rules

These are the rules for how work gets done here. They exist because each one was violated at least once and it
cost real time.

1. **Nothing runs without approval.** No training, no long job, no destructive command until it is explicitly
   asked for. Proposals are written down first, not started.
2. **Prove it before building on it.** Any claim that a later step depends on gets verified first: the maths
   derived, or the measurement taken with its error bar. A trend without an interval is a hypothesis. A result
   from a single seed is a hypothesis. Say which it is.
3. **Every event goes into PROGRESS.md with a UTC date and time**, including failures, dead ends and mistakes.
   A log that only records wins is not a log.
4. **Full disclosure.** When something breaks, when a number moves, when a previously stated figure turns out
   wrong, it is said immediately and in plain terms, along with what it invalidates.
5. **Say when the premise is wrong.** Including when the instruction is based on something inaccurate. That is
   more useful than agreement.
6. **Few files.** A new file needs a reason. Prefer extending an existing one. This whole repository should
   stay small enough to read in one sitting.
7. **Kill only exact, verified PIDs.** Confirm `/proc/<pid>/cmdline` and that the process actually holds the
   GPU memory. Never a name pattern, never a memory threshold. `setsid nohup ... &` returns the wrong PID.
8. **Hard-cap GPU memory per process.** Never gate a launch on free VRAM: free VRAM is a race, a cap is not.
   Measure the requirement first rather than guessing a cap and discovering it by crashing.
9. **Determinism by default.** Seed everything, record the seed, and make a result reproducible from the log.
