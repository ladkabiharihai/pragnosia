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

- `/mnt/kv_cache` 3.5 TB, 2.7 TB free. Everything large lives here.
- `/` 495 GB, 266 GB free.

## Data: what we actually have

Updated 2026-09-11. The clean of 2026-09-10 deleted everything; this is what has been re-acquired since,
measured, not planned.

| what | state |
|---|---|
| raw corpora | `/mnt/kv_cache/hf_dl`, 470 GB, 200.8B tokens by streaming length measurement (`src/corpus.py`). Eight sources: dclm .413, fineweb-edu .322, starcoderdata .130, finemath .036, open-web-math .030, arxiv .030, wikipedia .023, algebraic-stack .015 (achieved token shares). No Gutenberg, by instruction. |
| tokenizers | `/mnt/kv_cache/tok_v{2048,4096,8192,16384,32768,65536}.json`, BPE, trained on a held-out slice. Compression 2.653 / 3.017 / 3.374 / 3.677 / 3.906 / 4.061 bytes per token. Compute-optimal choice by width: 8192 at d512, 16384 at d768 to 1024, 32768 at d1536 and up. |
| tokenized | `/mnt/kv_cache/tok1b_v8192/<source>.bin`, uint16, 1.01B tokens in the achieved mix. This is the ONLY tokenized slice; the 200B corpus is raw. |
| checkpoints | `/mnt/kv_cache/run1` (spin 77.6M, 120k steps, `ckpt.pt` plus `snaps/step_*.pt` every 5k, 12 GB) and `/mnt/kv_cache/run2` (attention control, same schedule). Trajectory probes in `<run>/probe_suite.json`. |
| baselines | `/mnt/kv_cache/hf_home`, `hf_cache`: GPT-2 124M (the BPB anchor), Pythia-160M, SmolLM2, Qwen2.5-0.5B. |

**Not ours to delete:** `/mnt/kv_cache/lmcache_disk` (153 GB) is production's vLLM KV-cache offload.
**Not to be used:** `/mnt/kv_cache/spin` holds the previous spin lineage's HF repo for reference only; nothing
is taken from it (instruction of 2026-09-10).

Disk: 2.7 TB free of 3.5 TB.

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
