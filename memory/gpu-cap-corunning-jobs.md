---
name: gpu-cap-corunning-jobs
description: Any job co-running on the shared H100 must be hard-capped with util.cap_gpu, never launched behind a free-memory check
metadata:
  type: feedback
---

Before launching any GPU job that co-runs with prod or a training run, cap it with `util.cap_gpu(gib)`
(wraps `torch.cuda.set_per_process_memory_fraction`). Do NOT gate the launch on a free-VRAM check.

**Why:** on 2026-09-07 I launched `exp/bptt.py` behind a ">20 GiB free" check. It passed at 27.8 GiB free, then
the job used 25.9 GiB (I had estimated 10-15) and drove the card to 1.9 GiB free with prod's vLLM on it. Nothing
crashed, but only by luck. A guard that compares free memory against an *estimate* of a job's usage is not a
guard - it fails exactly when the estimate is wrong, which is when you need it. Truncated BPTT was the specific
trap: it holds K chunks of activations in the graph at once, so memory grows linearly in K.

**How to apply:** `from util import cap_gpu; cap_gpu(12)` at the top of any co-running experiment. The job then
OOMs against its own ceiling instead of consuming prod's headroom. Recovery, if needed, is still by exact
verified PID checked against `/proc/<pid>/cmdline` - see [[prod-safety-shared-h100]].
