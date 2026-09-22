---
name: gpu-contention-flagship-cost
description: Co-running experiments alongside the flagship trainer costs it ~40% throughput; keep the card clear during a deadline run
metadata:
  type: project
---

Measured 2026-09-09 on the H100: the flagship trainer (train400f.py, 191M) sustains 124-137 ktok/s when it is
the only non-prod job on the card, and drops to 74 ktok/s while side experiments run alongside it, even when
each of those is hard-capped to 8-11 GiB. The cap protects prod from an OOM; it does nothing about contention
for SMs.

**Why:** over a multi-day run that 40% tax is worth roughly 30 hours, which is larger than any single
experiment that could be run in the same window. On a fixed GPU deadline the throughput IS the deliverable.

**How to apply:** during a deadline-critical training run, treat the card as reserved. Batch diagnostic and
measurement work into a deliberate window, or run it after the flagship finishes. When an experiment genuinely
must co-run, quantify what it costs the trainer first and say so, rather than assuming a memory cap makes it
free. See [[gpu-cap-corunning-jobs]] for the separate memory-safety rule.
