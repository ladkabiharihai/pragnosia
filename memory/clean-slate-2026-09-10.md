---
name: clean-slate-2026-09-10
description: Second full reset; all code, data, tokenizer and checkpoints deleted, five-file structure established
metadata:
  type: project
---

On 2026-09-10 the user ordered a full reset after roughly two months of work on a growing recurrent model that
could not be proven to investors. Deleted: 176 source files (recoverable at git 9139b56), 584GB of tokenized
corpora (290.2B tokens across 12 streams), the 32k tokenizer, and 46 checkpoints including p124f.pt, which was
the only model carrying verified claims. The checkpoints and corpora are NOT recoverable.

What replaced it: VISION.md with the eight goals and no architecture, CONSTRAINTS.md with measured hardware and
data facts plus nine working rules, PROGRESS.md as a UTC-timestamped append-only journal, README.md, and
src/common.py which enforces production safety, GPU capping, verified-PID identification, determinism and
logging in code.

**Why:** the previous effort lost more time to process and measurement failures than to modelling ones. Caps
guessed and discovered by crashing, the wrong PID killed leading to two trainers on one card, results quoted
from a single seed, audit tools printing a criterion different from the one they tested, and a learning rate
held constant while the model quadrupled. The rules and the guards exist to make each of those harder.

**How to apply:** do not propose an architecture unless asked; the vision deliberately contains none. Do not
start any run without explicit approval. Correct the user when a premise is wrong, which happened twice on the
day of the reset (there was no raw text on disk to re-tokenize from, and the token ids were trainable without a
tokenizer while text metrics were not). See [[gpu-cap-corunning-jobs]] and
[[gpu-contention-flagship-cost]].
