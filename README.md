# Pragnosia

A research effort with eight goals, stated in [VISION.md](VISION.md). Nothing else is settled: there is no
chosen architecture, and the vision file deliberately does not propose one.

## Where things are

| file | what it is |
|---|---|
| [VISION.md](VISION.md) | the eight goals. The work is accountable to these and nothing else. |
| [CONSTRAINTS.md](CONSTRAINTS.md) | what hardware, data and time we actually have, measured; and the rules for how work is done. Read before proposing anything. |
| [PROGRESS.md](PROGRESS.md) | dated log of everything that happens, including failures. Append only. |
| `src/common.py` | shared infrastructure: production safety, GPU capping, pid identification, determinism, logging, corpus access. |
| `tests/test_common.py` | verifies the guards actually guard. |
| `logs/` | one file per run, not tracked. |

Five files. A sixth needs a reason.

## Current state, 2026-09-10

**No data, no tokenizer, no checkpoints, no architecture.** All deleted deliberately. The previous effort is
in git history at `9139b56` if any of it is worth recovering.

Before any training can happen: source corpora have to be downloaded, a tokenizer built, and the corpora
tokenized. Each of those is a decision to be made deliberately rather than inherited.

## Running things

```bash
/opt/code/parakeet_env/bin/python tests/test_common.py     # verify the guards
```

Every run should use `src/common.py:Run`, which writes one log file and appends a dated entry to PROGRESS.md
on completion, including on failure:

```python
from src.common import Run, require_prod, cap_gpu
require_prod()                      # refuse to start if production is unhealthy
with Run("name", note="what and why", seed=0) as r:
    cap_gpu(8.0)                    # measured, not guessed
    r.log("...")
```

## How work is done here

The full list is in CONSTRAINTS.md. The three that matter most:

- **Nothing runs without approval.** Proposals are written before anything is started.
- **Prove it before building on it.** A trend without an error bar is a hypothesis; a single seed is a
  hypothesis. Say which it is.
- **Full disclosure.** Failures, wrong numbers and mistaken premises are reported immediately, along with what
  they invalidate.
