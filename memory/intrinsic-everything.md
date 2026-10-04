---
name: intrinsic-everything
description: "User requires every training decision to be the model's own - rate, schedule, WHEN to grow and HOW BIG; no hand-set sizes or thresholds"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab0dd831-724a-4bd0-be9e-dd7e1ea6f491
  modified: 2026-09-29T08:14:18.327Z
---

Everything must be intrinsic: no hand-set learning rate or schedule, no preset growth steps, and no hand-picked model size ("why are we growing the model to 118 manually why not intrinsically", 2026-09-29). A hand-set starting size is acceptable only as a CONTROL, never as the deliverable.

**Why:** VISION.md goal 8 (everything intrinsic) and goal 7 (continuous growth); investors judge against the vision.
**How to apply:** when growth doesn't pay, fix the growth operator/decision rather than picking a bigger size. Validate on long horizons before full runs (the 6k-step intrinsic-rate result collapsed over 120k steps). State at 2026-10-04: rate = schedule-free Prodigy at defaults; growth trials (W/d/R) decide size and attention reach. Reach trials MUST be judged on long carried streams (train.py --heldout-long 32768): judged on 8k windows the model grew attention to 8,192 and paid +0.034 bits/byte past 8k; on long streams it chose 1,024 and is flat to 1M (run82). User said 98M is enough to test; scaling held until gaps close. Deadline extended ~15 days from 2026-09-29 (to ~2026-10-14). Related: [[report-times-in-ist]].
