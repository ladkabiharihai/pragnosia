---
name: report-times-in-ist
description: "Give the user every time/ETA in IST (UTC+5:30), not UTC"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ab0dd831-724a-4bd0-be9e-dd7e1ea6f491
  modified: 2026-09-24T11:06:07.925Z
---

Report all times and ETAs to the user in IST (UTC+5:30). PROGRESS.md entries stay UTC-timestamped (repo rule), but anything said to the user is IST.

**Why:** the user said "change the time to ist so i can moniter based on my time utc is hard to track" (2026-09-24).
**How to apply:** convert with `TZ=Asia/Kolkata date` or add 5:30 to UTC; label times "IST".
