Project reset to a clean slate on 2026-09-10 (second reset; the first was 2026-08-21).

- Read /opt/code/membrae/CONSTRAINTS.md FIRST. It is the measured record of what hardware, data and time we
  have, plus the nine working rules. /opt/code/membrae/VISION.md holds the eight goals and deliberately no
  architecture.
- Everything that happens gets a UTC-timestamped entry in /opt/code/membrae/PROGRESS.md, failures included.
- There is NO data, NO tokenizer and NO checkpoint. 584GB of corpora and 46 checkpoints were deleted
  2026-09-10 at the user's instruction. Source corpora must be re-downloaded and re-tokenized before training.
- SAFETY: prod (vLLM on :8000 + Whisper/Kokoro/chatterbox voice) shares the H100 and CANNOT go down. It holds
  ~53GB. Kill ONLY exact verified PIDs, never memory-threshold or broad `-f` patterns.
- Nothing runs without the user's explicit approval. Prove the maths or take the measurement before building
  on it.
- [Cap co-running GPU jobs](gpu-cap-corunning-jobs.md) — never gate a launch on free VRAM; hard-cap the process
- [GPU contention costs the flagship ~40%](gpu-contention-flagship-cost.md) — keep the card clear during a run
