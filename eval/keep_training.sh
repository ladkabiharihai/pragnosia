#!/bin/bash
# KEEP A TRAINING RUN GOING (training efficiency). run70 died at the cap at 18:20 UTC on 2026-09-30. Its launch script then
# reported success, the pipeline evaluated the wrong checkpoint, and the card sat idle ~10.5 hours until a human looked.
# This wrapper runs the trainer. If the trainer exits with an error, it resumes at once from the run's own last checkpoint,
# which carries its step, shape, optimizer state and stage clock. It gives up only when two attempts in a row fail
# without the checkpoint moving forward. The done marker is written only on success.
# usage: eval/keep_training.sh <log> <DONE_MARKER> <src.train args...>       (the args must include --out DIR)
# To stop a run on purpose, kill THIS script's PID first and the trainer second; otherwise it is resumed.
set -u
cd "$(dirname "$0")/.."
LOG=$1; MARK=$2; shift 2
ARGS=("$@")
OUT=""; for ((k=0; k<${#ARGS[@]}; k++)); do [ "${ARGS[$k]}" = "--out" ] && OUT=${ARGS[$((k+1))]}; done
[ -n "$OUT" ] || { echo "keep_training: --out is required" >&2; exit 2; }
step_of () { .venv/bin/python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$1" 2>/dev/null || echo -1; }
last=-2; fails=0
while true; do
  .venv/bin/python -u -m src.train "${ARGS[@]}" >> "$LOG" 2>&1 && { echo "$MARK $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$LOG"; exit 0; }
  rc=$?
  s=$(step_of "$OUT/ckpt.pt")
  if [ "$s" = "$last" ]; then fails=$((fails + 1)); else fails=1; last=$s; fi
  echo "TRAINER EXITED with code $rc at $(date -u '+%Y-%m-%d %H:%M UTC'); last checkpoint at step $s (failure $fails there)" >> "$LOG"
  # PATIENT (2026-10-06): a restart that collides with a side job (the 6-hourly scorecard) dies of out-of-memory at
  # capture; twice in a row and the card sat idle 14 hours overnight. Now: up to 6 attempts at one checkpoint, 10
  # minutes apart, so a collision waits the side job out
  if [ "$s" -lt 0 ] || [ "$fails" -ge 6 ]; then
    echo "${MARK%_DONE}_FAILED $(date -u '+%Y-%m-%d %H:%M UTC'): no progress past step $s across $fails attempts; stopping for a human" >> "$LOG"
    exit 1
  fi
  [ "$fails" -ge 2 ] && sleep 600
  # resume from the run's own checkpoint (a copy, since the resumed run rewrites ckpt.pt); --resume-stage is dropped
  # because the checkpoint carries the stage clock
  NEW=(); skip=0
  for ((k=0; k<${#ARGS[@]}; k++)); do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case "${ARGS[$k]}" in --resume|--resume-stage) skip=1; continue;; esac
    NEW+=("${ARGS[$k]}")
  done
  cp "$OUT/ckpt.pt" "$OUT/resume_from.pt"
  ARGS=("${NEW[@]}" --resume "$OUT/resume_from.pt")
  echo "=== RESUMED AUTOMATICALLY $(date -u '+%Y-%m-%d %H:%M UTC') from step $s ===" >> "$LOG"
done
