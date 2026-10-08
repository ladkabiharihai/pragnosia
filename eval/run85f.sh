#!/bin/bash
# RUN85, THE FINAL PHASE (plan approved by the user 2026-10-06: "the vision resolved anyhow, highest priority"). From run85's
# latest checkpoint:
#   recall ON with a smooth start (--recall 16 --recall-gated 1: per-head gates born at -4): the 4,000-step A/B found it
#     pays on 32k held-out streams (-0.034 nats/token) and gives needles 92% at 32k, 79% at 64k (PROGRESS.md 2026-10-05)
#   the data mix CHOSEN BY THE MODEL (--mix-auto 500, learning progress on held-out slices) over 12 sources: the main
#     corpora, the retrieval documents (needles, lookup, far needles) and the instruction corpora (TRAIN splits only)
#   NO growth trials in this phase (2026-10-06): with recall's 8k-key carried stores (~4 GiB over 16 stream groups) a
#     grown branch's capture passed the 34 GiB cap at the first trial (R, 398,000), and a trial that cannot fit crashes the
#     trainer and repeats on every resume; the last six trials all decided STAY. The step size held (--freeze-d 1).
# usage: eval/run85f.sh <run85 checkpoint copy>
set -u
cd "$(dirname "$0")/.."
if pgrep -f '^(\S*/)?python[0-9.]* -u -m src\.train' >/dev/null || pgrep -f "^/bin/bash eval/keep_training.sh logs/train85.out" >/dev/null; then echo "run85f: a trainer or run85 wrapper is already running; refusing" >&2; exit 3; fi
SEED=$1
cp $SEED /mnt/kv_cache/run85/resume_from.pt
PLAN="R|W|d,R|W|d,R|W|d,R|W|d,R|W|d,R|W|d"
echo "=== RUN85 FINAL PHASE (run85f.sh) $(date -u '+%Y-%m-%d %H:%M UTC') from $SEED: recall on (gated), the mix chosen by the model over 12 sources ===" >> logs/train85.out
eval/keep_training.sh logs/train85.out TRAIN85_DONE --steps 2400000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 \
  --carry-half 1 --recall 16 --recall-gated 1 --mix-auto 500 \
  --heldout-long 32768 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 21 --freeze-d 1 \
  --data /mnt/kv_cache/tokfinal_v8192 --save-every 4000 --eval-every 4000 --cap 34 \
  --resume /mnt/kv_cache/run85/resume_from.pt --out /mnt/kv_cache/run85
