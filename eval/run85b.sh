#!/bin/bash
# RUN85, RESUMED (2026-10-05) from its own checkpoint after the recall A/B (eval/recall_ab.sh) paused it. Identical to
# eval/run85.sh except: nothing is deleted, the checkpoint carries the step, optimizer and stage clock (no --resume-stage),
# and RECALL (M: each sliding attention query also attends to its top-k keys older than the window) is among the growth
# options, so the model decides by trial on its 32k-token held-out streams whether retrieval past the window pays.
# usage: eval/run85b.sh <run85 checkpoint copy>
set -u
cd "$(dirname "$0")/.."
if pgrep -f '^(\S*/)?python[0-9.]* -u -m src\.train' >/dev/null || pgrep -f "^/bin/bash eval/keep_training.sh logs/train85.out" >/dev/null; then echo "run85b: a trainer or run85 wrapper is already running; refusing" >&2; exit 3; fi
SEED=$1
cp $SEED /mnt/kv_cache/run85/resume_from.pt
# (12:20 UTC) M offered again: the trials now price compute as measured TIME (train.py), so recall's extra time per step
# counts against it; it must buy more held-out gain per hour than staying (2 of the 8 held-out streams are retrieval
# documents, so retrieval past the window counts)
PLAN="M|R|W|d,M|R|W|d,M|R|W|d,M|R|W|d,M|R|W|d,M|R|W|d"
echo "=== RUN85 RESUMED $(date -u '+%Y-%m-%d %H:%M UTC') from $SEED, growth options $PLAN (fixes of the recall review in) ===" >> logs/train85.out
eval/keep_training.sh logs/train85.out TRAIN85_DONE --steps 2400000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 \
  --carry-half 1 --grow-auto "$PLAN" --trial-steps 1000 --trial-frac 16 \
  --heldout-long 32768 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 17 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_mix --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34 \
  --resume /mnt/kv_cache/run85/resume_from.pt --out /mnt/kv_cache/run85
