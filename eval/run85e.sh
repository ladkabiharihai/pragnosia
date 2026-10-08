#!/bin/bash
# RUN85, RESUMED FROM ITS LAST HEALTHY CHECKPOINT (step 360,000) WITH THE STEP SIZE HELD (2026-10-05 12:55 UTC).
# After the automatic resume at 360,000 the shared Prodigy step size ratcheted 9.69e-4 -> 2.73e-3 (it only rises) and the
# weights degraded steadily: the trainer's own held-out bits/byte at a 1,024 window 1.0394 (360k) -> 1.0480 (368k) ->
# 1.0521 (372k), while the 2,048 window the model grew was worth -0.004 at the same weights (PROGRESS.md). So: back to
# 360,000, --freeze-d 1 holds the step size at the model's own estimate there (9.69e-4; schedule-free at a constant
# step, as published for SF-AdamW), and the trial rules fixed today (time-priced, same batches) re-decide R and M.
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
PLAN="R|W|d,R|W|d,R|W|d,R|W|d,R|W|d,R|W|d"   # (2026-10-06) M lost its trial at 368,600 (2.6498 -> 2.6939 in 1,000 steps); recall is a fine-tuned variant instead
echo "=== RUN85 RESUMED (run85e.sh, step size held, cap 34) $(date -u '+%Y-%m-%d %H:%M UTC') from $SEED, growth options $PLAN (fixes of the recall review in) ===" >> logs/train85.out
eval/keep_training.sh logs/train85.out TRAIN85_DONE --steps 2400000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 \
  --carry-half 1 --grow-auto "$PLAN" --trial-steps 1000 --trial-frac 16 \
  --heldout-long 32768 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 20 --freeze-d 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_mix --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34 \
  --resume /mnt/kv_cache/run85/resume_from.pt --out /mnt/kv_cache/run85
