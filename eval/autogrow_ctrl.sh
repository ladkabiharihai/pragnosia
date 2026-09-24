#!/bin/bash
# CONTROLS for the model-decided growth run (logs/autogrow2.out: start d512, widened itself to d768 at step 4,200,
# held-out BPB 1.3928 at 10,000 steps). Same steps, data, seed, intrinsic rate from 3e-3, NO growth:
# fixed at the start width (d512) and fixed at the grown width (d768) from step 0.
set -u
cd "$(dirname "$0")/.."
OUT=eval/autogrow.txt
echo "=== model-decided growth vs fixed size, 10,000 steps x 16,384 tokens, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
echo "grown (d512 -> d768 at 4,200, self-triggered): BPB $(grep -oE 'BPB [0-9.]+' logs/autogrow2.out | tail -1 | cut -d' ' -f2)  wall $(grep -oE 'END train  ok  \([0-9.]+ min' logs/autogrow2.out | grep -oE '[0-9.]+ min')" >> $OUT
for D in 512 768; do
  .venv/bin/python -u -m src.train --steps 10000 --d $D --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms \
    --intrinsic-lr 1 --lr 3e-3 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 10000 --save-every 100000 --cap 24 --out /mnt/kv_cache/autogrow_tmp > logs/autogrow_fixed$D.out 2>&1
  echo "fixed d$D: BPB $(grep -oE 'BPB [0-9.]+' logs/autogrow_fixed$D.out | tail -1 | cut -d' ' -f2)  wall $(grep -oE 'END train  ok  \([0-9.]+ min' logs/autogrow_fixed$D.out | grep -oE '[0-9.]+ min')" >> $OUT
done
echo CTRL_DONE >> $OUT
