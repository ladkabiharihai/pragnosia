#!/bin/bash
# IS THE SCHEDULE-FREE AVERAGE WHAT HIDES NEW CAPACITY? From rl_psf (12L d512, 30,000 steps), widen to 768 at once and
# train 8,000 steps (same data and seed as eval/growth_long.sh, whose unchanged run gives 34k 1.2454, 36k 1.2402,
# 38k 1.2352). Optimizer memory carried in both; B also restarts the schedule-free average at the growth.
set -u
cd "$(dirname "$0")/.."
OUT=eval/avg_test.txt; echo "=== averaging-clock test, $(date -u '+%Y-%m-%d %H:%M UTC') (unchanged: 32k 1.2508 34k 1.2454 36k 1.2402 38k 1.2352) ===" >> $OUT
run() { N=$1; shift
  mkdir -p /mnt/kv_cache/at_$N; cp /mnt/kv_cache/rl_psf/ckpt.pt /mnt/kv_cache/at_$N/seed.pt
  .venv/bin/python -u -m src.train --steps 38000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms --optim psf \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 2 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 2000 --save-every 100000 --cap 13 --resume /mnt/kv_cache/at_$N/seed.pt --out /mnt/kv_cache/at_$N "$@" > logs/avg_$N.out 2>&1
  echo "$N: $(grep -oE 'GREW.*loss [0-9.]+ -> [0-9.]+' logs/avg_$N.out | grep -oE 'loss [0-9.]+ -> [0-9.]+') | $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/avg_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') $(grep -oE 'Error.*' logs/avg_$N.out | head -1)" >> $OUT; }
run widen_carry --grow-schedule 30000:w768 &
run widen_restart --grow-schedule 30000:w768 --sf-restart 1 &
wait; echo AVG_DONE >> $OUT
