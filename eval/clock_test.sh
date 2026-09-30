#!/bin/bash
# DO PER-WEIGHT AVERAGING CLOCKS SWITCH NEW CAPACITY ON? From rl_sfp (12L d512, 30,000 steps with the fused schedule-free
# Prodigy), 8,000 more steps each, same data and seed: unchanged | widen to 768 with one clock per group (as the package)
# | widen to 768 with a clock per weight (new slots start at 0).
set -u
cd "$(dirname "$0")/.."
until grep -q "END train" logs/rate_long_sfp.out; do sleep 30; done
OUT=eval/clock_test.txt; echo "=== averaging-clock test on the fused optimizer, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; shift
  mkdir -p /mnt/kv_cache/ct_$N; cp /mnt/kv_cache/rl_sfp/ckpt.pt /mnt/kv_cache/ct_$N/seed.pt
  .venv/bin/python -u -m src.train --steps 38000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms --optim sfp \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 2 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 2000 --save-every 100000 --cap 12 --resume /mnt/kv_cache/ct_$N/seed.pt --out /mnt/kv_cache/ct_$N "$@" > logs/clock_$N.out 2>&1
  echo "$N: $(grep -oE 'GREW.*loss [0-9.]+ -> [0-9.]+' logs/clock_$N.out | grep -oE 'loss [0-9.]+ -> [0-9.]+') | $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/clock_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') $(grep -oE 'resumed optimizer state|not loadable' logs/clock_$N.out | head -1) $(grep -oE 'Error.*' logs/clock_$N.out | head -1)" >> $OUT; }
run unchanged &
run widen_shared --grow-schedule 30000:w768 --sf-clock 0 &
run widen_perweight --grow-schedule 30000:w768 --sf-clock 1 &
wait; echo CLOCK_DONE >> $OUT
