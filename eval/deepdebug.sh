#!/bin/bash
# WHY DOES TRAINING DIVERGE AFTER A MID-RUN DEEPEN? A plain deepen (12L d512, window 2,048) trains fine
# (6.70 -> 6.70 at the event, 6.14 at step 600, 1 skip), with or without the captured micro-step. The growth
# test that diverged differed in two ways; each is isolated here.
set -u
cd "$(dirname "$0")/.."
OUT=eval/deepdebug.txt; echo "=== round 4 (fixes: skipped steps do not move, NaN-initialised guard history, guard reset at growth, new layers ramp) $(date -u '+%Y-%m-%d %H:%M UTC') (plain deepen: fine, see PROGRESS) ===" >> $OUT
run() { N=$1; shift
  .venv/bin/python -u -m src.train --steps 600 --d 512 --layers 12 --mixer grspin --norm rms \
    --lr 1.5e-3 --new-lr 1e-3 --warm 50 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
    --data /mnt/kv_cache/tokfull_v8192 --eval-every 100000 --save-every 100000 --cap 20 "$@" \
    --out /mnt/kv_cache/deepdebug_tmp > logs/deepdebug_$N.out 2>&1
  echo "$N: $(grep -E '^\[.*\] (200|400|600)/600' logs/deepdebug_$N.out | grep -oE 'loss [0-9.]+' | tr '\n' ' ') | $(grep -oE '(GREW|DEEPENED).*loss [0-9.]+ -> [0-9.]+' logs/deepdebug_$N.out | tr '\n' ' ') | skips $(grep -oE 'total [0-9]+' logs/deepdebug_$N.out | tail -1) $(grep -oE '(Error|DIVERGED).*' logs/deepdebug_$N.out | head -1)" >> $OUT; }
run fixed-plain-at-300 --ctx 2048 --batch 8 --accum 1 --grow-schedule 300:d --new-warm 100
run fixed-widen-then-deepen --ctx 2048 --batch 8 --accum 1 --grow-schedule 150:w768,300:d --new-warm 100
rm -rf /mnt/kv_cache/deepdebug_tmp; echo DEBUG_DONE >> $OUT
