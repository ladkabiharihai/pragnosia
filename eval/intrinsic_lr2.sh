#!/bin/bash
# ROUND 2: the hypergradient lowers an over-high rate well (start 3e-3 -> 1.4734 held-out, beating the tuned
# schedule's 1.5644) and raises a low one badly (start 1e-4 -> 1.7481). If the result does not care HOW high it
# starts, the start is not a knob: the optimizer finds its rate from above. Starts 1e-2 and 3e-2, same run.
set -u
cd "$(dirname "$0")/.."
OUT=eval/intrinsic_lr.txt; echo "=== round 2: start-high sensitivity, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; shift
  .venv/bin/python -u -m src.train --steps 6000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 6000 --save-every 100000 --cap 20 --out /mnt/kv_cache/ilr_tmp "$@" > logs/ilr_$N.out 2>&1
  echo "$N: loss@2k/4k/6k $(grep -E '\] (2000|4000|6000)/6000' logs/ilr_$N.out | grep -oE 'loss [0-9.]+' | cut -d' ' -f2 | tr '\n' ' ')| held-out $(grep -oE 'BPB [0-9.]+' logs/ilr_$N.out | tail -1) | final $(grep -E '6000/6000' logs/ilr_$N.out | grep -oE 'rates .*' ) | skips $(grep -oE 'total [0-9]+' logs/ilr_$N.out | tail -1) $(grep -oE 'DIVERGED|Error' logs/ilr_$N.out | head -1)" >> $OUT; }
run intrinsic-from-1e-2 --intrinsic-lr 1 --lr 1e-2
run intrinsic-from-3e-2 --intrinsic-lr 1 --lr 3e-2
rm -rf /mnt/kv_cache/ilr_tmp; echo ILR2_DONE >> $OUT
