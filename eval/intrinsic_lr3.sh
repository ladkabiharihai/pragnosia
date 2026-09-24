#!/bin/bash
# ROUND 3: with the halve-on-spike back-off, does ANY high start work? 3e-2 diverged without it; 1e-1 is extreme.
set -u
cd "$(dirname "$0")/.."
OUT=eval/intrinsic_lr.txt; echo "=== round 3: halve on a rejected spike, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; shift
  .venv/bin/python -u -m src.train --steps 6000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 6000 --save-every 100000 --cap 20 --out /mnt/kv_cache/ilr_tmp "$@" > logs/ilr_$N.out 2>&1
  echo "$N: loss@2k/4k/6k $(grep -E '\] (2000|4000|6000)/6000' logs/ilr_$N.out | grep -oE 'loss [0-9.]+' | cut -d' ' -f2 | tr '\n' ' ')| held-out $(grep -oE 'BPB [0-9.]+' logs/ilr_$N.out | tail -1) | final $(grep -E '6000/6000' logs/ilr_$N.out | grep -oE 'rates .*' ) | skips $(grep -oE 'total [0-9]+' logs/ilr_$N.out | tail -1) $(grep -oE 'DIVERGED|Error' logs/ilr_$N.out | head -1)" >> $OUT; }
run backoff-from-3e-2 --intrinsic-lr 1 --lr 3e-2
run backoff-from-1e-1 --intrinsic-lr 1 --lr 1e-1
run backoff-from-1e-2 --intrinsic-lr 1 --lr 1e-2
rm -rf /mnt/kv_cache/ilr_tmp; echo ILR3_DONE >> $OUT
