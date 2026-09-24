#!/bin/bash
# GOAL 8, THE LEARNING RATE: does the optimizer's own rate (hypergradient over the displacement horizon) match
# the hand-tuned schedule at equal tokens, and does it forget where it started? 12L d512 grspin rms, window
# 2,048, 16,384 tokens a step, 6,000 steps (98M tokens), held-out BPB at the end. Same seed and data for all.
set -u
cd "$(dirname "$0")/.."
OUT=eval/intrinsic_lr.txt; echo "=== intrinsic rate vs tuned schedule, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; shift
  .venv/bin/python -u -m src.train --steps 6000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 6000 --save-every 100000 --cap 20 --out /mnt/kv_cache/ilr_tmp "$@" > logs/ilr_$N.out 2>&1
  echo "$N: loss@2k/4k/6k $(grep -E '\] (2000|4000|6000)/6000' logs/ilr_$N.out | grep -oE 'loss [0-9.]+' | cut -d' ' -f2 | tr '\n' ' ')| held-out $(grep -oE 'BPB [0-9.]+' logs/ilr_$N.out | tail -1) | final rates $(grep -E '6000/6000' logs/ilr_$N.out | grep -oE 'rates .*' ) $(grep -oE 'DIVERGED|Error' logs/ilr_$N.out | head -1)" >> $OUT; }
run tuned-cosine --lr 1.5e-3 --warm 200
run intrinsic-from-1e-4 --intrinsic-lr 1 --lr 1e-4
run intrinsic-from-3e-3 --intrinsic-lr 1 --lr 3e-3
run intrinsic-beta0.05 --intrinsic-lr 1 --lr 1e-4 --hyper-beta 0.05
rm -rf /mnt/kv_cache/ilr_tmp; echo ILR_DONE >> $OUT
