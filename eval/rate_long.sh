#!/bin/bash
# GOAL 8, THE LONG TEST the first intrinsic rate never got: 30,000 steps (491M tokens, ~11 tokens/param), not 6,000.
# 12L d512 grspin rms, window 2,048, 16,384 tokens a step, held-out BPB every 5,000 steps. Same seed and data.
#   tuned   the hand-set schedule: trust ratio 1.5e-3, 200 warm-up, cosine to 5% over exactly these 30,000 steps
#   psf     schedule-free Prodigy, published defaults: no rate, no schedule, no warm-up
#   hyper   the hypergradient from 1e-2, which collapsed over 120k steps at full scale
set -u
cd "$(dirname "$0")/.."
run() { N=$1; shift
  .venv/bin/python -u -m src.train --steps 30000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 5000 --save-every 100000 --cap 16 --out /mnt/kv_cache/rl_$N "$@" > logs/rate_long_$N.out 2>&1
  echo "$N: $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/rate_long_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') $(grep -oE 'DIVERGED|Error' logs/rate_long_$N.out | head -1)" >> eval/rate_long.txt; }
echo "=== 30,000-step test, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> eval/rate_long.txt
run tuned --lr 1.5e-3 --warm 200 &
run psf --optim psf &
run hyper --intrinsic-lr 1 --lr 1e-2 &
wait
echo RATE_LONG_DONE >> eval/rate_long.txt
