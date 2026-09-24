#!/bin/bash
# GOAL 7 + 8 TOGETHER: the model decides WHEN to grow. Intrinsic rate from a high start; each time it has annealed
# one e-fold the model is saturated at its size, grows by the next event (widen to 768, then double the depth),
# and its rates restart. 12L d512 start, window 2,048, 16,384 tokens a step, 10,000 steps. Mechanics test:
# events fire on their own, every event exact, training continues, no divergence.
set -u
cd "$(dirname "$0")/.."

.venv/bin/python -u -m src.train --steps 10000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms \
  --intrinsic-lr 1 --lr 3e-3 --new-lr 1e-4 --grow-auto w768,d --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --eval-every 10000 --save-every 100000 --cap 24 --out /mnt/kv_cache/autogrow_tmp > logs/autogrow3.out 2>&1
echo "exit $?" >> logs/autogrow3.out
