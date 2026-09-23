#!/bin/bash
# END-TO-END TEST of the 12B growth schedule, compressed: the real shapes (grspin, RMSNorm, fused scan, windows
# 2,048 and 8,192 at 16,384 tokens a micro-step) through all four events in 1,400 steps (each stage has one clean 200-step speed window), at the real cap.
# Checks: loss unchanged across every event, training continues, stage 4 fits under the cap after the split.
set -u
cd "$(dirname "$0")/.."
.venv/bin/python -u -m src.train --steps 1400 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --lr 1.5e-3 --new-lr 1e-3 --warm 100 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --grow-schedule 300:w768,700:d,1100:a2,1100:w1024 --new-warm 100 \
  --data /mnt/kv_cache/tokfull_v8192 --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --eval-every 100000 --save-every 100000 --cap 34 --out /mnt/kv_cache/growtest_tmp > logs/growtest.out 2>&1
echo "exit $?" >> logs/growtest.out
