#!/bin/bash
# STAGE 4 AS IT WILL RUN: 24 layers x d1024 grspin rms, micro-step split in two (batch 4 at 2,048 / 1 at 8,192,
# accumulation 8 = 65,536 tokens an update), under the production-safe 34 GiB cap. Speed from the 400-step line.
set -u
cd "$(dirname "$0")/.."
.venv/bin/python -u -m src.train --steps 400 --d 1024 --layers 24 --ctx 8192 --batch 1 --accum 8 --mixer grspin --norm rms \
  --lr 1.5e-3 --warm 50 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --eval-every 100000 --save-every 100000 --cap 34 --out /mnt/kv_cache/stage4_tmp > logs/stage4.out 2>&1
echo "L24 d1024 split (B4@2048/B1@8192, accum 8), cap 34: $(grep -E '400/400' logs/stage4.out | grep -oE '[0-9]+ ktok/s') $(grep -oE '[0-9.]+ GiB reserved after capture|OutOfMemory' logs/stage4.out | head -1)" >> eval/stages.txt
rm -rf /mnt/kv_cache/stage4_tmp
