#!/bin/bash
# VISION GOAL 7, FIRST HALF: CONTINUOUS GROWTH, measured as the experiment that decides whether growth is worth
# having. Growing a TRAINED model is not function-preserving in this architecture: measured 2026-09-23, run45
# widened 768 -> 1024 goes from loss 1.980 to 2.390 on real text, because LayerNorm over padded zeros shifts
# every normalisation once the residual stream has a non-zero mean. So the claim tested is the useful one:
#   A  512 wide for 4,000 steps, then GROWN to 768 for 4,000 more        (the growth path)
#   B  512 wide for 8,000 steps                                           (did growing help?)
#   C  768 wide from scratch for 8,000 steps                              (did growing save training?)
# Same data, seed, window, schedule shape and tokens per step. 32,768 tokens per step, 0.26B tokens per run.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; OUT=eval/growth.txt
T="--ctx 2048 --batch 4 --accum 4 --mixer rspin --lr 1.5e-3 --scan-chunk 64 --chunks 4 --warm 300 --data /mnt/kv_cache/tokfull_v8192 --eval-every 1000 --save-every 4000 --cap 24 --seed 0"
echo "=== growth experiment $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
$P -u -m src.train --steps 8000 --d 512 $T --grow-at 4000 --grow-to 768 --out /mnt/kv_cache/grow_A > logs/grow_A.out 2>&1
$P -u -m src.train --steps 8000 --d 512 $T --out /mnt/kv_cache/grow_B > logs/grow_B.out 2>&1
$P -u -m src.train --steps 8000 --d 768 $T --out /mnt/kv_cache/grow_C > logs/grow_C.out 2>&1
for R in A B C; do
  echo "--- $R" >> $OUT
  grep -E "params|GREW|eval step|final loss|END train" logs/grow_$R.out | cut -c1-120 >> $OUT
done
