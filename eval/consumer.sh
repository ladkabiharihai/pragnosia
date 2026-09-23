#!/bin/bash
# VISION GOAL 6: TRAINS ON A CONSUMER GPU, RUNS INFERENCE WITHIN 4 GB. Measured, with the memory cap enforced by
# the process itself (torch's per-process memory fraction), so "it fits" means it ran under the cap, not that a
# number was read off a bigger card.
#   1. real training under an 8 GiB cap -- the class of an RTX 4060 / 3070 -- 400 steps, loss must fall
#   2. real training under a 16 GiB cap at a 16,384-token window -- an RTX 4080 / 4070 Ti Super
#   3. which widths fit a 24 GB card (RTX 4090 / 3090) at a 4,096-token window, measured, not extrapolated
# Inference within 4 GB is in eval/results.txt: 1.22 GiB decoding, 3.50 GiB holding a million tokens.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; OUT=eval/consumer.txt
echo "=== consumer GPU $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run () {  # cap ctx d steps label
  $P -u -m src.train --steps $4 --d $3 --ctx $2 --batch 1 --accum 4 --mixer rspin --lr 1.5e-3 --scan-chunk 64 \
     --chunks 8 --warm 50 --data /mnt/kv_cache/tokfull_v8192 --eval-every 100000 --save-every 100000 --cap $1 --seed 0 \
     --out /mnt/kv_cache/consumer_tmp > logs/consumer_$5.out 2>&1
  echo "--- $5: cap $1 GiB, d$3, window $2" >> $OUT
  grep -E "capped|params|reserved after capture|[0-9]+/$4 loss|final loss|END train|OutOfMemory" logs/consumer_$5.out \
     | grep -vE "spike" | sed -n '1,3p;$p' | cut -c1-120 >> $OUT
}
run 8  4096  768 400 8GiB
run 16 16384 768 200 16GiB
echo "--- widths that fit a 24 GB card, window 4,096, micro-batch 1" >> $OUT
for D in 768 1024 1536 2048; do
  $P eval/bench_step.py "$PWD" rspin --d $D --ctx 4096 --batch 1 2>&1 | grep -E "ktok/s|OutOfMemory" >> $OUT
done
rm -rf /mnt/kv_cache/consumer_tmp
