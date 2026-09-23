#!/bin/bash
# TRAINING THROUGHPUT, FAIRLY. The earlier table ran every window at micro-batch 1. The carrier's scan runs in
# parallel over batch x heads, so at a 512-token window and batch 1 it had 6 blocks of work for 132 cores and
# the GPU sat mostly idle; attention has no such dependence. That comparison measured how the benchmark was
# set up. Here every window carries the SAME 16,384 tokens per micro-step for both mixers: batch = 16384 / window.
# (It also keeps the carrier's per-chunk state at the size run33 trained with, clear of the compile fault.)
set -u
cd "$(dirname "$0")/.."
OUT=eval/fairbench.txt
while pgrep -f "[s]rc.sft" > /dev/null || pgrep -f "[s]rc.train" > /dev/null; do sleep 20; done
echo "=== equal tokens per micro-step (16,384), $(date -u '+%Y-%m-%d %H:%M UTC') ===" > $OUT
for T in 512 1024 2048 4096 8192 16384; do
  B=$((16384 / T))
  .venv/bin/python eval/bench_step.py "$PWD" rspin attn --d 768 --ctx $T --batch $B 2>&1 | grep -E "ktok/s|OutOfMemory" >> $OUT
done
echo FAIR_DONE >> $OUT
