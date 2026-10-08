#!/bin/bash
# THE RECALL A/B, READ (2026-10-05): both arms (4,000 steps from run85's 368,000 checkpoint; rc1 trained with --recall 16,
# rc0 without), each measured with recall off (k=0) and on (k=16, the whole past kept): in-distribution needles 1k-64k and
# the loss on 8 long streams of the main mix by position. Waits for the A/B to finish; runs beside run85 (capped, light).
set -u
cd "$(dirname "$0")/.."
until grep -q RECALL_AB_DONE logs/recall_ab_0.out 2>/dev/null; do sleep 60; done
O=eval/recall_ab.txt
echo "RECALL A/B, read $(date -u '+%Y-%m-%d %H:%M UTC'): rc1 = 4,000 steps WITH recall 16, rc0 = the same 4,000 steps without; both from run85 step 368,000 (window 2,048)" > $O
for arm in 1 0; do
  echo "################ rc$arm" >> $O
  CAP=8 RECALL=0,16 NEEDLE_N=24 NEEDLE_DIST=1024,2048,4096,8192,16384,32768,65536 .venv/bin/python eval/recall_test.py /mnt/kv_cache/rc$arm/ckpt.pt needle 2>&1 | grep --line-buffered -vE "Warning|^\[gpu\]" >> $O
  CAP=8 RECALL=0,16 STREAM_N=8 .venv/bin/python eval/recall_test.py /mnt/kv_cache/rc$arm/ckpt.pt stream 2>&1 | grep --line-buffered -vE "Warning|^\[gpu\]" >> $O
done
grep -E "eval step|END train|FAILED" logs/recall_ab_1.out logs/recall_ab_0.out | sed 's/^/  /' >> $O
echo "RECALL_AB_EVAL_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
