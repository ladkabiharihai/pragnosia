#!/bin/bash
# END TO END: does the fused scan make whole training steps faster, not just one kernel? The same run with
# --fused-scan 0 and 1, 400 steps, card to itself, at two equal-token configurations (16,384 tokens a micro-
# step). Also proves the fused kernel survives CUDA-graph capture, which the trainer relies on.
set -u
cd "$(dirname "$0")/.."
OUT=eval/fusedbench.txt
until grep -q "8,192" eval/stateread.txt 2>/dev/null; do sleep 20; done
while pgrep -f "[s]rc.train" > /dev/null || pgrep -f "[s]rc.sft" > /dev/null; do sleep 20; done
echo "=== whole training steps, fused scan off vs on, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > $OUT
for cfg in "2048 8" "8192 2"; do
  set -- $cfg
  for F in 0 1; do
    .venv/bin/python -u -m src.train --steps 400 --d 768 --ctx $1 --batch $2 --accum 2 --mixer rspin --lr 1.5e-3 \
      --scan-chunk 64 --chunks 4 --warm 50 --data /mnt/kv_cache/tokfull_v8192 --eval-every 100000 --save-every 100000 \
      --cap 24 --seed 0 --fused-scan $F --out /mnt/kv_cache/fusedbench_tmp > logs/fusedbench_${1}_$F.out 2>&1
    echo "window $1 batch $2, fused $F: $(grep -E '400/400|200/400' logs/fusedbench_${1}_$F.out | tail -n 1 | grep -oE '[0-9]+ ktok/s|loss [0-9.]+' | tr '\n' ' ')  $(grep -oE 'reserved after capture|OutOfMemory|Error' logs/fusedbench_${1}_$F.out | head -n 1)" >> $OUT
  done
done
rm -rf /mnt/kv_cache/fusedbench_tmp
echo FUSED_DONE >> $OUT
