#!/bin/bash
# DOES STACKING MAKE GROWTH PAY? From the same trained model (rl_psf: 12L d512, 30,000 steps, held-out BPB 1.2572),
# 4,000 more steps each, same data and seed, schedule-free Prodigy:
#   unchanged | deepen to 24 layers, new layers zero-output (exact) | deepen to 24 layers by stacking copies
set -u
cd "$(dirname "$0")/.."
OUT=eval/stack_test.txt; echo "=== stacking test, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; shift
  mkdir -p /mnt/kv_cache/st_$N; cp /mnt/kv_cache/rl_psf/ckpt.pt /mnt/kv_cache/st_$N/seed.pt
  .venv/bin/python -u -m src.train --steps 34000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms --optim psf \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 2 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 1000 --save-every 100000 --cap 14 --resume /mnt/kv_cache/st_$N/seed.pt --out /mnt/kv_cache/st_$N "$@" > logs/stack_$N.out 2>&1
  echo "$N: $(grep -oE 'DEEPENED.*loss [0-9.]+ -> [0-9.]+' logs/stack_$N.out | grep -oE 'loss [0-9.]+ -> [0-9.]+') | BPB $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/stack_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') | params $(grep -oE '[0-9.]+M params' logs/stack_$N.out | tail -1) $(grep -oE 'Error.*' logs/stack_$N.out | head -1)" >> $OUT; }
run unchanged &
run zero --grow-schedule 30000:d --deepen-init zero &
run copy --grow-schedule 30000:d --deepen-init copy &
wait; echo STACK_DONE >> $OUT
