#!/bin/bash
# DOES GROWTH PAY OVER A LONG HORIZON? From rl_psf (12L d512, 30,000 steps), 20,000 more steps each: unchanged vs deepened
# to 24 layers by gated copies (the best operator at 4,000 steps). Held-out every 2,000; compare at EQUAL COMPUTE
# (the 81M model costs 1.88x per step).
set -u
cd "$(dirname "$0")/.."
OUT=eval/growth_long.txt; echo "=== long-horizon growth test, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; shift
  mkdir -p /mnt/kv_cache/gl_$N; cp /mnt/kv_cache/rl_psf/ckpt.pt /mnt/kv_cache/gl_$N/seed.pt
  .venv/bin/python -u -m src.train --steps 50000 --d 512 --layers 12 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms --optim psf \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 2 --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 2000 --save-every 100000 --cap 14 --resume /mnt/kv_cache/gl_$N/seed.pt --out /mnt/kv_cache/gl_$N "$@" > logs/growth_long_$N.out 2>&1
  echo "$N: $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/growth_long_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') $(grep -oE 'Error.*' logs/growth_long_$N.out | head -1)" >> $OUT; }
run unchanged &
run gate --grow-schedule 30000:d --deepen-init gate &
wait; echo GL_DONE >> $OUT
