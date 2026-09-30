#!/bin/bash
# THE DECISIVE GROWTH TEST, AFTER THE CROSSOVER. From run60's final 46M model (7.9B tokens; the 118M control shows a bigger
# model wins at equal compute beyond ~3B), converted to the fused optimizer with per-weight clocks (eval/psf2sfp.py).
# Real configuration. Held-out every 1,000 steps; compare at EQUAL COMPUTE (98M costs 2.1x a step, 81M 1.9x).
set -u
cd "$(dirname "$0")/.."
OUT=eval/post_growth.txt; echo "=== post-crossover growth test, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; STEPS=$2; shift 2
  mkdir -p /mnt/kv_cache/pg_$N; cp /mnt/kv_cache/post_base/seed_sfp.pt /mnt/kv_cache/pg_$N/seed.pt
  .venv/bin/python -u -m src.train --steps $STEPS --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp \
    --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 6 --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
    --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 --eval-every 1000 --save-every 100000 --cap 18 \
    --resume /mnt/kv_cache/pg_$N/seed.pt --out /mnt/kv_cache/pg_$N "$@" > logs/post_$N.out 2>&1
  echo "$N: $(grep -oE '(GREW|DEEPENED).*loss [0-9.]+ -> [0-9.]+' logs/post_$N.out | grep -oE 'loss [0-9.]+ -> [0-9.]+') | $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/post_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') $(grep -oE 'Error.*' logs/post_$N.out | head -1)" >> $OUT; }
run unchanged 133000 &
run widen 126000 --grow-schedule 120000:w768 &
run deepen 126000 --grow-schedule 120000:d --deepen-init gate &
wait; echo POST_DONE >> $OUT
