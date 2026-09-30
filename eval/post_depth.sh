#!/bin/bash
# POST-CROSSOVER DEPTH GROWTH, RE-RUN after the step-size carry fix (same checkpoint and configuration as
# eval/post_growth.sh). Two at a time for memory.
set -u
cd "$(dirname "$0")/.."
OUT=eval/post_growth.txt; echo "=== depth re-run after the d-carry fix, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
run() { N=$1; INIT=$2
  mkdir -p /mnt/kv_cache/pd_$N; cp /mnt/kv_cache/post_base/seed_sfp.pt /mnt/kv_cache/pd_$N/seed.pt
  .venv/bin/python -u -m src.train --steps 126000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp \
    --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 6 --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
    --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 --eval-every 1000 --save-every 100000 --cap 24 \
    --resume /mnt/kv_cache/pd_$N/seed.pt --out /mnt/kv_cache/pd_$N --grow-schedule 120000:d --deepen-init $INIT > logs/postd_$N.out 2>&1
  echo "$N (fixed): $(grep -oE 'DEEPENED.*loss [0-9.]+ -> [0-9.]+' logs/postd_$N.out | grep -oE 'loss [0-9.]+ -> [0-9.]+') | d after: $(grep -E '/126000' logs/postd_$N.out | head -1 | grep -oE 'd [0-9.e -]+$') | $(grep -oE 'eval step [0-9]+: BPB [0-9.]+' logs/postd_$N.out | sed 's/eval step //; s/: BPB /=/' | tr '\n' ' ') $(grep -oE 'Error.*' logs/postd_$N.out | head -1)" >> $OUT; }
run zero zero & run gate gate & wait
run copy copy
echo DEPTH_DONE >> $OUT
