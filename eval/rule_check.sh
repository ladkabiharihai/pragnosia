#!/bin/bash
# VALIDATE THE MARGINAL-RETURNS GROWTH RULE against a known answer. The 118M control shows the bigger model overtakes
# the 46M run at equal compute near the 46M run's step 47,000 (~3B tokens). One forced trial (4,000 steps a branch,
# widen to 768, the real configuration) at two points of the 46M run:
#   step 28,000 (before the crossover)  -> the rule should say STAY
#   step 120,000 (well after it)         -> the rule should say GROW
set -u
cd "$(dirname "$0")/.."
OUT=eval/rule_check.txt; echo "=== growth-rule validation, $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $OUT
chk() { N=$1; CK=$2; STEPS=$3
  mkdir -p /mnt/kv_cache/rc_$N; cp $CK /mnt/kv_cache/rc_$N/seed.pt
  .venv/bin/python -u -m src.train --steps $STEPS --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim psf \
    --grow-auto w768 --trial-steps 4000 --trial-now 1 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 3 \
    --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
    --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
    --eval-every 100000 --save-every 100000 --cap 30 --resume /mnt/kv_cache/rc_$N/seed.pt --out /mnt/kv_cache/rc_$N > logs/rule_$N.out 2>&1
  echo "$N: $(grep -E 'TRIAL w768: marginal|DECISION' logs/rule_$N.out | cut -c14-300 | tr '\n' ' ') $(grep -oE 'Error.*' logs/rule_$N.out | head -1)" >> $OUT; }
chk before28k /mnt/kv_cache/run60_intrinsic/seed28000.pt 38000
chk after120k /mnt/kv_cache/run60_intrinsic/ckpt.pt 130000
echo RULE_DONE >> $OUT
