#!/bin/bash
# RESTORE READING (2026-10-11). The knowledge + arithmetic run (168M, grown by its own trial) beat its 382k start on general text,
# facts, arithmetic, needles and open-book, but learning by reading invented facts fell 87 / 89 -> ~60 at 8k / 32k (its mix held
# retrieval at 11.5%). A training phase with retrieval raised, growth ON (the stage clock carried in the checkpoint decides when
# it trials), the step size held, 1 x 8, cap 32.
#   first (2026-10-11 08:32 UTC): 590k -> 598k, retrieval 30%: reading 61 / 56 -> 59 / 65 (+7, not enough)
#   second (armed 12:45 UTC): 598k -> 620k, retrieval 32% with far lookup (the test's own kind of document) raised to 10%, after
#   the final base's evaluations free the trainer slot (FINAL_BASE_EVALS_DONE)
# env: SEED STEPS OUT LOG WAITFILE WAITMARK WEIGHTS D2FRAC
cd "$(dirname "$0")/.."
SEED=${SEED:-/mnt/kv_cache/restore/ckpt_s598000.pt}; STEPS=${STEPS:-620000}; OUT=${OUT:-/mnt/kv_cache/restore2}; L=${LOG:-logs/restore2.out}
WAITFILE=${WAITFILE:-eval/final_base_v2.txt}; WAITMARK=${WAITMARK:-FINAL_BASE_EVALS_DONE}; D2FRAC=${D2FRAC:-0.32}
WEIGHTS=${WEIGHTS:-fineweb-edu:28,wikipedia:18,dclm:14,finemath:4,arxiv:4,farlookup2:6,farlookup:4,farread2:7,farread:1,recall:4,needle:3,farneedle:3,arith:3,instr_math:1}
until grep -q "$WAITMARK" $WAITFILE 2>/dev/null; do sleep 60; done
mkdir -p $OUT; cp $SEED $OUT/seed.pt
echo "=== READING RESTORE RUN from $SEED to $STEPS $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $L
( m0=0; while :; do sleep 120; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 30; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -qE "RESTORE_DONE|RESTORE_FAILED" $L && break; done ) &
( done_=" "; while :; do sleep 300                         # reading every 4,000 steps (side slot, under the lock)
  for c in $(ls $OUT/ckpt_s*.pt 2>/dev/null | sort -V); do s=$(basename $c .pt | sed 's/ckpt_s//')
    if [ $((s % 4000)) -eq 2000 ] && [[ "$done_" != *" $s "* ]]; then done_="$done_$s "
      MEMBRAE_CAP_MAX=6 N_Q=100 DIST=8192,32768 eval/guarded.sh 7 .venv/bin/python -u eval/learn_by_reading.py $c=restore2-$s 2>&1 | grep -E "^restore2" >> eval/restore2_reading.txt; fi; done
  grep -qE "RESTORE_DONE|RESTORE_FAILED" $L && break; done ) &
eval/keep_training.sh $L RESTORE_DONE --steps $STEPS --d 512 --layers 12 --ctx 8192 --batch 1 --accum 8 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 \
  --grow-auto "R|W|d,R|W|d,R|W|d,R|W|d,R|W|d,R|W|d" --trial-steps 1000 --trial-frac 16 --heldout-long 32768 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 35 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokrestore --data2-frac $D2FRAC \
  --weights $WEIGHTS --save-every 2000 --eval-every 2000 --cap 32 --resume $OUT/seed.pt --out $OUT
