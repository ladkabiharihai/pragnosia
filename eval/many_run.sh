#!/bin/bash
# MANY KINDS OF FACT (2026-10-12). The lookup the model reads with is bound to the six fact kinds its builders wrote: on two kinds
# no builder writes (eval/incontext.py INCONTEXT_KINDS=new) the base and every self-improved round sit at chance (24-35%), with
# 10 facts in plain view, while the trained kinds reach 97% (eval/si_newkinds_v2.txt). A training phase from the latest
# checkpoint (598k, 168M) with ~381 kinds of fact (eval/build_recall.py RECALL_KINDS=many: lists, and lists followed by
# 1k-12k tokens of real prose; the test's kinds held out) at 12% of tokens, growth ON (the stage clock carried in the checkpoint
# decides when it trials), the step size held, 1 x 8, cap 32. Every 4,000 steps the held-out kinds and the trained kinds
# are measured (eval/many_probe.txt, side slot, cap 6) -- the run stops when the held-out kinds stop rising or time runs out.
# 2026-10-12 12:17 UTC: paused at 616k so final pipeline v3 has the card (round 1 labelling ran ~1.7x slower beside it);
# resumed from ckpt_s616000.pt when FINAL_PIPELINE_DONE appears, without probes (NOPROBE=1: pretraining alone did not move the
# held-out kinds in 8,000 steps; the answer-only fine-tune does)
# env: SEED STEPS OUT LOG WAITFILE WAITMARK WEIGHTS D2FRAC NOPROBE
cd "$(dirname "$0")/.."
SEED=${SEED:-/mnt/kv_cache/restore/ckpt_s598000.pt}; STEPS=${STEPS:-650000}; OUT=${OUT:-/mnt/kv_cache/many}; L=${LOG:-logs/many.out}
WAITFILE=${WAITFILE:-eval/openbook_si_v2.txt}; WAITMARK=${WAITMARK:-GUARD: peak}; D2FRAC=${D2FRAC:-0.32}; PR=eval/many_probe.txt
WEIGHTS=${WEIGHTS:-fineweb-edu:28,wikipedia:18,dclm:14,finemath:4,arxiv:4,manyrecall:6,manyfar:6,farread2:6,farread:1,farlookup2:2,farlookup:1,recall:1,needle:2,farneedle:2,arith:4,instr_math:1}
probe () {   # <ckpt> <label>: held-out kinds, then trained kinds; powered hard, 20 contexts per row
  for kd in new old; do
    echo "== $2 kinds=$kd $(date -u '+%H:%M UTC')" >> $PR
    INCONTEXT_KINDS=$kd INCONTEXT_TRIALS=20 INCONTEXT_GPT2=0 MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 .venv/bin/python -u eval/incontext.py $1 powered 2>&1 |
      grep -E "^(10|30|60|120) " >> $PR; done; }
until grep -q "$WAITMARK" $WAITFILE 2>/dev/null; do sleep 60; done
mkdir -p $OUT; cp $SEED $OUT/seed.pt
echo "=== MANY-KINDS RUN from $SEED to $STEPS $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $L
echo "MANY-KINDS PROBE: hard four-way lookup, chance 25%, n=200 per row (+-6); new = held-out kinds (rivers, guilds), old = capitals / engines" >> $PR
( m0=0; while :; do sleep 120; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 30; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -qE "MANY_DONE|MANY_FAILED" $L && break; done ) &
[ -z "${NOPROBE:-}" ] && ( probe $SEED seed-598k; done_=" "; while :; do sleep 300
  for c in $(ls $OUT/ckpt_s*.pt 2>/dev/null | sort -V); do s=$(basename $c .pt | sed 's/ckpt_s//')
    if [ $((s % 4000)) -eq 2000 ] && [[ "$done_" != *" $s "* ]]; then done_="$done_$s "; probe $c many-$s; fi; done
  grep -qE "MANY_DONE|MANY_FAILED" $L && break; done ) &
eval/keep_training.sh $L MANY_DONE --steps $STEPS --d 512 --layers 12 --ctx 8192 --batch 1 --accum 8 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 \
  --grow-auto "R|W|d,R|W|d,R|W|d,R|W|d,R|W|d,R|W|d" --trial-steps 1000 --trial-frac 16 --heldout-long 32768 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 36 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokmany --data2-frac $D2FRAC \
  --weights $WEIGHTS --save-every 2000 --eval-every 2000 --cap 32 --resume $OUT/seed.pt --out $OUT
