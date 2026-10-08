#!/bin/bash
# THE GROWN MEMORY LAYER, A/B (2026-10-05; PROGRESS.md). run83 + one carrier layer on top whose heads all hold for long and
# whose writes start CLOSED (train.py --grow-schedule <start>:d --deepen-init memory: grown IN THE RUN, every existing weight's
# optimizer state carried), vs run83 unchanged. Both: one shared step size + carried windows (the repair of 2026-10-04), 25%
# far-needle documents (every needle >= 1,500 tokens back), 6,000 steps, both resuming run83's optimizer. (The first version
# started both cold: the step size restarted at 6e-5 against run83's 8.1e-4 and neither arm could learn; stopped.)
# Then READ by distance (eval/carrier_probe.py retain), needles, the 24-passage context test, held-out.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/memlayer.txt
echo "GROWN MEMORY LAYER (writes closed by default) vs control, from run83, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
C="--steps 344000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp --attn-sliding 1
   --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --spike 1e9 --scan-chunk 64 --chunks 4
   --fused-scan 1 --seed 14 --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 /mnt/kv_cache/tokfarneedle --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 30"
for arm in mem ctrl; do
  mkdir -p /mnt/kv_cache/ml_$arm; rm -f /mnt/kv_cache/ml_$arm/ckpt*.pt /mnt/kv_cache/ml_$arm/resume_from.pt; : > logs/ml_$arm.out
  cp /mnt/kv_cache/run83_final.pt /mnt/kv_cache/ml_$arm/seed.pt
  X=""; [ $arm = mem ] && X="--grow-schedule 338000:d --deepen-init memory"
  eval/keep_training.sh logs/ml_$arm.out ML_DONE $C $X --resume /mnt/kv_cache/ml_$arm/seed.pt --out /mnt/kv_cache/ml_$arm || exit 1
done
for arm in mem ctrl; do
  echo "######## $arm: $(grep -h 'eval step' logs/ml_$arm.out | tr '\n' ' ')" >> $O
  $P eval/carrier_probe.py /mnt/kv_cache/ml_$arm/ckpt.pt retain 2>&1 | grep -vE "Warning|^\[gpu\]" | cut -c1-40 >> $O
  NEEDLE_DIST=1024,2048,4096,7000 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/ml_$arm/ckpt.pt needle 2>&1 | grep distance >> $O
  MILLION_SAMPLES=24 $P eval/million.py /mnt/kv_cache/ml_$arm/ckpt.pt 8192,32768 2>&1 | grep "tokens of context" >> $O
done
echo "MEMLAYER_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
