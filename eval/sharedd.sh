#!/bin/bash
# REPAIRING THE CARRIER AFTER run83's STEP-SIZE RATCHET (2026-10-04; PROGRESS.md). run83 learned more (Q&A everyday 35.4%,
# facts 31/48, induction 99%) but its carrier timescales drifted when the spectral group's own step size ratcheted 3x, and
# long-document prediction across a window boundary fell from 1.04 to 1.26 bits/byte. From run83_final, 6,000 steps:
#   A  --share-d 1 + --carry 16 --carry-groups 16 --carry-half 1  (one step size; window starts with a carried state trained)
#   B  --share-d 1                                                   (one step size only)
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/sharedd.txt
echo "SHARED STEP SIZE (+/- carried windows), from run83_final, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
C="--steps 344000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp --attn-sliding 1
   --attn-window 1024 --share-d 1 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 13
   --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 34"
for arm in A B; do
  mkdir -p /mnt/kv_cache/sd_$arm; rm -f /mnt/kv_cache/sd_$arm/ckpt*.pt /mnt/kv_cache/sd_$arm/resume_from.pt
  cp /mnt/kv_cache/run83_final.pt /mnt/kv_cache/sd_$arm/seed.pt; : > logs/sd_$arm.out
  if [ $arm = A ]; then X="--carry 16 --carry-groups 16 --carry-half 1"; else X="--ctx-mix 2048,8192"; fi
  eval/keep_training.sh logs/sd_$arm.out SD_DONE $C $X --resume /mnt/kv_cache/sd_$arm/seed.pt --out /mnt/kv_cache/sd_$arm || exit 1
done
for ck in sd_A/ckpt.pt sd_B/ckpt.pt; do
  echo "== $ck: $(grep -h 'eval step' logs/$(dirname $ck).out | tr '\n' ' ')" >> $O
  MILLION_SAMPLES=24 $P eval/million.py /mnt/kv_cache/$ck 8192,32768 2>&1 | grep "tokens of context" >> $O
  NEEDLE_DIST=256,1024 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/$ck needle 2>&1 | grep distance >> $O
done
echo "SHAREDD_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
