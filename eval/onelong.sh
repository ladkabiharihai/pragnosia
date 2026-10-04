#!/bin/bash
# ONE LONG-RANGE ATTENTION LAYER (2026-10-03; PROGRESS.md). run82 chose a 1,024 window (all three attention layers): no
# context rot to 131k, but exact retrieval only to ~1k. run80's 8,192 on all three: retrieval to ~7k, but +0.034 at 32k.
# Can ONE layer at 8,192 (layer 7) restore retrieval while the other two stay short and the carrier keeps carrying?
# Two arms from run82_final, 6,000 steps, same data and seed: long7 (--set-window 7:8192) and ctrl (unchanged).
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/onelong.txt
echo "ONE LONG-RANGE LAYER vs control, from run82_final, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
C="--steps 244000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp
   --attn-sliding 1 --attn-window 1024 --spike 1e9 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 11
   --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 34"
for arm in long7 ctrl; do
  mkdir -p /mnt/kv_cache/ol_$arm; rm -f /mnt/kv_cache/ol_$arm/ckpt*.pt /mnt/kv_cache/ol_$arm/resume_from.pt
  cp /mnt/kv_cache/run82_final.pt /mnt/kv_cache/ol_$arm/seed.pt; : > logs/ol_$arm.out
  X=""; [ $arm = long7 ] && X="--set-window 7:8192"
  eval/keep_training.sh logs/ol_$arm.out OL_DONE $C $X --resume /mnt/kv_cache/ol_$arm/seed.pt --out /mnt/kv_cache/ol_$arm || exit 1
done
for arm in long7 ctrl; do
  echo "== $arm: $(grep -h 'eval step\|windows set' logs/ol_$arm.out | tr '\n' ' ')" >> $O
  MILLION_SAMPLES=24 $P eval/million.py /mnt/kv_cache/ol_$arm/ckpt.pt 8192,32768,131072 2>&1 | grep "tokens of context" >> $O
  NEEDLE_DIST=256,1024,4096,7000 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/ol_$arm/ckpt.pt needle 2>&1 | grep distance >> $O
  PASSKEY_LENS=2048,8192 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/ol_$arm/ckpt.pt passkey 2>&1 | grep -E "^[0-9]" >> $O
done
echo "ONELONG_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
