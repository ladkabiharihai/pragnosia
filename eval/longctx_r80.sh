#!/bin/bash
# PAST THE TRAINING WINDOW ON THE SLIDING-8K MODEL (2026-10-03). With the masked-row bug fixed, block-local models are flat
# from 8k to 131k of context (-0.0007 +- 0.0006), but run80 (sliding attention over 8,192, chosen by itself) is worse at
# 32k than at 8k by +0.0414 +- 0.0152. Its carrier never held more than 8,192 tokens of history in training. Two arms from
# run80_final, 6,000 steps, same data and seed, no skip guard:
#   ctrl   windows 2,048 / 8,192 (as run80)
#   long   windows 2,048 / 8,192 / 32,768 on the 8,192-entry position table (batch 1 x accumulation 2 = the same tokens)
# Then the paired 24-passage test at 8k / 32k / 131k on both, and the needles (attention must be unharmed).
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
for arm in ctrl long; do
  mkdir -p /mnt/kv_cache/lc80_$arm; rm -f /mnt/kv_cache/lc80_$arm/ckpt*.pt /mnt/kv_cache/lc80_$arm/resume_from.pt
  cp /mnt/kv_cache/run80_final.pt /mnt/kv_cache/lc80_$arm/seed.pt; : > logs/lc80_$arm.out
  if [ $arm = ctrl ]; then W="--ctx-mix 2048,8192 --batch 2 --accum 4"; else W="--ctx-mix 2048,8192,32768 --pos-table 8192 --batch 1 --accum 2"; fi
  eval/keep_training.sh logs/lc80_$arm.out LC80_DONE --steps 244000 --d 512 --layers 12 --ctx 8192 $W --mixer grspin --norm rms \
    --optim sfp --attn-sliding 1 --attn-window 8192 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 7 \
    --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
    --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 34 \
    --resume /mnt/kv_cache/lc80_$arm/seed.pt --out /mnt/kv_cache/lc80_$arm || exit 1
done
O=eval/longctx_r80.txt; echo "PAST THE TRAINING WINDOW, sliding-8k model, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
for arm in ctrl long; do
  echo "== $arm: $(grep -h 'eval step\|attention from' logs/lc80_$arm.out | tr '\n' ' ')" >> $O
  MILLION_SAMPLES=24 $P eval/million.py /mnt/kv_cache/lc80_$arm/ckpt.pt 8192,32768,131072 2>&1 | grep "tokens of context" >> $O
  NEEDLE_DIST=1024,4096,7000 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/lc80_$arm/ckpt.pt needle 2>&1 | grep -E "distance" >> $O
done
echo "LONGCTX_R80_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
