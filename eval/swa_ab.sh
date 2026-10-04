#!/bin/bash
# RETRIEVAL THROUGH ATTENTION, MADE CONSISTENT AND WIDER (2026-10-02; PROGRESS.md). Needles were found only inside the
# 512-token window, whose attention was block-local in training and something else at inference. Two arms from r98r
# (step 192,000, its own optimizer state), the same data (25% lookup + needle documents, seed 3), 6,000 steps each:
#   s512   a TRUE sliding window of 512 (the consistency fix alone)
#   s8192  a TRUE sliding window of 8,192 (KV cache at inference 3 layers x 8,192 x 768 x 2 x 2 bytes = 75 MB)
# Then needles by distance, passkey and powered lookup on both.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
C="--steps 198000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp
   --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 3 --attn-sliding 1
   --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 34"
O=eval/swa_ab.txt; echo "SLIDING-WINDOW ATTENTION A/B, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
for W in 512 8192; do
  mkdir -p /mnt/kv_cache/swa_$W; cp /mnt/kv_cache/r98r/ckpt.pt /mnt/kv_cache/swa_$W/seed.pt; : > logs/swa_$W.out
  eval/keep_training.sh logs/swa_$W.out SWA_DONE $C --attn-window $W --resume /mnt/kv_cache/swa_$W/seed.pt --out /mnt/kv_cache/swa_$W || exit 1
done
for W in 512 8192; do
  echo "######## window $W: $(grep -h 'eval step\|captured' logs/swa_$W.out | tr '\n' ' ')" >> $O
  NEEDLE_DIST=64,256,1024,4096,7000,16384 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/swa_$W/ckpt.pt needle 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O
  PASSKEY_LENS=512,2048,8192,32768 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/swa_$W/ckpt.pt passkey 2>&1 | grep -vE "Warning|^\[gpu\]|wanted" >> $O
  INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/swa_$W/ckpt.pt powered 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O
done
echo "SWA_AB_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
