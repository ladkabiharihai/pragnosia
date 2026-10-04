#!/bin/bash
# CAN THE CARRIER LEARN EXACT RETRIEVAL PAST THE ATTENTION WINDOW? (2026-10-03; PROGRESS.md) run82 retrieves only inside its
# 1,024-token attention window. Its needle data mostly sits inside that window. This arm trains on documents (2-8k tokens,
# inside one training window) whose every needle is >= 1,500 tokens before its question, so only the carrier can answer
# and the gradient reaches it. run82_final + 6,000 steps, otherwise exactly eval/onelong.sh's control (ol_ctrl), which used
# the ordinary needle data. Then needles by distance, passkey, the 24-passage context test, on both.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/farneedle.txt
echo "FAR-NEEDLE CURRICULUM vs control (ol_ctrl), from run82_final, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
mkdir -p /mnt/kv_cache/fn_far; rm -f /mnt/kv_cache/fn_far/ckpt*.pt /mnt/kv_cache/fn_far/resume_from.pt
cp /mnt/kv_cache/run82_final.pt /mnt/kv_cache/fn_far/seed.pt; : > logs/fn_far.out
eval/keep_training.sh logs/fn_far.out FN_DONE --steps 244000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin \
  --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --spike 1e9 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 \
  --fused-scan 1 --seed 11 --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokfarneedle --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 34 \
  --resume /mnt/kv_cache/fn_far/seed.pt --out /mnt/kv_cache/fn_far || exit 1
for ck in fn_far ol_ctrl; do
  echo "== $ck: $(grep -h 'eval step' logs/$( [ $ck = fn_far ] && echo fn_far || echo ol_ctrl ).out | tr '\n' ' ')" >> $O
  NEEDLE_PRE=256 NEEDLE_N=48 NEEDLE_DIST=512,1024,2048,4096,7000 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/$ck/ckpt.pt needle 2>&1 | grep distance >> $O
  PASSKEY_LENS=2048,8192 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/$ck/ckpt.pt passkey 2>&1 | grep -E "^[0-9]" >> $O
  MILLION_SAMPLES=24 $P eval/million.py /mnt/kv_cache/$ck/ckpt.pt 8192,32768 2>&1 | grep "tokens of context" >> $O
done
echo "FARNEEDLE_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
