#!/bin/bash
# MQAR past the attention window, with and without the short convolution, trained on the task alone (eval/mqar.py).
# d256, 4 layers (carriers 0-2, windowed attention 3), 2,048-token windows, 4,000 steps x 32,768 tokens, no rate.
set -u
cd "$(dirname "$0")/.."
C="--steps 4000 --d 256 --layers 4 --ctx 2048 --batch 16 --accum 1 --mixer grspin --norm rms --optim sfp
   --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 6 --data /mnt/kv_cache/tokmqar --save-every 4000 --eval-every 2000 --cap 12"
O=eval/mqar.txt; echo "MQAR A/B, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
for arm in noconv conv; do
  mkdir -p /mnt/kv_cache/mq_$arm; : > logs/mq_$arm.out
  X=""; [ $arm = conv ] && X="--short-conv 4"
  eval/keep_training.sh logs/mq_$arm.out MQ_DONE $C $X --out /mnt/kv_cache/mq_$arm || exit 1
  grep -h "params\|final loss" logs/mq_$arm.out >> $O
  .venv/bin/python eval/mqar.py test /mnt/kv_cache/mq_$arm/ckpt.pt 200 2>&1 | grep -v "^\[gpu\]" >> $O
done
echo "MQAR_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
