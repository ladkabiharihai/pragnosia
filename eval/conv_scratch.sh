#!/bin/bash
# CAN THE CARRIER RETRIEVE AT ALL, AND DOES A SHORT CONVOLUTION MAKE IT? (2026-10-02; PROGRESS.md) The late-added
# convolution never woke up (cold optimizer, step size ~3e-5), so this trains two small models FROM SCRATCH, identical
# except --short-conv 4: d384, 8 layers (attention at 3 and 7, window 512), 2,048-token windows, 20,000 steps x 16,384
# tokens, 25% lookup + needle documents. Then needles at 64..1,536 tokens before the question, prompts aligned to 512
# so the windowed attention is block-local as trained: past 512, only the carrier can answer.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
C="--steps 20000 --d 384 --layers 8 --ctx 2048 --batch 8 --accum 1 --mixer grspin --norm rms --optim sfp
   --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 5 --data /mnt/kv_cache/tokfull_v8192
   --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25
   --save-every 20000 --eval-every 5000 --cap 20"
for arm in noconv conv; do
  mkdir -p /mnt/kv_cache/cs_$arm; : > logs/cs_$arm.out
  X=""; [ $arm = conv ] && X="--short-conv 4"
  eval/keep_training.sh logs/cs_$arm.out CS_DONE $C $X --out /mnt/kv_cache/cs_$arm || exit 1
done
O=eval/conv_scratch.txt
{ echo "SHORT CONVOLUTION FROM SCRATCH, $(date -u '+%Y-%m-%d %H:%M UTC')"; grep -h "params\|eval step" logs/cs_noconv.out logs/cs_conv.out; } > $O
for arm in noconv conv; do NEEDLE_PRE=256 NEEDLE_ALIGN=512 NEEDLE_N=48 NEEDLE_DIST=64,256,768,1024,1536 INCONTEXT_GPT2=0 \
  $P eval/incontext.py /mnt/kv_cache/cs_$arm/ckpt.pt needle 2>&1 | grep -vE "Warning|^\[gpu\]|miss:" >> $O; done
echo "CONV_SCRATCH_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
