#!/bin/bash
# THE LONG-SPAN PHASE (2026-10-08, approved by the user): recall made long-stream prediction worse past ~100k tokens (+0.025
# bits/byte at 131k, 8 passages) and real-text reading at 32k stayed at closed-book level, because training kept only ONE
# window (8,192 keys) in recall's store: the model never saw a large store, nor a paragraph more than ~16k back in it. Here
# training keeps 65,536 keys (--recall-span; a zero-padded fixed buffer, the pads masked), 2 stream groups (to fit the
# store), and a 6th retrieval source: SQuAD-train paragraphs 2k-56k tokens before their questions (farread_long).
# From the final base (far-read 382k) + 3,000 steps, step size held. Adopt only if open-book at 32k improves, needles hold
# and 131k prediction stops degrading.
cd "$(dirname "$0")/.."
OUT=/mnt/kv_cache/far_long; mkdir -p $OUT; [ -f $OUT/seed.pt ] || cp /mnt/kv_cache/final_base_fr382.pt $OUT/seed.pt
L=logs/far_long.out; echo "=== FAR LONG RUN $(date -u '+%Y-%m-%d %H:%M UTC') ===" > $L
( m0=0; while :; do sleep 60; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 20; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -q FAR_LONG_DONE $L && break; done ) &
.venv/bin/python -u -m src.train --steps 385000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 2 --carry-half 1 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 28 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_far3 --data2-frac 0.25 \
  --recall-span 65536 --save-every 1000 --eval-every 1000 --cap 32 --resume $OUT/seed.pt --out $OUT >> $L 2>&1
echo "FAR_LONG_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $L
