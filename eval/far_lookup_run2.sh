#!/bin/bash
# LEARNING BY READING THAT LASTS, PART 2 (2026-10-07): 3,000 more steps from far_lookup's 375k (8k 81%, 32k 70%). Part 1: the final base (the A/B recall arm, rc1) + 3,000 steps with far-lookup documents
# (a list of facts, 1k-12k tokens of prose, then questions) as a 4th retrieval source (data2 25%: needles, lookup lists, far
# needles, far lookup); its own step size held (3.30e-3, under which recall was learned); held-out every 1,000, every
# checkpoint kept. Then eval/learn_by_reading.py at 8k / 32k, needles, bits/byte: does it replace rc1 as the final base?
cd "$(dirname "$0")/.."
OUT=/mnt/kv_cache/far_lookup2   # seed.pt copied in beforehand (far_lookup ckpt_s375000)
L=logs/far_lookup2.out; echo "=== FAR LOOKUP RUN $(date -u '+%Y-%m-%d %H:%M UTC') ===" > $L
( m0=0; while :; do sleep 60; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 20; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -q FAR_LOOKUP2_DONE $L && break; done ) &
.venv/bin/python -u -m src.train --steps 378000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 25 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_far --data2-frac 0.25 \
  --save-every 1000 --eval-every 1000 --cap 30 --resume $OUT/seed.pt --out $OUT >> $L 2>&1
echo "FAR_LOOKUP2_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $L
