#!/bin/bash
# READING REAL TEXT FAR BACK (2026-10-07): the final base (far-lookup 375k) + 3,000 steps with a 5th retrieval source,
# far reading (eval/build_recall.py farread: SQuAD TRAIN paragraphs, 0.5k-16k tokens of prose, then their questions in
# other words), data2 25% over needles, lookup lists, far needles, far lookup, far reading; the step size held (3.30e-3).
# Then open-book SciQ at 8k / 32k, learning by reading, needles, bits/byte: does it replace 375k as the final base?
cd "$(dirname "$0")/.."
OUT=/mnt/kv_cache/far_read; mkdir -p $OUT; [ -f $OUT/seed.pt ] || cp /mnt/kv_cache/final_base_far.pt $OUT/seed.pt
L=logs/far_read.out; echo "=== FAR READ RUN $(date -u '+%Y-%m-%d %H:%M UTC') ===" > $L
( m0=0; while :; do sleep 60; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 20; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -q FAR_READ_DONE $L && break; done ) &
.venv/bin/python -u -m src.train --steps 378000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 26 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_far2 --data2-frac 0.25 \
  --save-every 1000 --eval-every 1000 --cap 24 --resume $OUT/seed.pt --out $OUT >> $L 2>&1
echo "FAR_READ_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $L
