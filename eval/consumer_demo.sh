#!/bin/bash
# GOAL 6, TRAINING ON A CONSUMER GPU (2026-10-07): the flagship's exact training recipe (98M, carried 8,192-token windows,
# recall 16, schedule-free Prodigy, batch 2 x 4 accumulation) under a 22 GiB cap -- an RTX 4090 has 24 GB -- for 1,000 steps
# from the final base; the process's real GPU memory is sampled from nvidia-smi alongside. Output in its own directory.
cd "$(dirname "$0")/.."
OUT=/mnt/kv_cache/consumer_demo; L=logs/consumer_demo.out
echo "=== CONSUMER-GPU TRAINING DEMO $(date -u '+%Y-%m-%d %H:%M UTC'): cap 22 GiB ===" > $L
.venv/bin/python -u -m src.train --steps 373000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --recall 16 \
  --recall-gated 0 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 23 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_mix --data2-frac 0.25 \
  --save-every 100000 --eval-every 100000 --cap 22 --resume $OUT/seed.pt --out $OUT >> $L 2>&1 &
TP=$!; PEAK=0
while kill -0 $TP 2>/dev/null; do
  u=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$TP '$1==p {print $2}')
  [ -n "$u" ] && [ "$u" -gt "$PEAK" ] && PEAK=$u; sleep 5
done
echo "CONSUMER_DEMO_DONE $(date -u '+%Y-%m-%d %H:%M UTC'): peak process GPU memory $PEAK MiB (nvidia-smi)" >> $L
