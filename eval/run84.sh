#!/bin/bash
# RUN84 (2026-10-05): the main lineage, repaired. run83 learned the most (Q&A everyday 35.4%, facts 31/48, induction 99%) but
# its carrier drifted when the spectral group's own step size ratcheted. The repair that worked (eval/sharedd.txt, arm A):
# ONE shared step size (Prodigy as published) + carried windows (half of the micro-steps continue a stream, so a window that
# starts with a carried state is trained): long passages 1.26 -> 1.10 in 6,000 steps with general quality unchanged.
# 60,000 steps (~3.9B tokens), held-out every 4,000 (stop by hand if it rises), then the full pipeline -> scorecard_r84.
set -u
cd "$(dirname "$0")/.."
mkdir -p /mnt/kv_cache/run84; rm -f /mnt/kv_cache/run84/ckpt*.pt /mnt/kv_cache/run84/resume_from.pt
cp /mnt/kv_cache/run83_final.pt /mnt/kv_cache/run84/seed.pt
echo "=== RUN84 START $(date -u '+%Y-%m-%d %H:%M UTC') ===" > logs/train84.out
eval/keep_training.sh logs/train84.out TRAIN84_DONE --steps 398000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 \
  --carry-half 1 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 15 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34 \
  --resume /mnt/kv_cache/run84/seed.pt --out /mnt/kv_cache/run84 || exit 1
sed -i "s|logs/[a-z_0-9]*\.out >> \$OUT$|logs/train84.out >> \$OUT|" eval/scorecard.sh
cp /mnt/kv_cache/run84/ckpt.pt /mnt/kv_cache/run84_final.pt
exec eval/eval_ckpt.sh /mnt/kv_cache/run84_final.pt r84
