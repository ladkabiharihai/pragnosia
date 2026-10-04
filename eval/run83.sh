#!/bin/bash
# RUN83 (2026-10-03): run82 (97.7M, its own 1,024-token reach chosen on long streams; no context rot to 1M) trained LONGER,
# the size held at 98M (user: close the gaps before scaling). The remaining gaps against GPT-2 and SmolLM2 are facts and
# knowledge (facts 27 vs 33/48; Q&A far behind SmolLM2's 2T tokens), which come from data: run82 has seen ~16B tokens.
# 100,000 more steps (~6.5B tokens), reach still decided by its own trials on 32k carried streams, held-out every 8,000,
# then the vision pipeline -> eval/scorecard_r83.txt.
set -u
cd "$(dirname "$0")/.."
mkdir -p /mnt/kv_cache/run83; rm -f /mnt/kv_cache/run83/ckpt*.pt /mnt/kv_cache/run83/resume_from.pt
cp /mnt/kv_cache/run82_final.pt /mnt/kv_cache/run83/seed.pt
echo "=== RUN83 START $(date -u '+%Y-%m-%d %H:%M UTC') ===" > logs/train83.out
eval/keep_training.sh logs/train83.out TRAIN83_DONE --steps 338000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --trial-steps 1000 --trial-frac 16 \
  --heldout-long 32768 --spike 1e9 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 12 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 8000 --eval-every 8000 --cap 34 \
  --grow-auto "R,R,R,R" --resume /mnt/kv_cache/run83/seed.pt --out /mnt/kv_cache/run83 || exit 1
sed -i "s|logs/train8[0-9].out|logs/train83.out|" eval/scorecard.sh
cp /mnt/kv_cache/run83/ckpt.pt /mnt/kv_cache/run83_final.pt
exec eval/eval_ckpt.sh /mnt/kv_cache/run83_final.pt r83
