#!/bin/bash
# RUN81 (2026-10-03): run80 (97.7M, its own 8,192-token reach) continued with CARRIED-STATE training, the configuration
# that helped (carry80b: half fresh windows, 16 rotating stream groups, streams of up to 131k tokens, schedule-free step
# size free). 40,000 steps (~2.6B tokens), held-out every 4,000 to see whether its small general-text cost settles or
# grows, then the vision pipeline (fine-tune, abstention, scorecard) -> eval/scorecard_r81.txt.
set -u
cd "$(dirname "$0")/.."
mkdir -p /mnt/kv_cache/run81; rm -f /mnt/kv_cache/run81/ckpt*.pt /mnt/kv_cache/run81/resume_from.pt
cp /mnt/kv_cache/run80_final.pt /mnt/kv_cache/run81/seed.pt
echo "=== RUN81 START $(date -u '+%Y-%m-%d %H:%M UTC') ===" > logs/train81.out
eval/keep_training.sh logs/train81.out TRAIN81_DONE --steps 278000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 8192 --carry 16 --carry-groups 16 --carry-half 1 \
  --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 8 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34 \
  --resume /mnt/kv_cache/run81/seed.pt --out /mnt/kv_cache/run81 || exit 1
sed -i 's#logs/train80.out#logs/train81.out#' eval/scorecard.sh
cp /mnt/kv_cache/run81/ckpt.pt /mnt/kv_cache/run81_final.pt
exec eval/eval_ckpt.sh /mnt/kv_cache/run81_final.pt r81
