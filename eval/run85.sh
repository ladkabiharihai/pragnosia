#!/bin/bash
# RUN85 (2026-10-05): THE LONG RUN TO THE MEETING (user: "the vision complete, far ahead"; "scorecard every 6 hours";
# "retrieval beyond 1k, also long generation"). From run84 (run83 + the carrier repair: one shared step size + carried
# windows). Everything the model decides for itself, judged on 32k-token streams read window by window (how it is used):
#   growth   at each doubling of its stage it trials R (double the attention window: exact retrieval reaches as far as the
#            window, run80 found needles 96% at 4k with 8k), W (widen one step) and d (double the depth) against staying
#   rate     schedule-free Prodigy, one shared step size, no schedule
#   training carried windows (streams of up to 131k tokens: the cross-window case and long continuation are trained)
#   data     main mix + 25% retrieval documents: lookup lists, needles, and needles >= 1,500 tokens back (only the carrier can
#            answer those: the signal that selective writing needs)
# Held-out every 4,000; checkpoints every 4,000; eval/periodic_eval.sh scores the latest checkpoint every 6 hours.
set -u
cd "$(dirname "$0")/.."
SEED=${1:-/mnt/kv_cache/run84/ckpt.pt}
mkdir -p /mnt/kv_cache/run85; rm -f /mnt/kv_cache/run85/ckpt*.pt /mnt/kv_cache/run85/resume_from.pt
cp $SEED /mnt/kv_cache/run85/seed.pt
S0=$(.venv/bin/python -c "import torch;print(torch.load('/mnt/kv_cache/run85/seed.pt',map_location='cpu',weights_only=False)['step'])")
echo "=== RUN85 START $(date -u '+%Y-%m-%d %H:%M UTC') from $SEED (step $S0) ===" > logs/train85.out
eval/keep_training.sh logs/train85.out TRAIN85_DONE --steps 2400000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 \
  --carry-half 1 --grow-auto "R|W|d,R|W|d,R|W|d,R|W|d,R|W|d,R|W|d" --trial-steps 1000 --trial-frac 16 \
  --resume-stage $S0,4000 --heldout-long 32768 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 16 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_mix --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34 \
  --resume /mnt/kv_cache/run85/seed.pt --out /mnt/kv_cache/run85
