#!/bin/bash
# THE LONG INTRINSIC RUN (goals 7 + 8, and the model for goals 1, 3-5, 9). Nothing about its training is hand-set:
#   step size   schedule-free Prodigy, fused (--optim sfp): no rate, no schedule, no warm-up; matched the package over 30k
#               steps (1.2611 vs 1.2572, both beating the tuned schedule's 1.2720) at 437-455 ktok/s
#   averaging   one schedule-free clock per weight, so capacity added by growth switches on at once
#   size        starts at 12 layers x 512 (46M) and grows only when a TRIAL shows growing pays: at each doubling of the
#               stage's length it trains every option of the next plan entry (W = widen one step, d = double the depth)
#               from the same snapshot on the same batches, and keeps the best only if it turns compute into held-out
#               loss faster at the margin than staying, and is not behind. No size or growth step is written down.
#   memory      if a grown model does not fit the 34 GiB cap, the micro-step splits itself (same tokens per update)
#   data        as run60: 60% prose + 40% Wikipedia with the Q&A set mixed in from the first step
#   length      GROW_STEPS x 65,536 tokens; checkpoints every 8,000 updates
# Then: recall training, factual fine-tune, own-mistake abstention, full vision scorecard (eval/after_intrinsic5.sh).
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
STEPS=${GROW_STEPS:-450000}
PLAN=${GROW_PLAN:-"W|d,W|d,W|d,W|d"}
mkdir -p /mnt/kv_cache/run70_grow
$P -u -m src.train --steps $STEPS --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --grow-auto "$PLAN" --trial-steps 1000 --trial-frac 8 --deepen-init zero \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 8000 --eval-every 8000 --cap 34 --out /mnt/kv_cache/run70_grow > logs/train_grow70.out 2>&1
echo "TRAIN70_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/train_grow70.out
