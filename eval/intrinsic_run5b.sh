#!/bin/bash
# RESUME OF run70 after the out-of-memory crash at step ~157,000 (a trial's deepen branch at the 34 GiB cap). The model's own
# decision at step 152,000 was GROW by widening (DECISION line in logs/train_grow70.out); the checkpoint was written one
# step before it was applied, so it is re-applied here (152000:w768) and the new stage's clock starts there. The rest of
# the plan (three more "W|d" decisions) is left to the model.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
echo "=== RESUMED $(date -u '+%Y-%m-%d %H:%M UTC') from step 152,000, re-applying the model's GROW (W) decision ===" >> logs/train_grow70.out
$P -u -m src.train --steps 450000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --grow-schedule 152000:w768 --resume-stage 152000,4000 --grow-auto "W|d,W|d,W|d" \
  --trial-steps 1000 --trial-frac 16 --deepen-init zero \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 8000 --eval-every 8000 --cap 34 --resume /mnt/kv_cache/run70_grow/seed152000.pt \
  --out /mnt/kv_cache/run70_grow >> logs/train_grow70.out 2>&1 && echo "TRAIN70_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/train_grow70.out
