#!/bin/bash
# THE 98M TEST (user, 2026-10-01: "98M is sufficient to test the model"). run70's own decision at step 152,000 (~10B
# tokens) was GROW by widening, 46.2M -> 97.7M. This continues that model at 97.7M with no further growth trials to
# step 184,000 (the end of the stage the model would next have reconsidered at: 152k + 32k), ~2.1B tokens at the new
# width, then the same pipeline as every lineage: 8,000 steps with in-context-lookup data, factual fine-tune,
# own-mistake abstention, full scorecard -> eval/scorecard_r98.txt. Schedule-free Prodigy everywhere (no rate).
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
echo "=== 98M TEST $(date -u '+%Y-%m-%d %H:%M UTC'): from step 152,000 re-applying the model's GROW (W) decision; no further trials, to step 184,000 ===" >> logs/train_grow70.out
$P -u -m src.train --steps 184000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim sfp --grow-schedule 152000:w768 \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 8000 --eval-every 8000 --cap 34 --resume /mnt/kv_cache/run70_grow/seed152000.pt \
  --out /mnt/kv_cache/run70_grow >> logs/train_grow70.out 2>&1 || { echo "R98_TRAIN_FAILED" >> logs/train_grow70.out; exit 1; }
echo "R98_TRAIN_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/train_grow70.out
cp /mnt/kv_cache/run70_grow/ckpt.pt /mnt/kv_cache/run70_98M.pt
mkdir -p /mnt/kv_cache/r98r; cp /mnt/kv_cache/run70_98M.pt /mnt/kv_cache/r98r/seed.pt
$P -u -m src.train --steps 192000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokrecall_v8192 --data2-frac 0.25 \
  --save-every 4000 --eval-every 4000 --cap 34 --resume /mnt/kv_cache/r98r/seed.pt --out /mnt/kv_cache/r98r > logs/train_recall98.out 2>&1 || exit 1
exec eval/eval_ckpt.sh /mnt/kv_cache/r98r/ckpt.pt r98
