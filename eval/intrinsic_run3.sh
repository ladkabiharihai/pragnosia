#!/bin/bash
# THE INTRINSIC RUN (goals 7 + 8 at full scale): no hand-set rate, no schedule, no growth steps.
#   rate      NONE: schedule-free Prodigy at its published defaults (no rate, no schedule, no warm-up); beat the
#             tuned schedule over 30k steps (eval/rate_long.txt). The hypergradient it replaces collapsed (run58).
#   growth    DECIDED BY TRIAL: each time the stage's length doubles the model trains a grown copy 1,000 steps and
#             an unchanged one 1,000 steps on the same batches, and grows only if growing bought more held-out loss
#             per unit compute. Candidate order: widen to 768, double the depth, split micro-step + widen to 1024.
#             It may stop at any size.
#   data      as run52: 60% prose + 40% Wikipedia with the Q&A set mixed in from the first step
#   length    120,000 updates x 65,536 tokens = 7.9B tokens; checkpoints every 4,000
# Then the factual fine-tune and the same Q&A test as run52/run46, against GPT-2 and SmolLM2.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
mkdir -p /mnt/kv_cache/run60_intrinsic
$P -u -m src.train --steps 120000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --optim psf --grow-auto w768,d,a2+w1024 --trial-steps 1000 \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 4000 --eval-every 4000 --cap 34 --out /mnt/kv_cache/run60_intrinsic > logs/train_intrinsic3.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run60_intrinsic/ckpt.pt --out /mnt/kv_cache/run61_sft --data /mnt/kv_cache/sft_factual_v8192 --optim psf \
  --epochs 2 --cap 30 > logs/sft_intrinsic3.out 2>&1
echo "=== run60_intrinsic + factual fine-tune, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/qa7.txt
$P -u eval/qa.py /mnt/kv_cache/run61_sft/ckpt.pt=ours-intrinsic /mnt/kv_cache/run46_sft/ckpt.pt=ours-12L-previous-best 2>&1 \
  | grep --line-buffered -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/qa7.txt
echo INTRINSIC3_QA_DONE >> eval/qa7.txt
