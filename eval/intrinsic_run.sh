#!/bin/bash
# THE INTRINSIC RUN (goals 7 + 8 at full scale): no hand-set rate, no schedule, no growth steps.
#   rate      the optimizer's own (hypergradient over the displacement horizon), starting at 3e-3; new layers start
#             at 1e-4 and the hypergradient raises them
#   growth    the model grows when its rate has annealed one e-fold below the stage start, in the order
#             widen to 768, double the depth, split the micro-step + widen to 1024 (split keeps 34 GiB)
#   data      as run52: 60% prose + 40% Wikipedia with the Q&A set mixed in from the first step
#   length    87,000 updates x 65,536 tokens = 5.7B tokens; checkpoints every 4,000
# Then the factual fine-tune and the same Q&A test as run52/run46, against GPT-2 and SmolLM2.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
mkdir -p /mnt/kv_cache/run56_intrinsic
$P -u -m src.train --steps 87000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --intrinsic-lr 1 --lr 3e-3 --new-lr 1e-4 --grow-auto w768,d,a2+w1024 \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 4000 --eval-every 4000 --cap 34 --out /mnt/kv_cache/run56_intrinsic > logs/train_intrinsic.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run56_intrinsic/ckpt.pt --out /mnt/kv_cache/run57_sft --data /mnt/kv_cache/sft_factual_v8192 \
  --epochs 2 --cap 30 > logs/sft_intrinsic.out 2>&1
echo "=== run56_intrinsic + factual fine-tune, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/qa5.txt
$P -u eval/qa.py /mnt/kv_cache/run57_sft/ckpt.pt=ours-intrinsic /mnt/kv_cache/run46_sft/ckpt.pt=ours-12L-previous-best 2>&1 \
  | grep --line-buffered -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/qa5.txt
echo INTRINSIC_QA_DONE >> eval/qa5.txt
