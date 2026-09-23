#!/bin/bash
# SCALE, BY GROWTH. run47 (12 layers, knowledge-trained) doubled in depth to 24 layers, 205M parameters, with the
# function preserved exactly (loss 1.949016 -> 1.949016), then trained on the MIXED recipe: the published finding
# (Allen-Zhu & Li 2024) is that knowledge is far more answerable when Q&A-style data is mixed into pretraining
# than when it is fine-tuned on afterwards -- which is the recipe that just failed to move facts (47.9% -> 45.8%).
#   data: general prose (web, educational, Wikipedia) plus, at 50%, a file of 1.5B Wikipedia tokens with the
#         46M-token Q&A set three times in it -> ~46% Wikipedia, ~50% prose, ~4% Q&A, each pair seen ~3 times
#   old layers: trust-ratio steps, peak 3e-4 (a re-warm, below the original 1.5e-3); new layers: absolute steps,
#   peak 1e-3; cosine to 5%; fused scan; windows 2,048 and 8,192 at 16,384 tokens a micro-step
#   38,000 steps x 65,536 tokens = 2.49B tokens
# then the factual Q&A fine-tune on top and the same Q&A test, against the 12-layer run48 and the baselines.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
$P -u -m src.train --steps 38000 --d 768 --layers 24 --ctx 32768 --batch 2 --accum 4 --mixer rspin \
  --lr 3e-4 --new-lr 1e-3 --warm 500 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia \
  --data2 /mnt/kv_cache/tokmixqa_v8192 --data2-frac 0.5 \
  --save-every 2000 --eval-every 2000 --cap 30 --resume /mnt/kv_cache/run49_deep/seed.pt \
  --out /mnt/kv_cache/run49_deep > logs/train_deep.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run49_deep/ckpt.pt --out /mnt/kv_cache/run50_sft --data /mnt/kv_cache/sft_factual_v8192 \
  --epochs 2 --cap 20 > logs/sft_deep.out 2>&1
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/qa3.txt
HF_HOME=/mnt/kv_cache/hf_home $P eval/qa.py /mnt/kv_cache/run50_sft/ckpt.pt=ours-24-layer /mnt/kv_cache/run48_sft/ckpt.pt=ours-12-layer 2>&1 \
  | grep --line-buffered -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/qa3.txt
echo DEEP_DONE >> eval/qa3.txt
