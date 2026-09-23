#!/bin/bash
# THE FACTUAL-ACCURACY PUSH. The first fine-tune reached 47.9% on everyday facts; SmolLM2-135M, our size with
# ~130x our training text and no instruction tuning, reaches 62.5%. So the limit is how much the model has
# read, and the fix is knowledge-dense continued pretraining, then question-answer fine-tuning on top.
#   1. run47: run45 (the base, before any Q&A tuning) continued for ~2B tokens of prose with Wikipedia weighted
#      up -- the main stream restricted to web, educational and Wikipedia text, plus Wikipedia as a second corpus
#      at 35%: roughly 37% Wikipedia, 35% web, 27% educational, no code or mathematics. No unlikelihood term:
#      it cost compression and real words. On the fused scan (identical maths, ~10% faster whole steps), with
#      the dead state read off (removing it measured +-0.0001 bits per byte). Windows 2,048 and 8,192 at
#      16,384 tokens a micro-step, both measured to run.
#   2. run48: run47 fine-tuned on the factual Q&A mix (84k examples, short answers).
#   3. the same Q&A test as run46, so before and after are on identical questions.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
mkdir -p /mnt/kv_cache/run47_know
cp /mnt/kv_cache/run45_prose/ckpt.pt /mnt/kv_cache/run47_know/seed.pt
$P -u -m src.train --steps 438200 --d 768 --ctx 32768 --batch 2 --accum 4 --mixer rspin --lr 1.5e-3 \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --warm 200 --fused-scan 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia \
  --data2 /mnt/kv_cache/tokwiki_v8192 --data2-frac 0.35 \
  --save-every 2000 --eval-every 2000 --cap 28 --resume /mnt/kv_cache/run47_know/seed.pt \
  --out /mnt/kv_cache/run47_know > logs/train_know.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run47_know/ckpt.pt --out /mnt/kv_cache/run48_sft --data /mnt/kv_cache/sft_factual_v8192 \
  --epochs 2 --cap 16 > logs/sft_factual.out 2>&1
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/qa2.txt
HF_HOME=/mnt/kv_cache/hf_home $P eval/qa.py /mnt/kv_cache/run48_sft/ckpt.pt=ours-knowledge+factual /mnt/kv_cache/run46_sft/ckpt.pt=ours-first-finetune 2>&1 \
  | grep --line-buffered -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/qa2.txt
echo KNOW_DONE >> eval/qa2.txt
