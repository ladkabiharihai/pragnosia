#!/bin/bash
# AFTER run52_grow: the same factual Q&A fine-tune the earlier models got, then the same Q&A test (48 everyday
# facts, 1,000 held-out SciQ, ARC-Easy 4-way, greedy), against the previous best of ours and GPT-2 / SmolLM2-135M.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
until grep -q "END train" logs/train_grow.out; do sleep 60; done
$P -m src.sft --init /mnt/kv_cache/run52_grow/ckpt.pt --out /mnt/kv_cache/run53_sft --data /mnt/kv_cache/sft_factual_v8192 \
  --epochs 2 --cap 30 > logs/sft_grow.out 2>&1
echo "=== run52_grow + factual fine-tune, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/qa4.txt
$P eval/qa.py /mnt/kv_cache/run53_sft/ckpt.pt=ours-grown-319M /mnt/kv_cache/run46_sft/ckpt.pt=ours-12L-previous-best 2>&1 \
  | grep --line-buffered -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/qa4.txt
echo GROW_QA_DONE >> eval/qa4.txt
