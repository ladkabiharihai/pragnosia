#!/bin/bash
# GOALS 4-5: R-Tuning. run46_sft (our best, 47.9% everyday) fine-tuned one epoch on the factual mix with SciQ/ARC
# replaced by its OWN answers to 15,049 SciQ/ARC training questions: 1,274 it got right keep the right answer,
# 1,274 of the 13,775 it got wrong (sampled, balanced 1:1) become "I don't know." Then the same Q&A test, counting right / WRONG / declined.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
$P -m src.sft --init /mnt/kv_cache/run46_sft/ckpt.pt --out /mnt/kv_cache/run55_rtune_bal --data /mnt/kv_cache/sft_rtune_bal_v8192 \
  --epochs 1 --cap 20 > logs/sft_rtune_bal.out 2>&1
echo "=== R-Tuning (abstention from own mistakes) vs the same model before, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/abstain_rt2.txt
QA_BASELINES=0 $P -u eval/qa.py /mnt/kv_cache/run55_rtune_bal/ckpt.pt=after-balanced /mnt/kv_cache/run54_rtune/ckpt.pt=after-unbalanced 2>&1 \
  | grep --line-buffered -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/abstain_rt2.txt
echo RT_DONE >> eval/abstain_rt2.txt
