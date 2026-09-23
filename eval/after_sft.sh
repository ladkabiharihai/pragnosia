#!/bin/bash
# In priority order, one at a time, each only once the card is free:
#   1. question answering on the fine-tuned model against GPT-2 and SmolLM2   (what investors test first)
#   2. training throughput at equal tokens per step                            (the fair efficiency comparison)
#   3. the optimised fine-tuning loop, timed alone                             (did the three fixes pay)
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
while pgrep -f "[s]rc.sft" > /dev/null || pgrep -f "[s]rc.train" > /dev/null; do sleep 20; done
sleep 5
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/qa.txt
HF_HOME=/mnt/kv_cache/hf_home $P eval/qa.py /mnt/kv_cache/run46_sft/ckpt.pt=ours-finetuned /mnt/kv_cache/run45_prose/ckpt.pt=ours-before 2>&1 \
  | grep -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" >> eval/qa.txt
echo QA_DONE >> eval/qa.txt
eval/fairbench.sh
echo "=== optimised fine-tuning loop, timed alone, $(date -u '+%H:%M UTC') ===" > eval/sft_speed.txt
$P -m src.sft --init /mnt/kv_cache/run45_prose/ckpt.pt --out /mnt/kv_cache/sft_speedtest --epochs 0.05 --eval-every 100000 2>&1 \
  | grep -E "ktok/s|held-out|done" >> eval/sft_speed.txt
rm -rf /mnt/kv_cache/sft_speedtest
echo SPEED_DONE >> eval/sft_speed.txt
