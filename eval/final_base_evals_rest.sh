#!/bin/bash
# THE REST OF eval/final_base_evals.sh on 382k (paused 2026-10-08 10:00 UTC for the long-span run, not adopted): needles at
# 512k / 1M with the store in host memory, long generation with DRY (recall on and off). Appends to eval/final_base_fr.txt.
cd "$(dirname "$0")/.."
B=/mnt/kv_cache/final_base_fr382.pt; P=.venv/bin/python; O=eval/final_base_fr.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
until grep -q FAR_LONG_EVAL_DONE eval/far_long_eval.txt 2>/dev/null; do sleep 60; done
export MEMBRAE_NOLOCK=1
echo "== resumed $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 CAP=3.5 RECALL=16 NEEDLE_DIST=524288,1048576 NEEDLE_N=12 \
  eval/guarded.sh 4 $P -u eval/recall_test.py $B needle 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/lg_check.py $B@0 $B 2>&1 | grep --line-buffered -vE "$F" >> $O
echo "FINAL_BASE_EVALS_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
