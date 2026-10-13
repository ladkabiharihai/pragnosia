#!/bin/bash
# EVALUATE THE LONG-SPAN PHASE (2026-10-08) against 382k, in the trainer's slot once training ends: open-book SciQ (closed, 0,
# 2k, 8k, 32k), learning by reading (8k, 32k), needles (64k, 131k, 262k; store on the GPU), and prediction over 8 paired passages
# at 8k / 32k / 131k (recall on, store in host memory). 382k: open-book 51.3 / 84.0 / 72.7 / 76.0 / 50.0; reading 87 / 89;
# needles 95.8 / 91.7 / 79.2; prediction 0.9739 / 0.9735 / 1.0339 (recall off 1.0016 / 1.0087 / 1.0091).
cd "$(dirname "$0")/.."
O=eval/far_long_eval.txt; P=.venv/bin/python; C=${1:-/mnt/kv_cache/far_long/ckpt_s385000.pt}; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
until grep -q FAR_LONG_DONE logs/far_long.out 2>/dev/null && [ -f $C ]; do sleep 60; done
echo "FAR LONG EVAL $C, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
grep "eval step" logs/far_long.out >> $O
export MEMBRAE_NOLOCK=1
MEMBRAE_CAP_MAX=6 N=150 ONLY_HF=none eval/guarded.sh 7 $P -u eval/openbook.py $C=far-long 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 N_Q=100 DIST=8192,32768 eval/guarded.sh 7 $P -u eval/learn_by_reading.py $C=far-long 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=12 CAP=12 RECALL=16 NEEDLE_DIST=65536,131072,262144 NEEDLE_N=24 eval/guarded.sh 13 $P -u eval/recall_test.py $C needle 2>&1 | grep --line-buffered -vE "$F" >> $O
echo "== prediction, 8 paired passages, recall on (store in host memory)" >> $O
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 MILLION_SAMPLES=8 eval/guarded.sh 4 $P -u eval/million.py $C 8192,32768,131072 2>&1 | grep --line-buffered -E "tokens of context" >> $O
echo "FAR_LONG_EVAL_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
