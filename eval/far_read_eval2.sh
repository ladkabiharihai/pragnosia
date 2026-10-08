#!/bin/bash
# EVALUATE THE FAR-READ PHASE (2026-10-07) against the final base, in the trainer's slot once training ends: open-book SciQ
# (closed, 0, 2k, 8k, 32k), learning by reading (8k, 32k), needles (8k, 32k, 64k). Output eval/far_read_eval.txt.
cd "$(dirname "$0")/.."
O=eval/far_read_eval2.txt; P=.venv/bin/python; C=/mnt/kv_cache/far_read2/ckpt_s384000.pt
until grep -q FAR_READ2_DONE logs/far_read2.out 2>/dev/null && [ -f $C ]; do sleep 60; done
echo "FAR READ EVAL $C, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
grep "eval step" logs/far_read2.out >> $O
export MEMBRAE_NOLOCK=1 MEMBRAE_CAP_MAX=6
N=150 ONLY_HF=none eval/guarded.sh 7 $P -u eval/openbook.py $C=far-read-384k 2>&1 | grep --line-buffered -vE "Warning|^\[gpu\]|Loading" >> $O
N_Q=100 DIST=8192,32768 eval/guarded.sh 7 $P -u eval/learn_by_reading.py $C=far-read-384k 2>&1 | grep --line-buffered -vE "Warning|^\[gpu\]" >> $O
CAP=6 RECALL=16 NEEDLE_DIST=8192,32768,65536 NEEDLE_N=24 eval/guarded.sh 7 $P -u eval/recall_test.py $C needle 2>&1 | grep --line-buffered -vE "Warning|^\[gpu\]" >> $O
echo "FAR_READ_EVAL2_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
