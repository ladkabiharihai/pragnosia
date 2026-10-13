#!/bin/bash
# THE VISION MEASURES ON ONE CHECKPOINT (2026-10-11; user: "have you tested the latest checkpoint on vision"), as a side job beside
# a trainer (cap 7, guard 8: prod ~52 + trainer 32 + 8 under the card): same-text BPB (goal 1 / general text), generation on
# facts / arithmetic / reasoning (goal 1), the benchmark suite (goal 1), reading invented facts at 0 / 8k / 32k (goals 7, 9),
# needles 16k / 64k / 131k (goal 9), open-book at 2k / 8k / 32k (goal 9). usage: eval/vision_ckpt.sh <ckpt> <label>
cd "$(dirname "$0")/.."
C=$1; T=$2; P=.venv/bin/python; O=eval/vision_$T.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices|\[base|\[eval\]"
export MEMBRAE_CAP_MAX=7
echo "VISION MEASURES $T ($C), $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
eval/guarded.sh 8 $P -u eval/bpb_ckpt.py $C 2>&1 | grep --line-buffered -vE "$F" >> $O
eval/guarded.sh 8 $P -u eval/showcase_ckpts.py $C=$T 2>&1 | grep --line-buffered -vE "$F" >> $O
N_Q=100 DIST=0,8192,32768 eval/guarded.sh 8 $P -u eval/learn_by_reading.py $C=$T 2>&1 | grep --line-buffered -vE "$F" >> $O
CAP=7 RECALL=16 NEEDLE_DIST=16384,65536,131072 NEEDLE_N=24 eval/guarded.sh 8 $P -u eval/recall_test.py $C needle 2>&1 | grep --line-buffered -vE "$F" >> $O
N=150 DIST=2048,8192,32768 ONLY_HF=none eval/guarded.sh 8 $P -u eval/openbook.py $C=$T 2>&1 | grep --line-buffered -vE "$F" >> $O
BENCH_HF=0 N=500 eval/guarded.sh 8 $P -u eval/bench_suite.py $C=$T 2>&1 | grep --line-buffered -vE "$F" | grep -vE "^\s+\[" >> $O
echo "VISION_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
