#!/bin/bash
# EVALUATE THE READING RESTORE (2026-10-11) at its end (598k) beside the 590k start: reading 0 / 8k / 32k, needles 16k / 64k / 131k,
# open-book (closed, 2k, 8k, 32k), same-text BPB, the benchmark suite (ours). Trainer slot (MEMBRAE_NOLOCK) once training ends.
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/restore_eval.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices|\[base|\[eval\]"
C=/mnt/kv_cache/restore/ckpt_s598000.pt; B=/mnt/kv_cache/know/ckpt_s590000.pt
until grep -qE "RESTORE_DONE" logs/restore.out 2>/dev/null && [ -f $C ]; do sleep 60; grep -q RESTORE_FAILED logs/restore.out 2>/dev/null && exit 1; done
export MEMBRAE_NOLOCK=1
echo "READING RESTORE EVAL $(date -u '+%Y-%m-%d %H:%M UTC'): $C vs $B" > $O
MEMBRAE_CAP_MAX=10 eval/guarded.sh 11 $P -u eval/bpb_ckpt.py $B $C 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=10 N_Q=100 DIST=0,8192,32768 eval/guarded.sh 11 $P -u eval/learn_by_reading.py $B=know-590k $C=restore-598k 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=14 CAP=14 RECALL=16 NEEDLE_DIST=16384,65536,131072 NEEDLE_N=24 eval/guarded.sh 15 $P -u eval/recall_test.py $C needle 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=10 N=150 ONLY_HF=none eval/guarded.sh 11 $P -u eval/openbook.py $B=know-590k $C=restore-598k 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=10 BENCH_HF=0 N=500 eval/guarded.sh 11 $P -u eval/bench_suite.py $C=restore-598k 2>&1 | grep --line-buffered -vE "$F" | grep -vE "^\s+\[" >> $O
echo "RESTORE_EVAL_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
