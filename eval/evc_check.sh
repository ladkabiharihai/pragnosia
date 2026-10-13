#!/bin/bash
# THE CHANCE-RISE CORRECTION (RECALL_EVC=8192, src/model.py _evc_shift) ON THE FINAL BASE (2026-10-08): it fixed a million-token
# prediction (1.1180 -> 1.0405; recall off 1.0411) on one passage. Does it keep what recall is for? Needles at 128k / 256k
# (store in host memory), reading at 8k / 32k, open-book at 8k / 32k, and prediction over 8 paired passages at 8k / 32k /
# 131k, each against the same test without the correction (eval/final_base_fr.txt, eval/far_read_eval382.txt).
cd "$(dirname "$0")/.."
C=/mnt/kv_cache/final_base_fr382.pt; P=.venv/bin/python; O=eval/evc_check.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
export MEMBRAE_NOLOCK=1 RECALL_EVC=8192
echo "CHANCE-RISE CORRECTION N0=8192 on $C, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 CAP=3.5 RECALL=16 NEEDLE_DIST=131072,262144 NEEDLE_N=24 \
  eval/guarded.sh 4 $P -u eval/recall_test.py $C needle 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 N_Q=100 DIST=8192,32768 eval/guarded.sh 7 $P -u eval/learn_by_reading.py $C=fr382-evc 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 N=150 DIST=8192,32768 ONLY_HF=none eval/guarded.sh 7 $P -u eval/openbook.py $C=fr382-evc 2>&1 | grep --line-buffered -vE "$F" >> $O
for E in 8192 0; do echo "== prediction, 8 paired passages, RECALL_EVC=$E (recall on, store in host memory)" >> $O
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RECALL_EVC=$E RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 MILLION_SAMPLES=8 \
    eval/guarded.sh 4 $P -u eval/million.py $C 8192,32768,131072 2>&1 | grep --line-buffered -E "tokens of context" >> $O; done
echo "== prediction, 8 paired passages, recall OFF" >> $O
MEMBRAE_CAP_MAX=4 MILLION_RECALL=0 MILLION_SAMPLES=8 eval/guarded.sh 5 $P -u eval/million.py $C 8192,32768,131072 2>&1 | grep --line-buffered -E "tokens of context" >> $O
echo "EVC_CHECK_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
