#!/bin/bash
# THE OUTLIER-ONLY FAR KEYS (RECALL_EVC=8192 RECALL_EVC_MODE=mask, 2026-10-08): a far key is kept only if it beats the expected
# maximum of N chance scores. A million tokens (one passage), needles 128k / 256k (store in host memory), reading 8k / 32k.
cd "$(dirname "$0")/.."
C=/mnt/kv_cache/final_base_fr382.pt; P=.venv/bin/python; O=eval/evc_mask.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
export MEMBRAE_NOLOCK=1 RECALL_EVC=8192 RECALL_EVC_MODE=mask PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "OUTLIER-ONLY FAR KEYS (mask, N0=8192) on $C, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 eval/guarded.sh 4 $P -u eval/million.py $C 8192,32768,131072,524288,1048576 2>&1 | grep --line-buffered -E "tokens of context" >> $O
RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 CAP=3.5 RECALL=16 NEEDLE_DIST=131072,262144 NEEDLE_N=24 eval/guarded.sh 4 $P -u eval/recall_test.py $C needle 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 N_Q=100 DIST=8192,32768 eval/guarded.sh 7 $P -u eval/learn_by_reading.py $C=fr382-mask 2>&1 | grep --line-buffered -vE "$F" >> $O
echo "EVC_MASK_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
