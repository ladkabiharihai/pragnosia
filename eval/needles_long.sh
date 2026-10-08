#!/bin/bash
# RETRIEVAL FAR PAST 1K (2026-10-07; user: "we need retrieval beyond 1k"): the final base's needles at 512k and 1M tokens,
# in the TRAINER's slot while no trainer runs (MEMBRAE_NOLOCK). Recall keeps every key of the three attention layers (~19 KB a
# token in float32); on the GPU that is 18 GiB at 1M tokens. RECALL_OFFLOAD=1 keeps the store in host memory (model.py
# FarStore: the same function, exact picks up to numerical ties), so the GPU holds the model, the window and one streamed
# chunk: capped here at 3.5 GiB for PyTorch, the guard at 4 GiB + 0.5 for the CUDA context (it reports the true peak).
set -u
cd "$(dirname "$0")/.."
CK=${1:-/mnt/kv_cache/final_base_far.pt}; O=eval/needles_long.txt; P=.venv/bin/python
echo "NEEDLES FAR PAST THE WINDOW: $CK, $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MEMBRAE_NOLOCK=1 RECALL=16 RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 CAP=3.5
NEEDLE_DIST=${NEEDLE_DIST:-524288,1048576} NEEDLE_N=${NEEDLE_N:-12} eval/guarded.sh 4 $P -u eval/recall_test.py $CK needle 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O
echo "NEEDLES_LONG_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
