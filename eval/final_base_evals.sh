#!/bin/bash
# THE FINAL BASE'S OWN EVALUATIONS (2026-10-08), in the TRAINER's slot while no trainer runs (MEMBRAE_NOLOCK): the benchmark
# suite against GPT-2 and SmolLM2, needles 4k-256k (store on the GPU) and 512k / 1M (store in host memory, PyTorch capped at
# 3.5 GiB), long generation with DRY (recall on and off). usage: eval/final_base_evals.sh <base ckpt> <tag>
set -u
cd "$(dirname "$0")/.."
B=$1; T=$2; P=.venv/bin/python; O=eval/final_base_$T.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
export MEMBRAE_NOLOCK=1
echo "FINAL BASE EVALUATIONS $T: $B, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/bench_suite.py $B=base-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=12 CAP=12 RECALL=16 NEEDLE_DIST=4096,16384,65536,131072,262144 NEEDLE_N=24 eval/guarded.sh 13 $P -u eval/recall_test.py $B needle 2>&1 | grep --line-buffered -vE "$F" >> $O
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RECALL_OFFLOAD=1 MEMBRAE_CAP_MAX=3.5 CAP=3.5 RECALL=16 NEEDLE_DIST=524288,1048576 NEEDLE_N=12 \
  eval/guarded.sh 4 $P -u eval/recall_test.py $B needle 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/lg_check.py $B@0 $B 2>&1 | grep --line-buffered -vE "$F" >> $O
echo "FINAL_BASE_EVALS_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
