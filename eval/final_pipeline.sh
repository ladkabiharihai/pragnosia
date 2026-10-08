#!/bin/bash
# THE FINAL MODEL'S PIPELINE (2026-10-07): from the chosen base (368k + the short recall run) --
#   1. self-improvement rounds (eval/si_rounds.sh: label -> learn -> test, a curriculum of harder documents; compounded on 396k)
#   2. the full evaluation pipeline on the self-improved model (eval/eval_ckpt.sh: factual fine-tune, abstention-tune, scorecard)
#   3. abstention by self-consistency on the fine-tuned model (eval/selfcons.py), open-book reading at a distance (eval/openbook.py)
# every GPU step under eval/guarded.sh (one side job at a time). usage: eval/final_pipeline.sh <base ckpt> <tag>
set -u
cd "$(dirname "$0")/.."
B=$1; T=$2; P=.venv/bin/python; O=eval/final_pipeline_$T.txt
echo "FINAL PIPELINE $T from $B, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
eval/si_rounds.sh $B 1000 _$T; tail -6 eval/si_rounds_$T.txt >> $O
S=/mnt/kv_cache/si_${T}_r3_ft/ckpt.pt; [ -f $S ] || S=$(ls -t /mnt/kv_cache/si_${T}_r*_ft/ckpt.pt 2>/dev/null | head -1); [ -n "$S" ] || S=$B
echo "== the self-improved model: $S" >> $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 eval/eval_ckpt.sh $S final_$T >> $O 2>&1
MEMBRAE_CAP_MAX=6 eval/guarded.sh 6 $P -u eval/selfcons.py /mnt/kv_cache/final_${T}_sft/ckpt.pt 6 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O
MEMBRAE_CAP_MAX=6 N=150 eval/guarded.sh 6 $P -u eval/openbook.py $B=base-$T $S=self-improved-$T 2>&1 | grep -vE "Warning|^\[gpu\]|Loading" >> $O
echo "FINAL_PIPELINE_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
