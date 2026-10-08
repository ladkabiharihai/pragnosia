#!/bin/bash
# THE FINAL MODEL'S PIPELINE, v2 (2026-10-08), side slot (every step under eval/guarded.sh, one side job at a time):
#   1. self-improvement rounds (eval/si_rounds.sh); the self-improved model is the round with the best mean on the hard lookup
#      test (round 3 added nothing on 375k), not simply the last
#   2. factual fine-tune + abstention-tune + scorecard (eval/eval_ckpt.sh) on it
#   3. self-consistency abstention (eval/selfcons.py); open-book (base, self-improved, GPT-2, SmolLM2: guard 8, SmolLM2 took 6.7)
#   4. the benchmark suite, learning by reading and long generation (DRY now applied in the recall decode) on the self-improved model
# usage: eval/final_pipeline2.sh <base ckpt> <tag>        the base's own evaluations: eval/final_base_evals.sh
set -u
cd "$(dirname "$0")/.."
B=$1; T=$2; P=.venv/bin/python; O=eval/final_pipeline_$T.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
echo "FINAL PIPELINE v2 $T from $B, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
eval/si_rounds.sh $B 1000 _$T
S=$($P - "$T" <<'PY'
import re, sys
t = sys.argv[1]; txt = open(f"eval/si_rounds_{t}.txt").read()
best, arg = -1, None
for blk in re.split(r"(?m)^== ", txt):
    m = re.match(r"round (\d) result", blk)
    rows = [float(v) for v in re.findall(r"(?m)^\d+\s+([0-9.]+)%", blk)]
    if m and len(rows) == 4 and sum(rows) / 4 > best: best, arg = sum(rows) / 4, int(m.group(1))
print(f"/mnt/kv_cache/si_{t}_r{arg}_ft/ckpt.pt" if arg else "")
PY
)
[ -n "$S" ] && [ -f "$S" ] || S=$B
grep -E "^== round [0-9] result" -A6 eval/si_rounds_$T.txt | grep -E "^==|^[0-9]+ " >> $O
echo "== the self-improved model (best mean round): $S" >> $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 eval/eval_ckpt.sh $S final_$T >> $O 2>&1
MEMBRAE_CAP_MAX=6 eval/guarded.sh 6 $P -u eval/selfcons.py /mnt/kv_cache/final_${T}_sft/ckpt.pt 6 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=7.5 N=150 eval/guarded.sh 8 $P -u eval/openbook.py $B=base-$T $S=self-improved-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 N_Q=100 DIST=0,8192,32768 eval/guarded.sh 7 $P -u eval/learn_by_reading.py $S=self-improved-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/lg_check.py $S 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/bench_suite.py $S=self-improved-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
echo "FINAL_PIPELINE_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
