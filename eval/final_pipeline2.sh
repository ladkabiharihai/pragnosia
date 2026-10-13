#!/bin/bash
# THE FINAL MODEL'S PIPELINE, v2 (2026-10-08), side slot (every step under eval/guarded.sh, one side job at a time):
#   1. self-improvement rounds (eval/si_rounds.sh); the self-improved model is the round with the best mean on the hard lookup
#      test (round 3 added nothing on 375k), not simply the last
#   2. factual fine-tune + abstention-tune + scorecard (eval/eval_ckpt.sh) on it
#   3. self-consistency abstention (eval/selfcons.py); open-book (base, self-improved, GPT-2, SmolLM2: guard 8, SmolLM2 took 6.7)
#   4. the benchmark suite, learning by reading and long generation (DRY now applied in the recall decode) on the self-improved model
# v3 (2026-10-12), env: LOOK=1 first fine-tunes the base on lookup over eval/kinds.py's 381 kinds (gold answers, answer-only loss,
#   /mnt/kv_cache/lookgold: the held-out kinds went 24-35% -> 58-66% on 602k), and the rounds then label many kinds (SI_KINDS=many)
#   and are tested on the held-out kinds too; PCAP (default 12) caps every step, so the pipeline can share the card with a
#   trainer (prod 51 + trainer 32 + 8 < 93); OPENBOOK_HF=0 skips the baselines' open-book rows (measured already).
#   LOOKDATA (2026-10-12): the lookup rows alone (/mnt/kv_cache/lookgold) raised same-text bits/byte 1.0143 -> 1.0933; interleaved 1:1
#   with general text (/mnt/kv_cache/lookgold_mix, replay) 1.0171, with the held-out-kind gain kept (eval/look_forget.txt).
# usage: eval/final_pipeline2.sh <base ckpt> <tag>        the base's own evaluations: eval/final_base_evals.sh
set -u
cd "$(dirname "$0")/.."
B=$1; T=$2; P=.venv/bin/python; O=eval/final_pipeline_$T.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices"
echo "FINAL PIPELINE v2 $T from $B, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
PC=${PCAP:-12}; PG=$((PC + 1)); B0=$B
# FINITE (2026-10-12): the replay lookup fine-tune diverged to NaN in one of two identical runs (loss 1.4 -> 2.7 -> nan by step
# 3,100; 213 of 231 tensors non-finite) and the pipeline went on testing it. Every fine-tuned checkpoint is now checked.
finite () { $P -c "import torch,sys; m=torch.load(sys.argv[1],map_location='cpu',weights_only=False); m=m.get('model',m); sys.exit(0 if all(torch.isfinite(v).all() for v in m.values() if torch.is_tensor(v) and v.is_floating_point()) else 1)" "$1"; }
if [ "${LOOK:-0}" = 1 ]; then
  if [ -n "${LOOKCK:-}" ]; then        # LOOKCK: a lookup fine-tune made and tested already (eval/look_forget.sh)
    echo "== lookup fine-tune over many kinds: $LOOKCK (made and tested beforehand)" >> $O; L=$LOOKCK
  else
    echo "== lookup fine-tune over many kinds (gold answers; ${LOOKDATA:-/mnt/kv_cache/lookgold}) from $B" >> $O
    MEMBRAE_CAP_MAX=$PC eval/guarded.sh $PG $P -m src.sft --init $B --out /mnt/kv_cache/final_${T}_look --data ${LOOKDATA:-/mnt/kv_cache/lookgold} --optim psf --epochs 1 --batch 1 --row 4097 --cap $PC 2>&1 | grep -E "held-out|done" | tail -1 >> $O
    L=/mnt/kv_cache/final_${T}_look/ckpt.pt
  fi
  if [ -f $L ] && finite $L; then B=$L; else echo "LOOK FAILED: $L missing or non-finite; stopping" >> $O; exit 1; fi
fi
SI_CAP=$PC SI_GUARD=$PG eval/si_rounds.sh $B 1000 _$T
S=$($P - "$T" <<'PY'
import re, sys
t = sys.argv[1]; txt = open(f"eval/si_rounds_{t}.txt").read()
best, arg = -1, None
for blk in re.split(r"(?m)^== ", txt):
    m = re.match(r"round (\d) result", blk)
    rows = [float(v) for v in re.findall(r"(?m)^\d+\s+([0-9.]+)%", blk)]
    # 4 rows (trained kinds), or 8 with the held-out kinds (si_rounds.sh, 2026-10-12): the mean over all of them
    if m and len(rows) in (4, 8) and sum(rows) / len(rows) > best: best, arg = sum(rows) / len(rows), int(m.group(1))
print(f"/mnt/kv_cache/si_{t}_r{arg}_ft/ckpt.pt" if arg else "")
PY
)
[ -n "$S" ] && [ -f "$S" ] || S=$B
grep -E "^== round [0-9] result" -A11 eval/si_rounds_$T.txt | grep -E "^==|^--|^[0-9]+ " >> $O
echo "== the self-improved model (best mean round): $S" >> $O
MEMBRAE_CAP_MAX=$PC eval/guarded.sh $PG eval/eval_ckpt.sh $S final_$T >> $O 2>&1
MEMBRAE_CAP_MAX=$PC eval/guarded.sh $PG $P -u eval/selfcons.py /mnt/kv_cache/final_${T}_sft/ckpt.pt 6 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=$PC OPENBOOK_CAP=$PC N=150 eval/guarded.sh $PG $P -u eval/openbook.py $B0=base-$T $S=self-improved-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=$PC N_Q=100 DIST=0,8192,32768 eval/guarded.sh $PG $P -u eval/learn_by_reading.py $S=self-improved-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=$PC eval/guarded.sh $PG $P -u eval/lg_check.py $S 2>&1 | grep --line-buffered -vE "$F" >> $O
MEMBRAE_CAP_MAX=$PC eval/guarded.sh $PG $P -u eval/bench_suite.py $S=self-improved-$T 2>&1 | grep --line-buffered -vE "$F" >> $O
echo "FINAL_PIPELINE_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
