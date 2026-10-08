#!/bin/bash
# GOAL 3: SELF-IMPROVEMENT THAT COMPOUNDS (2026-10-06). Round 2 at round 1's difficulty added nothing (2026-10-03). Here each
# round moves to documents with MORE facts than the last (a curriculum the model climbs on its own labels):
#   label  eval/selfimprove.py lookup: invented facts, questions about them, the model's answer under 3 orderings of the
#          facts, the answer >= 2 agree on becomes the label (gold read only afterwards, as a diagnostic and a gate)
#   learn  src/sft.py on its own labelled documents, the loss on its answers only (a fine-tune that fits beside training)
#   test   eval/incontext.py powered: the held-out hard lookup test (other names, other wording, every distractor from the
#          same context), 10 / 30 / 60 / 120 facts, 500 questions each
# Every step under eval/guarded.sh 6 (one side job at a time, killed past 6 GiB).
# usage: eval/si_rounds.sh <start ckpt> [docs per round] [tag: outputs eval/si_rounds<tag>.txt, /mnt/kv_cache/si<tag>_r*]
set -u
cd "$(dirname "$0")/.."
CK=$1; ND=${2:-1000}; TAG=${3:-}; P=.venv/bin/python; O=eval/si_rounds$TAG.txt; G="eval/guarded.sh 6"
echo "SELF-IMPROVEMENT ROUNDS from $CK, $ND documents per round, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
echo "== round 0 (the start)" >> $O
MEMBRAE_CAP_MAX=6 INCONTEXT_GPT2=0 $G $P eval/incontext.py $CK powered 2>&1 | grep -vE "Warning|^\[gpu\]|GUARD" >> $O
r=0
for F in "20,40,80" "80,160" "160,320"; do
  r=$((r + 1)); D=/mnt/kv_cache/si${TAG}_r$r; rm -rf $D ${D}_ft; mkdir -p $D
  echo "== round $r: documents of $F facts, labelled by the model from $CK" >> $O
  MEMBRAE_CAP_MAX=6 SI_FACTS=$F $G $P -u eval/selfimprove.py lookup $CK $D $ND 2>&1 | grep -vE "Warning|^\[gpu\]|GUARD" | tail -3 >> $O
  GATE=$(grep -oE "against gold, [0-9.]+% right" $O | tail -1 | grep -oE "[0-9.]+" | head -1)
  if [ -z "$GATE" ] || [ "$(echo "$GATE < 50" | bc)" = 1 ]; then echo "GATE FAILED at round $r: labels $GATE% right; stopping" >> $O; break; fi
  MEMBRAE_CAP_MAX=6 $G $P -m src.sft --init $CK --out ${D}_ft --data $D --optim psf --epochs 1 --batch 1 --row 4097 --cap 6 2>&1 | grep -E "held-out|done" | tail -2 >> $O
  [ -f ${D}_ft/ckpt.pt ] || { echo "round $r fine-tune failed" >> $O; break; }
  CK=${D}_ft/ckpt.pt
  echo "== round $r result" >> $O
  MEMBRAE_CAP_MAX=6 INCONTEXT_GPT2=0 $G $P eval/incontext.py $CK powered 2>&1 | grep -vE "Warning|^\[gpu\]|GUARD" >> $O
done
echo "SI_ROUNDS_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
