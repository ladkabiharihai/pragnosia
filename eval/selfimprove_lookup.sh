#!/bin/bash
# GOAL 3, SELF-IMPROVEMENT ON A SKILL THE MODEL HAS (2026-10-03; PROGRESS.md). 1: the model self-labels invented lookup
# questions by agreement across 3 orderings of the facts (gold read only afterwards: the gate). 2: if most labels are
# right, two arms from the same checkpoint, 2,000 steps each: SELF (25% of rows its own self-labelled documents) and
# CONTROL (plain text only). 3: the powered lookup test (other syllables, other wording, hard distractors) on all three.
set -u
cd "$(dirname "$0")/.."
# usage: eval/selfimprove_lookup.sh <ckpt> [round tag, e.g. 2]   (round 2 starts from round 1's SELF model)
P=.venv/bin/python; CK=${1:-/mnt/kv_cache/run81_final.pt}; R=${2:-}; O=eval/selfimprove_lookup$R.txt
grep -q "against gold" $O 2>/dev/null || echo "SELF-IMPROVEMENT ON IN-CONTEXT LOOKUP, from $CK, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
mkdir -p /mnt/kv_cache/tokself$R
# labels already made for this round are reused (round 2's training never ran the first time: the gate's percentage was
# stored in R, the round tag, and the arms looked for a corpus "tokself84.8"; 2026-10-03)
[ -s /mnt/kv_cache/tokself$R/selflabel.bin ] && grep -q "against gold" $O 2>/dev/null || \
  $P -u eval/selfimprove.py lookup $CK /mnt/kv_cache/tokself$R 2000 2>&1 | grep -vE "Warning|^\[gpu\]" | tail -4 >> $O
GATE=$(grep -oE "against gold, [0-9.]+% right" $O | grep -oE "[0-9.]+" | head -1)
if [ -z "$GATE" ] || [ "$(echo "$GATE < 50" | bc)" = 1 ]; then echo "GATE FAILED: labels $GATE% right; not training on them" >> $O; exit 0; fi
C="--steps $(( $($P -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['step'])") + 2000 )) --d 512 --layers 12
   --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 8192 --spike 1e9
   --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 9 --data /mnt/kv_cache/tokfull_v8192
   --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --save-every 2000 --eval-every 2000 --cap 34"
for arm in self ctrl; do
  mkdir -p /mnt/kv_cache/si${R}_$arm; rm -f /mnt/kv_cache/si${R}_$arm/ckpt*.pt /mnt/kv_cache/si${R}_$arm/resume_from.pt; cp $CK /mnt/kv_cache/si${R}_$arm/seed.pt
  X=""; [ $arm = self ] && X="--data2 /mnt/kv_cache/tokself$R --data2-frac 0.25"
  eval/keep_training.sh logs/si${R}_$arm.out SI_DONE $C $X --resume /mnt/kv_cache/si${R}_$arm/seed.pt --out /mnt/kv_cache/si${R}_$arm || exit 1
done
for ck in $CK /mnt/kv_cache/si${R}_self/ckpt.pt /mnt/kv_cache/si${R}_ctrl/ckpt.pt; do
  echo "== $ck $(grep -h 'eval step' logs/si_*.out 2>/dev/null | head -0)" >> $O
  INCONTEXT_GPT2=0 $P eval/incontext.py $ck powered 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O
done
grep -h "eval step" logs/si${R}_self.out logs/si${R}_ctrl.out >> $O
echo "SELFIMPROVE_LOOKUP_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
