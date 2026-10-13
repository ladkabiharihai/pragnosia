#!/bin/bash
# DOES LOOKUP TRAINED OVER MANY KINDS BECOME GENERAL? (2026-10-12) The many-kinds pretraining moved the held-out kinds nothing in
# 4,000 steps (their answers are ~2% of a document's loss). The control: a fine-tune with the loss on answers only, over
# eval/kinds.py's 381 kinds with the generator's answers (eval/selfimprove.py SI_GOLD=1, /mnt/kv_cache/lookgold; NOT
# self-improvement), from the run's 602k checkpoint, then the held-out kinds (rivers, guilds) and the trained kinds, full size.
cd "$(dirname "$0")/.."; P=.venv/bin/python; O=eval/lookgold_test.txt; CK=${1:-/mnt/kv_cache/many/ckpt_s602000.pt}; D=/mnt/kv_cache/lookgold
echo "LOOKUP FINE-TUNE OVER MANY KINDS (gold answers, answer-only loss) from $CK, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
MEMBRAE_CAP_MAX=8 eval/guarded.sh 9 $P -m src.sft --init $CK --out ${D}_ft --data $D --optim psf --epochs 1 --batch 1 --row 4097 --cap 8 2>&1 | grep -E "held-out|done|Error|Traceback|GUARD" | tail -3 >> $O
for kd in new old; do echo "== kinds=$kd (new = held out: rivers, guilds; old = capitals, engines)" >> $O
  INCONTEXT_KINDS=$kd INCONTEXT_GPT2=0 MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/incontext.py ${D}_ft/ckpt.pt powered 2>&1 | grep -E "^(10|30|60|120) |by where" >> $O; done
echo "LOOKGOLD_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
