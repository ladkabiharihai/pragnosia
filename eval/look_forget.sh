#!/bin/bash
# DOES THE LOOKUP FINE-TUNE HAVE TO FORGET? (2026-10-12) The lookup fine-tune over many kinds (2,421 steps, answers only) carried lookup
# to never-seen kinds but raised same-text bits/byte 1.0143 -> 1.0933 (+7.8%) and broke chat answers (facts 0/10). Two variants
# from the same base (final_base_v3, 604k): MIX = the same lookup rows interleaved 1:1 with general text (loss on every token of
# it: replay), 600 = the first 600 lookup rows only. Each: bits/byte, held-out kinds, trained kinds (n=200 per row, +-6).
# In the paused trainer's slot (MEMBRAE_NOLOCK), cap 8. Output eval/look_forget.txt.
cd "$(dirname "$0")/.."; P=.venv/bin/python; O=eval/look_forget.txt; B=/mnt/kv_cache/final_base_v3.pt
echo "LOOKUP FINE-TUNE WITHOUT FORGETTING? from $B (bits/byte 1.0143; full lookup fine-tune 1.0933), $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
for V in mix 600; do D=/mnt/kv_cache/lookgold_$V
  echo "== variant $V" >> $O
  MEMBRAE_NOLOCK=1 MEMBRAE_CAP_MAX=8 eval/guarded.sh 9 $P -m src.sft --init $B --out ${D}_ft --data $D --optim psf --epochs 1 --batch 1 --row 4097 --cap 8 2>&1 | grep -E "done|Error|Traceback" | tail -1 >> $O
  MEMBRAE_NOLOCK=1 eval/guarded.sh 8 $P -u eval/bpb_ckpt.py ${D}_ft/ckpt.pt 2>&1 | grep -E "BPB" >> $O
  for kd in new old; do echo "-- kinds=$kd" >> $O
    MEMBRAE_NOLOCK=1 INCONTEXT_KINDS=$kd INCONTEXT_TRIALS=20 INCONTEXT_GPT2=0 MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/incontext.py ${D}_ft/ckpt.pt powered 2>&1 | grep -E "^(10|30|60|120) " >> $O; done
done
echo "LOOK_FORGET_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
