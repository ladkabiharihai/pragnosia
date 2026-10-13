#!/bin/bash
# DOES SELF-IMPROVEMENT CARRY TO NEW KINDS OF FACT? (2026-10-12, goal 3's open caveat.) The rounds' test (eval/incontext.py powered,
# hard) holds out names and wording but uses the same two fact kinds the rounds label (capitals, engine inventors). This asks
# the same test over two kinds no training builder and no round ever wrote (INCONTEXT_KINDS=new: rivers into seas, guild
# leaders), on the base and the self-improved rounds. In the trainer's slot while no trainer runs (MEMBRAE_NOLOCK), cap 8.
cd "$(dirname "$0")/.."; P=.venv/bin/python; O=eval/si_newkinds_v2.txt
echo "SELF-IMPROVEMENT ON NEW FACT KINDS (rivers -> seas, guild leaders), hard four-way, chance 25%; $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
for ck in /mnt/kv_cache/final_base_v2.pt /mnt/kv_cache/si_v2_r1_ft/ckpt.pt /mnt/kv_cache/si_v2_r3_ft/ckpt.pt; do
  echo "== $ck" >> $O
  MEMBRAE_NOLOCK=1 MEMBRAE_CAP_MAX=8 INCONTEXT_KINDS=new INCONTEXT_GPT2=0 eval/guarded.sh 9 $P -u eval/incontext.py $ck powered 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O
done
echo "SI_NEWKINDS_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
