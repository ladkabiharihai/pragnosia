#!/bin/bash
# Pause run85 at its step-360,000 checkpoint (exact verified PIDs: keep_training first, then the trainer), run the recall
# A/B from that checkpoint, then resume run85 from it with recall among its growth options (2026-10-05).
set -u
cd "$(dirname "$0")/.."
CK=/mnt/kv_cache/run85/ckpt.pt; KT=2327437; TR=2327439
m0=$(stat -c %Y $CK)
while true; do
  sleep 30
  m1=$(stat -c %Y $CK 2>/dev/null || echo 0)
  [ "$m1" != "$m0" ] || continue
  s1=$(stat -c %s $CK); sleep 15; s2=$(stat -c %s $CK)
  [ "$s1" = "$s2" ] || continue
  st=$(.venv/bin/python -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo -1)
  m0=$m1
  [ "$st" -ge 360000 ] && break
done
grep -q "keep_training.sh logs/train85.out" /proc/$KT/cmdline && kill $KT
sleep 2
grep -q "src.train" /proc/$TR/cmdline && kill $TR
while kill -0 $TR 2>/dev/null; do sleep 2; done
cp $CK /mnt/kv_cache/run85_s$st.pt
echo "=== RUN85 PAUSED $(date -u '+%Y-%m-%d %H:%M UTC') at its step-$st checkpoint for the recall A/B (eval/recall_ab.sh) ===" >> logs/train85.out
echo "- $(date -u '+%Y-%m-%d %H:%M:%S UTC') **run85 paused** at its step-$st checkpoint (PIDs $KT, $TR verified by cmdline) for the recall A/B; it resumes from the same checkpoint afterwards (eval/run85b.sh)." >> PROGRESS.md
eval/recall_ab.sh /mnt/kv_cache/run85_s$st.pt 4000
eval/run85b.sh /mnt/kv_cache/run85_s$st.pt
