#!/bin/bash
# Pause run85 at its next checkpoint (step >= 368,000) for the recall A/B, then resume it (2026-10-05, second attempt).
# The first attempt (eval/pause85_ab.sh) grepped /proc/PID/cmdline, whose arguments are NUL-separated: the wrapper was not
# killed, it resumed run85, and two trainings filled the card. Here the cmdline is read with NULs as spaces, the wrapper is
# killed first and checked dead, and nothing starts while any trainer is alive.
set -u
cd "$(dirname "$0")/.."
CK=/mnt/kv_cache/run85/ckpt.pt; KT=2327437; TR=2342042; PARENT=2327333
cmd () { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null; }
m0=$(stat -c %Y $CK)
while true; do
  sleep 30
  m1=$(stat -c %Y $CK 2>/dev/null || echo 0); [ "$m1" != "$m0" ] || continue
  s1=$(stat -c %s $CK); sleep 15; s2=$(stat -c %s $CK); [ "$s1" = "$s2" ] || continue
  st=$(.venv/bin/python -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo -1)
  m0=$m1; [ "$st" -ge 368000 ] && break
done
[[ "$(cmd $KT)" == *"keep_training.sh logs/train85.out"* ]] && kill $KT
sleep 3
if kill -0 $KT 2>/dev/null; then echo "wrapper $KT still alive; aborting without touching the trainer" >> logs/train85.out; exit 1; fi
[[ "$(cmd $PARENT)" == *"eval/run85.sh"* ]] && kill $PARENT 2>/dev/null
[[ "$(cmd $TR)" == *"-m src.train"* ]] && kill $TR
while kill -0 $TR 2>/dev/null; do sleep 2; done
sleep 5
if pgrep -f '^(\S*/)?python[0-9.]* -u -m src\.train' >/dev/null; then echo "a trainer is still alive after the pause; aborting the A/B" >> logs/train85.out; exit 1; fi
cp $CK /mnt/kv_cache/run85_s$st.pt
echo "=== RUN85 PAUSED $(date -u '+%Y-%m-%d %H:%M UTC') at its step-$st checkpoint for the recall A/B (eval/recall_ab.sh) ===" >> logs/train85.out
echo "- $(date -u '+%Y-%m-%d %H:%M:%S UTC') **run85 paused** at its step-$st checkpoint (wrapper $KT then trainer $TR, both verified dead) for the recall A/B." >> PROGRESS.md
eval/recall_ab.sh /mnt/kv_cache/run85_s$st.pt 4000
# run85 resumes once the review's fixes are in (flag file), or after 30 minutes regardless (the card must not sit idle)
for k in $(seq 1 60); do [ -f eval/.run85b_ok ] && break; sleep 30; done
setsid nohup eval/run85b.sh /mnt/kv_cache/run85_s$st.pt > logs/run85b.out 2>&1 < /dev/null &
