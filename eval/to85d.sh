#!/bin/bash
# Restart run85 with recall (M) back among its options, at run85c's first checkpoint after its 360,400 trial (step >= 368,000).
# Stop order: keep_training wrapper, verified dead, then the trainer; the trainer check is anchored (no shell can match it).
set -u
cd "$(dirname "$0")/.."
CK=/mnt/kv_cache/run85/ckpt.pt
cmd () { tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null; }
TRPAT='^(\S*/)?python[0-9.]* -u -m src\.train'
m0=$(stat -c %Y $CK)
while true; do
  sleep 30
  m1=$(stat -c %Y $CK 2>/dev/null || echo 0); [ "$m1" != "$m0" ] || continue
  s1=$(stat -c %s $CK); sleep 15; s2=$(stat -c %s $CK); [ "$s1" = "$s2" ] || continue
  st=$(.venv/bin/python -c "import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['step'])" 2>/dev/null || echo -1)
  m0=$m1; [ "$st" -ge 368000 ] && break
done
KT=$(pgrep -f "^/bin/bash eval/keep_training.sh logs/train85.out"); TR=$(pgrep -f "$TRPAT"); PC=$(pgrep -f "^/bin/bash eval/run85c.sh")
[[ "$(cmd $KT)" == *"keep_training.sh logs/train85.out"* ]] && kill $KT
sleep 3; if kill -0 $KT 2>/dev/null; then echo "wrapper $KT alive; aborting" >> logs/train85.out; exit 1; fi
[ -n "$PC" ] && kill $PC 2>/dev/null
[[ "$(cmd $TR)" == *"-m src.train"* ]] && kill $TR
while kill -0 $TR 2>/dev/null; do sleep 2; done; sleep 5
if pgrep -f "$TRPAT" >/dev/null; then echo "a trainer is still alive; not starting run85d" >> logs/train85.out; exit 1; fi
cp $CK /mnt/kv_cache/run85_s$st.pt
echo "- $(date -u '+%Y-%m-%d %H:%M:%S UTC') run85c stopped at its step-$st checkpoint (wrapper $KT, trainer $TR verified dead); run85d starts from it with M offered, cap 32." >> PROGRESS.md
setsid nohup eval/run85d.sh /mnt/kv_cache/run85_s$st.pt > logs/run85d.out 2>&1 < /dev/null &
