#!/usr/bin/env bash
# Capture the checkpoint immediately BEFORE the first growth and immediately AFTER it, so we can test whether
# 22 tok/param of training at the infant size produced a model worth growing -- and whether the grow preserved it.
LOG=/opt/code/membrae/logs/train_pragnosia.log
CKPT=/mnt/kv_cache/pragnosia_data/pragnosia.pt
D=/mnt/kv_cache/pragnosia_data
step_of(){ /opt/code/parakeet_env/bin/python -c "import torch;print(torch.load('$CKPT',map_location='cpu')['step'])" 2>/dev/null; }
ngrow(){ awk '/===== train_pragnosia started/{c=1;n=0} c&&/SELF-GREW/{n++} END{print n+0}' "$LOG"; }
base=$(ngrow); echo "$(date +%H:%M:%S) watching (grows so far: $base)"
# keep a rolling copy of the newest pre-growth checkpoint
while [ "$(ngrow)" -eq "$base" ]; do
  s=$(step_of); [ -n "$s" ] && cp -f "$CKPT" "$D/gate_before.pt" 2>/dev/null && echo "$(date +%H:%M:%S) pre-growth snapshot at step $s"
  sleep 60
done
line=$(awk '/===== train_pragnosia started/{c=1;buf=""} c{buf=buf$0"\n"} END{print buf}' "$LOG" | grep SELF-GREW | tail -1)
gstep=$(echo "$line" | grep -oE 'step [0-9]+' | grep -oE '[0-9]+')
echo "$(date +%H:%M:%S) GROWTH: $line"
# wait for the first checkpoint written AFTER the grow
for i in $(seq 1 90); do
  cs=$(step_of)
  if [ -n "$cs" ] && [ "$cs" -gt "$gstep" ] 2>/dev/null; then
    cp -f "$CKPT" "$D/gate_after.pt"; echo "$(date +%H:%M:%S) SAVED gate_after.pt (grow at $gstep, ckpt at $cs)"; break
  fi
  sleep 20
done
echo "$(date +%H:%M:%S) done: gate_before.pt + gate_after.pt ready to compare"
