#!/usr/bin/env bash
# Watch for the NEXT growth event; the moment it fires, snapshot the checkpoint written just after it.
# This gives (a) the exact step, so we can tell whether the sensor is rate-limited, and (b) a clean resume point.
LOG=/opt/code/membrae/logs/train_pragnosia.log
CKPT=/mnt/kv_cache/pragnosia_data/pragnosia.pt
before=$(awk '/===== train_pragnosia started/{c=1;n=0} c&&/SELF-GREW/{n++} END{print n+0}' "$LOG")
echo "$(date +%H:%M:%S) watching: $before grows so far"
while true; do
  now=$(awk '/===== train_pragnosia started/{c=1;n=0} c&&/SELF-GREW/{n++} END{print n+0}' "$LOG")
  if [ "$now" -gt "$before" ]; then
    line=$(awk '/===== train_pragnosia started/{c=1;buf=""} c{buf=buf$0"\n"} END{print buf}' "$LOG" | grep SELF-GREW | tail -1)
    gstep=$(echo "$line" | grep -oE 'step [0-9]+' | grep -oE '[0-9]+')
    echo "$(date +%H:%M:%S) GROWTH: $line"
    # wait for the first checkpoint written AFTER the grow (saves every 500 steps)
    for i in $(seq 1 60); do
      cs=$(/opt/code/parakeet_env/bin/python -c "import torch;print(torch.load('$CKPT',map_location='cpu')['step'])" 2>/dev/null)
      if [ -n "$cs" ] && [ "$cs" -gt "$gstep" ] 2>/dev/null; then
        cp "$CKPT" /mnt/kv_cache/pragnosia_data/grow_${now}_step${cs}.pt
        echo "$(date +%H:%M:%S) SAVED grow_${now}_step${cs}.pt (grow at $gstep, checkpoint at $cs)"
        break
      fi
      sleep 20
    done
    break
  fi
  sleep 20
done
