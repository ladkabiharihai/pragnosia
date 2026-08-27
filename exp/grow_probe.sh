#!/usr/bin/env bash
# After a growth event: test ONE checkpoint BELOW the grow and TWO ABOVE it (user's request), so we can see
# what the model was before it grew and how it recovers over the two saves that follow.
# Usage: exp/grow_probe.sh <grow_step>
G=${1:?usage: grow_probe.sh <grow_step>}
D=/mnt/kv_cache/pragnosia_data
below=$(( (G/500)*500 ))                        # last save strictly before the grow
a1=$(( below+500 )); a2=$(( below+1000 ))
echo "growth at step $G -> testing below=$below, above=$a1, $a2"
for s in $below $a1 $a2; do
  f=$D/probe_$s.pt
  if [ ! -f "$f" ]; then echo "  waiting for step $s ..."; fi
done
