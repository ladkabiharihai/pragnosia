#!/bin/bash
# RUN A SIDE JOB UNDER A HARD MEMORY WATCHDOG (2026-10-06). A per-process memory fraction did not bound a fine-tune (15.4 GiB
# observed under an 8 GiB cap) and the card filled to 7.81 MiB beside the training and prod. This runs the command, polls
# nvidia-smi every 2 s for the GPU memory of the command's own process tree (its PIDs only), and kills exactly those PIDs
# if their sum passes the limit. The voice services allocate on demand: the card must never fill.
# usage: eval/guarded.sh <limit GiB> <command...>
set -u
LIM=$1; shift
# ONE SIDE JOB AT A TIME (2026-10-06): prod ~51 + the trainer 34 + ONE side job 6 is the card's budget; every side job takes
# this lock first and waits for the one before it (the 6-hourly scorecard and experiments alike)
# MEMBRAE_NOLOCK=1: a job in the TRAINER's slot while no trainer runs (prod ~51 + one side job 7 + this <= the card)
if [ -z "${MEMBRAE_NOLOCK:-}" ]; then exec 9>/tmp/membrae_side_gpu.lock; flock 9; fi
export MEMBRAE_CAP_MAX=${MEMBRAE_CAP_MAX:-$LIM}
"$@" &
ROOT=$!; PEAK=0
tree () { local p=$1; echo $p; for c in $(pgrep -P $p); do tree $c; done; }
while kill -0 $ROOT 2>/dev/null; do
  sleep 2
  pids=" $(tree $ROOT | tr '\n' ' ') "
  used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
         awk -F', ' -v P="$pids" 'index(P, " " $1 " ") {s += $2} END {print s + 0}')
  [ "$used" -gt "$PEAK" ] && PEAK=$used
  if [ "$used" -gt $((LIM * 1024 + 512)) ]; then
    echo "GUARD $(date -u '+%H:%M:%S UTC'): $used MiB > $LIM GiB; killing $pids ($*)" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | awk -F', ' -v P="$pids" 'index(P, " " $1 " ")' |
      while IFS=', ' read -r p_ u_ rest_; do echo "   pid $p_ $u_ $rest_: $(tr '\0' ' ' < /proc/$p_/cmdline 2>/dev/null | cut -c1-80)" >&2; done
    for p in $pids; do kill $p 2>/dev/null; done
    sleep 3; for p in $pids; do kill -9 $p 2>/dev/null; done
    wait $ROOT 2>/dev/null; exit 99
  fi
done
wait $ROOT; rc=$?
echo "GUARD: peak $PEAK MiB of $LIM GiB ($*)" >&2
exit $rc
