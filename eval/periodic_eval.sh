#!/bin/bash
# THE SCORECARD EVERY 6 HOURS (user, 2026-10-05) while run85 trains: the latest checkpoint is copied and the whole pipeline
# runs on it beside the training -> eval/scorecard_r85_s<step>.txt. Decode-speed numbers from these runs share the card and
# are NOT used. (2026-10-06) Every job capped at 6 GiB (MEMBRAE_CAP_MAX) AND run under eval/guarded.sh, which kills the
# job's own PIDs if their real GPU memory passes 6 GiB: prod ~51 + the trainer 34 + this 6.5 stay under the card. The
# fine-tunes now use the fused scan (the python scan's backward allocated ~10 GiB outside the cap; 15.4 GiB observed).
# Stops when run85's trainer is gone.
set -u
cd "$(dirname "$0")/.."
sed -i "s|logs/[a-z_0-9]*\.out >> \$OUT$|logs/train85.out >> \$OUT|" eval/scorecard.sh
# (an automatic resume appends --resume after --out, so the pattern must not anchor on the end of the line)
TRPAT='^(\S*/)?python[0-9.]* -u -m src\.train .*--out /mnt/kv_cache/run85( |$)'
WAIT=${FIRST_SLEEP:-21600}
while pgrep -f "$TRPAT" >/dev/null; do
  sleep $WAIT; WAIT=21600
  pgrep -f "$TRPAT" >/dev/null || break
  [ -f /mnt/kv_cache/run85/ckpt.pt ] || continue
  ST=$(.venv/bin/python -c "import torch;print(torch.load('/mnt/kv_cache/run85/ckpt.pt',map_location='cpu',weights_only=False)['step'])")
  cp /mnt/kv_cache/run85/ckpt.pt /mnt/kv_cache/run85_s$ST.pt
  echo "periodic scorecard at step $ST, $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/periodic_eval.out
  MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 eval/eval_ckpt.sh /mnt/kv_cache/run85_s$ST.pt r85_s$ST >> logs/periodic_eval.out 2>&1
done
echo "periodic scorecards stopped $(date -u '+%Y-%m-%d %H:%M UTC'): run85's trainer is gone" >> logs/periodic_eval.out
