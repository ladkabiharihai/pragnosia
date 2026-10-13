#!/bin/bash
# CHOOSE THE FINAL BASE AND START THE FINAL PIPELINE (2026-10-11, while the user is away: "take decisions yourself"). The rule, fixed
# before the numbers: the reading restore (598k) replaces 590k only if reading invented facts at 8k + 32k gains >= 10 points and
# same-text BPB is no worse than +0.004; otherwise 590k. Then eval/final_pipeline2.sh (side slot: self-improvement, fine-tune,
# abstention, scorecard, self-consistency, open-book, reading, long generation, benchmarks) and eval/final_base_evals.sh (trainer
# slot: benchmarks, needles to 1M, long generation) on it, tag v2.
cd "$(dirname "$0")/.."
until grep -q RESTORE_EVAL_DONE eval/restore_eval.txt 2>/dev/null; do sleep 60; done
read B590 B598 R590 R598 < <(.venv/bin/python - <<'PY'
import re
t = open("eval/restore_eval.txt").read()
bpb = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"step (\d+)\s+BPB ([0-9.]+)", t)}
rd = {}
for lab in ("know-590k", "restore-598k"):
    m = re.search(rf"(?m)^{lab}\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%", t)
    rd[lab] = float(m.group(2)) + float(m.group(3)) if m else -1
print(bpb.get(590000, 9), bpb.get(598000, 9), rd["know-590k"], rd["restore-598k"])
PY
)
if .venv/bin/python -c "import sys; sys.exit(0 if ($R598 >= $R590 + 10 and $B598 <= $B590 + 0.004) else 1)"; then
  BASE=/mnt/kv_cache/restore/ckpt_s598000.pt; WHY="598k: reading 8k+32k $R598 vs $R590, BPB $B598 vs $B590"
else
  BASE=/mnt/kv_cache/know/ckpt_s590000.pt; WHY="590k: the restore did not meet the rule (reading 8k+32k $R598 vs $R590, BPB $B598 vs $B590)"
fi
cp $BASE /mnt/kv_cache/final_base_v2.pt
echo "- \`$(date -u '+%Y-%m-%d %H:%M') UTC\` **Final base chosen by the pre-set rule (eval/choose_final.sh): $WHY** -> /mnt/kv_cache/final_base_v2.pt; eval/final_pipeline2.sh and eval/final_base_evals.sh launched on it (tag v2)." >> PROGRESS.md
setsid nohup eval/final_pipeline2.sh /mnt/kv_cache/final_base_v2.pt v2 > /dev/null 2>&1 < /dev/null &
sleep 5
setsid nohup eval/final_base_evals.sh /mnt/kv_cache/final_base_v2.pt v2 > /dev/null 2>&1 < /dev/null &
