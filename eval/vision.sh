#!/bin/bash
# THE VISION GOALS THAT HAD NO EVIDENCE, one command, one job at a time. Then the head-to-head.
#   goal 5  abstains on what it does not know           eval/abstain.py    -> eval/abstain.txt
#   goal 7  continuous learning                         eval/continual.py  -> eval/continual.txt
#   goal 7  continuous growth                           eval/growth.sh     -> eval/growth.txt
#   goal 6  consumer GPU training, inference in 4 GB    eval/consumer.sh   -> eval/consumer.txt
#   goal 1  surpass the LLMs, controlled                eval/headtohead.sh -> eval/headtohead.txt
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
exec 9>/tmp/claude-1000/-opt-code-membrae/ab0dd831-724a-4bd0-be9e-dd7e1ea6f491/scratchpad/gpu.queue.lock
flock 9
clean () { grep -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" ; }
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/abstain.txt;  $P eval/abstain.py   2>&1 | clean >> eval/abstain.txt
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/continual.txt; $P eval/continual.py 2>&1 | clean >> eval/continual.txt
eval/growth.sh
eval/consumer.sh
echo "VISION_DONE $(date -u '+%H:%M UTC')" >> eval/consumer.txt
flock -u 9
eval/headtohead.sh
