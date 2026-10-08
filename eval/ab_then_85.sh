#!/bin/bash
# the recall A/B from run85's step-368000 checkpoint, then run85 resumes (after the review's fixes: eval/.run85b_ok, or 30 min)
cd "$(dirname "$0")/.."
echo "- $(date -u '+%Y-%m-%d %H:%M:%S UTC') **run85 stopped at its step-368000 checkpoint** (eval/pause85_ab2.sh: wrapper then trainer, verified dead); its trainer check then matched a shell's command line and aborted the A/B, so the card was idle ~5 min. Relaunched the A/B by hand (anchored trainer pattern)." >> PROGRESS.md
eval/recall_ab.sh /mnt/kv_cache/run85_s368000.pt 4000
for k in $(seq 1 60); do [ -f eval/.run85b_ok ] && break; sleep 30; done
eval/run85b.sh /mnt/kv_cache/run85_s368000.pt
