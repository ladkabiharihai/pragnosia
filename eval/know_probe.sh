#!/bin/bash
# SCORE THE KNOWLEDGE + ARITHMETIC RUN AS IT GOES (2026-10-08): every checkpoint at a multiple of 20,000 steps (and the 382k
# start once), under the side-job lock (one side job at a time beside the trainer and the final pipeline): same-text bits per
# byte (eval/bpb_ckpt.py), the benchmark suite on ours only (facts, arithmetic 4-way and exact, reasoning, HellaSwag, PIQA,
# ARC, OpenBookQA, LAMBADA, GSM8K), and learning by reading at 8k / 32k (it must survive). Output: eval/know_probe.txt
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/know_probe.txt; F="Warning|^\[gpu\]|Loading|torch_dtype|Token indices|\[base"
probe () {  # $1 ckpt  $2 label
  echo "==================== $2 ($1), $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
  MEMBRAE_CAP_MAX=6 eval/guarded.sh 7 $P -u eval/bpb_ckpt.py $1 2>&1 | grep --line-buffered -vE "$F" >> $O
  MEMBRAE_CAP_MAX=6 BENCH_HF=0 N=500 eval/guarded.sh 7 $P -u eval/bench_suite.py $1=$2 2>&1 | grep --line-buffered -vE "$F" | grep -vE "^\s+\[" >> $O
  MEMBRAE_CAP_MAX=6 N_Q=100 DIST=8192,32768 eval/guarded.sh 7 $P -u eval/learn_by_reading.py $1=$2 2>&1 | grep --line-buffered -vE "$F" | grep -E "^$2|^model" >> $O
}
[ -f $O ] || probe /mnt/kv_cache/final_base_fr382.pt start-382k
done_=" "
while :; do
  for c in $(ls /mnt/kv_cache/know/ckpt_s*.pt 2>/dev/null | sort -V); do
    s=$(basename $c .pt | sed 's/ckpt_s//')
    if [ $((s % 20000)) -eq 0 ] && [[ "$done_" != *" $s "* ]]; then probe $c know-$s; done_="$done_$s "; fi
  done
  grep -qE "KNOW_DONE|KNOW_FAILED" logs/know2.out 2>/dev/null && [ -z "$(ls /mnt/kv_cache/know/ckpt_s*.pt 2>/dev/null | sed 's/.*ckpt_s//; s/.pt//' | awk -v d="$done_" '$1 % 20000 == 0 && index(d, " " $1 " ") == 0')" ] && break
  sleep 300
done
echo "KNOW_PROBE_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
