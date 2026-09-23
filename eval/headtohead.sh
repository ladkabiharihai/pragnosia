#!/bin/bash
# THE HEAD-TO-HEAD. Our mixer against a transformer, both trained FROM SCRATCH, identical in everything except
# the mixer: same data (the full designed mix), same token count, same width and depth (99.5M against 97.6M
# parameters, within 2%, ours slightly larger), same embeddings, learned positions, feed-forward blocks,
# optimizer, schedule, window and seed. The transformer is full causal attention with QK-norm.
#
# Why this and not GPT-2: every win against GPT-2 is exposed to "a 2019 model" and "different data". This
# removes both, and isolates the one thing the project claims to have changed.
#
# Fairness of the learning rate: each architecture gets its OWN, chosen by the same procedure -- a factor-of-two
# grid around our usual setting on 0.2B tokens, lowest held-out bits per byte wins -- so neither is handicapped
# by a rate tuned for the other.
#
# usage: eval/headtohead.sh        logs: logs/h2h_*.out, summary appended to eval/headtohead.txt
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
SUM=eval/headtohead.txt
exec 9>/tmp/claude-1000/-opt-code-membrae/ab0dd831-724a-4bd0-be9e-dd7e1ea6f491/scratchpad/gpu.queue.lock
flock 9
echo "=== HEAD-TO-HEAD started $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $SUM
train () {   # mixer lr steps outdir log
  $P -u -m src.train --steps $3 --d 768 --ctx 8192 --batch 1 --accum 4 --mixer $1 --lr $2 \
    --scan-chunk 64 --chunks 4 --warm 500 --data /mnt/kv_cache/tokfull_v8192 \
    --save-every 4000 --eval-every 2000 --cap 28 --seed 0 --out $4 > $5 2>&1
}
lastbpb () { grep -E "eval step" "$1" | tail -n 1 | grep -oE "BPB [0-9.]+" | awk '{print $2}'; }
# 1. learning-rate sweep, 6,000 steps = 0.197B tokens each
for MX in rspin attn; do
  best=""; bestlr=""
  for LR in 7.5e-4 1.5e-3 3e-3; do
    D=/mnt/kv_cache/h2h_sweep_${MX}_${LR}; L=logs/h2h_sweep_${MX}_${LR}.out
    train $MX $LR 6000 $D $L
    b=$(lastbpb $L); echo "  sweep $MX lr $LR: bpb ${b:-FAILED}" >> $SUM
    rm -rf $D
    if [ -n "$b" ] && { [ -z "$best" ] || awk "BEGIN{exit !($b < $best)}"; }; then best=$b; bestlr=$LR; fi
  done
  echo "  -> $MX trains at lr $bestlr (sweep bpb $best)" >> $SUM
  eval "LR_$MX=$bestlr"
done
# 2. the real runs, 92,000 steps = 3.01B tokens each, each at its own best rate
for MX in rspin attn; do
  eval "LR=\$LR_$MX"
  D=/mnt/kv_cache/h2h_${MX}; L=logs/h2h_${MX}.out
  train $MX $LR 92000 $D $L
  echo "  full $MX at lr $LR: final bpb $(lastbpb $L)   $(grep -E 'END train' $L | cut -c14-60)" >> $SUM
done
echo "=== training done $(date -u '+%H:%M UTC') ===" >> $SUM
