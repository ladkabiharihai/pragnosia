#!/bin/bash
# FROM SCRATCH, ONE DOCUMENTED RECIPE. The old lineage went through ~10 continuation phases, each chosen by looking
# at the same probes; a model built by one recipe from random weights is the one a technical reviewer can reproduce.
# Everything measured on 2026-09-22/23 is applied at once:
#   shape    24 layers x d768, 1-in-4 attention (window 512): 18 carrier + 6 attention layers, ~186M parameters
#   speed    fused scan (identical maths, +9-11%), fused projections (same function), equal-token batching over
#            the windows, fused optimizer, CUDA-graphed step; the dead attention state read is off
#   data     60% prose (web, educational, a little maths and arXiv, no code) + 40% a file of all 5.94B tokens of
#            Wikipedia with the 46M-token Q&A set three times in it: ~30% web, ~23% educational, ~41% Wikipedia,
#            ~5% maths/arXiv, ~1% Q&A from the very first step (knowledge is more answerable when Q&A is mixed
#            into pretraining than added after)
#   schedule trust-ratio 1.5e-3 peak (the rate the 12-layer model trained at for 10B tokens), 2,000 warmup steps,
#            cosine to 5%; windows 2,048 and 8,192 at 16,384 tokens a micro-step, 4 micro-steps an update
#   length   183,000 updates x 65,536 tokens = 12.0B tokens; checkpoints every 4,000 updates
# Before it starts: the fused projections timed against the plain ones on this exact shape, so the speed claim is
# a measurement.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
while pgrep -f "[e]val/qa.py" > /dev/null; do sleep 15; done
COMMON="--d 768 --layers 24 --ctx 8192 --batch 2 --accum 4 --mixer rspin --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0"
echo "=== fused projections off vs on, 24 layers, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > eval/scratch_speed.txt
for FP in 0 1; do
  $P -u -m src.train --steps 300 $COMMON --lr 1.5e-3 --warm 50 --fused-proj $FP --data /mnt/kv_cache/tokfull_v8192 \
    --eval-every 100000 --save-every 100000 --cap 34 --out /mnt/kv_cache/scratch_speedtmp > logs/scratch_speed_$FP.out 2>&1
  echo "fused-proj $FP: $(grep -E '300/300' logs/scratch_speed_$FP.out | grep -oE '[0-9]+ ktok/s') $(grep -oE '[0-9.]+ GiB reserved after capture|OutOfMemory|Error' logs/scratch_speed_$FP.out | head -n 1)" >> eval/scratch_speed.txt
done
rm -rf /mnt/kv_cache/scratch_speedtmp
mkdir -p /mnt/kv_cache/run51_scratch
$P -u -m src.train --steps 183000 $COMMON --lr 1.5e-3 --warm 2000 --fused-proj 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 4000 --eval-every 4000 --cap 34 --out /mnt/kv_cache/run51_scratch > logs/train_scratch.out 2>&1
