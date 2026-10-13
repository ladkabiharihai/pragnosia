#!/bin/bash
# GOAL 6, TRAINING ON A CONSUMER GPU. The user's definition (2026-10-11): a 4-8 GB card, the one most people have, for training AND
# inference -- the first version of this demo (2026-10-07: 20.1 GiB under a 22 GiB cap, "a 24 GB RTX 4090") did not meet it.
# Now: the run's own recipe (carried 8,192-token windows, recall 16, schedule-free Prodigy, the step size held, growth trials ON)
# under a hard CAP (default 7.5 GiB, so the process with its CUDA context stays under 8 GB), with each layer recomputed in
# backward (--remat), micro-batch 1 x 8 accumulation (the same 8 sequences per update), 2 carried stream groups (NGRP=; not GROUPS, a bash built-in holding the group id 1000: it once made 1,000 groups), a growth trial at
# the start (--trial-now, short branches) shows growth bounded by the card: an option that does not fit is not a candidate.
# The process's real GPU memory (nvidia-smi, the CUDA context included) is sampled alongside.
# usage: eval/consumer_demo.sh [seed ckpt] [steps]      CAP=<GiB>
cd "$(dirname "$0")/.."
SEED=${1:-/mnt/kv_cache/know/ckpt_s590000.pt}; N=${2:-800}; CAP=${CAP:-7.5}
# 2026-10-13 (user: proceed with goal 6): CTX / ACCUM set the window (default 8,192 x 8 = 65,536 tokens per update, as the run;
# 2,048 x 32 keeps the tokens per update and cuts the captured graphs' activation pool, which scales with the window). OUT / L
# name the run. --pos-table keeps the checkpoint's 8,192-row position table (a shorter --ctx alone rebuilt it at the window size and
# the optimizer state no longer matched: "starting cold", then a crash in the compiled update). Takes the side lock (one side job at a time) unless NOLOCK=1 -- the main trainer is running again.
CTX=${CTX:-8192}; ACCUM=${ACCUM:-$((65536 / CTX))}
if [ -z "${NOLOCK:-}" ]; then exec 9>/tmp/membrae_side_gpu.lock; flock 9; fi
OUT=${OUT:-/mnt/kv_cache/consumer_demo}; L=${L:-logs/consumer_demo.out}; mkdir -p $OUT; cp $SEED $OUT/seed.pt
S0=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/seed.pt',map_location='cpu',weights_only=False)['step'])")
echo "=== CONSUMER-GPU TRAINING DEMO $(date -u '+%Y-%m-%d %H:%M UTC'): cap $CAP GiB, window $CTX x $ACCUM, from $SEED (step $S0), $N steps ===" > $L
MEMBRAE_NOLOCK=1 MEMBRAE_CAP_MAX=$CAP .venv/bin/python -u -m src.train --steps $((S0 + N)) --d 512 --layers 12 --ctx $CTX --pos-table ${PT:-8192} --batch 1 --accum $ACCUM \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups ${NGRP:-2} --carry-half 1 \
  --grow-auto "R|W|d,R|W|d" --trial-steps 100 --trial-frac 16 --trial-now ${TRIALNOW:-1} --heldout-long 32768 --remat 1 --spike 1e9 \
  --scan-chunk 64 --chunks 16 --fused-scan 1 --seed 34 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokrestore --data2-frac 0.30 \
  --weights fineweb-edu:30,wikipedia:18,dclm:14,finemath:4,arxiv:4,farread2:6,farread:1,farlookup2:4,farlookup:2,recall:4,needle:3,farneedle:3,arith:5,instr_math:2 \
  --save-every 1000000 --eval-every 1000000 --cap $CAP --resume $OUT/seed.pt --out $OUT >> $L 2>&1 &
TP=$!; PEAK=0
while kill -0 $TP 2>/dev/null; do
  u=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$TP '$1==p {print $2}')
  [ -n "$u" ] && [ "$u" -gt "$PEAK" ] && PEAK=$u; sleep 5
done
wait $TP; RC=$?
echo "CONSUMER_DEMO_DONE $(date -u '+%Y-%m-%d %H:%M UTC'): exit $RC, peak process GPU memory $PEAK MiB (nvidia-smi, CUDA context included)" >> $L
