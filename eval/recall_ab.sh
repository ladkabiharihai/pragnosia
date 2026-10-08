#!/bin/bash
# RECALL A/B (2026-10-05; user: "retrieval beyond 1k"). Untrained, recall (model.py Attn._recall: each sliding attention
# query also attends to its top-k keys older than the window) took needles at 1-32k from 0% to 100-31% but cost prediction
# past the first context window. Does TRAINING with it fix the cost and raise retrieval? Two arms from the same run85
# checkpoint, the same seed, data, steps and settings as run85 (carried windows, one shared step size, 25% retrieval
# documents); the only difference is --recall 16 vs 0. Run one after the other (sharing the card measures contention and
# puts prod at risk). Then run85 resumes from the same checkpoint with recall (M) among its growth options.
# usage: eval/recall_ab.sh <run85 checkpoint copy> [steps]
set -u
cd "$(dirname "$0")/.."
if pgrep -f '^(\S*/)?python[0-9.]* -u -m src\.train' >/dev/null; then echo "recall_ab: another trainer is running; refusing (two trainings fill the card, 2026-10-05)" >&2; exit 3; fi
SEED=$1; N=${2:-4000}
S0=$(.venv/bin/python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['step'])" $SEED)
COMMON=(--steps $((S0 + N)) --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp
  --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --spike 1e9 --scan-chunk 64
  --chunks 4 --fused-scan 1 --seed 16 --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
  --data2 /mnt/kv_cache/tokneedle_mix --data2-frac 0.25 --save-every $N --eval-every $N --cap 34 --resume $SEED)
for arm in 1 0; do
  out=/mnt/kv_cache/rc$arm; mkdir -p $out; rm -f $out/ckpt*.pt
  echo "=== RECALL A/B arm --recall $((arm * 16)) from $SEED (step $S0), $N steps, $(date -u '+%Y-%m-%d %H:%M UTC') ===" > logs/recall_ab_$arm.out
  .venv/bin/python -u -m src.train "${COMMON[@]}" --recall $((arm * 16)) --out $out >> logs/recall_ab_$arm.out 2>&1 \
    || echo "ARM $arm FAILED $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/recall_ab_$arm.out
done
echo "RECALL_AB_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/recall_ab_0.out
