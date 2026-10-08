#!/bin/bash
# WHY DOES CONTINUED TRAINING FROM 368k DEGRADE? (2026-10-07) Two 4,000-step arms from the best checkpoint, recall off, step
# size held, held-out every 1,000: A main corpora only; B main + 25% retrieval documents (run85e's recipe, which drifted
# 1.0372 -> 1.0488 over 28k steps). If A holds and B drifts, the repeated retrieval documents are the cause.
cd "$(dirname "$0")/.."
for arm in A B; do
  OUT=/mnt/kv_cache/drift_$arm; mkdir -p $OUT; rm -f $OUT/*; cp /mnt/kv_cache/run85_s368000_c.pt $OUT/seed.pt
  X=""; [ $arm = B ] && X="--data2 /mnt/kv_cache/tokneedle_mix --data2-frac 0.25"
  echo "=== DRIFT ARM $arm $(date -u '+%H:%M UTC') ===" > logs/drift_$arm.out
  .venv/bin/python -u -m src.train --steps 372000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
    --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --spike 1e9 \
    --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 24 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
    --sources dclm,fineweb-edu,wikipedia,finemath,arxiv $X --save-every 100000 --eval-every 1000 --cap 30 \
    --resume $OUT/seed.pt --out $OUT >> logs/drift_$arm.out 2>&1
done
echo "DRIFT_TEST_DONE $(date -u '+%H:%M UTC')" >> logs/drift_B.out
