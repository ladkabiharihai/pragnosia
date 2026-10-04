#!/bin/bash
# RUN80 (2026-10-02): the 98M lineage with CONSISTENT sliding-window attention, deciding its own REACH. User: "98M is
# sufficient to test the model" and close the gaps before scaling, so the only growth offered is R (double the attention
# window) vs staying, by the same trials as width and depth, judged on a held-out sample that mirrors the training mix
# (25% lookup + needle documents) at 8,192 tokens. From swa_512 (step 198,000; sliding 512, its own optimizer),
# 40,000 steps (~2.6B tokens), then the vision pipeline. NO SKIP GUARD (--spike 1e9): with it, four attempts entered a
# runaway of skipped steps at 199,000-201,200 (cause not yet understood; a skipped step is a true no-op since the
# optimizer fix); without it swa_8192 trained 6,000 steps cleanly. Schedule-free Prodigy's per-tensor rms clip bounds
# every update, and finiteness and the held-out evals still stop a real divergence. (factual fine-tune, own-mistake abstention, scorecard).
set -u
cd "$(dirname "$0")/.."
# a fresh start: no checkpoint of an earlier attempt may be left for the wrapper to resume from (attempt 4 resumed
# attempt 3's step-200,000 file, 2026-10-02)
mkdir -p /mnt/kv_cache/run80; rm -f /mnt/kv_cache/run80/ckpt.pt /mnt/kv_cache/run80/ckpt_prev.pt /mnt/kv_cache/run80/resume_from.pt
cp /mnt/kv_cache/swa_512/ckpt.pt /mnt/kv_cache/run80/seed.pt
C="--steps 238000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp
   --attn-sliding 1 --attn-window 512 --trial-steps 1000 --trial-frac 16 --resume-stage 198000,4000
   --heldout-T 8192 --heldout-data2 1 --spike 1e9
   --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0
   --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 4000 --eval-every 8000 --cap 34"
echo "=== RUN80 START $(date -u '+%Y-%m-%d %H:%M UTC') ===" > logs/train80.out
eval/keep_training.sh logs/train80.out TRAIN80_DONE $C --grow-auto "R,R,R,R" --resume /mnt/kv_cache/run80/seed.pt --out /mnt/kv_cache/run80 || exit 1
sed -i 's#logs/train_grow70.out#logs/train80.out#' eval/scorecard.sh
cp /mnt/kv_cache/run80/ckpt.pt /mnt/kv_cache/run80_final.pt
exec eval/eval_ckpt.sh /mnt/kv_cache/run80_final.pt r80
