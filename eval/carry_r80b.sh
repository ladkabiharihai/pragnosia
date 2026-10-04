#!/bin/bash
# CARRIED-STATE TRAINING ON THE SLIDING-8K MODEL (2026-10-03; PROGRESS.md). run80 pays +0.033 +- 0.015 bits/byte at 32k of
# context vs 8k (lc80_ctrl), and 32k training windows only reduced it to +0.021 +- 0.010 at a cost in general quality and
# speed. Here the model trains exactly as inference runs: 8,192-token windows, each continuing its stream with every
# layer's state carried in (verified: two carried windows = one 16k forward, 2.40320 vs 2.40305), streams restarted every
# 16 steps (up to 131k tokens of history). Same start, data, seed and steps as lc80_ctrl. Then the same tests.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
mkdir -p /mnt/kv_cache/carry80b; rm -f /mnt/kv_cache/carry80b/ckpt*.pt /mnt/kv_cache/carry80b/resume_from.pt
cp /mnt/kv_cache/run80_final.pt /mnt/kv_cache/carry80b/seed.pt; : > logs/carry80b.out
eval/keep_training.sh logs/carry80b.out CARRY80B_DONE --steps 244000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 \
  --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 8192 --carry 16 --carry-groups 16 --carry-half 1 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 7 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25 --save-every 3000 --eval-every 3000 --cap 34 \
  --resume /mnt/kv_cache/carry80b/seed.pt --out /mnt/kv_cache/carry80b || exit 1
O=eval/carry_r80b.txt; echo "CARRIED-STATE TRAINING vs control, $(date -u '+%Y-%m-%d %H:%M UTC')" > $O
echo "== carry80b (half fresh, 16 rotating groups): $(grep -h 'eval step\|CARRIED' logs/carry80b.out | tr '\n' ' ')" >> $O
MILLION_SAMPLES=24 $P eval/million.py /mnt/kv_cache/carry80b/ckpt.pt 8192,32768,131072 2>&1 | grep "tokens of context" >> $O
NEEDLE_DIST=1024,4096,7000,16384 INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/carry80b/ckpt.pt needle 2>&1 | grep -E "distance" >> $O
echo "== control (lc80_ctrl, eval/longctx_r80.txt):" >> $O; sed -n '/== ctrl/,/== long/p' eval/longctx_r80.txt | grep -v "== long" >> $O
echo "CARRY_R80B_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
