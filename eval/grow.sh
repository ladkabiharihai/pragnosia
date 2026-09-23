#!/bin/bash
# FROM SCRATCH, GROWING: 12B tokens, one documented recipe, depth AND width grown in place with the function
# preserved exactly at every event (measured end to end: eval/growtest.sh, logs/growtest.out).
#   shape     grspin: 1-in-4 attention (window 512, 64-dim heads), gated-delta carrier (128-dim heads), RMSNorm
#   growth    0-3B   12 layers x d512   (46M)   ~467 ktok/s
#             3-6B   12 layers x d768   (98M)   ~294 ktok/s   widen at update 45,776
#             6-9B   24 layers x d768   (183M)  ~157 ktok/s   deepen at 91,552 (new layers ramp in over 1,000)
#             9-12B  24 layers x d1024  (319M)  ~101 ktok/s   micro-step split + widen at 137,328
#   data      60% prose (web, educational, Wikipedia, a little maths and arXiv, no code) + 40% all 5.94B tokens of
#             Wikipedia with the 46M-token Q&A set three times in it, Q&A mixed in from the first step
#   schedule  trust-ratio 1.5e-3 peak, 2,000 warm-up, cosine to 5%; windows 2,048 and 8,192, 65,536 tokens an update
#   cap       34 GiB (production shares the card); stage 4 measured 22 GiB
set -u
cd "$(dirname "$0")/.."
mkdir -p /mnt/kv_cache/run52_grow
.venv/bin/python -u -m src.train --steps 183105 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms \
  --lr 1.5e-3 --warm 2000 --new-lr 1e-3 --new-warm 1000 --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 0 \
  --grow-schedule 45776:w768,91552:d,137328:a2,137328:w1024 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokmixqa_full_v8192 --data2-frac 0.4 \
  --save-every 4000 --eval-every 4000 --cap 34 --out /mnt/kv_cache/run52_grow > logs/train_grow.out 2>&1
