#!/bin/bash
# THE MATCHED TRANSFORMER CONTROL (2026-10-04; user approved). Our comparisons so far are against GPT-2 and SmolLM2, which
# differ in data and tokens. Here: two 98M models FROM SCRATCH on the same data (main mix + 25% lookup/needle documents),
# the same tokens (46,000 steps x 65,536 = 3.0B), the same optimizer (schedule-free Prodigy, one shared step size, no rate),
# the same seed, the same windows (2,048 / 8,192):
#   ours  the current design: gated delta-rule carrier, 1 layer in 4 sliding-window attention (1,024), d768, 12 layers (97.7M)
#   attn  a standard transformer: full causal attention every layer, learned positions, d768, 12 layers (97.6M)
# Then the full vision pipeline on both -> eval/scorecard_m_ours.txt, eval/scorecard_m_attn.txt.
set -u
cd "$(dirname "$0")/.."
C="--steps 46000 --d 768 --layers 12 --ctx 8192 --batch 2 --accum 4 --norm rms --optim sfp --share-d 1 --spike 1e9
   --ctx-mix 2048,8192 --chunks 4 --seed 21 --data /mnt/kv_cache/tokfull_v8192
   --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_v8192 --data2-frac 0.25
   --save-every 4000 --eval-every 4000 --cap 34"
for arm in ours attn; do
  mkdir -p /mnt/kv_cache/m_$arm; rm -f /mnt/kv_cache/m_$arm/ckpt*.pt /mnt/kv_cache/m_$arm/resume_from.pt; : > logs/m_$arm.out
  if [ $arm = ours ]; then X="--mixer grspin --attn-sliding 1 --attn-window 1024 --scan-chunk 64 --fused-scan 1"; else X="--mixer attn"; fi
  eval/keep_training.sh logs/m_$arm.out M_DONE $C $X --out /mnt/kv_cache/m_$arm || exit 1
  cp /mnt/kv_cache/m_$arm/ckpt.pt /mnt/kv_cache/m_${arm}_final.pt
done
for arm in ours attn; do
  sed -i "s|logs/[a-z_0-9]*\.out >> \$OUT$|logs/m_$arm.out >> \$OUT|" eval/scorecard.sh
  eval/eval_ckpt.sh /mnt/kv_cache/m_${arm}_final.pt m_$arm
done
echo "MATCHED_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> logs/m_attn.out
