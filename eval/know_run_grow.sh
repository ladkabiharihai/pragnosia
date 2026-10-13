#!/bin/bash
# THE KNOWLEDGE + ARITHMETIC PHASE (2026-10-08, approved by the user). The final base (far-read 382k) knew few facts (2 of 10
# everyday facts, SmolLM2 8) and no arithmetic (0 of 8; 4-way arithmetic 8.3%, below chance): ~25B tokens of mostly web text.
# ~3 days more from 382k with the mix moved toward knowledge-dense text and arithmetic (--weights, set by hand: the
# model-chosen mix degraded the model on 2026-10-07; a goal-8 compromise, stated):
#   main 78.5%: fineweb-edu 34, wikipedia 20 (one pass of all 5.9B tokens in ~30B), dclm 16, finemath 4, arxiv 4
#   data2 21.5%: synthetic arithmetic 6 (eval/build_arith.py), Orca-Math / MetaMath / GSM8K-train 3, chat Q&A 1 (train
#   splits only), retrieval 11.5 (far lookup 3, far reading 3, recall lists 2.5, needles 1.5, far needles 1.5) so reading survives
# (2026-10-08 ~14:00 UTC, the USER: growth must ALWAYS be enabled) restarted from its 385,000 checkpoint WITH the model's growth
# trials, as run85: at each doubling of the stage, wider / deeper / longer reach vs unchanged on the same batches, judged on
# 32k-token held-out streams, time-priced. (2026-10-09) micro-batch 1 x 8 accumulation (the same 8 sequences per update):
# at 2 x 4 a widened trial branch needed more than the 32 GiB cap with recall on; at 1 x 8 every branch fits (12 GiB base). Recipe otherwise as far-read (16 stream groups, carried state, step size held). Checkpoints every 5,000 steps kept;
# eval/know_probe.sh scores them. To stop on purpose: kill keep_training.sh first, then the trainer.
cd "$(dirname "$0")/.."
OUT=/mnt/kv_cache/know; cp $OUT/ckpt_s385000.pt $OUT/resume_385000.pt
L=logs/know2.out; echo "=== KNOWLEDGE + ARITHMETIC RUN, RESTARTED WITH GROWTH TRIALS $(date -u '+%Y-%m-%d %H:%M UTC') ===" >> $L
( m0=0; while :; do sleep 120; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 30; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -qE "KNOW_DONE|KNOW_FAILED" $L && break; done ) &
eval/keep_training.sh $L KNOW_DONE --steps 682000 --d 512 --layers 12 --ctx 8192 --batch 1 --accum 8 --mixer grspin --norm rms \
  --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 --carry-half 1 --grow-auto "R|W|d,R|W|d,R|W|d,R|W|d,R|W|d,R|W|d" --trial-steps 1000 --trial-frac 16 --heldout-long 32768 --spike 1e9 \
  --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 29 --freeze-d 1 --data /mnt/kv_cache/tokfull_v8192 \
  --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokknow --data2-frac 0.215 \
  --weights fineweb-edu:34,wikipedia:20,dclm:16,finemath:4,arxiv:4,arith:6,instr_math:3,instr_chat:1,farlookup:1,farlookup2:2,farread:0.5,farread2:2.5,recall:2.5,needle:1.5,farneedle:1.5 \
  --save-every 5000 --eval-every 5000 --cap 32 --resume $OUT/resume_385000.pt --out $OUT
