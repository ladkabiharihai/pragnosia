#!/bin/bash
# PAST THE 8,192-TOKEN WINDOW: does training with WRAPPED positions fix "more context, worse prediction"? (2026-10-01)
# Measured: the same final 256 tokens score 1.1195 bits/byte after 8,192 tokens of context and 1.1430 after 32,768
# to 1M, whatever the positions are set to at inference. Training never shows a state carried past one window. This arm
# is eval/conv_ab.sh's control in every respect (r98r, cold optimizer, seed 2, 25% lookup documents, 8,000 steps) except
# that a third of the windows are 32,768 tokens on the 8,192-entry position table (--pos-table 8192), so positions wrap
# inside a training window with the state full -- what inference does at every boundary. Then the same-256-token test
# on both arms.
# ARM / DATA2 (environment) name the arm and its second corpus: ARM=needle DATA2=/mnt/kv_cache/tokneedle_v8192 is the
# same arm with needle documents (eval/build_recall.py needle) beside the lookup ones, 50/50.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; ARM=${ARM:-long}; DATA2=${DATA2:-/mnt/kv_cache/tokrecall_v8192}
mkdir -p /mnt/kv_cache/ab_$ARM; cp /mnt/kv_cache/ab_ctrl/seed.pt /mnt/kv_cache/ab_$ARM/seed.pt
# batch 1 x accumulation 2: the batch is sized from the LONGEST window (tokens per micro-step = batch x 32,768), so
# this is the control's 65,536 tokens per optimizer step (batch 2 x 8,192 x 4); the first try at batch 2 asked for 4x
C="--steps 200000 --d 512 --layers 12 --ctx 8192 --batch 1 --accum 2 --mixer grspin --norm rms --optim sfp
   --ctx-mix 2048,8192,32768 --pos-table 8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 2
   --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 $DATA2 --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34"
: > logs/ab_$ARM.out
eval/keep_training.sh logs/ab_$ARM.out AB_${ARM^^}_DONE $C --resume /mnt/kv_cache/ab_$ARM/seed.pt --out /mnt/kv_cache/ab_$ARM || exit 1
O=eval/longctx_ab_$ARM.txt
{ echo "LONG-WINDOW A/B ($ARM, second corpus $DATA2), $(date -u '+%Y-%m-%d %H:%M UTC')"; grep -h "eval step\|micro-step\|split\|tok/s (last" logs/ab_$ARM.out | tail -5; } > $O
for arm in ctrl $ARM; do echo "== $arm" >> $O; $P eval/million.py /mnt/kv_cache/ab_$arm/ckpt.pt 8192,32768,131072 2>&1 | grep -E "tokens of context" >> $O; done
echo "LONGCTX_AB_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
