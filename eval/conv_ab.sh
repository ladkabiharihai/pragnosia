#!/bin/bash
# IN-CONTEXT LOOKUP: DOES A SHORT CONVOLUTION IN THE CARRIER FIX IT? (2026-10-01; PROGRESS.md)
# Two arms from the same 98M checkpoint (r98r, step 192,000), the same data in the same order (seed 2, 25% lookup
# documents), the same 8,000 steps, both with a cold optimizer (the conv arm must start cold: its new weights have no
# state). The only difference: arm B adds identity-initialised short causal convolutions (K=4) on every carrier's
# q, k, v (verified exact: same loss to 6 decimals, trunk difference 0.0). Then the powered lookup test on both.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
mkdir -p /mnt/kv_cache/ab_ctrl /mnt/kv_cache/ab_conv
$P -c "
import torch; sd = torch.load('/mnt/kv_cache/r98r/ckpt.pt', map_location='cpu', weights_only=False); sd.pop('opt', None)
torch.save(sd, '/mnt/kv_cache/ab_ctrl/seed.pt'); torch.save(sd, '/mnt/kv_cache/ab_conv/seed.pt')"
C="--steps 200000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp
   --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 2
   --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv
   --data2 /mnt/kv_cache/tokrecall_v8192 --data2-frac 0.25 --save-every 4000 --eval-every 4000 --cap 34"
: > logs/ab_ctrl.out; : > logs/ab_conv.out
eval/keep_training.sh logs/ab_ctrl.out AB_CTRL_DONE $C --resume /mnt/kv_cache/ab_ctrl/seed.pt --out /mnt/kv_cache/ab_ctrl || exit 1
eval/keep_training.sh logs/ab_conv.out AB_CONV_DONE $C --short-conv 4 --resume /mnt/kv_cache/ab_conv/seed.pt --out /mnt/kv_cache/ab_conv || exit 1
O=eval/conv_ab.txt
{ echo "SHORT-CONVOLUTION A/B, $(date -u '+%Y-%m-%d %H:%M UTC')"; grep -h "eval step\|ADDED\|COLD" logs/ab_ctrl.out logs/ab_conv.out; } > $O
for arm in ctrl conv; do INCONTEXT_GPT2=0 $P eval/incontext.py /mnt/kv_cache/ab_$arm/ckpt.pt powered 2>&1 | grep -vE "Warning|^\[gpu\]" >> $O; done
echo "CONV_AB_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
