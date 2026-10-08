#!/bin/bash
# THE FINAL MODEL, STEP 1, SECOND TRY (2026-10-07): as eval/final_recall.sh but the STEP SIZE FREE (Prodigy's own estimate,
# not held): with the step held at 9.69e-4 recall training taught the model to IGNORE far keys (needles at 8k+ 4% -> 0% from
# 2,000 to 4,000 steps), while the A/B's recall arm, whose step rose 2.7e-3 -> 3.3e-3, reached 92% at 32k. (First try: user approved the revert): from run85's best checkpoint (368,000: trainer bits/byte
# 1.0372) a SHORT recall run, as the 2026-10-05 A/B's recall arm (4,000 steps from 368k gave needles 100% to 16k, 92% at 32k,
# 79% at 64k and -0.034 nats/token on 32k streams): full-strength recall 16 (ungated: the smooth-start gates opened only
# -4 -> ~-3 in 4,000 final-phase steps), the run85 recipe otherwise (main corpora + 25% retrieval documents, carried
# windows, step size held, no growth trials, no instruction data: the final phase's instruction-heavy mix cost ~10% bits/
# byte). Held-out every 2,000 steps; checkpoints every 2,000 kept, so the best one can be chosen by held-out.
# usage: eval/final_recall.sh [steps]
set -u
cd "$(dirname "$0")/.."
N=${1:-6000}; OUT=/mnt/kv_cache/final_recall2; SEED=/mnt/kv_cache/run85_s368000_c.pt
if pgrep -f '^(\S*/)?python[0-9.]* -u -m src\.train' >/dev/null; then echo "a trainer is running; refusing" >&2; exit 3; fi
mkdir -p $OUT; rm -f $OUT/ckpt*.pt $OUT/resume_from.pt; cp $SEED $OUT/seed.pt
echo "=== FINAL RECALL RUN $(date -u '+%Y-%m-%d %H:%M UTC') from $SEED, $N steps ===" > logs/final_recall2.out
# keep every checkpoint it writes (the trainer keeps only ckpt.pt and ckpt_prev.pt), so the best can be chosen by held-out
( m0=0; while :; do sleep 60; [ -f $OUT/ckpt.pt ] || continue; m1=$(stat -c %Y $OUT/ckpt.pt); [ "$m1" = "$m0" ] && continue
  sleep 20; s_=$(.venv/bin/python -c "import torch;print(torch.load('$OUT/ckpt.pt',map_location='cpu',weights_only=False)['step'])" 2>/dev/null) || continue
  cp $OUT/ckpt.pt $OUT/ckpt_s$s_.pt; m0=$m1; grep -q FINAL_RECALL2_DONE logs/final_recall2.out && break; done ) &
eval/keep_training.sh logs/final_recall2.out FINAL_RECALL2_DONE --steps $((368000 + N)) --d 512 --layers 12 --ctx 8192 --batch 2 \
  --accum 4 --mixer grspin --norm rms --optim sfp --attn-sliding 1 --attn-window 1024 --share-d 1 --carry 16 --carry-groups 16 \
  --carry-half 1 --recall 16 --recall-gated 0 --spike 1e9 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 22 --freeze-d 0 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --data2 /mnt/kv_cache/tokneedle_mix \
  --data2-frac 0.25 --save-every 2000 --eval-every 2000 --cap 34 --resume $OUT/seed.pt --out $OUT
