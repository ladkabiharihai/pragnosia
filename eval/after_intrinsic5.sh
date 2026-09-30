#!/bin/bash
# AFTER THE LONG INTRINSIC RUN (run70): every stage with the fused schedule-free Prodigy (no rate anywhere).
#   1 goal 9: 8,000 more steps with 25% of sequences from eval/build_recall.py (in-context lookup)   -> run70r
#   2 factual fine-tune                                                                           -> run71_sft
#   3 goals 4-5: own right/wrong answers on SciQ/ARC TRAIN questions, balanced 1:1 with "I don't know" -> run72_abst
#   4 the full vision scorecard                                                                   -> eval/scorecard.txt
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
until grep -q TRAIN70_DONE logs/train_grow70.out 2>/dev/null; do sleep 120; done
mkdir -p /mnt/kv_cache/run70r; cp /mnt/kv_cache/run70_grow/ckpt.pt /mnt/kv_cache/run70r/seed.pt
S0=$($P -c "import torch; print(torch.load('/mnt/kv_cache/run70r/seed.pt', map_location='cpu', weights_only=False)['step'])")
$P -u -m src.train --steps $((S0 + 8000)) --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim sfp \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokrecall_v8192 --data2-frac 0.25 \
  --save-every 4000 --eval-every 4000 --cap 34 --resume /mnt/kv_cache/run70r/seed.pt --out /mnt/kv_cache/run70r > logs/train_recall70.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run70r/ckpt.pt --out /mnt/kv_cache/run71_sft --data /mnt/kv_cache/sft_factual_v8192 --optim psf \
  --epochs 2 --cap 30 > logs/sft_run71.out 2>&1
$P -u eval/rtune.py /mnt/kv_cache/run71_sft/ckpt.pt /mnt/kv_cache/rtune_run71.jsonl > logs/rtune_run71.out 2>&1
$P - <<'PY'
import json, random
rows = [json.loads(l) for l in open("/mnt/kv_cache/rtune_run71.jsonl")]
right = [r for r in rows if r["model_right"]]; idk = [r for r in rows if not r["model_right"]]
random.Random(0).shuffle(idk)
with open("/mnt/kv_cache/rtune_run71_bal.jsonl", "w") as f:
    for r in right + idk[:len(right)]: f.write(json.dumps(r) + "\n")
print(f"{len(right)} right + {min(len(right), len(idk))} I-don't-know")
PY
SFT_MIX=factual SFT_DROP=sciq,arc SFT_EXTRA=/mnt/kv_cache/rtune_run71_bal.jsonl $P eval/build_sft.py /mnt/kv_cache/sft_rtune71_v8192 > logs/build_rtune71.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run71_sft/ckpt.pt --out /mnt/kv_cache/run72_abst --data /mnt/kv_cache/sft_rtune71_v8192 --optim psf \
  --epochs 1 --cap 30 > logs/sft_run72.out 2>&1
sed -i 's#logs/train_intrinsic[0-9]*.out#logs/train_grow70.out#' eval/scorecard.sh
eval/scorecard.sh /mnt/kv_cache/run70r/ckpt.pt /mnt/kv_cache/run71_sft/ckpt.pt /mnt/kv_cache/run72_abst/ckpt.pt > logs/scorecard70.out 2>&1
echo AFTER70_DONE >> eval/scorecard.txt
