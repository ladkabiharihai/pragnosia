#!/bin/bash
# THE FULL VISION EVALUATION OF ONE PRETRAINED CHECKPOINT (run it with the card otherwise free of our jobs): factual fine-tune, own-mistake
# abstention (balanced), then the scorecard (Q&A right/wrong/declined vs GPT-2 and SmolLM2, context rot, facts, hard
# in-context lookup, one million tokens, generation quality, learning without forgetting). Fine-tunes use schedule-free
# Prodigy (no rate). usage: eval/eval_ckpt.sh <pretrained ckpt> <tag>       output: eval/scorecard_<tag>.txt
set -u
cd "$(dirname "$0")/.."
CK=$1; TAG=$2; P=.venv/bin/python
$P -m src.sft --init $CK --out /mnt/kv_cache/${TAG}_sft --data /mnt/kv_cache/sft_factual_v8192 --optim psf --epochs 2 --cap 30 > logs/sft_${TAG}.out 2>&1
$P -u eval/rtune.py /mnt/kv_cache/${TAG}_sft/ckpt.pt /mnt/kv_cache/rtune_${TAG}.jsonl > logs/rtune_${TAG}.out 2>&1
$P - "$TAG" <<'PY'
import json, random, sys
t = sys.argv[1]
rows = [json.loads(l) for l in open(f"/mnt/kv_cache/rtune_{t}.jsonl")]
right = [r for r in rows if r["model_right"]]; idk = [r for r in rows if not r["model_right"]]
random.Random(0).shuffle(idk)
with open(f"/mnt/kv_cache/rtune_{t}_bal.jsonl", "w") as f:
    for r in right + idk[:len(right)]: f.write(json.dumps(r) + "\n")
print(f"{len(right)} right + {min(len(right), len(idk))} I-don't-know")
PY
SFT_MIX=factual SFT_DROP=sciq,arc SFT_EXTRA=/mnt/kv_cache/rtune_${TAG}_bal.jsonl $P eval/build_sft.py /mnt/kv_cache/sft_rtune_${TAG} > logs/build_rtune_${TAG}.out 2>&1
$P -m src.sft --init /mnt/kv_cache/${TAG}_sft/ckpt.pt --out /mnt/kv_cache/${TAG}_abst --data /mnt/kv_cache/sft_rtune_${TAG} --optim psf --epochs 1 --cap 30 > logs/sft_${TAG}_abst.out 2>&1
cp eval/scorecard.txt eval/scorecard_prev.txt 2>/dev/null
eval/scorecard.sh $CK /mnt/kv_cache/${TAG}_sft/ckpt.pt /mnt/kv_cache/${TAG}_abst/ckpt.pt > logs/scorecard_${TAG}.out 2>&1
cp eval/scorecard.txt eval/scorecard_${TAG}.txt
echo "EVAL_${TAG}_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> eval/scorecard_${TAG}.txt
