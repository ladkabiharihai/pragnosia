#!/bin/bash
# AFTER THE INTRINSIC RUN (run60): the remaining goals on that lineage, every stage with schedule-free Prodigy at its
# defaults (no rate, no schedule anywhere). Starts when eval/intrinsic_run3.sh has finished its own fine-tune + Q&A.
#   1  goal 9: 6,000 more steps of pretraining with 25% of sequences from eval/build_recall.py (facts listed, then
#      asked in other wording), so the model learns in-context lookup           -> run60r
#   2  factual fine-tune of run60r                                               -> run62_sft
#   3  goals 4-5: its own right/wrong answers on SciQ/ARC TRAIN questions, balanced 1:1 with "I don't know",
#      fine-tuned from run62_sft                                                 -> run63_abst
#   4  the whole vision scorecard on run60r / run62_sft / run63_abst            -> eval/scorecard.txt
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
until grep -q INTRINSIC3_QA_DONE eval/qa7.txt 2>/dev/null; do sleep 60; done
mkdir -p /mnt/kv_cache/run60r
cp /mnt/kv_cache/run60_intrinsic/ckpt.pt /mnt/kv_cache/run60r/seed.pt
$P -u -m src.train --steps 126000 --d 512 --layers 12 --ctx 8192 --batch 2 --accum 4 --mixer grspin --norm rms --optim psf \
  --ctx-mix 2048,8192 --scan-chunk 64 --chunks 4 --fused-scan 1 --seed 1 \
  --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv \
  --data2 /mnt/kv_cache/tokrecall_v8192 --data2-frac 0.25 \
  --save-every 2000 --eval-every 2000 --cap 34 --resume /mnt/kv_cache/run60r/seed.pt --out /mnt/kv_cache/run60r > logs/train_recall.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run60r/ckpt.pt --out /mnt/kv_cache/run62_sft --data /mnt/kv_cache/sft_factual_v8192 --optim psf \
  --epochs 2 --cap 30 > logs/sft_run62.out 2>&1
$P -u eval/rtune.py /mnt/kv_cache/run62_sft/ckpt.pt /mnt/kv_cache/rtune_run62.jsonl > logs/rtune_run62.out 2>&1
$P - <<'PY'
import json, random
rows = [json.loads(l) for l in open("/mnt/kv_cache/rtune_run62.jsonl")]
right = [r for r in rows if r["model_right"]]; idk = [r for r in rows if not r["model_right"]]
random.Random(0).shuffle(idk)
with open("/mnt/kv_cache/rtune_run62_bal.jsonl", "w") as f:
    for r in right + idk[:len(right)]: f.write(json.dumps(r) + "\n")
print(f"{len(right)} right + {min(len(right), len(idk))} I-don't-know")
PY
SFT_MIX=factual SFT_DROP=sciq,arc SFT_EXTRA=/mnt/kv_cache/rtune_run62_bal.jsonl $P eval/build_sft.py /mnt/kv_cache/sft_rtune62_v8192 > logs/build_rtune62.out 2>&1
$P -m src.sft --init /mnt/kv_cache/run62_sft/ckpt.pt --out /mnt/kv_cache/run63_abst --data /mnt/kv_cache/sft_rtune62_v8192 --optim psf \
  --epochs 1 --cap 30 > logs/sft_run63.out 2>&1
eval/scorecard.sh /mnt/kv_cache/run60r/ckpt.pt /mnt/kv_cache/run62_sft/ckpt.pt /mnt/kv_cache/run63_abst/ckpt.pt > logs/scorecard2.out 2>&1
echo AFTER_INTRINSIC_DONE >> eval/scorecard.txt
