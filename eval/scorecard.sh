#!/bin/bash
# THE VISION SCORECARD, ONE COMMAND, ONE MODEL LINEAGE. Every goal in VISION.md measured on the same checkpoints
# against GPT-2 124M (and SmolLM2-135M where it applies), identical inputs, card to itself, one job at a time.
#
# usage: eval/scorecard.sh <pretrained ckpt> <fine-tuned ckpt> <abstention-tuned ckpt>     output: eval/scorecard.txt
#   1  surpass the LLMs         Q&A (everyday / SciQ / ARC), context rot, facts, generation quality   qa.py, nine.py, genqual2.py
#   2  a cognitive model        the profile below: memory (9), learning (7), metacognition (4-5), adaptation (7-8)
#   3  self improving           gate: are its own majority answers mostly right?                       selfimprove.py (gate only)
#   4  knows what it knows      calibration + selective accuracy                                       nine.py, abstain.py
#   5  abstains                 right / WRONG / declined, before and after learning from own mistakes  qa.py
#   6  consumer GPU / 4 GB      inference memory to 1M tokens; training memory and speed               million.py, eval/stages.txt
#   7  growth + learning        model-decided growth (training log), learning without forgetting      train log, continual.py
#   8  everything intrinsic     rate and growth set by the model (training log)                        train log
#   9  context rot              prediction vs length; HARD in-context lookup                           nine.py, incontext.py
set -u
cd "$(dirname "$0")/.."
BASE=$1; SFT=$2; ABST=$3
P=.venv/bin/python
OUT=eval/scorecard.txt
: > $OUT
sec () { echo "" >> $OUT; echo "################ $1" >> $OUT; echo "   $(date -u '+%Y-%m-%d %H:%M UTC')" >> $OUT; }
clean () { grep -vE "Warning|Loading|^\[gpu\]|transformers|warn\(|it/s\]" ; }
sec "1, 5  QUESTION ANSWERING, right / WRONG / declined: fine-tuned, abstention-tuned, GPT-2, SmolLM2 ($SFT, $ABST)"
$P -u eval/qa.py "$SFT=ours-finetuned" "$ABST=ours-abstaining" 2>&1 | clean >> $OUT
sec "1, 4, 9  CONTEXT ROT, FACTS + CALIBRATION, INDUCTION, INFERENCE MEMORY ($BASE)"
$P eval/nine.py "$BASE" final 2>&1 | clean >> $OUT
sec "9  IN-CONTEXT LOOKUP, HARD: every wrong option is another answer from the same context ($SFT)"
$P eval/incontext.py "$SFT" hard 2>&1 | clean >> $OUT
sec "9  RETRIEVAL: passkey (standard, out of distribution), needles by distance (real prose), lookup with error bars ($SFT / $BASE)"
PASSKEY_LENS=512,2048,8192,32768 $P eval/incontext.py "$BASE" passkey 2>&1 | clean | grep -v wanted >> $OUT
NEEDLE_DIST=64,256,1024,4096,7000,16384 INCONTEXT_GPT2=0 $P eval/incontext.py "$BASE" needle 2>&1 | clean | grep -v miss: >> $OUT
$P eval/incontext.py "$SFT" powered 2>&1 | clean >> $OUT
sec "9  CONTEXT PAST THE TRAINING WINDOW: the same passages after 8k vs 32k of context, 24 passages, paired ($BASE)"
MILLION_SAMPLES=24 $P eval/million.py "$BASE" 8192,32768 2>&1 | clean | grep "tokens of context" >> $OUT
sec "6  ONE MILLION TOKENS OF CONTEXT: memory and bits per byte ($BASE)"
$P eval/million.py "$BASE" 2>&1 | clean >> $OUT
sec "1  GENERATION QUALITY: words, surprise and variety together, against GPT-2 ($BASE)"
$P eval/genqual2.py "$BASE" 2>&1 | clean >> $OUT
sec "7  LEARNING WITHOUT FORGETTING, NO RATE: whole model, + self-rehearsal, new layers only ($SFT)"
$P eval/continual.py "$SFT" intrinsic 2>&1 | clean >> $OUT
sec "7, 8  GROWTH AND RATE DECIDED BY THE MODEL (from its training log)"
grep -E "RECONSIDER|TRIAL|DECISION|GREW|REACH|DEEPENED|SPLIT|eval step" logs/train83.out >> $OUT
echo "SCORECARD_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $OUT
