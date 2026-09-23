#!/bin/bash
# THE WHOLE SCORECARD, ONE COMMAND. Reproduces every number in REPORT.md against GPT-2 124M on identical
# inputs. Each section is a separate script so any one can be rerun alone; they run one after another because
# two jobs sharing the card measure contention, which looks real and is not (found three times on 2026-09-22).
#
# usage: eval/run_all.sh [checkpoint]        output: eval/results.txt
# expects: the 8k tokenizer and corpus at /mnt/kv_cache, GPT-2 via transformers, ~60-75 minutes on an H100.
set -u
cd "$(dirname "$0")/.."
CK=${1:-/mnt/kv_cache/run45_prose/ckpt.pt}
P=.venv/bin/python
OUT=eval/results.txt
: > $OUT
sec () { echo "" >> $OUT; echo "################ $1" >> $OUT; echo "   $(date -u '+%Y-%m-%d %H:%M UTC')  checkpoint $CK" >> $OUT; }
clean () { grep -vE "Warning|Loading|^\[gpu\]|transformers|warn\(" ; }
sec "1,3,6,9  context rot / inference efficiency / facts + calibration / induction"
$P eval/nine.py "$CK" final 2>&1 | clean >> $OUT
sec "2  training efficiency: our mixer against full attention, same harness, card to itself"
for T in 512 1024 2048 4096 8192 16384 32768; do
  $P eval/bench_step.py "$PWD" rspin attn --d 768 --ctx $T --batch 1 2>&1 | grep -E "ktok/s" >> $OUT
done
sec "4  large-context handling: one million tokens"
$P eval/million.py "$CK" 2>&1 | clean >> $OUT
sec "5  coherence, under BOTH decoding rules"
$P eval/coherence2.py "$CK" 2>&1 | clean >> $OUT
sec "7  generation quality: words, surprise and variety together, against GPT-2"
$P eval/genqual2.py "$CK" 2>&1 | clean >> $OUT
sec "8  long generation: 32,768 tokens, is each 4,096-token block still text"
$P eval/isittext.py "$CK" final 2>&1 | clean >> $OUT
sec "which layers do the work: silencing each in turn"
$P eval/whodoes.py "$CK" 2>&1 | clean >> $OUT
echo "" >> $OUT; echo "ALL_DONE $(date -u '+%H:%M UTC')" >> $OUT
