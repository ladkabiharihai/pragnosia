#!/bin/bash
# TESTS OF THE OUT-OF-MEMORY FIX (2026-10-01; PROGRESS.md "why run70 ran out of memory"). Small model, a few minutes.
#   1  an out-of-graph eval fails with OOM (injected: MEMBRAE_TEST_OOM=N fails the Nth such call once): the graphs must
#      be released, the eval redone, the graphs recaptured, and training must continue with EXACTLY the losses of an
#      undisturbed control run
#   2  the run70 case: the exactness check right after a growth's capture fails; the growth must complete, exact
#   3  eval/keep_training.sh: the trainer is killed mid-run; it must resume from its checkpoint and finish
# output: eval/oom_test.txt
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python; O=eval/oom_test.txt; : > $O
C="--d 256 --layers 4 --ctx 2048 --batch 2 --accum 1 --mixer grspin --norm rms --optim sfp --scan-chunk 64 --chunks 4
   --fused-scan 1 --seed 3 --data /mnt/kv_cache/tokfull_v8192 --sources dclm,fineweb-edu,wikipedia,finemath,arxiv --cap 8"
rm -rf /mnt/kv_cache/oomtest_*; mkdir -p /mnt/kv_cache/oomtest_a /mnt/kv_cache/oomtest_b /mnt/kv_cache/oomtest_c /mnt/kv_cache/oomtest_d /mnt/kv_cache/oomtest_e
echo "== 1  control vs injected OOM at the first eval (step 100)" >> $O
$P -u -m src.train $C --steps 600 --eval-every 100 --save-every 1000 --out /mnt/kv_cache/oomtest_a > logs/oomtest_a.out 2>&1; echo "control exit $?" >> $O
MEMBRAE_TEST_OOM=1 $P -u -m src.train $C --steps 600 --eval-every 100 --save-every 1000 --out /mnt/kv_cache/oomtest_b > logs/oomtest_b.out 2>&1; echo "injected exit $?" >> $O
grep -E "released|eval step" logs/oomtest_b.out >> $O
paste <(grep -oE "^\[ *[0-9.]+s\] [0-9]+/600 loss [0-9.]+" logs/oomtest_a.out | awk '{print $(NF-2), $NF}') \
      <(grep -oE "^\[ *[0-9.]+s\] [0-9]+/600 loss [0-9.]+" logs/oomtest_b.out | awk '{print $NF}') | sed 's/^/  step, control loss, injected loss: /' >> $O
echo "== 2  injected OOM in the exactness check right after a deepen's capture (step 200), then a widen (step 300)" >> $O
MEMBRAE_TEST_OOM=2 $P -u -m src.train $C --steps 400 --eval-every 1000 --save-every 1000 --grow-schedule 200:d,300:w384 --out /mnt/kv_cache/oomtest_c > logs/oomtest_c.out 2>&1; echo "exit $?" >> $O
grep -E "released|DEEPENED|GREW|/400 loss" logs/oomtest_c.out | cut -c1-200 >> $O
echo "== 3  automatic resume: trainer killed mid-run (after step 4,000), checkpoints every 1,000" >> $O
# long enough to be killed mid-run (the first version finished 600 steps before the kill), and the child is identified
# through tr: grep cannot read /proc/<pid>/cmdline directly, so the first redo never killed anything either
eval/keep_training.sh logs/oomtest_d.out OOMTEST_DONE $C --steps 20000 --eval-every 100000 --save-every 1000 --out /mnt/kv_cache/oomtest_d &
W=$!
until grep -qE " (4|5|6)[0-9]00/20000 loss" logs/oomtest_d.out 2>/dev/null; do sleep 1; done
T=$(pgrep -P $W | head -1); CMD=$(tr '\0' ' ' < /proc/$T/cmdline)
case "$CMD" in *src.train*) kill $T; echo "killed trainer $T (${CMD:0:50})" >> $O;; *) echo "child $T is not the trainer: $CMD" >> $O;; esac
wait $W; echo "wrapper exit $?" >> $O
grep -E "TRAINER EXITED|RESUMED|OOMTEST_|resumed from" logs/oomtest_d.out | cut -c1-160 >> $O
grep -oE "[0-9]+/20000 loss [0-9.]+" logs/oomtest_d.out | awk 'NR%10==0' >> $O
echo "OOM_TEST_DONE $(date -u '+%Y-%m-%d %H:%M UTC')" >> $O
rm -rf /mnt/kv_cache/oomtest_*
