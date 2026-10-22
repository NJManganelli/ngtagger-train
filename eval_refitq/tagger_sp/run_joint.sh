#!/bin/bash
# Jointly trained track-quality heads on the heavy lane (tagger_v5/joint_heads/PLAN.md); run by ops/spheavy from
# the frozen copy's root after run_all.sh built and trained SAMPLES (its manifests, track and jet caches are the
# inputs). Rerun-safe: every stage skips work already done.
#   eval_refitq/tagger_sp/run_joint.sh SAMPLES...       (ttbar htt)
# Stages: extra columns -> lambda scan (fit) -> lambda choice + identity check -> MI -> head quality -> conifer
# check -> jet caches per head version -> 10 trials per joint cell (+ top-up) -> trial choice -> analysis of every
# cell; results copied to OUT/results_joint_<samples>/. Logs in OUT/logs (run.log, "end rc=N").
# Exit: 0 ok; 1 a step failed; 3 material (identity check failed: stop; conifer mismatch or a cell without a
# kept trial: reported at the end).
set -u
if [ $# -gt 0 ]; then SAMPLES=("$@"); else SAMPLES=(ttbar htt); fi
PY=${TAGGER_PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}
D=eval_refitq/tagger_sp
OUT=$($PY -c "import sys; sys.path.insert(0, '$D'); import common as C; print(C.OUT)") || exit 1
LOG=$OUT/logs
mkdir -p "$LOG"
TAG=$(IFS=_; echo "${SAMPLES[*]}")
export OMP_NUM_THREADS=2 PY D LOG TAG SAMPLES_STR="${SAMPLES[*]}"
export JSON_ROOT=${JSON_ROOT:-/opt/homebrew/include}
say() { echo "$(date -u +%FT%TZ) joint $*" | tee -a "$LOG/run.log"; }
fail() { say "$* FAILED"; say "end rc=1"; exit 1; }
say "start [${SAMPLES[*]}] in $PWD"
MATERIAL=""
# outputs of another code revision (e.g. a run frozen before the base-score and h_nmerged fixes) must not be reused
# by the rerun-safe stages: stop and say what to move aside
REV=r1
if [ -e "$OUT/joint_heads/fits" ] && [ "$(cat "$OUT/joint_heads/REVISION" 2>/dev/null)" != "$REV" ]; then
  say "stale joint outputs of another revision: move aside $OUT/joint_heads, $OUT/cache/jets_joint_*, $OUT/preds/*_joint_*, $OUT/models/*/joint_* first"
  say "end rc=1"
  exit 1
fi
mkdir -p "$OUT/joint_heads" && echo $REV > "$OUT/joint_heads/REVISION"

for st in extra fit choose mi quality conifer jets; do
  $PY -u $D/joint_heads.py $st >> "$LOG/joint_${st}_$TAG.log" 2>&1 || fail "joint_heads $st"
  say "$st ok"
  if [ $st = choose ]; then
    say "  $(tail -1 "$LOG/joint_choose_$TAG.log")"
    if ! $PY -c "import json,sys; sys.path.insert(0,'$D'); import joint_heads as J; \
sys.exit(0 if json.load(open(J.jdir('choice.json')))['identity_all_pass'] else 1)"; then
      say "MATERIAL: the identity check (lambda = 0 joint vs independent heads) failed: stop"
      say "end rc=3"
      exit 3
    fi
  fi
  if [ $st = conifer ] && ! $PY -c "import json,sys; sys.path.insert(0,'$D'); import joint_heads as J; \
sys.exit(0 if json.load(open(J.jdir('conifer.json')))['pass'] else 1)"; then
    MATERIAL="$MATERIAL conifer-mismatch"
    say "MATERIAL: conifer differs from the booster (reported; nothing downstream depends on it)"
  fi
done

train_cells() {   # lines "mask variant" on stdin, seeds in $1
  SEEDS_STR=$1 xargs -P 4 -L 1 bash -c \
    '$PY -u $D/train.py --mask $0 --variant $1 --samples $SAMPLES_STR --seeds $SEEDS_STR --threads 2 \
       >> $LOG/train_$0_$1_$TAG.log 2>&1 || { echo "train $0 $1 rc=$?" >&2; exit 255; }'
}
for m in IIII AAAA AAAI IAAA; do for v in joint_ref joint_A joint_B; do echo "$m $v"; done; done \
  | train_cells "0 1 2 3 4 5 6 7 8 9" || fail "training (first 10 trials)"
say "first 10 trials per joint cell done"
$PY -u $D/trial_select.py > "$LOG/select_joint_$TAG.log" 2>&1 || fail "trial selection"
TOP=$(awk '/^TOPUP/ {print $2, $3}' "$LOG/select_joint_$TAG.log")
if [ -n "$TOP" ]; then
  say "top-up for: $(echo "$TOP" | tr '\n' ',')"
  echo "$TOP" | train_cells "10 11 12 13 14 15 16 17 18 19" || fail "training (top-up)"
  $PY -u $D/trial_select.py > "$LOG/select_joint_$TAG.log" 2>&1 || fail "trial selection"
fi
grep 'joint_' "$LOG/select_joint_$TAG.log" | grep -v '^TOPUP' | while read -r line; do say "  $line"; done
$PY -u $D/analyze.py --boot 300 > "$LOG/analyze_joint_$TAG.log" 2>&1 || fail "analysis"
say "analysis ok"
R=$OUT/results_joint_$TAG
mkdir -p "$R"
cp "$OUT/selection.json" "$OUT/summary.json" "$OUT/tagger_sp.pdf" "$R/" || fail "copy of the results"
cp "$OUT"/joint_heads/{choice,mi,quality,conifer}.json "$R/" || fail "copy of the head results"
grep -q "NO KEPT TRIAL" "$LOG/select_joint_$TAG.log" && MATERIAL="$MATERIAL cell-without-kept-trial"
if [ -n "$MATERIAL" ]; then
  say "MATERIAL:$MATERIAL"
  say "end rc=3"
  exit 3
fi
say "end rc=0"
