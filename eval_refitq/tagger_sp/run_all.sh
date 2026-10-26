#!/bin/bash
# The jet-tagging study on the heavy lane: run by ops/spheavy from the frozen copy's root (or from the
# repository root). Rerun-safe: build.py reuses caches and heads made from the same inputs, train.py skips
# trials whose predictions exist for the same samples, test events and features.
#   eval_refitq/tagger_sp/run_all.sh build SAMPLES...   freeze the file set; per mask: track caches,
#                                                       track-quality heads, jet caches
#   eval_refitq/tagger_sp/run_all.sh train SAMPLES...   10 trials per cell (4 cells at a time, 2 threads each);
#                                                       trial screen and choice (trial_select.py); 10 more
#                                                       trials for cells with fewer than 3 kept; analysis
#   eval_refitq/tagger_sp/run_all.sh all SAMPLES...     build, then train
# SAMPLES: ttbar [htt]. TAGGER_SP_TIER=slim: the slim tagger tier (slimtier.py; ttbar only). TRIAL_SEEDS / TOPUP_SEEDS:
# the trials per cell (default 10 + a top-up of 10). Outputs under common.OUT; logs in OUT/logs (run.log: one line per step, ends with
# "end rc=N"); the selection, summary and PDF are copied to OUT/results_<samples>/ at the end.
# Exit: 0 ok; 1 a step failed; 3 a cell has no kept trial after the top-up (material: no result for it).
set -u
PHASE=${1:?phase: build | train | all}
shift
SAMPLES=("${@:-ttbar}")
PY=${TAGGER_PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}
D=eval_refitq/tagger_sp
OUT=$($PY -c "import sys; sys.path.insert(0, '$D'); import common as C; print(C.OUT)") || exit 1
LOG=$OUT/logs
mkdir -p "$LOG"
TAG=$(IFS=_; echo "${SAMPLES[*]}")
# trials per cell (the policy: 10, and 10 more for a cell with fewer than 3 kept); a test run may use fewer.
# Top-up seeds must start at 10 (trial_select.TOPUP_SEEDS).
TRIAL_SEEDS=${TRIAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}
TOPUP_SEEDS=${TOPUP_SEEDS:-10 11 12 13 14 15 16 17 18 19}
export OMP_NUM_THREADS=2 PY D LOG TAG SAMPLES_STR="${SAMPLES[*]}"
say() { echo "$(date -u +%FT%TZ) tagger $*" | tee -a "$LOG/run.log"; }
fail() { say "$* FAILED"; say "end rc=1"; exit 1; }
say "start $PHASE [${SAMPLES[*]}] tier ${TAGGER_SP_TIER:-canonical} trials [$TRIAL_SEEDS] top-up [$TOPUP_SEEDS] in $PWD"

build() {
  $PY -c "import sys; sys.path.insert(0, '$D'); import common as C; print('frozen file set:', \
C.freeze_fileset(sys.argv[1:]))" "${SAMPLES[@]}" >> "$LOG/fileset_$TAG.log" 2>&1 || fail "file set"
  say "$(tail -1 "$LOG/fileset_$TAG.log")"
  for m in IIII AAAA AAAI IAAA; do
    $PY -u $D/build.py --mask $m --samples "${SAMPLES[@]}" >> "$LOG/build_${m}_$TAG.log" 2>&1 || fail "build $m"
    say "build $m ok"
  done
}

# lines "mask variant" on stdin, seeds in $1 ("0 1 ... 9")
train_cells() {
  SEEDS_STR=$1 xargs -P 4 -L 1 bash -c \
    '$PY -u $D/train.py --mask $0 --variant $1 --samples $SAMPLES_STR --seeds $SEEDS_STR --threads 2 \
       >> $LOG/train_$0_$1_$TAG.log 2>&1 || { echo "train $0 $1 rc=$?" >&2; exit 255; }'
}

select_trials() {
  $PY -u $D/trial_select.py > "$LOG/select_$TAG.log" 2>&1 || fail "trial selection"
}

train() {
  for m in IIII AAAA AAAI IAAA; do for v in baseline track trackmva; do echo "$m $v"; done; done \
    | train_cells "$TRIAL_SEEDS" || fail "training (first trials)"
  say "first trials per cell done (seeds $TRIAL_SEEDS)"
  select_trials
  TOP=$(awk '/^TOPUP/ {print $2, $3}' "$LOG/select_$TAG.log")
  if [ -n "$TOP" ]; then
    say "top-up for: $(echo "$TOP" | tr '\n' ',')"
    echo "$TOP" | train_cells "$TOPUP_SEEDS" || fail "training (top-up)"
    select_trials
  fi
  grep -v '^TOPUP' "$LOG/select_$TAG.log" | while read -r line; do say "  $line"; done
  $PY -u $D/analyze.py --boot 300 > "$LOG/analyze_$TAG.log" 2>&1 || fail "analysis"
  say "analysis ok"
  mkdir -p "$OUT/results_$TAG"
  cp "$OUT/selection.json" "$OUT/summary.json" "$OUT/tagger_sp.pdf" "$OUT/results_$TAG/" || fail "copy of the results"
  if grep -q "NO KEPT TRIAL" "$LOG/select_$TAG.log"; then
    say "MATERIAL: a cell has no kept trial after the top-up"
    say "end rc=3"
    exit 3
  fi
}

case $PHASE in
  build) build ;;
  train) train ;;
  all) build; train ;;
  *) fail "unknown phase $PHASE" ;;
esac
say "end rc=0"
