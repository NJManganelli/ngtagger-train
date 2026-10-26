#!/bin/bash
# The data-volume check on the slim tier: retrain cells on a fixed subset of N training events (the same events for
# every mask; slimtier.training_subset) and evaluate them on the full slim test set. Run from the repository (or
# frozen copy) root with TAGGER_SP_TIER=slim and TAGGER_SP_OUT = a folder of its own whose `cache` links to the full
# run's cache (so results_ttbar/ of the full run is untouched).
#   eval_refitq/tagger_sp/run_subset.sh N "MASKS" VARIANT "SEEDS"      e.g. 800 "IIII AAAA" baseline "0 1 2 3 4 5"
# The masks train in parallel (2 threads each); then trial_select.py and analyze.py; the outputs are copied to
# OUT/results_sub<N>/. No top-up.
set -u
N=${1:?training events}
MASKS=${2:-IIII AAAA}
VARIANT=${3:-baseline}
SEEDS=${4:-0 1 2 3 4 5}
PY=${TAGGER_PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}
D=eval_refitq/tagger_sp
OUT=$($PY -c "import sys; sys.path.insert(0, '$D'); import common as C; print(C.OUT)") || exit 1
[ "${TAGGER_SP_TIER:-}" = slim ] || { echo "TAGGER_SP_TIER=slim required"; exit 2; }
mkdir -p "$OUT/logs"
say() { echo "$(date -u +%FT%TZ) subset $*" | tee -a "$OUT/logs/run.log"; }
say "start N=$N masks [$MASKS] variant $VARIANT seeds [$SEEDS] in $PWD (OUT $OUT)"
pids=()
for m in $MASKS; do
  OMP_NUM_THREADS=2 $PY -u $D/train.py --mask $m --variant $VARIANT --samples ttbar --seeds $SEEDS --threads 2 \
    --train-events $N >> "$OUT/logs/train_${m}_${VARIANT}_sub$N.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { say "training FAILED"; say "end rc=1"; exit 1; }
say "training done"
$PY -u $D/trial_select.py > "$OUT/logs/select_sub$N.log" 2>&1 || { say "selection FAILED"; say "end rc=1"; exit 1; }
grep -v '^TOPUP' "$OUT/logs/select_sub$N.log" | while read -r line; do say "  $line"; done
$PY -u $D/analyze.py --boot 300 > "$OUT/logs/analyze_sub$N.log" 2>&1 || { say "analysis FAILED"; say "end rc=1"; exit 1; }
mkdir -p "$OUT/results_sub$N"
cp "$OUT/selection.json" "$OUT/summary.json" "$OUT/tagger_sp.pdf" "$OUT/results_sub$N/"
say "end rc=0"
