#!/bin/bash
# One tagger_sp training cell through the heavy lane, from a frozen copy of eval_refitq/tagger_sp +
# src/ngtagger (untracked code such as linout.py and tagger_sp_linout.yaml is copied automatically and
# its content recorded; the diff of tracked files, e.g. train.py, is stored as the job's tree.patch).
#   ops/tagger/run_tagger_cell.sh PREREG.txt RECIPE MASK VARIANT [train.py options ...]
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRE=$1; RECIPE=$2; MASK=$3; VAR=$4; shift 4
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec "$R/ops/spheavy" run --kind tagger --name "tagger-$RECIPE-$MASK-$VAR" --frozen --tree eval_refitq/tagger_sp \
  --prereg "@$PRE" --gate --disk-path /Volumes/WDMac -- \
  python eval_refitq/tagger_sp/train.py --recipe "$RECIPE" --mask "$MASK" --variant "$VAR" "$@"
