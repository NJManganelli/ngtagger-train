#!/bin/bash
# A census job through the heavy lane, from a frozen copy of eval_refitq/combinatorics + src/ngtagger:
# the lock-aware successor of site_v2/wp2/run_cfgz.sh (which only polled for tagger_sp/train.py).
#   ops/census/run_census.sh LABEL OUTDIR INPUT.root PREREG.txt [tp_findability options ...]
# The LUTs the census reads beside itself (luts/, silently skipped by ot_luts() when absent) and the input
# are declared data files: a copy without them is refused at launch. The fast gate runs first (once per
# frozen-copy manifest). Outputs go to OUTDIR (never into the copy, which is read-only).
set -u
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L=$1; OUT=$2; IN=$3; PRE=$4; shift 4
DATA=(--data "$IN")
for f in "$R"/eval_refitq/combinatorics/luts/VMPhiCorrL{1,2,3,4,5,6}.tab \
         "$R"/eval_refitq/combinatorics/luts/TP_L1L2A_stubptinnercut.dat; do DATA+=(--data "$f"); done
mkdir -p "$OUT"
exec "$R/ops/spheavy" run --kind census --name "census-$L" --frozen --tree eval_refitq/combinatorics \
  "${DATA[@]}" --prereg "@$PRE" --gate --disk-path "$OUT" -- \
  python -u eval_refitq/combinatorics/tp_findability.py -i "$IN" --cache-dir "$OUT/$L" -o "$OUT/$L.json" "$@"
