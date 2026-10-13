#!/bin/bash
# P0.3 pre-registration amendment 1c, the demonstration job (heavy lane), seed 20261014:
#   rank n   (sheet v1):           diagnosis (fit 3-4 / test 5-10; flag churn), known-good (fit 7-8,
#                                  test 3-6 and 9-10), known-bads (a)-(e), KB-f, KB-h1, KB-h2, KB-g report;
#   rank n+1 (recipe option (a)):  known-good and known-bads on the same split and seed;
# then registers p03.table.a1c.*. It STOPS there: no gated test, files 1-2 not read.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
export P03_OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
RUNS="${RUNS:-n:diagnosis n:known-good n:known-bad n+1:known-good n+1:known-bad}"
for r in $RUNS; do
    rank="${r%%:*}"; m="${r#*:}"
    "$PY" "$here/table.py" "$m" --rank "$rank" --rep "$REP" --proc "$PROC"; rc=$?
    echo "$m rank $rank exit $rc"
    [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ] || exit "$rc"
done
"$PY" "$here/registry_seed.py" --only p03.table.a1c --overwrite || exit 24
echo "amendment-1c demonstration done (gated test NOT run)"
