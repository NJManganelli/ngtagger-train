#!/bin/bash
# P0.3 pre-registration amendment 1d, the demonstration job (heavy lane), seed 20261014, split fit 7-8 /
# test 3-6 and 9-10 (as amendments 1b and 1c):
#   centre median           (sheet v2.5):                  known-good, known-bads (a)-(e), KB-q95, KB-c;
#   centre trimmed_mean_25  (the alternative measured for the physicist): the same;
# then registers p03.table.a1d.*. It STOPS there: no gated test, files 1-2 not read.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
export P03_OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
RUNS="${RUNS:-median:known-good median:known-bad trimmed_mean_25:known-good trimmed_mean_25:known-bad}"
for r in $RUNS; do
    c="${r%%:*}"; m="${r#*:}"
    "$PY" "$here/table.py" "$m" --centre "$c" --rep "$REP" --proc "$PROC"; rc=$?
    echo "$m centre $c exit $rc"
    [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ] || exit "$rc"
done
"$PY" "$here/registry_seed.py" --only p03.table.a1d --overwrite || exit 24
"$PY" "$here/registry_seed.py" --only p03.d10c --overwrite || exit 25
echo "amendment-1d demonstration done (gated test NOT run)"
