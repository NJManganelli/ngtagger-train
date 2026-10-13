#!/bin/bash
# P0.3 pre-registration amendment 1b, the demonstration job (heavy lane): the amended rule on
#   1. fit 3-4 / test 5-10 (the v1 failure split): diagnosis, reported (also a second flagged-row list);
#   2. the demonstration split (fit 7-8, test 3-6 and 9-10), seed 20261014: must pass;
#   3. known-bads (a)-(e), the canonical reference, KB-f and KB-g on the demonstration split;
# then registers p03.table.a1b.*. It STOPS there: no gated test, no table fit on files 1-2
# (files 1-2 are not read; the gated run waits for the physicist's approval of amendment 1b).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
export P03_OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
MODES="${MODES:-diagnosis known-good known-bad}"
for m in $MODES; do
    "$PY" "$here/table.py" "$m" --rep "$REP" --proc "$PROC"; rc=$?
    echo "$m exit $rc"
    [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ] || exit "$rc"
done
"$PY" "$here/registry_seed.py" --only p03.table.a1b --overwrite || exit 24
echo "amendment-1b demonstration done (gated test NOT run)"
