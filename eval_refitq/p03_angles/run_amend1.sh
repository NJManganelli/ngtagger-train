#!/bin/bash
# P0.3 pre-registration amendment 1, the demonstration job (heavy lane): the amended rule on
#   1. the split where the v1 rule failed (fit 3-4, test 5-10): diagnosis, reported;
#   2. the new known-good split (fit 9-10, test 3-8): the demonstration;
#   3. the known-bads and the input perturbation on the new split (each must fail);
# then registers p03.table.a1.*. It STOPS there: no gated test, no table fit on files 1-2
# (the gated run waits for the physicist's approval of amendment 1).
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
"$PY" "$here/registry_seed.py" --only p03.table.a1 --overwrite || exit 24
echo "amendment-1 demonstration done (gated test NOT run)"
