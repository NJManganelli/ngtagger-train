#!/bin/bash
# P0.3 pre-registration amendment 2, the module-frame disc rows (heavy lane), seed 20261014, split fit 7-8 /
# test 3-6 and 9-10: known-good and known-bads under --frame module; then registers p03.table.a2.*.
# MODE=gated runs the gated test instead (fit 1-2, test 3-10; table.py refuses without the demonstration and
# $P03_OUT/amendment2_approval.txt).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
export P03_OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
MODES="${MODE:-known-good known-bad}"
for m in $MODES; do
    "$PY" "$here/table.py" "$m" --frame module --rep "$REP" --proc "$PROC"; rc=$?
    echo "$m exit $rc"
    [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ] || [ "$rc" -eq 5 ] || exit "$rc"
done
"$PY" "$here/registry_seed.py" --only p03.table.a2 --overwrite || exit 24
"$PY" "$here/registry_seed.py" --only p03.table.v1a1e-disclocal --overwrite || exit 25
echo "amendment-2 run done ($MODES)"
