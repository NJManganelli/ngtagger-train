#!/bin/bash
# P0.3 pre-registration amendment 1e, the demonstration job (heavy lane), seed 20261014, split fit 7-8 /
# test 3-6 and 9-10 (as amendments 1b-1d): known-good, known-bads (a)-(e), KB-q95, KB-c and KB-w; then
# registers p03.table.a1e.*. It STOPS there: no gated test, files 1-2 not read.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
# the era's output directory (common.ERA / common.OUT_DIR: canon_v3 in p03_angles, another era in p03_angles_<era>)
export P03_OUT="${P03_OUT:-$(cd "$here" && "$PY" -c 'import common; print(common.OUT_DIR)')}"
REP="${REP:-250}"
PROC="${PROC:-10}"
for m in known-good known-bad; do
    "$PY" "$here/table.py" "$m" --rep "$REP" --proc "$PROC"; rc=$?
    echo "$m exit $rc"
    [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ] || [ "$rc" -eq 5 ] || exit "$rc"
done
"$PY" "$here/registry_seed.py" --only p03.table.a1e --overwrite || exit 24
echo "amendment-1e demonstration done (gated test NOT run)"
