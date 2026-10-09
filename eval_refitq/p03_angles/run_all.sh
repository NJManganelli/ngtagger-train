#!/bin/bash
# P0.3, one heavy-lane job: module checks -> table sequence (run_table.sh) -> ROC -> other outputs
# -> summary PDF -> registry. Outputs in $P03_OUT (default /Volumes/WDMac/smartpixels_scratch/p03_angles).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
export P03_OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
export REP PROC
"$PY" -m ngtagger.spdefs.checks run --json "$P03_OUT/spdefs_checks_run.json" || exit 11
"$PY" -m ngtagger.spdefs.checks known-bad --json "$P03_OUT/spdefs_checks_known_bad.json" || exit 12
bash "$here/run_table.sh" || exit $?
TABLE="$(cat "$P03_OUT/p03_table_used.txt")"
echo "table for the ROC and the other outputs: $TABLE"
"$PY" "$here/roc.py" --table "$TABLE" --rep "$REP" --proc "$PROC" || exit 21
"$PY" "$here/extras.py" --table "$TABLE" --rep "$REP" || exit 22
"$PY" "$here/plots.py" || exit 23
"$PY" "$here/registry_seed.py" || exit 24
echo "P0.3 done"
