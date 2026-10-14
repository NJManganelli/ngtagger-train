#!/bin/bash
# P0.3 table sequence under pre-registration amendment 1e (preregistration.txt):
#   1. the rule's known-good demonstration (fit files 7-8, test 3-6 and 9-10);
#   2. known-bads (a)-(e), KB-q95, KB-c and KB-w on the same split (each must fail as declared);
#   3. only if 1 passed, every known-bad failed AND the physicist approved amendment 1e
#      ($OUT/amendment1e_approval.txt): the gated test (fit 1-2, test 3-10), judged once (exit 0 pass,
#      5 MINOR deviations: the table ships with them listed, 4 MATERIAL: back to the physicist);
#      otherwise the recipe fitted on files 1-2 WITHOUT the test (an UNTESTED table, for the ROC's centres).
# Writes the path of the table the later steps use to $OUT/p03_table_used.txt. Exit 0 when the
# sequence ran as designed (whatever the verdicts); the verdicts are in the JSON files.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
rm -f "$OUT/p03_table_used.txt"
"$PY" "$here/table.py" known-good --rep "$REP" --proc "$PROC"; kg=$?
"$PY" "$here/table.py" known-bad --rep "$REP" --proc "$PROC"; kb=$?
echo "known-good exit $kg (0 = pass), known-bad exit $kb (0 = every known-bad fails)"
[ "$kg" -eq 0 ] || [ "$kg" -eq 4 ] || [ "$kg" -eq 5 ] || exit "$kg"
[ "$kb" -eq 0 ] || [ "$kb" -eq 3 ] || exit "$kb"
if [ "$kg" -eq 0 ] && [ "$kb" -eq 0 ] && [ -f "$OUT/amendment1e_approval.txt" ]; then
    "$PY" "$here/table.py" gated --rep "$REP" --proc "$PROC"; g=$?
    echo "gated exit $g (0 = the table passes its test)"
    [ "$g" -eq 0 ] || [ "$g" -eq 4 ] || [ "$g" -eq 5 ] || exit "$g"
    echo "$OUT/angle_table_v1a1e_canon_v3.json" > "$OUT/p03_table_used.txt"
else
    echo "the gated test does not run (amendment 1e not approved, or D.0 step 4): fitting the UNTESTED table for the ROC"
    "$PY" "$here/table.py" fit-only --rep "$REP" --proc "$PROC" || exit $?
    echo "$OUT/angle_table_v1a1e_canon_v3_UNTESTED.json" > "$OUT/p03_table_used.txt"
fi
exit 0
