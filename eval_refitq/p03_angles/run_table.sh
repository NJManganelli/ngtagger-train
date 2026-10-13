#!/bin/bash
# P0.3 table sequence under pre-registration amendments 1, 1b and 1c (preregistration.txt; rank: RANK=n or n+1):
#   1. the rule's known-good demonstration (fit files 7-8, test 3-6 and 9-10; amendment 1c);
#   2. known-bads (a)-(e), KB-f, KB-h1 and KB-h2 on the same split (each must fail);
#   3. only if 1 passed, every known-bad failed AND the physicist approved amendment 1c
#      ($OUT/amendment1c_approval.txt): the gated test (fit 1-2, test 3-10), judged once;
#      otherwise the recipe fitted on files 1-2 WITHOUT the test (an UNTESTED table, for the ROC's centres).
# Writes the path of the table the later steps use to $OUT/p03_table_used.txt. Exit 0 when the
# sequence ran as designed (whatever the verdicts); the verdicts are in the JSON files.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}"
OUT="${P03_OUT:-/Volumes/WDMac/smartpixels_scratch/p03_angles}"
REP="${REP:-250}"
PROC="${PROC:-10}"
RANK="${RANK:-n}"
SFX=""; [ "$RANK" = "n+1" ] && SFX="_rankA"
rm -f "$OUT/p03_table_used.txt"
"$PY" "$here/table.py" known-good --rank "$RANK" --rep "$REP" --proc "$PROC"; kg=$?
"$PY" "$here/table.py" known-bad --rank "$RANK" --rep "$REP" --proc "$PROC"; kb=$?
echo "known-good exit $kg (0 = pass), known-bad exit $kb (0 = every known-bad fails)"
[ "$kg" -eq 0 ] || [ "$kg" -eq 4 ] || exit "$kg"
[ "$kb" -eq 0 ] || [ "$kb" -eq 3 ] || exit "$kb"
if [ "$kg" -eq 0 ] && [ "$kb" -eq 0 ] && [ -f "$OUT/amendment1c_approval.txt" ]; then
    "$PY" "$here/table.py" gated --rank "$RANK" --rep "$REP" --proc "$PROC"; g=$?
    echo "gated exit $g (0 = the table passes its test)"
    [ "$g" -eq 0 ] || [ "$g" -eq 4 ] || exit "$g"
    echo "$OUT/angle_table_v1a1c_canon_v3${SFX}.json" > "$OUT/p03_table_used.txt"
else
    echo "the gated test does not run (amendment 1b not approved, or D.0 step 4): fitting the UNTESTED table for the ROC"
    "$PY" "$here/table.py" fit-only --rank "$RANK" --rep "$REP" --proc "$PROC" || exit $?
    echo "$OUT/angle_table_v1a1c_canon_v3_UNTESTED${SFX}.json" > "$OUT/p03_table_used.txt"
fi
exit 0
