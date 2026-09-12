#!/bin/bash
# Every seeding study, on every process, kept SEPARATE.
#
# The three conditions are not interchangeable: PU200 ttbar and PU200 VBF
# H->tautau have nearly identical occupancy (pileup dominates the cluster count,
# so the hard process barely registers) but different heavy-flavour content,
# while noPU ttbar has ~45x fewer clusters and is the only independent lever on
# how cost scales with occupancy. Merging them would hide all three facts.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-/Users/nmangane/.pixi/envs/uproot/bin/python}
S=eval_refitq/combinatorics
D=../cmssw/work/otstub_arm
PAYLOAD=../cmssw/work/spxsmoke/spix_angle_response_Conv1D_Full-2bit_v4fixed.json
OUT=${OUT:-$S/results}
mkdir -p "$OUT"

declare -a NAMES=(ttbar_pu200 htt_pu200 ttbar_nopu)
declare -a INPUTS=(
  "$D/itot_truth_100ev.root,$D/itot_ttbar_f*_100ev.root"
  "$D/itot_htt_100ev.root"
  "$D/itot_nopu_100ev.root"
)
for i in 0 1 2; do
  n=${NAMES[$i]}; in=${INPUTS[$i]}
  echo "############ $n"
  $PY $S/validate_samples.py      -i "$in"                        > "$OUT/${n}_samples.txt"      2>&1
  $PY $S/seeder_design_basis.py   -i "$in" --payload "$PAYLOAD" \
                                  -o "$OUT/${n}_design_basis.json" > "$OUT/${n}_design_basis.txt" 2>&1
  $PY $S/projection_residuals.py  -i "$in"                        > "$OUT/${n}_projection_residuals.txt" 2>&1
  $PY $S/triplet_d0_solve.py      -i "$in"                        > "$OUT/${n}_triplet_d0_solve.txt"     2>&1
  $PY $S/tracklet_topology_cost.py -i "$in" --batch-events 8 \
                                  -o "$OUT/${n}_cost.json"        > "$OUT/${n}_cost.txt"          2>&1
  for f in samples design_basis projection_residuals triplet_d0_solve cost; do
    printf "  %-22s %s\n" "$f" "$(tail -1 "$OUT/${n}_${f}.txt" | cut -c1-70)"
  done
done
echo "############ results in $OUT"
