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
  $PY -u $S/validate_samples.py      -i "$in"                        > "$OUT/${n}_samples.txt"      2>&1
  $PY -u $S/seeder_design_basis.py   -i "$in" --payload "$PAYLOAD" \
                                  -o "$OUT/${n}_design_basis.json" > "$OUT/${n}_design_basis.txt" 2>&1
  $PY -u $S/projection_residuals.py  -i "$in"                        > "$OUT/${n}_projection_residuals.txt" 2>&1
  $PY -u $S/triplet_d0_solve.py      -i "$in"                        > "$OUT/${n}_triplet_d0_solve.txt"     2>&1
  # SPLIT BY BENCHMARK. All four at once costs a MEASURED 32.5 s/event, i.e. ~9
  # hours for the 1000-event ttbar set. The three quantized benchmarks are
  # RELATIVE comparisons against native and settle in a couple hundred events;
  # only native plausibly needs full statistics, and then only for the >500 um d0
  # band. So native runs on everything and the quantized ones are capped.
  $PY -u $S/tracklet_topology_cost.py -i "$in" --batch-events 8 --benchmark native \
                                  -o "$OUT/${n}_cost_native.json" > "$OUT/${n}_cost_native.txt" 2>&1
  for b in A_a3b5 B_a4b6 C_a3b7; do
    $PY -u $S/tracklet_topology_cost.py -i "$in" --batch-events 8 -n "${QNEV:-200}" \
        --benchmark "$b" --skip-ot -o "$OUT/${n}_cost_${b}.json" > "$OUT/${n}_cost_${b}.txt" 2>&1
  done
  for f in samples design_basis projection_residuals triplet_d0_solve cost_native; do
    printf "  %-22s %s\n" "$f" "$(tail -1 "$OUT/${n}_${f}.txt" | cut -c1-70)"
  done
done
echo "############ results in $OUT"
