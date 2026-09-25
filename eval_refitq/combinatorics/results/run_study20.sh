#!/bin/bash
# Study 20 (refit-update projection) on the 1000-event PU200 ttbar set (itot_tp).
#   bash run_study20.sh <mask> <cal|prod>      one activeSP mask, one covariance model
#   bash run_study20.sh all                    all 15 masks: calibrated first, then producer
# The replay gate always replays the producer's AAAA refit (the only refit table in the
# nanos): for other masks it validates the update machinery, not that scenario.
set -u
cd /Users/nmangane/smartpixels/ngtagger-train
PY=.pixi/envs/default/bin/python
S=eval_refitq/combinatorics
D=/Users/nmangane/smartpixels/cmssw/work/otstub_arm
MASKS="AAAA AAAI AAIA AIAA IAAA AAII AIAI AIIA IAAI IAIA IIAA AIII IAII IIAI IIIA"
one() {
  local mask=$1 model=$2 out=$S/results/refit_study20_${2}_$1
  local extra=()
  [ "$model" = cal ] && extra=(--refit-calibration $S/results/refit_calibration/refit_calibration_v3.json)
  mkdir -p "$out"
  $PY -u $S/spix_combinatorics_omnibus.py -i $D/itot_tp_f01_100ev.root --only refit-update \
      --geometry $D/spix_module_geometry_D121.json --proj-inputs "$D/itot_tp_f*_100ev.root" \
      --refit-mask "$mask" -o "$out" ${extra[@]+"${extra[@]}"} > "$out/run.log" 2>&1
  echo "[$(date +%H:%M)] $model $mask rc=$?"
}
if [ "${1:-}" = all ]; then
  for m in $MASKS; do [ -s $S/results/refit_study20_cal_$m/spix_ot_refit_projection_$m.pdf ] || one $m cal; done
  for m in $MASKS; do [ -s $S/results/refit_study20_prod_$m/spix_ot_refit_projection_$m.pdf ] || one $m prod; done
else
  one "$1" "$2"
fi
