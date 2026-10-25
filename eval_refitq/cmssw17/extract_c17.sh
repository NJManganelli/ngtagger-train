#!/bin/bash
# After the CMSSW_17 production (cmssw17_prod/ttbar_spring24, 10 files x 4 masks): the comparison's records of the
# CMSSW_17 side, base (extract.py) and ML (extract_ml.py), then the comparison itself (frozen tables required).
# Run from the ngtagger-train-cmssw17 worktree; light (host Python), about 10 min.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/src
PY=${PY:-$HOME/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}
L=${CMSSW17_COMPARE_OUT:-/Volumes/WDMac/smartpixels_scratch/cmssw17_compare}
for m in IIII AAAA AAAI IAAA; do
  nice -n 10 $PY eval_refitq/cmssw17/extract.py --side c17 --mask $m --file 1 2 3 4 5 6 7 8 9 10 >> $L/extract_c17.log 2>&1
  nice -n 10 $PY eval_refitq/cmssw17/extract_ml.py --side c17 --mask $m --file 1 2 3 4 5 6 7 8 9 10 >> $L/extract_ml_c17.log 2>&1
done
# the payload-overlap rule decided 2026-10-25: (a) "sole"
nice -n 10 $PY eval_refitq/cmssw17/run.py compare --mixed-rule sole > $L/compare_sole.log 2>&1
echo COMPARE-DONE
