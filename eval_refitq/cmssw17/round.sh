#!/bin/bash
# One full round of the comparison's checks on D121 alone (plan 3.4.4-3.4.7): freeze, the primary null, the
# pre-declared fallback, the four nulls with it, the planted defects (1-7, then 10), the defect report.
# Run from the ngtagger-train-cmssw17 worktree. Logs and outputs under $CMSSW17_COMPARE_OUT (default WDMac).
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD/src
PY=${PY:-$HOME/smartpixels/ngtagger-train/.pixi/envs/default/bin/python}
L=${CMSSW17_COMPARE_OUT:-/Volumes/WDMac/smartpixels_scratch/cmssw17_compare}
R="nice -n 10 $PY eval_refitq/cmssw17/run.py"
$R freeze > $L/freeze.log 2>&1
$R null --split oddeven > $L/null_oddeven_prefallback.log 2>&1
$R fallback > $L/fallback.log 2>&1
for s in oddeven halves randomA randomB; do $R null --split $s > $L/null_$s.log 2>&1; done
$R defects > $L/defects.log 2>&1
$R defects --defect no_1gev_rule > $L/defects_no1gev.log 2>&1
$R defects-report > $L/defects_report.log 2>&1 || true
echo ROUND-DONE
