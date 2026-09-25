#!/bin/bash
# Rebuild the explorer site from the regenerated nanos (itot_tp_f*_100ev.root:
# L1TTP truth at the POCA, tp_phi0, fixed producer Q). Same census
# configurations as the 2026-09-20/22 site (cache-incl a27db6c4 and cache-otonly
# 2a17ab70 manifests); run sequentially, one heavy job at a time.
set -eu
PY=/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python
S=/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics
IN="/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root"
cd /Users/nmangane/smartpixels/ngtagger-train
echo "[$(date +%H:%M)] census: IT+OT menu (as cache-incl)"
$PY -u $S/tp_findability.py -i "$IN" --chunk 4 --calib-events 100 --export-tracks \
    --it-ptmin 1.0 --min-layers 4 --min-it-layers 3 --cache-dir $S/cache-tp-incl \
    -o $S/results/census_tp_incl.json > $S/results/census_tp_incl.log 2>&1
echo "[$(date +%H:%M)] census: OT-only re-emulated baseline (as cache-otonly)"
$PY -u $S/tp_findability.py -i "$IN" --chunk 4 --calib-events 100 --export-tracks \
    --seed-classes ot --targets OL1,OL2,OL3,OL4,OL5,OL6 --min-layers 4 \
    --cache-dir $S/cache-tp-otonly -o $S/results/census_tp_otonly.json > $S/results/census_tp_otonly.log 2>&1
echo "[$(date +%H:%M)] export_site -> site_tp_new"
$PY -u $S/export_site.py --cache-dir $S/cache-tp-incl -o $S/site_tp_new \
    --baseline-input "$IN" --baseline-census $S/cache-tp-otonly \
    > $S/results/export_site_tp.log 2>&1
echo "[$(date +%H:%M)] done"
