#!/bin/bash
# Rebuild the explorer site (site_v2) from canon_v2 AAAA f03-f10, the FIRST 200
# EVENTS (= f03 + f04; f01/f02 are the cluster-CNN training events and are not
# inputs, nor are the *_rerun.root files). |eta| summary split X = 1.8.
# Run sequentially, one heavy job at a time; caches, logs and the site go to
# WDMac (internal disk rule).
#
# MASKS: the builds the IT+OT census runs. Each build's seeds follow only into
# that build's own surfaces, so every build is its own set of entries and its
# own track rows: the default is every build; the user may restrict it (e.g.
# AAAA,AAAI,IAAA). The OT-only baseline census follows OT layers only, which
# every build shares, so it is one set of entries whatever MASKS is.
set -eu
PY=/Users/nmangane/smartpixels/ngtagger-train/.pixi/envs/default/bin/python
S=/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics
C2=/Volumes/WDMac/smartpixels_scratch/canon_v2
W=/Volumes/WDMac/smartpixels_scratch/site_v2
IN="$C2/spstudy_AAAA_f03_100ev.root,$C2/spstudy_AAAA_f04_100ev.root,$C2/spstudy_AAAA_f05_100ev.root,$C2/spstudy_AAAA_f06_100ev.root,$C2/spstudy_AAAA_f07_100ev.root,$C2/spstudy_AAAA_f08_100ev.root,$C2/spstudy_AAAA_f09_100ev.root,$C2/spstudy_AAAA_f10_100ev.root"
NEV=200
MASKS=${MASKS:-AAAA,AAAI,AAIA,AIAA,IAAA,AAII,AIAI,AIIA,IAAI,IAIA,IIAA,AIII,IAII,IIAI,IIIA}
mkdir -p $W/logs
cd /Users/nmangane/smartpixels/ngtagger-train
echo "[$(date +%H:%M)] census: IT+OT menu, builds $MASKS"
caffeinate -i $PY -u $S/tp_findability.py -i "$IN" -n $NEV --chunk 4 --calib-events 100 \
    --export-tracks --it-ptmin 1.0 --min-layers 4 --min-it-layers 3 --masks "$MASKS" \
    --cache-dir $W/cache-tp-incl -o $W/census_tp_incl.json > $W/logs/census_tp_incl.log 2>&1
echo "[$(date +%H:%M)] census: OT-only re-emulated baseline"
caffeinate -i $PY -u $S/tp_findability.py -i "$IN" -n $NEV --chunk 4 --calib-events 100 \
    --export-tracks --seed-classes ot --targets OL1,OL2,OL3,OL4,OL5,OL6 --min-layers 4 \
    --cache-dir $W/cache-tp-otonly -o $W/census_tp_otonly.json > $W/logs/census_tp_otonly.log 2>&1
echo "[$(date +%H:%M)] export_site -> $W/site"
caffeinate -i $PY -u $S/export_site.py --cache-dir $W/cache-tp-incl -o $W/site \
    --baseline-input "$IN" --baseline-census $W/cache-tp-otonly --baseline-nev $NEV \
    --eta-x 1.8 > $W/logs/export_site.log 2>&1
echo "[$(date +%H:%M)] done"
