#!/bin/bash
# The upward one-parameter scan from 4/4/4/4 in the jet tagger, PROMPT collection (physicist, 2026-11-15): for each point
# (ext_scan.UP_SCAN, in the order given there) one after the other, each step through the heavy lane (frozen copy):
#   1. the enlarged heads sample for the point: its per-track scan on the further files with TWO workers, the heads
#      retrained on it, their evaluation, the joint classifier test of the parametrization with them, the models shipped
#      (spixbits/prompt_enlarged.py --scenario <point> --steps scan_one heads_one eval_one closure_one ship_one);
#   2. the tagger run (run_prompt_quant.sh all <point>: enlarged heads, re-encoded candidates);
#   3. outside the lane (light, a minute each): the paired comparisons with "implemented now" and with 4/4/4/4
#      (analyze.py --paired-with) and the results file (scan_1d_table.py).
# Started with nohup from the repository root; NOT itself a lane job. Rerun-safe: a finished point (both paired files
# exist) is skipped, a finished step inside a point is skipped by the step itself. The lane serialises the steps, so a
# scan never runs beside a tagger training.
#   eval_refitq/tagger_sp/scan_1d.sh [POINT ...]        default: every point
# Survives the loss of the external disk or the SD card (waits, then starts the same step again). Stops, without working
# around it: internal disk below 12 GB free (exit 4); a step failing 6 times (exit 1); a tagger training with NO kept
# trial (exit 3). A tagger training with fewer than three kept trials after the top-up does not stop the scan: the results
# file marks it "short" and does not quote it.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MAIN=$HOME/smartpixels/ngtagger-train
PY=$MAIN/.pixi/envs/default/bin/python
BASE=/Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt
SLIM=/Volumes/NJM256GBSD/smartpixels/slimtaggertier/RelValTTbar_14TeV
LOG=$HOME/smartpixels/scratch_local/spixbits/logs/tagger_scan_1d.log
REFS="implemented_now_refheads_enlarged alloc_4444_qheads_enlarged_reencoded"
if [ $# -gt 0 ]; then POINTS=("$@"); else
  POINTS=($(cd "$REPO/eval_refitq/spixbits" && PYTHONPATH=$REPO/src $PY -c "import ext_scan; print(' '.join(ext_scan.UP_SCAN))")) || exit 1
fi
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
disks() { [ -d /Volumes/WDMac/smartpixels_scratch ] && [ -d "$SLIM" ]; }
wait_disks() { local n=0; until disks; do [ $((n % 10)) -eq 0 ] && say "waiting for WDMac and the SD card"; n=$((n + 1)); sleep 60; done; }
# lane NAME KIND PREREG LOGFILE COMMAND...: one lane job, up to 6 attempts; the Docker check as the standing rule says
lane() {
  local name=$1 kind=$2 prereg=$3 out=$4; shift 4
  for try in 1 2 3 4 5 6; do
    wait_disks
    FREE=$(df -g "$HOME" | awk 'NR==2 {print $4}')
    if [ "$FREE" -lt 12 ]; then say "STOP: internal disk $FREE GB free, below the 12 GB rule"; exit 4; fi
    [ "$FREE" -lt 15 ] && say "WARNING: internal disk $FREE GB free (below 15)"
    if docker ps > /dev/null 2>&1; then DOCK=""; DSAY="normal Docker check"; else DOCK="--no-docker-check"; DSAY="--no-docker-check (Docker not running)"; fi
    say "$name: start, attempt $try, $DSAY, internal disk $FREE GB free"
    "$MAIN/ops/spheavy" run --name "$name" --kind "$kind" --frozen $DOCK --repo "$REPO" --python "$PY" --tree eval_refitq \
      --on-busy wait --prereg "$prereg" -- "$@" >> "$out" 2>&1
    rc=$?
    say "$name: lane exit $rc"
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && return 3
    sleep 120
  done
  say "STOP: $name failed 6 times"; exit 1
}
say "1D scan: start (${#POINTS[@]} points: ${POINTS[*]}) from $REPO at $(git -C "$REPO" rev-parse --short HEAD)"
for Q in "${POINTS[@]}"; do
  DIR=${Q}_qheads_enlarged_reencoded
  wait_disks
  if [ -f "$BASE/$DIR/paired_vs_alloc_4444_qheads_enlarged_reencoded.json" ] && [ -f "$BASE/$DIR/paired_vs_implemented_now_refheads_enlarged.json" ]; then
    say "$Q: done before, skipped"; continue
  fi
  PRE="Upward one-parameter scan from 4/4/4/4, prompt collection, point $Q (one of x, y, alpha, beta raised, the other three at the 4/4/4/4 settings). Recipe of the first pass: prompt parametrization, track-quality heads retrained on the point's own tracks of the enlarged sample, candidates re-encoded, three input sets, 10 trials + 10 where fewer than 3 kept, best three; read at the 10 % and 1 % light-jet fake rate of each tagger, paired with 'implemented now' and with 4/4/4/4 on the same test events. Expected: between 4/4/4/4 and 'implemented now', rising with the y bits (the first pass's losses followed y); x, alpha, beta: no prior claim. A point equal to 4/4/4/4 within errors is a legitimate outcome. Angles are placeholders."
  lane "spixbits-enlarged-$Q" replay "$PRE Step 1: enlarged scan (two workers), heads, evaluation, joint classifier test." \
    "$HOME/smartpixels/scratch_local/spixbits/logs/prompt_enlarged_1d_lane.log" \
    python eval_refitq/spixbits/prompt_enlarged.py --scenario "$Q" --workers 2 --steps scan_one heads_one eval_one closure_one ship_one || exit $?
  if [ ! -f "$BASE/$DIR/results_ttbar/summary.json" ]; then
    mkdir -p "$BASE"
    lane "tagger-prompt-$DIR" tagger "$PRE Step 2: the tagger run." "$BASE/lane_$DIR.log" \
      bash eval_refitq/tagger_sp/run_prompt_quant.sh all "$Q"
    rc=$?
    say "$Q: tagger run ended ($(tail -1 "$BASE/$DIR/logs/run.log" 2>/dev/null))"
    if [ $rc -eq 3 ]; then say "STOP: $Q ended with exit 3 (a tagger training has no kept trial)"; exit 3; fi
  fi
  for R in $REFS; do
    (cd "$REPO" && TAGGER_SP_TIER=slim TAGGER_SP_COLLECTION=prompt SLIMTIER_INVENTORY_DIR=$BASE/inventory TAGGER_SP_QUANT=$Q \
      TAGGER_SP_HEADSET=quantization_enlarged TAGGER_SP_CANDIDATES=reencoded PYTHONPATH=$REPO/src \
      $PY eval_refitq/tagger_sp/analyze.py --boot 300 --paired-with "$BASE/$R" >> "$BASE/$DIR/logs/paired.log" 2>&1) || say "$Q: paired comparison with $R FAILED"
  done
  (cd "$REPO" && $PY eval_refitq/tagger_sp/scan_1d_table.py >> "$LOG" 2>&1) || say "$Q: results file FAILED"
  say "$Q: point done ($(grep -c 'short' "$HOME/smartpixels/doc-archive/spix-bits-refit-study/results/ptcap/tagger_1d_scan_prompt.md" 2>/dev/null) 'short' marks in the results file)"
done
say "1D scan: all points done"
