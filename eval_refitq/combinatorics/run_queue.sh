#!/bin/bash
# Queued follow-ups, run STRICTLY IN SEQUENCE after the combined-architecture
# census finishes.
#
# Sequential is not laziness. Every one of these peaks at 1-2 GB, this machine
# has repeatedly been at a few hundred MB of headroom with other work running,
# and a concurrent launch has already cost one census run mid-flight. The queue
# waits on the census PID rather than a timer so it cannot start early.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-.pixi/envs/default/bin/python}
S=eval_refitq/combinatorics
R=$S/results
D=../cmssw/work/otstub_arm
TT="$D/itot_tp_f01_100ev.root"
ALL="$D/itot_tp_f*_100ev.root"
WAIT_PID=${WAIT_PID:-}
mkdir -p "$R"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting on census pid $WAIT_PID"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  echo "[queue] census finished, starting at $(date)"
fi

step () {                     # step <name> <logfile> <cmd...>
  local name=$1 log=$2; shift 2
  echo "[queue] === $name"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    echo "[queue]     ok in $((SECONDS - t0))s -> $log"
  else
    echo "[queue]     FAILED in $((SECONDS - t0))s, see $log"
    tail -5 "$log"
  fi
}

# 0. Conflict-graph scale: decides whether the interactive page can do
#    duplicate removal the way a real system would -- keyed on SHARED HITS,
#    never on truth. Grouping by TrackingParticle is an oracle no L1 system has,
#    and it distorts BOTH ways at once: real tracks deduplicated better than any
#    implementable DR could manage, fakes not deduplicated at all. Run first
#    because the answer changes what the exporter must ship.
#    10% of the sample, not a 50k toy: one low-statistics point is not a basis
#    for a claim about interactivity.
step "DR conflict graph scale (100 ev)" "$R/dr_scale.log" \
  $PY -u $S/dr_conflict_scale.py -i "$ALL" -n 100 --chunk 8 \
      -o "$R/dr_conflict_scale.json"

# 1. Quality MVA on full statistics. The 96.2% three-class accuracy and the
#    sigma(d0)-by-predicted-class calibration were measured on EIGHT events.
step "track quality MVA, full stats" "$R/tq_mva.log" \
  $PY -u $S/track_quality_mva.py --cache-dir $S/cache -o "$R/tq_mva.json"

# 2. Inter-system scattering, binned in |eta|. Material grows with |eta| while
#    the measurement variance's pT dependence does not, which is what breaks the
#    degeneracy that made the phi0 channel unusable barrel-wide -- and it is the
#    form a tkLayout x/X0-vs-eta curve can actually be compared against.
step "scattering vs eta" "$R/scattering_eta.log" \
  $PY -u $S/arch_comparison.py -i "$ALL" -n 200 --chunk 8 --scattering \
      -o "$R/arch_scattering_eta.json"

# 3. Parallel-architecture track rows, so the page can show all three designs.
#    Separate file from the census: its schema carries arch_mode and track_stage,
#    which the census shards cannot without moving their format version.
step "parallel architecture tracks" "$R/arch_tracks.log" \
  $PY -u $S/arch_comparison.py -i "$ALL" -n 400 --chunk 8 \
      --export-tracks "$R/arch_parallel_tracks.npz" \
      -o "$R/arch_comparison_1000.json"

echo "[queue] done at $(date)"
for f in "$R"/dr_scale.log "$R"/tq_mva.log "$R"/scattering_eta.log "$R"/arch_tracks.log; do
  printf "  %-28s %s\n" "$(basename "$f")" "$(tail -1 "$f" | cut -c1-70)"
done

# NOTE FOR ANYONE EDITING THIS FILE WHILE IT IS RUNNING: don't. bash reads a
# script by BYTE OFFSET, so inserting or deleting lines above the point it has
# reached makes it resume mid-token and execute garbage. If the queue is already
# waiting on a PID, kill it and relaunch after editing. This was learned the
# obvious way.
