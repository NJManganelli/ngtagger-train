#!/bin/bash
# The first pass of the jet-tagger retraining on the PROMPT collection: the runs of RUNSHEET_tagger_prompt.md one after
# the other, each through the heavy lane (ops/spheavy, frozen copy), in the order given. Started with nohup from the
# repository root; NOT itself a lane job. Rerun-safe: a finished run (its results_ttbar/summary.json exists) is skipped.
#   eval_refitq/tagger_sp/first_pass.sh            the whole pass
# Survives the loss of the external disk or the SD card: before and after a failed run it waits until both are back and
# starts the same run again (the build reuses finished jet caches, the training skips finished trials: at most the
# trials in flight are lost). Stops, without working around it, when the internal disk has less than 12 GB free, when a
# run fails 6 times, or when a run ends with exit 3 (a tagger training with no kept trial after the top-up).
# Docker: the lane's normal check when `docker ps` connects; --no-docker-check when the daemon is not running (standing
# rule of 2026-11-12); which one is written to the log for every run.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MAIN=$HOME/smartpixels/ngtagger-train
BASE=/Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt
SLIM=/Volumes/NJM256GBSD/smartpixels/slimtaggertier/RelValTTbar_14TeV
LOG=$HOME/smartpixels/scratch_local/spixbits/logs/tagger_first_pass.log
# quantization scenario | head set | candidates | output folder
RUNS=(
  "implemented_now|reference_enlarged||implemented_now_refheads_enlarged"
  "implemented_now|detector||implemented_now"
  "alloc_4444|quantization_enlarged|reencoded|alloc_4444_qheads_enlarged_reencoded"
  "alloc_6604|quantization_enlarged|reencoded|alloc_6604_qheads_enlarged_reencoded"
  "alloc_6622|quantization_enlarged|reencoded|alloc_6622_qheads_enlarged_reencoded"
  "alloc_5533|quantization_enlarged|reencoded|alloc_5533_qheads_enlarged_reencoded"
  "network_float|quantization_enlarged|reencoded|network_float_qheads_enlarged_reencoded"
)
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
disks() { [ -d /Volumes/WDMac/smartpixels_scratch ] && [ -d "$SLIM" ]; }
wait_disks() { local n=0; until disks; do [ $((n % 10)) -eq 0 ] && say "waiting for WDMac and the SD card"; n=$((n + 1)); sleep 60; done; }
say "first pass: start (${#RUNS[@]} runs) from $REPO at $(git -C "$REPO" rev-parse --short HEAD)"
for r in "${RUNS[@]}"; do
  IFS='|' read -r Q H CAND DIR <<< "$r"
  wait_disks
  if [ -f "$BASE/$DIR/results_ttbar/summary.json" ]; then say "$DIR: done before, skipped"; continue; fi
  ok=0
  for try in 1 2 3 4 5 6; do
    wait_disks
    FREE=$(df -g "$HOME" | awk 'NR==2 {print $4}')
    if [ "$FREE" -lt 12 ]; then say "STOP: internal disk $FREE GB free, below the 12 GB rule"; exit 4; fi
    if docker ps > /dev/null 2>&1; then DOCK=""; DSAY="normal Docker check"; else DOCK="--no-docker-check"; DSAY="--no-docker-check (Docker not running)"; fi
    say "$DIR: start, attempt $try, $DSAY, internal disk $FREE GB free"
    mkdir -p "$BASE"
    "$MAIN/ops/spheavy" run --name "tagger-prompt-$DIR" --kind tagger --frozen $DOCK --repo "$REPO" \
      --python "$MAIN/.pixi/envs/default/bin/python" --tree eval_refitq --on-busy wait \
      --prereg "Jet-tagger retraining on the PROMPT collection, slim tagger tier TTbar (8,500 events common to IIII, AAAA, AAAI, IAAA; RUNSHEET_tagger_prompt.md): quantization scenario '$Q', track-quality heads '$H', candidates' dxy and z0 '${CAND:-as produced}'. Twelve tagger trainings per run = four detector scenarios x three input sets (candidates only; + track quantities; + track quantities and track-quality heads); the ones this run cannot change are taken from the 'implemented now' run. 10 first trials + 10 more where fewer than 3 are kept; best three deployed. Working points: 10 % and 1 % fake rates of each retrained tagger itself. Expected: 'implemented now' close to the extended-collection slim-tier study; a quantization scenario at or below 'implemented now' in the SmartPixels-minus-IIII steps, the loss growing with the d0 width." \
      -- bash eval_refitq/tagger_sp/run_prompt_quant.sh all "$Q" "$H" $CAND >> "$BASE/lane_$DIR.log" 2>&1
    rc=$?
    say "$DIR: lane exit $rc ($(tail -1 "$BASE/$DIR/logs/run.log" 2>/dev/null))"
    if [ $rc -eq 0 ] && [ -f "$BASE/$DIR/results_ttbar/summary.json" ]; then ok=1; break; fi
    if [ $rc -eq 3 ]; then say "STOP: $DIR ended with exit 3 (a tagger training has no kept trial)"; exit 3; fi
    sleep 120
  done
  [ $ok -eq 1 ] || { say "STOP: $DIR failed 6 times"; exit 1; }
done
say "first pass: all runs done"
