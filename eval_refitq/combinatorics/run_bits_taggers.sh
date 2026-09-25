#!/bin/bash
# Do the taggers get worse when alpha and beta are quantised UPSTREAM?
#
# The bit grid measured cost, reach and resolution, but it ran without track
# rows, so nothing could be trained on it. This rebuilds three settings WITH
# --export-tracks and trains the cleanliness and d0-accuracy taggers on each,
# in both prompt and displaced variants, against the full-precision numbers.
#
# Quantising alpha/beta changes which hits are attached, so the tagger sees a
# different track population AND different angle features. That is the effect
# a post-hoc quantisation of the feature columns cannot reproduce.
#
# DO NOT EDIT WHILE RUNNING: bash reads this file by byte offset.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.pixi/envs/default/bin/python
S=eval_refitq/combinatorics
D=../cmssw/work/otstub_arm
ALL="$D/itot_tp_f*_100ev.root"
NEV=${NEV:-100}
OUT=$S/results/scan
CACHE=$S/cache-bits
mkdir -p "$OUT" "$CACHE"

for b in 3 5 7; do
  tag="taggers_a${b}_b${b}"
  [ -f "$OUT/$tag.json" ] && { echo "[bt] $tag done"; continue; }
  if [ "$(df -g /Users/nmangane | awk 'NR==2{print $4}')" -lt 5 ]; then
    echo "[bt] STOPPING: low disk"; exit 1
  fi
  echo "[bt] census a$b b$b  $(date +%H:%M:%S)"
  $PY -u $S/tp_findability.py -i "$ALL" -n "$NEV" --chunk 4 \
      --alpha-bits "$b" --beta-bits "$b" --export-tracks \
      --cache-dir "$CACHE" -o "$OUT/bits_tracks_a${b}_b${b}.json" \
      > "$OUT/$tag.census.log" 2>&1 || { echo "[bt] census FAILED"; continue; }
  dir=$(ls -dt $CACHE/tpcensus_* | head -1)
  echo "[bt] taggers on $dir  $(date +%H:%M:%S)"
  $PY -u $S/quality_taggers.py --census "$dir" --only clean,d0 \
      -o "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 \
    || echo "[bt] taggers FAILED for a$b b$b"
done
echo "[bt] all done $(date +%H:%M:%S)"
