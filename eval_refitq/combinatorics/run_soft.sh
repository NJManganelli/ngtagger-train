#!/bin/bash
# Soft-track (1-2 GeV) finding: can SmartPixels reach the tracks the OT cannot?
#
# WHY IT-ONLY SEEDS. The OT does not lose soft tracks at the seeding stage, it
# loses them at the STUB stage -- the module bend window is a front-end pT
# filter. MEASURED on 100 events: mean OT layers per TP falls from 2.46 above
# 2 GeV to 0.87 below it, while mean IT layers is FLAT at 2.45. Only 4.4% of
# 1-2 GeV TPs reach the >= 4 OT layers an OT track needs, against 42.3% just
# above threshold. So the OT cannot seed here at all, and soft finding needs
# >= 3 instrumented IT layers -- a 2-layer build cannot form a 3-hit mini-track.
#
# WHY THE OT STILL MATTERS, and this is the point of the A/B below. Soft tracks
# DO make stubs, just on the inner layers: at 1.0-1.5 GeV the per-layer
# occupancy is OL1 14.5%, OL2 25.3%, OL3 13.4%, then ~2% on OL4-OL6. And of the
# 1-2 GeV TPs with EXACTLY 3 IT layers -- the ones that would otherwise be a
# zero-degree-of-freedom triplet with no r-phi chi2 to reject on -- 59.5% have
# at least one stub on OL1 or OL2. One OT hit turns a fit with no handle into
# one with a handle, for 60% of exactly the population that needs it, and it
# matters most in the 3-layer build that is likelier to be built.
#
# So: run A requires 3 layers and follows the IT only; run B requires 4 and may
# take a confirmation from OL1-OL3. The difference is what the OT buys.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-.pixi/envs/default/bin/python}
S=eval_refitq/combinatorics
R=$S/results
D=../cmssw/work/otstub_arm
ALL="$D/itot_truth_100ev.root,$D/itot_ttbar_f*_100ev.root"
NEV=${NEV:-200}
MASKS=${MASKS:-AAAA,AAAI,AAIA,AIAA,IAAA}
mkdir -p "$R"

run () {                      # run <tag> <targets> <minlayers>
  local tag=$1 tgt=$2 ml=$3
  echo "[soft] === $tag  targets=$tgt  min_layers=$ml"
  local t0=$SECONDS
  $PY -u $S/tp_findability.py -i "$ALL" -n "$NEV" --chunk 4 --ptmin 1.0 \
      --seed-classes it --targets "$tgt" --min-layers "$ml" \
      --masks "$MASKS" --export-tracks \
      --cache-dir "$S/cache_soft_$tag" -o "$R/soft_$tag.json" \
      > "$R/soft_$tag.log" 2>&1 \
    && echo "[soft]     ok in $((SECONDS-t0))s -> $R/soft_$tag.log" \
    || { echo "[soft]     FAILED in $((SECONDS-t0))s"; tail -5 "$R/soft_$tag.log"; }
}

# A: IT only. In a 3-layer build a triplet consumes every layer, so there is no
#    fourth hit available and min_layers must be 3 -- demanding 4 would reject
#    every soft track by geometry rather than by quality.
run itonly IL1,IL2,IL3,IL4 3
# B: the same seeds, allowed one confirmation from the inner OT.
run withot IL1,IL2,IL3,IL4,OL1,OL2,OL3 4

echo "[soft] done at $(date)"
for t in itonly withot; do
  printf "  %-10s %s\n" "$t" "$(grep -m1 'TPs with' "$R/soft_$t.log" 2>/dev/null | cut -c1-80)"
done
