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
# take a confirmation from OL1-OL3; run C allows the OT confirmation but still
# only requires 3, which separates "the OT hit is available" from "the OT hit is
# mandatory". The A/B difference is what the OT buys; the B/C difference is what
# the four-layer rule costs in this regime.
#
# THE SEPTEMBER 16 RESULTS FROM THIS SCRIPT ARE NOT QUOTABLE. --seed-classes was
# a silent no-op, so every arm ran the full menu including OT and mixed seeds --
# visible in the old cache manifests, whose seed_tags list OL1+OL2, OL2+OL3 and
# IL4+OL1 under seed_classes [0]. An "IT-only" arm that contained OT seeds
# cannot answer whether the IT alone reaches soft tracks. Outputs are written
# under *_v2 so the retracted files stay on disk rather than being overwritten
# by numbers that look like a correction but share a filename.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-.pixi/envs/default/bin/python}
S=eval_refitq/combinatorics
R=$S/results
D=../cmssw/work/otstub_arm
ALL="$D/itot_tp_f*_100ev.root"
NEV=${NEV:-200}
MASKS=${MASKS:-AAAA,AAAI,AAIA,AIAA,IAAA}
mkdir -p "$R"

run () {                      # run <tag> <targets> <min_it_layers> <min_ot_conf>
  local tag=$1 tgt=$2 mil=$3 moc=$4
  echo "[soft] === $tag  targets=$tgt  min_it_layers=$mil  min_ot_conf=$moc"
  local t0=$SECONDS
  $PY -u $S/tp_findability.py -i "$ALL" -n "$NEV" --chunk 4 --ptmin 1.0 \
      --seed-classes it --targets "$tgt" \
      --min-it-layers "$mil" --min-ot-conf "$moc" \
      --masks "$MASKS" --export-tracks \
      --cache-dir "$S/cache_soft3_$tag" -o "$R/soft_${tag}_v3.json" \
      > "$R/soft_${tag}_v3.log" 2>&1 \
    && echo "[soft]     ok in $((SECONDS-t0))s -> $R/soft_${tag}_v3.log" \
    || { echo "[soft]     FAILED in $((SECONDS-t0))s"; tail -5 "$R/soft_${tag}_v3.log"; }
}

# THE RULE IS PER SYSTEM: 3xIT, or 3xIT + 1xOT. Never 2xIT + 1xOT -- an IT
# doublet completed by one distant OT stub cannot constrain d0 and is not a
# candidate worth forming. The v2 arms used a flat --min-layers and therefore
# allowed exactly that; their fake fractions rose with the IT lever arm
# (IL3+IL4 0.034 -> 0.067, IL1+IL4 0.202 -> 0.557) while every arity-3 seed was
# untouched (0.029 -> 0.029), which is the signature of the doublets being the
# whole effect. v2 is superseded, not corrected in place.
#
# A: 3xIT, IT targets only.
run itonly IL1,IL2,IL3,IL4 3 0
# B: 3xIT + 1xOT, the OT confirmation REQUIRED.
run withot IL1,IL2,IL3,IL4,OL1,OL2,OL3 3 1
# C: 3xIT with the OT confirmation OPTIONAL -- "3xIT or 3xIT + 1xOT". By
#    construction this accepts a superset of both A and B at the seeding stage,
#    so any efficiency it LOSES is the fit or the chi2 gate, not the rule.
run opportunistic IL1,IL2,IL3,IL4,OL1,OL2,OL3 3 0

echo "[soft] done at $(date)"
for t in itonly withot opportunistic; do
  printf "  %-14s %s\n" "$t" "$(grep -m1 'TPs with' "$R/soft_${t}_v3.log" 2>/dev/null | cut -c1-80)"
done
