#!/bin/bash
# The jet-tagging study on the PROMPT track collection (the baseline) for one quantization scenario, on the slim
# tagger tier; run by ops/spheavy from the frozen copy's root (or from the repository root). Rerun-safe like run_all.sh.
#   eval_refitq/tagger_sp/run_prompt_quant.sh PHASE QUANT [HEADSET [CANDIDATES]]
#     PHASE    build | train | all   (as run_all.sh)
#     QUANT    implemented_now, or a quantization scenario of the prompt parametrization (alloc_4444, up_y_6b, ...)
#     HEADSET  the track-quality heads (common.py). Default: the per-quantization-scenario heads of the ENLARGED sample
#              (844 training events; they exist for the first-pass quantization scenarios only), i.e.
#                a quantization scenario: quantization_enlarged  (heads retrained on that scenario's prompt tracks)
#                implemented_now:         reference_enlarged     (the "implemented now" heads of the same sample: its fair reference)
#              quantization / reference: the same two from the five scan files only (338 training events; they exist
#              for all 64 sub-area quantization scenarios).
#              detector: the heads retrained per detector scenario on today's prompt tracks with the full sample (the
#              comparison that shows what the smaller heads sample costs); allowed for both.
#     CANDIDATES  reencoded (default): the PUPPI candidate's own dxy and z0 re-derived from the shifted track with the
#              trigger chain's encoding (on today's tracks this reproduces the stored values exactly)
#              today: left as produced (a control: the tagger then sees today's d0 through the coarse dxy code)
# Output folder: /Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt/<QUANT>[_qheads|_refheads][_reencoded] (common.OUT).
# Order: implemented_now with HEADSET reference_enlarged FIRST. Every run uses one frozen file inventory and the same common
# events ($BASE/inventory), and takes by symbolic link the predictions of the cells it cannot change, which train.py
# then skips:
#   implemented_now, another   from implemented_now_refheads_enlarged: every cell of IIII and the "baseline" and "track"
#   head set                   cells (only the heads differ): 3 cells trained
#   a quantization scenario    from the "implemented now" run of its head set (reference_enlarged for
#                              quantization_enlarged, reference for quantization, detector for detector): every cell of
#                              IIII, and with CANDIDATES=today also the "baseline" cells: 9 cells trained (6 with today)
set -u
PHASE=${1:?phase: build | train | all}
QUANT=${2:?quantization scenario, or implemented_now}
BASE=/Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt
export TAGGER_SP_TIER=slim TAGGER_SP_COLLECTION=prompt SLIMTIER_INVENTORY_DIR=$BASE/inventory
mkdir -p "$BASE/inventory"
link() { for f in "$@"; do [ -e "$f" ] && [ ! -e "$OUT/preds/$(basename "$f")" ] && ln -s "$f" "$OUT/preds/$(basename "$f")"; done; return 0; }
FIRST=$BASE/implemented_now_refheads_enlarged      # the first run: the primary reference
if [ "$QUANT" = implemented_now ]; then      # one folder per head set: re-encoding today's tracks reproduces the stored candidates
  HEADSET=${3:-reference_enlarged}
  case $HEADSET in
    reference_enlarged) ;;
    reference|detector)
      [ -f "$FIRST/selection.json" ] || { echo "train implemented_now with the reference_enlarged heads first ($FIRST/selection.json missing)" >&2; exit 2; }
      OUT=$BASE/implemented_now$([ "$HEADSET" = reference ] && echo _refheads)
      mkdir -p "$OUT/preds"
      link "$FIRST"/preds/IIII_*.npz "$FIRST"/preds/*_baseline_s*.npz "$FIRST"/preds/*_track_s*.npz ;;
    *) echo "implemented_now: HEADSET reference_enlarged, reference or detector" >&2; exit 2 ;;
  esac
  export TAGGER_SP_HEADSET=$HEADSET
else
  HEADSET=${3:-quantization_enlarged}
  CANDIDATES=${4:-reencoded}
  case $HEADSET in
    quantization_enlarged) REF=$FIRST; SUF=_qheads_enlarged ;;
    quantization) REF=$BASE/implemented_now_refheads; SUF=_qheads ;;
    detector) REF=$BASE/implemented_now; SUF= ;;
    *) echo "a quantization scenario: HEADSET quantization_enlarged, quantization or detector" >&2; exit 2 ;;
  esac
  [ -f "$REF/selection.json" ] || { echo "train implemented_now with the matching heads first ($REF/selection.json missing)" >&2; exit 2; }
  export TAGGER_SP_QUANT=$QUANT TAGGER_SP_HEADSET=$HEADSET TAGGER_SP_CANDIDATES=$CANDIDATES
  OUT=$BASE/$QUANT$SUF$([ "$CANDIDATES" = reencoded ] && echo _reencoded)
  mkdir -p "$OUT/preds"
  link "$REF"/preds/IIII_*.npz
  [ "$CANDIDATES" = today ] && link "$REF"/preds/*_baseline_s*.npz
fi
exec bash "$(dirname "$0")/run_all.sh" "$PHASE" ttbar      # (a frozen copy keeps no execute permission)
