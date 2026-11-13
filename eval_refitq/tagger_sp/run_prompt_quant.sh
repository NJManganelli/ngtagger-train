#!/bin/bash
# The jet-tagging study on the PROMPT track collection (the baseline) for one quantization scenario, on the slim
# tagger tier; run by ops/spheavy from the frozen copy's root (or from the repository root). Rerun-safe like run_all.sh.
#   eval_refitq/tagger_sp/run_prompt_quant.sh PHASE QUANT [HEADSET]
#     PHASE    build | train | all   (as run_all.sh)
#     QUANT    implemented_now, or a quantization scenario of the prompt parametrization (alloc_4444, up_y_6b, ...)
#     HEADSET  detector (default): heads retrained per detector scenario on today's prompt tracks
#              quantization: heads retrained on that quantization scenario's prompt tracks (five files only)
# Output folder: /Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt/<QUANT>[_qheads] (common.OUT).
# "implemented_now" must be built and trained first: every other folder
#   - uses its frozen file inventory and common events (SLIMTIER_INVENTORY_DIR), so all quantization scenarios train
#     and test on the same events;
#   - takes from it, by symbolic link, the predictions of the cells a quantization scenario cannot change: every cell
#     of IIII (no SmartPixels layers) and the "baseline" cell of AAAA / AAAI / IAAA (candidates and jets only, which the
#     shift does not touch). train.py then skips them, so 6 of the 12 cells are trained per quantization scenario.
set -u
PHASE=${1:?phase: build | train | all}
QUANT=${2:?quantization scenario, or implemented_now}
HEADSET=${3:-detector}
BASE=/Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt
REF=$BASE/implemented_now
export TAGGER_SP_TIER=slim TAGGER_SP_COLLECTION=prompt TAGGER_SP_HEADSET=$HEADSET SLIMTIER_INVENTORY_DIR=$REF/cache
if [ "$QUANT" = implemented_now ]; then
  [ "$HEADSET" = detector ] || { echo "implemented_now has the detector-scenario heads only" >&2; exit 2; }
  mkdir -p "$REF/cache"
else
  export TAGGER_SP_QUANT=$QUANT
  [ -f "$REF/selection.json" ] || { echo "train implemented_now first ($REF/selection.json missing)" >&2; exit 2; }
  OUT=$BASE/$QUANT$([ "$HEADSET" = quantization ] && echo _qheads)
  mkdir -p "$OUT/preds"
  for f in "$REF"/preds/IIII_*.npz "$REF"/preds/*_baseline_s*.npz; do
    [ -e "$OUT/preds/$(basename "$f")" ] || ln -s "$f" "$OUT/preds/$(basename "$f")"
  done
fi
exec "$(dirname "$0")/run_all.sh" "$PHASE" ttbar
