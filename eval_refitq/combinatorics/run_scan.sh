#!/bin/bash
# Two scans, strictly sequential. DO NOT EDIT WHILE RUNNING: bash reads this
# file by byte offset, so an edit mid-flight makes it resume at the wrong place.
#
# Phase 1, the projection-window test. A doublet seed assumes d0 = 0, so at
# r = 3 cm a real 1 mm impact parameter throws the azimuth by 33 mrad and falls
# outside the phi road. If that is where displaced tracks are lost, widening the
# allowance should recover them -- at a cost in candidates that this measures.
#
# Phase 2, the alpha/beta bit grid, 2..7 bits each. The quantised angle feeds
# kap_a/s_kap, which the SEEDING and PROJECTION use as well as the KF, so a bit
# width changes which candidates exist and cannot be applied after the fact.
# 2 bits is degenerate by construction (QUANT_RESERVE = 3 leaves one code), so
# that row/column is a no-angle-information baseline rather than a coarse
# measurement.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.pixi/envs/default/bin/python
S=eval_refitq/combinatorics
D=../cmssw/work/otstub_arm
ALL="$D/itot_tp_f*_100ev.root"
NEV=${NEV:-100}
OUT=$S/results/scan
mkdir -p "$OUT"

free_gb () { df -g /Users/nmangane | awk 'NR==2{print $4}'; }

run () {                      # run <tag> <extra args...>
  local tag=$1; shift
  local log="$OUT/$tag.log"
  if [ -f "$OUT/$tag.json" ]; then echo "[scan] $tag already done"; return; fi
  if [ "$(free_gb)" -lt 5 ]; then
    echo "[scan] STOPPING: only $(free_gb) GB free"; exit 1
  fi
  echo "[scan] $tag  $(date +%H:%M:%S)"
  $PY -u $S/tp_findability.py -i "$ALL" -n "$NEV" --chunk 4 \
      --cache-dir $S/cache-scan -o "$OUT/$tag.json" "$@" > "$log" 2>&1 \
    || { echo "[scan] FAILED $tag (see $log)"; return; }
  echo "[scan]   done, $(grep -c 'chunk ' "$log") chunks, $(free_gb) GB free"
}

echo "=== phase 1: projection-window allowance ==="
for d0 in 0.0 0.05 0.1 0.2; do
  run "win_${d0}" --d0-window-cm "$d0"
done

echo "=== phase 2: alpha/beta bit grid, diagonal first ==="
for b in 2 3 4 5 6 7; do
  run "bits_a${b}_b${b}" --alpha-bits "$b" --beta-bits "$b"
done
echo "=== phase 2b: the rest of the 6x6 grid ==="
for a in 2 3 4 5 6 7; do
  for b in 2 3 4 5 6 7; do
    [ "$a" = "$b" ] && continue
    run "bits_a${a}_b${b}" --alpha-bits "$a" --beta-bits "$b"
  done
done
echo "[scan] all done $(date +%H:%M:%S)"
