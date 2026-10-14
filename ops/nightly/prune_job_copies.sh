#!/bin/bash
# Nightly prune of frozen code copies (plan C.3-4 TODO): for spheavy job directories older than 30 days,
# remove only the bulky frozen copy (copy/). The job's provenance stays: launch.json, manifest.json,
# tree.patch, untracked.tar.gz, exit.json and the guard log are kept, so the code can be rebuilt from
# the recorded commit + patch + untracked archive. A job whose lock is still held is never touched.
set -u
JOBS=/Users/nmangane/smartpixels/sp_jobs
LOG=/Users/nmangane/smartpixels/heavy_lane/nightly.log
DAYS=${PRUNE_DAYS:-30}
stamp() { date -u +%FT%TZ; }
held=$(/Users/nmangane/smartpixels/ngtagger-train/ops/spheavy status 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('holder') or {}).get('job_id',''))" 2>/dev/null)
n=0
while IFS= read -r d; do
  j=$(basename "$(dirname "$d")")
  [ "$j" = "$held" ] && continue
  [ -f "$(dirname "$d")/launch.json" ] || continue          # only real spheavy job dirs
  chmod -R u+w "$d" 2>/dev/null; rm -rf "$d" && n=$((n+1)) && echo "$(stamp) prune_job_copies removed $d" >> "$LOG"
done < <(find "$JOBS" -mindepth 2 -maxdepth 2 -type d -name copy -mtime +"$DAYS" 2>/dev/null)
echo "$(stamp) prune_job_copies done: $n copies older than $DAYS days removed" >> "$LOG"
