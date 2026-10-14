#!/bin/bash
# Nightly snapshot of the results registry to WDMac (plan C.3-5 TODO): one dated directory per night,
# unchanged files hard-linked to the previous snapshot (rsync --link-dest), so each night costs only
# what changed. Skips cleanly when WDMac is not mounted.
set -u
SRC=/Users/nmangane/smartpixels/sp_registry/
DST=/Volumes/WDMac/smartpixels_scratch/sp_registry_backup
LOG=/Users/nmangane/smartpixels/heavy_lane/nightly.log
stamp() { date -u +%FT%TZ; }
if [ ! -d /Volumes/WDMac/smartpixels_scratch ]; then echo "$(stamp) registry_backup SKIP WDMac not mounted" >> "$LOG"; exit 0; fi
mkdir -p "$DST"
today=$(date -u +%F)
prev=$(ls -1d "$DST"/20??-??-?? 2>/dev/null | grep -v "/$today$" | tail -1)
link=(); [ -n "$prev" ] && link=(--link-dest="$prev")
rsync -a --delete ${link[@]+"${link[@]}"} "$SRC" "$DST/$today/"; rc=$?
if [ $rc -eq 0 ]; then
  echo "$(stamp) registry_backup OK $DST/$today (link-dest ${prev:-none}) $(du -sh "$DST/$today" | cut -f1)" >> "$LOG"
else
  echo "$(stamp) registry_backup FAILED rc=$rc (under launchd: grant /bin/bash access to removable volumes or Full Disk Access)" >> "$LOG"; exit 1
fi
