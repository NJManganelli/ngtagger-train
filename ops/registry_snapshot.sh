#!/bin/bash
# Snapshot of the results registry to WDMac, run by ops/spheavy after every heavy job (the registry
# changes only when jobs register results). One dated directory per day on WDMac
# (sp_registry_backup/YYYY-MM-DD), unchanged files hard-linked to the previous day's (rsync
# --link-dest). Does nothing when the registry is unchanged since the last snapshot or WDMac is not
# mounted. Never fails the job that triggered it.
set -u
SRC=/Users/nmangane/smartpixels/sp_registry/
DST=/Volumes/WDMac/smartpixels_scratch/sp_registry_backup
LOG=/Users/nmangane/smartpixels/heavy_lane/registry_snapshot.log
stamp() { date -u +%FT%TZ; }
if [ ! -d /Volumes/WDMac/smartpixels_scratch ]; then echo "$(stamp) SKIP WDMac not mounted" >> "$LOG"; exit 0; fi
mkdir -p "$DST"
if [ -f "$DST/.last" ] && [ -z "$(find "$SRC" -newer "$DST/.last" -print -quit 2>/dev/null)" ]; then exit 0; fi
today=$(date -u +%F)
prev=$(ls -1d "$DST"/20??-??-?? 2>/dev/null | grep -v "/$today$" | tail -1)
link=(); [ -n "$prev" ] && link=(--link-dest="$prev")
rsync -a --delete ${link[@]+"${link[@]}"} "$SRC" "$DST/$today/"; rc=$?
if [ $rc -eq 0 ]; then
  touch "$DST/.last"
  echo "$(stamp) OK $DST/$today (link-dest ${prev:-none}) $(du -sh "$DST/$today" | cut -f1)${1:+ after $1}" >> "$LOG"
else
  echo "$(stamp) FAILED rc=$rc${1:+ after $1}" >> "$LOG"
fi
exit 0
