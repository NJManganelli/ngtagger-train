#!/bin/bash
# drive_v2.sh (cmssw/work/canon_v2, sha256 ce9619eb...) run INSIDE the machine-wide heavy lane. Changes
# from drive_v2.sh, nothing else: (1) it refuses to start unless the heavy-lane launcher started it
# (SP_JOB_ID set), so the lock is held for the whole production, every legacy job is waited for first
# and no host heavy job can slip into the gaps between files; (2) "$mon" is quoted, so a monitor file
# name with spaces cannot leave the EXIT trap waiting forever on a monitor that never sees its stop file.
#   ~/smartpixels/ngtagger-train/ops/spheavy run --kind cmssw --name canon_v3-production --mem-min-free 15 \
#       --disk-path /Volumes/WDMac -- \
#       bash ~/smartpixels/ngtagger-train/ops/cmssw/drive_v2_locked.sh canon_v3:AAAI_f01 canon_v3:AAAI_f02 ...
# (each <dir>:<tag> a separate argument; the launcher wants >= 10 GB free on both disks and >= 15 % free
# memory before it starts, then this script's own guards apply per file). Original header:
# Host-side sequential driver over several work dirs: one cmsRun at a time (4 threads), resource monitor
# every 30 s, disk guards (internal < 11 GB -> kill cmsRun, abort; WDMac < 20 GB -> abort), finished ROOT
# files moved to WDMac.   usage: bash drive3.sh <dir>:<tag> ...
if [ -z "$SP_JOB_ID" ]; then
  echo "drive_v2_locked.sh: start me through the heavy-lane launcher (ops/spheavy run --kind cmssw ... -- bash $0 ...)" >&2
  exit 9
fi
#   canon_v2:IAAA_f02    -> /work/canon_v2/spstudy_IAAA_f02_cfg.py, logs IAAA_f02.*, spstudy_*_100ev*.root -> WDMac canon_v2
#   coexist_v2:coexist15_f01_100ev -> /work/coexist_v2/coexist15_f01_100ev_cfg.py, logs <tag>.*, coexist15_*_100ev.root -> WDMac coexist_v2
#   coexist_v2:smoke3    -> as above, output stays in the work dir
H=/Users/nmangane/smartpixels/cmssw/work
C=cmssw-el9-native-1
MINGB=11
WDMINGB=20
free_gb() { df -g / | awk 'NR==2{print $4}'; }
wd_free_gb() { df -g /Volumes/WDMac | awk 'NR==2{print $4}'; }
LOG=$H/canon_v2/drive.log
# host memory guard (a concurrent host study may be running): pause between files if swap grew > 2 GB or free memory < 15%
SWAPGROWMB=2048
MEMMINPCT=15
swap_mb() { sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); printf "%d", v}'; }
mem_free_pct() { memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }
SWAP0=$(swap_mb)
# source-tree guard: production must run on an unchanged arm checkout
SRC=/Users/nmangane/smartpixels/cmssw/work/arm/CMSSW_20_1_0_pre1/src
src_sig() { echo "$(git -C $SRC rev-parse HEAD 2>/dev/null) $(git -C $SRC status --porcelain 2>/dev/null | grep -v '^?? L1Trigger/TrackFindingTracklet/data/spixel_smear_all_configs_barrel_CalV1_v2p1_compound.json$' | md5)"; }
SIG0=$(src_sig)
# optional pin: EXPECT_HEAD=<sha prefix> refuses to start on any other commit
if [ -n "$EXPECT_HEAD" ] && [ "${SIG0#$EXPECT_HEAD}" = "$SIG0" ]; then echo "ABORT HEAD ${SIG0%% *} != expected $EXPECT_HEAD" | tee -a $LOG; exit 8; fi
echo "SRC $SIG0 $(date -u +%FT%T) lane-job $SP_JOB_ID" >> $LOG
# The Docker VM keeps virtiofs read handles on files it has read: a plain unlink leaves the space allocated
# until Docker restarts. Copy, verify the size, truncate (frees the blocks behind the held handle), unlink.
mvfile() {
  local f=$1 d=$2/$(basename $1)
  cp "$f" "$d" && [ "$(stat -f %z "$f")" = "$(stat -f %z "$d")" ] && : > "$f" && rm -f "$f" && echo "MOVED $(basename $f) -> $2" >> $LOG \
    || { echo "ABORT move failed $(basename $f)" | tee -a $LOG; exit 6; }
}
first=${1#*:}; last=${@: -1}; last=${last#*:}
mon=$H/canon_v2/monitor_${first}_to_${last}.csv
rm -f "$mon.stop"
bash $H/gate200/monitor.sh "$mon" 30 &
MONPID=$!
caffeinate -i -w $$ &
trap 'touch "$mon.stop"; wait $MONPID 2>/dev/null' EXIT
for item in "$@"; do
  d=${item%%:*}; tag=${item#*:}; W=$H/$d
  case $d in
    canon_v2) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_100ev*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v2 ;;
    canon_v3) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_100ev*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v3 ;;
    canon_v4) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_100ev*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v4 ;;
    canon_v4_closure) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*ev_*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v4/closure ;;
    canon_v4_test) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_10ev*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v4_test ;;
    canon_v2_htt) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_htt_f*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v2_htt ;;
    canon_v2_minbias) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_minbias_f*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v2_minbias ;;
    canon_v2_t1tttt) cfg=spstudy_${tag}_cfg.py; pat="spstudy_*_t1tttt_f*.root"; DEST=/Volumes/WDMac/smartpixels_scratch/canon_v2_t1tttt ;;
    coexist_v2) cfg=${tag}_cfg.py; pat="coexist15_*_100ev.root"; DEST=/Volumes/WDMac/smartpixels_scratch/coexist_v2 ;;
    *) echo "ABORT unknown dir $d" | tee -a $LOG; exit 2 ;;
  esac
  if [ "$(wd_free_gb)" -lt $WDMINGB ]; then echo "ABORT WDMac $(wd_free_gb) GB before $tag" | tee -a $LOG; exit 3; fi
  if [ $(( $(swap_mb) - SWAP0 )) -gt $SWAPGROWMB ] || [ "$(mem_free_pct)" -lt $MEMMINPCT ]; then
    echo "PAUSE memory before $tag: swap $(swap_mb) MB (start $SWAP0), free $(mem_free_pct)%" | tee -a $LOG; exit 7; fi
  sig=$(src_sig); if [ "$sig" != "$SIG0" ]; then echo "ABORT source tree changed before $tag: $sig (start $SIG0)" | tee -a $LOG; exit 8; fi
  if [ "$(free_gb)" -lt $MINGB ]; then echo "ABORT disk $(free_gb) GB before $tag" | tee -a $LOG; exit 3; fi
  if [ -n "$(docker exec $C ps -C cmsRun -o pid=)" ]; then echo "ABORT another cmsRun is running before $tag" | tee -a $LOG; exit 4; fi
  for attempt in 1 2; do
    echo "START $d:$tag $(date -u +%FT%T) disk=$(free_gb)GB" >> $LOG
    docker exec $C bash /work/spx_arm_env.sh bash -c "cd /work/$d; s=\$(date +%s); echo RUN-START \$(date -u +%FT%T) > $tag.run.log; /usr/bin/time -v cmsRun /work/$d/$cfg >> $tag.run.log 2> $tag.err.log; rc=\$?; echo RUN-EXIT rc=\$rc wall=\$(( \$(date +%s) - s ))s end=\$(date -u +%FT%T) >> $tag.run.log" > $W/$tag.exec.log 2>&1 &
    JOB=$!
    while kill -0 $JOB 2>/dev/null; do
      if [ "$(free_gb)" -lt $MINGB ]; then
        echo "ABORT disk $(free_gb) GB during $tag; killing cmsRun" | tee -a $LOG
        docker exec $C pkill -TERM -x cmsRun; wait $JOB; exit 3
      fi
      sleep 30
    done
    wait $JOB
    echo "END $d:$tag $(date -u +%FT%T) $(grep RUN-EXIT $W/$tag.run.log)" >> $LOG
    # one retry for a transient Frontier DNS failure at startup (seen twice on 2026-09-30)
    if [ $attempt = 1 ] && ! grep -q "RUN-EXIT rc=0" $W/$tag.run.log && grep -q "No more servers/proxies" $W/$tag.err.log \
       && grep -q "Name or service not known" $W/$tag.err.log; then
      echo "RETRY $tag after Frontier DNS failure; waiting 300 s" >> $LOG; sleep 300
      echo "RETRY $tag getent: $(docker exec $C getent hosts cmsfrontier.cern.ch | head -1)" >> $LOG; continue
    fi
    break
  done
  for f in $(ls $W/$pat 2>/dev/null); do mvfile "$f" $DEST; done
  grep -q "RUN-EXIT rc=0" $W/$tag.run.log || { echo "ABORT $tag nonzero rc" | tee -a $LOG; exit 5; }
done
echo "ALL-DONE $(date -u +%FT%T)" >> $LOG
