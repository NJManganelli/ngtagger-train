#!/bin/zsh
# FAST class-C study: sequential training queue (ONE job at a time; resource checks before each). Logs on WDMac emu_fast/logs.
cd "$(dirname "$0")"
PY=~/smartpixels/ngtagger-train/.pixi/envs/default/bin/python
L=/Volumes/WDMac/smartpixels_scratch/emu_fast/logs
export OMP_NUM_THREADS=4
if (( $# )); then Q=("$@"); else
Q=("barrel full" "barrel narrow" "barrel dws" "barrel tiny" "barrel proj1d" "barrel narrow 16 12"
   "disc full" "disc narrow" "disc dws" "disc tiny" "disc proj1d"); fi
for job in $Q; do
  set -- ${=job}
  df -g / | awk 'NR==2 && $4 < 10 {print "DISK LOW"; exit 1}' || exit 1
  sw=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d M); echo "swap used ${sw}M"
  ex=""; [[ -n $3 ]] && ex="--wx $3 --wy $4"
  n="$1_$2${3:+_w$3x$4}"
  echo "=== $n $(date)"
  $PY fast_train.py --region $1 --arch $2 ${=ex} > $L/train_$n.log 2>&1
  grep '"epoch"' $L/train_$n.log | tail -1; tail -1 $L/train_$n.log
done
echo "=== queue done $(date)"
