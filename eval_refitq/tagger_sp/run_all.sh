#!/bin/zsh
# Full study rerun (adding nano files = rerunning this). Sequential, CPU, 2 threads.
#   eval_refitq/tagger_sp/run_all.sh [samples...]   (default: ttbar htt)
# build.py retrains a mask's track-MVA heads only when its file list changed and rebuilds
# the jet caches only when the heads or the feature list changed; train.py always retrains.
set -e
cd "${0:A:h}/../.."
SAMPLES=(${@:-ttbar htt})
SEEDS=(0 1 2 3 4 5 6 7 8 9)
export OMP_NUM_THREADS=2
LOG=/Volumes/WDMac/smartpixels_scratch/tagger_sp/logs
mkdir -p $LOG
rm -f /Volumes/WDMac/smartpixels_scratch/tagger_sp/preds/*.npz
for m in IIII AAAA AAAI IAAA; do
  pixi run python eval_refitq/tagger_sp/build.py --mask $m --samples $SAMPLES > $LOG/build_$m.log 2>&1
  for v in baseline track trackmva; do
    pixi run python eval_refitq/tagger_sp/train.py --mask $m --variant $v --samples $SAMPLES \
      --seeds $SEEDS > $LOG/train_${m}_$v.log 2>&1
  done
done
pixi run python eval_refitq/tagger_sp/analyze.py --boot 300 > $LOG/analyze.log 2>&1
