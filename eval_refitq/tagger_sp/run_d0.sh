#!/bin/zsh
# PV-(x, y) d0 UPPER-BOUND study (d0pv.py, d0_analyze.py): inputs, track-level validation, the
# variants A/B/C/D x L2/L3 x {AAAA, IIII, AAAI, IAAA} x 10 seeds with the "linout" recipe (the
# ladder's DeepSetHGQ2 without the ReLU on the class logits, which leaves dead, exactly uniform
# outputs), the same on AAAA with the ladder recipe (for the degeneracy comparison), a seeded-run
# reproducibility rerun, analysis. Needs the ladder's jet caches (run_all.sh). Sequential, CPU,
# 2 threads, ~3.5 h:
#   caffeinate -i eval_refitq/tagger_sp/run_d0.sh
#   RESUME=1 caffeinate -i eval_refitq/tagger_sp/run_d0.sh     (skip cells whose 10 predictions exist)
# Since the output-ReLU fix the framework model has linear logits, so "ladder" builds the same model as
# "linout" and the ReLU comparison cannot be regenerated; its stored predictions remain.
# Outputs: /Volumes/WDMac/smartpixels_scratch/tagger_sp/d0_upper_bound/ (d0_upper_bound.pdf, summary.json).
# One heavy job at a time on the shared host: every heavy step waits (60 s polls) while a census run
# (tp_findability.py, which has priority) or another tagger_sp training is active; waits -> logs/gate_wait.log.
set -e
cd "${0:A:h}/../.."
SEEDS=(0 1 2 3 4 5 6 7 8 9)
export OMP_NUM_THREADS=2
OUTD=/Volumes/WDMac/smartpixels_scratch/tagger_sp/d0_upper_bound
LOG=$OUTD/logs
mkdir -p $LOG

gate() {  # the bracketed patterns keep pgrep from matching its own command line
  local t0=$SECONDS
  while pgrep -f "[t]p_findability.py" > /dev/null || pgrep -f "[t]agger_sp/train.py" > /dev/null; do
    sleep 60
  done
  if (( SECONDS > t0 )); then
    echo "$(date +%FT%T) waited $(( SECONDS - t0 )) s before: $*" >> $LOG/gate_wait.log
  fi
}

gate inputs
pixi run python eval_refitq/tagger_sp/d0pv.py lut > $LOG/lut.log 2>&1
pixi run python eval_refitq/tagger_sp/d0pv.py side > $LOG/side.log 2>&1
pixi run python eval_refitq/tagger_sp/d0pv.py validate > $LOG/validate.log 2>&1

cell() {  # recipe mask variant d0
  local pdir=$OUTD/preds
  [[ $1 == linout ]] && pdir=$OUTD/linout/preds
  local done=($pdir/$2_$3_$4_s<0-9>.npz(N))
  if [[ -n $RESUME ]] && (( ${#done} == ${#SEEDS} )); then
    return
  fi
  gate train $*
  pixi run python eval_refitq/tagger_sp/train.py --recipe $1 --mask $2 --variant $3 --d0 $4 --seeds $SEEDS \
    > $LOG/train_$1_$2_$3_$4.log 2>&1
}
for m in AAAA IIII AAAI IAAA; do          # priority: AAAA and the OT controls first
  for v in track trackmva; do
    for x in A B C D; do cell linout $m $v $x; done
  done
done
for v in track trackmva; do
  for x in A B C D; do cell ladder AAAA $v $x; done
done
gate repro
pixi run python eval_refitq/tagger_sp/train.py --recipe linout --mask AAAA --variant track --d0 A --seeds 0 1 \
  --tag _repro > $LOG/train_repro.log 2>&1
gate analyze
pixi run python eval_refitq/tagger_sp/d0_analyze.py --boot 300 > $LOG/analyze.log 2>&1
