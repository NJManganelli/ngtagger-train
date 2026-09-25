#!/bin/bash
# Disc CNNs on the barrel+disc dataset (data_all/ -> WDMac cluster_cnn_data_all):
# a separate disc model, and one unified barrel+disc model (half the sample from each).
cd "$(dirname "$0")"
PY=../../.pixi/envs/default/bin/python
for args in "--variant disc --subdet 2" "--variant unified --subdet 0"; do
  echo "=== $args $(date)"
  $PY train.py $args --data data_all --epochs 8 --ntrain 3000000 --threads 4
done
echo "=== evaluate $(date)"
$PY evaluate_disc.py
echo "=== done $(date)"
