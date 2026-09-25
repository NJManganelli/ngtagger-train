#!/bin/bash
# Sequential training of all variants (models/ -> /Volumes/WDMac/smartpixels_scratch/cluster_cnn_models)
cd "$(dirname "$0")"
PY=../../.pixi/envs/default/bin/python
for args in "--variant base" "--variant small --lr 5e-3" "--variant bla" "--variant base --clean" "--variant pos"; do
  echo "=== $args $(date)"
  $PY train.py $args --epochs 8 --ntrain 3000000 --threads 4
done
echo "=== done $(date)"
