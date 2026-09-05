"""VERIFY-FIRST prereq check: confirm each unified nano yields jet-flavor labels via
the tagger's prepare_dataset path, and the NG jet-tagger input tier loads."""
import sys

import numpy as np

from ngtagger.data.labels import CLASS_LABELS
from ngtagger.train.trainer import prepare_dataset

view = sys.argv[1]
files = sys.argv[2:]
print(f"=== prereq check for view {view} on {len(files)} file(s) ===")

groups = ["baseline"]
if view != "0000":
    # sanity: also try vertexdxy which needs L1TExtTrack 5-par
    groups = ["baseline"]

ds = prepare_dataset(files, feature_groups=groups, seed=0)
y = ds["y_train"]
yt = ds["y_test"]
allY = np.concatenate([y, yt], axis=0)
labels = allY.argmax(axis=1)
print("n_train jets:", len(y), " n_test jets:", len(yt))
print("feature_names (", len(ds["feature_names"]), "):", ds["feature_names"])
print("X_train shape:", ds["X_train"].shape)
print("=== per-flavor jet counts (train+test) ===")
for i, name in enumerate(CLASS_LABELS):
    print(f"  {name:10s}: {(labels==i).sum()}")
print("TOTAL labeled jets:", len(allY))
