"""Step 3b: which non-angle COMBINATIONS carry the separation? Small HGB fits on feature subsets.

Hypothesis tested: the non-angle HGB mostly learns a CLUSTER-SHAPE vs ORIGIN-POINTING consistency
(sizeY vs |z|/r, sizeX vs r-phi incidence), i.e. the classic pixel cluster-shape filter, rather
than any single raw feature. Same event-level split as s3_separability (same RNG seed).
Writes results/sep_subsets.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import _common as c
from s3_separability import prepare

SUBSETS = {
    "sizeY": ["sizeY"],
    "sizeY + |z| + r (z-shape vs origin)": ["sizeY", "absZ", "globalR"],
    "sizeX": ["sizeX"],
    "sizeX + localX + r + layer (rphi-shape vs origin)": ["sizeX", "localX", "globalR", "layer"],
    "shape only: size,sizeX,sizeY,charge,qPerPix,sigX,sigY,layer": ["size", "sizeX", "sizeY", "charge", "qPerPix", "sigX", "sigY", "layer"],
    "position only: |z|, r, localX, localY, layer": ["absZ", "globalR", "localX", "localY", "layer"],
    "sizeX,sizeY,|z|,r,localX,layer": ["sizeX", "sizeY", "absZ", "globalR", "localX", "layer"],
    "all non-angle (as s3)": ["layer", "size", "sizeX", "sizeY", "charge", "qPerPix", "absZ", "localX", "localY", "globalR", "sigX", "sigY"],
}


def main():
    d = prepare(c.load())
    tr = d[~d["is_test"]].sample(1_000_000, random_state=1)
    te = d[d["is_test"]].sample(1_000_000, random_state=2)
    rows = []
    for name, f in SUBSETS.items():
        clf = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=63, random_state=0).fit(tr[f], tr["unlinked"])
        s = clf.predict_proba(te[f])[:, 1]
        row = {"features": name, "AUC all": roc_auc_score(te["unlinked"], s)}
        for L in (1, 2, 3, 4):
            m = (te["layer"] == L).to_numpy()
            row[f"AUC L{L}"] = roc_auc_score(te["unlinked"][m], s[m])
        rows.append(row)
        print(row, flush=True)
    t = pd.DataFrame(rows).set_index("features")
    (c.HERE / "results" / "sep_subsets.md").write_text(
        "HGB on feature subsets (1M train / 1M test clusters, event-level split)\n\n" + c.md(t, "{:.3f}"))


if __name__ == "__main__":
    main()
