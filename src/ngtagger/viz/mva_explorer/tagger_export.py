"""Panel 3 exporter: Stage-4 jet-tagger per-jet prediction dumps -> one
float32 row matrix + meta for the browser.

Input: the npz dumps written by ngtagger.train.prediction_dump (one per
(cell, seed), produced by the Stage-4 matrix re-run).  Every (cell, seed)
becomes a row group of the shared table; columns are uniform across groups
(charge-prob columns are -1-filled for cells without a charge head, gated by
the per-group has_charge flag)."""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from ngtagger.data.labels import CHARGE_CLASS_LABELS, CLASS_LABELS

DUMPS_DIR = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage4/pred_dumps"

# canonical dropdown orders (user preference: refit views first, baseline last)
VIEW_ORDER = ["1111", "1100", "0000"]
VARIANT_ORDER = ["baseline", "refitbdt", "vertexdxy", "both", "both__chargehead"]
VARIANT_LABELS = {
    "baseline": "baseline", "refitbdt": "+refitBDT",
    "vertexdxy": "+vtxDxy", "both": "+both",
    "both__chargehead": "both+chargeHead",
}

KIN_COLUMNS = ["pt", "abs_eta", "phi", "nconst"]

X_DEFAULTS = {
    "pt": {"lo": 5.0, "hi": 250.0, "n": 10, "log": True},
    "abs_eta": {"lo": 0.0, "hi": 2.4, "n": 8, "log": False},
    "phi": {"lo": -3.1416, "hi": 3.1416, "n": 8, "log": False},
    "nconst": {"lo": -0.5, "hi": 16.5, "n": 17, "log": False},
    "score": {"lo": 0.0, "hi": 1.0, "n": 20, "log": False},
}

_DUMP_RE = re.compile(r"^(?P<cell>.+)__s(?P<seed>\d+)\.npz$")


def parse_cell_key(cell: str) -> tuple[str, str]:
    """'1111__both__chargehead' -> ('1111', 'both__chargehead')"""
    view, variant = cell.split("__", 1)
    return view, variant


def _sort_key(view: str, variant: str, seed: int):
    vi = VIEW_ORDER.index(view) if view in VIEW_ORDER else len(VIEW_ORDER)
    gi = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER)
    return (vi, gi, seed)


def export_tagger(out_dir: str, dumps_dir: str = DUMPS_DIR) -> dict:
    from ngtagger.train.prediction_dump import load_predictions

    files = sorted(glob.glob(os.path.join(dumps_dir, "*.npz")))
    entries = []
    for path in files:
        m = _DUMP_RE.match(os.path.basename(path))
        if not m:
            continue
        view, variant = parse_cell_key(m.group("cell"))
        entries.append((view, variant, int(m.group("seed")), m.group("cell"), path))
    if not entries:
        raise FileNotFoundError(f"no prediction dumps (*__s<seed>.npz) in {dumps_dir}")
    entries.sort(key=lambda e: _sort_key(e[0], e[1], e[2]))

    columns = ([f"prob_{c}" for c in CLASS_LABELS]
               + [f"prob_{c}" for c in CHARGE_CLASS_LABELS]
               + ["label", "charge_label"] + KIN_COLUMNS)
    n_id = len(CLASS_LABELS)
    n_q = len(CHARGE_CLASS_LABELS)

    groups, mats = [], []
    row_offset = 0
    for view, variant, seed, cell, path in entries:
        d = load_predictions(path)
        if d["class_labels"] and list(d["class_labels"]) != list(CLASS_LABELS):
            raise ValueError(f"{path}: class labels {d['class_labels']} != {CLASS_LABELS}")
        n = len(d["class_probs"])
        mat = np.full((n, len(columns)), -1.0, dtype="<f4")
        mat[:, :n_id] = d["class_probs"]
        has_charge = "charge_probs" in d
        if has_charge:
            mat[:, n_id:n_id + n_q] = d["charge_probs"]
        mat[:, n_id + n_q] = d["y_true"]
        if "charge_true" in d:
            mat[:, n_id + n_q + 1] = d["charge_true"]
        kin = d["kinematics"]
        for j, name in enumerate(KIN_COLUMNS):
            if name in kin:
                mat[:, n_id + n_q + 2 + j] = kin[name]
        mats.append(mat)
        groups.append({
            "id": f"{cell}__s{seed}", "cell": cell, "view": view,
            "variant": variant, "variant_label": VARIANT_LABELS.get(variant, variant),
            "seed": seed, "has_charge": bool(has_charge),
            "row_offset": row_offset, "n_rows": int(n),
            "meta": d.get("meta", {}),
        })
        row_offset += n

    all_rows = np.concatenate(mats, axis=0)
    meta = {
        "id": "tagger", "type": "table", "file": "tagger.bin",
        "title": "Stage-4 jet tagger (per-jet test predictions)",
        "columns": columns,
        "score_columns": [f"prob_{c}" for c in CLASS_LABELS],
        "charge_score_columns": [f"prob_{c}" for c in CHARGE_CLASS_LABELS],
        "label_column": "label",
        "charge_label_column": "charge_label",
        "class_names": list(CLASS_LABELS),
        "charge_class_names": list(CHARGE_CLASS_LABELS),
        "view_order": VIEW_ORDER,
        "variant_order": VARIANT_ORDER,
        "variant_labels": VARIANT_LABELS,
        "groups": groups,
        "x_defaults": X_DEFAULTS,
        "notes": [
            "one row group per (cell, seed); test split fixed by the dataset "
            "seed (12345) so all seeds of a cell share the same jets.",
            "charge-prob columns are -1-filled where the cell has no charge "
            "head (has_charge=false).",
            "~1300 test jets per view: per-bin counts are displayed because "
            "thin statistics are expected.",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    all_rows.tofile(os.path.join(out_dir, "tagger.bin"))
    with open(os.path.join(out_dir, "tagger_meta.json"), "w") as f:
        json.dump(meta, f)
    return meta
