"""Panel 2 exporter: per-track Stage-3 refit-quality BDT scores + conditioning
variables from the FAT COHERENT nanos, stored as one float32 row matrix the
browser bins on the fly.

Data mapping (user's choice):
  nano_fat_1111_coopt_file{1..10}.root -> model/tables refitq_AAAA
  nano_fat_1100_coopt_file{1..10}.root -> model/tables refitq_AAII
  nano_fat_0000_baseline (no refit tables) -> excluded, noted in meta.

Scoring REUSES the repo's bit-faithful conifer walker + the exact Stage-3
feature assembly (ngtagger.train.refitquality.build_spec_dataset with the same
5-par prompt-L1TTrack framing as eval_refitq/stage3/train_stage3.py); rows are
restricted to refit-performed tracks (the producer never scores passthrough).

The export is dataset-agnostic: add_view() just appends another (files, config)
pair, so a future nano_pG export can add all 15 configs without touching the JS.
"""

from __future__ import annotations

import json
import os

import numpy as np

NANO_DIR = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano"
MODELS_DIR = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage3/models"

# view (activeSP bits) -> (file glob, refit config suffix)
DEFAULT_VIEWS = {
    "1111": ("nano_fat_1111_coopt_file{i}.root", "AAAA"),
    "1100": ("nano_fat_1100_coopt_file{i}.root", "AAII"),
}

COLUMNS = ["score", "label", "pt", "abs_eta", "phi", "z0", "abs_d0",
           "nstub", "chi2rphi_bin", "chi2rz_bin"]

# per-column default browser binning: {lo, hi, n, log}
X_DEFAULTS = {
    "pt": {"lo": 2.0, "hi": 100.0, "n": 12, "log": True},
    "abs_eta": {"lo": 0.0, "hi": 2.4, "n": 12, "log": False},
    "phi": {"lo": -3.1416, "hi": 3.1416, "n": 12, "log": False},
    "z0": {"lo": -15.0, "hi": 15.0, "n": 12, "log": False},
    "abs_d0": {"lo": 0.0, "hi": 0.5, "n": 10, "log": False},
    "nstub": {"lo": 3.5, "hi": 7.5, "n": 4, "log": False},
    "chi2rphi_bin": {"lo": -0.5, "hi": 15.5, "n": 16, "log": False},
    "chi2rz_bin": {"lo": -0.5, "hi": 15.5, "n": 16, "log": False},
    "score": {"lo": 0.0, "hi": 1.0, "n": 20, "log": False},
}


def tkq_rows(ref, var, hits, config: str, conifer_json: str,
             label: str = "genuine") -> np.ndarray:
    """(ref, var, hits) awkward tables for one config -> float32 (n, ncol)
    matrix in COLUMNS order, refit-performed rows only.  Split out from the
    file driver so tests can feed tiny synthetic tables."""
    import awkward as ak

    from ngtagger.train.refitquality import build_spec_dataset, conifer_json_walk

    X, y, _names, aux = build_spec_dataset(
        ref, var, hits, config, label=label,
        refit_only=True, require_truth=False,
        seed_npar=5, track_npar=5, spec_version=1)
    with open(conifer_json) as f:
        model = json.load(f)
    margin = conifer_json_walk(model, X).astype(np.float64)
    score = 1.0 / (1.0 + np.exp(-margin))

    # build_spec_dataset applied the refit mask to X/y; aux["refit_mask"] is
    # the full-length mask, so apply it to the raw conditioning columns too.
    refit_mask = aux["refit_mask"].astype(bool)
    cols = {}
    for b in ("pt", "eta", "phi", "z0", "d0", "nStubs",
              "hwChi2RPhi", "hwChi2RZ"):
        v = ak.to_numpy(ak.flatten(ref[b])).astype(np.float64)
        if len(v) != len(refit_mask):
            raise RuntimeError("ref column does not align with the refit mask")
        cols[b] = v[refit_mask]
    if len(cols["pt"]) != len(y):
        raise RuntimeError("refit-masked columns do not align with the labels")
    out = np.stack([
        score,
        y.astype(np.float64),
        cols["pt"],
        np.abs(cols["eta"]),
        cols["phi"],
        cols["z0"],
        np.abs(cols["d0"]),
        cols["nStubs"],
        cols["hwChi2RPhi"],
        cols["hwChi2RZ"],
    ], axis=1).astype("<f4")
    assert out.shape[1] == len(COLUMNS)
    return out


def export_tkquality(out_dir: str, nano_dir: str = NANO_DIR,
                     models_dir: str = MODELS_DIR, n_files: int = 10,
                     views: dict | None = None,
                     max_events: int | None = None) -> dict:
    """Score every view and write tkq.bin + tkq_meta.json into out_dir."""
    from ngtagger.train.refitquality import load_refit_tables

    views = views or DEFAULT_VIEWS
    groups = []
    mats = []
    row_offset = 0
    for view, (pattern, cfg) in views.items():
        files = [os.path.join(nano_dir, pattern.format(i=i))
                 for i in range(1, n_files + 1)]
        files = [f for f in files if os.path.exists(f)]
        if not files:
            raise FileNotFoundError(f"no nano files for view {view} in {nano_dir}")
        conifer_json = os.path.join(models_dir, f"refitq_{cfg}_conifer.json")
        ref, var, hits = load_refit_tables(files, cfg, "L1TTrack", max_events)
        rows = tkq_rows(ref, var, hits, cfg, conifer_json)
        mats.append(rows)
        groups.append({
            "id": view, "view": view, "refit_config": cfg,
            "model": os.path.basename(conifer_json),
            "n_files": len(files),
            "row_offset": row_offset, "n_rows": int(len(rows)),
        })
        row_offset += len(rows)

    all_rows = np.concatenate(mats, axis=0)
    meta = {
        "id": "tkq", "type": "table", "file": "tkq.bin",
        "title": "Stage-3 refit-quality BDT (tkquality)",
        "columns": COLUMNS,
        "score_columns": ["score"],
        "label_column": "label",
        "class_names": ["genuine"],
        "groups": groups,
        "x_defaults": X_DEFAULTS,
        "notes": [
            "score = sigmoid(raw conifer logit margin), REFIT_BDT_FEATURES v1 "
            "(24), 5-par prompt L1TTrack framing, refit-performed tracks only.",
            "0000 baseline nanos carry no refit tables -> no tkquality entry "
            "for 0000 (nothing to score).",
            "dataset-agnostic export: future nano_pG exports can append all 15 "
            "configs as extra groups without JS changes.",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    all_rows.tofile(os.path.join(out_dir, "tkq.bin"))
    with open(os.path.join(out_dir, "tkq_meta.json"), "w") as f:
        json.dump(meta, f)
    return meta
