"""Shared data prep for the model-space study (Part A).

Builds, aligned to the SPEC-ORDER 17-feature matrix rows (refit tracks only):
  - the spec17 baseline (exact producer contract, via build_spec_dataset)
  - per-layer feature blocks from the per-hit link table (exactly one crossing
    per (track, layer) in this production, verified by probe_hits.py)
  - guard-clipped chi2 variants (the PU nano is PRE-guard: chi2 tails up to
    3.3e9 are numerical-Jacobian pathology; physical cap ~2e6 per spec §6b)

All studies here share the SAME split seeds so per-seed PAIRED AUC deltas can
be quoted (the split sigma ~0.011 dwarfs any single-cell difference at these
stats; pairing removes it).
"""
from __future__ import annotations

import numpy as np
import awkward as ak

from ngtagger.train.refitquality import (
    _HIT_COLS,
    _SENTINEL,
    build_spec_dataset,
    load_refit_tables,
)

NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_pu100_TrkSmartPix_withGen.root"
CHI2_GUARD = 2.0e6  # spec §6b: >2e6 is numerical pathology on this pre-guard nano
LAYERS = (1, 2, 3, 4)
SEEDS = tuple(range(8))
TEST_FRACTION = 0.2

# per-hit columns exposed per layer (missing crossing/hit -> np.nan)
PERLAYER_COLS = ["windowMult", "resX", "resY", "pullX", "pullY",
                 "pullAlpha", "pullBeta", "chi2IncRPhi", "chi2IncRZ"]


def log1p_guard(x: np.ndarray) -> np.ndarray:
    """log1p with the §6b physical cap; NaN passes through (xgb missing)."""
    return np.log1p(np.clip(x, 0.0, CHI2_GUARD))


def split(n: int, seed: int, test_fraction: float = TEST_FRACTION):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_fraction)
    return idx[n_test:], idx[:n_test]


def load_dataset(config: str = "AAAA", files: list[str] | None = None):
    """Returns dict with spec17 X/names, y, per-layer blocks, aux."""
    files = files or [NANO]
    ref, var, hits = load_refit_tables(files, config)

    X, y, names, aux = build_spec_dataset(ref, var, hits, config, refit_only=True)

    # --- global hit index (same construction as build_spec_dataset) ---
    counts = ak.to_numpy(ak.num(ref["genuine"]))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_tracks = int(offsets[-1])
    h = {b: ak.to_numpy(ak.flatten(hits[b])) for b in _HIT_COLS}
    ev_of_hit = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits["trackIdx"])))
    gidx = h["trackIdx"].astype(np.int64) + offsets[ev_of_hit]

    # --- per-layer scatter: one crossing per (track, layer) [verified] ---
    per_layer = {}
    accepted = h["pullX"] > _SENTINEL  # accepted == KF position update applied
    for L in LAYERS:
        mL = h["layer"] == L
        rows = gidx[mL]
        blk = {}
        blk["crossed"] = np.zeros(n_tracks, np.float32)
        blk["crossed"][rows] = 1.0
        blk["accepted"] = np.zeros(n_tracks, np.float32)
        blk["accepted"][rows[accepted[mL]]] = 1.0
        for col in PERLAYER_COLS:
            v = np.full(n_tracks, np.nan, np.float32)
            vals = h[col][mL].astype(np.float32)
            ok = vals > _SENTINEL
            v[rows[ok]] = vals[ok]
            if col.startswith("chi2"):
                v = log1p_guard(v)
            blk[col] = v
        per_layer[L] = blk

    refit_mask = aux["refit_mask"]
    for L in LAYERS:
        for k in per_layer[L]:
            per_layer[L][k] = per_layer[L][k][refit_mask]

    return {
        "X_spec": X, "y": y, "spec_names": list(names), "aux": aux,
        "per_layer": per_layer, "config": config,
    }


def spec_guarded(X_spec: np.ndarray, spec_names: list[str]) -> np.ndarray:
    """spec17 with the two chi2 totals log1p-guard-clipped (features 9, 10)."""
    Xg = X_spec.copy()
    for f in ("chi2IncRPhiTot", "chi2IncRZTot"):
        i = spec_names.index(f)
        Xg[:, i] = log1p_guard(Xg[:, i].astype(np.float64)).astype(np.float32)
    return Xg


def perlayer_matrix(per_layer: dict, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Stack the requested per-layer columns into (N, 4*len(cols))."""
    mats, names = [], []
    for L in LAYERS:
        for c in cols:
            mats.append(per_layer[L][c])
            names.append(f"{c}_L{L}")
    return np.stack(mats, axis=1).astype(np.float32), names


def paired_auc_deltas(aucs_by_set: dict[str, list[float]], baseline: str) -> dict:
    """Per-seed paired deltas vs the baseline set (same splits)."""
    base = np.asarray(aucs_by_set[baseline])
    out = {}
    for k, v in aucs_by_set.items():
        d = np.asarray(v) - base
        out[k] = {
            "auc_mean": float(np.mean(v)), "auc_std": float(np.std(v)),
            "delta_mean": float(d.mean()), "delta_std": float(d.std()),
            "delta_min": float(d.min()), "delta_max": float(d.max()),
            "n_seeds_improved": int((d > 0).sum()),
        }
    return out
