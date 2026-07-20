"""Engineered jet-charge Q_kappa baseline (model-space study B.2.3): the
must-beat benchmark for the learned charge head."""

from __future__ import annotations

import awkward as ak
import numpy as np


def jet_charge_kappa(charge, pt, kappa: float = 0.5, norm: str = "sum_pow"):
    """Per-jet engineered charge from constituent charges and pts.

        norm="sum_pow":  Q = sum_i q_i pt_i^kappa / (sum_i pt_i)^kappa
                         (the study's B.2.3 formula -- default)
        norm="pow_sum":  Q = sum_i q_i pt_i^kappa /  sum_i pt_i^kappa
                         (classical Field-Feynman variant)

    Accepts numpy (n_jets, n_const) zero-padded tensors or awkward jagged
    (…, constituent) arrays; padding entries (q = pt = 0) contribute nothing.
    Jets with no constituents (sum pt = 0) get Q = 0.
    """
    xp = ak if isinstance(charge, ak.Array) or isinstance(pt, ak.Array) else np
    num = xp.sum(charge * pt**kappa, axis=-1)
    if norm == "sum_pow":
        den = xp.sum(pt, axis=-1) ** kappa
    elif norm == "pow_sum":
        den = xp.sum(pt**kappa, axis=-1)
    else:
        raise ValueError(f"unknown norm '{norm}'; use 'sum_pow' or 'pow_sum'")
    return xp.where(den > 0, num / xp.where(den > 0, den, 1.0), 0.0)


def jet_charge_from_features(X: np.ndarray, feature_names: list[str],
                             kappa: float = 0.5, norm: str = "sum_pow"):
    """Q_kappa from the padded training tensor (N, n_const, n_feat).

    The constituent charge is reconstructed from the baseline one-hot charge
    flags (charge is known only for tracked constituents; neutrals are 0 by
    construction), so the baseline consumes exactly the columns the model
    sees."""
    idx = {n: feature_names.index(n) for n in feature_names}
    plus = ["isElectronPlus", "isMuonPlus", "isChargedHadronPlus"]
    minus = ["isElectronMinus", "isMuonMinus", "isChargedHadronMinus"]
    missing = [n for n in plus + minus + ["pt"] if n not in idx]
    if missing:
        raise ValueError(f"feature_names missing baseline charge columns: {missing}")
    q = sum(X[..., idx[n]] for n in plus) - sum(X[..., idx[n]] for n in minus)
    return jet_charge_kappa(q, X[..., idx["pt"]], kappa=kappa, norm=norm)


def evaluate_charge_baseline(q_kappa, charge_class) -> dict:
    """Benchmark numbers for Q_kappa against the 3-class truth
    (0: q-, 1: neutral, 2: q+): per-class mean/count and the ROC AUC of
    Q_kappa separating q+ from q- (the headline the learned head must beat)."""
    q_kappa = np.asarray(ak.to_numpy(q_kappa) if isinstance(q_kappa, ak.Array) else q_kappa)
    charge_class = np.asarray(charge_class)
    out = {
        "n_per_class": np.bincount(charge_class, minlength=3).tolist(),
        "mean_q_per_class": [
            float(q_kappa[charge_class == c].mean()) if (charge_class == c).any() else float("nan")
            for c in range(3)
        ],
    }
    sel = charge_class != 1
    if (charge_class[sel] == 2).any() and (charge_class[sel] == 0).any():
        from sklearn.metrics import roc_auc_score

        out["auc_pm"] = float(roc_auc_score(charge_class[sel] == 2, q_kappa[sel]))
    else:
        out["auc_pm"] = float("nan")
    return out
