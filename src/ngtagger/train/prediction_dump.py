"""Per-jet test-set prediction dumps (default-on for the trainer path).

Every training that goes through ngtagger.train.trainer.run_training — and the
Stage-4 matrix runner — persists an npz next to the model with, per test jet:
all class probabilities, charge-head probabilities where present, the true
labels, and cheap jet-level kinematics.  These dumps feed the MVA-explorer
jet-tagger panel (ngtagger.viz.mva_explorer.tagger_export) but are useful on
their own for any post-hoc study without re-running inference.
"""

from __future__ import annotations

import json
import os

import numpy as np


def dump_predictions(path: str, *, class_probs: np.ndarray,
                     class_labels: list[str], y_true: np.ndarray,
                     kinematics: dict | None = None,
                     charge_probs: np.ndarray | None = None,
                     charge_true: np.ndarray | None = None,
                     charge_labels: list[str] | None = None,
                     meta: dict | None = None) -> str:
    """Write one compressed npz prediction dump.

    class_probs: (n, n_classes) float; y_true: (n,) int class index or (n, n_classes)
    one-hot (converted); kinematics: {name: (n,) array} of cheap jet-level
    quantities (pt, abs_eta, phi, nconst, ...); meta: JSON-serializable dict
    recorded as a string array.
    """
    class_probs = np.asarray(class_probs, dtype=np.float32)
    y_true = np.asarray(y_true)
    if y_true.ndim == 2:  # one-hot -> index
        y_true = y_true.argmax(axis=1)
    y_true = y_true.astype(np.int16)
    n = len(class_probs)
    if len(y_true) != n:
        raise ValueError(f"y_true length {len(y_true)} != n {n}")

    payload = {
        "class_probs": class_probs,
        "y_true": y_true,
        "class_labels": np.asarray(class_labels, dtype=object),
    }
    for name, arr in (kinematics or {}).items():
        a = np.asarray(arr)
        if len(a) != n:
            raise ValueError(f"kinematics[{name!r}] length {len(a)} != n {n}")
        payload[f"kin_{name}"] = a.astype(np.float32)
    if charge_probs is not None:
        cp = np.asarray(charge_probs, dtype=np.float32)
        if len(cp) != n:
            raise ValueError("charge_probs length mismatch")
        payload["charge_probs"] = cp
        payload["charge_labels"] = np.asarray(
            charge_labels or [f"q{i}" for i in range(cp.shape[1])], dtype=object)
    if charge_true is not None:
        ct = np.asarray(charge_true)
        if ct.ndim == 2:
            ct = ct.argmax(axis=1)
        payload["charge_true"] = ct.astype(np.int16)
    payload["meta_json"] = np.asarray(json.dumps(meta or {}))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_predictions(path: str) -> dict:
    """Inverse of dump_predictions: npz -> plain dict (kinematics regrouped)."""
    with np.load(path, allow_pickle=True) as f:
        out = {"class_probs": f["class_probs"], "y_true": f["y_true"],
               "class_labels": [str(s) for s in f["class_labels"]],
               "kinematics": {k[len("kin_"):]: f[k] for k in f.files
                              if k.startswith("kin_")},
               "meta": json.loads(str(f["meta_json"]))}
        for k in ("charge_probs", "charge_true"):
            if k in f.files:
                out[k] = f[k]
        if "charge_labels" in f.files:
            out["charge_labels"] = [str(s) for s in f["charge_labels"]]
    return out


def dump_test_predictions(model, ds: dict, path: str, seed: int | None = None,
                          extra_meta: dict | None = None) -> str | None:
    """Trainer-path hook: predict on ds['X_test'] and dump (default-on in
    run_training).  Tolerates models whose predict returns [id, charge] lists
    and datasets that lack the optional kinematics keys."""
    X = ds.get("X_test")
    if X is None or len(X) == 0:
        return None
    prob = model.predict(X)
    charge_probs = None
    if isinstance(prob, (list, tuple)):
        prob, charge_probs = prob[0], (prob[1] if len(prob) > 1 else None)
    kin = {}
    for src, name in (("reco_pt_test", "pt"), ("reco_eta_test", "abs_eta"),
                      ("reco_phi_test", "phi"), ("nconst_test", "nconst"),
                      ("truth_pt_test", "truth_pt")):
        if src in ds:
            v = np.asarray(ds[src], dtype=np.float32)
            kin[name] = np.abs(v) if name == "abs_eta" else v
    return dump_predictions(
        path,
        class_probs=prob,
        class_labels=list(ds.get("class_labels", [])),
        y_true=ds["y_test"],
        kinematics=kin,
        charge_probs=charge_probs,
        charge_true=ds.get("charge_test"),
        charge_labels=list(ds.get("charge_class_labels", [])) or None,
        meta={"seed": seed, "n_test": int(len(X)), **(extra_meta or {})},
    )
