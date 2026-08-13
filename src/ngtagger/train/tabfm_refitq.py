"""TabFM (tabular foundation model) baselines on the SmartPixels refit tables.

Two studies, both reusing the tier-study feature matrices from refitquality.py
so results are directly comparable to the xgboost tiers:

  binary     : genuine-vs-fake, the same target as the deployed TrackQuality
               GBDT -- a foundation-model ceiling estimate for the feature set.
  multiclass : track ORIGIN identification (particle species + combinatorial
               fakes). Historically the classic-7 BDT separates electrons from
               other origins well; this measures how far the refit features
               push per-species separation, and which species are mutually
               confusable (the confusion matrix is the deliverable).

TabFM predicts by in-context learning: the "training" set is shown to the
frozen model as context at inference time, so cost scales with context size,
not with epochs. Contexts are therefore SUBSAMPLED (class-balanced by default)
and the evaluation set is capped; both are explicit knobs, and every reported
number carries the sizes it was computed at.

Both studies are teacher candidates for a hardware BDT (student), so the
per-row predicted probabilities are persisted alongside the metrics.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ngtagger.train.refitquality import (
    _SENTINEL,
    _fake_rates,
    _split,
    build_refitq_dataset,
    load_refit_tables,
)

# ---------------------------------------------------------------------------
# Track-origin classes for the multiclass study.
#
# Composition of the PU200 TT reference sample (|pdgId| of the matched TP):
#   pion 63.9%, kaon 18.5%, proton 13.7%, electron 1.9%, fake 1.4%, muon 0.6%,
#   hyperons (3112/3222/3312) together ~0.05%.
#
# NOT a class: TAUS. A tau decays after ~87 um, well inside the beam pipe, so
# it never leaves a track -- its charged decay products (mostly pions) are what
# the tracker sees. Identifying "tau tracks" needs parent/decay-chain truth,
# which the nano truth columns do not carry (only the matched TP's own pdgId).
#
# Protons ARE included (13.7%): a large, physically distinct class that the
# hadron-vs-hadron question cannot ignore.
#
# Hyperons are dropped rather than lumped: at ~100 tracks per 190k they cannot
# be learned or measured, and folding them into "hadron" would blur the
# kaon/proton comparison this study is for.
PDG_CLASSES = {
    11: "electron",
    13: "muon",
    211: "pion",
    321: "kaon",
    2212: "proton",
}
FAKE_CLASS = "fake"
# Rare species deliberately excluded from the multiclass target (see above).
DROPPED_PDG = (3112, 3222, 3312, 3334, 15)


def build_origin_labels(ref_flat: dict, var_flat: dict | None = None):
    """Multiclass track-origin target from the reference-track truth columns.

    Returns (y int64, class_names list[str], keep bool-mask). Rows failing the
    mask carry no usable target: truth-'unknown' tracks (the associator had no
    answer) and the dropped rare species. Combinatoric tracks become the
    'fake' class; a genuine/looselyGenuine track becomes its species class.
    """
    pdg = np.abs(ref_flat["tpPdgId"].astype(np.int64))
    comb = ref_flat["combinatoric"].astype(bool)
    unknown = ref_flat["unknown"].astype(bool)
    loose = ref_flat["looselyGenuine"].astype(bool)

    names = [PDG_CLASSES[k] for k in sorted(PDG_CLASSES)] + [FAKE_CLASS]
    y = np.full(len(pdg), -1, np.int64)
    for i, code in enumerate(sorted(PDG_CLASSES)):
        y[loose & (pdg == code)] = i
    y[comb] = len(names) - 1  # fake

    keep = y >= 0
    keep &= ~unknown
    for code in DROPPED_PDG:
        keep &= ~(pdg == code)
    return y, names, keep


def _subsample(y: np.ndarray, n_max: int, seed: int, balanced: bool):
    """Row indices for a context/eval subsample. balanced=True draws equally
    from each class (capped by the rarest), which is what makes a few-thousand
    row TabFM context informative when fakes are ~1% of the sample."""
    rng = np.random.default_rng(seed)
    if not balanced:
        idx = rng.permutation(len(y))[:n_max]
        return np.sort(idx)
    classes = np.unique(y)
    per = max(1, n_max // len(classes))
    picks = []
    for c in classes:
        rows = np.flatnonzero(y == c)
        picks.append(rng.permutation(rows)[:per])
    return np.sort(np.concatenate(picks))


def resolve_device(device: str | None = None) -> str:
    """cuda > mps > cpu. MPS matters here: on Apple Silicon the CPU path
    saturates all cores for ~an hour per run, and the in-context forward pass
    is exactly the dense-matmul workload the GPU is for. TabFM loads in
    bfloat16; if a torch/MPS build cannot honor that the caller can force
    --device cpu."""
    if device is not None:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_tabfm(device: str | None = None, verbose: bool = False, model_type: str = "classification"):
    """Load a frozen TabFM checkpoint (pytorch backend)."""
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_torch

    device = resolve_device(device)
    if verbose:
        print(f"  loading TabFM {model_type} checkpoint on {device} ...")
    return tabfm_torch.load(model_type=model_type, device=device), device


# NOTE: do NOT try to work around TabFM prediction failures by predicting in
# chunks. Its preprocessing (quantile normalization / outlier clipping) is fit
# on the QUERY BATCH, so chunked predictions differ materially from a single
# call (measured: up to 0.5 in probability) -- and it does not fix the failure
# anyway. Reduce the query size or the feature count instead.


def _fit_predict(model, X_ctx, y_ctx, X_eval, n_estimators, seed, verbose):
    """sklearn-API fit/predict_proba on the frozen model, with a finiteness
    guard: TabFM can return an all-NaN block for some (context, ensemble)
    combinations from perfectly finite inputs, which otherwise only surfaces
    later as an opaque error inside a metric."""
    from tabfm import TabFMClassifier

    clf = TabFMClassifier(model=model, n_estimators=n_estimators,
                          random_state=seed, verbose=verbose)
    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_eval)
    bad = ~np.isfinite(proba)
    if bad.any():
        raise RuntimeError(
            f"TabFM returned {bad.sum()}/{bad.size} non-finite probabilities "
            f"(context={len(X_ctx)}, eval={len(X_eval)}, n_estimators={n_estimators}) "
            "from finite inputs. Retry with a smaller --max-context or different "
            "--n-estimators.")
    return clf, proba


def _metrics_binary(y_true, proba, names):
    from sklearn.metrics import roc_auc_score

    p1 = proba[:, 1]
    out = {"test_auc": float(roc_auc_score(y_true, p1)),
           "fake_rates": _fake_rates(y_true, p1)}
    return out


def _metrics_multiclass(y_true, proba, names):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    pred = np.argmax(proba, axis=1)
    present = np.unique(np.concatenate([y_true, pred]))
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(names)))
    # Row-normalized confusion = per-true-class prediction distribution.
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ovr = {}
    for i, nm in enumerate(names):
        yi = (y_true == i).astype(int)
        if yi.sum() and (yi == 0).sum():
            ovr[nm] = float(roc_auc_score(yi, proba[:, i]))
    # Class similarity: symmetrized off-diagonal confusion. High => the pair is
    # hard to tell apart with these features.
    sim = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim[f"{names[i]}|{names[j]}"] = float(0.5 * (cm_norm[i, j] + cm_norm[j, i]))
    return {"accuracy": float((pred == y_true).mean()),
            "balanced_accuracy": float(np.nanmean(np.diag(cm_norm)[np.isin(np.arange(len(names)), present)])),
            "per_class_auc_ovr": ovr,
            "confusion_counts": cm.tolist(),
            "confusion_rownorm": np.round(cm_norm, 4).tolist(),
            "pair_confusability": dict(sorted(sim.items(), key=lambda kv: -kv[1]))}


def train_refitq_tabfm(files: list[str], output_dir: str, config: str = "AAII",
                       track_table: str = "L1TTrack", crossref_track_table: str | None = None,
                       tier: str = "D", label: str = "looselyGenuine",
                       multiclass: bool = False,
                       max_context: int = 4096, max_eval: int = 8192,
                       balanced_context: bool = True, balanced_eval: bool = True,
                       n_estimators: int = 8,
                       test_fraction: float = 0.2, seed: int = 0,
                       max_events: int | None = None, device: str | None = None,
                       verbose: bool = True):
    """Train/evaluate TabFM on the refit feature matrix.

    multiclass=False -> genuine-vs-fake (comparable to the xgboost tiers).
    multiclass=True  -> track-origin species + fakes (see PDG_CLASSES).

    Returns (clf, metrics dict). Writes <output_dir>/tabfm_<mode>_<tier>-<config>_meta.json
    and a .npz of the test-set probabilities (teacher-model distillation input).
    """
    os.makedirs(output_dir, exist_ok=True)
    ref, var, hits = load_refit_tables(files, config, track_table, max_events,
                                       crossref_track_table=crossref_track_table)
    # Feature matrix: identical construction to the xgboost tier study.
    X, y_binary, names, info = build_refitq_dataset(ref, var, hits, tier, config,
                                                    label=label, require_truth=True)

    import awkward as ak

    from ngtagger.train.refitquality import _REF_HW, _REF_FLOAT, _REF_TRUTH

    ref_flat = {b: ak.to_numpy(ak.flatten(ref[b])) for b in (_REF_HW + _REF_FLOAT + _REF_TRUTH)}

    if multiclass:
        y, class_names, keep = build_origin_labels(ref_flat)
        X, y = X[keep], y[keep]
        mode = "multiclass"
    else:
        y, class_names = y_binary, ["fake", label]
        mode = "binary"

    train_idx, test_idx = _split(len(X), test_fraction, seed)
    ctx = train_idx[_subsample(y[train_idx], max_context, seed, balanced_context)]
    # Evaluation is class-balanced by DEFAULT: the metrics (AUC; fake rate at a
    # threshold set by the positive quantile) are computed within class, so
    # balancing does not bias them -- it just keeps every available rare-class
    # row instead of sampling ~1% of them, which is the whole statistical power
    # for fakes/muons. Natural proportions remain available for accuracy-style
    # readings via balanced_eval=False.
    ev = test_idx[_subsample(y[test_idx], max_eval, seed + 1, balanced_eval)]
    X_ctx, y_ctx, X_ev, y_ev = X[ctx], y[ctx], X[ev], y[ev]

    ctx_counts = {class_names[c]: int((y_ctx == c).sum()) for c in np.unique(y_ctx)}
    ev_counts = {class_names[c]: int((y_ev == c).sum()) for c in np.unique(y_ev)}
    if verbose:
        print(f"TabFM[{mode}] tier {tier} config {config}: nfeat={X.shape[1]}, "
              f"context={len(X_ctx)} {ctx_counts}, eval={len(X_ev)} {ev_counts}")

    model, dev = _load_tabfm(device=device, verbose=verbose)
    clf, proba = _fit_predict(model, X_ctx, y_ctx, X_ev, n_estimators, seed, verbose=False)

    metrics = (_metrics_multiclass(y_ev, proba, class_names) if multiclass
               else _metrics_binary(y_ev, proba, class_names))
    metrics.update({"mode": mode, "tier": tier, "config": config, "features": names,
                    "n_features": len(names), "class_names": class_names,
                    "context_size": int(len(X_ctx)), "context_counts": ctx_counts,
                    "balanced_context": bool(balanced_context),
                    "balanced_eval": bool(balanced_eval),
                    "eval_size": int(len(X_ev)), "eval_counts": ev_counts,
                    "n_estimators": n_estimators, "seed": seed, "device": dev,
                    "track_table": track_table, "crossref_track_table": crossref_track_table,
                    "label": (label if not multiclass else None),
                    "input_files": [os.path.basename(f) for f in files]})

    tag = f"{mode}_{tier}-{config}"
    with open(os.path.join(output_dir, f"tabfm_{tag}_meta.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Teacher outputs for a later BDT student (row order == eval subsample).
    np.savez_compressed(os.path.join(output_dir, f"tabfm_{tag}_teacher.npz"),
                        X_eval=X_ev, y_eval=y_ev, proba=proba,
                        feature_names=np.array(names), class_names=np.array(class_names))

    if verbose:
        _report(metrics, class_names, multiclass)
        print(f"  metrics -> {output_dir}/tabfm_{tag}_meta.json")
        print(f"  teacher probabilities -> {output_dir}/tabfm_{tag}_teacher.npz")
    return clf, metrics


def _report(m: dict, names: list[str], multiclass: bool):
    if not multiclass:
        fr = "  ".join(f"FR@{e}={v:.4f}" for e, v in m["fake_rates"].items())
        print(f"  TabFM binary: test AUC = {m['test_auc']:.4f}  {fr}")
        return
    print(f"  TabFM multiclass: accuracy = {m['accuracy']:.4f}, "
          f"balanced accuracy = {m['balanced_accuracy']:.4f}")
    print("  per-class one-vs-rest AUC:")
    for k, v in sorted(m["per_class_auc_ovr"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:10s} {v:.4f}")
    print("  row-normalized confusion (rows = true class):")
    hdr = " " * 12 + "".join(f"{n[:7]:>9s}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        print(f"    {n:8s}" + "".join(f"{v:9.3f}" for v in m["confusion_rownorm"][i]))
    print("  most confusable pairs:")
    for k, v in list(m["pair_confusability"].items())[:5]:
        print(f"    {k:22s} {v:.3f}")
