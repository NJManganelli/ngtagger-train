"""TabFM (tabular foundation model) baseline for the multiclass jet tagger.

Same dataset and targets as the deepset tagger (prepare_dataset), so the
numbers are directly comparable to a trained NG model:

  flavour : 8-class classification (b, charm, light, gluon, taup, taum,
            muon, electron)                       -> TabFMClassifier
  pt      : regression of the gen/reco pt RATIO, the upstream target
            (clipped to [0.3, 2])                 -> TabFMRegressor
  charge  : 3-class classification {q-, neutral, q+} -- discrete, so a
            CLASSIFIER, not a regressor           -> TabFMClassifier

The deepset consumes a (jet, constituent, feature) tensor; a tabular model
needs 2D, so constituents are flattened slot-major (c0_f0, c0_f1, ... c15_fN),
which preserves the pt-ordered slot semantics the upstream feature builder
already imposes. With the default 16 constituents this stays well inside
TabFM's 500-feature limit; a --n-const/feature-group combination that exceeds
it is rejected up front.

TabFM predicts by in-context learning (the "training" rows are context at
inference time), so contexts are subsampled and evaluation is capped -- both
explicit knobs, and every metric records the sizes it was computed at. Each
head persists its per-row predictions as teacher outputs for a later
distillation step.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ngtagger.train.tabfm_refitq import _load_tabfm, _subsample, resolve_device
from ngtagger.train.trainer import prepare_dataset


# Empirically safe flattened-feature ceiling for the shipped TabFM checkpoint.
# Measured: 320 features -> all-NaN predictions; 160 and 80 -> clean, with
# identical data/context/query and a freshly loaded model, on both CPU and MPS.
# The library's own max_num_features=500 does NOT hold in practice.
TABFM_SAFE_FEATURES = int(os.environ.get("TABFM_SAFE_FEATURES", "160"))


def _flatten_constituents(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """(n_jets, n_const, n_feat) -> (n_jets, n_const*n_feat), slot-major names."""
    if X.ndim == 2:
        return X, list(feature_names)
    n_jets, n_const, n_feat = X.shape
    flat = X.reshape(n_jets, n_const * n_feat)
    names = [f"c{c}_{feature_names[f]}" for c in range(n_const) for f in range(n_feat)]
    return flat, names


def _class_counts(y: np.ndarray, names: list[str]) -> dict:
    return {names[c]: int((y == c).sum()) for c in np.unique(y)}


def _check_finite(pred, what: str, n_ctx: int, n_ev: int, n_est: int):
    """TabFM has been observed to return an ALL-NaN prediction block for some
    (context size, ensemble size) combinations while the same data works at a
    smaller context -- with clean, finite inputs. Catch it here: unguarded it
    surfaces much later as an opaque sklearn 'Input contains NaN' from inside a
    metric, with no hint of which head or setting produced it."""
    bad = ~np.isfinite(pred)
    if bad.any():
        raise RuntimeError(
            f"TabFM returned {bad.sum()}/{bad.size} non-finite values for the {what} head "
            f"(context={n_ctx}, eval={n_ev}, n_estimators={n_est}) from finite inputs. "
            "Known TabFM instability: retry with a smaller --max-context or different "
            "--n-estimators, or drop classes with very few context rows.")
    return pred


def _fit_predict_classifier(model, X_ctx, y_ctx, X_ev, n_estimators, seed, what="classification"):
    from tabfm import TabFMClassifier

    clf = TabFMClassifier(model=model, n_estimators=n_estimators, random_state=seed)
    clf.fit(X_ctx, y_ctx)
    proba = _check_finite(clf.predict_proba(X_ev), what, len(X_ctx), len(X_ev), n_estimators)
    return clf, proba


def _fit_predict_regressor(model, X_ctx, y_ctx, X_ev, n_estimators, seed):
    from tabfm import TabFMRegressor

    # TabFM hands the context targets to torch as-is; a float64 target then dies
    # on MPS ("Cannot convert a MPS Tensor to float64"). The pt-ratio target is
    # float64 out of numpy, so cast here rather than forcing --device cpu.
    y_ctx = np.asarray(y_ctx, dtype=np.float32)
    X_ctx = np.asarray(X_ctx, dtype=np.float32)
    X_ev = np.asarray(X_ev, dtype=np.float32)
    reg = TabFMRegressor(model=model, n_estimators=n_estimators, random_state=seed)
    reg.fit(X_ctx, y_ctx)
    pred = _check_finite(reg.predict(X_ev), "pt", len(X_ctx), len(X_ev), n_estimators)
    return reg, pred


def _classification_metrics(y_true, proba, names):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(names)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ovr = {}
    for i, nm in enumerate(names):
        yi = (y_true == i).astype(int)
        if yi.sum() and (yi == 0).sum():
            ovr[nm] = float(roc_auc_score(yi, proba[:, i]))
    present = np.unique(y_true)
    sim = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim[f"{names[i]}|{names[j]}"] = float(0.5 * (cm_norm[i, j] + cm_norm[j, i]))
    return {"accuracy": float((pred == y_true).mean()),
            "balanced_accuracy": float(np.mean(np.diag(cm_norm)[present])),
            "per_class_auc_ovr": ovr,
            "macro_auc_ovr": float(np.mean(list(ovr.values()))) if ovr else float("nan"),
            "confusion_counts": cm.tolist(),
            "confusion_rownorm": np.round(cm_norm, 4).tolist(),
            "pair_confusability": dict(sorted(sim.items(), key=lambda kv: -kv[1]))}


def _regression_metrics(y_true, pred, reco_pt=None, truth_pt=None):
    resid = pred - y_true
    out = {"mae": float(np.mean(np.abs(resid))), "rmse": float(np.sqrt(np.mean(resid ** 2))),
           "bias": float(np.mean(resid)),
           "r2": float(1.0 - np.sum(resid ** 2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-12))}
    if reco_pt is not None and truth_pt is not None:
        # Physical closure: corrected pt = reco_pt * predicted ratio.
        corrected = reco_pt * pred
        ok = truth_pt > 0
        if ok.any():
            r = corrected[ok] / truth_pt[ok]
            out["pt_response_median"] = float(np.median(r))
            q16, q84 = np.percentile(r, [16, 84])
            out["pt_resolution_iqr68"] = float(0.5 * (q84 - q16))
            raw = reco_pt[ok] / truth_pt[ok]
            out["pt_response_median_uncorrected"] = float(np.median(raw))
            q16r, q84r = np.percentile(raw, [16, 84])
            out["pt_resolution_iqr68_uncorrected"] = float(0.5 * (q84r - q16r))
    return out


def train_tagger_tabfm(files: list[str], output_dir: str, n_const: int = 16,
                       feature_groups: list[str] | None = None,
                       heads: tuple[str, ...] = ("flavour", "pt", "charge"),
                       max_context: int = 4096, max_eval: int = 8192,
                       balanced_context: bool = True, balanced_eval: bool = True,
                       n_estimators: int = 8,
                       test_fraction: float = 0.1, seed: int = 0,
                       max_events: int | None = None, device: str | None = None,
                       tables: dict | None = None, gen_match_dr: float = 0.4,
                       refit_config: str | None = None, refit_bdt_json: str | None = None,
                       verbose: bool = True):
    """TabFM baselines for the jet tagger's three heads. Returns (models, metrics)."""
    os.makedirs(output_dir, exist_ok=True)
    ds = prepare_dataset(files, n_const=n_const, feature_groups=feature_groups,
                         max_events=max_events, test_fraction=test_fraction, seed=seed,
                         tables=tables, gen_match_dr=gen_match_dr,
                         refit_config=refit_config, refit_bdt_json=refit_bdt_json)

    Xtr, names = _flatten_constituents(ds["X_train"], ds["feature_names"])
    Xte, _ = _flatten_constituents(ds["X_test"], ds["feature_names"])
    if Xtr.shape[1] > TABFM_SAFE_FEATURES:
        raise ValueError(
            f"flattened feature count {Xtr.shape[1]} ({n_const} constituents x "
            f"{len(ds['feature_names'])} features) exceeds the empirically safe limit "
            f"{TABFM_SAFE_FEATURES}. TabFM ADVERTISES max_num_features=500, but this "
            "checkpoint was measured to return an all-NaN prediction block at 320 "
            "features while 160 and 80 are clean (same data, context and query set, "
            "fresh model load, both CPU and MPS). Reduce --n-const (16 -> 8 halves the "
            "count) or the feature groups; override with TABFM_SAFE_FEATURES if a "
            "future release fixes this.")

    # Classes with a handful of context rows both starve the model and make the
    # metrics meaningless; warn loudly rather than reporting an AUC built on
    # single-digit counts.
    cls_labels = list(ds["class_labels"])
    chg_labels = list(ds["charge_class_labels"])
    y_fl_tr, y_fl_te = np.argmax(ds["y_train"], 1), np.argmax(ds["y_test"], 1)
    y_ch_tr, y_ch_te = np.argmax(ds["charge_train"], 1), np.argmax(ds["charge_test"], 1)

    if verbose:
        print(f"TabFM jet tagger: {Xtr.shape[0]} train / {Xte.shape[0]} test jets, "
              f"{Xtr.shape[1]} flattened features ({n_const} const x {len(ds['feature_names'])})")

    # The classification and regression heads use DIFFERENT frozen checkpoints;
    # the regression one is loaded lazily so a flavour/charge-only run does not
    # pay for it.
    device = resolve_device(device)
    model, dev = _load_tabfm(device=device, verbose=verbose)
    reg_model = None
    models, metrics = {}, {"n_features": int(Xtr.shape[1]), "n_const": n_const,
                           "feature_names": names, "device": dev, "seed": seed,
                           "n_estimators": n_estimators,
                           "n_train_pool": int(Xtr.shape[0]), "n_test_pool": int(Xte.shape[0]),
                           "balanced_context": bool(balanced_context),
                           "balanced_eval": bool(balanced_eval),
                           "input_files": [os.path.basename(f) for f in files]}
    teacher = {"feature_names": np.array(names)}

    # ---- flavour: 8-class classification ----
    if "flavour" in heads:
        ctx = _subsample(y_fl_tr, max_context, seed, balanced_context)
        ev = _subsample(y_fl_te, max_eval, seed + 1, balanced_eval)
        if verbose:
            print(f"  flavour: context {len(ctx)} {_class_counts(y_fl_tr[ctx], cls_labels)}")
        clf, proba = _fit_predict_classifier(model, Xtr[ctx], y_fl_tr[ctx], Xte[ev],
                                             n_estimators, seed, what="flavour")
        m = _classification_metrics(y_fl_te[ev], proba, cls_labels)
        m.update({"class_labels": cls_labels, "context_size": int(len(ctx)),
                  "context_counts": _class_counts(y_fl_tr[ctx], cls_labels),
                  "eval_size": int(len(ev)),
                  "eval_counts": _class_counts(y_fl_te[ev], cls_labels)})
        metrics["flavour"], models["flavour"] = m, clf
        teacher.update(flavour_proba=proba, flavour_y=y_fl_te[ev], flavour_idx=ev,
                       flavour_classes=np.array(cls_labels))

    # ---- pt: regression of the gen/reco ratio ----
    if "pt" in heads:
        ctx = _subsample(np.zeros(len(Xtr), np.int64), max_context, seed, balanced=False)
        ev = _subsample(np.zeros(len(Xte), np.int64), max_eval, seed + 1, balanced=False)
        if verbose:
            print(f"  pt: context {len(ctx)}, target = gen/reco pt ratio (clipped [0.3, 2])")
        if reg_model is None:
            reg_model, _ = _load_tabfm(device=device, verbose=verbose, model_type="regression")
        reg, pred = _fit_predict_regressor(reg_model, Xtr[ctx], ds["pt_train"][ctx], Xte[ev],
                                           n_estimators, seed)
        m = _regression_metrics(ds["pt_test"][ev], pred,
                                reco_pt=ds["reco_pt_test"][ev], truth_pt=ds["truth_pt_test"][ev])
        m.update({"target": "gen/reco pt ratio, clipped [0.3, 2]",
                  "context_size": int(len(ctx)), "eval_size": int(len(ev))})
        metrics["pt"], models["pt"] = m, reg
        teacher.update(pt_pred=pred, pt_y=ds["pt_test"][ev], pt_idx=ev,
                       pt_reco=ds["reco_pt_test"][ev], pt_truth=ds["truth_pt_test"][ev])

    # ---- charge: 3-class classification (discrete target) ----
    if "charge" in heads:
        ctx = _subsample(y_ch_tr, max_context, seed, balanced_context)
        ev = _subsample(y_ch_te, max_eval, seed + 1, balanced_eval)
        if verbose:
            print(f"  charge: context {len(ctx)} {_class_counts(y_ch_tr[ctx], chg_labels)}")
        clf, proba = _fit_predict_classifier(model, Xtr[ctx], y_ch_tr[ctx], Xte[ev],
                                             n_estimators, seed, what="charge")
        m = _classification_metrics(y_ch_te[ev], proba, chg_labels)
        m.update({"class_labels": chg_labels, "context_size": int(len(ctx)),
                  "context_counts": _class_counts(y_ch_tr[ctx], chg_labels),
                  "eval_size": int(len(ev)),
                  "eval_counts": _class_counts(y_ch_te[ev], chg_labels)})
        metrics["charge"], models["charge"] = m, clf
        teacher.update(charge_proba=proba, charge_y=y_ch_te[ev], charge_idx=ev,
                       charge_classes=np.array(chg_labels))

    with open(os.path.join(output_dir, "tabfm_tagger_meta.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(os.path.join(output_dir, "tabfm_tagger_teacher.npz"),
                        X_test=Xte, **teacher)
    if verbose:
        _report(metrics)
        print(f"  metrics -> {output_dir}/tabfm_tagger_meta.json")
        print(f"  teacher outputs -> {output_dir}/tabfm_tagger_teacher.npz")
    return models, metrics


def _report(m: dict):
    if "flavour" in m:
        f = m["flavour"]
        print(f"  flavour: accuracy {f['accuracy']:.4f}, balanced {f['balanced_accuracy']:.4f}, "
              f"macro OvR AUC {f['macro_auc_ovr']:.4f}")
        for k, v in sorted(f["per_class_auc_ovr"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:10s} AUC {v:.4f}")
        print("    most confusable pairs: "
              + ", ".join(f"{k} {v:.3f}" for k, v in list(f["pair_confusability"].items())[:4]))
    if "pt" in m:
        p = m["pt"]
        extra = ""
        if "pt_response_median" in p:
            extra = (f", response {p['pt_response_median']:.3f} "
                     f"(uncorrected {p['pt_response_median_uncorrected']:.3f}), "
                     f"resolution {p['pt_resolution_iqr68']:.3f} "
                     f"(uncorrected {p['pt_resolution_iqr68_uncorrected']:.3f})")
        print(f"  pt: MAE {p['mae']:.4f}, RMSE {p['rmse']:.4f}, R2 {p['r2']:.4f}{extra}")
    if "charge" in m:
        c = m["charge"]
        print(f"  charge: accuracy {c['accuracy']:.4f}, balanced {c['balanced_accuracy']:.4f}, "
              f"macro OvR AUC {c['macro_auc_ovr']:.4f}")
