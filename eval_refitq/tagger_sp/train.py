"""Train the NG jet tagger (framework DeepSetHGQ2 via ngtagger.train.trainer.run_training)
for one mask x feature variant x seeds, on the study's 80% training events, and dump
test-set predictions.

  pixi run python eval_refitq/tagger_sp/train.py --mask AAAA --variant baseline --seeds 0 1 2

Inputs: build.py jet caches (all samples given, TTbar + HTT combined). Acceptance
common.PT_MIN / ETA_MAX. Feature columns = common.VARIANTS[variant] resolved through
ngtagger.data.features (cumulative variants). Class weighting: the framework's
(trainer.class_pt_weights, training_config.weight_method). Training jets are shuffled
per seed before keras' validation_split (which takes the LAST fraction).
Outputs: OUT/models/<mask>/<variant>/s<seed>/ (framework model dir) and
OUT/preds/<mask>_<variant>_s<seed>.npz (softmax probabilities of the test jets + jet
kinematics + event identity).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_sp.yaml")


def load_mask(mask, samples):
    """Concatenate the jet caches of `mask` over `samples` (study acceptance applied)."""
    parts = []
    names = None
    for si, s in enumerate(samples):
        fs = C.files(s, mask)
        if not fs:
            raise FileNotFoundError(f"no {s} files for {mask}")
        for n, _ in fs:
            d = np.load(os.path.join(C.cache_dir("jets"), C.tag(s, mask, n) + ".npz"))
            fn = [str(x) for x in d["feature_names"]]
            if names is None:
                names = fn
            elif fn != names:
                raise RuntimeError("feature names differ between caches")
            acc = (d["pt"] > C.PT_MIN) & (np.abs(d["eta"]) < C.ETA_MAX)
            ev = d["evid"][d["entry"][acc]]
            parts.append({
                "X": d["X"][acc], "label": d["label"][acc], "target_pt": d["target_pt"][acc],
                "pt": d["pt"][acc], "eta": d["eta"][acc], "is_test": d["is_test"][acc],
                "sample": np.full(acc.sum(), si), "file": np.full(acc.sum(), n),
                "entry": d["entry"][acc], "evid": ev,
            })
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return D, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, choices=C.MASKS)
    ap.add_argument("--variant", required=True, choices=sorted(C.VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--samples", nargs="+", default=["ttbar", "htt"], choices=sorted(C.SAMPLES))
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gpu", action="store_true", help="use the Metal GPU (default: CPU only)")
    ap.add_argument("--tag", default="", help="suffix of the output names (dev runs)")
    a = ap.parse_args()

    C.tf_setup(a.threads, a.gpu)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file://" + os.path.join(C.OUT, "mlruns"))
    from ngtagger.data.features import resolve_feature_groups
    from ngtagger.data.labels import CLASS_LABELS
    from ngtagger.train.trainer import run_training

    D, names = load_mask(a.mask, a.samples)
    cols = resolve_feature_groups(C.VARIANTS[a.variant])
    ci = [names.index(c) for c in cols]
    X = D["X"][:, :, ci]
    y = np.eye(len(CLASS_LABELS))[D["label"]]
    te = D["is_test"].astype(bool)
    counts = {s: {c: [int(((D["label"] == i) & (D["sample"] == si) & ~te).sum()),
                      int(((D["label"] == i) & (D["sample"] == si) & te).sum())]
                  for i, c in enumerate(CLASS_LABELS)} for si, s in enumerate(a.samples)}
    print(f"{a.mask} {a.variant}: {len(cols)} features, train {int((~te).sum())} / test {int(te.sum())} jets; "
          f"[train, test] per class {json.dumps(counts)}", flush=True)

    tr_idx = np.flatnonzero(~te)
    for seed in a.seeds:
        t0 = time.time()
        perm = tr_idx[np.random.default_rng(seed).permutation(len(tr_idx))]
        ds = {
            "X_train": X[perm], "y_train": y[perm], "pt_train": D["target_pt"][perm],
            "reco_pt_train": D["pt"][perm], "charge_train": None,
            "X_test": X[te], "y_test": y[te], "pt_test": D["target_pt"][te],
            "reco_pt_test": D["pt"][te], "reco_eta_test": D["eta"][te],
            "feature_names": cols, "class_labels": CLASS_LABELS,
        }
        name = f"{a.mask}_{a.variant}_s{seed}{a.tag}"
        odir = os.path.join(C.OUT, "models", a.mask, a.variant + a.tag, f"s{seed}")
        model = run_training(a.config, [], odir, seed=seed, mlflow_run_name=name, dataset=ds)
        logits = model.predict(X[te])[0]
        z = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        h = model.history.history
        os.makedirs(os.path.join(C.OUT, "preds"), exist_ok=True)
        np.savez_compressed(
            os.path.join(C.OUT, "preds", name + ".npz"), prob=prob.astype(np.float32),
            label=D["label"][te], pt=D["pt"][te], eta=D["eta"][te], sample=D["sample"][te],
            fileno=D["file"][te], entry=D["entry"][te], evid=D["evid"][te],
            samples=np.array(a.samples), classes=np.array(CLASS_LABELS), features=np.array(cols),
            val_loss=np.array(h.get("val_loss", [])), loss=np.array(h.get("loss", [])),
            train_counts=json.dumps(counts), seed=seed)
        print(f"{name}: {len(h.get('loss', []))} epochs, best val_loss {model.best_val_loss():.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
