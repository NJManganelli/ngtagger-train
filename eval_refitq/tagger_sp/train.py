"""Train the NG jet tagger (framework DeepSetHGQ2 via ngtagger.train.trainer.run_training)
for one mask x feature variant x seeds, on the study's 80% training events, and dump
test-set predictions.

  pixi run python eval_refitq/tagger_sp/train.py --mask AAAA --variant baseline --seeds 0 1 2

Inputs: build.py jet caches (all samples given, TTbar + HTT combined). Acceptance
common.PT_MIN / ETA_MAX. Feature columns = common.VARIANTS[variant] resolved through
ngtagger.data.features (cumulative variants). Class weighting: the framework's
(trainer.class_pt_weights, training_config.weight_method).
Validation: the jets of the study's fixed validation events (common.val_table: 10% of each
sample's training events, the same events for every mask, cell and trial). They are put LAST and
keras' validation_split (which takes the last fraction) is set to take exactly them; the other
training jets are shuffled per seed. keras is seeded before the model is built, so a trial is
reproducible from its seed.
Outputs: OUT/models/<mask>/<variant>/s<seed>/ (framework model dir, with the trial's config) and
OUT/preds/<mask>_<variant>_s<seed>.npz: softmax probabilities of the test jets + jet kinematics +
event identity, and of the validation jets with the trial's validation classification loss
(val_closs: class-weighted cross-entropy of the jet-class output, the framework's weights) -- the
trial screen and choice (trial_select.py) read only these. A trial whose file exists for the same samples,
test events and features is skipped (rerun-safe); --force retrains it.

--d0 A|B|C|D: the PV-(x, y) d0 upper-bound study (d0pv.py): the variant's d0 inputs are
transformed (d0pv.apply), outputs go to OUT/d0_upper_bound/{models,preds} as
<mask>_<variant>_<d0>_s<seed>; the four variants share each seed's initialisation and shuffling.

--recipe linout: tagger_sp_linout.yaml (linout.py: the framework DeepSetHGQ2 with linear class
logits instead of a ReLU on them, which leaves 0-50% of the test jets of a seed with an exactly
uniform output); outputs under <out>/linout/.
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
CONFIG_LINOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_sp_linout.yaml")


def load_mask(mask, samples, variant=None):
    """Concatenate the jet caches of `mask` over `samples` (study acceptance applied), for
    exactly the files build.py used (its manifest); the variant's caches (common.jets_dir)."""
    with open(C.manifest_path(mask)) as f:
        man = json.load(f)
    if man["samples"] != list(samples):
        raise RuntimeError(f"{mask}: built for samples {man['samples']}, asked for {samples}")
    parts, test_keys = [], []
    names = None
    for si, s in enumerate(samples):
        fs = [n for s2, n, _ in man["files"] if s2 == s]
        if not fs:
            raise FileNotFoundError(f"no {s} files for {mask}")
        caches = {n: np.load(os.path.join(C.jets_dir(variant), C.tag(s, mask, n) + ".npz")) for n in fs}
        val = C.val_table(s, {n: d["evid"] for n, d in caches.items()},
                          {n: d["test_entries"] for n, d in caches.items()})
        for n, d in caches.items():
            fn = [str(x) for x in d["feature_names"]]
            if names is None:
                names = fn
            elif fn != names:
                raise RuntimeError("feature names differ between caches")
            use_ev = C.event_filter(d["evid"])                        # slim tier: events common to every mask
            test_keys.append(C.event_key(si, n, d["evid"][d["test_entries"] & use_ev]))
            acc = (d["pt"] > C.PT_MIN) & (np.abs(d["eta"]) < C.ETA_MAX) & use_ev[d["entry"]]
            ev = d["evid"][d["entry"][acc]]
            parts.append({
                "X": d["X"][acc], "label": d["label"][acc], "target_pt": d["target_pt"][acc],
                "pt": d["pt"][acc], "eta": d["eta"][acc], "is_test": d["is_test"][acc],
                "sample": np.full(acc.sum(), si), "file": np.full(acc.sum(), n),
                "entry": d["entry"][acc], "evid": ev, "is_val": val[n][d["entry"][acc]],
            })
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    D["test_event_keys"] = np.sort(np.concatenate(test_keys))
    if (D["is_val"] & D["is_test"]).any():
        raise RuntimeError(f"{mask}: validation and test jets overlap")
    return D, names


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


def trial_config(config, odir, n_val, n_all):
    """The recipe's yaml with validation_split set so that keras' split (the last
    floor-complement of n_all * (1 - split)) takes exactly the n_val validation jets put last."""
    import yaml

    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["training_config"]["validation_split"] = (n_val - 0.5) / n_all
    os.makedirs(odir, exist_ok=True)
    p = os.path.join(odir, "config_trial.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, choices=C.MASKS)
    ap.add_argument("--variant", required=True, choices=sorted(C.VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--samples", nargs="+", default=["ttbar", "htt"], choices=sorted(C.SAMPLES))
    ap.add_argument("--config", default=None, help="default: the --recipe's yaml")
    ap.add_argument("--recipe", choices=["ladder", "linout"], default="ladder",
                    help="linout: DeepSetHGQ2 with linear class logits (linout.py); outputs under <out>/linout")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gpu", action="store_true", help="use the Metal GPU (default: CPU only)")
    ap.add_argument("--tag", default="", help="suffix of the output names (dev runs)")
    ap.add_argument("--force", action="store_true", help="retrain trials whose prediction file exists")
    ap.add_argument("--d0", choices=["A", "B", "C", "D"], default=None,
                    help="PV-(x, y) d0 upper-bound variant (d0pv.py); outputs under OUT/d0_upper_bound")
    a = ap.parse_args()

    C.tf_setup(a.threads, a.gpu)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file://" + os.path.join(C.OUT, "mlruns"))
    import keras
    import yaml

    from ngtagger.data.features import resolve_feature_groups
    from ngtagger.data.labels import CLASS_LABELS
    from ngtagger.train.trainer import class_pt_weights, run_training

    D, names = load_mask(a.mask, a.samples, a.variant)
    X = D["X"]
    cols = resolve_feature_groups(C.VARIANTS[a.variant])
    out, vname = C.OUT, a.variant
    if a.d0:
        import d0pv

        X, names = d0pv.apply(X, names, d0pv.load_side(a.mask, a.samples, D), a.d0)
        cols = cols + d0pv.extra_columns(a.d0)
        out, vname = d0pv.OUTD, f"{a.variant}_{a.d0}"
    config = a.config or CONFIG
    if a.recipe == "linout":
        import linout  # noqa: F401  (registers DeepSetHGQ2LinOut)

        config = a.config or CONFIG_LINOUT
        out = os.path.join(out, "linout")
    ci = [names.index(c) for c in cols]
    X = X[:, :, ci]
    y = np.eye(len(CLASS_LABELS))[D["label"]]
    te = D["is_test"].astype(bool)
    counts = {s: {c: [int(((D["label"] == i) & (D["sample"] == si) & ~te).sum()),
                      int(((D["label"] == i) & (D["sample"] == si) & te).sum())]
                  for i, c in enumerate(CLASS_LABELS)} for si, s in enumerate(a.samples)}
    print(f"{a.mask} {vname}: {len(cols)} features, train {int((~te).sum())} / test {int(te.sum())} jets; "
          f"[train, test] per class {json.dumps(counts)}", flush=True)

    va = D["is_val"].astype(bool)
    tr_idx, va_idx = np.flatnonzero(~te & ~va), np.flatnonzero(va)
    print(f"  validation: {len(va_idx)} jets of the fixed validation events; fit: {len(tr_idx)} jets", flush=True)
    wmethod = yaml.safe_load(open(config))["training_config"].get("weight_method", "onlyclass")
    for seed in a.seeds:
        t0 = time.time()
        name = f"{a.mask}_{vname}_s{seed}{a.tag}"
        pfile = os.path.join(out, "preds", name + ".npz")
        if os.path.exists(pfile) and not a.force:
            with np.load(pfile) as old:
                current = [str(x) for x in old["samples"]] == list(a.samples) and \
                    np.array_equal(old["test_event_keys"], D["test_event_keys"]) and \
                    [str(x) for x in old["features"]] == list(cols)
            if current:
                print(f"{name}: exists, skipped", flush=True)
                continue
            print(f"{name}: stale prediction file (another sample set or split), retrained", flush=True)
        order = np.concatenate([tr_idx[np.random.default_rng(seed).permutation(len(tr_idx))], va_idx])
        ds = {
            "X_train": X[order], "y_train": y[order], "pt_train": D["target_pt"][order],
            "reco_pt_train": D["pt"][order], "charge_train": None,
            "X_test": X[te], "y_test": y[te], "pt_test": D["target_pt"][te],
            "reco_pt_test": D["pt"][te], "reco_eta_test": D["eta"][te],
            "feature_names": cols, "class_labels": CLASS_LABELS,
        }
        odir = os.path.join(out, "models", a.mask, vname + a.tag, f"s{seed}")
        cfg = trial_config(config, odir, len(va_idx), len(order))
        keras.utils.set_random_seed(seed)       # the framework seeds only at fit, after the weights exist
        model = run_training(cfg, [], odir, seed=seed, mlflow_run_name=name, dataset=ds)
        prob = softmax(model.predict(X[te])[0])
        vprob = softmax(model.predict(X[va_idx])[0])
        # the trial's validation classification loss: class-weighted cross-entropy with the framework's
        # weights (computed over the whole fit + validation set, as run_training does)
        w = class_pt_weights(y[order], D["pt"][order], wmethod)[len(tr_idx):]
        ce = -np.log(np.clip(vprob[np.arange(len(va_idx)), D["label"][va_idx]], 1e-12, None))
        val_closs = float(np.sum(w * ce) / np.sum(w))
        h = model.history.history
        os.makedirs(os.path.join(out, "preds"), exist_ok=True)
        tmp = pfile[:-4] + ".tmp.npz"
        np.savez_compressed(
            tmp, prob=prob.astype(np.float32),
            label=D["label"][te], pt=D["pt"][te], eta=D["eta"][te], sample=D["sample"][te],
            fileno=D["file"][te], entry=D["entry"][te], evid=D["evid"][te],
            val_prob=vprob.astype(np.float32), val_label=D["label"][va_idx], val_pt=D["pt"][va_idx],
            val_eta=D["eta"][va_idx], val_sample=D["sample"][va_idx], val_evid=D["evid"][va_idx],
            val_closs=val_closs,
            samples=np.array(a.samples), classes=np.array(CLASS_LABELS), features=np.array(cols),
            val_loss=np.array(h.get("val_loss", [])), loss=np.array(h.get("loss", [])),
            train_counts=json.dumps(counts), seed=seed, test_event_keys=D["test_event_keys"])
        os.replace(tmp, pfile)
        print(f"{name}: {len(h.get('loss', []))} epochs, best val_loss {model.best_val_loss():.4f}, "
              f"val_closs {val_closs:.4f}, {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
