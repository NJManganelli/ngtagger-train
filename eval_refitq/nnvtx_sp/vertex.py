"""E2E NNVtx (weight NN + soft histogram + pattern NN + association NN, float;
ngtagger.train.nnvtx.build_e2e_model) per variant, without ("stock") and with ("mva")
the per-track MVA scores, several seeds, event-level K-fold cross-validation.

Inputs per event: the variant's SELECTED tracks (the GTT vertexer input; sub2gev
selections), sorted by pT, at most MAXT.
  stock  (min(pT, 128), trkMVA1, |eta|)              -- the GTT E2E inputs
  mva    stock + (P(clean), P(d0acc), P(z0acc), log sigma_hat(d0), log sigma_hat(z0))
         OT heads for V0, SP heads for V1/V3 (ngtagger train-trackmva, canonical family)
Targets: GenVtx_z (Huber, delta 0.5 cm) + HS flag per track (BCE, weight 5).

Cross-validation (the pilot's scheme): event folds 0..4 on f03-f10 = the train-trackmva
folds; f01/f02 (fold -1, in-sample for the cluster CNN) are in every training set and
never predicted. Fold-k events are predicted by a model trained on the other folds
(+ f01/f02); the MVA scores that model sees in training AND at prediction are
scores[k] of the NESTED cross-fit, so every score is out of sample. Seeds vary the
initialisation and the 15% early-stopping split. Weight-output bias starts at +1
(the pilot's warm start; the zero bias collapsed 2/3 seeds there).

  pixi run python eval_refitq/nnvtx_sp/vertex.py [--variants V0,V1,V3] [--arms stock,mva] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvcommon import (DATASET, FILES, MVA_COLL, MVA_STEM, N_FOLDS, PRED, event_folds, load_dataset,  # noqa: E402
                    selections)

MAXT = 300  # >= the largest selected-track count per event (297, V3): nothing is clipped
PT_CAP = 128.0
N_MVA = 5


def tf_setup(threads):
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # CPU only (tensorflow-metal is installed)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)


class Padded:
    """(event, slot) view of the selected tracks of one variant, pT-descending."""

    def __init__(self, T, n_ev, sel, pre):
        rows = np.flatnonzero(sel)
        ev = T["evt"][rows]
        order = np.lexsort((-T[pre + "pt"][rows], ev))
        rows, ev = rows[order], ev[order]
        start = np.searchsorted(ev, np.arange(n_ev))
        rank = np.arange(len(rows)) - start[ev]
        keep = rank < MAXT
        self.idx = np.full((n_ev, MAXT), -1, np.int64)
        self.idx[ev[keep], rank[keep]] = rows[keep]
        self.mask = (self.idx >= 0).astype(np.float32)
        self.n_clipped = int((~keep).sum())
        self.n_clipped_events = int(np.unique(ev[~keep]).size)

    def g(self, arr, fill=0.0):
        return np.where(self.idx >= 0, arr[np.maximum(self.idx, 0)], fill)


def standardize(X, mask, ev):
    m = mask[ev] > 0
    mu, sd = X[ev][m].mean(0), X[ev][m].std(0) + 1e-6
    return ((X - mu) / sd * mask[..., None]).astype(np.float32)


def warm_bias(model, b=1.0):
    lay = model.get_layer("weight_out")
    lay.bias.assign(np.full(lay.bias.shape, b, dtype="float32"))


def split_val(tr_ev, seed):
    rng = np.random.default_rng(1000 + seed)
    p = rng.permutation(tr_ev)
    nv = int(round(0.15 * len(p)))
    return np.sort(p[nv:]), np.sort(p[:nv])


def load_scores(T, E):
    """{coll: (N_FOLDS, n_tracks, 5)} nested cross-fitted scores, checked row-aligned."""
    d = np.load(MVA_STEM + "_oof.npz", allow_pickle=False)
    assert [str(f) for f in d["files"]] == FILES, "oof scores from other files"
    assert np.array_equal(d["ev_file"], E["file"]) and np.array_equal(d["ev_entry"], E["entry"])
    assert np.array_equal(d["track_evt"], T["evt"])
    folds = event_folds(E)
    assert np.array_equal(d["folds_ev"], folds), "fold assignment differs from train-trackmva"
    assert list(d["outputs"]) == ["clean", "d0acc", "z0acc", "log_sd0", "log_sz0"]
    return {c: d[f"{c}_scores"] for c in ("base", "refit")}, folds


def run_variant(T, E, v, arms, seeds, epochs, patience, scores, folds, pred_path, log):
    import keras
    from ngtagger.train.nnvtx import build_e2e_model

    n_ev = len(E["evt"])
    sel, pre, hs = selections(T)[v]
    P = Padded(T, n_ev, sel, pre)
    log(f"[{v}] {int(sel.sum())} selected tracks ({sel.sum() / n_ev:.1f}/ev), clipped at MAXT={MAXT}: "
        f"{P.n_clipped} tracks in {P.n_clipped_events} events")
    z0 = P.g(T[pre + "z0"]).astype(np.float32)
    y = P.g(hs.astype(np.float32)).astype(np.float32)
    gz = E["genz"].astype(np.float32).reshape(-1, 1)
    stock = np.stack([P.g(np.minimum(T[pre + "pt"], PT_CAP)), P.g(T["b_trkMVA1"]), P.g(np.abs(T[pre + "eta"]))], -1)
    S = scores[MVA_COLL[v]]
    out = {"idx": P.idx, "mask": P.mask, "folds": folds}
    if os.path.exists(pred_path):  # merge with earlier invocations (same inputs required)
        old = dict(np.load(pred_path))
        assert all(np.array_equal(old[k], out[k]) for k in ("idx", "folds")), f"{pred_path}: other inputs"
        out = {**old, **out}
    for arm in arms:
        for seed in seeds:
            zp = np.full(n_ev, np.nan)
            ap = np.zeros_like(P.mask)
            ep, t0 = [], time.perf_counter()
            for k in range(N_FOLDS):
                te = np.flatnonzero(folds == k)
                tr, va = split_val(np.flatnonzero(folds != k), seed)
                X = stock
                if arm == "mva":
                    X = np.concatenate([stock, np.stack([P.g(S[k, :, j]) for j in range(N_MVA)], -1)], -1)
                assert np.isfinite(X[P.mask > 0]).all(), "non-finite input"
                X = standardize(X, P.mask, np.concatenate([tr, va]))
                keras.utils.set_random_seed(seed * 100 + k)
                m = build_e2e_model(X.shape[-1], 0, max_tracks=MAXT)
                warm_bias(m)
                m.compile(optimizer=keras.optimizers.Adam(1e-3),
                          loss=[keras.losses.Huber(delta=0.5), keras.losses.BinaryCrossentropy()],
                          loss_weights=[1.0, 5.0])
                ins, tg = [z0, X, P.mask], [gz, y]
                h = m.fit([a[tr] for a in ins], [a[tr] for a in tg],
                          validation_data=([a[va] for a in ins], [a[va] for a in tg]),
                          epochs=epochs, batch_size=32, verbose=0,
                          callbacks=[keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)])
                ep.append(len(h.history["loss"]))
                pz, pa = m.predict([a[te] for a in ins], verbose=0, batch_size=64)
                zp[te], ap[te] = pz[:, 0], pa
                keras.backend.clear_session()
            out[f"{arm}_s{seed}_z"] = zp
            out[f"{arm}_s{seed}_assoc"] = ap.astype(np.float32)
            out[f"{arm}_s{seed}_epochs"] = np.array(ep)
            t = folds >= 0
            eff = np.mean(np.abs(zp[t] - E["genz"][t]) < 0.15)
            from sklearn.metrics import roc_auc_score
            mk = (P.mask[t] > 0)
            auc = roc_auc_score(y[t][mk], ap[t][mk])
            log(f"[{v}/{arm}/s{seed}] {time.perf_counter() - t0:.0f}s epochs {ep} eff {eff:.3f} auc {auc:.4f}")
            np.savez_compressed(pred_path, **out)  # checkpoint after every seed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="V0,V1,V3")
    ap.add_argument("--arms", default="stock,mva")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-events", type=int, default=None, help="smoke test only")
    ap.add_argument("--pred", default=PRED, help="output pattern with {variant}")
    a = ap.parse_args()
    tf_setup(a.threads)
    log = lambda s: print(s, flush=True)  # noqa: E731
    T, E = load_dataset(DATASET)
    scores, folds = load_scores(T, E)
    if a.max_events:  # smoke test: keep events < max_events (tracks are event-ordered)
        keep_t = T["evt"] < a.max_events
        T = {k: v[keep_t] for k, v in T.items()}
        E = {k: v[:a.max_events] for k, v in E.items()}
        scores = {c: s[:, keep_t] for c, s in scores.items()}
        folds = folds[:a.max_events]
    for v in a.variants.split(","):
        run_variant(T, E, v, a.arms.split(","), [int(s) for s in a.seeds.split(",")], a.epochs, a.patience,
                    scores, folds, a.pred.format(variant=v), log)


if __name__ == "__main__":
    main()
