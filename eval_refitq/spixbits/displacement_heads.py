"""Per-track DISPLACEMENT heads (physicist, 2026-11-14): from track quantities, where was the track's particle produced?
Second version (physicist: no regression of the production radius; "a multi-classifier that targets 'inside layer k' /
'outside layer k, inside layer k+1' ... for valid layers in a scenario", which "translates DIRECTLY to a range,
mathematically, based on eta"). The definitions (surfaces, trajectory, classes, per-surface inputs, range) are in
displacement.py, shared with the jet tagger and the pair algorithm.

  [TAGGER_SP_COLLECTION=prompt TAGGER_SP_OUT=/Volumes/WDMac/smartpixels_scratch/tagger_v5_prompt/heads_input] \\
      python displacement_heads.py counts | train [--masks ...] [--trials 20] | report
  (without the two variables: the extended collection, the tagger study's own caches under tagger_v5)

Detector scenarios AAAA, AAAI, IAAA; IIII has no SmartPixels surface to divide on and gets no displacement head.
Sample, split and trial policy: those of the heads retrained per detector scenario (retrain_heads.py): today's
("implemented now") tracks of canon_v5 TTbar files 1-3, 5, 7-10 and seven H->tautau files (1,150 events; files 4 and
6 left out), the tagger study's split by event identity, 20 trials (trial 0 deterministic; 1-19 with row and column
subsampling 0.8, seed = trial), the best trial by validation loss, a non-finite one discarded. XGBoost hist, depth 3,
100 rounds, learning rate 0.1 (as the first version), one softmax model per window (fitted |d0| <= 0.02 cm, beyond).
Target: displacement.true_class of the particle's production radius (hypot(tp_vx, tp_vy)) on the particle's own
trajectory (its true tanL and z0): truth is the label only. A track without a particle has no class: it is left out
of the training and of the figures (fakes stay with the "clean" head); at application it gets outputs like any track.
Inputs (displacement.assemble): the 38 inputs of the track-quality heads, the fitted |d0| and |d0| / sigma(d0), and
the per-surface inputs of displacement.slot_inputs from the refit-hit table (no truth column: refuse_truth_columns).
Control ("without the per-surface inputs"): the same classes from the 38 inputs + |d0| and |d0| / sigma only, to tell
what the per-surface inputs add from what the new classes change (classes_without_surfaces.json; not used downstream).
Comparison: the first version (five classes by fixed radius 2.85 / 5.95 / 10.3 / 14.5 cm, the 38 inputs + |d0| and
|d0| / sigma; its stored models under models/<detector scenario>/d0/, trained by this file at commit 46af795) and a
cut on |d0| or |d0| / sigma, on the same test tracks.
Truth for the figures: K0S pions = a pi+ and a pi- tracking particle with a bit-identical production vertex and
nothing else charged-pion at it, invariant mass within 1 MeV of the K0S mass (l1-flow-bphys-study/COUNTS.md).
Outputs under ~/smartpixels/scratch_local/spixbits/displacement_heads/<collection>/ (second version: models_v2/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tagger_sp"))
os.environ.setdefault("TAGGER_SP_OUT", "/Volumes/WDMac/smartpixels_scratch/tagger_v5")
import common as C          # noqa: E402  (tagger_sp/common.py)
import displacement as D    # noqa: E402
import joint_heads as J     # noqa: E402
import retrain_heads as RH  # noqa: E402

OUT = Path.home() / "smartpixels/scratch_local/spixbits/displacement_heads" / C.COLLECTION
MASKS = ["AAAA", "AAAI", "IAAA"]
V1_EDGES = np.array([2.85, 5.95, 10.3, 14.5])             # cm: the first version's class boundaries
R_PROMPT = 1.0                                            # cm: "produced inside 1 cm"
XGB = {"tree_method": "hist", "max_depth": 3, "eta": 0.1, "nthread": 4}
ROUNDS = 100
MODELS = {"classes": 2, "classes_without_surfaces": 1}     # model name: the input version (1 = without the per-surface inputs)
M_PI, M_K0S = 0.13957, 0.497611


def k0s_flags(A, tab):
    """Per track of table `tab` (flattened): its particle is a pion of a K0S -> pi+ pi- (see the docstring)."""
    import awkward as ak
    out = []
    for ev in range(len(A["L1TTP_pt"])):
        g = lambda c: ak.to_numpy(A[f"L1TTP_{c}"][ev])
        pdg = g("pdgId")
        pi = np.flatnonzero(np.abs(pdg) == 211)
        key = np.stack([g("evt")[pi].astype(np.float64), g("bx")[pi].astype(np.float64), g("vx")[pi], g("vy")[pi], g("vz")[pi]], 1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        is_k = np.zeros(len(pdg), bool)
        two = np.flatnonzero(cnt[inv] == 2)
        if len(two):
            o = two[np.argsort(inv[two], kind="stable")]
            a, b = pi[o[0::2]], pi[o[1::2]]
            pt, eta, phi = g("pt").astype(np.float64), g("eta").astype(np.float64), g("phi").astype(np.float64)
            p4 = lambda i: (pt[i] * np.cos(phi[i]), pt[i] * np.sin(phi[i]), pt[i] * np.sinh(eta[i]),
                            np.sqrt((pt[i] * np.cosh(eta[i])) ** 2 + M_PI ** 2))
            xa, xb = p4(a), p4(b)
            m = np.sqrt(np.maximum((xa[3] + xb[3]) ** 2 - sum((xa[k] + xb[k]) ** 2 for k in range(3)), 0.0))
            ok = (pdg[a] * pdg[b] < 0) & (np.abs(m - M_K0S) < 1e-3)
            is_k[a[ok]] = is_k[b[ok]] = True
        ti = ak.to_numpy(A[f"{tab}_tpIdx"][ev])
        out.append(np.where(ti >= 0, is_k[np.maximum(ti, 0)], False))
    return np.concatenate(out)


def extra(sample, mask, n, fn):
    """Per track, in the track cache's order: the particle's production radius [cm] (NaN without one), its PDG code
    and the K0S-pion flag."""
    import awkward as ak
    import uproot
    out = OUT / "cache" / (C.tag(sample, mask, n) + ".npz")
    if out.exists():
        return out
    tab = C.OT_TABLE
    t = uproot.open(fn)["Events"]
    # sigma(d0) of the track the tagger reads: the SmartPixels refit's where it exists, else the OT track's
    cov = f"{C.TRACK_TABLE}_cov_d0_d0" if f"{C.TRACK_TABLE}_cov_d0_d0" in t.keys() else f"{tab}_cov_d0_d0"
    cols = [f"{tab}_{c}" for c in ("pt", "tpIdx", "tp_vx", "tp_vy", "tpPdgId")] + [cov] + [
        f"L1TTP_{c}" for c in ("pdgId", "evt", "bx", "vx", "vy", "vz", "pt", "eta", "phi")]
    parts = []
    for start in range(0, t.num_entries, 10):
        A = t.arrays(cols, entry_start=start, entry_stop=min(start + 10, t.num_entries))
        f = lambda c: ak.to_numpy(ak.flatten(A[f"{tab}_{c}"]))
        has = f("tpIdx") >= 0
        parts.append({"pt": f("pt"), "r_prod": np.where(has, np.hypot(f("tp_vx"), f("tp_vy")), np.nan).astype(np.float32),
                      "pdg": np.where(has, f("tpPdgId"), 0).astype(np.int32), "k0s": k0s_flags(A, tab),
                      "sig_d0": np.sqrt(np.abs(ak.to_numpy(ak.flatten(A[cov])))).astype(np.float32)})
    E = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tc = np.load(os.path.join(C.cache_dir("tracks"), C.tag(sample, mask, n) + ".npz"))
    if not np.array_equal(E["pt"], tc["T_b_pt"].astype(E["pt"].dtype)):
        raise RuntimeError(f"{fn}: not aligned with the track cache")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".tmp.npz", **E)
    os.replace(str(out) + ".tmp.npz", out)
    return out


def hit_inputs(sample, mask, n, fn):
    """Per track, in the track cache's order: the per-surface inputs and the track's crossing radii (displacement.slot_inputs)."""
    import awkward as ak
    import uproot
    out = OUT / "cache_v2" / (C.tag(sample, mask, n) + ".npz")
    if out.exists():
        return out
    sp, ot = C.TRACK_TABLE, C.OT_TABLE
    hit = D.hit_table(sp)
    t = uproot.open(fn)["Events"]
    D.refuse_truth_columns(t, hit)
    A = t.arrays([f"{hit}_{c}" for c in D.HIT_COLUMNS + ["layer", "trackIdx"]] + [f"{sp}_{c}" for c in ("tanL", "z0", "l1TrackIdx")] + [f"{ot}_pt"])
    ntrk = ak.to_numpy(ak.num(A[f"{sp}_tanL"]))
    if not np.array_equal(ak.to_numpy(ak.flatten(A[f"{sp}_l1TrackIdx"])), np.concatenate([np.arange(k) for k in ntrk])):
        raise RuntimeError(f"{fn}: {sp} is not in the order of {ot}")
    f = lambda c: ak.to_numpy(ak.flatten(A[c]))
    first = np.repeat(np.cumsum(ntrk) - ntrk, ak.to_numpy(ak.num(A[f"{hit}_layer"])))
    X, radii = D.slot_inputs(mask, f(f"{sp}_tanL"), f(f"{sp}_z0"), first + f(f"{hit}_trackIdx"), f(f"{hit}_layer"),
                             {c: f(f"{hit}_{c}") for c in D.HIT_COLUMNS})
    tc = np.load(os.path.join(C.cache_dir("tracks"), C.tag(sample, mask, n) + ".npz"))
    if not np.array_equal(f(f"{ot}_pt"), tc["T_b_pt"].astype(np.float32)):
        raise RuntimeError(f"{fn}: not aligned with the track cache")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".tmp.npz", slots=X, radii=radii.astype(np.float32))
    os.replace(str(out) + ".tmp.npz", out)
    return out


def load(mask):
    """The heads' tracks of the mask (retrain_heads.load: split flags, files 4 and 6 marked) with the displacement truth,
    the per-surface inputs and the true class."""
    T, man = RH.load(mask)
    ex = [dict(np.load(extra(s, mask, n, p))) for s, n, p in man["files"]]
    for k in ("r_prod", "pdg", "k0s", "sig_d0"):
        T["x_" + k] = np.concatenate([e[k] for e in ex])
    hi = [np.load(hit_inputs(s, mask, n, p)) for s, n, p in man["files"]]
    T["x_slots"], T["x_radii"] = (np.concatenate([h[k] for h in hi]) for k in ("slots", "radii"))
    assert len(T["x_r_prod"]) == len(T["b_pt"]) == len(T["x_slots"])
    T["x_class"] = np.where(np.isfinite(T["x_r_prod"]), D.true_class(np.nan_to_num(T["x_r_prod"]), T["b_tp_tanL"], T["b_tp_z0"], mask), -1)
    return T


def inputs(T, version):
    """version 2: displacement.assemble; version 1: the 38 inputs + |d0| and |d0| / sigma(d0)."""
    d0 = np.abs(T["r_d0"])
    X = D.assemble(J.features(T, "refit"), d0, T["x_sig_d0"], T["x_slots"])
    return X if version == 2 else X[:, :-len(D.INPUT_NAMES)]


def model_dir(mask, window):
    return OUT / "models_v2" / mask / window


def fit_trial(X, y, tr, va, n_class, t):
    import xgboost as xgb
    par = dict(XGB, seed=t, objective="multi:softprob", num_class=n_class, eval_metric="mlogloss",
               **({"subsample": 0.8, "colsample_bynode": 0.8} if t else {}))
    b = xgb.train(par, xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=ROUNDS)
    p = b.predict(xgb.DMatrix(X[va]))
    return b, float(-np.mean(np.log(np.clip(p[np.arange(len(va)), y[va]], 1e-12, 1))))


def train(a):
    from ngtagger.train import trackmva as tm
    rec_path = OUT / "trials_v2.json"
    rec = json.load(open(rec_path)) if rec_path.exists() else {}
    for mask in a.masks:
        T = load(mask)
        ok, y, n_class = T["x_class"] >= 0, T["x_class"], D.n_edges(mask) + 1
        for wname, w in tm.windows(T, "refit").items():
          for name, version in MODELS.items():
            key = f"{mask}/{wname}" + ("" if version == 2 else "/without_surfaces")
            d = model_dir(mask, wname)
            d.mkdir(parents=True, exist_ok=True)
            if key in rec and (d / f"{name}.json").exists() and (d / "shape.json").exists():
                continue
            X = inputs(T, version)
            tr = np.flatnonzero(w & ok & ~T["_test"] & ~T["_val"] & ~T["_left_out"])
            va = np.flatnonzero(w & ok & T["_val"] & ~T["_left_out"])
            t0, best, losses = time.time(), None, {}
            for t in range(a.trials):
                b, loss = fit_trial(X, y, tr, va, n_class, t)
                losses[t] = loss if np.isfinite(loss) else None
                if np.isfinite(loss) and (best is None or loss < best[1]):
                    best = (t, loss, b)
            best[2].save_model(str(d / f"{name}.json"))
            shape = D.fit_shape(T["x_r_prod"][tr], y[tr], T["x_radii"][tr], mask)      # the convention inside a class: training tracks
            json.dump({"what": "displacement.fit_shape on the training tracks of this window", "r_max_cm": D.R_MAX,
                       "grid": D.SHAPE_GRID.tolist(), "position_quantiles_by_class": shape.tolist()}, open(d / "shape.json", "w"))
            good = [v for v in losses.values() if v is not None]
            rec[key] = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_inputs": int(X.shape[1]), "n_classes": n_class,
                        "best_trial": best[0], "best_val_loss": best[1], "val_loss_trials": losses, "n_failed": a.trials - len(good),
                        "val_loss_sd": float(np.std(good)), "class_counts_train": np.bincount(y[tr], minlength=n_class).tolist()}
            json.dump(rec, open(rec_path, "w"), indent=1)
            print(f"{C.COLLECTION} {key}: best trial {best[0]} loss {best[1]:.4f} (sd {np.std(good):.4f}) {time.time() - t0:.0f} s", flush=True)


def predict(mask, T, name="classes"):
    """Second version: (P (n, N + 1) class probabilities, Q (n, 3) low / median / high radius [cm]), routed by window."""
    import xgboost as xgb
    from ngtagger.train import trackmva as tm
    X = inputs(T, MODELS[name])
    P, Q = np.full((len(X), D.n_edges(mask) + 1), np.nan), np.full((len(X), len(D.QUANTILES)), np.nan)
    for wname, w in tm.windows(T, "refit").items():
        b = xgb.Booster()
        b.load_model(str(model_dir(mask, wname) / f"{name}.json"))
        P[w] = b.predict(xgb.DMatrix(X[w]))
        shape = np.array(json.load(open(model_dir(mask, wname) / "shape.json"))["position_quantiles_by_class"])
        Q[w] = D.radius_range(P[w], T["x_radii"][w], mask, shape)
    return P, Q


def predict_v1(mask, T):
    """First version (stored models): the five probabilities of the fixed-radius classes."""
    import xgboost as xgb
    from ngtagger.train import trackmva as tm
    X = inputs(T, 1)
    P = np.full((len(X), 5), np.nan)
    for wname, w in tm.windows(T, "refit").items():
        b = xgb.Booster()
        b.load_model(str(OUT / "models" / mask / "d0" / wname / "band.json"))
        P[w] = b.predict(xgb.DMatrix(X[w]))
    return P


def counts(a):
    """Matched head-training tracks per class, and what the cap and the acceptance do."""
    out = {}
    for mask in a.masks:
        T = load(mask)
        N = D.n_edges(mask)
        ok = ~T["_left_out"] & (T["x_class"] >= 0)
        radii, _, n_true = D.crossings(T["b_tp_tanL"], T["b_tp_z0"], mask)
        raw = (radii < np.nan_to_num(T["x_r_prod"])[:, None]).sum(1)
        _, _, n_trk = D.crossings(T["r_tanL"], T["r_z0"], mask)
        acc = T["x_slots"][:, 2::len(D.SLOT_NAMES)][:, :D.N_SLOTS]
        e = {"tracks": int((~T["_left_out"]).sum()), "with_a_particle": int(ok.sum()),
             "crossed_surfaces_per_track": np.bincount(n_trk[~T["_left_out"]], minlength=D.N_SLOTS + 1).tolist(),
             "fraction_with_fewer_than_N_crossed": float(np.mean(n_trk[~T["_left_out"]] < N)),
             "fraction_beyond_the_cap": float(np.mean(raw[ok] > N)),
             "fraction_beyond_the_cap_of_class_1_and_higher": float(np.mean(raw[ok & (T["x_class"] > 0)] > N)),
             "class_by_track_trajectory_equals_class_by_true_trajectory": float(np.mean(
                 D.true_class(np.nan_to_num(T["x_r_prod"]), T["r_tanL"], T["r_z0"], mask)[ok] == T["x_class"][ok])),
             "accepted_hits_in_slots": int(np.nansum(acc[~T["_left_out"]])), "accepted_hits_of_the_refit": int(T["r_spixNAcceptedHits"][~T["_left_out"]].sum())}
        for name, sel in (("all", ok), ("train", ok & ~T["_test"] & ~T["_val"]), ("validation", ok & T["_val"]), ("test", ok & T["_test"]),
                          ("k0s_pions_all", ok & T["x_k0s"]), ("k0s_pions_test", ok & T["x_k0s"] & T["_test"])):
            e[name] = np.bincount(T["x_class"][sel], minlength=N + 1).tolist()
        out[mask] = e
        print(C.COLLECTION, mask, json.dumps(e), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"classes": "displacement.true_class", "masks": out}, open(OUT / "counts_v2.json", "w"), indent=1)


def roc_points(score, sig, bkg, rates=(0.01, 0.001)):
    """AUC of signal against background by `score`, and the signal efficiency at the background rates given."""
    from sklearn.metrics import roc_auc_score
    s, b = score[sig], score[bkg]
    if len(s) < 5 or len(b) < 50:
        return None
    out = {"auc": float(roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(b))], np.r_[s, b])), "n_signal": int(len(s)), "n_background": int(len(b))}
    for x in rates:
        out[f"eff@{x:g}"] = float(np.mean(s > np.quantile(b, 1 - x)))
    return out


def boot_diff(s2, s1, sig, bkg, rate=0.01, n=200, seed=1):
    """v2 - v1 efficiency at the background rate, with a bootstrap over tracks (signal and background resampled; tracks
    of one event are treated as independent: the error is a lower bound)."""
    rng = np.random.default_rng(seed)
    si, bi = np.flatnonzero(sig), np.flatnonzero(bkg)
    if len(si) < 5 or len(bi) < 50:
        return None
    eff = lambda sc, i, j: np.mean(sc[i] > np.quantile(sc[j], 1 - rate))
    d = [eff(s2, i, j) - eff(s1, i, j) for i, j in ((rng.choice(si, len(si)), rng.choice(bi, len(bi))) for _ in range(n))]
    return {"diff": float(eff(s2, si, bi) - eff(s1, si, bi)), "sd": float(np.std(d))}


def calibration(T, P, Q, te):
    """Does the true radius fall in the quoted range as often as stated; are the class probabilities calibrated."""
    r, cls = T["x_r_prod"], T["x_class"]
    out = {}
    sels = {"all": te, "fitted |d0| <= 0.02 cm": te & (np.abs(T["r_d0"]) <= 0.02), "fitted |d0| > 0.02 cm": te & (np.abs(T["r_d0"]) > 0.02),
            "true radius >= 1 cm": te & (r >= R_PROMPT), "median >= 1 cm": te & (Q[:, 1] >= R_PROMPT), "K0S pions": te & T["x_k0s"]}
    for name, m in sels.items():
        if m.sum() < 20:
            continue
        out[name] = {"n": int(m.sum()), "below_low (stated 0.16)": float(np.mean(r[m] < Q[m, 0])), "below_median (stated 0.50)": float(np.mean(r[m] < Q[m, 1])),
                     "below_high (stated 0.84)": float(np.mean(r[m] < Q[m, 2])), "inside_low_high (stated 0.68)": float(np.mean((r[m] >= Q[m, 0]) & (r[m] < Q[m, 2])))}
    rel = {}
    for k in range(1, P.shape[1]):                      # P(class >= k) against the observed fraction, in bins of the probability
        p, y = P[te, k:].sum(1), cls[te] >= k
        edges = [0, 0.01, 0.03, 0.1, 0.3, 0.6, 0.9, 1.0000001]
        b = np.digitize(p, edges) - 1
        rel[f"P(class >= {k})"] = [{"bin": [edges[i], min(edges[i + 1], 1.0)], "n": int((b == i).sum()), "mean_probability": float(p[b == i].mean()),
                                    "observed_fraction": float(y[b == i].mean())} for i in range(len(edges) - 1) if (b == i).sum() >= 20]
    out["class_probabilities"] = rel
    return out


def assignment(P, cls, m, n=200, seed=2):
    """How well the class is assigned on the tracks m: the log loss (bootstrap sd over tracks), and, given that the track is
    displaced (the probabilities of classes >= 1 renormalised), how often the most probable class is the true one."""
    i = np.flatnonzero(m)
    ll = -np.log(np.clip(P[i, cls[i]], 1e-12, 1))
    rng = np.random.default_rng(seed)
    out = {"n": int(len(i)), "log_loss": float(ll.mean()), "log_loss_sd": float(np.std([ll[rng.integers(0, len(i), len(i))].mean() for _ in range(n)]))}
    d = i[cls[i] >= 1]
    if len(d):
        out["right_class_given_displaced"] = float(np.mean(P[d, 1:].argmax(1) + 1 == cls[d]))
    return out


def report(a):
    res = {}
    for mask in a.masks:
        T = load(mask)
        N = D.n_edges(mask)
        te = T["_test"] & ~T["_left_out"] & (T["x_class"] >= 0)       # files 4 and 6 are read by nothing here
        r, cls = T["x_r_prod"], T["x_class"]
        d0 = np.abs(T["r_d0"])
        P, Q = predict(mask, T)
        P1 = predict_v1(mask, T)
        P0, _ = predict(mask, T, "classes_without_surfaces")
        layers = [k for k in range(4) if mask[k] == "A"]                # the k-th active barrel layer
        bkg = te & (r < R_PROMPT)
        conf = np.zeros((N + 1, N + 1), int)
        np.add.at(conf, (cls[te], P[te].argmax(1)), 1)
        e = {"n_test_with_particle": int(te.sum()), "n_test_inside_1cm": int(bkg.sum()), "confusion_true_class_by_predicted_class": conf.tolist(),
             "calibration": calibration(T, P, Q, te), "separation_from_inside_1cm": {},
             "class_assignment": {who: {name: assignment(p, cls, m) for name, p in (("second version", P), ("without the per-surface inputs", P0))}
                                  for who, m in (("all", te), ("true class >= 1", te & (cls >= 1)), ("K0S pions, true class >= 1", te & (cls >= 1) & T["x_k0s"]))}}
        for k in range(1, N + 1):
            scores = {"|d0|": d0, "|d0|/sigma": d0 / np.maximum(T["x_sig_d0"], 1e-6),
                      "first version": P1[:, layers[k - 1] + 1:].sum(1), "second version": P[:, k:].sum(1),
                      "without the per-surface inputs": P0[:, k:].sum(1)}
            for label, signal in ((f"beyond the {k}-th crossed active surface (class >= {k})", cls >= k),
                                  (f"beyond barrel layer {layers[k - 1] + 1} (r >= {V1_EDGES[layers[k - 1]]} cm)", r >= V1_EDGES[layers[k - 1]])):
                for who, sel in (("all particles", np.ones(len(r), bool)), ("K0S pions", T["x_k0s"])):
                    sig = te & signal & sel
                    x = {s: roc_points(sc, sig, bkg) for s, sc in scores.items()}
                    x["second - first, eff@0.01"] = boot_diff(scores["second version"], scores["first version"], sig, bkg)
                    x["second - without the per-surface inputs, eff@0.01"] = boot_diff(scores["second version"], scores["without the per-surface inputs"], sig, bkg)
                    e["separation_from_inside_1cm"][f"{label}, {who}"] = x
        res[mask] = e
        print(C.COLLECTION, mask, "done", flush=True)
    json.dump({"collection": C.COLLECTION, "masks": res}, open(OUT / "report_v2.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("counts", "train", "report"))
    ap.add_argument("--masks", nargs="+", default=MASKS)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    {"counts": counts, "train": train, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
