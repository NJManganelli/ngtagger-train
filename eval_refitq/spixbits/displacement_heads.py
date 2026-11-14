"""Per-track DISPLACEMENT heads (physicist, 2026-11-14: "a 'displaced track' classifier and even regression could be
outputs" of the track heads; first-line version, to be passed into a track-pair algorithm downstream): from the
inputs of the existing track-quality heads, where was the track's particle produced?

  [TAGGER_SP_COLLECTION=prompt TAGGER_SP_OUT=/Volumes/WDMac/smartpixels_scratch/tagger_v5_prompt/heads_input] \\
      python displacement_heads.py counts | train [--masks ...] [--trials 20] | report
  (without the two variables: the extended collection, the tagger study's own caches under tagger_v5)

Sample, split and trial policy: those of the heads retrained per detector scenario (retrain_heads.py): today's
("implemented now") tracks of canon_v5 TTbar files 1-3, 5, 7-10 and seven H->tautau files (1,150 events; files 4 and
6 left out), the tagger study's split by event identity, 20 trials (trial 0 deterministic; 1-19 with row and column
subsampling 0.8, seed = trial), the best trial by validation loss, a non-finite one discarded. XGBoost hist, depth 3,
100 rounds, learning rate 0.1, one model per window as the heads (SmartPixels detector scenarios: prompt |d0| <= 0.02
cm and displaced; IIII: one window, the 13 outer-tracker inputs).
Target: the transverse production radius r of the track's tracking particle (hypot(tp_vx, tp_vy), cm; present on
both track collections). A track without a particle has no radius: it is left out of the training and of the
figures (fakes stay with the existing "clean" head); at application it gets outputs like any track.
Outputs:
  band    five classes by r [cm]: < 2.85 (inside barrel layer 1), 2.85-5.95, 5.95-10.3, 10.3-14.5, >= 14.5 (softmax);
          used as the four cumulative probabilities P(r >= 2.85), P(r >= 5.95), P(r >= 10.3), P(r >= 14.5)
  radius  regression of ln(max(r, 0.1 cm)), squared error (the floor: below 1 mm every track is "prompt")
Input variants:
  heads   exactly the heads' inputs (38, or 13 for IIII; no truth column; the heads withhold d0 itself and see it
          only through the window)
  d0      those plus the fitted |d0| and |d0| / sigma(d0) of the track (quantities the tagger already reads): the
          variant to compare fairly with a cut on d0, which the first variant cannot be
Truth for the figures: K0S pions = a pi+ and a pi- tracking particle with a bit-identical production vertex and
nothing else charged-pion at it, invariant mass within 1 MeV of the K0S mass (l1-flow-bphys-study/COUNTS.md).
Outputs under ~/smartpixels/scratch_local/spixbits/displacement_heads/<collection>/.
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
sys.path.insert(0, str(HERE.parent / "tagger_sp"))
os.environ.setdefault("TAGGER_SP_OUT", "/Volumes/WDMac/smartpixels_scratch/tagger_v5")
import common as C          # noqa: E402  (tagger_sp/common.py)
import joint_heads as J     # noqa: E402
import retrain_heads as RH  # noqa: E402

OUT = Path.home() / "smartpixels/scratch_local/spixbits/displacement_heads" / C.COLLECTION
EDGES = np.array([2.85, 5.95, 10.3, 14.5])                # cm: barrel layers 1-4
BANDS = ["< 2.85", "2.85-5.95", "5.95-10.3", "10.3-14.5", ">= 14.5"]
R_FLOOR, R_PROMPT = 0.1, 1.0                              # cm: the regression's floor; "produced inside 1 cm"
XGB = {"tree_method": "hist", "max_depth": 3, "eta": 0.1, "nthread": 4}
ROUNDS = 100
VARIANTS = ("heads", "d0")
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


def load(mask):
    """The heads' tracks of the mask (retrain_heads.load: split flags, files 4 and 6 marked) with the displacement truth."""
    T, man = RH.load(mask)
    ex = [dict(np.load(extra(s, mask, n, p))) for s, n, p in man["files"]]
    for k in ("r_prod", "pdg", "k0s", "sig_d0"):
        T["x_" + k] = np.concatenate([e[k] for e in ex])
    assert len(T["x_r_prod"]) == len(T["b_pt"])
    return T


def inputs(T, coll, variant):
    X = J.features(T, coll)
    if variant == "d0":
        p = "r_" if coll == "refit" else "b_"
        d0 = np.abs(T[p + "d0"])
        X = np.column_stack([X, d0, d0 / np.maximum(T["x_sig_d0"], 1e-6)])
    return X.astype(np.float32)


def band_of(r):
    return np.searchsorted(EDGES, r, side="right")


def fit_trial(X, y, tr, va, kind, t):
    import xgboost as xgb
    par = dict(XGB, seed=t, **({"subsample": 0.8, "colsample_bynode": 0.8} if t else {}),
               **({"objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss"} if kind == "band"
                  else {"objective": "reg:squarederror", "eval_metric": "rmse"}))
    b = xgb.train(par, xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=ROUNDS)
    p = b.predict(xgb.DMatrix(X[va]))
    if kind == "band":
        loss = float(-np.mean(np.log(np.clip(p[np.arange(len(va)), y[va].astype(int)], 1e-12, 1))))
    else:
        loss = float(np.sqrt(np.mean((p - y[va]) ** 2)))
    return b, loss


def train(a):
    from ngtagger.train import trackmva as tm
    rec_path = OUT / "trials.json"
    rec = json.load(open(rec_path)) if rec_path.exists() else {}
    for mask in a.masks:
        coll = J.coll_of(mask)
        T = load(mask)
        ok = np.isfinite(T["x_r_prod"])
        y = {"band": band_of(T["x_r_prod"]).astype(np.float32), "radius": np.log(np.maximum(T["x_r_prod"], R_FLOOR)).astype(np.float32)}
        for variant in VARIANTS:
            X = inputs(T, coll, variant)
            for wname, w in tm.windows(T, coll).items():
                tr = np.flatnonzero(w & ok & ~T["_test"] & ~T["_val"] & ~T["_left_out"])
                va = np.flatnonzero(w & ok & T["_val"] & ~T["_left_out"])
                for kind in ("band", "radius"):
                    key = f"{mask}/{variant}/{wname}/{kind}"
                    d = OUT / "models" / mask / variant / wname
                    d.mkdir(parents=True, exist_ok=True)
                    if key in rec and (d / f"{kind}.json").exists():
                        continue
                    t0, best, losses = time.time(), None, {}
                    for t in range(a.trials):
                        b, loss = fit_trial(X, y[kind], tr, va, kind, t)
                        losses[t] = loss if np.isfinite(loss) else None
                        if np.isfinite(loss) and (best is None or loss < best[1]):
                            best = (t, loss, b)
                    best[2].save_model(str(d / f"{kind}.json"))
                    good = [v for v in losses.values() if v is not None]
                    rec[key] = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_inputs": int(X.shape[1]), "best_trial": best[0],
                                "best_val_loss": best[1], "val_loss_trials": losses, "n_failed": a.trials - len(good),
                                "val_loss_sd": float(np.std(good)), "band_counts_train": np.bincount(y["band"][tr].astype(int), minlength=5).tolist()}
                    json.dump(rec, open(rec_path, "w"), indent=1)
                    print(f"{C.COLLECTION} {key}: best trial {best[0]} loss {best[1]:.4f} (sd {np.std(good):.4f}) {time.time() - t0:.0f} s", flush=True)


def predict(mask, variant, T, coll):
    """(P (n, 5) band probabilities, ln r) from the stored models, routed by window as the heads."""
    import xgboost as xgb
    from ngtagger.train import trackmva as tm
    X = inputs(T, coll, variant)
    P, L = np.full((len(X), 5), np.nan), np.full(len(X), np.nan)
    for wname, w in tm.windows(T, coll).items():
        i = np.flatnonzero(w)
        for kind in ("band", "radius"):
            b = xgb.Booster()
            b.load_model(str(OUT / "models" / mask / variant / wname / f"{kind}.json"))
            p = b.predict(xgb.DMatrix(X[i]))
            if kind == "band":
                P[i] = p
            else:
                L[i] = p
    return P, L


def counts(a):
    """Step 1 (a): matched head-training tracks per production-radius band."""
    out = {}
    for mask in a.masks:
        T = load(mask)
        use = ~T["_left_out"]
        ok = use & np.isfinite(T["x_r_prod"])
        b = band_of(T["x_r_prod"])
        e = {"tracks": int(use.sum()), "with_a_particle": int(ok.sum()), "particle_has_radius": bool(np.isfinite(T["x_r_prod"][ok]).all()),
             "inside_1cm": int((ok & (T["x_r_prod"] < R_PROMPT)).sum())}
        for name, sel in (("all", ok), ("train", ok & ~T["_test"] & ~T["_val"]), ("validation", ok & T["_val"]), ("test", ok & T["_test"]),
                          ("k0s_pions_all", ok & T["x_k0s"]), ("k0s_pions_test", ok & T["x_k0s"] & T["_test"])):
            e[name] = np.bincount(b[sel], minlength=5).tolist()
        out[mask] = e
        print(C.COLLECTION, mask, json.dumps(e), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"bands_cm": BANDS, "masks": out}, open(OUT / "counts.json", "w"), indent=1)


def roc_points(score, sig, bkg, rates=(0.01, 0.001)):
    """AUC of signal against background by `score`, and the signal efficiency at the background rates given."""
    from sklearn.metrics import roc_auc_score
    s, b = score[sig], score[bkg]
    if len(s) < 5 or len(b) < 50:
        return None
    out = {"auc": float(roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(b))], np.r_[s, b])), "n_signal": int(len(s)), "n_background": int(len(b))}
    for x in rates:
        thr = np.quantile(b, 1 - x)
        out[f"eff@{x:g}"] = float(np.mean(s > thr))
    return out


def report(a):
    res = {}
    for mask in a.masks:
        coll = J.coll_of(mask)
        T = load(mask)
        te = T["_test"] & ~T["_left_out"] & np.isfinite(T["x_r_prod"])     # files 4 and 6 are read by nothing here
        r = T["x_r_prod"]
        p = "r_" if coll == "refit" else "b_"
        d0 = np.abs(T[p + "d0"])
        sig_d0 = d0 / np.maximum(T["x_sig_d0"], 1e-6)
        bkg = te & (r < R_PROMPT)
        e = {"n_test_with_particle": int(te.sum()), "n_test_inside_1cm": int(bkg.sum()), "variants": {}}
        scores = {"|d0|": {k: d0 for k in (1, 2, 3)}, "|d0|/sigma": {k: sig_d0 for k in (1, 2, 3)}}
        for variant in VARIANTS:
            P, L = predict(mask, variant, T, coll)
            scores[variant] = {k: P[:, k + 1:].sum(1) for k in (0, 1, 2, 3)}
            band = band_of(r)
            conf = np.zeros((5, 5), int)
            np.add.at(conf, (band[te], P[te].argmax(1)), 1)
            lr = np.log(np.maximum(r, R_FLOOR))
            reg = {}
            for name, sel in (("all", te), ("r >= 1 cm", te & (r >= 1)), ("r >= 5.95 cm", te & (r >= EDGES[1])), ("K0S pions, r >= 1 cm", te & T["x_k0s"] & (r >= 1))):
                if sel.sum() >= 20:
                    d = (L - lr)[sel]
                    reg[name] = {"n": int(sel.sum()), "median": float(np.median(d)), "w68_ln_r": float(np.quantile(np.abs(d - np.median(d)), 0.6827)),
                                 "w68_of_target": float(np.quantile(np.abs(lr[sel] - np.median(lr[sel])), 0.6827))}
            e["variants"][variant] = {"confusion_true_band_by_predicted_band": conf.tolist(), "regression": reg}
        e["separation_from_inside_1cm"] = {}
        for label, k, lo in (("produced beyond barrel layer 1 (r >= 2.85 cm)", 0, EDGES[0]), ("beyond layer 2 (r >= 5.95 cm)", 1, EDGES[1]),
                             ("beyond layer 3 (r >= 10.3 cm)", 2, EDGES[2])):
            for who, extra_sel in (("all particles", np.ones(len(r), bool)), ("K0S pions", T["x_k0s"])):
                sig = te & (r >= lo) & extra_sel
                e["separation_from_inside_1cm"][f"{label}, {who}"] = {
                    s: roc_points(sc[k if s in VARIANTS else 1], sig, bkg) for s, sc in scores.items()}
        res[mask] = e
        print(C.COLLECTION, mask, "done", flush=True)
    json.dump({"collection": C.COLLECTION, "bands_cm": BANDS, "masks": res}, open(OUT / "report.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("counts", "train", "report"))
    ap.add_argument("--masks", nargs="+", default=["IIII", "AAAA", "AAAI", "IAAA"])
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    {"counts": counts, "train": train, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
