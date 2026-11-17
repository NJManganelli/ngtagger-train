"""Displacement heads per QUANTIZATION SCENARIO (physicist, 2026-11-14): the second version's class model retrained on
the tracks each quantization scenario would deliver, and today's model applied to them.

  python displacement_heads_q.py train | report  [--collections prompt extended] [--masks ...] [--quant ...] [--trials 20]

Tracks: the Python replay of each quantization scenario with its per-surface record (ext_scan.py --version p3 for the
prompt collection, x3 for the extended one): "implemented now", network float, 6/6/0/4, 6/6/2/2, 5/5/3/3, 4/4/4/4 on
the enlarged sample (TTbar files 1-3, 5, 7-10 and seven H->tautau files; 4 and 6 never read), the tagger study's split
by event identity (the enlarged heads sample's event_split.npz). The same outer-tracker tracks in every quantization
scenario of a detector scenario, so differences between scenarios are paired.
Model: as displacement_heads.py (XGBoost depth 3, 100 rounds, 20 trials, best by validation loss, one softmax model per
window of the scenario's own fitted |d0| at 0.02 cm); classes and range from displacement.py; target = the particle's
class on its own true trajectory (truth is the label only).
Inputs: the 38 inputs of the track-quality heads as the scenario delivers them, its |d0| and |d0| / sigma(d0), and the
per-surface inputs in the list the replay can give ("magnitudes": absolute pulls in x, y and of both angles, hit or
not, candidates in the window; no signed residuals). A second model without the per-surface inputs ("without
surfaces"): the only one a tagger run through the parametrization can be given.
Per quantization scenario q the report compares, on the test tracks:
  own        the model trained on q, applied to q
  today's    the model trained on "implemented now" (the replay's), applied to q's inputs
  reference  the model trained on "implemented now", applied to "implemented now"
with a bootstrap over events paired between q and the reference (200 replicates).
Outputs under ~/smartpixels/scratch_local/spixbits/displacement_heads_q/<collection>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import displacement as D             # noqa: E402
import displacement_heads as DH      # noqa: E402  (the trial fit; tagger_sp modules only, never the replay's)

LOCAL = Path.home() / "smartpixels/scratch_local/spixbits"
SCAN = {"prompt": Path("/Volumes/WDMac/smartpixels_scratch/spixbits/prompt_tracks_v3"),
        "extended": Path("/Volumes/WDMac/smartpixels_scratch/spixbits/ext_v3")}
SPLIT = LOCAL / "heads_retrained_per_quantization_prompt_enlarged/event_split.npz"
FILES = (5, 7, 8, 9, 10, 1, 2, 3, 101, 102, 103, 104, 105, 107, 108)        # H->tautau file n = 100 + n
REFERENCE = "implemented_now"
QUANT = [REFERENCE, "network_float", "alloc_6604", "alloc_6622", "alloc_5533", "alloc_4444"]
MODELS = {"classes": True, "classes_without_surfaces": False}                # model name: with the per-surface inputs
D0_SPLIT = 0.02
N_BOOT = 200
MISSING = -999.0


def out_dir(coll):
    return LOCAL / "displacement_heads_q" / coll


def event_key(run, lumi, event):
    return (np.asarray(run, np.int64) * 1000003 + np.asarray(lumi, np.int64)) * 1000003 + np.asarray(event, np.int64)


def load(coll, mask, quant):
    """One quantization scenario's tracks: the two input matrices, the label, the split flags, the event key."""
    sp = np.load(SPLIT)
    key_s = event_key(*sp["evid"].T)
    o = np.argsort(key_s)
    parts = []
    for f in FILES:
        t = np.load(SCAN[coll] / "truth" / f"{mask}_f{f:02d}_truth.npz")
        u = np.load(SCAN[coll] / "units" / f"{mask}_f{f:02d}_00.npz")
        g = lambda k: u[f"{quant}__{k}"]
        a, surf, acc, pull = g("a").astype(np.float64), g("ps_surface"), g("ps_acc"), g("ps_pull")
        ti, j = np.nonzero(surf > 0)
        hits = {"accepted": acc[ti, j].astype(np.float64), "window_mult": g("ps_ncand")[ti, j].astype(np.float64),
                **{f"abs_pull_{n}": pull[ti, j, i].astype(np.float64) for i, n in enumerate(("x", "y", "alpha", "beta"))}}
        slots, radii = D.slot_inputs(mask, a[:, 2], a[:, 3], ti, surf[ti, j], hits, "magnitudes")
        X38 = np.column_stack([t["ot_inputs"], g("inputs")]).astype(np.float64)
        X38[~np.isfinite(X38)] = MISSING
        d0 = np.abs(a[:, 4])
        key = event_key(t["run"], t["lumi"], t["event"])
        i = o[np.clip(np.searchsorted(key_s[o], key), 0, len(o) - 1)]
        assert (key_s[i] == key).all(), f"{mask} file {f}: an event is not in the split"
        r = np.where(t["matched"], t["tp_r_prod"], np.nan).astype(np.float64)
        cls = np.where(np.isfinite(r), D.true_class(np.nan_to_num(r), np.sinh(t["tp_eta"].astype(np.float64)), t["tp_z0"].astype(np.float64), mask), -1)
        parts.append({"X": D.assemble(X38, d0, np.sqrt(np.abs(g("cov_diag")[:, 4].astype(np.float64))), slots), "radii": radii, "d0": d0,
                      "r": r, "cls": cls, "k0s": t["tp_k0s"], "test": sp["test"][i], "val": sp["val"][i], "key": key})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def columns(T, with_surfaces):
    return T["X"] if with_surfaces else T["X"][:, :-len(D.input_names("magnitudes"))]


def windows(T):
    return {"prompt": T["d0"] <= D0_SPLIT, "displaced": T["d0"] > D0_SPLIT}


def model_dir(coll, mask, quant, window):
    return out_dir(coll) / "models" / mask / quant / window


def train(a):
    for coll in a.collections:
        rec_path = out_dir(coll) / "trials.json"
        rec = json.load(open(rec_path)) if rec_path.exists() else {}
        for mask in a.masks:
            for quant in a.quant:
                T = None
                for wname in ("prompt", "displaced"):
                    for name, with_surfaces in MODELS.items():
                        key = f"{mask}/{quant}/{wname}/{name}"
                        d = model_dir(coll, mask, quant, wname)
                        if key in rec and (d / f"{name}.json").exists() and (d / "shape.json").exists():
                            continue
                        if T is None:
                            T = load(coll, mask, quant)
                        d.mkdir(parents=True, exist_ok=True)
                        w, ok, y, n_class = windows(T)[wname], T["cls"] >= 0, T["cls"], D.n_edges(mask) + 1
                        X = columns(T, with_surfaces)
                        tr, va = np.flatnonzero(w & ok & ~T["test"] & ~T["val"]), np.flatnonzero(w & ok & T["val"])
                        t0, best, losses = time.time(), None, {}
                        for t in range(a.trials):
                            b, loss = DH.fit_trial(X, y, tr, va, n_class, t)
                            losses[t] = loss if np.isfinite(loss) else None
                            if np.isfinite(loss) and (best is None or loss < best[1]):
                                best = (t, loss, b)
                        best[2].save_model(str(d / f"{name}.json"))
                        shape = D.fit_shape(T["r"][tr], y[tr], T["radii"][tr], mask)
                        json.dump({"what": "displacement.fit_shape on the training tracks of this window", "r_max_cm": D.R_MAX,
                                   "grid": D.SHAPE_GRID.tolist(), "position_quantiles_by_class": shape.tolist()}, open(d / "shape.json", "w"))
                        good = [v for v in losses.values() if v is not None]
                        rec[key] = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_inputs": int(X.shape[1]), "best_trial": best[0],
                                    "best_val_loss": best[1], "val_loss_sd": float(np.std(good)), "n_failed": a.trials - len(good),
                                    "class_counts_train": np.bincount(y[tr], minlength=n_class).tolist()}
                        json.dump(rec, open(rec_path, "w"), indent=1)
                        print(f"{coll} {key}: best trial {best[0]} loss {best[1]:.4f} (sd {np.std(good):.4f}) {time.time() - t0:.0f} s", flush=True)


def predict(coll, mask, model_quant, T, name):
    """Class probabilities of the model trained on `model_quant`, on the tracks T (routed by T's own window)."""
    import xgboost as xgb
    X = columns(T, MODELS[name])
    P = np.full((len(X), D.n_edges(mask) + 1), np.nan)
    for wname, w in windows(T).items():
        b = xgb.Booster()
        b.load_model(str(model_dir(coll, mask, model_quant, wname) / f"{name}.json"))
        P[w] = b.predict(xgb.DMatrix(X[w]))
    return P


def metrics(P, T, sel, w):
    """The figures of one model on the tracks `sel` with track weights w (1, or a bootstrap replicate's event counts):
    efficiency at 1 % of the tracks produced inside 1 cm for "beyond the k-th crossed surface" (all particles, K0S
    pions); among the tracks of class >= 1 the log loss and how often the most probable displaced class is the true one."""
    out = {}
    cls, r = T["cls"], T["r"]
    bkg = sel & (r < DH.R_PROMPT) & (w > 0)
    for k in range(1, P.shape[1]):
        s = P[:, k:].sum(1)
        o = np.argsort(s[bkg])
        cw = np.cumsum(w[bkg][o])
        thr = s[bkg][o][min(np.searchsorted(cw, 0.99 * cw[-1]), len(o) - 1)]
        for who, m in (("all", sel & (cls >= k)), ("k0s", sel & (cls >= k) & T["k0s"])):
            out[f"eff@1%_{who}_k{k}"] = float(np.sum(w[m] * (s[m] > thr)) / max(np.sum(w[m]), 1e-12))
    d = sel & (cls >= 1)
    out["log_loss_displaced"] = float(np.sum(w[d] * -np.log(np.clip(P[d, cls[d]], 1e-12, 1))) / np.sum(w[d]))
    out["right_class_displaced"] = float(np.sum(w[d] * (P[d, 1:].argmax(1) + 1 == cls[d])) / np.sum(w[d]))
    return out


def report(a):
    for coll in a.collections:
        res = {}
        for mask in a.masks:
            T = {q: load(coll, mask, q) for q in a.quant}
            ref = T[REFERENCE]
            for q in a.quant:
                assert np.array_equal(T[q]["key"], ref["key"]) and np.array_equal(T[q]["cls"], ref["cls"]), "the scenarios' tracks are not the same"
            sel = ref["test"] & (ref["cls"] >= 0)
            ev, inv = np.unique(ref["key"][sel], return_inverse=True)
            rng = np.random.default_rng(3)
            W = [np.ones(len(ref["key"]))]
            for _ in range(N_BOOT):
                w = np.zeros(len(ref["key"]))
                w[sel] = np.bincount(rng.integers(0, len(ev), len(ev)), minlength=len(ev))[inv]
                W.append(w)
            out = {"n_test_with_particle": int(sel.sum()), "n_test_events": int(len(ev)), "n_test_inside_1cm": int((sel & (ref["r"] < DH.R_PROMPT)).sum()),
                   "class_counts_test": np.bincount(ref["cls"][sel], minlength=D.n_edges(mask) + 1).tolist(), "quantization_scenarios": {}}
            for name in MODELS:
                M = {}       # (which, q) -> list over replicates of the metric dict
                for q in a.quant:
                    for which, model_quant in (("own", q), ("today's", REFERENCE)):
                        if which == "own" or q != REFERENCE:
                            P = predict(coll, mask, model_quant, T[q], name)
                            M[(which, q)] = [metrics(P, T[q], sel, w) for w in W]
                R = M[("own", REFERENCE)]
                for (which, q), m in M.items():
                    e = {}
                    for k in m[0]:
                        v, r0 = np.array([x[k] for x in m]), np.array([x[k] for x in R])
                        e[k] = {"value": float(v[0]), "sd": float(np.std(v[1:])), "minus_reference": float(v[0] - r0[0]),
                                "minus_reference_sd": float(np.std(v[1:] - r0[1:]))}
                    out["quantization_scenarios"].setdefault(q, {}).setdefault(name, {})[which] = e
            # what the per-surface inputs add within each quantization scenario (paired over events)
            for q in a.quant:
                P = {name: predict(coll, mask, q, T[q], name) for name in MODELS}
                m = {name: [metrics(P[name], T[q], sel, w) for w in W[:51]] for name in MODELS}
                out["quantization_scenarios"][q]["with_minus_without_surfaces"] = {
                    k: {"diff": float(m["classes"][0][k] - m["classes_without_surfaces"][0][k]),
                        "sd": float(np.std([x[k] - y[k] for x, y in zip(m["classes"][1:], m["classes_without_surfaces"][1:])]))} for k in m["classes"][0]}
            res[mask] = out
            print(coll, mask, "done", flush=True)
        json.dump({"collection": coll, "reference": REFERENCE, "bootstrap": "over events, paired with the reference", "masks": res},
                  open(out_dir(coll) / "report.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "report"))
    ap.add_argument("--collections", nargs="+", default=["prompt", "extended"])
    ap.add_argument("--masks", nargs="+", default=DH.MASKS)
    ap.add_argument("--quant", nargs="+", default=QUANT)
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    {"train": train, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
