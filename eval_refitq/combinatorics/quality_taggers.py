"""Six quality taggers x two variants, on the joint finding+fitting scenario.

WHY TWO VARIANTS. A single discriminant trained on a prompt-dominated sample
learns "displaced implies suspicious": measured on the deployed-style scores,
all three keep ~80% of clean PROMPT tracks at a 0.9 cut but only ~60% of clean
DISPLACED ones, a 20-point penalty on tracks whose every hit came from the right
particle. The Phase-2 system splits prompt and extended tracking for the same
reason. The split here is by an OBSERVABLE -- the FITTED |d0| -- so the variant
a track is scored under is decidable online; splitting on the true d0 would not
be deployable. Inside a window every track is displaced (or prompt), so |d0|
carries no discriminating power there and cannot be used as a proxy for guilt.

WHY SIX TARGETS. Contamination is one question; the accuracy of each fitted
parameter is five more, and they are not the same question. Measured over 134
composition cells, a correct IT hit multiplies sigma(d0) by 0.46 and
sigma(1/pT) by 0.98, while a correct OT hit does the reverse (1.17 and 0.51),
and the predicted per-track widths for d0, z0 and 1/pT are mutually
uncorrelated (rho 0.32, -0.03, 0.34). So one score cannot rank all five.

Each parameter target asks: is this parameter as accurate as a CLEAN track's
would be, i.e. |residual| below the 68.3rd percentile of the clean population in
the same variant window. That is scale-free, per-parameter and per-variant.

Consumes the census cache (needs --export-tracks).
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF   # noqa: E402
import kf_emulation as KF     # noqa: E402

# label -> (residual column, scale, unit) ; None = the cleanliness target
TARGETS = {
    "clean": (None, 1.0, ""),
    "invpt": ("d_kappa", 1.0, "1/GeV"),
    "phi0": ("d_phi0", 1e3, "mrad"),
    "d0": ("d_d0", 1e4, "um"),
    "cot": ("d_cot", 1e3, "1e-3"),
    "z0": ("d_z0", 1e4, "um"),
}
D0_SPLIT_CM = 0.02          # 200 um, on the FITTED d0


def sig(x, nmin=100):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < nmin:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def load(cache_dir, nev=None, census=None):
    if census:
        ds = [census]
    else:
        ds = sorted(glob.glob(f"{cache_dir}/tpcensus_*"),
                    key=lambda p: Path(p).stat().st_mtime)
        if not ds:
            raise SystemExit(f"no census under {cache_dir}")
    C = TF.load(ds[-1], nev=nev, with_tracks=True)
    if not C["tracks"]:
        raise SystemExit("census carries no track rows; rerun --export-tracks")
    T = np.concatenate([C["tracks"][i] for i in sorted(C["tracks"])])
    return T, {c: j for j, c in enumerate(C["track_cols"])}, Path(ds[-1]).name


def make_target(T, c, name, win):
    """Boolean label plus the rows it is defined on, inside one variant window."""
    if name == "clean":
        y = (T[:, c["n_wrong"]] <= 1)
        ok = win & (T[:, c["own_tpidx"]] >= 0)
        return y, ok, None
    col, scale, unit = TARGETS[name]
    r = np.abs(T[:, c[col]]) * scale
    ok = win & np.isfinite(r)
    clean = ok & (T[:, c["n_wrong"]] == 0)
    if clean.sum() < 1000:
        return None, None, None
    thr = float(np.percentile(r[clean], 68.3))
    return (r <= thr), ok, thr


def fit_one(X, y, tr, te, seed=1):
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    from sklearn.metrics import roc_auc_score
    g = GB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[tr], y[tr])
    s = g.predict_proba(X[te])[:, 1]
    return g, s, float(roc_auc_score(y[te], s))


def importance(g, X, y, rows, feats, rng, n=40000):
    """Permutation importance in AUC, on a subsample -- cheap and sufficient."""
    from sklearn.metrics import roc_auc_score
    if len(rows) > n:
        rows = rows[rng.choice(len(rows), n, replace=False)]
    Xs, ys = X[rows].copy(), y[rows]
    base = roc_auc_score(ys, g.predict_proba(Xs)[:, 1])
    out = []
    for j, f in enumerate(feats):
        keep = Xs[:, j].copy()
        Xs[:, j] = keep[rng.permutation(len(keep))]
        out.append((f, base - roc_auc_score(ys, g.predict_proba(Xs)[:, 1])))
        Xs[:, j] = keep
    return sorted(out, key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--census", default=None,
                    help="an explicit census directory, instead of the most "
                         "recent one under --cache-dir. Needed to point at a "
                         "quantised scan build rather than the full-precision "
                         "census.")
    ap.add_argument("--max-train", type=int, default=1_500_000)
    ap.add_argument("--drop", default="",
                    help="comma list of features to withhold, e.g. "
                         "d0,d0_over_sigma,sig_kf_d0 -- the d0-derived ones, "
                         "which a prompt-dominated sample lets a tagger use as "
                         "a proxy for guilt")
    ap.add_argument("--only", default="",
                    help="comma list of targets to run (default all six)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    T, c, cname = load(a.cache_dir, a.nev, a.census)
    drop = {x.strip() for x in a.drop.split(",") if x.strip()}
    feats = [f for f in KF.MVA_FEATURES if f in c and f not in drop]
    if drop:
        print(f"withheld: {', '.join(sorted(drop))}")
    want = {x.strip() for x in a.only.split(",") if x.strip()} or set(TARGETS)
    X = np.nan_to_num(T[:, [c[f] for f in feats]], posinf=0, neginf=0)
    rng = np.random.default_rng(1)
    d0f = np.abs(T[:, c["d0"]])
    WIN = {"prompt": d0f <= D0_SPLIT_CM, "displaced": d0f > D0_SPLIT_CM}
    ev = T[:, c["event"]]
    is_te = np.isin(ev, np.unique(ev)[::3])       # split by EVENT, never by track
    print(f"{cname}: {len(T):,} tracks, {len(feats)} features")
    for k, w in WIN.items():
        print(f"  window {k:<10} |fitted d0| {'<=' if k == 'prompt' else '>'} "
              f"{D0_SPLIT_CM * 1e4:.0f} um : {int(w.sum()):>10,} tracks")

    bench = None
    try:
        bench = json.load(open(Path(a.census or "").parent
                               / Path(a.census or "").name
                               / "manifest.json"))["config"].get("bench")
    except Exception:
        pass
    print(f"  alpha/beta quantisation: "
          f"{'none (full float)' if not bench else str(bench) + ' bits'}")
    J = {"census": cname, "bench": bench, "features": feats, "d0_split_um": D0_SPLIT_CM * 1e4,
         "n_tracks": int(len(T)), "targets": {}}
    models, scores = {}, {}
    for tname in [t for t in TARGETS if t in want]:
        for wname, win in WIN.items():
            y, ok, thr = make_target(T, c, tname, win)
            if y is None:
                continue
            tr = np.flatnonzero(ok & ~is_te)
            te = np.flatnonzero(ok & is_te)
            if len(tr) > a.max_train:
                tr = tr[rng.choice(len(tr), a.max_train, replace=False)]
            if len(tr) < 20000 or len(te) < 5000 or len(np.unique(y[tr])) < 2:
                print(f"{tname}/{wname}: too few rows, skipped")
                continue
            t0 = time.perf_counter()
            g, s, auc = fit_one(X, y, tr, te)
            models[(tname, wname)] = g
            scores[(tname, wname)] = (te, s)
            col, scale, unit = TARGETS[tname]
            line = (f"\n{tname:>6} / {wname:<10} AUC {auc:.4f}  "
                    f"({len(tr):,} train, {len(te):,} test, "
                    f"{100 * y[te].mean():.1f}% positive, "
                    f"{time.perf_counter() - t0:.0f}s)")
            print(line)
            rec = {"auc": round(auc, 4), "n_train": int(len(tr)),
                   "positive_fraction": round(float(y[te].mean()), 4)}
            if thr is not None:
                rec["target"] = (f"|{col}| <= {thr:.4g} ({thr * 1:.4g} {unit}, "
                                 f"the clean-track 68.3 percentile)")
                r = np.abs(T[te, c[col]]) * scale
                print(f"       keep the best-scoring fraction -> actual "
                      f"sigma({tname}) [{unit}]")
                rec["operating"] = []
                for f in (1.0, 0.5, 0.2, 0.1):
                    thr_s = np.percentile(s, 100 * (1 - f))
                    mk = s >= thr_s
                    v = sig(T[te, c[col]][mk] * scale)
                    print(f"         {100 * f:5.0f}%  {v:10.4g}")
                    rec["operating"].append({"kept": f, "sigma": v})
            else:
                rec["target"] = "n_wrong <= 1 (owner-referenced)"
                for f in (1.0, 0.5, 0.2, 0.1):
                    thr_s = np.percentile(s, 100 * (1 - f))
                    mk = s >= thr_s
                    print(f"       keep {100 * f:5.0f}%  purity "
                          f"{100 * y[te][mk].mean():5.1f}%  sigma(d0) "
                          f"{sig(T[te, c['d_d0']][mk] * 1e4):7.0f} um")
                    rec.setdefault("operating", []).append(
                        {"kept": f, "purity": round(float(y[te][mk].mean()), 4),
                         "sigma_d0_um": sig(T[te, c["d_d0"]][mk] * 1e4)})
            # DOES THE SCORE PENALISE REAL DISPLACEMENT? Keep rate at a matched
            # 20% overall, for the tracks that are genuinely good, banded by the
            # owner's TRUE impact parameter. A falling rate means the tagger is
            # rejecting displacement itself -- a flavour tagger's signal.
            thr20 = np.percentile(s, 80)
            d0true = np.abs(T[te, c["tp_d0"]]) * 1e4
            print("       keep rate for GOOD tracks at a 20% overall cut, "
                  "by true |d0|:")
            rec["fairness"] = []
            for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 1e9)):
                mk = (y[te] == 1) & (d0true >= lo) & (d0true < hi)
                if mk.sum() < 200:
                    continue
                kr = float((s[mk] >= thr20).mean())
                lab = f"{lo}-{'inf' if hi > 1e8 else int(hi)} um"
                print(f"         {lab:>12}  {int(mk.sum()):>9,} good tracks  "
                      f"kept {100 * kr:5.1f}%")
                rec["fairness"].append({"band_um": lab, "n": int(mk.sum()),
                                        "keep_rate": round(kr, 4)})
            imp = importance(g, X, y, te, feats, rng)
            rec["top_features"] = [{"feature": f, "delta_auc": round(v, 4)}
                                   for f, v in imp[:5]]
            print("       top features: " + ", ".join(
                f"{f} {v:+.4f}" for f, v in imp[:5]))
            J["targets"][f"{tname}/{wname}"] = rec

    # ---- does the prompt variant transfer to displaced tracks? -----------
    print("\nCROSS-APPLICATION: the prompt-trained model scoring DISPLACED "
          "tracks, against the displaced-trained one on the same tracks")
    # Compared at a MATCHED KEEP RATE, not at a fixed score. A fixed 0.9 cut is
    # vacuous on a hard target -- a well-calibrated model simply never emits a
    # score that high, and both columns read 0.0%. Holding the kept fraction
    # equal asks the only question that matters: of the tracks whose parameter
    # really is accurate, how many does each model keep?
    KEEP = 0.20
    print(f"  at a matched {100 * KEEP:.0f}% keep rate on displaced tracks")
    print(f"  {'target':>8}  {'own AUC':>8} {'prompt AUC':>11}"
          f" {'kept own':>9} {'kept prompt':>12}"
          f" {'good kept, own':>15} {'good kept, prompt':>18}"
          f" {'purity own':>11} {'purity prompt':>14}")
    from sklearn.metrics import roc_auc_score
    J["cross"] = {}
    for tname in [t for t in TARGETS if t in want]:
        if (tname, "displaced") not in models or (tname, "prompt") not in models:
            continue
        y, ok, _ = make_target(T, c, tname, WIN["displaced"])
        te = np.flatnonzero(ok & is_te)
        res = {}
        for lab, key in (("own", (tname, "displaced")), ("prompt", (tname, "prompt"))):
            sc = models[key].predict_proba(X[te])[:, 1]
            thr = np.percentile(sc, 100 * (1 - KEEP))
            mk = sc >= thr
            res[lab] = {"auc": float(roc_auc_score(y[te], sc)),
                        # the REALISED fraction: a near-binary score has ties at
                        # the percentile, so the match can fail and must show it
                        "kept": float(mk.mean()),
                        # of all the genuinely good tracks, the share retained
                        "good_kept": float(y[te][mk].sum() / max(y[te].sum(), 1)),
                        "purity": float(y[te][mk].mean())}
        print(f"  {tname:>8}  {res['own']['auc']:8.4f} {res['prompt']['auc']:11.4f}"
              f" {100 * res['own']['kept']:8.1f}% {100 * res['prompt']['kept']:11.1f}%"
              f" {100 * res['own']['good_kept']:14.1f}%"
              f" {100 * res['prompt']['good_kept']:17.1f}%"
              f" {100 * res['own']['purity']:10.1f}%"
              f" {100 * res['prompt']['purity']:13.1f}%")
        J["cross"][tname] = {"keep_rate": KEEP,
                             **{f"{k}_{m}": round(v[m], 4)
                                for k, v in res.items() for m in v}}

    # ---- are six taggers six DIFFERENT taggers? -------------------------
    for wname in WIN:
        ks = [t for t in TARGETS if t in want and (t, wname) in scores]
        if len(ks) < 2:
            continue
        print(f"\nrank correlation between the {len(ks)} scores, {wname} window "
              f"(1.00 would mean one tagger suffices)")
        print("        " + "".join(f"{k:>9}" for k in ks))
        J.setdefault("rho", {})[wname] = {}
        base_te = scores[(ks[0], wname)][0]
        for i, ka in enumerate(ks):
            row = []
            for kb in ks:
                ta, sa = scores[(ka, wname)]
                tb, sb = scores[(kb, wname)]
                n = min(len(sa), len(sb))
                ra = np.argsort(np.argsort(sa[:n]))
                rb = np.argsort(np.argsort(sb[:n]))
                rho = float(np.corrcoef(ra, rb)[0, 1])
                row.append(rho)
                J["rho"][wname][f"{ka}|{kb}"] = round(rho, 3)
            print(f"  {ka:>6}" + "".join(f"{v:9.2f}" for v in row))
    if a.out:
        json.dump(J, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
