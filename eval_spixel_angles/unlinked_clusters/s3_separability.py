"""Step 3: can unlinked clusters be told apart from TP-linked ones?

 a. single-feature ROC AUC per layer (non-angle features; plus the angle-presence leak)
 b. HistGradientBoosting, event-level 50/50 split: non-angle features; WITH angles (synthetic
    model); WITH angles + origin-compatibility features. AUC per layer, evaluated also against
    linked sub-populations, permutation importances.
 c. autoencoder trained on TP-linked clusters only (non-angle features), reconstruction error as
    an anomaly score for unlinked clusters.

Positive class = unlinked (tpIdx = -1) throughout; AUC 0.5 = indistinguishable.
Writes results/separability.json, results/sep_*.md, figs/f3_*.png.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

import _common as c

RES = c.HERE / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260924)
PT_MIN = 2.0

NA = ["layer", "size", "sizeX", "sizeY", "charge", "qPerPix", "absZ", "localX", "localY",
      "globalR", "sigX", "sigY"]
WA = NA + ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta"]
WO = WA + ["z0imp", "dphiNorm"]
SINGLE = ["size", "sizeX", "sizeY", "charge", "qPerPix", "absZ"]


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    r = d["globalR"].to_numpy()
    bend = np.arcsin(np.minimum(1.0, r * c.C_CURV / (2 * PT_MIN)))
    d["dphiNorm"] = d["dphiDir"] / bend
    for k in ("localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "z0imp", "dphiNorm"):
        d[k] = d[k].where(d[k] > -900, np.nan)   # HGB handles NaN natively
    ev = d["evid"].unique()
    test_ev = set(RNG.choice(ev, size=len(ev) // 2, replace=False))
    d["is_test"] = d["evid"].isin(test_ev)
    return d


def subgroups(t: pd.DataFrame) -> dict:
    lk = t["unlinked"] == 0
    return {
        "all linked": lk,
        "linked pT>=2": lk & (t["tpPt"] >= 2),
        "linked pT<0.3": lk & (t["tpPt"] < 0.3),
        "linked secondary vr>1cm": lk & (t["tpVr"] > 1),
        "linked shared frac<0.9": lk & (t["tpChargeFrac"] < 0.9),
    }


def single_feature(d):
    rows = []
    for L in (1, 2, 3, 4, "all"):
        m = np.ones(len(d), bool) if L == "all" else (d["layer"] == L).to_numpy()
        y = d.loc[m, "unlinked"].to_numpy()
        row = {"layer": L}
        for f in SINGLE:
            row[f] = auc(y, d.loc[m, f].to_numpy())
        # the leak that existed with no noiseSet: "has no angle" as the score
        row["noAngle (no noiseSet)"] = auc(y, 1 - d.loc[m, "hasAlpha_T"].to_numpy())
        row["noBeta (noise v6)"] = auc(y, 1 - d.loc[m, "hasBeta"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows).set_index("layer")


def fit_hgb(d, feats, name, max_train=2_000_000):
    tr = d[~d["is_test"]]
    if len(tr) > max_train:
        tr = tr.sample(max_train, random_state=1)
    te = d[d["is_test"]]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                                         early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=0)
    t0 = time.time()
    clf.fit(tr[feats], tr["unlinked"])
    s = clf.predict_proba(te[feats])[:, 1]
    out = {"model": name, "n_iter": int(clf.n_iter_), "fit_s": round(time.time() - t0, 1)}
    for L in (1, 2, 3, 4, "all"):
        m = np.ones(len(te), bool) if L == "all" else (te["layer"] == L).to_numpy()
        out[f"AUC L{L}"] = auc(te["unlinked"].to_numpy()[m], s[m])
    # evaluated against each linked sub-population (unlinked vs that population only)
    u = (te["unlinked"] == 1).to_numpy()
    sub = {}
    for gname, gm in subgroups(te).items():
        m = u | gm.to_numpy()
        sub[gname] = auc(te["unlinked"].to_numpy()[m], s[m])
    # working points
    fpr, tpr, thr = roc_curve(te["unlinked"], s)
    wp = {}
    for target in (0.5, 0.8, 0.9):
        i = np.searchsorted(tpr, target)
        wp[f"linked kept (1-FPR) at {int(target*100)}% unlinked rejected"] = float(1 - fpr[i])
    for f_ in (0.01, 0.05):
        i = np.searchsorted(fpr, f_)
        wp[f"unlinked rejected at {int(f_*100)}% linked loss"] = float(tpr[i])
    ptm = (te["tpPt"] >= 2).to_numpy()
    for f_ in (0.01, 0.05):
        thr_p = np.quantile(s[ptm], 1 - f_)
        wp[f"unlinked rejected at {int(f_*100)}% loss of pT>=2 linked"] = float(np.mean(s[u] > thr_p))
    # permutation importance on a test subsample, AUC drop
    sub_te = te.sample(min(300_000, len(te)), random_state=2)
    pi = permutation_importance(clf, sub_te[feats], sub_te["unlinked"], scoring="roc_auc",
                                n_repeats=3, random_state=0, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    return out, sub, wp, imp, s, te


def autoencoder(d):
    import torch
    torch.manual_seed(0)
    feats_log = ["size", "sizeX", "sizeY", "charge", "qPerPix"]
    feats_lin = ["absZ", "localX", "localY", "sigX", "sigY"]

    def matrix(x):
        cols = [np.log(np.maximum(x[f].to_numpy(np.float64), 1.0)) for f in feats_log]
        cols += [x[f].to_numpy(np.float64) for f in feats_lin]
        # radius relative to its layer mean (so the layer one-hot carries the gross radius)
        cols += [x["globalR"].to_numpy(np.float64) - x.groupby("layer")["globalR"].transform("mean").to_numpy()]
        oh = np.stack([(x["layer"].to_numpy() == L).astype(np.float64) for L in (1, 2, 3, 4)], 1)
        return np.concatenate([np.stack(cols, 1), oh], 1)

    tr = d[(~d["is_test"]) & (d["unlinked"] == 0)]
    tr = tr.sample(min(1_500_000, len(tr)), random_state=3)
    te = d[d["is_test"]]
    Xtr, Xte = matrix(tr), matrix(te)
    nc = len(feats_log) + len(feats_lin) + 1
    mu, sd = Xtr[:, :nc].mean(0), Xtr[:, :nc].std(0) + 1e-9
    Xtr[:, :nc] = (Xtr[:, :nc] - mu) / sd
    Xte[:, :nc] = (Xte[:, :nc] - mu) / sd
    nin = Xtr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(nin, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
        torch.nn.Linear(4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, nin))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    losses = []
    for ep in range(8):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 4096):
            b = xt[perm[i:i + 4096]]
            loss = ((net(b) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        losses.append(tot / len(xt))
        for g in opt.param_groups:
            g["lr"] *= 0.7
    with torch.no_grad():
        xe = torch.tensor(Xte, dtype=torch.float32)
        err = np.concatenate([((net(xe[i:i + 200000]) - xe[i:i + 200000]) ** 2)[:, :nc].mean(1).numpy()
                              for i in range(0, len(xe), 200000)])
    out = {"epochs_loss": losses, "bottleneck": 4, "n_train_linked": int(len(tr))}
    y = te["unlinked"].to_numpy()
    for L in (1, 2, 3, 4, "all"):
        m = np.ones(len(te), bool) if L == "all" else (te["layer"] == L).to_numpy()
        out[f"AUC L{L}"] = auc(y[m], err[m])
    u = y == 1
    out["vs subgroups"] = {g: auc(y[u | gm.to_numpy()], err[u | gm.to_numpy()]) for g, gm in subgroups(te).items()}
    out["median err linked"] = float(np.median(err[~u]))
    out["median err unlinked"] = float(np.median(err[u]))
    out["frac unlinked above linked q95"] = float(np.mean(err[u] > np.quantile(err[~u], 0.95)))
    out["frac unlinked above linked q99"] = float(np.mean(err[u] > np.quantile(err[~u], 0.99)))
    return out, err, te


def main():
    d = prepare(c.load())
    res = {}
    sf = single_feature(d)
    res["single_feature_auc"] = sf.reset_index().astype({"layer": str}).to_dict(orient="list")
    (RES / "sep_single_feature.md").write_text(
        "ROC AUC, positive = unlinked, score = feature value (AUC < 0.5 means unlinked sit LOWER)\n\n" + c.md(sf, "{:.3f}"))
    print(c.md(sf, "{:.3f}"))

    hgb_rows, sub_rows, wp_rows, imps, scores = [], {}, {}, {}, {}
    for feats, name in ((NA, "HGB non-angle"), (WA, "HGB + angles (SYNTHETIC)"),
                        (WO, "HGB + angles + origin-compat (SYNTHETIC)")):
        out, sub, wp, imp, s, te = fit_hgb(d, feats, name)
        hgb_rows.append(out); sub_rows[name] = sub; wp_rows[name] = wp; imps[name] = imp; scores[name] = s
        print(out, sub, wp, imp.head(8).to_dict(), flush=True)
    hgb = pd.DataFrame(hgb_rows).set_index("model")
    subt = pd.DataFrame(sub_rows).T
    wpt = pd.DataFrame(wp_rows).T
    impt = pd.DataFrame(imps)
    (RES / "sep_hgb.md").write_text(
        "Per-layer test AUC (event-level 50/50 split)\n\n" + c.md(hgb, "{:.4f}")
        + "\n\nAUC of unlinked vs each linked sub-population (same model, test set)\n\n" + c.md(subt, "{:.4f}")
        + "\n\nWorking points (test set)\n\n" + c.md(wpt, "{:.4f}")
        + "\n\nPermutation importance (mean ROC-AUC drop, 300k test clusters, 3 repeats)\n\n" + c.md(impt, "{:.4f}"))
    res["hgb"] = hgb.reset_index().to_dict(orient="list")
    res["hgb_subgroups"] = subt.to_dict()
    res["hgb_working_points"] = wpt.to_dict()
    res["hgb_importance"] = impt.to_dict()

    ae, err, te_ae = autoencoder(d)
    res["autoencoder"] = ae
    (RES / "sep_autoencoder.md").write_text("```\n" + json.dumps(ae, indent=1) + "\n```\n")
    print(json.dumps(ae, indent=1))
    (RES / "separability.json").write_text(json.dumps(res, indent=1, default=float))

    # ---------- figures
    te = d[d["is_test"]]
    y = te["unlinked"].to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        axs[0].plot(fpr, tpr, label=f"{name}: AUC {roc_auc_score(y, s):.3f}")
    fpr, tpr, _ = roc_curve(te_ae["unlinked"], err)
    axs[0].plot(fpr, tpr, label=f"AE recon. error (non-angle): AUC {roc_auc_score(te_ae['unlinked'], err):.3f}")
    for f in ("sizeX", "qPerPix"):
        sc = te[f].to_numpy() * (1 if roc_auc_score(y, te[f]) >= 0.5 else -1)
        fpr, tpr, _ = roc_curve(y, sc)
        axs[0].plot(fpr, tpr, ls=":", label=f"{f} alone: AUC {roc_auc_score(y, sc):.3f}")
    axs[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axs[0].set_xlabel("TP-linked clusters flagged (false-positive rate) [fraction]")
    axs[0].set_ylabel("unlinked clusters flagged (true-positive rate) [fraction]")
    axs[0].legend(fontsize=7); axs[0].set_title("ROC, all layers, test events")
    s = scores["HGB non-angle"]
    bins = np.linspace(0, 1, 51)
    for lab, m, col in (("TP-linked", y == 0, "#1f77b4"), ("linked pT>=2", (y == 0) & (te["tpPt"].to_numpy() >= 2), "#17becf"),
                        ("unlinked", y == 1, "#d62728")):
        axs[1].hist(s[m], bins, histtype="step", density=True, color=col, label=lab)
    axs[1].set_yscale("log"); axs[1].set_xlabel("HGB non-angle score P(unlinked) [unitless]")
    axs[1].set_ylabel("density [1/unit score]"); axs[1].legend(fontsize=8)
    bins = np.logspace(-3, 2, 61)
    for lab, m, col in (("TP-linked (training class)", y == 0, "#1f77b4"), ("unlinked", y == 1, "#d62728")):
        axs[2].hist(err[m], bins, histtype="step", density=True, color=col, label=lab)
    axs[2].set_xscale("log"); axs[2].set_yscale("log")
    axs[2].set_xlabel("AE reconstruction error (mean sq., standardised units) [unitless]")
    axs[2].set_ylabel("density [1/unit error]"); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(c.FIGS / "f3_separability.png", dpi=105)


if __name__ == "__main__":
    main()
