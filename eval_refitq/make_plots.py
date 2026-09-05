"""Evaluation plots for the SmartPixels digiRefit refit-quality BDT study.

Produces (to eval_refitq/):
  1. ROC curves: one panel per config overlaying tiers A/B/C/D, plus a panel of
     tier D across the four configs. Low-FPR-zoomed x axis (tracking-quality ROC).
  2. Resolution: sigma(d0) core and sigma(z0-GenVtx_z) core, reference vs each
     variant, over GENUINE hard-interaction prompt tracks.
  3. Sidecar variable distributions for AIII (1000) and AAAI (1110): every
     per-track extension column and per-hit column (sentinels excluded), split
     refit-performed vs passthrough (per-track) and by selHitClass (per-hit).

CAVEAT (stated on every relevant figure): production ran useAngles=alphaBeta
for ALL variants; the tiers ablate BDT INPUT FEATURES only, not the refit KF.

Run:  pixi run python eval_refitq/make_plots.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve

from ngtagger.train.refitquality import (
    CONFIG_ACTIVESP, SMARTPIXELS_CONFIGS, _split, build_refitq_dataset,
    load_refit_tables,
)

NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/nano_pu100_TrkSmartPix_withGen.root"
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
OUT = HERE
LABEL = "genuine"
SEED = 0
TEST_FRAC = 0.2
CAVEAT = ("caveat: production useAngles=alphaBeta for ALL variants; "
          "tiers ablate BDT INPUT FEATURES only, not the refit KF")


def _load_model(tag):
    m = xgb.XGBClassifier()
    m.load_model(os.path.join(MODELS, f"refitq_{tag}_xgb.json"))
    return m


def _core_sigma(x):
    q16, q84 = np.percentile(x, [16, 84])
    return (q84 - q16) / 2.0


# ---------------------------------------------------------------------------
# 1. ROC curves
# ---------------------------------------------------------------------------
def plot_roc():
    # per-config tables cached; tier A is config-independent (built off AIII)
    tables = {c: load_refit_tables([NANO], c) for c in SMARTPIXELS_CONFIGS}
    roc = {}  # (tier, config) -> (fpr, tpr)

    def _roc(tier, cfg):
        ref, var, hits = tables[cfg]
        X, y, _, _ = build_refitq_dataset(ref, var, hits, tier, cfg, label=LABEL)
        _, test = _split(len(X), TEST_FRAC, SEED)
        tag = "A" if tier == "A" else f"{tier}-{cfg}"
        proba = _load_model(tag).predict_proba(X[test])[:, 1]
        fpr, tpr, _ = roc_curve(y[test], proba)
        return fpr, tpr

    fpr_a, tpr_a = _roc("A", "AIII")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    colors = {"A": "k", "B": "tab:blue", "C": "tab:green", "D": "tab:red"}
    for i, cfg in enumerate(SMARTPIXELS_CONFIGS):
        ax = axes[i]
        ax.plot(fpr_a, tpr_a, color=colors["A"], ls="--", lw=1.6, label="A (ref hw, baseline)")
        for tier in ("B", "C", "D"):
            fpr, tpr = _roc(tier, cfg)
            ax.plot(fpr, tpr, color=colors[tier], lw=1.6, label=f"{tier}")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.0)
        ax.set_ylim(0.5, 1.005)
        ax.set_xlabel("fake acceptance (FPR, log)")
        ax.set_ylabel("genuine efficiency (TPR)")
        ax.set_title(f"{cfg}  (activeSP={CONFIG_ACTIVESP[cfg]})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=9)

    # panel 5: tier D across configs
    ax = axes[4]
    dcol = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    for cfg, c in zip(SMARTPIXELS_CONFIGS, dcol):
        fpr, tpr = _roc("D", cfg)
        ax.plot(fpr, tpr, color=c, lw=1.6, label=f"D-{cfg} ({CONFIG_ACTIVESP[cfg]})")
    ax.plot(fpr_a, tpr_a, "k--", lw=1.4, label="A baseline")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("fake acceptance (FPR, log)")
    ax.set_ylabel("genuine efficiency (TPR)")
    ax.set_title("Tier D across configs")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    # panel 6: AUC matrix text + caveats
    ax = axes[5]
    ax.axis("off")
    am = json.load(open(os.path.join(MODELS, "auc_matrix.json")))["results"]
    seed_std = json.load(open(os.path.join(MODELS, "auc_seed_std.json")))
    lines = ["AUC matrix (seed 0 test split)", "",
             f"  A baseline: {am['A']:.4f}", "",
             "  tier   " + " ".join(f"{c:>7}" for c in SMARTPIXELS_CONFIGS)]
    for tier in ("B", "C", "D"):
        lines.append(f"  {tier}    " + " ".join(f"{am[tier][c]:7.4f}" for c in SMARTPIXELS_CONFIGS))
    lines += ["", "AUC std over 8 split seeds:"]
    for k, v in seed_std.items():
        lines.append(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
    lines += ["", "=> tier/config spread (<0.003) is SMALLER than",
              "   the per-split std (~0.011): differences are",
              "   statistically indistinguishable.", "",
              "479 fakes / 16845 genuine (2.8%) => rough estimate."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10,
            family="monospace", transform=ax.transAxes)

    fig.suptitle("Refit-quality BDT ROC (genuine vs fake, held-out test)  —  " + CAVEAT,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "roc_refitquality.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 2. Resolution
# ---------------------------------------------------------------------------
def plot_resolution():
    import uproot
    import awkward as ak

    with uproot.open(f"{NANO}:Events") as t:
        g = ak.to_numpy(ak.flatten(t["L1TTrack_genuine"].array())).astype(bool)
        hard = ak.to_numpy(ak.flatten(t["L1TTrack_tpFromHardInteraction"].array())).astype(bool)
        sel = g & hard
        gvz = t["GenVtx_z"].array()
        gvz_pt = ak.to_numpy(ak.flatten(ak.broadcast_arrays(gvz, t["L1TTrack_z0"].array())[0]))
        d0ref = ak.to_numpy(ak.flatten(t["L1TTrack_d0"].array()))
        z0ref = ak.to_numpy(ak.flatten(t["L1TTrack_z0"].array()))
        d0v, z0v = {}, {}
        for c in SMARTPIXELS_CONFIGS:
            vt = f"L1TSmartPixelsTrackDigiRefit{c}"
            d0v[c] = ak.to_numpy(ak.flatten(t[f"{vt}_d0"].array()))
            z0v[c] = ak.to_numpy(ak.flatten(t[f"{vt}_z0"].array()))

    dz_ref = (z0ref - gvz_pt)[sel]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) sigma(d0) core bar chart
    ax = axes[0, 0]
    labels = ["REF"] + list(SMARTPIXELS_CONFIGS)
    sig_d0 = [_core_sigma(d0ref[sel])] + [_core_sigma(d0v[c][sel]) for c in SMARTPIXELS_CONFIGS]
    bars = ax.bar(labels, sig_d0, color=["gray", "tab:blue", "tab:green", "tab:orange", "tab:red"])
    for b, v in zip(bars, sig_d0):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$\sigma(d_0)$ core $=(q_{84}-q_{16})/2$  [cm]")
    ax.set_title(r"$d_0$ resolution proxy (genuine hard-interaction, true $d_0\approx0$)")
    ax.text(0.02, 0.95, "REF is 4-par: d0 pinned (spread=0);\nrefit ADDS d0 spread (angle scatter)",
            transform=ax.transAxes, va="top", fontsize=8, color="0.3")

    # (b) sigma(z0 - GenVtx_z) core bar chart
    ax = axes[0, 1]
    sig_z0 = [_core_sigma(dz_ref)] + [_core_sigma((z0v[c] - gvz_pt)[sel]) for c in SMARTPIXELS_CONFIGS]
    bars = ax.bar(labels, sig_z0, color=["gray", "tab:blue", "tab:green", "tab:orange", "tab:red"])
    for b, v in zip(bars, sig_z0):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$\sigma(z_0 - z_{\rm GenVtx})$ core  [cm]")
    ax.set_title(r"$z_0$ resolution vs hard-interaction $z$ (GenVtx_z)")

    # (c) d0 distributions overlaid
    ax = axes[1, 0]
    bins = np.linspace(-0.3, 0.3, 80)
    ax.hist(d0ref[sel], bins=bins, histtype="step", lw=1.5, label="REF (pinned)", color="gray")
    for c, col in zip(SMARTPIXELS_CONFIGS, ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
        ax.hist(d0v[c][sel], bins=bins, histtype="step", lw=1.3, label=c, color=col)
    ax.set_xlabel(r"$d_0$  [cm]")
    ax.set_ylabel("genuine hard-int tracks")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_title(r"$d_0$ distribution")

    # (d) z0 - GenVtx_z distributions
    ax = axes[1, 1]
    bins = np.linspace(-1.0, 1.0, 80)
    ax.hist(dz_ref, bins=bins, histtype="step", lw=1.5, label="REF", color="gray")
    for c, col in zip(SMARTPIXELS_CONFIGS, ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
        ax.hist((z0v[c] - gvz_pt)[sel], bins=bins, histtype="step", lw=1.3, label=c, color=col)
    ax.set_xlabel(r"$z_0 - z_{\rm GenVtx}$  [cm]")
    ax.set_ylabel("genuine hard-int tracks")
    ax.legend(fontsize=9)
    ax.set_title(r"$z_0$ kick vs hard-interaction vertex")

    fig.suptitle("Refit resolution: reference vs digiRefit variants  —  " + CAVEAT
                 + "\nMethod: robust core over genuine hard-interaction prompt tracks; "
                 "d0 true~0 (prompt TTbar); z0 ref = GenVtx_z (hard-interaction primary vertex z).",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "resolution_refitquality.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)
    return {"labels": labels, "sigma_d0": sig_d0, "sigma_z0_genvtx": sig_z0}


# ---------------------------------------------------------------------------
# 3. Sidecar distributions
# ---------------------------------------------------------------------------
_PERTRK_COLS = ["spixStatus", "spixRefitPerformed", "spixSeedCovOK", "spixParametrizedSeed",
                "spixAnyWindowTruncated", "spixNCrossings", "spixNAcceptedHits", "spixNKFUpdates",
                "spixLayerHitMask", "spixMaxWindowMult", "spixChi2IncRPhiTot", "spixChi2IncRZTot",
                "spixCompactWord"]
_PERHIT_COLS = ["layer", "windowMult", "windowTruncated", "hasAlpha", "hasBeta",
                "resX", "resY", "cotAlphaMeas", "cotBetaMeas", "sigAlpha", "sigBeta",
                "pullX", "pullY", "pullAlpha", "pullBeta", "chi2IncRPhi", "chi2IncRZ",
                "selHitClass", "parCotAlpha", "parCotBeta"]
_LOG_COLS = {"spixChi2IncRPhiTot", "spixChi2IncRZTot", "chi2IncRPhi", "chi2IncRZ",
             "pullAlpha", "pullBeta"}
_UNITS = {"resX": "cm", "resY": "cm", "spixChi2IncRPhiTot": "", "spixChi2IncRZTot": "",
          "sigAlpha": "rad", "sigBeta": "rad", "cotAlphaMeas": "", "cotBetaMeas": ""}


def _prep(vals, col):
    v = np.asarray(vals, dtype=np.float64)
    v = v[v > -900.0]  # sentinel exclusion
    if col in _LOG_COLS:
        s = np.sign(v)
        v = s * np.log10(1.0 + np.abs(v))
    return v


def plot_sidecar(cfg):
    import uproot
    import awkward as ak

    with uproot.open(f"{NANO}:Events") as t:
        vt = f"L1TSmartPixelsTrackDigiRefit{cfg}"
        ht = f"L1TSmartPixelsRefitHitDigiRefit{cfg}"
        perf = ak.to_numpy(ak.flatten(t[f"{vt}_spixRefitPerformed"].array())).astype(bool)
        pertrk = {c: ak.to_numpy(ak.flatten(t[f"{vt}_{c}"].array())) for c in _PERTRK_COLS
                  if f"{vt}_{c}" in t.keys()}
        selcls = ak.to_numpy(ak.flatten(t[f"{ht}_selHitClass"].array()))
        perhit = {c: ak.to_numpy(ak.flatten(t[f"{ht}_{c}"].array())) for c in _PERHIT_COLS
                  if f"{ht}_{c}" in t.keys()}

    # -- per-track figure: refit-performed vs passthrough --
    cols = list(pertrk.keys())
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, c in enumerate(cols):
        ax = axes[i]
        vp = _prep(pertrk[c][perf], c)
        vn = _prep(pertrk[c][~perf], c)
        lo = np.percentile(np.concatenate([vp, vn]) if len(vn) else vp, 0.5) if len(vp) else 0
        hi = np.percentile(np.concatenate([vp, vn]) if len(vn) else vp, 99.5) if len(vp) else 1
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 40)
        ax.hist(vp, bins=bins, histtype="step", lw=1.4, color="tab:green", density=True,
                label=f"refit ({perf.sum()})")
        if len(vn):
            ax.hist(vn, bins=bins, histtype="step", lw=1.4, color="tab:red", density=True,
                    label=f"passthrough ({(~perf).sum()})")
        xl = c + (r" [$\pm\log_{10}(1+|x|)$]" if c in _LOG_COLS else
                  (f" [{_UNITS[c]}]" if _UNITS.get(c) else ""))
        ax.set_xlabel(xl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-track extension columns  —  {cfg} (activeSP={CONFIG_ACTIVESP[cfg]})  "
                 "[refit-performed vs passthrough, sentinels excluded]\n" + CAVEAT, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(OUT, f"sidecar_pertrack_{cfg}.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    print("wrote", p1)

    # -- per-hit figure: split by selHitClass --
    cols = list(perhit.keys())
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    classes = sorted(np.unique(selcls).tolist())
    ccolors = {0: "tab:blue", 1: "tab:orange", 2: "tab:purple", -1: "0.5"}
    for i, c in enumerate(cols):
        ax = axes[i]
        allv = _prep(perhit[c], c)
        if len(allv) == 0:
            ax.axis("off"); continue
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 40)
        for cls in classes:
            m = (selcls == cls) & (perhit[c] > -900.0)
            v = _prep(perhit[c][m], c)
            if len(v):
                ax.hist(v, bins=bins, histtype="step", lw=1.3,
                        color=ccolors.get(cls, "k"), density=True, label=f"selClass={cls}")
        xl = c + (r" [$\pm\log_{10}(1+|x|)$]" if c in _LOG_COLS else
                  (f" [{_UNITS[c]}]" if _UNITS.get(c) else ""))
        ax.set_xlabel(xl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-hit link columns  —  {cfg} (activeSP={CONFIG_ACTIVESP[cfg]})  "
                 "[accepted hits only, split by selHitClass, sentinels excluded]\n"
                 "NB: per-hit table stores ACCEPTED hits only (hitAccepted==1 for all rows).",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(OUT, f"sidecar_perhit_{cfg}.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_roc()
    res = plot_resolution()
    json.dump(res, open(os.path.join(OUT, "resolution_numbers.json"), "w"), indent=2)
    for cfg in ("AIII", "AAAI"):
        plot_sidecar(cfg)
    print("done")
