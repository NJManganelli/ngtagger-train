"""Factorisation ladder of the jet-tagging study: metrics, paired event-bootstrap errors,
PDF + summary.json.

  pixi run python eval_refitq/tagger_sp/analyze.py [--boot 300]

Cells = (variant, mask), each with several seeds (train.py -> OUT/preds/<mask>_<variant>_s<seed>.npz).
Ladder (common.VARIANTS): L0 = baseline@IIII (OT reference), L1 = baseline@SP mask, L2 = track@SP,
L3 = trackmva@SP; controls L2' = track@IIII, L3' = trackmva@IIII. Steps reported:
  L1-L0 (refit tracks upstream only)   L2-L1 (+ refit track features)   L3-L2 (+ track-MVA scores)
  L2'-L0, L3'-L2' (the same feature additions on OT tracks)
  L2-L2', L3-L3' (better tracks at equal features)            L3-L0 (total)

Test sets hold the same EVENTS in every mask (the jets differ: each scenario re-emulates
PF/PUPPI/jets), so every difference is paired at the event level: each bootstrap replicate
draws multinomial counts over the union of test-event keys, every jet carries its event's count
as a weight, and every cell's seed-averaged metric is recomputed on that replicate. Row 0 of each
metric array is the nominal (unit-weight) value.

Metrics (softmax p_k), per region (inclusive; per sample; |eta| bins; pT bins):
  auc_<class> one-vs-rest (+ auc_tau for p_tau+ + p_tau-), macro_auc = mean of the 8 class AUCs
  b_vs_light  D = p_b/(p_b+p_light)          signal b,     background light (uds)
  c_vs_light  D = p_c/(p_c+p_light)          signal charm, background light
  tau_vs_jet  D = p_tau/(p_tau+p_light+p_g)  signal tau+-, background light+gluon
  <pair>_auc; <pair>_eff@1% / @10%: signal efficiency at the threshold giving 1% / 10%
  background efficiency INCLUSIVELY (per seed and replicate); in a bin, the same threshold
  (so a bin's efficiency and <pair>_mistag@X% are those of the inclusive working point).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLASSES = ["b", "charm", "light", "gluon", "taup", "taum", "muon", "electron"]
CI = {c: i for i, c in enumerate(CLASSES)}
PAIRS = {  # name: (signal, background, numerator, denominator) classes
    "b_vs_light": (["b"], ["light"], ["b"], ["b", "light"]),
    "c_vs_light": (["charm"], ["light"], ["charm"], ["charm", "light"]),
    "tau_vs_jet": (["taup", "taum"], ["light", "gluon"], ["taup", "taum"], ["taup", "taum", "light", "gluon"]),
}
MISTAGS = (0.01, 0.10)
SP_MASKS = [m for m in C.MASKS if m != C.BASELINE_MASK]
L0 = ("baseline", C.BASELINE_MASK)
STEPS = [  # (label, a, b): a - b ; mask "M" = the SP mask of the column
    ("L1-L0  refit tracks upstream", ("baseline", "M"), L0),
    ("L2-L1  + refit track features", ("track", "M"), ("baseline", "M")),
    ("L3-L2  + track-MVA scores", ("trackmva", "M"), ("track", "M")),
    ("L2'-L0  + track features, OT tracks", ("track", "IIII"), L0),
    ("L3'-L2'  + MVA scores, OT tracks", ("trackmva", "IIII"), ("track", "IIII")),
    ("L2-L2'  better tracks, = features", ("track", "M"), ("track", "IIII")),
    ("L3-L3'  better tracks, = features", ("trackmva", "M"), ("trackmva", "IIII")),
    ("L3-L0  total", ("trackmva", "M"), L0),
]
MAIN_STEPS = STEPS[:5]
KEY_METRICS = ["macro_auc", "b_vs_light_eff@10%", "b_vs_light_eff@1%", "c_vs_light_eff@10%",
               "tau_vs_jet_eff@10%", "tau_vs_jet_eff@1%"]
ALL_AUCS = [f"auc_{c}" for c in CLASSES] + ["auc_tau"]
STEP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b", "#a0a0a0"]
STEP_MARKERS = ["o", "s", "^", "v", "D"]


# ------------------------------------------------------------------ vectorised weighted metrics
def wauc(score, y, W):
    """Weighted ROC AUC per row of W (n_rep, n); ties count 1/2. NaN if a side is empty."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.full(len(W), np.nan)
    o = np.argsort(score, kind="stable")
    s = score[o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    Wo, yo = W[:, o], y[o]
    su = np.add.reduceat(Wo * yo, starts, axis=1)
    bu = np.add.reduceat(Wo * ~yo, starts, axis=1)
    below = np.cumsum(bu, axis=1) - bu
    den = su.sum(1) * bu.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (su * (below + 0.5 * bu)).sum(1) / den


def thresholds(score_b, W_b, mistag):
    """Per row: the threshold t with weighted background fraction above t <= mistag."""
    if score_b.size == 0:
        return np.full(len(W_b), np.nan)
    o = np.argsort(score_b)[::-1]
    sb = score_b[o]
    cw = np.cumsum(W_b[:, o], axis=1)
    tot = cw[:, -1:]
    k = (cw <= mistag * tot).sum(1)
    return np.where(k < len(sb), sb[np.minimum(k, len(sb) - 1)], -np.inf)


def frac_above(score, W, t):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * (score[None, :] > t[:, None])).sum(1) / W.sum(1)


def disc(prob, num, den):
    return prob[:, [CI[c] for c in num]].sum(1) / np.maximum(prob[:, [CI[c] for c in den]].sum(1), 1e-12)


def region_masks(d):
    R = {"incl": np.ones(len(d["label"]), bool)}
    for si, s in enumerate(d["samples"]):
        R[f"sample:{s}"] = d["sample"] == si
    ae = np.abs(d["eta"])
    for lo, hi in zip(C.ETA_BINS[:-1], C.ETA_BINS[1:]):
        R[f"abs_eta:{lo:g}-{hi:g}"] = (ae >= lo) & (ae < hi)
    for lo, hi in zip(C.PT_BINS[:-1], C.PT_BINS[1:]):
        R[f"pt:{lo:g}-{hi:g}" if hi < 1e8 else f"pt:>{lo:g}"] = (d["pt"] >= lo) & (d["pt"] < hi)
    return R


def cell_metrics(d, W):
    """{region: {metric: (n_rep,)}} for one seed's predictions d and replicate weights W."""
    prob, lab = d["prob"], d["label"]
    ptau = prob[:, CI["taup"]] + prob[:, CI["taum"]]
    istau = (lab == CI["taup"]) | (lab == CI["taum"])
    D = {n: disc(prob, num, den) for n, (_, _, num, den) in PAIRS.items()}
    S = {n: np.isin(lab, [CI[c] for c in sig]) for n, (sig, _, _, _) in PAIRS.items()}
    B = {n: np.isin(lab, [CI[c] for c in bkg]) for n, (_, bkg, _, _) in PAIRS.items()}
    T = {(n, m): thresholds(D[n][B[n]], W[:, B[n]], m) for n in PAIRS for m in MISTAGS}
    out = {}
    for reg, r in region_masks(d).items():
        Wr = W[:, r]
        o = {f"auc_{c}": wauc(prob[r, CI[c]], lab[r] == CI[c], Wr) for c in CLASSES}
        o["auc_tau"] = wauc(ptau[r], istau[r], Wr)
        o["macro_auc"] = np.nanmean(np.stack([o[f"auc_{c}"] for c in CLASSES]), axis=0) \
            if np.isfinite(np.stack([o[f"auc_{c}"] for c in CLASSES])).any() else np.full(len(W), np.nan)
        for n in PAIRS:
            sel = r & (S[n] | B[n])
            o[f"{n}_auc"] = wauc(D[n][sel], S[n][sel], W[:, sel])
            for m in MISTAGS:
                o[f"{n}_eff@{m:.0%}"] = frac_above(D[n][r & S[n]], W[:, r & S[n]], T[(n, m)])
                o[f"{n}_mistag@{m:.0%}"] = frac_above(D[n][r & B[n]], W[:, r & B[n]], T[(n, m)])
        out[reg] = o
    return out


# ------------------------------------------------------------------ loading
def load_preds():
    P = {}
    for f in sorted(glob.glob(os.path.join(C.OUT, "preds", "*.npz"))):
        m = re.fullmatch(r"([AI]{4})_(\w+?)_s(\d+)\.npz", os.path.basename(f))
        if not m or m.group(2) not in C.VARIANTS:
            continue              # dev runs carry a tag suffix
        d = np.load(f)
        P.setdefault((m.group(2), m.group(1)), {})[int(m.group(3))] = {
            "prob": d["prob"].astype(np.float64), "label": d["label"], "pt": d["pt"], "eta": d["eta"],
            "sample": d["sample"], "samples": [str(x) for x in d["samples"]],
            "key": d["sample"].astype(np.int64) * 10**7 + d["fileno"].astype(np.int64) * 10**4 + d["entry"],
            "train_counts": json.loads(str(d["train_counts"])), "n_epochs": len(d["loss"]),
            "features": [str(x) for x in d["features"]]}
    return P


def compute(P, n_boot, seed=12345):
    """{cell: {"seeds": {s: {reg: {metric: nominal}}}, "avg": {reg: {metric: (1+n_boot,)}}}}."""
    ev = np.unique(np.concatenate([d["key"] for seeds in P.values() for d in seeds.values()]))
    rng = np.random.default_rng(seed)
    cnt = np.vstack([np.ones(len(ev))] + [rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)))
                                          for _ in range(n_boot)]).astype(np.float64)
    out = {}
    for cell, seeds in P.items():
        t0 = time.time()
        per = {}
        for s, d in seeds.items():
            W = cnt[:, np.searchsorted(ev, d["key"])]
            per[s] = cell_metrics(d, W)
        regs = per[next(iter(per))].keys()
        avg = {r: {k: np.nanmean(np.stack([per[s][r][k] for s in per]), axis=0) for k in per[next(iter(per))][r]}
               for r in regs}
        out[cell] = {"seeds": {s: {r: {k: float(v[0]) for k, v in per[s][r].items()} for r in regs} for s in per},
                     "avg": avg}
        print(f"  {cell}: {len(seeds)} seeds ({time.time() - t0:.0f}s)", flush=True)
    return out


def seed_stats(M, cell, reg, k):
    v = np.array([M[cell]["seeds"][s][reg][k] for s in M[cell]["seeds"]], float)
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else np.nan,
            float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v))


def step_value(M, a, b, reg, k):
    """Nominal difference a - b of seed-averaged metrics, its event-bootstrap std and seed error."""
    if a not in M or b not in M:
        return None
    da, db = M[a]["avg"][reg][k], M[b]["avg"][reg][k]
    diff = da - db
    boot = float(np.nanstd(diff[1:], ddof=1)) if np.isfinite(diff[1:]).sum() > 2 else np.nan
    _, sa, na = seed_stats(M, a, reg, k)
    _, sb, nb = seed_stats(M, b, reg, k)
    seed_err = float(np.sqrt((sa ** 2 / na if na else np.nan) + (sb ** 2 / nb if nb else np.nan)))
    return {"diff": float(diff[0]), "boot_err": boot, "seed_err": seed_err}


def resolve(cell, mask):
    return (cell[0], mask if cell[1] == "M" else cell[1])


def ladder(M, regs):
    L = {}
    for mask in SP_MASKS:
        for lab, a, b in STEPS:
            for reg in regs:
                for k in M[next(iter(M))]["avg"][reg]:
                    v = step_value(M, resolve(a, mask), resolve(b, mask), reg, k)
                    if v is not None:
                        L.setdefault(mask, {}).setdefault(lab, {}).setdefault(reg, {})[k] = v
    return L


def counts(P):
    """{mask: {sample: {class: [train, test]}}} from the baseline cell's first seed."""
    out = {}
    for (var, mask), seeds in P.items():
        if var != "baseline":
            continue
        d = next(iter(seeds.values()))
        tc = d["train_counts"]
        out[mask] = {s: {c: [tc[s][c][0], int(((d["label"] == CI[c]) & (d["sample"] == si)).sum())]
                         for c in CLASSES} for si, s in enumerate(d["samples"])}
    return out


def roc_points(seeds, name):
    """Pooled (all seeds) ROC of a pairwise discriminant: (signal eff, background eff)."""
    sig, bkg, num, den = PAIRS[name]
    Ds, Db = [], []
    for d in seeds.values():
        D = disc(d["prob"], num, den)
        Ds.append(D[np.isin(d["label"], [CI[c] for c in sig])])
        Db.append(D[np.isin(d["label"], [CI[c] for c in bkg])])
    Ds, Db = np.sort(np.concatenate(Ds)), np.sort(np.concatenate(Db))
    t = np.unique(np.concatenate([Ds, Db]))[::-1]
    return 1 - np.searchsorted(Ds, t) / len(Ds), 1 - np.searchsorted(Db, t) / len(Db)


# ------------------------------------------------------------------ PDF helpers
def fmt(m, s=None, nd=3):
    if m is None or not np.isfinite(m):
        return "-"
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None and np.isfinite(s) else "")


def fmt_step(v, nd=3):
    if v is None or not np.isfinite(v["diff"]):
        return "-"
    err = np.hypot(v["boot_err"], v["seed_err"]) if np.isfinite(v["seed_err"]) else v["boot_err"]
    sig = v["diff"] / err if err and np.isfinite(err) and err > 0 else np.nan
    return f"{v['diff']:+.{nd}f} ± {err:.{nd}f} ({sig:+.1f}σ)"


def short(lab):
    return lab.split("  ")[0]


STEP_NOTE = ("Steps: " + "; ".join(f"{short(lab)} = {lab.split('  ', 1)[1]}" for lab, _, _ in STEPS)
             + ". Cell: difference of seed-averaged metrics ± (event bootstrap ⊕ seed error) (significance); "
             "the two error parts are listed separately in summary.json.")


def table_page(pdf, title, header, rows, note=None, fontsize=7.0, col_widths=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.02, 0.97, title, fontsize=11, weight="bold", va="top")
    ax = fig.add_axes([0.02, 0.10 if note else 0.03, 0.96, 0.83 if note else 0.9])
    ax.axis("off")
    if rows:
        tb = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                      colWidths=col_widths)
        tb.auto_set_font_size(False)
        tb.set_fontsize(fontsize)
        tb.scale(1, 1.22)
        for (r, c), cell in tb.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_text_props(weight="bold")
    if note:
        fig.text(0.02, 0.015, note, fontsize=7, va="bottom", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def text_page(pdf, title, lines):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.03, 0.96, title, fontsize=13, weight="bold", va="top")
    fig.text(0.03, 0.91, "\n".join(lines), fontsize=7.9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def mask_colors():
    return {"IIII": "#6b6b6b", "AAAA": "#2a78d6", "AAAI": "#eb6834", "IAAA": "#1baf7a"}


def plot_rocs(pdf, P):
    import matplotlib.pyplot as plt

    col = mask_colors()
    vars_ = [v for v in C.VARIANTS if any(c[0] == v for c in P)]
    fig, axes = plt.subplots(len(vars_), 3, figsize=(11.69, 3.2 * len(vars_) + 1.2), squeeze=False)
    for i, var in enumerate(vars_):
        for j, name in enumerate(PAIRS):
            ax = axes[i, j]
            for mask in C.MASKS:
                if (var, mask) not in P:
                    continue
                es, eb = roc_points(P[(var, mask)], name)
                ok = eb > 0
                ax.plot(es[ok], eb[ok], "--" if mask == C.BASELINE_MASK else "-", color=col[mask], lw=1.5,
                        label=mask)
            ax.set_yscale("log")
            ax.set_ylim(3e-3, 1.05)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.25, lw=0.5)
            sig, bkg = name.split("_vs_")
            ax.set_xlabel(f"{sig} efficiency")
            ax.set_ylabel(f"{bkg} mistag")
            ax.set_title(f"L{C.LEVEL[var]} ({var}): {name}", fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, title="scenario (IIII = OT baseline)")
    fig.suptitle("ROC curves on the test set (all seeds pooled)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def plot_steps(pdf, L, mask, what, regs_or_keys, xlabels, title, ylabel):
    """Step gains (MAIN_STEPS) vs classes or bins for one mask. what='metric:<k>' plots metric k
    over regions regs_or_keys; what='region:<reg>' plots the metrics regs_or_keys in region reg."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.69, 5.2))
    for i, (lab, _, _) in enumerate(MAIN_STEPS):
        ys, es = [], []
        for q in regs_or_keys:
            reg, k = (q, what.split(":", 1)[1]) if what.startswith("metric:") else (what.split(":", 1)[1], q)
            v = L[mask].get(lab, {}).get(reg, {}).get(k)
            ys.append(v["diff"] if v else np.nan)
            es.append(np.hypot(v["boot_err"], v["seed_err"]) if v else np.nan)
        x = np.arange(len(regs_or_keys)) + (i - 2) * 0.13
        ax.errorbar(x, ys, yerr=es, fmt=STEP_MARKERS[i], color=STEP_COLORS[i], ms=5, lw=1, capsize=2,
                    label=lab, mfc="white" if "OT tracks" in lab else STEP_COLORS[i])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(regs_or_keys)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(f"{mask}: {title}  (bars: event bootstrap ⊕ seed error)", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=C.OUT)
    a = ap.parse_args()
    t0 = time.time()
    P = load_preds()
    if not P:
        raise SystemExit("no predictions")
    print(f"cells: {sorted(P)}", flush=True)
    M = compute(P, a.boot)
    regs = list(M[next(iter(M))]["avg"])
    L = ladder(M, regs)
    N = counts(P)
    mva = {m: json.load(open(p)) for m in C.MASKS
           if os.path.exists(p := os.path.join(C.OUT, "mva", m, "heldout_check.json"))}
    samples = next(iter(next(iter(P.values())).values()))["samples"]
    cells = {f"{v}@{m}": {"level": ("L0" if (v, m) == L0 else
                                    f"L{C.LEVEL[v]}" + ("'" if m == C.BASELINE_MASK else "")),
                          "n_seeds": len(M[(v, m)]["seeds"]),
                          "n_epochs": [d["n_epochs"] for d in P[(v, m)].values()],
                          "features": next(iter(P[(v, m)].values()))["features"],
                          "per_seed": M[(v, m)]["seeds"],
                          "seed_mean_std": {r: {k: seed_stats(M, (v, m), r, k)[:2] for k in M[(v, m)]["avg"][r]}
                                            for r in regs}}
             for (v, m) in P}
    summary = {
        "what": "SmartPixels L1 jet multiclass tagging: factorisation ladder vs the OT baseline",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "samples": samples,
        "acceptance": {"pt_min_GeV": C.PT_MIN, "abs_eta_max": C.ETA_MAX},
        "split": {"test_fraction": C.TEST_FRACTION, "seed": C.SPLIT_SEED,
                  "ttbar_train_only_files": list(C.TTBAR_TRAIN_ONLY)},
        "variants": C.VARIANTS, "ladder_definition": {lab: [f"{x[0]}@{x[1]}" for x in (a_, b_)]
                                                       for lab, a_, b_ in STEPS},
        "n_bootstrap": a.boot, "jet_counts_train_test": N, "track_mva_heldout": mva,
        "cells": cells, "ladder": L,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"summary.json ({time.time() - t0:.0f}s)", flush=True)

    import matplotlib

    matplotlib.use("pdf")
    from matplotlib.backends.backend_pdf import PdfPages

    pdfp = os.path.join(a.out, "tagger_sp.pdf")
    with PdfPages(pdfp) as pdf:
        text_page(pdf, "SmartPixels L1 jet tagging: factorisation ladder vs the OT baseline", [
            "Scenarios: canonical coopt nanos (build 92bad91b2ba, disc refit), PU200. The tracks feeding the L1 vertex, PF, PUPPI",
            "and SC4 NG jets are the SmartPixels (SP) tracks of each IT active-layer mask (A/I = IT layer group active/inactive);",
            "IIII = passthrough (SP track = OT track): the OT-only baseline through the identical pipeline.",
            f"Samples: {samples} combined (TTbar RelVal; QQ->H->tautau RelVal adds taus). Split by event, identical in all",
            f"masks: {C.TEST_FRACTION:.0%} test (TTbar test events only from f03-f10; f01/f02 trained the cluster-angle CNN).",
            f"Jets: SC4 NG (L1puppiJetSC4NG), reco pT > {C.PT_MIN:g} GeV, |eta| < {C.ETA_MAX:g}; <= 16 constituents (framework cap)",
            "from L1SC4NGJetCands -> L1ExtPuppiCand; charged ones -> l1TrackIdx -> L1TSmartPixelsExtTrack (the track feeding PF).",
            "Labels (framework labels.py, identical in every scenario): GenJet hadron/parton flavour (dR < 0.4), GenVisTau",
            "(charge), prompt gen e/mu: b charm light gluon tau+ tau- muon electron; unlabelled (PU) jets dropped.",
            "Tagger (every cell): framework DeepSetHGQ2 (NGT DeepSet, quantisation-aware), framework loss, class weighting",
            "('onlyclass', to the b count) and callbacks. DEVIATION: batch 128 / lr 1e-3 instead of 2048 / 1e-2 (the stock",
            "schedule does not learn on ~5k jets: 2 steps/epoch). CPU, 2 threads.",
            "LADDER:  L0 = baseline inputs @ IIII (OT reference)   L1 = baseline inputs @ SP mask (refit tracks upstream only)",
            "  L2 = + track features of the constituent's track: rInv, tanL, z0, d0, chi2 x3, trkMVA1, dxy / dz wrt the L1 PV,",
            "       sigma(d0), sigma(z0) (fit covariance) and # accepted IT hits",
            "  L3 = L2 + track-MVA scores P(clean), P(d0 acc), P(z0 acc), log sigma_hat(d0), log sigma_hat(z0) (HGB heads fitted",
            "       per scenario on the SAME 80% training events -- correlation with the tagger ignored by design)",
            "  controls L2' / L3' = the same on IIII: slots filled from the OT track (0 IT hits), OT-only MVA heads.",
            "  Track truth everywhere = the OT track's TP (OT link). The MVA 'clean' = OT-genuine and no IT hit from another TP.",
            f"Errors: 'ev' = event-bootstrap std ({a.boot} paired resamples of the test events; seed-averaged metrics);",
            "'sd' = seed error of the difference of means; sigma = diff / (ev (+) sd) (conservative: they overlap).",
            "Caveats: limited statistics; single process mix (TTbar + H->tautau); MVA/tagger correlation ignored;",
            "16-constituent cap; eff@1% rests on few background jets; SP covariance = KF posterior (optimistic).",
        ])
        rows = []
        for mask in [m for m in C.MASKS if m in N]:
            for s, cc in N[mask].items():
                rows.append([mask, s] + [f"{cc[c][0]} / {cc[c][1]}" for c in CLASSES])
        table_page(pdf, "Jet counts per class: train / test (acceptance applied; train includes the 10% validation)",
                   ["mask", "sample"] + CLASSES, rows)
        # HEADLINE: the ladder
        for mask in [m for m in SP_MASKS if m in L]:
            hdr = ["metric (inclusive)", "L0 (IIII) abs."] + [short(lab) for lab, _, _ in STEPS]
            rows = []
            for k in KEY_METRICS + ["b_vs_light_auc", "c_vs_light_auc", "tau_vs_jet_auc"] + ALL_AUCS:
                m0 = seed_stats(M, L0, "incl", k) if L0 in M else (np.nan, np.nan, 0)
                rows.append([k, fmt(m0[0], m0[1])] + [fmt_step(L[mask].get(lab, {}).get("incl", {}).get(k))
                                                      for lab, _, _ in STEPS])
            table_page(pdf, f"LADDER ({mask} vs IIII): gain of each step, inclusive test set", hdr, rows,
                       fontsize=6.3, note=STEP_NOTE + " L0 column: absolute value ± seed std.")
        for mask in [m for m in SP_MASKS if m in L]:
            plot_steps(pdf, L, mask, "region:incl", ALL_AUCS + ["b_vs_light_auc", "c_vs_light_auc", "tau_vs_jet_auc"],
                       [k.replace("auc_", "").replace("_auc", "") for k in
                        ALL_AUCS + ["b_vs_light_auc", "c_vs_light_auc", "tau_vs_jet_auc"]],
                       "step gains per class (one-vs-rest AUC) and pairwise AUCs", "Δ AUC")
            bins = [r for r in regs if r.startswith(("abs_eta:", "pt:"))]
            for k, lab in (("b_vs_light_eff@10%", "Δ b eff (10% light mistag)"),
                           ("c_vs_light_eff@10%", "Δ c eff (10% light mistag)"),
                           ("tau_vs_jet_eff@10%", "Δ tau eff (10% light+gluon mistag)"),
                           ("macro_auc", "Δ macro AUC")):
                plot_steps(pdf, L, mask, f"metric:{k}", bins, [b.replace("abs_eta:", "|η| ").replace("pt:", "pT ")
                                                               for b in bins],
                           f"step gains in |η| and pT bins: {k}", lab)
            rows = []
            for reg in bins + [r for r in regs if r.startswith("sample:")]:
                for k in ("b_vs_light_eff@10%", "c_vs_light_eff@10%", "tau_vs_jet_eff@10%", "macro_auc"):
                    m0 = seed_stats(M, L0, reg, k) if L0 in M else (np.nan, np.nan, 0)
                    rows.append([reg, k, fmt(m0[0], m0[1])] + [fmt_step(L[mask].get(lab, {}).get(reg, {}).get(k))
                                                             for lab, _, _ in STEPS])
            table_page(pdf, f"LADDER ({mask}): step gains by region", ["region", "metric", "L0 (IIII) abs."]
                       + [short(lab) for lab, _, _ in STEPS], rows, fontsize=4.8,
                       note=STEP_NOTE + " Efficiencies at the INCLUSIVE 10%-mistag working point of each cell.")
        # absolute values per cell
        cl = [c for c in [(v, m) for v in C.VARIANTS for m in C.MASKS] if c in M]
        for chunk in (KEY_METRICS + ["b_vs_light_auc", "c_vs_light_auc", "tau_vs_jet_auc"], ALL_AUCS):
            rows = [[k] + [fmt(*seed_stats(M, c, "incl", k)[:2]) for c in cl] for k in chunk]
            table_page(pdf, "Absolute inclusive values per cell (mean ± seed std)", ["metric"]
                       + [f"{c[0]}@{c[1]}" for c in cl], rows, fontsize=5.3)
        plot_rocs(pdf, P)
        if mva:
            rows = [[m, v.get("coll"), v.get("n_test_tracks"), fmt(v.get("auc_clean")),
                     fmt(v.get("median_sigma_hat_d0_um"), None, 1), fmt(v.get("sigma68_d0_resid_um"), None, 1),
                     fmt(v.get("coverage_1sigma_d0"))] for m, v in mva.items()]
            table_page(pdf, "Track-MVA heads (L3 inputs): held-out check on the test events' tracks",
                       ["mask", "heads", "test tracks", "AUC P(clean)", "median sigma_hat(d0) [um]",
                        "68% |d0 resid| [um]", "frac |resid| < sigma_hat"], rows,
                       note="SP masks: refit heads on the SP tracks; IIII: OT-only heads. Truth = the OT track's TP.")
    print(f"wrote {pdfp} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
