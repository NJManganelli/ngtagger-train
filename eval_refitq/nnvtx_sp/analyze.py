"""Metrics, closure, figures (vector PDF), table pages and summary.json of the NNVtx study.

Headline sample: the test-eligible events (folds 0-4 = f03-f10, 800 events); every E2E
prediction and every MVA score used there is out of sample (vertex.py, train-trackmva).
Methods per variant: FH = offline float GTT fastHisto + the GTT eta-binned |dz| track-vertex
association (the sub2gev reference, same code), E2E-stock, E2E-mva (seed means).

Metrics (per method and variant):
  eff          fraction of events with |z_PV - GenVtx_z| < 0.15 cm (+ the curve vs the window)
  core / full  dz = z_PV - GenVtx_z: RMS about zero of the iterative 3-sigma-clipped core, and of all
  AUC          association ROC AUC over the selected tracks (label: HS track); FH score = -|dz_trk|/window(eta)
  WP           E2E: association score > 0.5; FH: inside the GTT TVA window
  assoc eff    sum(HS associated) / sum(HS selected); purity = sum(HS associated) / sum(associated)
  HS/ev        HS tracks associated per event
Errors: event bootstrap (N_BOOT replicas, the SAME replicas for every method/variant, so differences
are paired). E2E values are the mean over seeds; the seed spread (std over seeds) is quoted separately.

  pixi run python eval_refitq/nnvtx_sp/analyze.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nvcommon import (MVA_STEM, OUTDIR, PRED, PV_WINDOW, VARIANTS, VLABEL, core_rms, event_folds,  # noqa: E402
                      fasthisto, full_rms, load_dataset, selections, tva_dzmax)

N_BOOT = 1000
UM = 1e4
WP = 0.5
ARMS = ("stock", "mva")
METHODS = ("FH", "stock", "mva")
MLABEL = {"FH": "FastHisto + GTT TVA", "stock": "E2E NNVtx (stock inputs)", "mva": "E2E NNVtx + track-MVA scores"}
METRICS = ("eff", "core_um", "out_frac", "full_um", "auc", "assoc_eff", "purity", "hs_per_ev")
MFMT = {"eff": (100, "{:.1f}"), "core_um": (1, "{:.0f}"), "out_frac": (100, "{:.1f}"), "full_um": (1, "{:.0f}"),
        "auc": (1, "{:.4f}"), "assoc_eff": (100, "{:.1f}"), "purity": (100, "{:.1f}"), "hs_per_ev": (1, "{:.2f}")}
MHDR = {"eff": "PV eff %", "core_um": "dz core RMS [um]", "out_frac": "% outside core", "full_um": "dz full RMS [um]",
        "auc": "assoc AUC", "assoc_eff": "assoc eff % (WP)", "purity": "assoc purity % (WP)",
        "hs_per_ev": "HS assoc / ev (WP)"}
C = {"V0": "#0072B2", "V1": "#009E73", "V3": "#D55E00", "FH": "#000000", "stock": "#0072B2", "mva": "#D55E00"}
LS = {"FH": ":", "stock": "--", "mva": "-"}


def _load_sub2gev_analyze():
    spec = importlib.util.spec_from_file_location("sub2gev_analyze", os.path.join(HERE, "..", "sub2gev", "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2A = _load_sub2gev_analyze()  # legend_below / selections / run_pv / load_cache of the sub2gev study
legend_below = S2A.legend_below


def table_page(pdf, title, header, rows, notes=()):
    """The sub2gev table page, with the page width scaled to the column count (wide tables fit)."""
    import textwrap
    width = max(11.0, 2.0 * len(header))
    notes = [ln for s_ in notes for ln in textwrap.wrap(s_, int(17 * width))]
    fig = plt.figure(figsize=(width, 1.6 + 0.3 * len(rows) + 0.22 * len(notes)))
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    h = fig.get_figheight()
    ax = fig.add_axes([0.01, 0.02 + 0.22 * len(notes) / h, 0.98, 0.9 - (0.22 * len(notes) + 0.4) / h])
    ax.axis("off")
    t = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=header, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(list(range(len(header))))
    for i, s_ in enumerate(notes):
        fig.text(0.02, 0.02 + 0.22 * (len(notes) - 1 - i) / h, s_, fontsize=7.5)
    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------- per-config inputs
class Cfg:
    """One (variant, method[, seed]) on the test events: per-event z, per-track association."""

    def __init__(self, z, rows, ev, score, passed, y, genz):
        self.dz = z - genz                            # per test event
        self.found = np.abs(self.dz) < PV_WINDOW
        n = len(genz)
        self.n_hs_sel = np.bincount(ev, weights=y, minlength=n)
        self.n_assoc = np.bincount(ev, weights=passed, minlength=n)
        self.n_hs_assoc = np.bincount(ev, weights=passed & y, minlength=n)
        o = np.argsort(score, kind="stable")
        s = score[o]
        self.gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        self.y_s, self.ev_s, self.ng = y[o].astype(float), ev[o], int(self.gid[-1]) + 1 if len(s) else 0
        self.score, self.y, self.passed = score, y, passed

    def auc(self, c):
        w = c[self.ev_s].astype(float)
        pos = np.bincount(self.gid, weights=w * self.y_s, minlength=self.ng)
        neg = np.bincount(self.gid, weights=w * (1 - self.y_s), minlength=self.ng)
        below = np.cumsum(neg) - neg
        return float(np.sum(pos * (below + 0.5 * neg)) / (pos.sum() * neg.sum()))

    def stats(self, c):
        n = c.sum()
        dz = np.repeat(self.dz, c) * UM
        core, out = core_rms(dz)
        return {"eff": float(c @ self.found / n), "core_um": core, "out_frac": out,
                "full_um": float(np.sqrt(c @ (self.dz * UM) ** 2 / n)), "auc": self.auc(c),
                "assoc_eff": float(c @ self.n_hs_assoc / (c @ self.n_hs_sel)),
                "purity": float(c @ self.n_hs_assoc / max(c @ self.n_assoc, 1)),
                "hs_per_ev": float(c @ self.n_hs_assoc / n)}


def build_configs(T, E, test):
    genz = E["genz"][test]
    pos = np.full(len(test), -1)
    pos[test] = np.arange(test.sum())
    S = selections(T)
    ev_all = T["evt"]
    sl = np.searchsorted(ev_all, np.arange(len(test) + 1))
    cfgs, zfh_all, extra = {}, {}, {}
    for v in VARIANTS:
        sel, pre, hs = S[v]
        zfh = np.full(len(test), np.nan)
        for i in range(len(test)):
            s = slice(sl[i], sl[i + 1])
            m = sel[s]
            zfh[i] = fasthisto(T[pre + "z0"][s][m], T[pre + "pt"][s][m])[0]
        zfh_all[v] = zfh
        rows = np.flatnonzero(sel & test[ev_all])
        ev = pos[ev_all[rows]]
        y = hs[rows]
        dzt = np.abs(T[pre + "z0"][rows] - zfh[ev_all[rows]])
        win = tva_dzmax(T[pre + "eta"][rows])
        cfgs[(v, "FH")] = [Cfg(zfh[test], rows, ev, -dzt / win, dzt < win, y, genz)]
        P = np.load(PRED.format(variant=v))
        idx = P["idx"]
        assert np.array_equal(P["folds"], event_folds(E))
        extra[v] = {"sel_per_ev": float(sel[test[ev_all]].sum() / test.sum()),
                    "clipped_tracks": int(sel[test[ev_all]].sum() - (idx[test] >= 0).sum()), "epochs": {}}
        for arm in ARMS:
            seeds = sorted({int(k.split("_s")[1].split("_")[0]) for k in P.files if k.startswith(arm + "_s")})
            cfgs[(v, arm)] = []
            for sd in seeds:
                flat = np.zeros(len(ev_all), np.float32)
                ok = idx >= 0
                flat[idx[ok]] = P[f"{arm}_s{sd}_assoc"][ok]
                sc = flat[rows]
                cfgs[(v, arm)].append(Cfg(P[f"{arm}_s{sd}_z"][test], rows, ev, sc, sc > WP, y, genz))
                extra[v]["epochs"][f"{arm}_s{sd}"] = P[f"{arm}_s{sd}_epochs"].tolist()
    return cfgs, zfh_all, extra


def bootstrap_all(cfgs, n_test):
    """{key: {metric: (1 + N_BOOT,) seed-mean values}} (index 0 = central), + per-seed centrals."""
    rng = np.random.default_rng(12345)
    reps = [np.ones(n_test, np.int64)] + [np.bincount(rng.integers(0, n_test, n_test), minlength=n_test)
                                          for _ in range(N_BOOT)]
    B, seeds = {}, {}
    for key, lst in cfgs.items():
        per = np.array([[[cf.stats(c)[m] for m in METRICS] for c in reps] for cf in lst])  # (seed, rep, metric)
        B[key] = {m: per[:, :, j].mean(0) for j, m in enumerate(METRICS)}
        seeds[key] = {m: per[:, 0, j].tolist() for j, m in enumerate(METRICS)}
        print(f"bootstrap {key} done", flush=True)
    return B, seeds


# ----------------------------------------------------------------------------- formatting
def val(B, key, m):
    return float(B[key][m][0]), float(np.nanstd(B[key][m][1:], ddof=1))


def diff(B, a, b, m):
    d = B[a][m] - B[b][m]
    return float(d[0]), float(np.nanstd(d[1:], ddof=1))


def seed_sd(seeds, key, m):
    v = seeds[key][m]
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def fmt(v, e, m, sd=None):
    sc, f = MFMT[m]
    s = (f + " +- " + f).format(v * sc, e * sc)
    return s + (" [" + f.format(sd * sc) + "]" if sd is not None else "")


def seed_err_of_diff(seeds, a, b, m):
    """Seed-induced error of a difference of seed means: sqrt(sA^2/nA + sB^2/nB)."""
    t = 0.0
    for k in (a, b):
        if k[1] != "FH":
            t += seed_sd(seeds, k, m) ** 2 / len(seeds[k][m])
    return float(np.sqrt(t))


def matched_operating_points(cfgs):
    """E2E association at the FastHisto-TVA operating point of the same variant (central values,
    seed mean +- seed std; no bootstrap): efficiency at the FH purity, purity at the FH efficiency."""
    out = {}
    for v in VARIANTS:
        fh = cfgs[(v, "FH")][0]
        y = fh.y.astype(bool)
        fe, fp = float(fh.passed[y].mean()), float((fh.passed & y).sum() / fh.passed.sum())
        out[v] = {"FH_eff": fe, "FH_purity": fp}
        for arm in ARMS:
            e_at_p, p_at_e = [], []
            for cf in cfgs[(v, arm)]:
                o = np.argsort(-cf.score, kind="stable")
                yy = cf.y[o].astype(float)
                tp = np.cumsum(yy)
                eff, pur = tp / yy.sum(), tp / np.arange(1, len(yy) + 1)
                e_at_p.append(float(eff[pur >= fp].max()))
                p_at_e.append(float(pur[np.searchsorted(eff, fe)]))
            out[v][arm] = {"eff_at_FH_purity": [float(np.mean(e_at_p)), float(np.std(e_at_p, ddof=1))],
                           "purity_at_FH_eff": [float(np.mean(p_at_e)), float(np.std(p_at_e, ddof=1))]}
    return out


# ----------------------------------------------------------------------------- closure
def closure(T, E, test, zfh, B):
    """(1) per-event FH z vs the sub2gev code path on the sub2gev caches (same events);
    (2) FH metrics vs the sub2gev summary.json."""
    T2, E2 = S2A.load_cache(S2A.FILES_CLEAN)
    S2 = S2A.selections(T2)
    n2 = E2["ev_event"].size
    key2 = {(int(f), int(e)): i for i, (f, e) in enumerate(zip(E2["ev_file"], E2["ev_event"]))}
    mine = np.flatnonzero(test)
    j = np.array([key2[(int(E["file"][i]) + 1, int(E["event"][i]))] for i in mine])
    out = {}
    for v in VARIANTS:
        z2 = S2A.run_pv(T2, n2, S2[v][0], S2[v][1])
        d = np.abs(zfh[v][mine] - z2[j])
        out[v] = {"n_events": int(len(mine)), "max_abs_dz_cm": float(np.nanmax(d)),
                  "n_differ_gt_1um": int(np.sum(d > 1e-4)), "nan_mismatch": int(np.sum(np.isnan(zfh[v][mine]) != np.isnan(z2[j])))}
    ref = json.load(open(os.path.join(S2A.OUTDIR, "summary.json")))["f03-f10"]["q3_bootstrap"]
    rows = []
    for v in VARIANTS:
        for m, m2 in (("eff", "eff"), ("core_um", "core_um"), ("full_um", "full_um"), ("hs_per_ev", "hs_assoc_per_ev"),
                      ("assoc_eff", "hs_assoc_eff"), ("purity", "assoc_purity")):
            a = B[(v, "FH")][m][0]
            b = ref[f"{v}:{m2}"][0]
            out[v][m] = {"this": float(a), "sub2gev": float(b), "diff": float(a - b)}
            sc, f = MFMT[m]
            rows.append([v, MHDR[m], f.format(a * sc), f.format(b * sc), f"{(a - b) * sc:.2g}"])
    return out, rows


# ----------------------------------------------------------------------------- figures
def fig_window(pdf, cfgs, test_n):
    w = np.logspace(-3, 0.5, 70)

    def curve(key):
        c = np.array([[np.mean(np.abs(cf.dz) < x) for x in w] for cf in cfgs[key]])
        return c.mean(0), c.min(0), c.max(0)

    for group, keys, title in (
            ("by variant", [[(v, m) for m in METHODS] for v in VARIANTS], "per variant: FastHisto vs E2E"),
            ("by method", [[(v, m) for v in VARIANTS] for m in METHODS], "per method: V0 vs V1 vs V3")):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained", sharey=True)
        for ax, ks in zip(axs, keys):
            for k in ks:
                mu, lo, hi = curve(k)
                col = C[k[1]] if group == "by variant" else C[k[0]]
                lab = (MLABEL[k[1]] if group == "by variant" else VLABEL[k[0]])
                ax.plot(w * UM, mu, color=col, ls=LS[k[1]] if group == "by variant" else "-", lw=2, label=lab)
                if k[1] != "FH":
                    ax.fill_between(w * UM, lo, hi, color=col, alpha=0.2, lw=0)
            ax.axvline(PV_WINDOW * UM, color="k", lw=0.8, ls=":")
            ax.set(xscale="log", xlabel="window |z_PV - z_gen| [um]",
                   title=VLABEL[ks[0][0]] if group == "by variant" else MLABEL[ks[0][1]])
            ax.title.set_fontsize(9)
            ax.grid(alpha=0.25, which="both")
        axs[0].set_ylabel("fraction of events (PV-finding efficiency)")
        fig.suptitle(f"PV-finding efficiency vs window, {group} ({test_n} test events; E2E: seed mean, band = seed min-max; "
                     "dotted line 0.15 cm)", fontsize=10)
        legend_below(fig, ncol=3)
        pdf.savefig(fig)
        plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained", sharey=True)
    b = np.linspace(-3000, 3000, 121)
    for ax, v in zip(axs, VARIANTS):
        for m in METHODS:
            d = np.concatenate([cf.dz for cf in cfgs[(v, m)]]) * UM
            wts = np.full(d.size, 1.0 / len(cfgs[(v, m)]))
            ax.hist(np.clip(d, b[0], b[-1]), bins=b, weights=wts, histtype="step", lw=2, color=C[m], ls=LS[m],
                    label=MLABEL[m])
        ax.set(xlabel="z_PV - z_gen [um] (overflow in edge bins)", yscale="log", title=VLABEL[v])
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("events (E2E: per-seed average)")
    fig.suptitle("PV z residual", fontsize=10)
    legend_below(fig, ncol=3)
    pdf.savefig(fig)
    plt.close(fig)


def fig_roc(pdf, cfgs):
    from sklearn.metrics import roc_curve
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained", sharey=True)
    for ax, v in zip(axs, VARIANTS):
        for m in METHODS:
            for i, cf in enumerate(cfgs[(v, m)]):
                fpr, tpr, _ = roc_curve(cf.y, cf.score)
                ax.plot(fpr, tpr, color=C[m], ls=LS[m], lw=1.5 if m == "FH" else 1.0,
                        label=MLABEL[m] + (" (each seed)" if m != "FH" else "") if i == 0 else None)
                y = cf.y.astype(bool)
                ax.plot(np.mean(cf.passed[~y]), np.mean(cf.passed[y]), marker="o" if m == "FH" else "s", ms=5,
                        color=C[m], ls="none",
                        label=("working point (FH: TVA window; E2E: score > 0.5)" if (m == "FH" and i == 0) else None))
        ax.set(xscale="log", xlim=(1e-3, 1), ylim=(0.5, 1.0), xlabel="fraction of non-HS selected tracks associated",
               title=VLABEL[v])
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.25, which="both")
    axs[0].set_ylabel("fraction of HS selected tracks associated")
    fig.suptitle("Track-to-PV association ROC (test events; HS = TP with (evt, bx) = (0, 0))", fontsize=10)
    legend_below(fig, ncol=2)
    pdf.savefig(fig)
    plt.close(fig)


def fig_deltas(pdf, B, seeds):
    """Paired differences with bootstrap errors: MVA gain and the variant split."""
    ms = ("eff", "core_um", "auc", "purity", "hs_per_ev")
    fig, axs = plt.subplots(1, len(ms), figsize=(16, 4.8), layout="constrained")
    comps = [(f"{v}: E2E+MVA - E2E stock", (v, "mva"), (v, "stock"), C[v]) for v in VARIANTS]
    comps += [(f"{a}-{b} ({m})", (a, m), (b, m), C[m]) for m in METHODS for a, b in (("V3", "V0"), ("V1", "V0"), ("V3", "V1"))]
    for ax, m in zip(axs, ms):
        sc = MFMT[m][0]
        for i, (lab, a, b, col) in enumerate(comps):
            d, e = diff(B, a, b, m)
            ax.errorbar(d * sc, -i, xerr=e * sc, fmt="o", color=col, ms=4, capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(-np.arange(len(comps)))
        ax.set_yticklabels([c[0] for c in comps] if m == ms[0] else [""] * len(comps), fontsize=7)
        ax.set_xlabel({"eff": "delta PV eff [%]", "core_um": "delta dz core RMS [um]", "auc": "delta assoc AUC",
                       "purity": "delta purity [%] (WP)", "hs_per_ev": "delta HS assoc / ev"}[m])
        ax.grid(alpha=0.25)
    fig.suptitle("Paired differences (event bootstrap, 1 sigma). Rows 1-3: gain from the MVA inputs per variant; "
                 "then V3-V0 split into (V1-V0) + (V3-V1) per method", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    T, E = load_dataset()
    folds = event_folds(E)
    test = folds >= 0
    cfgs, zfh, extra = build_configs(T, E, test)
    B, seeds = bootstrap_all(cfgs, int(test.sum()))
    clo, clo_rows = closure(T, E, test, zfh, B)
    rep = json.load(open(MVA_STEM + "_report.json"))
    mop = matched_operating_points(cfgs)

    summ = {"n_test_events": int(test.sum()), "n_boot": N_BOOT, "wp_e2e": WP, "pv_window_cm": PV_WINDOW,
            "inputs": extra, "closure": clo, "matched_operating_points": mop, "values": {}, "seed_values": {}, "diffs": {}}
    for key in cfgs:
        k = f"{key[0]}:{key[1]}"
        summ["values"][k] = {m: dict(zip(("value", "boot_err"), val(B, key, m)),
                                     seed_sd=seed_sd(seeds, key, m)) for m in METRICS}
        summ["seed_values"][k] = seeds[key]
    comps = {}
    for v in VARIANTS:
        comps[f"{v}: mva - stock"] = ((v, "mva"), (v, "stock"))
        comps[f"{v}: stock - FH"] = ((v, "stock"), (v, "FH"))
        comps[f"{v}: mva - FH"] = ((v, "mva"), (v, "FH"))
    for m in METHODS:
        for a, b in (("V3", "V0"), ("V1", "V0"), ("V3", "V1")):
            comps[f"{m}: {a} - {b}"] = ((a, m), (b, m))
    for name, (a, b) in comps.items():
        summ["diffs"][name] = {m: dict(zip(("value", "boot_err"), diff(B, a, b, m)),
                                       seed_err=seed_err_of_diff(seeds, a, b, m)) for m in METRICS}

    os.makedirs(OUTDIR, exist_ok=True)
    with PdfPages(f"{OUTDIR}/nnvtx_sp.pdf") as pdf:
        table_page(pdf, "E2E NNVtx on SmartPixels refit tracks: definitions", ["item", "definition"], [
            ["sample", f"canon_v1 mask AAAA, PU200 ttbar; {int(test.sum())} test events (f03-f10), f01/f02 training only"],
            ["V0", VLABEL["V0"] + ": L1TTrack, GTT prompt selection (pT >= 2 GeV, |eta| <= 2.4, ...)"],
            ["V1", VLABEL["V1"] + ": L1TSmartPixelsTrack, same selection"],
            ["V3", VLABEL["V3"] + ": refitted tracks have no pT floor"],
            ["FH", "offline float GTT fastHisto (256 x 0.16 cm, 3-bin window) + GTT eta-binned TVA (sub2gev code)"],
            ["E2E", "ngtagger.train.nnvtx.build_e2e_model (weight MLP 10-10, soft histogram, Conv1D 16-16, "
                    "assoc MLP 20-20), float, trained on the variant's selected tracks"],
            ["stock inputs", "min(pT,128), trkMVA1, |eta|  (+ |z0 - z_PV| for the association net)"],
            ["+MVA inputs", "P(clean), P(d0acc), P(z0acc), log sigma_hat(d0), log sigma_hat(z0); OT heads (V0), "
                            "SP heads (V1/V3); nested cross-fitted (always out of sample)"],
            ["CV", "5 event folds on f03-f10 (+ f01/f02 in every training set); seeds 0,1,2 (init + early-stop split)"],
            ["truth", "PV = GenVtx_z; HS track = its TP has (evt,bx)=(0,0): V0 via L1TTrack_tpIdx, V1/V3 via "
                      "L1TSmartPixelsTrack_tpIdx (cluster majority)"],
            ["WP", "E2E association score > 0.5; FH: inside the TVA window"],
            ["errors", f"+- event bootstrap ({N_BOOT}, paired); [ ] = std over seeds (E2E)"]])
        table_page(pdf, "Closure: FastHisto path vs the sub-2 GeV study (same events, f03-f10)",
                   ["variant", "metric", "this study", "sub2gev summary.json", "difference"], clo_rows,
                   notes=["Per-event z_PV identical to the sub2gev code path: " + "; ".join(
                       f"{v}: max |dz| = {clo[v]['max_abs_dz_cm']:.1e} cm, {clo[v]['n_differ_gt_1um']} events differ > 1 um"
                       for v in VARIANTS)])
        rows = []
        for coll in ("base", "refit"):
            for w, r in rep[coll]["perf"].items():
                rows.append([{"base": "OT (L1TTrack)", "refit": "SP (L1TSmartPixelsTrack)"}[coll], w, r["n_tracks"],
                             f"{r['clean_frac']:.3f}", f"{r['auc_clean']:.3f}", f"{r['auc_d0acc']:.3f}",
                             f"{r['auc_z0acc']:.3f}", f"{r['z0acc_target_um']:.0f}", f"{r['sd0_coverage_68']:.3f}",
                             f"{r['sd0_pull68_all']:.2f}",
                             f"{r['sz0_rank_sigma68_best_frac']['1.0']:.0f} / {r['sz0_rank_sigma68_best_frac']['0.1']:.0f}"])
        table_page(pdf, "Track MVA heads retrained on canon_v1 (outer out-of-sample scores, test events)",
                   ["tracks", "window", "N", "clean frac", "AUC clean", "AUC d0acc", "AUC z0acc", "z0acc target [um]",
                    "sd0 68% coverage", "d0 pull sigma68", "z0 sigma68 all / best 10% by sigma_hat [um]"], rows,
                   notes=["ngtagger train-trackmva --family canonical --refit-truth sp --colls refit base --train-only "
                          "_f01_ _f02_ (nested cross-fit, 5 folds). SP window = fitted |d0| <= / > 200 um. d0-derived "
                          "features and the z0 value are withheld from every head."])
        for v in VARIANTS:
            rows = []
            for m in METHODS:
                rows.append([MLABEL[m]] + [fmt(*val(B, (v, m), k), k, None if m == "FH" else seed_sd(seeds, (v, m), k))
                                           for k in METRICS])
            table_page(pdf, f"{VLABEL[v]}: PV finding and association ({int(test.sum())} test events)",
                       ["method"] + [MHDR[k] for k in METRICS], rows,
                       notes=[f"{extra[v]['sel_per_ev']:.1f} selected tracks / event; {extra[v]['clipped_tracks']} "
                              f"selected tracks beyond the E2E input size (not scored by E2E, counted as not associated).",
                              "+- event bootstrap; [ ] std over the 3 seeds. E2E rows are seed means."])
        rows = []
        for name, (a, b) in comps.items():
            rows.append([name] + [fmt(*diff(B, a, b, k), k, seed_err_of_diff(seeds, a, b, k)) for k in METRICS])
        table_page(pdf, "Paired differences (A - B), same events", ["comparison"] + [MHDR[k] for k in METRICS], rows,
                   notes=["+- paired event bootstrap of the seed-mean difference; [ ] seed-induced error of the "
                          "difference of seed means, sqrt(sA^2/3 + sB^2/3) (FH has no seed error).",
                          "V3 - V0 = (V1 - V0) + (V3 - V1): refit of >= 2 GeV tracks, then admitting sub-2 GeV refit tracks. "
                          "V1 is a diagnostic only."])
        rows = []
        for v in VARIANTS:
            r = [VLABEL[v], f"{100 * mop[v]['FH_eff']:.1f}", f"{100 * mop[v]['FH_purity']:.1f}"]
            for arm in ARMS:
                for k in ("eff_at_FH_purity", "purity_at_FH_eff"):
                    r.append("{:.1f} [{:.1f}]".format(*(100 * np.array(mop[v][arm][k]))))
            rows.append(r)
        table_page(pdf, "Association at the FastHisto operating point (like-for-like with the GTT TVA window)",
                   ["variant", "FH assoc eff %", "FH purity %", "E2E stock: eff % at FH purity",
                    "E2E stock: purity % at FH eff", "E2E+MVA: eff % at FH purity", "E2E+MVA: purity % at FH eff"], rows,
                   notes=["The E2E score > 0.5 working point sits at much higher purity / lower efficiency than the TVA "
                          "window, so the WP columns of the other tables are not like-for-like with FH. Here the E2E "
                          "threshold is moved to the FH purity (or efficiency) of the same variant. Central values, "
                          "seed mean [std over seeds]; no bootstrap."])
        rows = []
        for key in cfgs:
            if key[1] == "FH":
                continue
            for i in range(len(seeds[key]["eff"])):
                rows.append([key[0], key[1], i] + [MFMT[k][1].format(seeds[key][k][i] * MFMT[k][0])
                                                     for k in METRICS] + [str(extra[key[0]]["epochs"][f"{key[1]}_s{i}"])])
        table_page(pdf, "E2E per seed (central values)", ["variant", "arm", "seed"] + [MHDR[k] for k in METRICS]
                   + ["epochs per fold"], rows)
        fig_window(pdf, cfgs, int(test.sum()))
        fig_roc(pdf, cfgs)
        fig_deltas(pdf, B, seeds)
    summ["track_mva_report"] = {c: rep[c]["perf"] for c in ("base", "refit")}
    with open(f"{OUTDIR}/summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(f"wrote {OUTDIR}/nnvtx_sp.pdf and summary.json", flush=True)


if __name__ == "__main__":
    main()
