"""FAST class-C study: vector-PDF figures from fast_eval / fast_ort_bench / fast_quant outputs. One shared legend OUTSIDE the axes,
statistics on separate table pages.
  fast_plots.py <variant> [...]      the variants to overlay (as in fast_eval: <cnn>:<draw>); ref:knn always first
Figures (cluster_cnn/figs/): fast_classC_closure.pdf, fast_pit.pdf, fast_corr.pdf, fast_emitted_CvsT.pdf, fast_cnn_tradeoff.pdf,
fast_ort.pdf (if the bench JSON exists).
"""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fast_common import FAST, RES, FIGS

VARS = sys.argv[1:] or ["ref:knn"]
GROUPS = [("barrel", "IL2-4"), ("barrel", "IL1"), ("disc", "ID1-4")]
ANG = ("alpha", "beta")
COL = ["#222222", "#e8742c", "#2a78d6", "#1f9e76", "#c23b8a", "#8c6d31", "#7a5cc7", "#d4a017", "#5aa0a8", "#999999"]
HB = {"alpha": np.linspace(-70, 70, 141), "beta": np.linspace(-90, 90, 181)}; HR = np.linspace(-60, 60, 241)
vn = lambda v, reg: v.replace("{reg}", reg)          # variants may be region templates, e.g. "{reg}_tiny:a2"
H = lambda reg, v: dict(np.load(os.path.join(FAST, "hists", f"{reg}__{vn(v, reg).replace(':', '__')}.npz")))
E = {reg: pd.read_csv(os.path.join(RES, f"fast_eval_{reg}.csv")) for reg in ("barrel", "disc") if os.path.exists(os.path.join(RES, f"fast_eval_{reg}.csv"))}
LABEL = {"knn": "knn donor (10-NN, 11 obs)", "nodonor": "no donor (CNN only)", "a1arh": "A1 autoregr. histogram (11 obs)",
         "a1mdn": "A1 bivariate MDN (11 obs)", "a2": "A2 single network (image)"}
lab = lambda v: f"{LABEL.get(v.split(':')[1], v.split(':')[1])} | CNN {v.split(':')[0].replace('{reg}_', 'new ')}"


def table_pages(pdf, df, title, per=28):
    for i in range(0, len(df), per):
        t = df.iloc[i:i + per]
        f, a = plt.subplots(figsize=(min(26, 2 + 1.5 * len(t.columns)), 0.33 * len(t) + 1.4)); a.axis("off")
        tb = a.table(cellText=t.astype(str).values, colLabels=list(t.columns), loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.auto_set_column_width(list(range(len(t.columns))))
        a.set_title(title + (f" ({i // per + 1})" if len(df) > per else ""), fontsize=10); pdf.savefig(f, bbox_inches="tight"); plt.close(f)


def fmt(df):
    return df.apply(lambda c: c.map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x))


def closure_pdf():
    with PdfPages(os.path.join(FIGS, "fast_classC_closure.pdf")) as pdf:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        for j, (reg, g) in enumerate(GROUPS):
            if reg not in E:
                continue
            T = H(reg, "classT_route")
            for i, a in enumerate(ANG):
                x = ax[i, j]; h = T[f"res_T_{g}_{a}"]; x.stairs(h / h.sum(), HR, color="#9e9e9e", lw=2.5, label="class T route (the target)")
                for k, v in enumerate(VARS):
                    try:
                        h = H(reg, v)[f"res_C_{g}_{a}"]
                    except (FileNotFoundError, KeyError):
                        continue
                    x.stairs(h / h.sum(), HR, color=COL[k % len(COL)], lw=1.3, label=lab(v))
                x.set_yscale("log"); x.set_ylim(1e-6, 0.3); x.set_title(f"{g} CMS {a}: class-C route output - truth (class-T clusters)", fontsize=9)
                x.set_xlabel("residual [deg]"); x.grid(True, color="#eeeeee")
        hl = {}
        for x in ax.flat:
            for hh, ll in zip(*x.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=3, fontsize=8, frameon=False)
        pdf.savefig(fig); plt.close(fig)
        for reg in E:
            d = E[reg]; d = d[(d.angle != "corr") & (d.variant.isin([vn(v, reg) for v in VARS] + ["classT_route"]))]
            cols = ["variant", "group", "angle", "core", "full", "gt10", "cnn_core", "cnn_full", "leak_strong", "leak_F1", "leak_own", "sat_C"]
            table_pages(pdf, fmt(d[[c for c in cols if c in d]]), f"{reg}: class-C closure (C route - truth on class T) [deg], leak AUC gain")


def pit_pdf():
    with PdfPages(os.path.join(FIGS, "fast_pit.pdf")) as pdf:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True); e = np.linspace(0, 1, 21)
        for j, (reg, g) in enumerate(GROUPS):
            if reg not in E:
                continue
            for i, a in enumerate(ANG):
                x = ax[i, j]
                for k, v in enumerate(VARS):
                    if v.endswith("nodonor"):
                        continue
                    try:
                        h = H(reg, v)[f"pit_{g}_{a}"]
                    except (FileNotFoundError, KeyError):
                        continue
                    if h.sum():
                        x.stairs(h / h.sum() * 20, e, color=COL[k % len(COL)], lw=1.3, label=lab(v))
                x.axhline(1, color="#9e9e9e", lw=2, label="perfect calibration"); x.set_ylim(0, 2.5)
                x.set_title(f"{g} {a}: PIT of truth ({'marginal' if a == 'alpha' else 'conditional on true alpha residual; knn: marginal'})", fontsize=8.5)
                x.set_xlabel("PIT"); x.grid(True, color="#eeeeee")
        hl = {}
        for x in ax.flat:
            for hh, ll in zip(*x.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=3, fontsize=8, frameon=False)
        pdf.savefig(fig); plt.close(fig)
        for reg in E:
            d = E[reg]; d = d[(d.angle != "corr") & d.variant.isin([vn(v, reg) for v in VARS])]
            table_pages(pdf, fmt(d[["variant", "group", "angle", "pit_ece", "pit_tail"]]),
                        f"{reg}: PIT calibration (ECE = mean |F(q)-q|, q=0.05..0.95; tail = P(PIT<0.025 or >0.975), 0.05 expected)")


def corr_pdf():
    with PdfPages(os.path.join(FIGS, "fast_corr.pdf")) as pdf:
        vs = [v for v in VARS if not v.endswith("nodonor")]
        for reg, g in GROUPS:
            if reg not in E:
                continue
            fig, ax = plt.subplots(1, len(vs) + 1, figsize=(4 * (len(vs) + 1), 4.6), constrained_layout=True)
            ex, ey = np.linspace(-30, 30, 61), np.linspace(-60, 60, 61)
            items = [("TRUE residual (class T)", H(reg, vs[0])[f"r2d_true_{g}"])] + [(lab(v), H(reg, v).get(f"r2d_draw_{g}")) for v in vs]
            for x, (t, h) in zip(np.atleast_1d(ax), items):
                if h is None:
                    continue
                im = x.pcolormesh(ex, ey, np.log10(h.T / h.sum() + 1e-7), vmin=-7, vmax=-1, cmap="viridis")
                x.set_title(t, fontsize=7.5); x.set_xlabel("r_alpha [deg]"); x.set_ylabel("r_beta [deg]")
            fig.colorbar(im, ax=ax, location="bottom", shrink=0.4, label="log10 fraction"); fig.suptitle(f"{g}: drawn (r_alpha, r_beta) vs true", fontsize=10)
            pdf.savefig(fig); plt.close(fig)
        for reg in E:
            d = E[reg]; d = d[(d.angle == "corr") & d.variant.isin([vn(v, reg) for v in VARS])]
            table_pages(pdf, fmt(d[["variant", "group", "pearson_draw", "pearson_true", "spearman_draw", "spearman_true", "quadL1", "quad_draw", "quad_true"]]),
                        f"{reg}: alpha/beta dependence of the drawn residual vs truth (class T); quad = (++, +-, -+, --) fractions with both |r| > 5")


def emitted_pdf():
    with PdfPages(os.path.join(FIGS, "fast_emitted_CvsT.pdf")) as pdf:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        for j, (reg, g) in enumerate(GROUPS):
            if reg not in E:
                continue
            T = H(reg, "classT_route")
            for i, a in enumerate(ANG):
                x = ax[i, j]; e = HB[a]; h = T[f"emit_T_{g}_{a}"]
                x.stairs(h / h.sum() / np.diff(e), e, color="#9e9e9e", lw=2.5, label="class T: emitted")
                for k, v in enumerate(VARS):
                    try:
                        h = H(reg, v)[f"emit_CU_{g}_{a}"]
                    except (FileNotFoundError, KeyError):
                        continue
                    x.stairs(h / h.sum() / np.diff(e), e, color=COL[k % len(COL)], lw=1.3, label="class C unlinked: " + lab(v))
                x.set_yscale("log"); x.set_ylim(1e-6, 0.3); x.grid(True, color="#eeeeee")
                x.set_title(f"{g} CMS {a}: emitted angle", fontsize=9); x.set_xlabel("angle [deg]"); x.set_ylabel("density")
        hl = {}
        for x in ax.flat:
            for hh, ll in zip(*x.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=2, fontsize=8, frameon=False)
        pdf.savefig(fig); plt.close(fig)
        for reg in E:
            d = E[reg]; d = d[(d.angle != "corr") & d.variant.isin([vn(v, reg) for v in VARS])]
            cols = ["variant", "group", "angle", "emit_T_rms", "emit_CU_rms", "emit_CL_rms", "emit_CU_gt60", "emit_CL_gt60",
                    "emit_CU_KS_vs_knnref", "emit_CL_KS_vs_knnref", "emit_CU_KS_vs_T"]
            table_pages(pdf, fmt(d[[c for c in cols if c in d]]), f"{reg}: emitted angles, class C (CU unlinked, CL linked bad truth) vs class T and vs knn(ref)")


if __name__ == "__main__":
    closure_pdf(); pit_pdf(); corr_pdf(); emitted_pdf()
    print("wrote", [f for f in os.listdir(FIGS) if f.startswith("fast_")])
