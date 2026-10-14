#!/usr/bin/env python
"""P0.3 step 5: the summary PDF (vector; legends outside the axes; units on every axis) from the
JSON outputs of table.py, roc.py and extras.py.

    python plots.py [--out .../p03_summary.pdf]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402
import table as T  # noqa: E402

from ngtagger.spdefs import angles as SA  # noqa: E402

# reference palette (dataviz skill), categorical slots in fixed order; text in ink, not series colour
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42})
A4L = (11.69, 8.27)


PNG_DIR: list = []          # optional PNG copies of every page (visual checks)


def save(pdf, fig):
    pdf.savefig(fig)
    if PNG_DIR:
        n = len(list(Path(PNG_DIR[0]).glob("page_*.png"))) + 1
        fig.savefig(Path(PNG_DIR[0]) / f"page_{n:02d}.png", dpi=80)
    plt.close(fig)


def load(name):
    p = CM.OUT_DIR / name
    return json.load(open(p)) if p.exists() else None


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=A4L)
    fig.text(0.05, 0.95, title, fontsize=13, color=INK, va="top", weight="bold")
    fig.text(0.05, 0.90, "\n".join(lines), fontsize=8, color=INK, va="top", family="monospace")
    save(pdf, fig)


def key_label(surface_name):
    return "signed 1/cotθ (= tanθ) [dimensionless]" if surface_name.startswith("ID") else \
        "signed cotθ [dimensionless]"


def bin_x(lo, hi, cap):
    lo, hi = max(lo, -cap), min(hi, cap)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def pulls_pages(pdf, res, title):
    """Per key x angle and bin: the weighting width and the centre of the test pulls before the table
    (published and census sigma) and after it (the (i) statistics), with the adequacy bands."""
    if not res or "bins" not in res:
        return
    for key, rows in res["bins"].items():
        for ang in SA.ANGLES:
            groups = {}
            for r in rows:
                groups.setdefault(f"{r['surface']} f{r['facing']:+d}" if r["surface"].startswith("ID") else r["surface"],
                                  []).append(r)
            names = list(groups)
            for what, band, ylab in (("width", (1 - T.BAND_W, 1 + T.BAND_W), "pull weighting width [published σ units]"),
                                     ("centre", (-T.BAND_C, T.BAND_C), "pull centre [published σ units]")):
                fig, axs = plt.subplots(3, 4, figsize=A4L, constrained_layout=True)
                for ax, g in zip(axs.flat, names):
                    cap = 3.0 if g.startswith("IL") else 1.0
                    xs = [bin_x(r["lo"], r["hi"], cap) for r in groups[g]]
                    x = np.array([v[0] for v in xs])
                    xe = np.array([v[1] for v in xs])

                    def get(r, k):
                        d = r[ang].get(k)
                        return np.nan if not d else d[what]
                    yb = np.array([get(r, "before") for r in groups[g]], float)
                    ya = np.array([r[ang]["after"][what] for r in groups[g]], float)
                    sa = np.array([r[ang]["after"]["sd_" + what] for r in groups[g]], float)
                    ax.axhspan(*band, color=GRID, alpha=0.6, lw=0, label="adequacy band (i)")
                    ax.plot(x, yb, "o", color=C2, ms=3, label="before (published σ)")
                    if g.startswith("IL"):
                        yc = np.array([get(r, "before_census") for r in groups[g]], float)
                        ax.plot(x, yc, "s", color=C3, ms=3, label="before (today's census σ)")
                    ax.errorbar(x, ya, xerr=xe, yerr=sa, fmt="o", color=C1, ms=3, lw=1, label="after the table (test)")
                    ax.set_title(g, fontsize=8, color=INK)
                    ax.set_xlabel(key_label(g), fontsize=7)
                    ax.set_ylabel(ylab, fontsize=7)
                    if what == "width":
                        ax.set_ylim(0, max(3.0, np.nanmax(np.r_[yb, ya]) * 1.05) if np.isfinite(np.nanmax(np.r_[yb, ya])) else 3)
                for ax in list(axs.flat)[len(names):]:
                    ax.axis("off")
                h, lab = axs.flat[0].get_legend_handles_labels()
                fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=7)
                fig.suptitle(f"{title}: {ang} pull {what} per bin, {key} key", color=INK, fontsize=10)
                save(pdf, fig)


def coverage_page(pdf, res, title):
    if not res or "bins" not in res:
        return
    fig, axs = plt.subplots(2, 2, figsize=A4L, constrained_layout=True)
    t = res["judgement"]["ii"]["t_N"]
    for ax, lv in zip(axs.flat, SA.LEVELS):
        z = []
        for key, rows in res["bins"].items():
            for r in rows:
                for ang in SA.ANGLES:
                    c = r[ang]["coverage"][f"{lv:g}"]
                    if r["n_test"] >= SA.MIN_FIT and c["z"] is not None and np.isfinite(c["z"]):
                        z.append(c["z"])
        z = np.array(z)
        ax.hist(z, bins=np.linspace(-8, 8, 65), color=C1, label=f"bins (N = {len(z)})")
        xx = np.linspace(-8, 8, 400)
        ax.plot(xx, len(z) * 0.25 * np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color=C2, label="unit Gaussian")
        for s in (-t, t):
            ax.axvline(s, color=INK2, ls="--", lw=1)
        ax.set_xlabel(f"coverage z at {lv:.1%} [exact-shape null, amendments 1-1e; σ units]")
        ax.set_ylabel("bins [count]")
        ax.set_title(f"level {lv:.1%}; dashed: ±t(N) = {t:.2f}", fontsize=8)
    h, lab = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    fig.suptitle(f"{title}: coverage z-scores (both keys, both angles)", fontsize=10)
    save(pdf, fig)


def roc_pages(pdf, roc):
    if not roc:
        return
    cmb = roc["combos"]
    Lg = roc["definitions"]["level_grid"]
    lvl_idx = [Lg.index(lv) for lv in SA.LEVELS]
    for floor in (2.0, 1.5, 1.0):
        fig, axs = plt.subplots(2, 5, figsize=A4L, constrained_layout=True)
        groups = [SA.SURFACE_NAMES[s] for s in SA.SURFACES] + ["barrel", "discs"]
        for ax, g in zip(axs.flat, groups):
            for cut, col in zip(("S1", "beta", "both"), (C1, C2, C3)):
                n = cmb[f"F{floor:g}/k1/fixed/{cut}"]["nominal"][g]
                e, r = np.array(n["eff_sig_all"], float), np.array(n["retention"], float)
                ax.plot(e, r, "-", color=col, label=f"{cut} (all signal)")
                ax.plot(e[lvl_idx], r[lvl_idx], "o", color=col, ms=4)
                eT = np.array(n["eff_sig_T"], float)
                ax.plot(eT, r, ":", color=col, lw=1, label=f"{cut} (class-T signal)")
            ax.set_xlim(0.9, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(g, fontsize=8)
            ax.set_xlabel("signal efficiency [fraction of signal clusters]", fontsize=7)
            ax.set_ylabel("cluster retention [fraction of all clusters]", fontsize=7)
        h, lab = axs.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=6, fontsize=7)
        fig.suptitle(f"Cluster-level angle ROC, pT floor {floor:g} GeV, k = 1 (dots: the 95 / 97.5 / 99 / 99.7 % levels)",
                     fontsize=10)
        save(pdf, fig)
    # retention at 99 % signal efficiency vs k
    fig, axs = plt.subplots(1, 3, figsize=A4L, constrained_layout=True)
    for ax, cut in zip(axs, ("S1", "beta", "both")):
        for g, col in zip(("barrel", "discs", "IL1"), (C1, C2, C3)):
            for mode, ls in (("fixed", "-"), ("scaled", "--")):
                ks, ys = [], []
                for k in (1.0, 0.67, 0.0):
                    m = "fixed" if k == 1.0 else mode
                    c = cmb.get(f"F2/k{k:g}/{m}/{cut}")
                    if c:
                        ks.append(k)
                        ys.append(c["ret_at_99"][g]["sig_all"])
                ax.plot(ks, ys, ls, marker="o", color=col, label=f"{g}, {mode}")
        ax.set_xlabel("angle improvement k [dimensionless; 1 = emulated]")
        ax.set_ylabel("retention at 99 % signal efficiency [fraction]")
        ax.set_title(f"{cut}, floor 2 GeV, all-class signal", fontsize=8)
        ax.invert_xaxis()
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=6)
    fig.suptitle("Cluster retention at 99 % signal efficiency vs the angle improvement k", fontsize=10)
    save(pdf, fig)


def response_page(pdf):
    p = CM.OUT_DIR / "p03_response.npz"
    if not p.exists():
        return
    Z = np.load(p)
    meta = json.loads(str(Z["meta"]))
    e = np.array(meta["local_deg_edges"])[:-1]
    e2 = np.r_[e[::4], e[-1] + 0.5]
    for ang in SA.ANGLES:
        fig, axs = plt.subplots(3, 4, figsize=A4L, constrained_layout=True)
        for ax, g in zip(axs.flat, meta["groups"]):
            h = Z[f"{g['tag']}/{ang}/local_emitted_vs_true_deg"].astype(float)
            h = h.reshape(len(e) // 4, 4, len(e) // 4, 4).sum((1, 3))      # 0.5 -> 2 deg for display
            norm = h.sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                ph = np.where(norm > 0, h / norm, np.nan)
            nz = np.where(h.sum(1) > 0)[0]
            if len(nz) == 0:
                ax.axis("off")
                continue
            lo, hi = e2[max(nz[0] - 2, 0)], e2[min(nz[-1] + 3, len(e2) - 1)]
            im = ax.pcolormesh(e2, e2, ph.T, cmap="Blues", vmin=0, vmax=np.nanpercentile(ph, 99.5), rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=C2, lw=0.8, label="emitted = true")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{g['tag']} (n = {g['n']})", fontsize=8)
            ax.set_xlabel(f"true atan(local cot{ang}) [deg]", fontsize=7)
            ax.set_ylabel(f"emitted atan(local cot{ang}) [deg]", fontsize=7)
            fig.colorbar(im, ax=ax, label="p(emitted | true) per 2 deg [fraction]")
        for ax in list(axs.flat)[len(meta["groups"]):]:
            ax.axis("off")
        h_, lab = axs.flat[0].get_legend_handles_labels()
        fig.legend(h_, lab, loc="outside lower center")
        fig.suptitle(f"Class-T response p(emitted | true), {ang}, module frame, files 1-2, pT >= 1 GeV", fontsize=10)
        save(pdf, fig)


def summary_lines(kg, kb, gt, roc, ex):
    L = []
    for name, r in (("amendment-1e known-good demonstration (fit 7-8, test 3-6, 9-10)", kg),
                    ("gated test (fit 1-2, test 3-10)", gt)):
        if not r:
            L.append(f"{name}: not run")
            continue
        j = r["judgement"]
        L.append(f"{name}: PASS = {j['pass']}, verdict {j['materiality']['verdict']}  [(i) {j['i']['fail']} / {j['i']['bins']} bins outside the widened band, "
                 f"worst {j['i']['worst_fraction_of_widened_band']:.2f} of it; (ii) N = {j['ii']['N']}, t(N) = "
                 f"{j['ii']['t_N']:.2f}, max |z| = {j['ii']['max_abs_z']:.2f}, beyond t: {j['ii']['n_beyond_t']}, "
                 f"chi2 failing: {j['ii']['chi2_fail']}]")
        for k, v in j["per_region"].items():
            L.append(f"    {k:16s} bins {v['bins']:4d}  max|z| {v['max_abs_z']:.2f}  worst band fraction {v['worst_band_fraction']:.2f}")
    if kb:
        for n, v in kb["known_bads"].items():
            j = v["judgement"]
            L.append(f"known-bad {n:15s}: fails = {v['fails_as_required']}  [(i) {j['i']['fail']} bins; (ii) beyond t "
                     f"{j['ii']['n_beyond_t']}, max |z| {j['ii']['max_abs_z']:.1f}; (iii) {len(j['iii']['fail_cells'])} cells]")
    if roc:
        L.append("")
        L.append("ROC, floor 2 GeV, k = 1: retention at 99 % signal efficiency (all signal / class-T signal)"
                 + ("" if gt and gt["judgement"]["pass"] else "  [cut centres from the UNTESTED table: provisional]"))
        for cut in ("S1", "beta", "both"):
            c = roc["combos"][f"F2/k1/fixed/{cut}"]["ret_at_99"]
            L.append(f"    {cut:5s} " + "  ".join(f"{g} {c[g]['sig_all']:.3f}/{c[g]['sig_T']:.3f}"
                                              for g in ["barrel", "discs"] + [SA.SURFACE_NAMES[s] for s in SA.SURFACES]))
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CM.OUT_DIR / "p03_summary.pdf"))
    ap.add_argument("--png-dir", help="also write every page as PNG here (visual checks)")
    a = ap.parse_args(argv)
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
        for f in Path(a.png_dir).glob("page_*.png"):
            f.unlink()
        PNG_DIR.append(a.png_dir)
    kg, kb, gt = (load(T.result_name(m)) for m in ("known-good", "known-bad", "gated"))
    roc, ex = load("p03_roc.json"), load("p03_extras.json")
    with PdfPages(a.out) as pdf:
        text_page(pdf, "P0.3 angle definitions, uncertainty table and cluster-level ROC (canon_v3 AAAA)",
                  [f"Provenance: canon_v3 TTbar AAAA files 1-10, 150X D121 PU200; definition sheet {SA.SHEET_VERSION}; "
                   "ngtagger.spdefs.",
                   "Bias: emulated class-T angles (truth + 1.5 x PixelAV residual); the table models the emulator.", ""]
                  + summary_lines(kg, kb, gt, roc, ex))
        pulls_pages(pdf, gt or kg, "gated test" if gt else "known-good demonstration")
        coverage_page(pdf, kg, "known-good demonstration")
        if gt:
            coverage_page(pdf, gt, "gated test")
        roc_pages(pdf, roc)
        response_page(pdf)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
