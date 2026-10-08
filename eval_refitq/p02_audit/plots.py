#!/usr/bin/env python
"""P0.2 refit-v2 audit, step 3: the summary PDF from p02_results.json (vector; legends outside; units on axes).

    python plots.py [--json .../p02_results.json] [--out .../p02_summary.pdf]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

OUT_DIR = Path("/Volumes/WDMac/smartpixels_scratch/p02_refit_audit")
SCEN = ("AAAA", "AAAI", "IAAA")
# fixed identity colours (reference palette slots 1-3; OT = neutral baseline), the same on every page
COL = {"AAAA": "#2a78d6", "AAAI": "#eb6834", "IAAA": "#1baf7a", "OT": "#7a7974"}
MK = {"AAAA": "o", "AAAI": "s", "IAAA": "^", "OT": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ETA = ("0-1.1", "1.1-1.8", "1.8-2.4")
PT = ("2-3", "3-5", "5-10", "10-inf")
PT_LAB = ("2-3", "3-5", "5-10", ">10")
SURF = ("IL4", "IL3", "IL2", "IL1", "ID4", "ID3", "ID2", "ID1")
ORG = ("prompt", "hf", "displaced_inner", "beyond_il1")
ORG_LAB = {"all": "all origins", "prompt": "prompt", "hf": "b/c daughter", "displaced_inner": "other displaced",
           "beyond_il1": "born beyond IL1"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "errorbar.capsize": 2, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
A4L = (11.69, 8.27)


def v(d, scale=100.0):
    return (None, None) if not d or d.get("v") is None else (d["v"] * scale, (d.get("se") or 0.0) * scale)


def fmt(d, scale=100.0, nd=1, unit="%"):
    x, e = v(d, scale)
    return "n/a" if x is None else f"{x:.{nd}f} ± {e:.{nd}f}{unit}"


def legend_right(fig, handles=None, labels=None, title=None):
    if handles is None:
        handles, labels = [], []
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=False, title=title)


def header(fig, title, sub):
    fig.text(0.04, 0.965, title, fontsize=12, weight="bold", va="top")
    fig.text(0.04, 0.937, "\n".join(textwrap.wrap(sub, 175)), fontsize=8, color=INK2, va="top")


def logfmt(ax, axis="y"):
    """plain-number tick labels on a log axis (1, 2, 5 per decade)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    a.set_minor_formatter(NullFormatter())


def scen_points(ax, xs, getter, series=SCEN, dx=0.12, scale=100.0, line=True, positive=False):
    n = len(series)
    for i, s in enumerate(series):
        off = (i - (n - 1) / 2) * dx
        ys, es, xx = [], [], []
        for j, x in enumerate(xs):
            y, e = v(getter(s, x), scale)
            if y is not None and (y > 0 or not positive):
                xx.append(j + off); ys.append(y); es.append(min(e, 0.95 * y) if positive else e)
        lab = "OT (= IIII baseline)" if s == "OT" else f"SmartPixels {s}"
        ax.errorbar(xx, ys, yerr=es, fmt=MK[s], color=COL[s], ms=4.5, lw=1.2, label=lab, ls="-" if line else "none")


def scen_bars(ax, cats, getter, series=SCEN, scale=100.0, width=0.26):
    n = len(series)
    for i, s in enumerate(series):
        xs, ys, es = [], [], []
        for j, c in enumerate(cats):
            y, e = v(getter(s, c), scale)
            if y is not None:
                xs.append(j + (i - (n - 1) / 2) * width); ys.append(y); es.append(e)
        ax.bar(xs, ys, width=width * 0.92, color=COL[s], label=f"SmartPixels {s}" if s != "OT" else "OT", zorder=2)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, lw=0.8, zorder=3)


def page_title(pdf, R):
    P = R["provenance"]
    fig = plt.figure(figsize=A4L)
    header(fig, "P0.2: refit-v2 do-no-harm audit, fallback to the OT track, benefit baselines",
           "canon_v2 TTbar PU200, files 3-10 (800 events), scenarios AAAA / AAAI / IAAA; OT tracks identical in all "
           "three (= the IIII baseline); errors: event-level bootstrap, "
           f"{R['bootstrap']['replicas']} replicas, every fit re-done per replica")
    pop = R["population"]
    lines = [
        f"Status: {P['status']}.   Bias: {P['bias']}",
        "",
        f"Code: {P['code_repo']} @ {P['code_commit'][:12]} ({P['code_files_uncommitted']}); "
        f"shared-tree patch sha256 {P['tree_patch_sha256'][:16]} (provenance_tree.patch)",
        f"Inputs: {len(P['inputs'])} files, sha256 in p02_results.json / inputs.sha256; build {P['input_build']}",
        f"Generated {P['generated']}; python {P['python']}, numpy {P['numpy']}, uproot {P['uproot']}",
        "",
        f"Population: {pop['n_population']:,} truth-matched OT tracks with true pT >= 2 GeV, |eta| < 2.4 "
        f"(of {pop['n_tracks']:,}; {pop['excluded_tp_pt_lt_2']:,} with true pT < 2 GeV excluded). "
        f"Prompt {pop['per_origin']['prompt']:,}, b/c daughters {pop['per_origin']['hf']:,}, other displaced "
        f"{pop['per_origin']['displaced_inner']:,}, born beyond IL1 {pop['per_origin']['beyond_il1']:,}; "
        f"hard-interaction share {100 * pop['hard_interaction_share']:.1f}%.",
        "",
        "Definitions (choices to weigh are marked *):",
    ]
    D = R["definitions"]
    for k in ("harm", "origin", "core", "hit_class", "first_wrong_hit", "wrong_hit_origin", "P", "attachment",
              "fallback"):
        val = D[k] if isinstance(D[k], str) else "; ".join(f"{a}: {b}" for a, b in D[k].items())
        lines += textwrap.wrap(f"- {k}: {val}", 175, subsequent_indent="    ")
    lines += ["", "* prompt radius 100 um, IL1 radius 2.75 cm, shared threshold 0.9, true-pT >= 2 GeV population, "
              "b/c pT split at 5 GeV, fallback resolution = B.1 core per reconstructed bin, one global threshold."]
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, 190, subsequent_indent="    ") or [""]
    fig.text(0.04, 0.89, "\n".join(wrapped), fontsize=7.3, va="top", family="DejaVu Sans")
    pdf.savefig(fig); plt.close(fig)


def page_harm_bins(pdf, R):
    fig, axs = plt.subplots(3, 3, figsize=A4L, sharex=True)
    header(fig, "Harm of prompt tracks per |eta| x pT bin (H, the replay reference)",
           "Harm = SmartPixels worse than OT by > 0.5 mm in |d0 - true| or > 1 mm in |z0 - true|; "
           "prompt = not a b/c daughter, born within 100 um of the beam line")
    for r, hk in enumerate(("d0", "z0", "either")):
        for c, e in enumerate(ETA):
            ax = axs[r, c]
            scen_points(ax, PT, lambda s, p: R["scenarios"][s]["reference_E5"]["H"][f"eta{e}|pt{p}"][hk])
            ax.set_xticks(range(4)); ax.set_xticklabels(PT_LAB)
            if r == 0:
                ax.set_title(f"|eta| {e}")
            if r == 2:
                ax.set_xlabel("true pT bin [GeV]")
            ax.set_ylabel({"d0": "harmed in d0 [% of tracks]", "z0": "harmed in z0 [% of tracks]",
                           "either": "harmed in d0 or z0 [%]"}[hk])
            ax.set_ylim(bottom=0)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.88, bottom=0.08, hspace=0.25, wspace=0.28)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_harm_origin(pdf, R):
    cats = ["prompt|incl", "hf|eta0-2.4|pt2-5", "hf|eta0-2.4|pt5-inf", "displaced_inner|incl", "beyond_il1|incl"]
    labs = ["prompt", "b/c\n2-5 GeV", "b/c\n>= 5 GeV", "other\ndisplaced", "beyond\nIL1"]
    fig, axs = plt.subplots(1, 3, figsize=A4L)
    header(fig, "Harm by origin (all |eta|; b/c daughters in two true-pT bins)",
           "Same harm definition; origin from the particle's genealogy (b/c ancestor) and production radius: prompt "
           "< 100 um; other displaced 0.1 mm-2.75 cm (not b/c); beyond IL1 > 2.75 cm (not b/c); log scale")
    for ax, hk in zip(axs, ("d0", "z0", "either")):
        scen_points(ax, cats, lambda s, k: R["t2"][k]["harm"][s][hk], line=False, dx=0.18, positive=True)
        ax.set_yscale("log"); logfmt(ax)
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(labs, fontsize=7.5)
        ax.set_ylabel({"d0": "harmed in d0 (> 0.5 mm) [% of tracks]", "z0": "harmed in z0 (> 1 mm) [% of tracks]",
                       "either": "harmed in d0 or z0 [% of tracks]"}[hk])
        ax.set_xlabel("track origin")
        ax.set_title({"d0": "d0", "z0": "z0", "either": "d0 or z0"}[hk])
    fig.subplots_adjust(left=0.06, right=0.84, top=0.86, bottom=0.16, wspace=0.3)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_resolution(pdf, R):
    fig, axs = plt.subplots(2, 3, figsize=A4L, sharex=True)
    header(fig, "Resolution core of prompt tracks: SmartPixels vs OT",
           "Core = RMS about zero of a mode-seeded iterative 3-sigma clip (plan B.1); bins in true pT")
    for r, p in enumerate(("d0", "z0")):
        for c, e in enumerate(ETA):
            ax = axs[r, c]
            scen_points(ax, PT, lambda s, pt: R["t2"][f"prompt|eta{e}|pt{pt}"]["arms"][s][p]["core"],
                        series=("OT",) + SCEN, scale=1.0, dx=0.14, positive=True)
            ax.set_yscale("log"); logfmt(ax)
            ax.set_xticks(range(4)); ax.set_xticklabels(PT_LAB)
            ax.set_ylabel(f"{p} core width [um]")
            if r == 0:
                ax.set_title(f"|eta| {e}")
            else:
                ax.set_xlabel("true pT bin [GeV]")
    fig.subplots_adjust(left=0.07, right=0.84, top=0.88, bottom=0.09, hspace=0.18, wspace=0.28)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_tails(pdf, R):
    fig, axs = plt.subplots(2, 3, figsize=A4L)
    header(fig, "Tails beyond common absolute thresholds, prompt tracks (all pT)",
           "Fraction of tracks with |reconstructed - true| above each threshold; the same thresholds for every arm")
    for r, (p, thr) in enumerate((("d0", ("0.1", "0.2", "0.5", "1", "5")), ("z0", ("0.5", "1", "5")))):
        for c, e in enumerate(ETA):
            ax = axs[r, c]
            scen_points(ax, thr, lambda s, t: R["t2"][f"prompt|eta{e}|incl"]["arms"][s][p][f"frac_gt_{t}mm"],
                        series=("OT",) + SCEN, dx=0.12, positive=True)
            ax.set_yscale("log"); logfmt(ax)
            ax.set_xticks(range(len(thr))); ax.set_xticklabels(thr)
            ax.set_xlabel(f"|{p} residual| threshold [mm]")
            ax.set_ylabel(f"tracks beyond threshold [%]")
            if r == 0:
                ax.set_title(f"|eta| {e}")
    fig.subplots_adjust(left=0.07, right=0.84, top=0.88, bottom=0.09, hspace=0.35, wspace=0.28)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_pulls(pdf, R):
    fig, axs = plt.subplots(1, 3, figsize=A4L)
    header(fig, "Reported uncertainties: pull core and coverage, prompt tracks (all pT)",
           "Pull = residual / reported sigma (cov_d0_d0, cov_z0_z0 of each collection); a calibrated error gives "
           "core 1, coverage 68.3% (|pull| < 1)")
    cats = [(e, p) for p in ("d0", "z0") for e in ETA]
    labs = [f"{p}\n{e}" for e, p in cats]
    scen_points(axs[0], cats, lambda s, k: R["t2"][f"prompt|eta{k[0]}|incl"]["arms"][s][k[1]]["pull_core"],
                series=("OT",) + SCEN, scale=1.0, dx=0.14, line=False)
    axs[0].set_ylabel("pull core width [1]"); axs[0].axhline(1, color=INK2, lw=0.8, ls=":")
    scen_points(axs[1], cats, lambda s, k: R["t2"][f"prompt|eta{k[0]}|incl"]["arms"][s][k[1]]["pull_cov1"],
                series=("OT",) + SCEN, dx=0.14, line=False)
    axs[1].set_ylabel("tracks with |pull| < 1 [%]"); axs[1].axhline(68.3, color=INK2, lw=0.8, ls=":")
    scen_points(axs[2], cats, lambda s, k: R["t2"][f"prompt|eta{k[0]}|incl"]["arms"][s][k[1]]["pull_cov2"],
                series=("OT",) + SCEN, dx=0.14, line=False)
    axs[2].set_ylabel("tracks with |pull| < 2 [%]"); axs[2].axhline(95.4, color=INK2, lw=0.8, ls=":")
    for ax in axs:
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(labs, fontsize=6.8)
        ax.set_xlabel("parameter and true |eta| bin")
    fig.subplots_adjust(left=0.06, right=0.84, top=0.86, bottom=0.14, wspace=0.3)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_first_wrong(pdf, R):
    fig, axs = plt.subplots(1, 3, figsize=A4L, sharey=True)
    header(fig, "Where harmed prompt tracks pick up their first wrong IT hit",
           "Share of harmed prompt tracks (d0 or z0) by the surface of the first wrong accepted hit in the refit's "
           "visit order (outside-in); 'none' = harmed without any wrong hit")
    cats = list(SURF) + ["none"]
    for ax, e in zip(axs, ETA):
        scen_bars(ax, cats, lambda s, k: R["scenarios"][s]["attribution"][f"prompt|eta{e}"]["either"]
                  ["first_wrong_surface_share"][k])
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
        ax.set_xlabel("surface of the first wrong hit")
        ax.set_title(f"|eta| {e}")
        ax.text(0.98, 0.97, "harmed tracks:\n" + "\n".join(
            f"{s} {R['scenarios'][s]['attribution'][f'prompt|eta{e}']['either']['n_harmed']:,}" for s in SCEN),
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
    axs[0].set_ylabel("harmed prompt tracks [%]")
    fig.subplots_adjust(left=0.06, right=0.84, top=0.86, bottom=0.1, wspace=0.12)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_classes(pdf, R):
    fig, axs = plt.subplots(2, 2, figsize=A4L)
    header(fig, "What the wrong hits on harmed prompt tracks are",
           "Class T = cluster on another particle's true path (angle emulated from its truth); class C = no such "
           "particle (CNN angle); shared = dominant particle < 90% of the charge; noise = no particle")
    A = {s: R["scenarios"][s]["attribution"]["prompt"]["either"] for s in SCEN}
    cats = ["wrong_T", "wrong_C", "noise", "shared_other"]
    labs = ["wrong, class T", "wrong, class C", "noise (class C)", "shared, other\nparticle dominant"]
    ax = axs[0, 0]
    scen_bars(ax, cats, lambda s, k: A[s]["first_wrong_class_share"][k])
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("first wrong hits [%]"); ax.set_title("class of the first wrong hit"); ax.set_xlabel("hit class")
    ax = axs[0, 1]
    cats2 = ["other_interaction", "same_interaction", "unlinked", "shared"]
    labs2 = ["other pp\ninteraction", "same pp\ninteraction", "unlinked\n(noise)", "shared\ncluster"]
    scen_bars(ax, cats2, lambda s, k: A[s]["first_wrong_origin_share"][k])
    ax.set_xticks(range(4)); ax.set_xticklabels(labs2, fontsize=7)
    ax.set_ylabel("first wrong hits [%]"); ax.set_title("origin of the first wrong hit"); ax.set_xlabel("origin")
    for ax, lab in ((axs[1, 0], "harmed"), (axs[1, 1], "unharmed")):
        hc = ["shared_own", "shared_other", "wrong_T", "wrong_C", "noise"]
        scen_bars(ax, hc, lambda s, k: A[s][f"hit_class_share_{lab}"][k])
        ax.set_xticks(range(len(hc))); ax.set_xticklabels(["shared, own\ndominant", "shared, other\ndominant",
                                                           "wrong,\nclass T", "wrong,\nclass C", "noise"], fontsize=7)
        corr = "\n".join(f"{s} {fmt(A[s][f'hit_class_share_{lab}']['correct'], nd=2)}" for s in SCEN)
        ax.set_title(f"all accepted hits on {lab} prompt tracks")
        ax.text(0.02, 0.97, "correct hits:\n" + corr, transform=ax.transAxes, ha="left", va="top", fontsize=7,
                color=INK2)
        ax.set_ylabel("accepted hits [%]"); ax.set_xlabel("hit class")
        if lab == "unharmed":
            ax.set_yscale("log"); logfmt(ax)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.86, bottom=0.1, hspace=0.55, wspace=0.25)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_P(pdf, R):
    fig, axs = plt.subplots(1, 2, figsize=A4L, sharey=True)
    header(fig, "Wrong-hit rate P per surface (the replay reference) and correct-hit attachment",
           "P = wrong accepted hits / accepted hits (truth-matched tracks, true pT >= 2 GeV); IL4 (IL3 for AAAI) "
           "is the first surface the outside-in refit visits")
    for ax, key, title in ((axs[0], "P", "all origins"), (axs[1], "P_prompt", "prompt tracks")):
        scen_bars(ax, SURF, lambda s, k: R["scenarios"][s]["reference_E5"][key].get(k))
        ax.set_xticks(range(len(SURF))); ax.set_xticklabels(SURF)
        ax.set_xlabel("surface"); ax.set_title(title)
    axs[0].set_ylabel("wrong-hit rate P [% of accepted hits]")
    fig.subplots_adjust(left=0.06, right=0.84, top=0.86, bottom=0.1, wspace=0.08)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_benefit(pdf, R):
    fig, axs = plt.subplots(2, 3, figsize=A4L, sharex=True)
    header(fig, "Benefit baselines for the refit decision (prompt tracks)",
           "Top: correct-hit attachment at the innermost active barrel surface (IL1 for AAAA / AAAI, IL2 for IAAA), "
           "tracks crossing it. Bottom: tracks within 100 um of the true d0")
    for c, e in enumerate(ETA):
        ax = axs[0, c]
        scen_points(ax, PT, lambda s, p: R["scenarios"][s]["benefit"]["attach_innermost_active"][f"prompt|eta{e}|pt{p}"])
        ax.set_title(f"|eta| {e}"); ax.set_ylabel("correct hit attached [%]")
        ax = axs[1, c]
        scen_points(ax, PT, lambda s, p: (R["scenarios"]["AAAA"]["benefit"]["within_100um_d0"][f"prompt|eta{e}|pt{p}"]["OT"]
                                          if s == "OT" else
                                          R["scenarios"][s]["benefit"]["within_100um_d0"][f"prompt|eta{e}|pt{p}"]["SP"]),
                    series=("OT",) + SCEN, dx=0.14)
        ax.set_ylabel("|d0 - true| < 100 um [% of tracks]"); ax.set_xlabel("true pT bin [GeV]")
        ax.set_xticks(range(4)); ax.set_xticklabels(PT_LAB)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.86, bottom=0.09, hspace=0.18, wspace=0.28)
    legend_right(fig)
    pdf.savefig(fig); plt.close(fig)


def page_roc(pdf, R):
    fig, axs = plt.subplots(1, 3, figsize=A4L, sharey=True)
    header(fig, "Fallback to the OT parameters: ROC (working point set on files 3-6, quoted on files 7-10)",
           "Revert when chi2 = sum over d0, z0 of (SP - OT)^2 / (sigma_SP^2 + sigma_OT^2) exceeds a threshold; "
           "sigma = each arm's measured core per reconstructed |eta| x pT bin (files 3-6)")
    LS = {"all": "-", "prompt": "--", "beyond_il1": ":"}
    for ax, s in zip(axs, SCEN):
        F = R["scenarios"][s]["fallback"]
        for split, alpha, lw in (("fit_files_3-6", 0.45, 1.0), ("test_files_7-10", 1.0, 1.5)):
            roc = F["roc"][split]
            x = [100 * a for a in roc["unharmed_revert"]]
            for o, ls in LS.items():
                y = roc["harmed_revert"] if o == "all" else roc.get(f"harmed_revert_{o}")
                if y is None:
                    continue
                lab = {"all": "all origins", "prompt": "prompt", "beyond_il1": "born beyond IL1"}[o]
                ax.plot(x, [100 * a for a in y], ls=ls, color=COL[s] if split.startswith("test") else COL["OT"],
                        lw=lw, alpha=alpha, label=f"{lab}, {'files 7-10' if split.startswith('test') else 'files 3-6'}")
        wp = F["wp"]["all"]["test_files_7-10"]
        xx, xe = v(wp["unharmed_revert_fraction"]); yy, ye = v(wp["harmed_revert_fraction"])
        ax.errorbar([xx], [yy], xerr=[xe], yerr=[ye], fmt="o", color=INK, ms=5, label="working point (files 7-10)", zorder=5)
        ax.axvline(1.2, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlim(0.05, 100); ax.set_ylim(0, 100); logfmt(ax, "x")
        ax.set_xlabel("unharmed tracks reverted [%]")
        ax.set_title(f"{s}\nharm removed {fmt(wp['harmed_revert_fraction'])}\n"
                     f"at {fmt(wp['unharmed_revert_fraction'], nd=2)} unharmed reverted", fontsize=8)
    axs[0].set_ylabel("harmed tracks reverted [%]")
    fig.subplots_adjust(left=0.06, right=0.78, top=0.83, bottom=0.1, wspace=0.1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, ls=ls, lw=1.5) for ls in LS.values()]
    l = ["all origins", "prompt", "born beyond IL1"]
    h += [Line2D([], [], color=COL["OT"], lw=1.0, alpha=0.6), Line2D([], [], color=INK, marker="o", ls="none")]
    l += ["files 3-6 (where the\nworking point is set)", "working point,\nfiles 7-10"]
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.79, 0.5), frameon=False,
               title="line style: origin of the\nharmed tracks; colour:\nscenario, files 7-10")
    pdf.savefig(fig); plt.close(fig)


def table_page(pdf, title, sub, rows, colw=None, fs=7.0):
    fig = plt.figure(figsize=A4L)
    header(fig, title, sub)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86]); ax.axis("off")
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="center", colLoc="center",
                 colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#f0efec")
    pdf.savefig(fig); plt.close(fig)


def page_fallback_cost(pdf, R):
    thr = ", ".join(f"{s} chi2 > {R['scenarios'][s]['fallback']['threshold_chi2']['v']:.1f} ± "
                    f"{R['scenarios'][s]['fallback']['threshold_chi2']['se']:.1f}" for s in SCEN)
    rows = [["scenario", "origin", "tracks", "harmed", "harm removed", "unharmed reverted", "all reverted",
             "H before", "H after"]]
    for s in SCEN:
        F = R["scenarios"][s]["fallback"]
        for o in ("all",) + ORG:
            w = F["wp"][o]["test_files_7-10"]
            rows.append([s, ORG_LAB.get(o, o), f"{w['n']:,}", f"{w['n_harmed']:,}", fmt(w["harmed_revert_fraction"]),
                         fmt(w["unharmed_revert_fraction"], nd=2), fmt(w["revert_fraction"], nd=2),
                         fmt(w["H_before"]), fmt(w["H_after"])])
    table_page(pdf, "Fallback working point on files 7-10: what it removes",
               "Working point: 1.2% of unharmed tracks (all origins) reverted on files 3-6; threshold " + thr +
               ". H = harmed fraction (d0 or z0); reverted tracks carry the OT parameters and are not harmed by "
               "definition", rows, fs=7)
    short = lambda d: "n/a" if v(d)[0] is None else f"{v(d)[0]:.1f}±{v(d)[1]:.1f}"
    hdr = ["scenario", "origin"]
    keys = (("d0", "outside_core", "d0 out-core"), ("d0", "frac_gt_0.2mm", "d0>0.2mm"),
            ("d0", "frac_gt_0.5mm", "d0>0.5mm"), ("z0", "frac_gt_1mm", "z0>1mm"))
    for _, _, lab in keys:
        hdr += [f"{lab} SP", "FB", "OT"]
    rows = [hdr]
    for s in SCEN:
        C = R["scenarios"][s]["fallback"]["cost"]
        for o in ("all",) + ORG:
            r = [s, ORG_LAB.get(o, o)]
            for p, k, _ in keys:
                r += [short(C[o][p][a].get(k)) for a in ("SP", "FB", "OT")]
            rows.append(r)
    table_page(pdf, "Fallback cost on files 7-10 [% of tracks]: SmartPixels (SP) vs fallback (FB) vs OT",
               "Outside core: fraction beyond 3 core widths, the core refitted on each collection (B.1); tails: "
               "fractions beyond common absolute thresholds. Paired FB - SP differences with errors are in the JSON",
               rows, fs=6.4)
    rows = [["scenario"] + [k.split(" (")[0] for k in R["scenarios"]["AAAA"]["fallback"]["alternatives"]]]
    for s in SCEN:
        A = R["scenarios"][s]["fallback"]["alternatives"]
        rows.append([s] + [f"{fmt(a['test_harmed_revert'])} at {fmt(a['test_unharmed_revert'], nd=2)}" for a in A.values()])
    rows.append(["main (measured resolution, d0 + z0)"] + [""] * (len(rows[0]) - 1))
    for s in SCEN:
        w = R["scenarios"][s]["fallback"]["wp"]["all"]["test_files_7-10"]
        rows.append([s, f"{fmt(w['harmed_revert_fraction'])} at {fmt(w['unharmed_revert_fraction'], nd=2)}"] + [""] * (len(rows[0]) - 2))
    table_page(pdf, "Fallback discriminant alternatives (context): harm removed on files 7-10 at 1.2% unharmed reverted on files 3-6",
               "refit shift chi2 = the refit's own da^T (C_OT - C_refit)^-1 da; reported-covariance chi2 = (SP - OT)^2 / "
               "(cov_SP + cov_OT) in d0 and z0; the main discriminant uses measured resolutions", rows, fs=7)


def page_tables(pdf, R):
    rows = [["true |eta|", "true pT [GeV]"] + [f"H {s} (d0 or z0)" for s in SCEN] + [f"H {s} d0 / z0" for s in SCEN] + ["prompt tracks"]]
    for e in ETA:
        for p, pl in zip(PT, PT_LAB):
            k = f"eta{e}|pt{p}"
            H = {s: R["scenarios"][s]["reference_E5"]["H"][k] for s in SCEN}
            rows.append([e, pl] + [fmt(H[s]["either"], nd=2) for s in SCEN]
                        + [f"{v(H[s]['d0'])[0]:.2f} / {v(H[s]['z0'])[0]:.2f}%" for s in SCEN] + [f"{H['AAAA']['n']:,}"])
    table_page(pdf, "Replay reference: H (harmed fraction of prompt tracks) per bin",
               "Bootstrap standard errors (variances and per-event counts in the JSON / npz for paired checks)", rows, fs=6.8)
    rows = [["surface"] + [f"P {s} all" for s in SCEN] + [f"P {s} prmpt" for s in SCEN] + [f"attach {s} prompt" for s in SCEN]]
    for x in SURF:
        r = [x]
        for key in ("P", "P_prompt"):
            r += [fmt(R["scenarios"][s]["reference_E5"][key].get(x), nd=2) if R["scenarios"][s]["reference_E5"][key].get(x) else "-" for s in SCEN]
        r += [fmt(R["scenarios"][s]["benefit"]["attach_per_surface"].get(x), nd=2) if R["scenarios"][s]["benefit"]["attach_per_surface"].get(x) else "-" for s in SCEN]
        rows.append(r)
    table_page(pdf, "Replay reference: wrong-hit rate P per surface; correct-hit attachment per surface",
               "P = wrong accepted / accepted hits; attachment = tracks crossing the surface whose accepted hit is "
               "correct", rows, fs=7)
    rows = [["true |eta|", "true pT [GeV]"] + [f"attach. innermost {s}" for s in SCEN] + [f"within 100 um {s}" for s in SCEN] + ["within 100 um OT"]]
    for e in ETA:
        for p, pl in zip(PT, PT_LAB):
            k = f"prompt|eta{e}|pt{p}"
            B = {s: R["scenarios"][s]["benefit"] for s in SCEN}
            rows.append([e, pl] + [fmt(B[s]["attach_innermost_active"][k], nd=2) for s in SCEN]
                        + [fmt(B[s]["within_100um_d0"][k]["SP"], nd=2) for s in SCEN]
                        + [fmt(B["AAAA"]["within_100um_d0"][k]["OT"], nd=2)])
    table_page(pdf, "Benefit baselines per bin (prompt tracks)",
               "Innermost active barrel surface: IL1 (AAAA, AAAI), IL2 (IAAA); within 100 um = |d0 - true d0| < 100 um",
               rows, fs=6.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT_DIR / "p02_results.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "p02_summary.pdf"))
    args = ap.parse_args()
    R = json.load(open(args.json))
    here = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "--no-optional-locks", "log", "-1", "--format=%h", "--", "plots.py"], cwd=here,
                            capture_output=True, text=True).stdout.strip()
    with PdfPages(args.out, metadata={"Title": "P0.2 refit-v2 do-no-harm audit",
                                      "Subject": f"analysis {R['provenance']['code_commit'][:12]}; plots {commit or 'uncommitted'}"}) as pdf:
        page_title(pdf, R)
        page_harm_bins(pdf, R)
        page_harm_origin(pdf, R)
        page_resolution(pdf, R)
        page_tails(pdf, R)
        page_pulls(pdf, R)
        page_first_wrong(pdf, R)
        page_classes(pdf, R)
        page_P(pdf, R)
        page_benefit(pdf, R)
        page_roc(pdf, R)
        page_fallback_cost(pdf, R)
        page_tables(pdf, R)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
