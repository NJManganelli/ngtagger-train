"""PV-(x, y) d0 upper-bound study: degeneracy screening, matched-seed comparisons, PDF + summary.json.

  pixi run python eval_refitq/tagger_sp/d0_analyze.py [--boot 300]

Inputs: train.py --d0 predictions, two recipes:
  linout  (PRIMARY) OUT/d0_upper_bound/linout/preds/<mask>_<variant>_<d0>_s<seed>.npz -- the ladder's
          DeepSetHGQ2 with linear class logits, every mask
  ladder  OUT/d0_upper_bound/preds/... -- the ladder recipe as is (ReLU on the class logits), AAAA
and d0pv.py validate (validation.json / validation_hists.npz).

Metrics, working points and regions are the ladder's (analyze.cell_metrics: the matched "menu" WP =
the udsg fake rate of the stock bTagScore > 0.15 on the IIII test jets; 10% / 1% udsg mistag;
thresholds per trained tagger; same event-bootstrap seed and replicates for every cell).

DEGENERACY (per trained tagger = cell x seed): the ladder recipe's ReLU on the logits leaves some
test jets with logits exactly 0 -> uniform softmax, D_b = 1/3; matched-WP thresholds can then sit
on that tie block (the efficiency and the achieved fake rate collapse). A seed is DEGENERATE if
> 1% of its test jets share one identical output vector (uniform or otherwise), or if any of its
b-vs-udsg thresholds (menu, 10%, 1%) sits on a tie block holding >= 1% of the udsg test jets.

COMPARISONS X - R (B-A, C-A, D-A; D-C supplementary) are made on the MATCHED NON-DEGENERATE seeds
(degenerate in neither cell; the "all seeds" version is kept for reference): difference of
seed-averaged metrics, its event-bootstrap std ('ev'), the unpaired seed error of the difference
of means (ladder convention) and the PAIRED seed error (std over seeds of the per-seed difference
/ sqrt(n)). The pairing is real: train.py --d0 seeds keras before the build, the seeded runs are
bit-for-bit reproducible (checked: *_repro), so X and R share each seed's initialisation and
shuffling. Totals: ev (+) seed error; 95% upper limit = diff + 1.645 x total (one-sided).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as AZ  # noqa: E402
import common as C  # noqa: E402
import d0pv  # noqa: E402

RECIPES = {"linout": os.path.join(d0pv.OUTD, "linout", "preds"), "ladder": os.path.join(d0pv.OUTD, "preds")}
PRIMARY = "linout"
RECIPE_NAME = {"linout": "linout (linear class logits)", "ladder": "ladder recipe (ReLU on the class logits)"}
LEVELS = {"track": "L2", "trackmva": "L3"}
VARIANTS = ["A", "B", "C", "D"]
DIFFS = [("B", "B-A  d0 wrt the true PV (x, y)"), ("C", "C-A  + d0 / sigma_hat(d0)"),
         ("D", "D-A  true-PV d0 + its significance")]
COMPARISONS = [("B", "A"), ("C", "A"), ("D", "A"), ("D", "C")]
COL = {"A": "#6b6b6b", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a"}     # reference palette slots 1-3 + grey
MRK = {"A": "D", "B": "o", "C": "s", "D": "^"}
GREY = "#6b6b6b"
HEAD_METRICS = ["b_vs_udsg_eff@menu", "b_vs_udsg_eff@10%", "b_vs_udsg_eff@1%"]
OTHER_METRICS = ["b_vs_udsg_auc", "macro_auc", "c_vs_udsg_eff@menu", "c_vs_udsg_eff@10%",
                 "tau_vs_jet_eff@menu", "tau_vs_jet_eff@10%", "b_vs_light_eff@10%"]
KEEP = HEAD_METRICS + OTHER_METRICS
MASK_ORDER = ["AAAA", "AAAI", "IAAA", "IIII"]
DEGEN_FRAC = 0.01
CL95 = 1.645


# ------------------------------------------------------------------ loading / screening
def load_preds(pdir):
    P = {}
    for f in sorted(glob.glob(os.path.join(pdir, "*.npz"))):
        m = re.fullmatch(r"([AI]{4})_(track|trackmva)_([ABCD])_s(\d+)\.npz", os.path.basename(f))
        if not m:
            continue
        d = np.load(f)
        P.setdefault((m.group(2), m.group(3), m.group(1)), {})[int(m.group(4))] = {
            "prob": d["prob"].astype(np.float64), "label": d["label"], "pt": d["pt"], "eta": d["eta"],
            "sample": d["sample"], "samples": [str(x) for x in d["samples"]],
            "key": C.event_key(d["sample"], d["fileno"].astype(np.int64), d["evid"]),
            "train_counts": json.loads(str(d["train_counts"])), "n_epochs": len(d["loss"]),
            "features": [str(x) for x in d["features"]], "test_event_keys": d["test_event_keys"]}
    return P


def screen(d):
    """Degeneracy diagnostics of one trained tagger (nominal test set)."""
    p, lab = d["prob"], d["label"]
    uniform = np.all(np.abs(p - 1.0 / p.shape[1]) < 1e-6, axis=1)
    _, cnt = np.unique(np.round(p, 6), axis=0, return_counts=True)
    D = AZ.disc(p, ["b"], ["b", "light", "gluon"])
    b, u = lab == AZ.CI["b"], np.isin(lab, [AZ.CI["light"], AZ.CI["gluon"]])
    one = np.ones((1, int(u.sum())))
    out = {"uniform_frac": float(uniform.mean()), "uniform_frac_b": float(uniform[b].mean()),
           "uniform_frac_udsg": float(uniform[u].mean()), "max_identical_output_frac": float(cnt.max() / len(p)),
           "auc_b_udsg": float(AZ.wauc(D[b | u], b[b | u], np.ones((1, int((b | u).sum()))))[0]), "wp": {}}
    tied = False
    for x in AZ.MISTAGS[:3]:
        t = AZ.thresholds(D[u], one, x)[0]
        tie = float(np.mean(D[u] == t))
        out["wp"][AZ.pct(x)] = {"threshold": float(t), "target_fake": x, "achieved_fake": float(np.mean(D[u] > t)),
                                "b_eff": float(np.mean(D[b] > t)), "tie_block_udsg_frac": tie}
        tied |= tie >= DEGEN_FRAC
    out["threshold_on_tie"] = bool(tied)
    out["degenerate"] = bool(out["max_identical_output_frac"] > DEGEN_FRAC or tied)
    return out


def replicate_weights(P_all, n_boot, seed=12345):
    """Event-bootstrap counts over the test events (identical in every cell of every recipe)."""
    ev = None
    for P in P_all.values():
        for c, s in P.items():
            for d in s.values():
                if ev is None:
                    ev = d["test_event_keys"]
                elif not np.array_equal(d["test_event_keys"], ev):
                    raise RuntimeError(f"{c}: test events differ from the other cells")
                if np.setdiff1d(d["key"], ev).size:
                    raise RuntimeError(f"{c}: jets outside the test events")
    rng = np.random.default_rng(seed)
    cnt = np.vstack([np.ones(len(ev))] + [rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)))
                                          for _ in range(n_boot)]).astype(np.float64)
    print(f"  {len(ev)} test events, identical in all cells", flush=True)
    return ev, cnt


def per_seed_metrics(P, ev, cnt):
    """{cell: {seed: {region: {metric: (1 + n_boot,)}}}} for the KEEP metrics."""
    out = {}
    for cell, seeds in P.items():
        out[cell] = {}
        for s, d in seeds.items():
            W = cnt[:, np.searchsorted(ev, d["key"])]
            out[cell][s] = {r: {k: v[k] for k in KEEP} for r, v in AZ.cell_metrics(d, W).items()}
    return out


def compare(Ma, Mb, seeds, reg, k):
    A = np.stack([Ma[s][reg][k] for s in seeds])
    B = np.stack([Mb[s][reg][k] for s in seeds])
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN slices (e.g. an empty bin)
        diff = np.nanmean(A, 0) - np.nanmean(B, 0)
        a0, b0 = A[:, 0], B[:, 0]
        n = len(seeds)
        ok = np.isfinite(a0) & np.isfinite(b0)
        r = float(np.corrcoef(a0[ok], b0[ok])[0, 1]) if ok.sum() > 2 else np.nan
        return {"diff": float(diff[0]),
                "boot_err": float(np.nanstd(diff[1:], ddof=1)) if np.isfinite(diff[1:]).sum() > 2 else np.nan,
                "seed_err": float(np.sqrt(np.nanvar(a0, ddof=1) / n + np.nanvar(b0, ddof=1) / n)) if n > 1 else np.nan,
                "paired_seed_err": float(np.nanstd(a0 - b0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "seed_corr": r, "n_seeds": n, "seeds": list(seeds),
                "per_seed_x": [float(x) for x in a0], "per_seed_ref": [float(x) for x in b0]}


def steps(M, SCR, regs):
    """{mask: {level: {"X-R": {"matched": {reg: {metric: ...}}, "all": {...}}}}}"""
    out = {}
    for (var, mode, mask) in M:
        for x, ref in COMPARISONS:
            a, b = (var, x, mask), (var, ref, mask)
            if mode != ref or a not in M:
                continue
            common = sorted(set(M[a]) & set(M[b]))
            clean = [s for s in common if not SCR[a][s]["degenerate"] and not SCR[b][s]["degenerate"]]
            node = out.setdefault(mask, {}).setdefault(LEVELS[var], {}).setdefault(f"{x}-{ref}", {})
            for sel, seeds in (("matched", clean), ("all", common)):
                if len(seeds) < 2:
                    continue
                node[sel] = {reg: {k: compare(M[a], M[b], seeds, reg, k) for k in KEEP} for reg in regs}
    return out


def err(v, paired=True):
    s = v["paired_seed_err"] if paired else v["seed_err"]
    return float(np.hypot(v["boot_err"], s)) if np.isfinite(s) else v["boot_err"]


def sig(v, paired=True):
    e = err(v, paired)
    return v["diff"] / e if e and np.isfinite(e) and e > 0 else np.nan


def fmt_step(v, nd=3, n=False):
    """diff ± err (sigma) with the paired seed error | the unpaired (ladder) one."""
    if v is None or not np.isfinite(v["diff"]):
        return "-"
    out = (f"{v['diff']:+.{nd}f} ± {err(v):.{nd}f} ({sig(v):+.1f}σ) | ±{err(v, False):.{nd}f} "
           f"({sig(v, False):+.1f}σ)")
    return out + (f" [{v['n_seeds']}]" if n else "")


def get(S, mask, lv, comp, sel, reg, k):
    return S.get(mask, {}).get(lv, {}).get(comp, {}).get(sel, {}).get(reg, {}).get(k)


# ------------------------------------------------------------------ plots
def plt_setup():
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8a8a86", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e4e3df",
                         "grid.linewidth": 0.6, "lines.linewidth": 1.5})
    return plt


def errbars(ax, x, vs, color, marker, label, ms=4.5):
    """Thick bar: event bootstrap (+) paired seed error; thin whisker: (+) unpaired seed error."""
    ys = [v["diff"] if v else np.nan for v in vs]
    ax.errorbar(x, ys, yerr=[err(v, False) if v else np.nan for v in vs], fmt="none", ecolor=color,
                elinewidth=0.8, capsize=2.5)
    ax.errorbar(x, ys, yerr=[err(v) if v else np.nan for v in vs], fmt=marker, color=color, ms=ms,
                elinewidth=2.6, capsize=0, label=label)


def page_validation_hists(pdf, plt, V, H):
    """d0 of prompt hard-scatter tracks: A (origin) vs B (true PV) vs the wrong sign."""
    panels = [(m, "sp:refit:prompt_hs", "SP track (refit)") for m in ("AAAA", "AAAI", "IAAA")] + \
             [("IIII", "ot:all:prompt_hs", "OT track (every scenario)")] + \
             [(m, "sp:refit:pileup", "SP track (refit), pile-up") for m in ("AAAA",)] + \
             [("IIII", "ot:all:pileup", "OT track, pile-up")]
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 7.4))
    for ax, (m, lab, title) in zip(axes.flat, panels):
        h = H[f"{m}__{lab.replace(':', '--')}"]
        e = H["edges_ot"] if lab.startswith("ot") else H["edges_sp"]
        ctr = 0.5 * (e[1:] + e[:-1])
        for row, (name, color, ls) in enumerate((("A: d0 wrt origin", GREY, "-"),
                                                 ("B: d0 wrt true PV (x, y)", COL["B"], "-"),
                                                 ("wrong sign: d0 + correction", COL["C"], "--"))):
            ax.step(ctr, h[row], where="mid", color=color, ls=ls, lw=1.3, label=name)
        s = V["scenarios"][m][lab]
        ax.set_title(f"{m}: {title}", fontsize=9)
        ax.text(0.02, 0.97, f"σ68 A {s['sigma68_A_um']:.1f} µm\nσ68 B {s['sigma68_B_um']:.1f} µm\n"
                            f"σ68 wrong {s['sigma68_wrong_sign_um']:.1f} µm\nRMS corr {s['rms_corr_um']:.1f} µm\n"
                            f"slope {s['slope_d0_on_corr']:.2f} ± {s['slope_err']:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.5, family="monospace", color="#52514e")
        lim = 150 if not lab.startswith("ot") else 2000
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("track d0 [µm]")
        ax.set_ylabel(f"tracks / {e[1] - e[0]:.3g} µm")
        ax.grid(alpha=0.6)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Validation: d0 of prompt hard-scatter tracks (TP from the hard interaction, produced < 1 µm from "
                 "GenVtx) and of pile-up tracks,\nall study events (TTbar + H→ττ, every file); correction = "
                 "x_PV sin φ − y_PV cos φ (true PV, track φ). OT d0 binned on its 24.4 µm grid.", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def page_inclusive(pdf, plt, S, recipe, sel="matched"):
    """Δ b eff (inclusive) for B-A, C-A, D-A at the three WPs, every mask x level."""
    cells = [(m, lv) for m in MASK_ORDER for lv in ("L2", "L3") if m in S and lv in S[m]]
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.9))
    for ax, k in zip(axes, HEAD_METRICS):
        for i, (x, lab) in enumerate(DIFFS):
            vs = [get(S, m, lv, f"{x}-A", sel, "incl", k) for m, lv in cells]
            errbars(ax, np.arange(len(cells)) + (i - 1) * 0.22, vs, COL[x], MRK[x], lab)
        ax.axhline(0, color="#0b0b0b", lw=0.6)
        ax.set_xticks(np.arange(len(cells)))
        ax.set_xticklabels([f"{m}\n{lv}" + ("'" if m == C.BASELINE_MASK else "") for m, lv in cells], fontsize=7)
        ax.set_xlabel("scenario / ladder level (IIII = OT control)")
        ax.set_ylabel("Δ b-jet efficiency (absolute fraction)")
        wp = k.split("@")[1]
        ax.set_title(f"b vs udsg at {'the menu-matched' if wp == 'menu' else wp} udsg mistag", fontsize=9)
        ax.grid(axis="y", alpha=0.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{RECIPE_NAME[recipe]}: inclusive gain over A at matched fake rate, matched non-degenerate seeds.\n"
                 "Thick bar: event bootstrap ⊕ paired seed error; thin whisker: ⊕ unpaired seed error (ladder "
                 "convention)", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    pdf.savefig(fig)
    plt.close(fig)


def page_bins(pdf, plt, S, mask, regs, recipe):
    bins = [r for r in regs if r.startswith(("abs_eta:", "pt:"))]
    xl = [b.replace("abs_eta:", "|η| ").replace("pt:", "pT ").replace(">", "> ") + (" GeV" if b.startswith("pt:") else "")
          for b in bins]
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 7.6))
    for r, k in enumerate(["b_vs_udsg_eff@menu", "b_vs_udsg_eff@10%"]):
        for c, lv in enumerate(["L2", "L3"]):
            ax = axes[r, c]
            for i, (x, lab) in enumerate(DIFFS):
                vs = [get(S, mask, lv, f"{x}-A", "matched", b, k) for b in bins]
                errbars(ax, np.arange(len(bins)) + (i - 1) * 0.2, vs, COL[x], MRK[x], lab, ms=4)
            ax.axhline(0, color="#0b0b0b", lw=0.6)
            ax.axvline(2.5, color="#8a8a86", lw=0.6, ls=":")
            ax.set_xticks(np.arange(len(bins)))
            ax.set_xticklabels(xl, fontsize=7)
            ax.set_xlabel("jet |η| bin (dimensionless) | jet pT bin [GeV]")
            ax.set_ylabel("Δ b-jet efficiency (absolute fraction)")
            wp = k.split("@")[1]
            ax.set_title(f"{lv}{chr(39) if mask == C.BASELINE_MASK else ''}: b eff at the inclusive "
                         f"{'menu-matched' if wp == 'menu' else wp} udsg-mistag threshold", fontsize=9)
            ax.grid(axis="y", alpha=0.6)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{RECIPE_NAME[recipe]}, {mask}{' (OT control)' if mask == C.BASELINE_MASK else ''}: gain over A "
                 "in |η| and pT bins (thick: bootstrap ⊕ paired seed error; thin: ⊕ unpaired)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_per_seed(pdf, plt, M, SCR, recipe, k, masks):
    """Per-seed values of metric k for A/B/C/D, lines joining the same seed; degenerate seeds hollow."""
    cols = [m for m in MASK_ORDER if m in masks]
    fig, axes = plt.subplots(2, len(cols), figsize=(11.69, 7.4), squeeze=False)
    for r, var in enumerate(("track", "trackmva")):
        for c, mask in enumerate(cols):
            ax = axes[r, c]
            cells = [(var, x, mask) for x in VARIANTS]
            if not all(cl in M for cl in cells):
                ax.axis("off")
                continue
            seeds = sorted(set.intersection(*(set(M[cl]) for cl in cells)))
            off = np.linspace(-0.12, 0.12, len(seeds))       # one horizontal offset per seed
            for j, s in enumerate(seeds):
                ax.plot(np.arange(4) + off[j], [M[cl][s]["incl"][k][0] for cl in cells], color="#c3c2b7", lw=0.7,
                        zorder=1)
            for i, cl in enumerate(cells):
                y = np.array([M[cl][s]["incl"][k][0] for s in seeds])
                dg = np.array([SCR[cl][s]["degenerate"] for s in seeds])
                xx = i + off
                ax.scatter(xx[~dg], y[~dg], s=16, color=COL[cl[1]], marker=MRK[cl[1]], zorder=3,
                           label=f"variant {cl[1]}" if (r, c) == (0, 0) else None)
                ax.scatter(xx[dg], y[dg], s=22, facecolors="none", edgecolors=COL[cl[1]], marker=MRK[cl[1]],
                           zorder=3, label="degenerate seed (hollow)" if (r, c, i) == (0, 0, 0) else None)
                if (~dg).any():
                    ax.hlines(np.nanmean(y[~dg]), i - 0.3, i + 0.3, color="#0b0b0b", lw=1.5, zorder=4,
                              label="mean of non-degenerate seeds" if (r, c, i) == (0, 0, 0) else None)
            ax.set_xticks(range(4))
            ax.set_xticklabels(VARIANTS)
            ax.set_xlim(-0.5, 3.5)
            ax.set_xlabel("d0 variant (category)")
            ax.set_ylabel(f"{k.replace('_eff@', ' eff @ ')} (fraction)")
            ax.set_title(f"{mask} {LEVELS[var]}{chr(39) if mask == C.BASELINE_MASK else ''}", fontsize=9)
            ax.grid(axis="y", alpha=0.6)
    hl = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    fig.legend(list(hl.values()), list(hl), loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"{RECIPE_NAME[recipe]}: per-seed {k} on the test set (grey lines join the same seed across "
                 "variants)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def page_ladder_recipe(pdf, plt, S):
    """Ladder recipe (AAAA): every seed vs matched non-degenerate seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.9))
    cats = [(lv, sel) for lv in ("L2", "L3") for sel in ("all", "matched")]
    for ax, k in zip(axes, HEAD_METRICS):
        for i, (x, lab) in enumerate(DIFFS):
            vs = [get(S, "AAAA", lv, f"{x}-A", sel, "incl", k) for lv, sel in cats]
            errbars(ax, np.arange(len(cats)) + (i - 1) * 0.22, vs, COL[x], MRK[x], lab)
        ax.axhline(0, color="#0b0b0b", lw=0.6)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels([f"AAAA {lv}\n{'all seeds' if sel == 'all' else 'non-degenerate'}"
                            + (f"\n(n = {get(S, 'AAAA', lv, 'B-A', sel, 'incl', k)['n_seeds']} for B-A)"
                               if get(S, "AAAA", lv, "B-A", sel, "incl", k) else "") for lv, sel in cats], fontsize=7)
        ax.set_xlabel("ladder level / seed selection")
        ax.set_ylabel("Δ b-jet efficiency (absolute fraction)")
        wp = k.split("@")[1]
        ax.set_title(f"b vs udsg at {'the menu-matched' if wp == 'menu' else wp} udsg mistag", fontsize=9)
        ax.grid(axis="y", alpha=0.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{RECIPE_NAME['ladder']}, AAAA: every seed (ladder averaging) vs matched non-degenerate seeds. "
                 "Thick: ⊕ paired seed error; thin: ⊕ unpaired", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


# ------------------------------------------------------------------ main
def reproducibility():
    out = {}
    for pdir in RECIPES.values():
        for f in sorted(glob.glob(os.path.join(pdir, "*_repro.npz"))):
            ref = f.replace("_repro.npz", ".npz")
            if os.path.exists(ref):
                a, b = np.load(f)["prob"], np.load(ref)["prob"]
                out[os.path.relpath(ref, d0pv.OUTD)] = float(np.abs(a - b).max()) if a.shape == b.shape else None
    return {"max_abs_prob_diff": out,
            "reproducible": bool(out) and all(v is not None and v < 1e-5 for v in out.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    P_all = {r: P for r, d in RECIPES.items() if (P := load_preds(d))}
    if PRIMARY not in P_all:
        raise SystemExit("no linout predictions")
    fr_menu, n_udsg_ref = AZ.stock_reference()
    AZ.WP_LABEL[fr_menu] = "menu"
    AZ.WP_NAME[fr_menu] = "menu-matched"
    AZ.MISTAGS = [fr_menu] + [x for x in AZ.MISTAGS if x != fr_menu]
    print(f"menu WP: udsg fake rate {fr_menu:.4f} ({n_udsg_ref} udsg test jets)", flush=True)
    ev, cnt = replicate_weights(P_all, a.boot)
    M, SCR, S = {}, {}, {}
    for r, P in P_all.items():
        SCR[r] = {cell: {s: screen(d) for s, d in seeds.items()} for cell, seeds in P.items()}
        M[r] = per_seed_metrics(P, ev, cnt)
        regs = list(next(iter(next(iter(M[r].values())).values())))
        S[r] = steps(M[r], SCR[r], regs)
        print(f"  {r}: {len(P)} cells, {sum(len(s) for s in P.values())} trained taggers, "
              f"{sum(v['degenerate'] for c in SCR[r].values() for v in c.values())} degenerate "
              f"({time.time() - t0:.0f}s)", flush=True)
    REP = reproducibility()
    print(f"seeded-run reproducibility: {REP}", flush=True)
    one = next(iter(next(iter(P_all[PRIMARY].values())).values()))
    n_test = {c: int((one["label"] == AZ.CI[c]).sum()) for c in AZ.CLASSES}
    degen = {r: {f"{v}_{x}@{m}": {"n_seeds": len(SCR[r][(v, x, m)]),
                                  "n_degenerate": int(sum(d["degenerate"] for d in SCR[r][(v, x, m)].values())),
                                  "degenerate_seeds": sorted(s for s, d in SCR[r][(v, x, m)].items() if d["degenerate"]),
                                  "per_seed": SCR[r][(v, x, m)]}
                 for (v, x, m) in SCR[r]} for r in SCR}
    ladder = json.load(open(os.path.join(C.OUT, "summary.json")))
    cross = {}
    for r in M:
        for (v, x, m) in M[r]:
            if x != "A":
                continue
            lc = ladder["cells"].get(f"{v}@{m}")
            for k in HEAD_METRICS + ["b_vs_udsg_auc", "macro_auc"]:
                vals = np.array([M[r][(v, x, m)][s]["incl"][k][0] for s in M[r][(v, x, m)]])
                cross.setdefault(f"{v}@{m}", {}).setdefault(k, {})[r] = [float(np.nanmean(vals)),
                                                                         float(np.nanstd(vals, ddof=1))]
                cross[f"{v}@{m}"][k]["ladder_v2_run"] = lc["seed_mean_std"]["incl"].get(k) if lc else None
    V = json.load(open(os.path.join(d0pv.OUTD, "validation.json")))
    H = np.load(os.path.join(d0pv.OUTD, "validation_hists.npz"))
    summary = {
        "what": "UPPER BOUND: does track d0 relative to the TRUE PV (x, y) help L1 jet b-tagging? Variants A (ladder "
                "inputs), B (d0 wrt the true PV), C (A + d0/sigma_hat(d0)), D (B + C) at L2/L3 (SP masks) and "
                "L2'/L3' (IIII)",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": {"A": "ladder inputs (track / trackmva)",
                     "B": "trk_d0, vtxpv_dxy and the PuppiCand dxy (re-encoded) taken wrt GenVtx (x, y): "
                          "d0 - (x_PV sin phi_trk - y_PV cos phi_trk); every track of the event",
                     "C": "A + trk_d0sig = d0[um] / exp(trkmva_log_sd0) (frozen track-MVA head; 0 without a track)",
                     "D": "B + C (significance of the true-PV d0)"},
        "d0_slots": {"changed_in_B": ["trk_d0 (track)", "vtxpv_dxy (vertexdxy_pv; = trk_d0)",
                                      "dxy (baseline: PuppiCand hwDxy = F(|floor(d0*256)|)/64 cm, table "
                                      "puppi_dxy_lut.json, exact on every charged constituent)"],
                     "unchanged": ["trk_sig_d0 (KF covariance)", "trackmva heads (d0 withheld; window routing by "
                                   "|d0| from the origin kept)"]},
        "recipes": {"linout": "PRIMARY. The ladder's DeepSetHGQ2 with the ReLU on the class logits (Dense_out_jetID) "
                              "removed (tagger_sp_linout.yaml); everything else the ladder's (split, "
                              "HGQ/EBOPs, widths, hyperparameters, class weights, acceptance, matched WPs)",
                    "ladder": "the ladder v2 recipe as is (framework DeepSetHGQ2); AAAA only",
                    "why_linout": "with the ReLU on the logits, jets whose 8 pre-activations are all negative get "
                                  "exactly uniform outputs and no gradient; in the ladder recipe most seeds have such "
                                  "dead jets (up to ~50% of the b jets) and matched-WP thresholds tie at D = 1/3",
                    "seeding": "train.py --d0 seeds keras before the build: A/B/C/D share each seed's initialisation "
                               "and shuffling; seeded runs are bit-for-bit reproducible"},
        "degeneracy_definition": f"seed degenerate if > {DEGEN_FRAC:.0%} of its test jets share one identical output "
                                 f"vector, or a b-vs-udsg threshold (menu/10%/1%) sits on a tie block of >= "
                                 f"{DEGEN_FRAC:.0%} of the udsg test jets",
        "menu_wp": {"udsg_fake_rate": fr_menu, "n_udsg_test_jets_IIII": n_udsg_ref},
        "n_bootstrap": a.boot, "n_test_jets_per_class": n_test,
        "error_convention": "comparisons on matched non-degenerate seeds ('matched'; 'all' = every seed); err = "
                            "event-bootstrap std (+) paired seed error (headline) or (+) unpaired seed error "
                            "(ladder convention); upper_limit_95 = diff + 1.645 x err(paired)",
        "seeded_run_reproducibility": REP, "validation": V, "degeneracy": degen,
        "steps": S, "A_cross_check": cross,
        "per_seed_metrics": {r: {f"{v}_{x}@{m}": {s: {k: float(M[r][(v, x, m)][s]["incl"][k][0]) for k in KEEP}
                                                   for s in M[r][(v, x, m)]} for (v, x, m) in M[r]} for r in M},
    }
    for r in S:
        for m in S[r]:
            for lv in S[r][m]:
                for comp in S[r][m][lv]:
                    for sel in S[r][m][lv][comp]:
                        for v in S[r][m][lv][comp][sel]["incl"].values():
                            v["upper_limit_95"] = v["diff"] + CL95 * err(v) if np.isfinite(v["diff"]) else np.nan
    with open(os.path.join(d0pv.OUTD, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"summary.json ({time.time() - t0:.0f}s)", flush=True)

    plt = plt_setup()
    from matplotlib.backends.backend_pdf import PdfPages

    SP = S[PRIMARY]
    pdfp = os.path.join(d0pv.OUTD, "d0_upper_bound.pdf")
    with PdfPages(pdfp) as pdf:
        headline = []
        for m in MASK_ORDER:
            for lv in ("L2", "L3"):
                v = get(SP, m, lv, "B-A", "matched", "incl", "b_vs_udsg_eff@menu")
                w = get(SP, m, lv, "B-A", "matched", "incl", "b_vs_udsg_eff@10%")
                if v and w:
                    headline.append(f"  {m} {lv}{chr(39) if m == C.BASELINE_MASK else ' '}  B-A @menu {v['diff']:+.3f} ± "
                                    f"{err(v):.3f} (95% UL {v['upper_limit_95']:+.3f})   @10% {w['diff']:+.3f} ± "
                                    f"{err(w):.3f} (95% UL {w['upper_limit_95']:+.3f})   [{v['n_seeds']} seeds]")
        nd = {r: (sum(v["n_degenerate"] for v in degen[r].values()), sum(v["n_seeds"] for v in degen[r].values()))
              for r in degen}
        AZ.text_page(pdf, "Upper bound: L1 jet b-tagging with track d0 relative to the TRUE PV (x, y)", [
            "Question: would a 'PV (x, y) readout' of the E2E NNVtx help the jet tagger? Upper bound: give the tagger the TRUE",
            "PV transverse position (GenVtx x, y). The beamspot sits at the origin (GenVtx mean ~1 um, spread ~10 / 5-6 um), so",
            "today's L1 d0 is d0 wrt the beamspot; the new information is the PV offset, whose projection on a track",
            "(x sin phi - y cos phi) has RMS 8.5 um, vs refit d0 sigma68 24-55 um by |eta| (AAAA prompt HS) and OT 476 um.",
            "Variants: A ladder inputs (L2 track / L3 trackmva); B d0 wrt the true PV in trk_d0, vtxpv_dxy and the PF/PUPPI",
            "  candidate dxy (re-encoded), every track of the event; C A + d0/sigma_hat(d0) (frozen track-MVA head); D B + C.",
            "Protocol: ladder v2 split, inputs, hyperparameters, class weights, matched WPs (menu = udsg fake rate "
            f"{fr_menu:.3f}), 10 seeds.",
            "RECIPE: the ladder's DeepSetHGQ2 puts a ReLU on the class logits; jets with all logits <= 0 get an exactly uniform",
            f"output and no gradient. Ladder recipe (AAAA): {nd.get('ladder', (0, 0))[0]} of {nd.get('ladder', (0, 0))[1]} "
            "trained taggers degenerate. PRIMARY results use 'linout' = the same",
            f"network with linear logits: {nd[PRIMARY][0]} of {nd[PRIMARY][1]} degenerate. Comparisons on matched "
            "non-degenerate seeds; A/B/C/D share",
            f"each seed's init and shuffling (keras seeded before the build; reruns bit-identical: {REP['reproducible']}).",
            "Errors: event bootstrap (+) paired seed error (headline) | (+) unpaired seed error (ladder convention).",
            "",
            "HEADLINE (linout, b vs udsg efficiency, B - A = true-PV d0 minus origin d0; 95% UL = one-sided upper limit):",
        ] + headline + ["", f"Test jets per class: {n_test}"])
        page_validation_hists(pdf, plt, V, H)
        rows = []
        for m in MASK_ORDER:
            R = V["scenarios"][m]
            for lab, v in R.items():
                if not (isinstance(v, dict) and "sigma68_A_um" in v) or "not_refit" in lab:
                    continue
                rows.append([m, lab.replace(":", " / "), v["n"], f"{v['mean_corr_um']:+.2f}", f"{v['rms_corr_um']:.2f}",
                             f"{v['sigma68_A_um']:.1f}", f"{v['sigma68_B_um']:.1f}", f"{v['sigma68_wrong_sign_um']:.1f}",
                             f"{v['quadrature_shrink_sigma68_um']:+.1f}",
                             f"{v['slope_d0_on_corr']:.3f} ± {v['slope_err']:.3f}",
                             f"{v['median_A_um']:+.2f} / {v['median_B_um']:+.2f}"])
            v = R.get("sp:refit:prompt_hs")
            if v:
                for e, b in v["by_abs_eta"].items():
                    rows.append([m, f"sp / refit / prompt_hs, |η| {e}", b["n"], "", f"{b['rms_corr_um']:.2f}",
                                 f"{b['sigma68_A_um']:.1f}", f"{b['sigma68_B_um']:.1f}", "",
                                 f"{b['quadrature_shrink_um']:+.1f}", "", ""])
        AZ.table_page(pdf, "Validation table: d0 of prompt hard-scatter and pile-up tracks, A vs B (true-PV correction)",
                      ["mask", "tracks", "n", "mean corr [µm]", "RMS corr [µm]", "σ68 A [µm]", "σ68 B [µm]",
                       "σ68 wrong sign [µm]", "√(A²−B²) [µm]", "slope d0 vs corr", "median A / B [µm]"], rows,
                      fontsize=6.2, max_rows=34,
                      note="Expectation for prompt hard-scatter tracks: mean correction ~0, B narrower than A by the RMS "
                           "correction in quadrature (~8.5 µm), the wrong sign broader, slope of d0 on the correction = +1. "
                           "Pile-up: slope 0 and B broader (the hard-scatter PV is the wrong vertex). OT tracks: d0 sits on "
                           "a 24.4 µm grid and the correction is ~2% of the resolution (σ68 cannot resolve it; the slope "
                           "can). Window routing of the refit MVA heads flips for "
                           + ", ".join(f"{m} {V['scenarios'][m].get('sp:window_routing_flip_frac', 0):.2%}"
                                       for m in MASK_ORDER if m != C.BASELINE_MASK) + " of the SP tracks (kept as A).")
        # degeneracy table
        rows = []
        for r in ("ladder", "linout"):
            if r not in SCR:
                continue
            for m in MASK_ORDER:
                for var in ("track", "trackmva"):
                    for x in VARIANTS:
                        sc = SCR[r].get((var, x, m))
                        if not sc:
                            continue
                        u = np.array([d["uniform_frac"] for d in sc.values()])
                        ub = np.array([d["uniform_frac_b"] for d in sc.values()])
                        auc = np.array([d["auc_b_udsg"] for d in sc.values()])
                        ties = sum(sum(w["tie_block_udsg_frac"] >= DEGEN_FRAC for w in d["wp"].values())
                                   for d in sc.values())
                        rows.append([r, f"{m} {LEVELS[var]} {x}", len(sc), sum(d["degenerate"] for d in sc.values()),
                                     ",".join(str(s) for s, d in sorted(sc.items()) if d["degenerate"]) or "-",
                                     f"{u.mean():.3f} / {u.max():.3f}", f"{ub.max():.3f}", ties,
                                     f"{auc.mean():.3f} ± {auc.std(ddof=1):.3f} ({auc.min():.2f}-{auc.max():.2f})"])
        AZ.table_page(pdf, "Degeneracy screen per trained tagger (cell x seed)",
                      ["recipe", "cell", "seeds", "degen.", "degenerate seeds", "uniform frac\nmean / max",
                       "uniform frac\nb jets, max", "WP ties", "b/udsg AUC per seed\nmean ± std (range)"],
                      rows, fontsize=5.6, max_rows=40,
                      col_widths=[0.06, 0.09, 0.05, 0.05, 0.16, 0.12, 0.12, 0.08, 0.21],
                      note=summary["degeneracy_definition"] + ". Uniform output = all 8 probabilities 1/8 (logits all 0 "
                           "after the ReLU of the ladder recipe). 'WP thresholds on a tie' counts (seed, WP) pairs among "
                           "menu/10%/1%.")
        page_inclusive(pdf, plt, SP, PRIMARY)
        masks_p = sorted({m for (_, _, m) in M[PRIMARY]})
        for k in HEAD_METRICS:
            page_per_seed(pdf, plt, M[PRIMARY], SCR[PRIMARY], PRIMARY, k, masks_p)
        rows = []
        for m in MASK_ORDER:
            for lv, var in (("L2", "track"), ("L3", "trackmva")):
                if (var, "A", m) not in M[PRIMARY]:
                    continue
                for k in HEAD_METRICS + OTHER_METRICS:
                    vals = np.array([M[PRIMARY][(var, "A", m)][s]["incl"][k][0] for s in M[PRIMARY][(var, "A", m)]
                                     if not SCR[PRIMARY][(var, "A", m)][s]["degenerate"]])
                    rows.append([f"{m} {lv}{chr(39) if m == C.BASELINE_MASK else ''}", k,
                                 AZ.fmt(np.nanmean(vals), np.nanstd(vals, ddof=1)) if len(vals) > 1 else "-"]
                                + [fmt_step(get(SP, m, lv, f"{x}-{r}", "matched", "incl", k), n=True)
                                   for x, r in COMPARISONS])
        AZ.table_page(pdf, f"{RECIPE_NAME[PRIMARY]}: inclusive results on matched non-degenerate seeds",
                      ["cell", "metric", "A"] + [f"{x} − {r}" for x, r in COMPARISONS], rows,
                      fontsize=4.9, max_rows=40, col_widths=[0.055, 0.105, 0.08, 0.19, 0.19, 0.19, 0.19],
                      note="Cell: difference of seed-averaged metrics ± (event bootstrap ⊕ paired seed error) (σ) | "
                           "± (event bootstrap ⊕ unpaired seed error, ladder convention) (σ) [matched seeds]. A: mean ± "
                           "seed std. D−C: the true-PV effect with the significance present. eff@X = efficiency at the "
                           f"tagger's own threshold giving X background fake rate (menu = {fr_menu:.3f} udsg); c and tau "
                           f"use the same fake rates (tau: light+gluon background). Test jets: {n_test}.")
        for m in MASK_ORDER:
            if m in SP:
                page_bins(pdf, plt, SP, m, regs, PRIMARY)
        rows = []
        bins = [r for r in regs if r.startswith(("abs_eta:", "pt:", "sample:"))]
        for m in MASK_ORDER:
            for lv in ("L2", "L3"):
                if m not in SP or lv not in SP[m]:
                    continue
                for reg in bins:
                    for k in HEAD_METRICS[:2]:
                        rows.append([f"{m} {lv}", reg, k]
                                    + [fmt_step(get(SP, m, lv, f"{x}-A", "matched", reg, k)) for x, _ in DIFFS])
        AZ.table_page(pdf, f"{RECIPE_NAME[PRIMARY]}: results by region (b eff at the INCLUSIVE threshold of each WP)",
                      ["cell", "region", "metric", "B − A", "C − A", "D − A"], rows, fontsize=5.2, max_rows=42,
                      col_widths=[0.06, 0.09, 0.11, 0.2466, 0.2466, 0.2466],
                      note="Efficiency in a bin at the tagger's inclusive threshold. Errors as on the inclusive table "
                           "(paired | unpaired seed error), matched non-degenerate seeds.")
        if "ladder" in S:
            page_ladder_recipe(pdf, plt, S["ladder"])
            page_per_seed(pdf, plt, M["ladder"], SCR["ladder"], "ladder", "b_vs_udsg_eff@10%",
                          sorted({m for (_, _, m) in M["ladder"]}))
            rows = []
            for lv in ("L2", "L3"):
                for k in HEAD_METRICS + ["b_vs_udsg_auc"]:
                    for sel in ("all", "matched"):
                        rows.append([f"AAAA {lv}", k, sel] + [fmt_step(get(S["ladder"], "AAAA", lv, f"{x}-{r}", sel,
                                                                           "incl", k), n=True)
                                                              for x, r in COMPARISONS])
            AZ.table_page(pdf, f"{RECIPE_NAME['ladder']}, AAAA: every seed vs matched non-degenerate seeds",
                          ["cell", "metric", "seeds"] + [f"{x} − {r}" for x, r in COMPARISONS], rows, fontsize=5.2,
                          col_widths=[0.06, 0.11, 0.05, 0.195, 0.195, 0.195, 0.195],
                          note="Errors as on the linout tables (paired | unpaired seed error) [number of seeds].")
        rows = []
        for c, d in cross.items():
            for k, v in d.items():
                rows.append([c, k] + [AZ.fmt(*v[r]) if r in v else "-" for r in ("ladder", "linout")]
                            + [AZ.fmt(*v["ladder_v2_run"]) if v.get("ladder_v2_run") else "-"])
        AZ.table_page(pdf, "Cross-check: variant A per recipe (all seeds, mean ± seed std) vs the ladder v2 run",
                      ["cell", "metric", "A, ladder recipe (this study)", "A, linout", "ladder v2 run"], rows,
                      fontsize=6.5, max_rows=40,
                      note="The ladder-recipe A differs from the ladder v2 run only by seeding keras before the build "
                           "(same inputs and network). linout removes the ReLU on the class logits.")
    print(f"wrote {pdfp} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
