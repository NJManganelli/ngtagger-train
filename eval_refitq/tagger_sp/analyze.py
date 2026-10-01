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
  b_vs_udsg   D = p_b/(p_b+p_light+p_g)      signal b,     background udsg (light + gluon)
  c_vs_udsg   D = p_c/(p_c+p_light+p_g)      signal charm, background udsg
  tau_vs_jet  D = p_tau/(p_tau+p_light+p_g)  signal tau+-, background udsg
  b_vs_light, c_vs_light: the same with light (uds) only
  WORKING POINTS (per trained tagger = per cell and seed): loose / medium / tight = the threshold
  on D giving 10% / 1% / 0.1% background efficiency on the cell's own test jets (thresholds
  thr@X% in summary.json); <pair>_eff@X% = signal efficiency there. In a bin, the INCLUSIVE
  threshold (so <pair>_mistag@X% in a bin is that WP's mistag in the bin).
  Reachability: n_bkg x mistag < 1 -> the WP is not resolvable with our test jets (flagged);
  < 10 -> poorly determined.
  Stock NG tagger (emulated scores in the nano, per scenario's jets; not retrained): b eff and
  udsg mistag of bTagScore > 0.15 (menu cut), test jets in acceptance.
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
    "b_vs_udsg": (["b"], ["light", "gluon"], ["b"], ["b", "light", "gluon"]),
    "c_vs_udsg": (["charm"], ["light", "gluon"], ["charm"], ["charm", "light", "gluon"]),
    "tau_vs_jet": (["taup", "taum"], ["light", "gluon"], ["taup", "taum"], ["taup", "taum", "light", "gluon"]),
    "b_vs_light": (["b"], ["light"], ["b"], ["b", "light"]),
    "c_vs_light": (["charm"], ["light"], ["charm"], ["charm", "light"]),
}
WP_PAIRS = ["b_vs_udsg", "c_vs_udsg", "tau_vs_jet"]
MISTAGS = (0.10, 0.01, 0.001)      # loose / medium / tight (background = udsg; tau: light+gluon jets)
WP_NAME = {0.10: "loose", 0.01: "medium", 0.001: "tight"}
STOCK_B_CUT = 0.15                 # the stock NG menu cut on bTagScore


def pct(m):
    return f"{m * 100:g}%"
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
KEY_METRICS = ["macro_auc", "b_vs_udsg_eff@10%", "b_vs_udsg_eff@1%", "b_vs_udsg_eff@0.1%",
               "c_vs_udsg_eff@10%", "c_vs_udsg_eff@1%", "tau_vs_jet_eff@10%", "tau_vs_jet_eff@1%",
               "b_vs_light_eff@10%", "c_vs_light_eff@10%"]
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
    thr = {f"{n}_thr@{pct(m)}": T[(n, m)] for n in PAIRS for m in MISTAGS}
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
                o[f"{n}_eff@{pct(m)}"] = frac_above(D[n][r & S[n]], W[:, r & S[n]], T[(n, m)])
                o[f"{n}_mistag@{pct(m)}"] = frac_above(D[n][r & B[n]], W[:, r & B[n]], T[(n, m)])
        if reg == "incl":
            o.update(thr)
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
            "key": C.event_key(d["sample"], d["fileno"].astype(np.int64), d["evid"]),
            "train_counts": json.loads(str(d["train_counts"])), "n_epochs": len(d["loss"]),
            "features": [str(x) for x in d["features"]], "test_event_keys": d["test_event_keys"]}
    return P


def compute(P, n_boot, seed=12345):
    """{cell: {"seeds": {s: {reg: {metric: nominal}}}, "avg": {reg: {metric: (1+n_boot,)}}}}."""
    # the test EVENTS (from the split, incl. events without an accepted labelled jet) must be
    # identical in every cell: the pairing of the bootstrap relies on it
    ev = next(iter(next(iter(P.values())).values()))["test_event_keys"]
    for c, s in P.items():
        for d in s.values():
            if not np.array_equal(d["test_event_keys"], ev):
                raise RuntimeError(f"{c}: test events differ from the other cells")
            if np.setdiff1d(d["key"], ev).size:
                raise RuntimeError(f"{c}: jets outside the test events")
    print(f"  {len(ev)} test events, identical in all cells", flush=True)
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
                for k in [k for k in M[next(iter(M))]["avg"][reg] if "_thr@" not in k]:
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


def reachability(P):
    """{cell: {pair: {wp: n_bkg_test_jets * mistag}}} (seed independent)."""
    out = {}
    for (v, m), seeds in P.items():
        d = next(iter(seeds.values()))
        out[f"{v}@{m}"] = {n: {pct(x): float(np.isin(d["label"], [CI[c] for c in PAIRS[n][1]]).sum() * x)
                               for x in MISTAGS} for n in WP_PAIRS}
    return out


def stock_wp(n_boot=300, seed=7):
    """The stock NG tagger (scores emulated in the nano, per scenario's jets) on each mask's
    test jets in acceptance: b eff / udsg mistag of bTagScore > STOCK_B_CUT, and its b eff at
    the 10 / 1 / 0.1% udsg-mistag thresholds of bTagScore; event-bootstrap errors."""
    out = {}
    for mask in C.MASKS:
        mp = C.manifest_path(mask)
        if not os.path.exists(mp):
            continue
        man = json.load(open(mp))
        b, lab, key = [], [], []
        for s, n, _ in man["files"]:
            d = np.load(os.path.join(C.cache_dir("jets"), C.tag(s, mask, n) + ".npz"))
            if "stock_bTagScore" not in d.files:
                return {}
            sel = d["is_test"] & (d["pt"] > C.PT_MIN) & (np.abs(d["eta"]) < C.ETA_MAX)
            b.append(d["stock_bTagScore"][sel])
            lab.append(d["label"][sel])
            key.append(C.event_key(man["samples"].index(s), n, d["evid"][d["entry"][sel]]))
        b, lab, key = np.concatenate(b).astype(float), np.concatenate(lab), np.concatenate(key)
        ev, inv = np.unique(key, return_inverse=True)
        rng = np.random.default_rng(seed)
        W = np.vstack([np.ones(len(ev))] + [rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)))
                                            for _ in range(n_boot)]).astype(float)[:, inv]
        isb, isu = lab == CI["b"], np.isin(lab, [CI["light"], CI["gluon"]])
        cut = np.full(len(W), STOCK_B_CUT)
        r = {"b_eff@cut": frac_above(b[isb], W[:, isb], cut), "udsg_mistag@cut": frac_above(b[isu], W[:, isu], cut),
             "light_mistag@cut": frac_above(b[lab == CI["light"]], W[:, lab == CI["light"]], cut)}
        for x in MISTAGS:
            t = thresholds(b[isu], W[:, isu], x)
            r[f"b_eff@{pct(x)}"] = frac_above(b[isb], W[:, isb], t)
            r[f"thr@{pct(x)}"] = t
        out[mask] = {k: [float(v[0]), float(np.nanstd(v[1:], ddof=1))] for k, v in r.items()}
        out[mask].update({"n_b": int(isb.sum()), "n_udsg": int(isu.sum()),
                          "frac_udsg_score0": float(np.mean(b[isu] == 0))})
    return out


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


def table_page(pdf, title, header, rows, note=None, fontsize=7.0, col_widths=None, max_rows=30):
    """One or more table pages (max_rows body rows each)."""
    if len(rows) > max_rows:
        for i in range(0, len(rows), max_rows):
            table_page(pdf, f"{title} ({i // max_rows + 1}/{-(-len(rows) // max_rows)})", header,
                       rows[i:i + max_rows], note, fontsize, col_widths, max_rows)
        return
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
        for j, name in enumerate(WP_PAIRS):
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
    L = {m: v for m, v in ladder(M, regs).items() if ("baseline", m) in M}
    N = counts(P)
    RCH = reachability(P)
    STK = stock_wp(a.boot)
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
        "working_points": {
            "definition": "per trained tagger (cell, seed): threshold on D giving 10/1/0.1% background "
                          "efficiency (loose/medium/tight) on that cell's test jets; D per pair in the "
                          "module docstring (b/c: background udsg; tau: light+gluon jets)",
            "thresholds_per_seed": {f"{v}@{m}": {s: {k: x for k, x in M[(v, m)]["seeds"][s]["incl"].items()
                                                      if "_thr@" in k} for s in M[(v, m)]["seeds"]}
                                    for (v, m) in M},
            "expected_bkg_jets_above_threshold": RCH,
            "stock_ng_tagger": {"cut": f"bTagScore > {STOCK_B_CUT}", "per_mask": STK}},
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
            for k in KEY_METRICS + ["b_vs_udsg_auc", "c_vs_udsg_auc", "tau_vs_jet_auc", "b_vs_light_auc", "c_vs_light_auc"] + ALL_AUCS:
                m0 = seed_stats(M, L0, "incl", k) if L0 in M else (np.nan, np.nan, 0)
                rows.append([k, fmt(m0[0], m0[1])] + [fmt_step(L[mask].get(lab, {}).get("incl", {}).get(k))
                                                      for lab, _, _ in STEPS])
            table_page(pdf, f"LADDER ({mask} vs IIII): gain of each step, inclusive test set", hdr, rows,
                       fontsize=6.3, note=STEP_NOTE + " L0 column: absolute value ± seed std.")
        for mask in [m for m in SP_MASKS if m in L]:
            plot_steps(pdf, L, mask, "region:incl", ALL_AUCS + ["b_vs_udsg_auc", "c_vs_udsg_auc", "tau_vs_jet_auc", "b_vs_light_auc", "c_vs_light_auc"],
                       [k.replace("auc_", "").replace("_auc", "") for k in
                        ALL_AUCS + ["b_vs_udsg_auc", "c_vs_udsg_auc", "tau_vs_jet_auc", "b_vs_light_auc", "c_vs_light_auc"]],
                       "step gains per class (one-vs-rest AUC) and pairwise AUCs", "Δ AUC")
            bins = [r for r in regs if r.startswith(("abs_eta:", "pt:"))]
            for k, lab in (("b_vs_udsg_eff@10%", "Δ b eff (loose: 10% udsg mistag)"),
                           ("b_vs_udsg_eff@1%", "Δ b eff (medium: 1% udsg mistag)"),
                           ("c_vs_udsg_eff@10%", "Δ c eff (10% udsg mistag)"),
                           ("tau_vs_jet_eff@10%", "Δ tau eff (10% light+gluon mistag)"),
                           ("macro_auc", "Δ macro AUC")):
                plot_steps(pdf, L, mask, f"metric:{k}", bins, [b.replace("abs_eta:", "|η| ").replace("pt:", "pT ")
                                                               for b in bins],
                           f"step gains in |η| and pT bins: {k}", lab)
            rows = []
            for reg in bins + [r for r in regs if r.startswith("sample:")]:
                for k in ("b_vs_udsg_eff@10%", "b_vs_udsg_eff@1%", "c_vs_udsg_eff@10%", "tau_vs_jet_eff@10%",
                          "macro_auc"):
                    m0 = seed_stats(M, L0, reg, k) if L0 in M else (np.nan, np.nan, 0)
                    rows.append([reg, k, fmt(m0[0], m0[1])] + [fmt_step(L[mask].get(lab, {}).get(reg, {}).get(k))
                                                             for lab, _, _ in STEPS])
            table_page(pdf, f"LADDER ({mask}): step gains by region", ["region", "metric", "L0 (IIII) abs."]
                       + [short(lab) for lab, _, _ in STEPS], rows, fontsize=4.8,
                       note=STEP_NOTE + " Efficiencies at the INCLUSIVE 10%-mistag working point of each cell.")
        # working points per cell
        for n in WP_PAIRS:
            rows = []
            for c in [c for c in [(v, m) for v in C.VARIANTS for m in C.MASKS] if c in M]:
                row = [f"{c[0]}@{c[1]}"]
                for x in MISTAGS:
                    e = seed_stats(M, c, "incl", f"{n}_eff@{pct(x)}")
                    t = seed_stats(M, c, "incl", f"{n}_thr@{pct(x)}")
                    nb = RCH[f"{c[0]}@{c[1]}"][n][pct(x)]
                    flag = " [unreachable]" if nb < 1 else (" [<10 bkg jets]" if nb < 10 else "")
                    row += [fmt(e[0], e[1]) + flag, fmt(t[0], t[1], 4)]
                rows.append(row)
            hdr = ["cell"] + [f"{WP_NAME[x]} ({pct(x)}): {k}" for x in MISTAGS for k in ("eff", "threshold on D")]
            table_page(pdf, f"Working points per cell: {n} (mean ± seed std over the trained taggers)", hdr,
                       rows, fontsize=6,
                       note="Threshold = D value giving the background efficiency on the cell's own test jets "
                            "(per seed; per-seed values in summary.json working_points). [unreachable]: "
                            "n_bkg x mistag < 1 test jet; [<10 bkg jets]: the WP rests on < 10 background jets.")
        if STK:
            rows = [[m, f"{v['n_b']} / {v['n_udsg']}", fmt(*v["b_eff@cut"]), fmt(*v["udsg_mistag@cut"]),
                     fmt(*v["light_mistag@cut"]), fmt(*v["b_eff@10%"]), fmt(*v["b_eff@1%"]),
                     fmt(v["frac_udsg_score0"])] for m, v in STK.items()]
            table_page(pdf, f"Stock NG tagger (nano scores, not retrained): bTagScore > {STOCK_B_CUT} on each "
                            "scenario's test jets",
                       ["mask", "test b / udsg jets", "b eff @cut", "udsg mistag @cut", "light mistag @cut",
                        "b eff @10% udsg", "b eff @1% udsg", "frac udsg with score 0"], rows,
                       note="Errors: event bootstrap. The stock model was trained elsewhere on other inputs; it is "
                            "evaluated as emulated in each scenario's nano (jets re-emulated from that scenario's tracks).")
        # absolute values per cell
        cl = [c for c in [(v, m) for v in C.VARIANTS for m in C.MASKS] if c in M]
        for chunk in (KEY_METRICS + ["b_vs_udsg_auc", "c_vs_udsg_auc", "tau_vs_jet_auc", "b_vs_light_auc", "c_vs_light_auc"], ALL_AUCS):
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
