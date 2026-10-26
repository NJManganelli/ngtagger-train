"""SmartPixels jet tagging: the deployed tagger of every cell vs the OT baseline; paired event-bootstrap
errors and the trial spread; PDF + summary.json.

  python eval_refitq/tagger_sp/analyze.py [--boot 300]      (after trial_select.py)

Cells = (variant, mask). Each cell has several trials (train.py -> OUT/preds/<mask>_<variant>_s<seed>.npz);
trial_select.py screens and chooses them on VALIDATION jets (OUT/selection.json). A cell's DEPLOYED tagger
is the mean softmax output of its deployed trials (best 3 by validation classification loss); the single
best trial is reported beside it. The test set is read here only.
HEADLINE: with vs without SmartPixels per SP mask at equal inputs: <variant>@<SP mask> - <variant>@IIII, for
the three input sets (baseline = L1-L0, track = L2-L2', trackmva = L3-L3'). The other ladder steps
(common.VARIANTS): L2-L1 (+ refit track features), L3-L2 (+ track-MVA scores), L2'-L0, L3'-L2' (the same
additions on OT tracks), L3-L0 (total).

Test sets hold the same EVENTS in every mask (the jets differ: each scenario re-emulates PF/PUPPI/jets), so
every difference is paired at the event level: each bootstrap replicate draws multinomial counts over the
union of test-event keys, every jet carries its event's count as a weight, and every cell's metric is
recomputed on that replicate. Index 0 along axis 0 of each metric array is the nominal (unit-weight) value.
Errors of a difference: 'boot' = the std of the replicates; 'seed' = the trial spread: the SD of the kept
trials' single-model values over sqrt(number deployed) (the ensemble) or the SD itself (the single best),
in quadrature for the two cells. Significance = diff / (boot (+) seed).

Metrics (softmax p_k), per region (inclusive; per sample; |eta| bins; pT bins):
  auc_<class> one-vs-rest (+ auc_tau for p_tau+ + p_tau-), macro_auc = mean of the 8 class AUCs
  b_vs_udsg   D = p_b/(p_b+p_light+p_g)      signal b,     background udsg (light + gluon)
  c_vs_udsg   D = p_c/(p_c+p_light+p_g)      signal charm, background udsg
  tau_vs_jet  D = p_tau/(p_tau+p_light+p_g)  signal tau+-, background udsg
  b_vs_light, c_vs_light: the same with light (uds) only
  MATCHED WORKING POINTS (per tagger and replicate): the threshold on D at which the background fraction
  equals the target EXACTLY, ties split (ngtagger.definitions.matched_wp: score == threshold accepted with
  the fraction that makes it exact; `tie_thresholds` is its vectorised form, checked against it on the
  nominal replicate). Targets: "menu" = the udsg fake rate of the stock menu cut bTagScore > 0.15 on the
  IIII (OT baseline) test jets; 10%; 1% (statistically weak). <pair>_eff@X = signal efficiency there; in a
  bin, the INCLUSIVE threshold (so <pair>_mistag@X in a bin is that working point's mistag in the bin).
  Reachability: n_bkg x mistag < 1 -> not resolvable with our test jets (flagged); < 10 -> poorly determined.
  Stock NG tagger (emulated scores in the nano, per scenario's jets; not retrained): b eff and udsg mistag of
  bTagScore > 0.15 (menu cut), and its b eff at the matched points, ties split, test jets in acceptance.
`thresholds` / `frac_above` (ties dropped) are kept for the finished d0 study (d0_analyze.py) and as the fast
gate's named known-bad; nothing here uses them.
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
MISTAGS = [0.10, 0.01, 0.001]      # loose / medium / tight (background = udsg; tau: light+gluon jets)
WP_NAME = {0.10: "loose", 0.01: "medium", 0.001: "tight"}
WP_LABEL = {}                      # value -> key label override ("menu", set in main)
STOCK_B_CUT = 0.15                 # the stock NG menu cut on bTagScore
REPORT_MISTAGS = [0.10, 0.01]      # with the menu-matched point (main); 0.1% is not resolvable here


def pct(m):
    return WP_LABEL.get(m, f"{m * 100:g}%")


def wp_title(x):
    return f"{WP_NAME.get(x, pct(x))} ({x * 100:.3g}% fake rate)"


def stock_reference():
    """The udsg fake rate of the stock menu cut bTagScore > STOCK_B_CUT on the OT baseline
    (IIII) test jets in acceptance -- the fake-rate target of the "menu" matched WP."""
    man = json.load(open(C.manifest_path(C.BASELINE_MASK)))
    sc, lab = [], []
    for s, n, _ in man["files"]:
        d = np.load(os.path.join(C.cache_dir("jets"), C.tag(s, C.BASELINE_MASK, n) + ".npz"))
        sel = d["is_test"] & (d["pt"] > C.PT_MIN) & (np.abs(d["eta"]) < C.ETA_MAX) & C.event_filter(d["evid"])[d["entry"]]
        sc.append(d["stock_bTagScore"][sel])
        lab.append(d["label"][sel])
    sc, lab = np.concatenate(sc), np.concatenate(lab)
    u = np.isin(lab, [CI["light"], CI["gluon"]])
    return float(np.mean(sc[u] > STOCK_B_CUT)), int(u.sum())
SP_MASKS = [m for m in C.MASKS if m != C.BASELINE_MASK]
L0 = ("baseline", C.BASELINE_MASK)
STEPS = [  # (label, a, b): a - b ; mask "M" = the SP mask of the column. The first three are the HEADLINE
    ("L1-L0  baseline inputs: SmartPixels - OT", ("baseline", "M"), L0),
    ("L2-L2'  + track inputs: SmartPixels - OT", ("track", "M"), ("track", "IIII")),
    ("L3-L3'  + track-MVA inputs: SmartPixels - OT", ("trackmva", "M"), ("trackmva", "IIII")),
    ("L2-L1  + refit track features", ("track", "M"), ("baseline", "M")),
    ("L3-L2  + track-MVA scores", ("trackmva", "M"), ("track", "M")),
    ("L2'-L0  + track features, OT tracks", ("track", "IIII"), L0),
    ("L3'-L2'  + MVA scores, OT tracks", ("trackmva", "IIII"), ("track", "IIII")),
    ("L3-L0  total", ("trackmva", "M"), L0),
]
MAIN_STEPS = STEPS[:3]
# jointly trained track-quality heads (joint_heads.py): reference (uncoupled), A (outputs apart), B (errors apart)
JOINT_STEPS = [
    ("Jref  joint heads, reference: SmartPixels - OT", ("joint_ref", "M"), ("joint_ref", "IIII")),
    ("JA  joint heads, outputs apart: SmartPixels - OT", ("joint_A", "M"), ("joint_A", "IIII")),
    ("JB  joint heads, errors apart: SmartPixels - OT", ("joint_B", "M"), ("joint_B", "IIII")),
    ("JA-Jref  outputs apart - reference, SmartPixels", ("joint_A", "M"), ("joint_ref", "M")),
    ("JB-Jref  errors apart - reference, SmartPixels", ("joint_B", "M"), ("joint_ref", "M")),
    ("JA-Jref(OT)  outputs apart - reference, OT", ("joint_A", "IIII"), ("joint_ref", "IIII")),
    ("JB-Jref(OT)  errors apart - reference, OT", ("joint_B", "IIII"), ("joint_ref", "IIII")),
    ("Jref-L3  joint reference - today's heads, SmartPixels", ("joint_ref", "M"), ("trackmva", "M")),
]
# the with-vs-without-SmartPixels step of each input set (the headline pages)
HEAD_LABEL = {a[0]: lab for lab, a, b in STEPS + JOINT_STEPS if a[0] == b[0] and a[1] == "M" and b[1] == "IIII"}
KEY_METRICS = ["macro_auc", "b_vs_udsg_auc", "b_vs_udsg_eff@menu", "b_vs_udsg_eff@10%", "b_vs_udsg_eff@1%",
               "c_vs_udsg_auc", "c_vs_udsg_eff@menu", "c_vs_udsg_eff@10%", "c_vs_udsg_eff@1%",
               "tau_vs_jet_auc", "tau_vs_jet_eff@menu", "tau_vs_jet_eff@10%", "tau_vs_jet_eff@1%",
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
    """Per replicate: the threshold t with weighted background fraction above t <= mistag (ties DROPPED:
    the finished d0 study's estimator and the fast gate's known-bad; the study uses tie_thresholds)."""
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


def tie_thresholds(score_b, W_b, target):
    """Per replicate (index along axis 0 of W_b): (t, p) of the matched working point with ties split, as
    ngtagger.definitions.matched_wp.tie_split_threshold: accept score > t fully and score == t with the
    fraction p, so that the weighted background fraction accepted equals `target` exactly."""
    n_rep = len(W_b)
    if score_b.size == 0:
        return np.full(n_rep, np.nan), np.full(n_rep, np.nan)
    o = np.argsort(score_b, kind="stable")
    sb = score_b[o]
    starts = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1]])
    u = sb[starts]                                             # distinct values, ascending
    Wu = np.add.reduceat(W_b[:, o], starts, axis=1)            # (n_rep, n_distinct) weight at each value
    at_or_above = np.cumsum(Wu[:, ::-1], axis=1)[:, ::-1]
    want = target * Wu.sum(1)
    i = np.clip((at_or_above >= want[:, None]).sum(1) - 1, 0, len(u) - 1)   # last value with >= want at or above
    r = np.arange(n_rep)
    at = Wu[r, i]
    above = at_or_above[r, i] - at
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(at > 0, (want - above) / at, 0.0)
    return u[i], np.clip(p, 0.0, 1.0)


def frac_tie(score, W, t, p):
    """Per replicate: weighted fraction accepted at (t, p) (score > t, plus p of score == t)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((W * (score[None, :] > t[:, None])).sum(1)
                + p * (W * (score[None, :] == t[:, None])).sum(1)) / W.sum(1)


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
    """{region: {metric: (n_rep,)}} for one tagger's predictions d and replicate weights W (matched working
    points with ties split)."""
    prob, lab = d["prob"], d["label"]
    ptau = prob[:, CI["taup"]] + prob[:, CI["taum"]]
    istau = (lab == CI["taup"]) | (lab == CI["taum"])
    D = {n: disc(prob, num, den) for n, (_, _, num, den) in PAIRS.items()}
    S = {n: np.isin(lab, [CI[c] for c in sig]) for n, (sig, _, _, _) in PAIRS.items()}
    B = {n: np.isin(lab, [CI[c] for c in bkg]) for n, (_, bkg, _, _) in PAIRS.items()}
    T = {(n, m): tie_thresholds(D[n][B[n]], W[:, B[n]], m) for n in PAIRS for m in MISTAGS}
    thr = {f"{n}_thr@{pct(m)}": T[(n, m)][0] for n in PAIRS for m in MISTAGS}
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
                o[f"{n}_eff@{pct(m)}"] = frac_tie(D[n][r & S[n]], W[:, r & S[n]], *T[(n, m)])
                o[f"{n}_mistag@{pct(m)}"] = frac_tie(D[n][r & B[n]], W[:, r & B[n]], *T[(n, m)])
        if reg == "incl":
            o.update(thr)
        out[reg] = o
    return out


# ------------------------------------------------------------------ loading
def load_trials():
    """{(variant, mask): {seed: test predictions}} of every trial of the current build's samples (dev files
    with a tag suffix and stale files of another sample set ignored)."""
    P = {}
    for f in sorted(glob.glob(os.path.join(C.OUT, "preds", "*.npz"))):
        m = re.fullmatch(r"([AI]{4})_(\w+?)_s(\d+)\.npz", os.path.basename(f))
        if not m or m.group(2) not in C.VARIANTS:
            continue              # dev runs carry a tag suffix
        d = np.load(f)
        if [str(x) for x in d["samples"]] != C.built_samples(m.group(1)):
            continue              # stale: an earlier run's sample set
        P.setdefault((m.group(2), m.group(1)), {})[int(m.group(3))] = {
            "prob": d["prob"].astype(np.float64), "label": d["label"], "pt": d["pt"], "eta": d["eta"],
            "sample": d["sample"], "samples": [str(x) for x in d["samples"]],
            "key": C.event_key(d["sample"], d["fileno"].astype(np.int64), d["evid"]),
            "train_counts": json.loads(str(d["train_counts"])), "n_epochs": len(d["loss"]),
            "features": [str(x) for x in d["features"]], "test_event_keys": d["test_event_keys"]}
    return P


def load_preds():
    """{(variant, mask): {"trials": {seed: d} (kept trials), "ens": d (mean softmax of the deployed trials),
    "best": d (the single best), "sel": the cell's selection}} from the trials and OUT/selection.json."""
    sp = os.path.join(C.OUT, "selection.json")
    if not os.path.exists(sp):
        raise SystemExit(f"no {sp}: run trial_select.py first")
    S = json.load(open(sp))["cells"]
    P = {}
    for cell, trials in load_trials().items():
        sel = S.get(f"{cell[0]}@{cell[1]}")
        if sel is None or sorted(trials) != sel["seeds"]:
            raise RuntimeError(f"{cell}: selection.json is stale (rerun trial_select.py)")
        if not sel["deployed_seeds"]:
            print(f"  {cell}: NO KEPT TRIAL, no result", flush=True)
            continue
        dep = [trials[s] for s in sel["deployed_seeds"]]
        for d in dep[1:]:
            if not (np.array_equal(d["key"], dep[0]["key"]) and np.array_equal(d["label"], dep[0]["label"])):
                raise RuntimeError(f"{cell}: the trials' test jets differ")
        ens = dict(dep[0], prob=np.mean([d["prob"] for d in dep], axis=0))
        P[cell] = {"trials": {s: trials[s] for s in sel["kept_seeds"]}, "ens": ens,
                   "best": trials[sel["best_seed"]], "sel": sel}
    return P


def test_events(P):
    """The study's test-event keys, which must be identical in every cell (the pairing of the bootstrap
    relies on it; they include events without an accepted labelled jet)."""
    ev = next(iter(P.values()))["ens"]["test_event_keys"]
    for c, x in P.items():
        for d in list(x["trials"].values()) + [x["ens"]]:
            if not np.array_equal(d["test_event_keys"], ev):
                raise RuntimeError(f"{c}: test events differ from the other cells")
            if np.setdiff1d(d["key"], ev).size:
                raise RuntimeError(f"{c}: jets outside the test events")
    print(f"  {len(ev)} test events, identical in all cells", flush=True)
    return ev


def replicates(ev, n_boot, seed=12345):
    """(1 + n_boot, n_events) event weights: the unit weights, then multinomial resamples."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(len(ev))] + [rng.multinomial(len(ev), np.full(len(ev), 1 / len(ev)))
                                           for _ in range(n_boot)]).astype(np.float64)


def compute(P, ev, cnt):
    """{cell: {"ens"|"best": {reg: {metric: (1+n_boot,)}}, "trials": {s: {reg: {metric: nominal}}},
    "n_deployed": n}}."""
    out = {}
    for cell, x in P.items():
        t0 = time.time()
        W = cnt[:, np.searchsorted(ev, x["ens"]["key"])]
        tr = {s: {r: {k: float(v[0]) for k, v in m.items()} for r, m in cell_metrics(d, W[:1]).items()}
              for s, d in x["trials"].items()}
        out[cell] = {"ens": cell_metrics(x["ens"], W), "best": cell_metrics(x["best"], W), "trials": tr,
                     "n_deployed": len(x["sel"]["deployed_seeds"])}
        check_matched(x["ens"], out[cell]["ens"]["incl"], cell)
        print(f"  {cell}: deployed {x['sel']['deployed_seeds']}, {len(tr)} kept ({time.time() - t0:.0f}s)",
              flush=True)
    return out


def check_matched(d, m, cell):
    """The vectorised tie split equals ngtagger.definitions.matched_wp on the nominal replicate."""
    from ngtagger.definitions.matched_wp import matched_point

    for n in WP_PAIRS:
        sig, bkg, num, den = PAIRS[n]
        D = disc(d["prob"], num, den)
        S, B = np.isin(d["label"], [CI[c] for c in sig]), np.isin(d["label"], [CI[c] for c in bkg])
        for x in MISTAGS:
            ref = matched_point(D[S], D[B], x)
            if not (abs(ref["efficiency"] - m[f"{n}_eff@{pct(x)}"][0]) < 1e-12
                    and abs(ref["realised_background"] - m[f"{n}_mistag@{pct(x)}"][0]) < 1e-12):
                raise RuntimeError(f"{cell} {n}@{pct(x)}: tie split differs from definitions.matched_wp")


def seed_stats(M, cell, reg, k):
    """Mean, SD and number of the KEPT trials' single-model values."""
    v = np.array([M[cell]["trials"][s][reg][k] for s in M[cell]["trials"]], float)
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else np.nan,
            float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v))


def step_value(M, a, b, reg, k, which="ens"):
    """Nominal difference a - b of the deployed ("ens") or single best ("best") taggers, its event-bootstrap
    std and its trial-spread error (kept-trial SD, over sqrt(n deployed) for the ensemble)."""
    if a not in M or b not in M:
        return None
    diff = M[a][which][reg][k] - M[b][which][reg][k]
    boot = float(np.nanstd(diff[1:], ddof=1)) if np.isfinite(diff[1:]).sum() > 2 else np.nan
    se = 0.0
    for c in (a, b):
        sd = seed_stats(M, c, reg, k)[1]
        se += sd ** 2 / (M[c]["n_deployed"] if which == "ens" else 1)
    return {"diff": float(diff[0]), "boot_err": boot, "seed_err": float(np.sqrt(se))}


def resolve(cell, mask):
    return (cell[0], mask if cell[1] == "M" else cell[1])


def ladder(M, regs, which="ens"):
    L = {}
    for mask in SP_MASKS:
        for lab, a, b in STEPS + JOINT_STEPS:
            for reg in regs:
                for k in [k for k in M[next(iter(M))][which][reg] if "_thr@" not in k]:
                    v = step_value(M, resolve(a, mask), resolve(b, mask), reg, k, which)
                    if v is not None:
                        L.setdefault(mask, {}).setdefault(lab, {}).setdefault(reg, {})[k] = v
    return L


def counts(P):
    """{mask: {sample: {class: [train, test]}}} from the baseline cells."""
    out = {}
    for (var, mask), x in P.items():
        if var != "baseline":
            continue
        d = x["ens"]
        tc = d["train_counts"]
        out[mask] = {s: {c: [tc[s][c][0], int(((d["label"] == CI[c]) & (d["sample"] == si)).sum())]
                         for c in CLASSES} for si, s in enumerate(d["samples"])}
    return out


def roc_points(d, name):
    """ROC of a pairwise discriminant of one tagger: (signal eff, background eff)."""
    sig, bkg, num, den = PAIRS[name]
    D = disc(d["prob"], num, den)
    Ds = np.sort(D[np.isin(d["label"], [CI[c] for c in sig])])
    Db = np.sort(D[np.isin(d["label"], [CI[c] for c in bkg])])
    t = np.unique(np.concatenate([Ds, Db]))[::-1]
    return 1 - np.searchsorted(Ds, t) / len(Ds), 1 - np.searchsorted(Db, t) / len(Db)


def reachability(P):
    """{cell: {pair: {wp: n_bkg_test_jets * mistag}}}."""
    out = {}
    for (v, m), x in P.items():
        d = x["ens"]
        out[f"{v}@{m}"] = {n: {pct(t): float(np.isin(d["label"], [CI[c] for c in PAIRS[n][1]]).sum() * t)
                               for t in MISTAGS} for n in WP_PAIRS}
    return out


def stock_wp(ev, cnt):
    """The stock NG tagger (scores emulated in each nano, per scenario's jets; not retrained) on each mask's
    test jets in acceptance: b eff / udsg mistag of bTagScore > STOCK_B_CUT, and its b eff at the matched
    points (ties split: the scores are 8-bit), on the trained taggers' paired event replicates.
    Returns {mask: {metric: [nominal, bootstrap std]} + counts; "<metric>_minus_IIII": [diff, std]}."""
    R, out = {}, {}
    for mask in C.MASKS:
        mp = C.manifest_path(mask)
        if not os.path.exists(mp):
            continue
        man = json.load(open(mp))
        b, lab, key = [], [], []
        for s, n, _ in man["files"]:
            d = np.load(os.path.join(C.cache_dir("jets"), C.tag(s, mask, n) + ".npz"))
            sel = d["is_test"] & (d["pt"] > C.PT_MIN) & (np.abs(d["eta"]) < C.ETA_MAX) & C.event_filter(d["evid"])[d["entry"]]
            b.append(d["stock_bTagScore"][sel])
            lab.append(d["label"][sel])
            key.append(C.event_key(man["samples"].index(s), n, d["evid"][d["entry"][sel]]))
        b, lab, key = np.concatenate(b).astype(float), np.concatenate(lab), np.concatenate(key)
        if np.setdiff1d(key, ev).size:
            raise RuntimeError(f"stock {mask}: jets outside the test events")
        W = cnt[:, np.searchsorted(ev, key)]
        isb, isu, isl = lab == CI["b"], np.isin(lab, [CI["light"], CI["gluon"]]), lab == CI["light"]
        cut = np.full(len(W), STOCK_B_CUT)
        r = {"b_eff@cut": frac_above(b[isb], W[:, isb], cut), "udsg_mistag@cut": frac_above(b[isu], W[:, isu], cut),
             "light_mistag@cut": frac_above(b[isl], W[:, isl], cut)}
        for x in MISTAGS:
            t, p = tie_thresholds(b[isu], W[:, isu], x)
            r[f"b_eff@{pct(x)}"] = frac_tie(b[isb], W[:, isb], t, p)
            r[f"udsg_mistag@{pct(x)}"] = frac_tie(b[isu], W[:, isu], t, p)
        R[mask] = r
        out[mask] = {k: [float(v[0]), float(np.nanstd(v[1:], ddof=1))] for k, v in r.items()}
        out[mask].update({"n_b": int(isb.sum()), "n_udsg": int(isu.sum()),
                          "frac_udsg_score0": float(np.mean(b[isu] == 0))})
    for mask in SP_MASKS:
        if mask in R and C.BASELINE_MASK in R:
            for k in R[mask]:
                d = R[mask][k] - R[C.BASELINE_MASK][k]
                out[mask][f"{k}_minus_IIII"] = [float(d[0]), float(np.nanstd(d[1:], ddof=1))]
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
    return f"{v['diff']:+.{nd}f} ± {err:.{nd}f}" + (f" ({sig:+.1f}σ)" if np.isfinite(sig) else "")


def short(lab):
    return lab.split("  ")[0]


STEP_NOTE = ("Steps: " + "; ".join(f"{short(lab)} = {lab.split('  ', 1)[1]}" for lab, _, _ in STEPS)
             + ". Cell: difference of the deployed taggers (mean of the best 3 trials) ± (event bootstrap ⊕ "
             "trial spread) (significance); the two error parts are listed separately in summary.json.")


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


def plot_rocs(pdf, P, STK=None):
    """Efficiency vs mistag of every cell's deployed tagger, with the matched working points and the
    statistical floor (< 10 background test jets above threshold)."""
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
                es, eb = roc_points(P[(var, mask)]["ens"], name)
                ok = eb > 0
                ax.plot(es[ok], eb[ok], "--" if mask == C.BASELINE_MASK else "-", color=col[mask], lw=1.5,
                        label=mask)
                if STK and var == "baseline" and name == "b_vs_udsg" and mask in STK:
                    ax.plot(STK[mask]["b_eff@cut"][0], STK[mask]["udsg_mistag@cut"][0], "*", ms=10,
                            color=col[mask], mec="k", mew=0.5,
                            label=f"stock NG bTagScore>{STOCK_B_CUT}" if mask == C.BASELINE_MASK else None)
            nb = min(np.isin(P[c]["ens"]["label"], [CI[x] for x in PAIRS[name][1]]).sum()
                     for c in P if c[0] == var)
            floor = 10.0 / max(nb, 1)
            ax.axhspan(1e-4, floor, color="#dddddd", alpha=0.6, lw=0,
                       label="< 10 bkg test jets (unmeasurable)" if (i, j) == (0, 0) else None)
            for x in MISTAGS:
                ax.axhline(x, color="k", lw=0.6, ls=":")
                ax.text(0.99, x * 1.1, wp_title(x), ha="right", va="bottom", fontsize=6,
                        transform=ax.get_yaxis_transform())
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 1.05)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.25, lw=0.5)
            sig, bkg = name.split("_vs_")
            ax.set_xlabel(f"{sig} efficiency")
            ax.set_ylabel(f"{bkg} mistag")
            ax.set_title(f"L{C.LEVEL[var]} ({var}): {name}", fontsize=9)
    hl = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    fig.legend(list(hl.values()), list(hl), loc="lower center", ncol=6, frameon=False,
               title="scenario (IIII = OT baseline)")
    fig.suptitle("Efficiency vs mistag on the test set (deployed tagger of each cell). Dotted: the matched "
                 "working points; grey: < 10 background test jets", fontsize=10)
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
        x = np.arange(len(regs_or_keys)) + (i - (len(MAIN_STEPS) - 1) / 2) * 0.15
        ax.errorbar(x, ys, yerr=es, fmt=STEP_MARKERS[i], color=STEP_COLORS[i], ms=5, lw=1, capsize=2,
                    label=lab, mfc="white" if "OT tracks" in lab else STEP_COLORS[i])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(regs_or_keys)))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(f"{mask}: {title}  (bars: event bootstrap ⊕ trial spread)", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def nominal_boot(m):
    """{reg: {metric: [nominal, bootstrap std]}} of a metric dict of (1+n_boot,) arrays."""
    return {r: {k: [float(v[0]), float(np.nanstd(v[1:], ddof=1)) if np.isfinite(v[1:]).sum() > 2 else None]
                for k, v in x.items()} for r, x in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--out", default=C.OUT)
    a = ap.parse_args()
    t0 = time.time()
    # matched "menu" WP: the udsg fake rate of the stock menu cut on IIII, applied to every cell
    global MISTAGS
    fr_menu, n_udsg_ref = stock_reference()
    WP_LABEL[fr_menu] = "menu"
    WP_NAME[fr_menu] = "menu-matched"
    MISTAGS = [fr_menu] + REPORT_MISTAGS
    print(f"menu WP: udsg fake rate of bTagScore > {STOCK_B_CUT} on IIII = {fr_menu:.4f} "
          f"({n_udsg_ref} udsg test jets)", flush=True)
    P = load_preds()
    if not P:
        raise SystemExit("no deployed tagger")
    print(f"cells: {sorted(P)}", flush=True)
    ev = test_events(P)
    cnt = replicates(ev, a.boot)
    M = compute(P, ev, cnt)
    regs = list(M[next(iter(M))]["ens"])
    L = {m: v for m, v in ladder(M, regs).items() if ("baseline", m) in M}
    LB = ladder(M, ["incl"], "best")
    N = counts(P)
    RCH = reachability(P)
    STK = stock_wp(ev, cnt)
    mva = {m: json.load(open(p)) for m in C.MASKS
           if os.path.exists(p := os.path.join(C.OUT, "mva", m, "heldout_check.json"))}
    samples = next(iter(P.values()))["ens"]["samples"]
    variants = [v for v in C.VARIANTS if any(c[0] == v for c in P)]
    sel_keys = ("launched", "failed", "outliers", "kept", "kept_seeds", "deployed_seeds", "best_seed", "flag")
    cells = {f"{v}@{m}": {"level": ("L0" if (v, m) == L0 else
                                    f"L{C.LEVEL[v]}" + ("'" if m == C.BASELINE_MASK else "")),
                          "selection": {k: P[(v, m)]["sel"][k] for k in sel_keys},
                          "n_epochs_kept": {s: d["n_epochs"] for s, d in P[(v, m)]["trials"].items()},
                          "features": P[(v, m)]["ens"]["features"],
                          "deployed": nominal_boot(M[(v, m)]["ens"]),
                          "single_best": nominal_boot({"incl": M[(v, m)]["best"]["incl"]}),
                          "kept_trials_incl": {s: x["incl"] for s, x in M[(v, m)]["trials"].items()},
                          "kept_mean_sd_incl": {k: seed_stats(M, (v, m), "incl", k)[:2]
                                                for k in M[(v, m)]["ens"]["incl"]}}
             for (v, m) in P}
    head = {mask: {lab.split("  ")[1]: L[mask][lab]["incl"] for lab in HEAD_LABEL.values() if lab in L[mask]}
            for mask in L}
    head_best = {mask: {lab.split("  ")[1]: {k: LB[mask][lab]["incl"][k] for k in KEY_METRICS}
                        for lab in HEAD_LABEL.values() if lab in LB.get(mask, {})} for mask in LB}
    summary = {
        "what": "SmartPixels L1 jet multiclass tagging on canon_v5 (refit v3): deployed taggers vs the OT baseline",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "samples": samples,
        "inputs": {"sp_masks": C.CANON["sp"], "IIII": C.CANON[C.BASELINE_MASK]},
        "acceptance": {"pt_min_GeV": C.PT_MIN, "abs_eta_max": C.ETA_MAX},
        "split": {"test_fraction": C.TEST_FRACTION, "validation_fraction_of_training_events": C.VAL_FRACTION,
                  "seed": C.SPLIT_SEED, "ttbar_train_only_files": list(C.TTBAR_TRAIN_ONLY),
                  "n_test_events": int(len(ev))},
        "selection_rules": json.load(open(os.path.join(C.OUT, "selection.json")))["rules"],
        "variants": C.VARIANTS, "steps_definition": {lab: [f"{x[0]}@{x[1]}" for x in (a_, b_)]
                                                     for lab, a_, b_ in STEPS + JOINT_STEPS},
        "headline_definition": "per SP mask and input set: deployed tagger on the mask minus deployed tagger on "
                               "IIII (inclusive test set); diff, boot_err (paired event bootstrap), seed_err "
                               "(trial spread)",
        "headline": head, "headline_single_best": head_best,
        "n_bootstrap": a.boot, "jet_counts_train_test": N, "track_mva_heldout": mva,
        "cells": cells, "steps": L,
        "working_points": {
            "definition": "matched fake rates, ties split (ngtagger.definitions.matched_wp): every cell's deployed "
                          "tagger gets the threshold on D at which the background (udsg; tau: light+gluon jets) "
                          "fraction of its own test jets in acceptance equals the target exactly, recomputed in "
                          "every replicate. 'menu' = the udsg fake rate of the stock menu cut bTagScore > 0.15 on "
                          "the IIII test jets; '10%'; '1%' (statistically weak).",
            "reference": {"stock_cut": f"bTagScore > {STOCK_B_CUT}", "mask": C.BASELINE_MASK,
                          "udsg_fake_rate_menu": fr_menu, "n_udsg_test_jets": n_udsg_ref},
            "fake_rate_targets": {pct(x): x for x in MISTAGS},
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
        text_page(pdf, "SmartPixels L1 jet tagging on canon_v5 (refit v3): retrained taggers vs the OT baseline", [
            "Scenarios: canonical coopt nanos, PU200. The tracks feeding the L1 vertex, PF, PUPPI and SC4 NG jets are the",
            "SmartPixels (SP) tracks of each IT active-layer mask (A/I = IT layer group active/inactive). SP masks: canon_v5",
            "(refit v3); IIII (passthrough, SP track = OT track, the OT-only baseline): canon_v4 (canon_v5 has no IIII).",
            f"Samples: {samples} combined. Split by event identity, identical in all masks: {C.TEST_FRACTION:.0%} test (TTbar",
            "test events only from f03-f10: f01/f02 trained the cluster-angle CNN); validation = a fixed "
            f"{C.VAL_FRACTION:.0%} of the training",
            "events, the same for every trial, cell and mask (early stopping, trial screen and choice only).",
            f"Jets: SC4 NG (L1puppiJetSC4NG), reco pT > {C.PT_MIN:g} GeV, |eta| < {C.ETA_MAX:g}; <= 16 constituents (framework cap).",
            "Labels (framework labels.py, identical in every scenario): b charm light gluon tau+ tau- muon electron.",
            "Tagger (every cell): framework DeepSetHGQ2 with upstream's ReLU on the class logits; framework loss, class",
            "weighting and callbacks; batch 128 / lr 1e-3 (the stock 2048 / 1e-2 does not learn on ~5k jets). CPU.",
            "Input sets: baseline = upstream hardware inputs; track = + the constituent track's parameters, fit quality,",
            "  resolution and IT hits; trackmva = + track-quality head outputs (gradient-boosted, fitted per scenario on",
            "  the training events).",
            "TRIALS (training policy): 10 per cell (+10 if fewer than 3 kept); failed = degenerate on the validation jets",
            "  (> 1% identical outputs or a working-point threshold on a tie block); outliers by a robust rule; the",
            "  DEPLOYED tagger = mean softmax of the 3 kept trials with the lowest validation loss (single best beside it).",
            "HEADLINE: with vs without SmartPixels per mask at equal inputs = deployed tagger on the mask minus on IIII.",
            "Working points: matched fake rates with ties split; menu = udsg fake rate of the stock cut bTagScore > 0.15",
            "  on IIII; 10%; 1% (weak).",
            f"Errors: 'boot' = paired event bootstrap ({a.boot} resamples of the common test events); 'seed' = trial",
            "  spread (kept-trial SD / sqrt(3) for the deployed ensemble); sigma = diff / (boot (+) seed).",
            "DESCRIPTIVE ONLY: the SmartPixels angles are placeholders and the cluster CNN is a stand-in for the sensor.",
            "Caveats: limited statistics; head/tagger correlation ignored; 16-constituent cap; recipe not certified",
            "saturated; SP covariance = KF posterior.",
        ])
        rows = [[f"{v}@{m}", x["sel"]["launched"], x["sel"]["failed"], x["sel"]["outliers"], x["sel"]["kept"],
                 " ".join(map(str, x["sel"]["deployed_seeds"])),
                 " ".join(f"{x['sel']['trials'][str(s)]['val_closs']:.3f}" for s in x["sel"]["deployed_seeds"]),
                 x["sel"]["flag"] or ""] for (v, m), x in sorted(P.items())]
        table_page(pdf, "Trials per cell: launched, failed, outliers, kept, deployed (chosen on validation jets)",
                   ["cell", "launched", "failed", "outliers", "kept", "deployed seeds", "their validation loss", "flag"],
                   rows, note="Failed = degenerate on the validation jets (the d0 study's screen); outliers: robust "
                              "z of the validation loss or b vs udsg AUC beyond 3.5 (cells with >= 5 non-failed).")
        rows = []
        for mask in [m for m in C.MASKS if m in N]:
            for s, cc in N[mask].items():
                rows.append([mask, s] + [f"{cc[c][0]} / {cc[c][1]}" for c in CLASSES])
        table_page(pdf, "Jet counts per class: train / test (acceptance applied; train includes the validation jets)",
                   ["mask", "sample"] + CLASSES, rows)
        # HEADLINE: one page per input set
        for v in variants:
            rows = []
            for k in KEY_METRICS:
                o = cells.get(f"{v}@{C.BASELINE_MASK}", {}).get("deployed", {}).get("incl", {}).get(k)
                row = [k, fmt(*o) if o else "-"]
                for mask in SP_MASKS:
                    x = cells.get(f"{v}@{mask}", {}).get("deployed", {}).get("incl", {}).get(k)
                    lab = HEAD_LABEL[v]
                    row += [fmt(*x) if x else "-", fmt_step(L.get(mask, {}).get(lab, {}).get("incl", {}).get(k))]
                rows.append(row)
            table_page(pdf, f"HEADLINE, input set '{v}': with vs without SmartPixels (deployed taggers, inclusive)",
                       ["metric", "IIII (OT)"] + [h for m in SP_MASKS for h in (m, f"{m} - IIII")], rows,
                       fontsize=6.3, note="Absolute values ± event bootstrap. Differences ± (event bootstrap ⊕ trial "
                                          "spread) (significance). Descriptive only (placeholder angles).")
        rows = []
        for k in KEY_METRICS:
            rows.append([k] + [fmt_step(LB.get(m, {}).get(lab, {}).get("incl", {}).get(k))
                               for lab, _, _ in MAIN_STEPS for m in SP_MASKS])
        table_page(pdf, "HEADLINE with the single best trial per cell (instead of the best-3 ensemble)",
                   ["metric"] + [f"{lab.split('  ')[0].split('-')[0]}: {m}" for lab, _, _ in MAIN_STEPS
                                 for m in SP_MASKS], rows, fontsize=6.0,
                   note="Error = event bootstrap ⊕ kept-trial SD (one model's spread).")
        if STK:
            ks = ["b_eff@cut", "udsg_mistag@cut"] + [f"b_eff@{pct(x)}" for x in MISTAGS]
            rows = [[m, f"{STK[m]['n_b']} / {STK[m]['n_udsg']}"] + [fmt(*STK[m][k]) for k in ks]
                    + [fmt(*STK[m][f"b_eff@{pct(MISTAGS[0])}_minus_IIII"]) if m != C.BASELINE_MASK else "-",
                       fmt(*STK[m]["b_eff@10%_minus_IIII"]) if m != C.BASELINE_MASK else "-"]
                    for m in STK]
            table_page(pdf, "Control: the stock NG tagger (nano scores, not retrained) on each scenario's test jets",
                       ["mask", "test b / udsg jets"] + ks + ["b eff @menu - IIII", "b eff @10% - IIII"], rows,
                       fontsize=6.5, note="Matched points with ties split (8-bit scores); errors: the same paired "
                                          "event bootstrap as the trained taggers.")
        for mask in [m for m in SP_MASKS if m in L]:
            hdr = ["metric (inclusive)", "L0 (IIII) abs."] + [short(lab) for lab, _, _ in STEPS]
            rows = []
            for k in KEY_METRICS + ["b_vs_light_auc", "c_vs_light_auc"] + ALL_AUCS:
                o = cells.get(f"baseline@{C.BASELINE_MASK}", {}).get("deployed", {}).get("incl", {}).get(k)
                rows.append([k, fmt(*o) if o else "-"] + [fmt_step(L[mask].get(lab, {}).get("incl", {}).get(k))
                                                          for lab, _, _ in STEPS])
            table_page(pdf, f"All steps ({mask} vs IIII), inclusive test set", hdr, rows, fontsize=6.0,
                       note=STEP_NOTE + " L0 column: absolute value ± event bootstrap.")
        for mask in [m for m in SP_MASKS if m in L and any(lab in L[m] for lab, _, _ in JOINT_STEPS)]:
            hdr = ["metric (inclusive)"] + [short(lab) for lab, _, _ in JOINT_STEPS]
            rows = [[k] + [fmt_step(L[mask].get(lab, {}).get("incl", {}).get(k)) for lab, _, _ in JOINT_STEPS]
                    for k in KEY_METRICS + ["b_vs_light_auc", "c_vs_light_auc"] + ALL_AUCS]
            table_page(pdf, f"Jointly trained track-quality heads ({mask}), inclusive test set", hdr, rows,
                       fontsize=6.0, note="Steps: " + "; ".join(f"{short(lab)} = {lab.split('  ', 1)[1]}"
                                                               for lab, _, _ in JOINT_STEPS)
                       + ". Deployed taggers (best-3 mean); ± (event bootstrap ⊕ trial spread) (significance).")
        for mask in [m for m in SP_MASKS if m in L]:
            aucs = ALL_AUCS + ["b_vs_udsg_auc", "c_vs_udsg_auc", "tau_vs_jet_auc", "b_vs_light_auc", "c_vs_light_auc"]
            plot_steps(pdf, L, mask, "region:incl", aucs, [k.replace("auc_", "").replace("_auc", "") for k in aucs],
                       "with vs without SmartPixels per input set: per class (one-vs-rest AUC) and pairwise AUCs",
                       "Δ AUC")
            bins = [r for r in regs if r.startswith(("abs_eta:", "pt:"))]
            for k, lab in (("b_vs_udsg_eff@menu", "Δ b eff (menu-matched udsg fake rate)"),
                           ("b_vs_udsg_eff@10%", "Δ b eff (10% udsg mistag)"),
                           ("b_vs_udsg_eff@1%", "Δ b eff (1% udsg mistag)"),
                           ("c_vs_udsg_eff@10%", "Δ c eff (10% udsg mistag)"),
                           ("tau_vs_jet_eff@10%", "Δ tau eff (10% light+gluon mistag)"),
                           ("macro_auc", "Δ macro AUC")):
                plot_steps(pdf, L, mask, f"metric:{k}", bins, [b.replace("abs_eta:", "|η| ").replace("pt:", "pT ")
                                                               for b in bins],
                           f"with vs without SmartPixels in |η| and pT bins: {k}", lab)
            rows = []
            for reg in bins + [r for r in regs if r.startswith("sample:")]:
                for k in ("b_vs_udsg_eff@menu", "b_vs_udsg_eff@10%", "b_vs_udsg_eff@1%", "c_vs_udsg_eff@10%",
                          "tau_vs_jet_eff@menu", "tau_vs_jet_eff@10%", "macro_auc"):
                    o = cells.get(f"baseline@{C.BASELINE_MASK}", {}).get("deployed", {}).get(reg, {}).get(k)
                    rows.append([reg, k, fmt(*o) if o else "-"] + [fmt_step(L[mask].get(lab, {}).get(reg, {}).get(k))
                                                                  for lab, _, _ in STEPS])
            table_page(pdf, f"All steps ({mask}) by region", ["region", "metric", "L0 (IIII) abs."]
                       + [short(lab) for lab, _, _ in STEPS], rows, fontsize=4.8,
                       note=STEP_NOTE + " Efficiencies at the INCLUSIVE matched working point of each cell.")
        for n in WP_PAIRS:
            rows = []
            for c in [c for c in [(v, m) for v in C.VARIANTS for m in C.MASKS] if c in M]:
                row = [f"{c[0]}@{c[1]}"]
                for x in MISTAGS:
                    e = cells[f"{c[0]}@{c[1]}"]["deployed"]["incl"][f"{n}_eff@{pct(x)}"]
                    t = cells[f"{c[0]}@{c[1]}"]["deployed"]["incl"][f"{n}_thr@{pct(x)}"]
                    nb = RCH[f"{c[0]}@{c[1]}"][n][pct(x)]
                    flag = " [unreachable]" if nb < 1 else (" [<10 bkg jets]" if nb < 10 else "")
                    row += [fmt(*e) + flag, fmt(t[0], None, 4)]
                rows.append(row)
            hdr = ["cell"] + [f"{wp_title(x)}: {k}" for x in MISTAGS for k in ("eff", "threshold on D")]
            table_page(pdf, f"Working points per cell: {n} (deployed tagger; eff ± event bootstrap)", hdr,
                       rows, fontsize=6,
                       note="Threshold = D value at which the background fraction of the cell's own test jets equals "
                            "the target (ties split). [unreachable]: n_bkg x mistag < 1 test jet; [<10 bkg jets]: "
                            "the point rests on < 10 background jets.")
        cl = [c for c in [(v, m) for v in C.VARIANTS for m in C.MASKS] if c in M]
        for chunk in (KEY_METRICS + ["b_vs_light_auc", "c_vs_light_auc"], ALL_AUCS):
            rows = [[k] + [fmt(*cells[f"{c[0]}@{c[1]}"]["deployed"]["incl"][k]) for c in cl] for k in chunk]
            table_page(pdf, "Absolute inclusive values per cell (deployed tagger ± event bootstrap)", ["metric"]
                       + [f"{c[0]}@{c[1]}" for c in cl], rows, fontsize=5.3)
        plot_rocs(pdf, P, STK)
        if mva:
            rows = [[m, v.get("coll"), v.get("n_test_tracks"), fmt(v.get("auc_clean")),
                     fmt(v.get("median_sigma_hat_d0_um"), None, 1), fmt(v.get("sigma68_d0_resid_um"), None, 1),
                     fmt(v.get("coverage_1sigma_d0"))] for m, v in mva.items()]
            table_page(pdf, "Track-quality heads (trackmva inputs): held-out check on the test events' tracks",
                       ["mask", "heads", "test tracks", "AUC P(clean)", "median sigma_hat(d0) [um]",
                        "68% |d0 resid| [um]", "frac |resid| < sigma_hat"], rows,
                       note="SP masks: refit heads on the SP tracks (canon_v5); IIII: OT-only heads (canon_v4). "
                            "Truth = the OT track's TP.")
    print(f"wrote {pdfp} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
