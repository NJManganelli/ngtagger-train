"""spixbits analysis, step 1: per-arm summaries with the paired event bootstrap (PLAN.md section 6).

Reads the scan's units read-only (never *.invalid-*), one arm of one scenario at a time, and writes one small cache
file per arm x scenario to ~/smartpixels/scratch_local/spixbits_analysis/cache/. An arm x scenario is processed only
when all five scan files hold it; a cache entry is redone when its unit files changed. Re-runnable at any time.

  python core.py [--scen AAAA AAAI IAAA] [--arms NAME ...] [--status]

Rules implemented (PLAN.md 6, reviews/stats-review-01.md 1, 3, 5):
- fixed population per scenario = every track of the truth file; residual = delivered - tracking particle, the
  outer-tracker values where the arm's refit did not run;
- width = 68 % (and 95 %) quantile of |residual| about zero;
- one multinomial event-weight matrix (1,000 replicas), keyed by (run, lumi, event), the same for every arm and
  scenario;
- cells by the particle's |eta|, pT and origin; counts are of the population and do not depend on the arm.

Frames: delivered parameter 4 = d0 [cm], 3 = z0 [cm]. Hit quantities are stored in code order (localX, localY,
code alpha, code beta) = SmartPixels (y, x, beta, alpha); everything written here is named in the SmartPixels frame.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
# Which scan is analysed: SPIXBITS_SCAN_DIR = the scan's output directory (holding units/ and arms.json), SPIXBITS_TAG =
# a short name that keeps this scan's caches and results apart (results/<tag>/, README_<tag>.md). Unset = the first
# scan (random choice under the cap of three clusters per sub-area). run_all.py sets both from --scan-dir and --tag.
SCAN = Path(os.environ.get("SPIXBITS_SCAN_DIR") or HOME / "smartpixels/scratch_local/spixbits").expanduser()
TAG = os.environ.get("SPIXBITS_TAG", "")
UNITS = SCAN / "units"
CACHE_ROOT = HOME / "smartpixels/scratch_local/spixbits_analysis"
CACHE = CACHE_ROOT / "cache" if not TAG else CACHE_ROOT / f"cache_{TAG}"
FILES = (5, 7, 8, 9, 10)
SCENARIOS = ("AAAA", "AAAI", "IAAA")
NREP = 1000
SEED = 20261111
ETA_EDGES = (0.0, 1.1, 1.8, 2.4)
PT_EDGES = (2.0, 3.0, 5.0, 10.0, np.inf)
ETA_NAMES = ("|eta| 0-1.1", "|eta| 1.1-1.8", "|eta| 1.8-2.4")
PT_NAMES = ("pT 2-3", "pT 3-5", "pT 5-10", "pT > 10")
ORIGIN_NAMES = ("prompt", "b/c daughters", "displaced inside IL1", "born beyond IL1")
HITQ = ("y", "x", "beta", "alpha")  # stored order localX, localY, code alpha, code beta
OT_ARM = "outer_tracker_only"
CM2UM = 1.0e4
VERSION = 4


def registry():
    return json.load(open(SCAN / "arms.json"))["arms"]


def discover():
    """{scenario: {arm: {file: path}}} from the unit files that exist (a file that exists is complete)."""
    out = {s: {} for s in SCENARIOS}
    for p in sorted(UNITS.glob("*/*.npz")):
        if p.parent.name == "truth":
            continue
        m = re.match(r"([A-Z]{4})_f(\d+)_", p.name)
        if not m or m.group(1) not in out:
            continue
        try:
            keys = np.load(p).files
        except Exception:  # being renamed / unreadable: skip, picked up at the next run
            continue
        for arm in {k.split("__")[0] for k in keys}:
            out[m.group(1)].setdefault(arm, {})[int(m.group(2))] = p
    return out


def complete(disc, scen):
    return sorted(a for a, d in disc[scen].items() if all(f in d for f in FILES))


class Truth:
    """Truth of one scenario, the five scan files concatenated in FILES order."""

    def __init__(self, scen):
        parts = [np.load(UNITS / "truth" / f"{scen}_f{f:02d}_truth.npz") for f in FILES]
        self.n_per_file = [len(p["run"]) for p in parts]
        cat = lambda k: np.concatenate([p[k] for p in parts])
        for k in ("run", "lumi", "event", "track", "tp_pt", "tp_eta", "tp_d0", "tp_z0", "origin", "ot_a",
                  "ot_cov_diag", "slot_surface", "crossing_cluster", "crossing_cluster_sent_under_cap", "disc_going"):
            setattr(self, k, cat(k))
        self.n = len(self.run)
        key = np.stack([self.run, self.lumi, self.event], 1)
        self.event_keys, self.ev = np.unique(key, axis=0, return_inverse=True)
        self.ev = self.ev.ravel()
        self.n_events = len(self.event_keys)
        aeta = np.abs(self.tp_eta)
        self.ieta = np.clip(np.digitize(aeta, ETA_EDGES) - 1, 0, 2)
        self.ipt = np.clip(np.digitize(self.tp_pt, PT_EDGES) - 1, 0, 3)
        self.prompt = self.origin == 0
        # innermost visited surface = the last used slot (visits run outside-in)
        used = self.slot_surface >= 0
        self.n_slots = used.sum(1)
        self.inner_slot = np.maximum(self.n_slots - 1, 0)
        rows = np.arange(self.n)
        self.inner_surface = np.where(self.n_slots > 0, self.slot_surface[rows, self.inner_slot], -1)
        self.inner_has_crossing = (self.n_slots > 0) & (self.crossing_cluster[rows, self.inner_slot] >= 0)
        self.inner_sent = self.crossing_cluster_sent_under_cap[rows, self.inner_slot]
        self.p = self.tp_pt * np.cosh(self.tp_eta)


def weights(truth):
    """(NREP, n_events) multinomial weights; the event list is sorted by key, so every scenario gets the same matrix."""
    rng = np.random.default_rng(SEED)
    return rng.multinomial(truth.n_events, np.full(truth.n_events, 1.0 / truth.n_events), size=NREP).astype(np.int16)


# ---------------------------------------------------------------- cells
def pooled_cells(truth, threshold):
    """Prompt cells after the plan's pooling order, applied per |eta| range: pT > 10 into 5-10; then 3-5 into 2-3;
    then the two forward |eta| ranges together. Returns (labels per track, -1 = not prompt; list of names; counts)."""
    cnt = np.zeros((3, 4), int)
    np.add.at(cnt, (truth.ieta[truth.prompt], truth.ipt[truth.prompt]), 1)
    opts = ([[0], [1], [2], [3]], [[0], [1], [2, 3]], [[0, 1], [2, 3]])
    names = {(0,): "pT 2-3", (1,): "pT 3-5", (2,): "pT 5-10", (3,): "pT > 10", (2, 3): "pT >= 5", (0, 1): "pT 2-5"}

    def level(c):
        for k, o in enumerate(opts):
            if all(c[g].sum() >= threshold for g in o):
                return k
        return None

    lev = [level(cnt[i]) for i in range(3)]
    groups = []  # (eta rows, pt group)
    if lev[1] is None or lev[2] is None:
        k = level(cnt[1] + cnt[2])
        k = 2 if k is None else k
        rows = [((0,), opts[lev[0] if lev[0] is not None else 2]), ((1, 2), opts[k])]
    else:
        rows = [((i,), opts[lev[i] if lev[i] is not None else 2]) for i in range(3)]
    labels = np.full(truth.n, -1, np.int16)
    out_names, counts = [], []
    for etas, o in rows:
        for g in o:
            m = truth.prompt & np.isin(truth.ieta, etas) & np.isin(truth.ipt, g)
            labels[m] = len(out_names)
            en = ETA_NAMES[etas[0]] if len(etas) == 1 else "|eta| 1.1-2.4"
            out_names.append(f"{en}, {names[tuple(g)]} GeV")
            counts.append(int(m.sum()))
    return labels, out_names, counts


def partitions(truth):
    """The cell sets every arm is read on. Each is a partition (labels, names); -1 = not in any cell."""
    P = {}
    P["origin"] = (truth.origin.astype(np.int16), list(ORIGIN_NAMES))
    lab, names = np.full(truth.n, -1, np.int16), []
    hi = truth.tp_pt >= 5.0
    for i in range(3):
        for h in (False, True):
            lab[truth.prompt & (truth.ieta == i) & (hi == h)] = len(names)
            names.append(f"prompt, {ETA_NAMES[i]}, {'pT >= 5' if h else 'pT 2-5'} GeV")
    for o in (1, 2):
        for h in (False, True):
            lab[(truth.origin == o) & (hi == h)] = len(names)
            names.append(f"{ORIGIN_NAMES[o]}, {'pT >= 5' if h else 'pT 2-5'} GeV")
    P["six"] = (lab, names)
    # the two forward |eta| ranges pooled (the plan's last pooling step), for readings that need 1,000 tracks
    lab, names = np.full(truth.n, -1, np.int16), []
    for h in (False, True):
        lab[truth.prompt & (truth.ieta >= 1) & (hi == h)] = len(names)
        names.append(f"prompt, |eta| 1.1-2.4, {'pT >= 5' if h else 'pT 2-5'} GeV")
    P["forward_pooled"] = (lab, names)
    lab, names, _ = pooled_cells(truth, 200)
    P["twelve"] = (lab, ["prompt, " + n for n in names])
    return P


# ---------------------------------------------------------------- bootstrap engines
def boot_widths(absr, ev, labels, ngroups, W, qs=(0.68, 0.95), chunk=100):
    """Quantiles of |r| about zero per group: point (ngroups, len(qs)) and replicas (ngroups, NREP) of qs[0].
    Replica quantile = the smallest value whose cumulated event weight reaches q x the replica's total weight."""
    point = np.full((ngroups, len(qs)), np.nan, np.float32)
    reps = np.full((ngroups, W.shape[0]), np.nan, np.float32)
    for g in range(ngroups):
        idx = np.flatnonzero(labels == g)
        if len(idx) < 20:
            continue
        o = np.argsort(absr[idx], kind="stable")
        vals, evs = absr[idx][o], ev[idx][o]
        m = len(vals)
        for j, q in enumerate(qs):
            point[g, j] = vals[min(int(np.ceil(q * m)) - 1, m - 1)]
        for c0 in range(0, W.shape[0], chunk):
            cs = np.cumsum(W[c0:c0 + chunk][:, evs], axis=1, dtype=np.int32)
            tgt = np.ceil(qs[0] * cs[:, -1]).astype(np.int32)
            pos = (cs >= tgt[:, None]).argmax(axis=1)
            reps[g, c0:c0 + chunk] = vals[pos]
    return point, reps


def boot_auc(score, ev, pos, neg, W, chunk=100):
    """Area under the curve of `score` separating pos from neg (higher = pos); point and replicas."""
    idx = np.flatnonzero(pos | neg)
    o = np.argsort(score[idx], kind="stable")
    idx = idx[o]
    ip, evs = pos[idx], ev[idx]
    cn = np.cumsum(~ip)
    point = float((cn[ip]).sum() / (ip.sum() * (~ip).sum())) if ip.any() and (~ip).any() else np.nan
    reps = np.full(W.shape[0], np.nan, np.float32)
    for c0 in range(0, W.shape[0], chunk):
        w = W[c0:c0 + chunk][:, evs].astype(np.float32)
        wn = np.where(ip[None, :], 0.0, w)
        cs = np.cumsum(wn, axis=1, dtype=np.float64)
        num = (np.where(ip[None, :], w, 0.0) * cs).sum(1)
        wp = w[:, ip].sum(1)
        reps[c0:c0 + chunk] = num / np.maximum(wp * cs[:, -1], 1e-9)
    return point, reps


def per_event(x, ev, n_events, mask=None):
    """Sum of x per event (float32), for mean-type statistics bootstrapped as W @ sums."""
    if mask is not None:
        x = np.where(mask, x, 0)
    return np.bincount(ev, weights=np.asarray(x, np.float64), minlength=n_events).astype(np.float32)


# ---------------------------------------------------------------- one arm
ARM_KEYS = ("a", "cov", "refit", "forced_a", "forced_cov", "forced_refit", "n_accepted", "hit_code", "hit_abs_pull",
            "hit_bits", "hit_cluster", "hit_choice", "hit_candidates", "hit_candidates_classC", "crossing_in_window",
            "il1_crossing_dx_um", "il1_crossing_dy_um")


def load_arm(paths, arm, keys=ARM_KEYS):
    parts = [np.load(paths[f]) for f in FILES]
    return {k: np.concatenate([p[f"{arm}__{k}"] for p in parts]) for k in keys}


def rms5(x):
    x = x[np.isfinite(x)]
    k = x[np.abs(x) < 5.0]
    return (float(np.sqrt(np.mean(k.astype(np.float64) ** 2))) if len(k) else np.nan,
            float(1.0 - len(k) / max(len(x), 1)), int(len(x)))


def summarize(arm, A, T, W, P, ref_clusters, forced_trunc):
    """Everything kept for one arm of one scenario. A = arm arrays (None for the outer-tracker-only pseudo-arm)."""
    out = {}
    ot = A is None
    a = T.ot_a if ot else A["a"]
    r = {"d0": (a[:, 4] - T.tp_d0) * CM2UM, "z0": (a[:, 3] - T.tp_z0) * CM2UM}
    for pname in ("origin", "six", "forward_pooled", "twelve"):
        lab, names = P[pname]
        for q, v in r.items():
            pt, rp = boot_widths(np.abs(v).astype(np.float32), T.ev, lab, len(names), W)
            out[f"w_{pname}_{q}_point"] = pt
            out[f"w_{pname}_{q}_reps"] = rp if pname != "twelve" else rp.astype(np.float16)
    stats = {}
    pr = T.prompt
    stats["n_prompt"] = int(pr.sum())
    if ot:
        out["stats"] = json.dumps(stats)
        return out
    # forced refits (precision effect): the arm's measurements on the clusters "implemented now" accepted
    fa = A["forced_a"]
    fr = {"d0": (fa[:, 4] - T.tp_d0) * CM2UM, "z0": (fa[:, 3] - T.tp_z0) * CM2UM}
    for pname in ("origin", "six"):
        lab, names = P[pname]
        for q, v in fr.items():
            pt, rp = boot_widths(np.abs(v).astype(np.float32), T.ev, lab, len(names), W)
            out[f"fw_{pname}_{q}_point"] = pt
            out[f"fw_{pname}_{q}_reps"] = rp.astype(np.float16)
    # forced RMS inside a fixed truncation, per |eta| range (prompt, pT inclusive): per-event sums
    if forced_trunc is not None:
        for q in ("d0", "z0"):
            s2 = np.zeros((3, T.n_events), np.float32)
            cn = np.zeros((3, T.n_events), np.float32)
            for i in range(3):
                m = pr & (T.ieta == i) & (np.abs(fr[q]) < forced_trunc[q][i])
                s2[i] = per_event(fr[q].astype(np.float64) ** 2, T.ev, T.n_events, m)
                cn[i] = per_event(np.ones(T.n), T.ev, T.n_events, m)
            out[f"frms_{q}_s2"], out[f"frms_{q}_n"] = s2, cn
    # forced covariance diagonals (for the covariance forecast, PLAN 4a): kept per track only for the anchors
    out["_forced_var"] = np.stack([A["forced_cov"][:, 14], A["forced_cov"][:, 12]], 1)  # d0d0, z0z0 [cm^2]
    # mean-type statistics as per-event sums: prompt tracks
    refit = A["refit"]
    pe = {}
    pe["n"] = per_event(pr, T.ev, T.n_events)
    pe["refit"] = per_event(pr & refit, T.ev, T.n_events)
    pe["tail100"] = per_event(pr & (np.abs(r["d0"]) > 100.0), T.ev, T.n_events)
    rows = np.arange(T.n)
    code_in = A["hit_code"][rows, T.inner_slot]
    den = pr & T.inner_has_crossing
    pe["inner_den"] = per_event(den, T.ev, T.n_events)
    pe["inner_correct"] = per_event(den & (code_in == 1), T.ev, T.n_events)
    out["pe_names"] = np.array(list(pe))
    out["pe"] = np.stack([pe[k] for k in pe])
    stats["refit_fraction_prompt"] = float(refit[pr].mean())
    stats["refit_fraction_all"] = float(refit.mean())
    stats["forced_refit_fraction_prompt"] = float(A["forced_refit"][pr].mean())
    nd = max(int(den.sum()), 1)
    inwin = A["crossing_in_window"][rows, T.inner_slot]
    stats["inner"] = {
        "n_tracks_with_crossing_cluster": int(den.sum()),
        "correct_cluster_accepted": float((den & (code_in == 1)).sum() / nd),
        "other_cluster_of_the_particle": float((den & (code_in == 2)).sum() / nd),
        "wrong_cluster": float((den & (code_in >= 3)).sum() / nd),
        "no_hit": float((den & (code_in == 0)).sum() / nd),
        "correct_not_in_window_or_not_sent": float((den & ~inwin).sum() / nd),
        "correct_in_window_not_accepted": float((den & inwin & (code_in != 1)).sum() / nd),
        "correct_in_window_no_hit_taken": float((den & inwin & (code_in == 0)).sum() / nd),
        "correct_dropped_by_cap": float((den & ~T.inner_sent).sum() / nd),
    }
    hc = A["hit_code"][pr]
    acc = hc > 0
    nacc = max(int(acc.sum()), 1)
    bits = A["hit_bits"][pr]
    cand = A["hit_candidates"][pr].astype(np.int64)
    candC = A["hit_candidates_classC"][pr].astype(np.int64)
    isC = (bits & 16) > 0
    stats["hits"] = {
        "accepted_per_track": float(acc.sum() / pr.sum()),
        "wrong_cluster_share_of_accepted": float((hc >= 3).sum() / nacc),
        "other_own_cluster_share_of_accepted": float((hc == 2).sum() / nacc),
        "no_angle_truth_share_of_accepted": float((acc & isC).sum() / nacc),
        "no_angle_truth_share_of_candidates": float(candC.sum() / max(cand.sum(), 1)),
        "wrong_share_among_accepted_without_angle_truth": float(((hc >= 3) & isC).sum() / max((acc & isC).sum(), 1)),
        "wrong_share_among_accepted_with_angle_truth": float(((hc >= 3) & ~isC).sum() / max((acc & ~isC).sum(), 1)),
        "wrong_accepted_per_track_without_angle_truth": float(((hc >= 3) & isC).sum() / pr.sum()),
        "wrong_accepted_per_track_with_angle_truth": float(((hc >= 3) & ~isC).sum() / pr.sum()),
    }
    # honesty: correct-hit pulls (innovation pulls of the accepted crossing cluster), prompt tracks
    hp = A["hit_abs_pull"][pr].astype(np.float32)
    surf = T.slot_surface[pr]
    pulls = {}
    for sel_name, sel in (("all surfaces", hc == 1), ("barrel layer 1", (hc == 1) & (surf == 1)),
                          ("barrel layers 2-4", (hc == 1) & (surf >= 2) & (surf <= 4)), ("discs", (hc == 1) & (surf >= 11))):
        d = {}
        for j, qn in enumerate(HITQ):
            v = hp[..., j][sel]
            v = v[v > 0]
            if len(v) < 50:
                continue
            rm, beyond, n = rms5(v)
            d[qn] = {"rms_within_5": rm, "share_beyond_5": beyond, "n": n, "q68_over_0.9945": float(np.quantile(v, 0.68) / 0.9945)}
        pulls[sel_name] = d
    stats["correct_hit_pull"] = pulls
    cov = A["cov"]
    ok = pr & refit
    tp = {}
    for q, ci in (("d0", 14), ("z0", 12)):
        sig = np.sqrt(np.maximum(cov[ok, ci], 1e-30)) * CM2UM
        rm, beyond, n = rms5(r[q][ok] / sig)
        tp[q] = {"rms_within_5": rm, "share_beyond_5": beyond, "n": n,
                 "q68_over_0.9945": float(np.quantile(np.abs(r[q][ok] / sig), 0.68) / 0.9945),
                 "median_sigma_um": float(np.median(sig))}
        for i in range(3):
            mm = ok & (T.ieta == i)
            s_ = np.sqrt(np.maximum(cov[mm, ci], 1e-30)) * CM2UM
            tp[q][ETA_NAMES[i]] = rms5(r[q][mm] / s_)[0]
    stats["track_pull"] = tp
    # same accepted clusters as the anchors (ordered list of accepted cluster or none per slot), prompt
    same = {}
    for name, hcl in ref_clusters.items():
        same[name] = float((A["hit_cluster"][pr] == hcl[pr]).all(1).mean())
    stats["same_clusters_share"] = same
    # single-hit axis: emulated position of the barrel layer-1 crossing cluster minus the helix, p >= 40 GeV, prompt
    sh = {}
    for qn, k in (("y", "il1_crossing_dx_um"), ("x", "il1_crossing_dy_um")):
        v = A[k][pr & (T.p >= 40.0)]
        v = v[np.isfinite(v)]
        if len(v) > 30:
            w68 = float(np.quantile(np.abs(v), 0.68))
            core = v[np.abs(v) < 5 * w68]
            sh[qn] = {"n": int(len(v)), "w68_um": w68, "rms_um": float(np.sqrt(np.mean(core.astype(np.float64) ** 2))),
                      "rms_all_um": float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))}
    stats["single_hit_layer1_p40"] = sh
    # secondary: b/c daughters against prompt by |d0| and |d0|/sigma(d0)
    d0 = np.abs(a[:, 4]) * CM2UM
    sg = d0 / (np.sqrt(np.maximum(cov[:, 14], 1e-30)) * CM2UM)
    bc = T.origin == 1
    bcs = bc & (np.abs(T.tp_d0) * CM2UM < 200.0)
    aucs = {}
    for nm, sc, pos in (("abs_d0", d0, bc), ("abs_d0_over_sigma", sg, bc), ("abs_d0__true_below_200um", d0, bcs),
                        ("abs_d0_over_sigma__true_below_200um", sg, bcs)):
        pt, rp = boot_auc(sc.astype(np.float32), T.ev, pos, pr, W)
        aucs[nm] = pt
        out[f"auc_{nm}_reps"] = rp
    stats["auc_bc_vs_prompt"] = aucs
    stats["prompt_share_d0_off_by_more_than_100um"] = float((np.abs(r["d0"][pr]) > 100).mean())
    out["stats"] = json.dumps(stats)
    return out


def signature(paths):
    return json.dumps({str(f): [paths[f].name, paths[f].stat().st_mtime_ns, paths[f].stat().st_size] for f in FILES} | {"v": VERSION})


def cache_path(scen, arm):
    return CACHE / scen / f"{arm}.npz"


def load_cache(scen, arm):
    p = cache_path(scen, arm)
    return np.load(p, allow_pickle=False) if p.exists() else None


def run(scens, only=None, log=print):
    disc = discover()
    for scen in scens:
        arms = complete(disc, scen)
        if "implemented_now" not in arms or "network_float" not in arms:
            log(f"{scen}: priority arms not complete yet, skipped")
            continue
        T = Truth(scen)
        W = weights(T)
        P = partitions(T)
        (CACHE / scen).mkdir(parents=True, exist_ok=True)
        meta = {"n_tracks": T.n, "n_events": T.n_events, "n_per_file": T.n_per_file,
                "origin_counts": np.bincount(T.origin, minlength=4).tolist(),
                "cells": {k: {"names": v[1], "counts": [int((v[0] == g).sum()) for g in range(len(v[1]))]} for k, v in P.items()},
                "prompt_12_counts": [[int((T.prompt & (T.ieta == i) & (T.ipt == j)).sum()) for j in range(4)] for i in range(3)],
                "event_keys_head": T.event_keys[:3].tolist()}
        bits_lab, bits_names, bits_counts = pooled_cells(T, 1000)
        meta["cells"]["twelve_pooled_to_1000"] = {"names": bits_names, "counts": bits_counts}
        # cap: share of crossing clusters not sent, per surface, prompt tracks (the same in every capped arm)
        cap = {}
        for s in (1, 2, 3, 4, 11, 12, 13, 14):
            m = (T.slot_surface == s) & (T.crossing_cluster >= 0) & T.prompt[:, None]
            cap[str(s)] = [float((m & ~T.crossing_cluster_sent_under_cap).sum() / max(m.sum(), 1)), int(m.sum())]
        meta["correct_cluster_dropped_by_cap_per_surface"] = cap
        json.dump(meta, open(CACHE / scen / "_meta.json", "w"), indent=1)
        # anchors first: their accepted clusters and forced truncation are used by every other arm
        ref_clusters, forced_trunc, anchors = {}, None, ("implemented_now", "network_float", "alloc_4444")
        for an in anchors:
            if an in arms:
                ref_clusters[an] = load_arm(disc[scen][an], an, ("hit_cluster",))["hit_cluster"]
        nf = load_arm(disc[scen]["network_float"], "network_float", ("forced_a",))["forced_a"]
        forced_trunc = {}
        for q, j, tv in (("d0", 4, T.tp_d0), ("z0", 3, T.tp_z0)):
            rr = np.abs(nf[:, j] - tv) * CM2UM
            forced_trunc[q] = [5.0 * float(np.quantile(rr[T.prompt & (T.ieta == i)], 0.68)) for i in range(3)]
        del nf
        todo = [OT_ARM] + arms
        if only:
            todo = [a for a in todo if a in only]
        for arm in todo:
            sig = "ot-v%d" % VERSION if arm == OT_ARM else signature(disc[scen][arm])
            c = load_cache(scen, arm)
            if c is not None and str(c["signature"]) == sig:
                continue
            t0 = time.time()
            A = None if arm == OT_ARM else load_arm(disc[scen][arm], arm)
            if A is not None and len(A["refit"]) != T.n:
                log(f"{scen} {arm}: track count differs from the truth file, skipped")
                continue
            out = summarize(arm, A, T, W, P, ref_clusters, forced_trunc)
            fv = out.pop("_forced_var", None)
            if fv is not None and (arm in ("network_float", "bending_float") or arm.startswith("forecast_")):
                out["forced_var"] = fv.astype(np.float32)
            out["signature"] = np.array(sig)
            out["forced_trunc"] = np.array([forced_trunc["d0"], forced_trunc["z0"]])
            tmp = cache_path(scen, arm).with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **out)
            tmp.rename(cache_path(scen, arm))
            log(f"{scen} {arm}: {time.time() - t0:.1f} s")
            del A, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        d = discover()
        for s in SCENARIOS:
            print(s, len(complete(d, s)), "arms complete of", len(registry()))
        sys.exit(0)
    run(a.scen, a.arms)
