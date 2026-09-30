"""Sub-2 GeV SmartPixels refit study: Q1 (what the tracks are), Q2 (z0 resolution),
Q3 (GTT fastHisto PV in an offline float emulator, V0 vs V3) + the emulator closure.

  pixi run python eval_refitq/sub2gev/build_cache.py [--input-set S]    # once (per-file numpy caches)
  pixi run python eval_refitq/sub2gev/analyze.py [--input-set S] [--outdir D] [--compare-set S2]
      [--fh-nbins 256 --fh-window 3]                                       # figures/tables/JSON under D

Input sets: canon_v1 (barrel-only refit; default), canon_v2 (refit with the discs).
--compare-set: the same events of another input set (same L1TTrack rows, checked) enter as the
paired reference variants V1ref/V3ref (e.g. canon_v2 vs canon_v1 on identical events).
--fh-nbins/--fh-window: the Q3 fastHisto binning (the closure against CMSSW always uses 256/3).
|eta| breakdown (track level): |eta| < 1.1, 1.1-1.8, 1.8-2.4 of the track (Q1/Q2: the OT track's
eta, so the same track sits in the same bin for OT and refit; Q3: the variant's own track eta).

Samples: headline = f03..f10 (800 ev; f01/f02 are in-sample for the cluster CNN), check = f01..f10.
Definitions (all stated again on the table pages):
  refitted        L1TSmartPixelsTrack_spixRefitPerformed (else passthrough copy of the OT track)
  moved below 2   refitted, OT pT >= 2 GeV, refit pT < 2 GeV
  saturated       refit 15-bit 1/R word hwRinv = 0x3FFF or 0x4000 (outside the track-word pT range)
  V0              OT tracks (L1TTrack), standard GTT prompt selection (pT >= 2 GeV, ...)
  V3              SmartPixels tracks, standard selection, except refitted tracks have NO pT floor
  V1 (diagnostic) SmartPixels tracks, standard selection (the CMSSW wiring); used for the closure
                  and to split V3 - V0 into (refit of >= 2 GeV tracks) + (admitted sub-2 GeV tracks)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import (FH_NBINS, FH_WINDOW, FILES_ALL, FILES_CLEAN, PV_WINDOW, RINV_SAT, bootstrap,  # noqa: E402
                    core_rms, fasthisto, full_rms, load_cache, standard_selection, tva_dzmax)

N_BOOT = 1000
VLABEL = {"V0": "V0 OT tracks", "V3": "V3 SP tracks, sub-2 GeV refit tracks admitted",
          "V1": "V1 SP tracks, 2 GeV floor (diagnostic)",
          "V3m": "V3m SP tracks, only the moved-below-2 tracks admitted (diagnostic)"}
VARIANTS = tuple(VLABEL)
DIFFS = (("V3", "V0"), ("V1", "V0"), ("V3", "V1"), ("V3m", "V1"))
REF_DIFFS = (("V3", "V3ref"), ("V1", "V1ref"))  # with --compare-set: same events, the other input set
UM = 1e4  # cm -> um
ETA_BINS = ((0.0, 1.1), (1.1, 1.8), (1.8, 2.4))
ETA_LABEL = tuple(f"|eta| {lo:g}-{hi:g}" for lo, hi in ETA_BINS)
REF_LABEL = ""  # set in main() to the compare set's name
# Okabe-Ito (CVD-safe), fixed order per entity
C = {"V0": "#0072B2", "V3": "#D55E00", "V1": "#009E73", "V3m": "#CC79A7", "OT": "#0072B2", "SP": "#D55E00",
     "G": "#009E73", "GW": "#56B4E9", "W": "#D55E00", "R": "#E69F00", "U": "#999999", "UO": "#CC79A7",
     "V3ref": "#E69F00", "V1ref": "#56B4E9"}
CLASS_LABEL = {
    "G": "genuine low-pT: true pT < 2, all IT hits correct (OT overestimated)",
    "GW": "true pT < 2 but >= 1 wrong IT hit",
    "W": "pulled down by wrong IT hits: true pT >= 2, >= 1 wrong IT hit",
    "R": "true pT >= 2, all IT hits correct (refit pT mismeasured)",
    "UO": "SP-unmatched, OT track truth-matched",
    "U": "fake: SP- and OT-unmatched",
}


# ----------------------------------------------------------------------------- helpers
def table_page(pdf, title, header, rows, notes=()):
    notes = [ln for s in notes for ln in textwrap.wrap(s, 190)]
    n = len(rows)
    fig = plt.figure(figsize=(11, 1.6 + 0.3 * n + 0.22 * len(notes)))
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    ax = fig.add_axes([0.01, 0.02 + 0.22 * len(notes) / fig.get_figheight(), 0.98,
                       0.9 - (0.22 * len(notes) + 0.4) / fig.get_figheight()])
    ax.axis("off")
    t = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=header, loc="upper center",
                 cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(list(range(len(header))))
    for i, s in enumerate(notes):
        fig.text(0.02, 0.02 + 0.22 * (len(notes) - 1 - i) / fig.get_figheight(), s, fontsize=7.5)
    pdf.savefig(fig)
    plt.close(fig)


def legend_below(fig, ncol=3):
    h, lab = [], []
    for ax in fig.axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="outside lower center", ncol=ncol, fontsize=8, frameon=False)


def pm(v, s, fmt="{:.3f}"):
    return (fmt + " +- " + fmt).format(v, s)


def events_slices(evidx, nev):
    return np.searchsorted(evidx, np.arange(nev + 1))


def eta_bin(eta):
    """Index into ETA_BINS of |eta| (-1 outside |eta| < 2.4)."""
    a = np.abs(np.asarray(eta, float))
    i = np.searchsorted([b[0] for b in ETA_BINS[1:]], a, side="right")
    return np.where(a < ETA_BINS[-1][1], i, -1)


def popcount(x):
    x = np.asarray(x, np.int64)
    return sum((x >> b) & 1 for b in range(8))


def n_barrel_disc(T, pre="sp_"):
    """Accepted refit hits on IL1-4 (spixLayerHitMask bits 0-3) and ID1-4 (bits 4-7)."""
    m = T[pre + "spixLayerHitMask"].astype(np.int64)
    return popcount(m & 0x0F), popcount(m & 0xF0)


# ----------------------------------------------------------------------------- Q1
def classify(T):
    """Truth class of every row (meaningful for refitted rows); SP truth = weighted majority."""
    sp_m = T["sp_tpIdx"] >= 0
    pruned = T["sp_tpIdx"] <= -3  # matched to a TP not written to L1TTP (|p| <= 1 GeV)
    ptt = np.where(pruned, 0.5, T["sptp_pt"])  # placeholder < 2 for pruned (|p| <= 1 GeV)
    nwrong = T["sp_spixNAcceptedHits"] - T["sp_tpNITClusters"]
    low = ptt < 2.0
    cls = np.full(T["sp_pt"].size, "U", dtype="<U2")
    matched = sp_m | pruned
    cls[matched & low & (nwrong == 0)] = "G"
    cls[matched & low & (nwrong > 0)] = "GW"
    cls[matched & ~low & (nwrong > 0)] = "W"
    cls[matched & ~low & (nwrong == 0)] = "R"
    cls[~matched & (T["ot_tpIdx"] >= 0)] = "UO"
    return cls, nwrong, ptt


def q1(T, E, tag, pdf, summ):
    nev = E["ev_event"].size
    ref = T["sp_spixRefitPerformed"] != 0
    moved = ref & (T["ot_pt"] >= 2.0) & (T["sp_pt"] < 2.0)
    sat = ref & np.isin(T["sp_hwRinv"], RINV_SAT)
    sat_neg = ref & (T["sp_hwRinv"] == 0x4000)
    sp_lt2 = T["sp_pt"] < 2.0
    moved_up = ref & (T["ot_pt"] < 2.0) & (T["sp_pt"] >= 2.0)
    cls, nwrong, ptt = classify(T)
    T["cls"] = cls

    def per_ev(m):
        c = np.bincount(T["evidx"][m], minlength=nev)
        return c.mean(), c.std(), c

    rows = []
    for name, m in (("all L1TTrack (= L1TSmartPixelsTrack rows)", np.ones_like(ref)),
                    ("refitted", ref),
                    ("OT pT < 2 GeV (1.95-2.0; outside V0 anyway)", T["ot_pt"] < 2.0),
                    ("SP pT < 2 GeV (any)", sp_lt2),
                    ("moved below 2: refitted, OT pT >= 2, refit pT < 2", moved),
                    ("  of which refit pT < 1.9 GeV", moved & (T["sp_pt"] < 1.9)),
                    ("saturated 1/R word (0x3FFF or 0x4000), all", sat),
                    ("  of which 0x4000 (negative; the out-of-range LUT case)", sat_neg),
                    ("  of which also moved below 2", sat & moved),
                    ("moved up: refitted, OT pT < 2, refit pT >= 2", moved_up)):
        mu, sd, _ = per_ev(m)
        rows.append([name, int(m.sum()), f"{mu:.2f}", f"{sd:.2f}"])
    table_page(pdf, f"Q1 [{tag}] track counts ({nev} PU200 ttbar events; {common.INPUT_LABEL})",
               ["population", "N tracks", "mean / event", "std / event"], rows,
               notes=["'saturated' = the refit 1/R word is at its 15-bit limit (pT below ~1.9 GeV); "
                      "0x4000 is the value that indexed past the stock GTT pT LUT (bug fixed by the guard 549aaec59fa)."])
    summ["counts"] = {r[0].strip(): {"N": r[1], "per_event": float(r[2])} for r in rows}
    sat_pt = T["sp_pt"][sat]
    summ["saturated_sp_pt_range"] = [float(sat_pt.min()), float(sat_pt.max())] if sat_pt.size else None
    summ["moved_lt19_vs_saturated"] = {"moved_and_pt_lt_1p9": int((moved & (T["sp_pt"] < 1.9)).sum()),
                                       "moved_and_saturated": int((moved & sat).sum())}

    # classification table
    def class_rows(m):
        out = []
        n = m.sum()
        for k in ("G", "GW", "W", "R", "UO", "U"):
            mk = m & (cls == k)
            hs = mk & (T["sptp_evt"] == 0) & (T["sptp_bx"] == 0)
            out.append([k, CLASS_LABEL[k], int(mk.sum()), f"{100 * mk.sum() / max(n, 1):.1f}",
                        int(hs.sum()), f"{np.median(T['sp_pt'][mk]):.2f}" if mk.any() else "-",
                        f"{np.median(T['ot_pt'][mk]):.2f}" if mk.any() else "-",
                        f"{np.nanmedian(ptt[mk]):.2f}" if mk.any() and np.isfinite(ptt[mk]).any() else "-"])
        return out

    hdr = ["class", "definition", "N", "%", "N hard-scatter", "med refit pT", "med OT pT", "med true pT"]
    notes = ["Truth = SmartPixels weighted cluster-majority TP (L1TSmartPixelsTrack_tpIdx). 'wrong IT hit' = accepted IT "
             "cluster not owned by that TP (spixNAcceptedHits - tpNITClusters > 0).",
             "Hard-scatter = TP with eventId (evt, bx) = (0, 0). pT in GeV."]
    table_page(pdf, f"Q1 [{tag}] classification of the tracks moved below 2 GeV (N = {moved.sum()})", hdr,
               class_rows(moved), notes)
    table_page(pdf, f"Q1 [{tag}] classification of the saturated-word refit tracks (N = {sat.sum()})", hdr,
               class_rows(sat), notes)
    summ["class_moved"] = {r[0]: {"N": r[2], "pct": float(r[3]), "N_HS": r[4]} for r in class_rows(moved)}
    summ["class_saturated"] = {r[0]: {"N": r[2], "pct": float(r[3]), "N_HS": r[4]} for r in class_rows(sat)}

    # truth agreement / cross-check with the OT truth and the selHitClass counting
    m = moved
    both = m & (T["ot_tpIdx"] >= 0) & (T["sp_tpIdx"] >= 0)
    same = both & (T["ot_tpIdx"] == T["sp_tpIdx"])
    wrong_sel = T["hit_nacc_other"] + T["hit_nacc_noise"]
    rows = [["moved tracks", int(m.sum())],
            ["OT truth-matched (loosely genuine, L1TTrack_tpIdx >= 0)", int((m & (T["ot_tpIdx"] >= 0)).sum())],
            ["SP truth-matched (weighted majority)", int((m & (T["sp_tpIdx"] >= 0)).sum())],
            ["both matched, same TP", int(same.sum())],
            ["both matched, different TP", int((both & ~same).sum())],
            ["OT truth pT < 2 GeV (OT-matched)", int((m & (T["ottp_pt"] < 2)).sum())],
            [">= 1 wrong IT hit (SP-truth counting)", int((m & (nwrong > 0)).sum())],
            [">= 1 wrong IT hit (selHitClass 1/2, relative to the OT TP)", int((m & (wrong_sel > 0)).sum())],
            ["mean accepted IT hits", f"{T['sp_spixNAcceptedHits'][m].mean():.2f}"],
            ["mean wrong IT hits (SP truth)", f"{nwrong[m].mean():.2f}"]]
    # how much of the OT-overestimated population does the refit correct?
    over = ref & (T["ot_pt"] >= 2.0) & (T["sp_tpIdx"] >= 0) & (T["sptp_pt"] < 2.0)
    rows += [["[context] refitted, OT pT >= 2, SP-truth pT < 2 (OT overestimated)", int(over.sum())],
             ["[context]   of which moved below 2 by the refit", int((over & moved).sum())],
             ["[context] refitted, OT pT >= 2, SP-truth pT >= 2, moved below 2 (lost)",
              int((ref & (T['ot_pt'] >= 2) & (T['sptp_pt'] >= 2) & moved).sum())],
             ["[context] refitted, OT pT >= 2, SP-truth pT >= 2 (denominator)",
              int((ref & (T['ot_pt'] >= 2) & (T['sptp_pt'] >= 2)).sum())]]
    table_page(pdf, f"Q1 [{tag}] truth cross-checks for the moved tracks", ["quantity", "value"], rows,
               notes=["L1TTrack_tpIdx = stock loosely-genuine OT match; L1TSmartPixelsTrack_tpIdx = weighted majority over "
                      "OT stub clusters (1/2 each) and accepted IT clusters (1 each)."])
    summ["moved_crosschecks"] = {r[0]: r[1] for r in rows}

    # figures
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    _, _, cm = per_ev(moved)
    _, _, cs = per_ev(sat)
    b = np.arange(0, max(cm.max(), 1) + 2) - 0.5
    axs[0].hist(cm, bins=b, histtype="step", lw=2, color=C["SP"], label="moved below 2 GeV (refit)")
    axs[0].hist(cs, bins=b, histtype="step", lw=2, color=C["UO"], label="saturated 1/R word")
    axs[0].set_xlabel("tracks per event")
    axs[0].set_ylabel("events")
    bins = np.linspace(1.3, 3.0, 69)
    axs[1].hist(T["ot_pt"][moved], bins=bins, histtype="step", lw=2, color=C["OT"], label="OT pT (moved tracks)")
    axs[1].hist(T["sp_pt"][moved], bins=bins, histtype="step", lw=2, color=C["SP"], label="refit pT (moved tracks)")
    mt = moved & np.isfinite(T["sptp_pt"])
    axs[1].hist(T["sptp_pt"][mt], bins=bins, histtype="step", lw=2, color="k", ls="--",
                label="true pT (SP truth, moved tracks)")
    axs[1].set_xlabel("pT [GeV]")
    axs[1].set_ylabel("tracks / 0.025 GeV")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Q1 [{tag}] tracks the refit moves below 2 GeV")
    legend_below(fig, ncol=3)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for k in ("R", "W", "G", "GW", "UO", "U"):
        mk = moved & (cls == k)
        tp = np.where(np.isfinite(ptt[mk]), ptt[mk], np.nan)
        axs[0].scatter(T["ot_pt"][mk], T["sp_pt"][mk], s=6, color=C[k], label=f"{k}: {CLASS_LABEL[k]}")
        axs[1].scatter(tp, T["sp_pt"][mk], s=6, color=C[k])
        axs[2].scatter(tp, T["ot_pt"][mk], s=6, color=C[k])
    axs[0].set(xlabel="OT pT [GeV]", ylabel="refit pT [GeV]", xlim=(1.95, 4), ylim=(1.3, 2.05))
    axs[1].set(xlabel="true pT (SP truth) [GeV]", ylabel="refit pT [GeV]", xlim=(1.5, 4), ylim=(1.3, 2.05))
    axs[2].set(xlabel="true pT (SP truth) [GeV]", ylabel="OT pT [GeV]", xlim=(1.5, 4), ylim=(1.95, 4))
    for ax in axs[1:]:
        ax.axvline(2.0, color="k", lw=0.8, ls=":")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Q1 [{tag}] moved tracks by truth class (unmatched classes have no true pT)")
    legend_below(fig, ncol=2)
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    mm = moved & np.isfinite(T["sptp_pt"])
    b = np.linspace(-0.6, 0.4, 101)
    ax.hist((T["ot_pt"][mm] - T["sptp_pt"][mm]) / T["sptp_pt"][mm], bins=b, histtype="step", lw=2,
            color=C["OT"], label="(OT pT - true) / true")
    ax.hist((T["sp_pt"][mm] - T["sptp_pt"][mm]) / T["sptp_pt"][mm], bins=b, histtype="step", lw=2,
            color=C["SP"], label="(refit pT - true) / true")
    ax.set(xlabel="relative pT residual", ylabel="tracks", title=f"Q1 [{tag}] moved, SP-truth-matched tracks")
    ax.grid(alpha=0.25)
    legend_below(fig, ncol=2)
    pdf.savefig(fig)
    plt.close(fig)
    return moved, sat


def q1_eta(T, E, tag, pdf, summ, moved, sat):
    """Track-level Q1 quantities per |eta| bin of the OT track."""
    nev = E["ev_event"].size
    ref = T["sp_spixRefitPerformed"] != 0
    nb, nd = n_barrel_disc(T)
    nwrong = T["sp_spixNAcceptedHits"] - T["sp_tpNITClusters"]
    eb = eta_bin(T["ot_eta"])
    both = (T["ot_tpIdx"] >= 0) & (T["sp_tpIdx"] >= 0)
    rows, out = [], {}
    for b, lab in enumerate(ETA_LABEL):
        m = eb == b
        r = m & ref
        sm = r & (T["sp_tpIdx"] >= 0)
        row = {"tracks / ev": m.sum() / nev, "refitted %": 100 * r.sum() / max(m.sum(), 1),
               ">=1 disc hit % (refitted)": 100 * np.mean(nd[r] > 0) if r.any() else np.nan,
               "mean barrel hits (refitted)": nb[r].mean() if r.any() else np.nan,
               "mean disc hits (refitted)": nd[r].mean() if r.any() else np.nan,
               "SP truth-matched % (refitted)": 100 * sm.sum() / max(r.sum(), 1),
               ">=1 wrong IT hit % (refitted, SP-matched)": 100 * np.mean(nwrong[sm] > 0) if sm.any() else np.nan,
               "SP TP != OT TP (both matched), N": int((r & both & (T["ot_tpIdx"] != T["sp_tpIdx"])).sum()),
               "SP-unmatched but OT-matched, N": int((r & (T["sp_tpIdx"] < 0) & (T["ot_tpIdx"] >= 0)).sum()),
               "moved below 2 / ev": (m & moved).sum() / nev, "saturated 1/R word / ev": (m & sat).sum() / nev}
        out[lab] = {k: float(v) for k, v in row.items()}
        rows.append([lab] + [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row.values()])
    hdr = ["OT track |eta|"] + list(next(iter(out.values())))
    table_page(pdf, f"Q1 [{tag}] track census per |eta| bin (OT track eta; {common.INPUT_LABEL})",
               hdr, [[r[0]] + r[1:] for r in rows],
               notes=["disc hit = accepted refit hit on ID1-ID4 (spixLayerHitMask bits 4-7); barrel = IL1-IL4. "
                      "wrong IT hit = accepted IT cluster not owned by the SP-truth TP. SP TP != OT TP = the refit's "
                      "cluster-majority TP differs from the OT loosely-genuine TP."])
    summ["q1_eta"] = out


# ----------------------------------------------------------------------------- Q2
def q2(T, tag, pdf, summ, moved, sat):
    cls = T["cls"]
    rows = []
    res_store = {}
    for name, m in (("moved below 2 (all SP-matched)", moved), ("saturated word (SP-matched)", sat)) + tuple(
            (f"moved, class {k}", moved & (cls == k)) for k in ("G", "GW", "W", "R")):
        mm = m & (T["sp_tpIdx"] >= 0)
        rot = (T["ot_z0"][mm] - T["sptp_z0"][mm]) * UM
        rsp = (T["sp_z0"][mm] - T["sptp_z0"][mm]) * UM
        res_store[name] = (rot, rsp)
        co, fo = core_rms(rot)
        cs, fs = core_rms(rsp)
        rows.append([name, int(mm.sum()), f"{co:.0f}", f"{100 * fo:.1f}", f"{full_rms(rot):.0f}",
                     f"{cs:.0f}", f"{100 * fs:.1f}", f"{full_rms(rsp):.0f}"])
    # OT-truth cross-check for the moved set (OT z0 vs its own loosely-genuine TP)
    mm = moved & (T["ot_tpIdx"] >= 0)
    rot = (T["ot_z0"][mm] - T["ottp_z0"][mm]) * UM
    rsp = (T["sp_z0"][mm] - T["ottp_z0"][mm]) * UM
    co, fo = core_rms(rot)
    cs, fs = core_rms(rsp)
    rows.append(["moved, OT-truth-matched (truth = OT TP)", int(mm.sum()), f"{co:.0f}", f"{100 * fo:.1f}",
                 f"{full_rms(rot):.0f}", f"{cs:.0f}", f"{100 * fs:.1f}", f"{full_rms(rsp):.0f}"])
    hdr = ["population", "N", "OT core RMS", "OT % out", "OT full RMS", "refit core RMS", "refit % out",
           "refit full RMS"]
    table_page(pdf, f"Q2 [{tag}] z0 residual (reco - true TP z0) of the sub-2 GeV refit tracks [um]", hdr, rows,
               notes=["core RMS = RMS about zero of the iterative 3-sigma-clipped core (bias counts); % out = fraction "
                      "outside the core; full RMS about zero. Truth = SP TP unless stated."])
    summ["z0_sub2"] = {r[0]: dict(zip(hdr[1:], r[1:])) for r in rows}

    # resolution vs true pT, all tracks
    edges = np.array([1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0, 50.0, 200.0])
    ref = T["sp_spixRefitPerformed"] != 0
    pops = {
        "OT, all OT-matched tracks": (T["ot_tpIdx"] >= 0, "ot_z0", "ottp_"),
        "OT, refitted rows (OT truth)": (ref & (T["ot_tpIdx"] >= 0), "ot_z0", "ottp_"),
        "refit, refitted rows (SP truth)": (ref & (T["sp_tpIdx"] >= 0), "sp_z0", "sptp_"),
        "SP passthrough rows (= OT, SP truth)": (~ref & (T["sp_tpIdx"] >= 0), "sp_z0", "sptp_"),
    }
    curves = {}
    rows = []
    for name, (m, zc, tp) in pops.items():
        vals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mb = m & (T[tp + "pt"] >= lo) & (T[tp + "pt"] < hi)
            r = (T[zc][mb] - T[tp + "z0"][mb]) * UM
            c, f = core_rms(r) if r.size >= 20 else (np.nan, np.nan)
            vals.append((c, f, full_rms(r) if r.size >= 20 else np.nan, r.size))
            rows.append([name, f"{lo:g}-{hi:g}", r.size, f"{c:.0f}", f"{100 * f:.1f}", f"{vals[-1][2]:.0f}"])
        curves[name] = np.array(vals)
    centers = np.sqrt(edges[:-1] * edges[1:])
    colors = {"OT, all OT-matched tracks": C["OT"], "OT, refitted rows (OT truth)": "#56B4E9",
              "refit, refitted rows (SP truth)": C["SP"], "SP passthrough rows (= OT, SP truth)": "#999999"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, v in curves.items():
        axs[0].plot(centers, v[:, 0], marker="o", color=colors[name], lw=2, label=name)
        axs[1].plot(centers, v[:, 2], marker="o", color=colors[name], lw=2)
    for ax, yl in zip(axs, ("z0 core RMS [um]", "z0 full RMS [um]")):
        ax.set(xscale="log", yscale="log", xlabel="true pT [GeV]", ylabel=yl)
        ax.axvline(2.0, color="k", lw=0.8, ls=":")
        ax.grid(alpha=0.25, which="both")
    fig.suptitle(f"Q2 [{tag}] z0 resolution vs true pT (all tracks; dotted: 2 GeV)")
    legend_below(fig, ncol=2)
    pdf.savefig(fig)
    plt.close(fig)
    table_page(pdf, f"Q2 [{tag}] z0 resolution vs true pT [um]",
               ["population", "true pT bin [GeV]", "N", "core RMS", "% out", "full RMS"], rows,
               notes=["Bins with < 20 tracks are left empty. Passthrough rows are OT copies (refit not performed)."])
    summ["z0_vs_pt"] = {n: {"edges": edges.tolist(), "core": v[:, 0].tolist(), "out_frac": v[:, 1].tolist(),
                            "full": v[:, 2].tolist(), "N": v[:, 3].tolist()} for n, v in curves.items()}

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    rot, rsp = res_store["moved below 2 (all SP-matched)"]
    b = np.linspace(-5000, 5000, 101)
    for ax, (lab, r, c) in zip(axs, (("OT z0 - true", rot, C["OT"]), ("refit z0 - true", rsp, C["SP"]))):
        ax.hist(np.clip(r, b[0], b[-1]), bins=b, histtype="step", lw=2, color=c, label=lab)
        ax.set(xlabel="z0 residual [um] (overflow in edge bins)", ylabel="tracks", yscale="log")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Q2 [{tag}] z0 residuals of the tracks moved below 2 GeV (SP truth)")
    legend_below(fig, ncol=2)
    pdf.savefig(fig)
    plt.close(fig)


def q2_eta(T, tag, pdf, summ):
    """z0 core/full RMS vs true pT per |eta| bin of the OT track: OT (OT truth) vs refit (SP truth)
    on refitted rows (+ the compare set's refit on the same rows)."""
    ref = T["sp_spixRefitPerformed"] != 0
    edges = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 200.0])
    eb = eta_bin(T["ot_eta"])
    pops = {"OT (refitted rows, OT truth)": (ref & (T["ot_tpIdx"] >= 0), "ot_z0", "ottp_", C["OT"]),
            "refit (refitted rows, SP truth)": (ref & (T["sp_tpIdx"] >= 0), "sp_z0", "sptp_", C["SP"])}
    if "csp_z0" in T:
        cref = T["csp_spixRefitPerformed"] != 0
        pops[f"refit {REF_LABEL} (same rows, its SP truth)"] = (cref & (T["csp_tpIdx"] >= 0), "csp_z0", "csptp_",
                                                                 C["V3ref"])
    rows, out = [], {}
    fig, axs = plt.subplots(1, len(ETA_BINS), figsize=(15, 4.8), layout="constrained", sharey=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for b, (ax, lab) in enumerate(zip(axs, ETA_LABEL)):
        for name, (m, zc, tp, col) in pops.items():
            vals = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mb = m & (eb == b) & (T[tp + "pt"] >= lo) & (T[tp + "pt"] < hi)
                r = (T[zc][mb] - T[tp + "z0"][mb]) * UM
                c, f = core_rms(r) if r.size >= 20 else (np.nan, np.nan)
                fr = full_rms(r) if r.size >= 20 else np.nan
                vals.append((c, f, fr, r.size))
                rows.append([lab, name, f"{lo:g}-{hi:g}", r.size, f"{c:.0f}", f"{100 * f:.1f}", f"{fr:.0f}"])
            v = np.array(vals)
            out[f"{lab} | {name}"] = {"edges": edges.tolist(), "core": v[:, 0].tolist(), "out_frac": v[:, 1].tolist(),
                                      "full": v[:, 2].tolist(), "N": v[:, 3].tolist()}
            ax.plot(centers, v[:, 0], marker="o", color=col, lw=2, label=name)
        ax.set(xscale="log", yscale="log", xlabel="true pT [GeV]", title=f"OT track {lab}")
        ax.grid(alpha=0.25, which="both")
    axs[0].set_ylabel("z0 core RMS [um]")
    fig.suptitle(f"Q2 [{tag}] z0 core resolution vs true pT per |eta| bin ({common.INPUT_LABEL})")
    legend_below(fig, ncol=3)
    pdf.savefig(fig)
    plt.close(fig)
    table_page(pdf, f"Q2 [{tag}] z0 resolution per |eta| bin and true pT [um]",
               ["OT track |eta|", "population", "true pT [GeV]", "N", "core RMS", "% out", "full RMS"], rows,
               notes=["Refitted rows only (the refit changed the track). Bins with < 20 tracks are empty. core RMS = "
                      "RMS about zero of the iterative 3-sigma-clipped core (clip about the mean)."])
    summ["z0_eta"] = out



# ----------------------------------------------------------------------------- Q3
def selections(T):
    ref = T["sp_spixRefitPerformed"] != 0
    common = (T["ot_nStubs"], T["ot_chi2BendRed"], T["ot_chi2ZRed"], T["ot_chi2XYRed"])  # SP rows carry the OT chi2s
    s0 = standard_selection(T["ot_pt"], T["ot_eta"], T["ot_z0"], *common)
    s1 = standard_selection(T["sp_pt"], T["sp_eta"], T["sp_z0"], *common)
    s3 = standard_selection(T["sp_pt"], T["sp_eta"], T["sp_z0"], *common, pt_min=np.where(ref, 0.0, 2.0))
    s3m = standard_selection(T["sp_pt"], T["sp_eta"], T["sp_z0"], *common,
                             pt_min=np.where(ref & (T["ot_pt"] >= 2.0), 0.0, 2.0))
    out = {"V0": (s0, "ot_", "ottp_"), "V1": (s1, "sp_", "sptp_"), "V3": (s3, "sp_", "sptp_"),
           "V3m": (s3m, "sp_", "sptp_")}
    if "csp_pt" in T:  # --compare-set: the other input set's SP tracks on the same rows
        cref = T["csp_spixRefitPerformed"] != 0
        out["V1ref"] = (standard_selection(T["csp_pt"], T["csp_eta"], T["csp_z0"], *common), "csp_", "csptp_")
        out["V3ref"] = (standard_selection(T["csp_pt"], T["csp_eta"], T["csp_z0"], *common,
                                           pt_min=np.where(cref, 0.0, 2.0)), "csp_", "csptp_")
    return out


def run_pv(T, nev, sel, pre, quantised=False, nbins=FH_NBINS, window=FH_WINDOW):
    sl = events_slices(T["evidx"], nev)
    z = np.full(nev, np.nan)
    for i in range(nev):
        s = slice(sl[i], sl[i + 1])
        m = sel[s]
        z[i] = fasthisto(T[pre + "z0"][s][m], T[pre + "pt"][s][m], quantised=quantised, nbins=nbins,
                         window=window)[0]
    return z


def closure(T, E, tag, pdf, summ):
    nev = E["ev_event"].size
    S = selections(T)
    rows, out = [], {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, (v, ref, lab) in zip(axs, (("V0", "ev_l1v_ot", "V0 (OT tracks) vs CMSSW L1Vertex, OT-fed (coexist nano)"),
                                       ("V1", "ev_l1v_sp", "V1 (SP tracks, 2 GeV floor) vs CMSSW L1Vertex (canonical nano)"))):
        sel, pre, _ = S[v]
        for q in (False, True):
            z = run_pv(T, nev, sel, pre, quantised=q)
            d = z - E[ref]
            g = E["ev_genz"]
            r = [f"{v} {'quantised-approx' if q else 'float'}", nev, f"{100 * np.mean(np.abs(d) < 0.01):.1f}",
                 f"{100 * np.mean(np.abs(d) < 0.08):.1f}", f"{100 * np.mean(np.abs(d) > 1.0):.1f}",
                 f"{np.median(np.abs(d)) * UM:.0f}", f"{100 * np.mean(np.abs(z - g) < PV_WINDOW):.1f}",
                 f"{100 * np.mean(np.abs(E[ref] - g) < PV_WINDOW):.1f}"]
            rows.append(r)
            out[r[0]] = r[1:]
            if not q:
                b = np.linspace(-0.2, 0.2, 81)
                ax.hist(np.clip(d, b[0], b[-1]), bins=b, histtype="step", lw=2, color=C[v],
                        label=f"{v} float emulator - CMSSW")
                ax.set(xlabel="z_PV(float emulator) - z_PV(CMSSW) [cm] (overflow in edge bins)", ylabel="events",
                       yscale="log", title=lab)
                ax.title.set_fontsize(9)
                ax.grid(alpha=0.25)
    fig.suptitle(f"Closure [{tag}]: offline float fastHisto vs CMSSW fastHistoEmulation")
    legend_below(fig, ncol=2)
    pdf.savefig(fig)
    plt.close(fig)
    hdr = ["emulator", "events", "% |dz|<0.01 cm", "% |dz|<0.08 cm (half bin)", "% |dz|>1 cm (other vertex)",
           "median |dz| [um]", f"eff |z-zgen|<{PV_WINDOW} emu %", "eff CMSSW %"]
    table_page(pdf, f"Closure [{tag}] float emulator vs CMSSW L1Vertex_z0 (same events, same tracks)", hdr, rows,
               notes=["The canonical nano's GTT is fed by the SmartPixels tracks (V1 wiring); the OT-only CMSSW vertex comes "
                      "from the coexist nano of the same events (identical L1TTrack).",
                      "quantised-approx: pT floored to 1/32 GeV, bin sums floored to integers and saturated at 511, z0 on "
                      "the word LSB; the GTT pT LUT, word-level cuts and the 1/x LUT are NOT modelled."])
    summ["closure"] = {k: dict(zip(hdr[1:], v)) for k, v in out.items()}


def q3(T, E, tag, pdf, summ, fh=(FH_NBINS, FH_WINDOW)):
    nev = E["ev_event"].size
    S = selections(T)
    variants = tuple(S)
    diffs = DIFFS + tuple(d for d in REF_DIFFS if d[1] in S)
    g = E["ev_genz"]
    Z = {v: run_pv(T, nev, s, pre, nbins=fh[0], window=fh[1]) for v, (s, pre, _) in S.items()}
    fh_txt = f"{fh[0]} bins of {2 * 20.46912512 / fh[0]:.4f} cm, {fh[1]}-bin window"
    ev = T["evidx"]
    admitted = S["V3"][0] & ~S["V1"][0]  # the sub-2 GeV refit tracks V3 adds

    # per-event association counts, own truth and (cross-check) OT-row truth for all variants
    per = {}
    for v, (s, pre, tp) in S.items():
        dz = np.abs(T[pre + "z0"] - Z[v][ev])
        assoc = s & (dz < tva_dzmax(T[pre + "eta"]))
        eb = eta_bin(T[pre + "eta"])
        for tl, tpre in (("own", tp), ("otrow", "ottp_")):
            hs = (T[tpre + "evt"] == 0) & (T[tpre + "bx"] == 0)
            per[(v, tl)] = dict(
                n_sel=np.bincount(ev[s], minlength=nev), n_assoc=np.bincount(ev[assoc], minlength=nev),
                n_hs_sel=np.bincount(ev[s & hs], minlength=nev), n_hs_assoc=np.bincount(ev[assoc & hs], minlength=nev),
                pt_hs_assoc=np.bincount(ev[assoc & hs], weights=T[pre + "pt"][assoc & hs], minlength=nev),
                n_adm_assoc=np.bincount(ev[assoc & admitted], minlength=nev),
                n_adm_hs_assoc=np.bincount(ev[assoc & hs & admitted], minlength=nev))
            for b in range(len(ETA_BINS)):
                e = eb == b
                per[(v, tl)].update({f"n_sel_eta{b}": np.bincount(ev[s & e], minlength=nev),
                                     f"n_assoc_eta{b}": np.bincount(ev[assoc & e], minlength=nev),
                                     f"n_hs_sel_eta{b}": np.bincount(ev[s & hs & e], minlength=nev),
                                     f"n_hs_assoc_eta{b}": np.bincount(ev[assoc & hs & e], minlength=nev)})

    def stats(idx, v, tl="own"):
        d = Z[v][idx] - g[idx]
        c, fo = core_rms(d * UM)
        p = per[(v, tl)]
        o = {"eff": np.mean(np.abs(d) < PV_WINDOW), "core_um": c, "out_frac": fo, "full_um": full_rms(d * UM),
             "hs_assoc_per_ev": p["n_hs_assoc"][idx].mean(),
             "hs_assoc_eff": p["n_hs_assoc"][idx].sum() / max(p["n_hs_sel"][idx].sum(), 1),
             "assoc_purity": p["n_hs_assoc"][idx].sum() / max(p["n_assoc"][idx].sum(), 1),
             "n_sel_per_ev": p["n_sel"][idx].mean()}
        for b in range(len(ETA_BINS)):
            o[f"hs_assoc_per_ev_eta{b}"] = p[f"n_hs_assoc_eta{b}"][idx].mean()
            o[f"hs_assoc_eff_eta{b}"] = p[f"n_hs_assoc_eta{b}"][idx].sum() / max(p[f"n_hs_sel_eta{b}"][idx].sum(), 1)
            o[f"assoc_purity_eta{b}"] = p[f"n_hs_assoc_eta{b}"][idx].sum() / max(p[f"n_assoc_eta{b}"][idx].sum(), 1)
        return o

    eta_keys = [f"{k}_eta{b}" for b in range(len(ETA_BINS)) for k in ("hs_assoc_per_ev", "hs_assoc_eff", "assoc_purity")]

    def all_stats(idx):
        out = {}
        for v in variants:
            for k, x in stats(idx, v).items():
                out[f"{v}:{k}"] = x
            so = stats(idx, v, "otrow")
            for k in ["hs_assoc_eff", "assoc_purity", "hs_assoc_per_ev"] + eta_keys:
                out[f"{v}:{k}:otrow"] = so[k]
        for a, b in diffs:
            for k in ["eff", "core_um", "full_um", "hs_assoc_per_ev", "hs_assoc_eff", "assoc_purity"] + eta_keys:
                out[f"{a}-{b}:{k}"] = out[f"{a}:{k}"] - out[f"{b}:{k}"]
            for k in eta_keys:
                out[f"{a}-{b}:{k}:otrow"] = out[f"{a}:{k}:otrow"] - out[f"{b}:{k}:otrow"]
            for k in ("hs_assoc_eff", "assoc_purity", "hs_assoc_per_ev"):
                out[f"{a}-{b}:{k}:otrow"] = out[f"{a}:{k}:otrow"] - out[f"{b}:{k}:otrow"]
        return out

    B = bootstrap(all_stats, nev, n_boot=N_BOOT)
    summ["q3_bootstrap"] = {k: [float(v[0]), float(v[1])] for k, v in B.items()}

    def f(k, scale=1.0, fmt="{:.3f}"):
        return pm(B[k][0] * scale, B[k][1] * scale, fmt)

    rows = []
    for v in variants:
        rows.append([v + (" (diagnostic)" if v in ("V1", "V3m") else ""), f(f"{v}:eff", 100, "{:.1f}"), f(f"{v}:core_um", 1, "{:.0f}"),
                     f(f"{v}:out_frac", 100, "{:.1f}"), f(f"{v}:full_um", 1, "{:.0f}"),
                     f(f"{v}:hs_assoc_per_ev", 1, "{:.2f}"), f(f"{v}:hs_assoc_eff", 100, "{:.1f}"),
                     f(f"{v}:assoc_purity", 100, "{:.1f}"), f"{B[f'{v}:n_sel_per_ev'][0]:.1f}"])
    for a, b in diffs:
        rows.append([f"{a} - {b} (paired)", f(f"{a}-{b}:eff", 100, "{:.1f}"), f(f"{a}-{b}:core_um", 1, "{:.0f}"), "",
                     f(f"{a}-{b}:full_um", 1, "{:.0f}"), f(f"{a}-{b}:hs_assoc_per_ev", 1, "{:.2f}"),
                     f(f"{a}-{b}:hs_assoc_eff", 100, "{:.1f}"), f(f"{a}-{b}:assoc_purity", 100, "{:.1f}"), ""])
    hdr = ["variant", f"PV eff % (|dz|<{PV_WINDOW} cm)", "dz core RMS [um]", "% outside core", "dz full RMS [um]",
           "HS tracks assoc / ev", "HS assoc eff %", "assoc purity %", "selected trk / ev"]
    table_page(pdf, f"Q3 [{tag}] PV finding, offline float fastHisto, {fh_txt} ({nev} events, {N_BOOT} event-bootstrap; "
               f"{common.INPUT_LABEL})", hdr, rows, notes=[
                   (f"V1ref / V3ref = V1 / V3 built from {REF_LABEL} on the SAME events and OT tracks (paired); "
                    "identical V0 in both sets (checked)." if "V3ref" in S else ""),
                   "dz = z_PV - GenVtx_z. HS = hard-scatter track: its TP has (evt, bx) = (0, 0) (V0: OT truth, V1/V3: SP truth). "
                   "Association = GTT TVA eta-binned |z0 - z_PV| < 0.37-1.6 cm.",
                   "V0 = OT tracks; V3 = SP tracks with the sub-2 GeV refit tracks admitted; V1 = SP tracks with the 2 GeV "
                   "floor (the CMSSW wiring); V3m = SP tracks, only the moved-below-2 tracks admitted.",
                   "HS assoc eff = sum(HS assoc)/sum(HS selected); purity = sum(HS assoc)/sum(assoc). Errors: event bootstrap "
                   "(paired for differences). V1 is a diagnostic only (splits V3 - V0).",
                   "V3 selection = standard prompt cuts; the 2 GeV floor is removed for refitted tracks only; passthrough "
                   "tracks keep it. V3m (diagnostic): floor removed only for refitted tracks with OT pT >= 2 (the tracks the "
                   "refit moved below 2). chi2/nStub cuts use the OT values (the refit does not rewrite them)."])
    rows2 = []
    for v in variants:
        rows2.append([v, f(f"{v}:hs_assoc_per_ev:otrow", 1, "{:.2f}"), f(f"{v}:hs_assoc_eff:otrow", 100, "{:.1f}"),
                      f(f"{v}:assoc_purity:otrow", 100, "{:.1f}")])
    for a, b in diffs:
        rows2.append([f"{a} - {b}", f(f"{a}-{b}:hs_assoc_per_ev:otrow", 1, "{:.2f}"),
                      f(f"{a}-{b}:hs_assoc_eff:otrow", 100, "{:.1f}"), f(f"{a}-{b}:assoc_purity:otrow", 100, "{:.1f}")])
    table_page(pdf, f"Q3 [{tag}] association cross-check with ONE truth for all variants (the OT row's loosely-genuine TP)",
               ["variant", "HS tracks assoc / ev", "HS assoc eff %", "assoc purity %"], rows2)

    # association per |eta| bin of the variant's own track
    rows3 = []
    for b, lab in enumerate(ETA_LABEL):
        for v in [x for x in ("V0", "V3", "V1", "V3ref", "V1ref") if x in S]:
            rows3.append([lab, v] + [f(f"{v}:{k}_eta{b}{t}", sc, fm) for t in ("", ":otrow")
                                     for k, sc, fm in (("hs_assoc_per_ev", 1, "{:.2f}"), ("hs_assoc_eff", 100, "{:.1f}"),
                                                       ("assoc_purity", 100, "{:.1f}"))])
        for a, bb in (("V3", "V0"), ("V1", "V0"), ("V3", "V1")) + tuple(d for d in REF_DIFFS if d[1] in S):
            rows3.append([lab, f"{a} - {bb} (paired)"] + [f(f"{a}-{bb}:{k}_eta{b}{t}", sc, fm) for t in ("", ":otrow")
                                                          for k, sc, fm in (("hs_assoc_per_ev", 1, "{:.2f}"),
                                                                            ("hs_assoc_eff", 100, "{:.1f}"),
                                                                            ("assoc_purity", 100, "{:.1f}"))])
    table_page(pdf, f"Q3 [{tag}] track-vertex association per |eta| bin (variant's own track eta; {fh_txt})",
               ["track |eta|", "variant", "HS tracks assoc / ev", "HS assoc eff %", "assoc purity %",
                "HS assoc / ev (OT-row truth)", "HS assoc eff % (OT-row truth)", "purity % (OT-row truth)"], rows3,
               notes=["Same definitions as the Q3 table, restricted to selected tracks in the |eta| bin. The GTT TVA "
                      "window grows with |eta| (0.37 cm at |eta| < 0.7 to 1.6 cm at 2.0-2.4).",
                      "Own truth: V0 = OT TP, V1/V3 = SP cluster-majority TP (a wrong disc/IT hit can flip it to "
                      "unmatched). OT-row truth: the OT track's TP for every variant (one truth; isolates the "
                      "reconstruction change from the truth-definition change)."])

    # admitted tracks and event migration
    p3 = per[("V3", "own")]
    found = {v: np.abs(Z[v] - g) < PV_WINDOW for v in Z}
    hs_adm = admitted & (T["sptp_evt"] == 0) & (T["sptp_bx"] == 0)
    rows = [["sub-2 GeV refit tracks admitted by V3 (passing all other cuts) / ev", f"{admitted.sum() / nev:.2f}"],
            ["  of which moved below 2 by the refit (OT pT >= 2) / ev", f"{np.sum(admitted & (T['ot_pt'] >= 2)) / nev:.2f}"],
            ["  of which saturated 1/R word / ev", f"{np.sum(admitted & np.isin(T['sp_hwRinv'], RINV_SAT)) / nev:.2f}"],
            ["  of which hard-scatter (SP truth)", f"{100 * hs_adm.sum() / max(admitted.sum(), 1):.1f} %"],
            ["  of which associated to the V3 PV / ev", f"{p3['n_adm_assoc'].mean():.3f}"],
            ["  of which associated AND hard-scatter / ev", f"{p3['n_adm_hs_assoc'].mean():.3f}"],
            ["events where V3 and V1 pick a different PV bin window (|dz| > 0.16 cm)",
             int(np.sum(np.abs(Z["V3"] - Z["V1"]) > 0.16))],
            ["events where V3 and V0 differ by > 0.16 cm", int(np.sum(np.abs(Z["V3"] - Z["V0"]) > 0.16))],
            ["V0 found & V3 found", int(np.sum(found["V0"] & found["V3"]))],
            ["V0 found & V3 lost", int(np.sum(found["V0"] & ~found["V3"]))],
            ["V0 lost & V3 found", int(np.sum(~found["V0"] & found["V3"]))],
            ["V0 lost & V3 lost", int(np.sum(~found["V0"] & ~found["V3"]))],
            ["V1 found & V3 lost", int(np.sum(found["V1"] & ~found["V3"]))],
            ["V1 lost & V3 found", int(np.sum(~found["V1"] & found["V3"]))]]
    for k in ("G", "GW", "W", "R", "UO", "U"):
        rows.append([f"admitted, class {k} (%)", f"{100 * np.sum(admitted & (T['cls'] == k)) / max(admitted.sum(), 1):.1f}"])
    table_page(pdf, f"Q3 [{tag}] what V3 adds: the admitted sub-2 GeV refit tracks and event migration",
               ["quantity", "value"], rows,
               notes=["'found' = |z_PV - GenVtx_z| < 0.15 cm. Classes as in Q1 (G genuine low-pT, W wrong-IT-hit pull-down, "
                      "R clean-hit mismeasured, U/UO unmatched)."])
    summ["q3_admitted_migration"] = {r[0].strip(): r[1] for r in rows}

    # figures
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    w = np.logspace(-3, 0.5, 60)
    for v, ls in [x for x in (("V0", "-"), ("V3", "-"), ("V1", "--"), ("V3m", ":"), ("V3ref", "-."),
                               ("V1ref", (0, (1, 3)))) if x[0] in Z]:
        d = np.abs(Z[v] - g)
        axs[0].plot(w * UM, [np.mean(d < x) for x in w], color=C[v], ls=ls, lw=2, label=VLABEL[v])
        b = np.linspace(-3000, 3000, 121)
        axs[1].hist(np.clip((Z[v] - g) * UM, b[0], b[-1]), bins=b, histtype="step", color=C[v], ls=ls, lw=2)
    axs[0].axvline(PV_WINDOW * UM, color="k", lw=0.8, ls=":")
    axs[0].set(xscale="log", xlabel="window |z_PV - z_gen| [um]", ylabel="fraction of events")
    axs[1].set(xlabel="z_PV - z_gen [um] (overflow in edge bins)", ylabel="events", yscale="log")
    for ax in axs:
        ax.grid(alpha=0.25, which="both")
    fig.suptitle(f"Q3 [{tag}] PV z (offline float fastHisto, {fh_txt}); dotted = 0.15 cm")
    legend_below(fig, ncol=3)
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    diff = per[("V3", "own")]["n_hs_assoc"] - per[("V0", "own")]["n_hs_assoc"]
    b = np.arange(diff.min() - 0.5, diff.max() + 1.5)
    ax.hist(diff, bins=b, histtype="step", lw=2, color=C["V3"], label="N(HS assoc, V3) - N(HS assoc, V0)")
    diff1 = per[("V3", "own")]["n_hs_assoc"] - per[("V1", "own")]["n_hs_assoc"]
    ax.hist(diff1, bins=b, histtype="step", lw=2, color=C["V1"], ls="--",
            label="N(HS assoc, V3) - N(HS assoc, V1) (sub-2 GeV admission only)")
    ax.set(xlabel="per-event difference in associated hard-scatter tracks", ylabel="events", yscale="log")
    ax.grid(alpha=0.25)
    legend_below(fig, ncol=1)
    pdf.savefig(fig)
    plt.close(fig)


def add_compare_set(T, E, files, name):
    """Merge the SP columns of another input set on the same events / L1TTrack rows as csp_* / csptp_*."""
    common.configure(name)
    T2, E2 = load_cache(files)
    assert np.array_equal(E["ev_event"], E2["ev_event"]) and np.array_equal(E["ev_file"], E2["ev_file"]), "events differ"
    for k in ("ot_pt", "ot_eta", "ot_z0", "ot_tpIdx", "evidx"):
        assert np.array_equal(T[k], T2[k]), f"{k} differs between the input sets"
    for k, v in T2.items():
        if k.startswith("sp_"):
            T["c" + k] = v
        elif k.startswith("sptp_"):
            T["c" + k] = v


def main():
    global REF_LABEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", choices=sorted(common.INPUT_SETS), default="canon_v1")
    ap.add_argument("--outdir", default=None, help="default: the input set's outdir")
    ap.add_argument("--compare-set", choices=sorted(common.INPUT_SETS), default=None)
    ap.add_argument("--fh-nbins", type=int, default=FH_NBINS)
    ap.add_argument("--fh-window", type=int, default=FH_WINDOW)
    ap.add_argument("--tags", default="f03-f10,f01-f10")
    a = ap.parse_args()
    common.configure(a.input_set, a.outdir)
    outdir, label = common.OUTDIR, common.INPUT_LABEL
    if a.compare_set:
        REF_LABEL = a.compare_set
        VLABEL["V3ref"] = f"V3 of {a.compare_set} (same events)"
        VLABEL["V1ref"] = f"V1 of {a.compare_set} (same events, diagnostic)"
    fh = (a.fh_nbins, a.fh_window)
    fsuf = "" if fh == (FH_NBINS, FH_WINDOW) else f"_b{fh[0]}_w{fh[1]}"
    os.makedirs(outdir, exist_ok=True)
    summary = {"input_set": a.input_set, "input_label": label, "compare_set": a.compare_set,
               "fh_nbins": fh[0], "fh_window": fh[1]}
    for tag, files in (("f03-f10", FILES_CLEAN), ("f01-f10", FILES_ALL)):
        if tag not in a.tags.split(","):
            continue
        common.configure(a.input_set, a.outdir)
        T, E = load_cache(files)
        if a.compare_set:
            add_compare_set(T, E, files, a.compare_set)
            common.configure(a.input_set, a.outdir)
        s = summary.setdefault(tag, {"n_events": int(E["ev_event"].size)})
        suffix = "" if tag == "f03-f10" else "_allfiles"
        with PdfPages(f"{outdir}/q1_tracks{suffix}.pdf") as pdf:
            moved, sat = q1(T, E, tag, pdf, s)
            q1_eta(T, E, tag, pdf, s, moved, sat)
        with PdfPages(f"{outdir}/q2_z0_resolution{suffix}.pdf") as pdf:
            q2(T, tag, pdf, s, moved, sat)
            q2_eta(T, tag, pdf, s)
        with PdfPages(f"{outdir}/q3_closure{suffix}.pdf") as pdf:
            closure(T, E, tag, pdf, s)
        with PdfPages(f"{outdir}/q3_pv{suffix}{fsuf}.pdf") as pdf:
            q3(T, E, tag, pdf, s, fh)
        print(f"done {tag}", flush=True)
    with open(f"{outdir}/summary{fsuf}.json", "w") as fh_:
        json.dump(summary, fh_, indent=1, default=str)


if __name__ == "__main__":
    main()
