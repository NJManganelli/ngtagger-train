"""The CMSSW_17 backport's statistical comparison (descriptive): shared definitions.

Plan: ~/smartpixels/doc-archive/cmssw17-backport-plan.md, section 3.4 (reviewed once, 2026-10-22, statistics and
physics; their changes are folded in there). The CMSSW_17 production of the old Phase2Spring24 TTbar sample is
compared with the CMSSW_20 production of the D121 TTbar RelVal (canon_v5 for the SmartPixels masks, canon_v4 for
IIII). It is a check that the CMSSW_17 values make sense, not an identity test.

Everything here is frozen with the tolerances (freeze.py) before the CMSSW_17 production is read.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CMSSW17_COMPARE_OUT", "/Volumes/WDMac/smartpixels_scratch/cmssw17_compare"))
GEOMETRY = os.environ.get("CMSSW17_GEOMETRY",
                          "/Users/nmangane/smartpixels/cmssw/work/otstub_arm/spix_module_geometry_D121.json")
HEADS_WORK = os.environ.get("TAGGER_SP_OUT_HEADS", "/Volumes/WDMac/smartpixels_scratch/heads_cmssw/work")

MASKS = ("AAAA", "AAAI", "IAAA", "IIII")
SP_MASKS = ("AAAA", "AAAI", "IAAA")
BASELINE_MASK = "IIII"
CLUSTER_MASK = "AAAA"          # the SmartPixels cluster table (emulated angles) is mask-independent: read it once

# the CMSSW_20 reference (D121 150X TTbar PU200 RelVal, files 1-10, 100 events each)
D121 = {"sp": Path("/Volumes/WDMac/smartpixels_scratch/canon_v5"),
        "IIII": Path("/Volumes/WDMac/smartpixels_scratch/canon_v4")}
D121_FILES = tuple(range(1, 11))
# the CMSSW_17 production of the old TTbar (10 jobs of 100 events per mask; written by the production phase)
C17 = Path(os.environ.get("CMSSW17_PRODUCTION", "/Volumes/WDMac/smartpixels_scratch/cmssw17_prod/ttbar_spring24"))
C17_FILES = tuple(range(1, 11))

# the head / cluster-CNN training files of the D121 side (tagger_sp/common.py TTBAR_TRAIN_ONLY): in-sample for
# the heads and the CNN-class angles; those quantities use the held-out (test) events of files 3-10 only
D121_TRAIN_ONLY = (1, 2)


def d121_path(mask: str, n: int) -> Path:
    base = D121["IIII"] if mask == BASELINE_MASK else D121["sp"]
    return base / f"spstudy_{mask}_f{n:02d}_100ev.root"


def c17_path(mask: str, n: int) -> Path:
    return C17 / f"spspring24_{mask}_f{n:02d}_100ev.root"


def side_path(side: str, mask: str, n: int) -> Path:
    return d121_path(mask, n) if side == "d121" else c17_path(mask, n)


# ---------------------------------------------------------------- truth: the 1 GeV rule (plan 3.4.2) -----------
TP_PT_MIN = 1.0                # the old sample's TrackingParticle selection: charged, pT >= 1 GeV, |y| < 5
EFF_PT_MIN, EFF_ETA_MAX = 2.0, 2.4   # per-particle efficiencies: charged hard-scatter particles above 2 GeV
EFF_PROD_R_MAX = 1.0                 # cm: produced within 1 cm of the beam line

# ---------------------------------------------------------------- bins (plan 3.4.4) ------------------------------
TRK_PT = (2.0, 3.0, 5.0, 10.0, 20.0, np.inf)
TRK_ETA = (0.0, 0.8, 1.4, 2.0, 2.4)
HIT_PT = (2.0, 5.0, 10.0, np.inf)
HEAD_PT = (2.0, 5.0, 10.0, np.inf)
HEAD_ETA = (0.0, 1.4, 2.4)
JET_PT = (30.0, 50.0, 100.0, 200.0, np.inf)
JET_ETA = (0.0, 1.5, 2.4)
JET_FLAVOURS = ("b1", "b2", "c", "q", "g")    # b with one B hadron, b with two or more, c, light quark, gluon
SURFACES = (1, 2, 3, 4, 11, 12, 13, 14)        # barrel IL1-IL4, discs ID1-ID4 (surface codes of the nano)
CONTROL_ETA_STEP = 0.1
CONTROL_Z0_CUT = 5.0           # cm: |z0| below / above, signed along the direction of flight
MERGE_ORDER = ("eta", "pt", "flavour")
MIN_EVENTS_PER_CELL = 50       # events contributing to a judged cell, on each side
POWER_DIVISOR = 2.93           # a cell is judged if its predicted error <= tolerance / 2.93 (80 % power to agree)
N_BOOT = 200
N_PERM = 200
ALPHA = 0.05                   # Holm, family-wise, per quantity (labels) and over the MATERIAL family

# ---------------------------------------------------------------- tolerances (plan 3.4.6) ------------------------
# class: (effect kind, tolerance, gross bound). "logratio": |log(a/b)|; "diff": |a - b|; fractions use the
# p-dependent rule of frac_tolerance(); shapes are the KS / total-variation distance.
TOL = {
    "multiplicity": ("logratio", np.log(1.05), np.log(1.5)),
    "efficiency": ("diff", 0.02, 0.10),
    "core_rms": ("logratio", np.log(1.15), np.log(1.5)),
    "full_rms": ("logratio", np.log(1.25), np.log(2.0)),
    "mean": ("diff_corerms", 0.2, 0.5),
    "chi2": ("logratio", np.log(1.15 ** 2), np.log(2.0)),
    "head_logsigma": ("diff", 0.1, 0.3),
    "head_pclean": ("diff", 0.05, 0.15),
    "head_coverage": ("diff", 0.05, 0.15),
    "shape": ("distance", 0.05, 0.20),
    # tagger inputs (pre-data calibration, 2026-10-25: the plan's table has no row for them): the mean difference in
    # units of the D121 standard deviation of the cell, with the values of the "Means (biases)" row; their shapes use
    # the "Shapes" row
    "tagger_input": ("diff_sd", 0.2, 0.5),
    "control_crossings": ("diff", 0.05, None),
    "control_discgoing": ("diff", 0.02, None),
    "control_classT_core": ("logratio", np.log(1.10), None),
    "control_classT_mean": ("diff_corerms", 0.1, None),
}
CONTROL_PER_TRACK_MIN = 0.999   # per-track visit-plan agreement


def frac_tolerance(p: float):
    """(tolerance, gross bound) of a fraction with D121 value p (plan 3.4.6): absolute 0.05 / 0.15 above 0.1,
    relative with a floor below (the statistics review: rare categories must stay visible). The rule is applied to
    the rarer side, min(p, 1 - p): a fraction near 1 is a rare complement (e.g. the refit flag at 0.999, whose rare
    category is "not performed": planted defect 2 of plan 3.4.7).
    Pre-data tool calibration (2026-10-22, accepted by the coordinator 2026-10-25): set after round 1 of the D121 nulls
    and planted defects, before any CMSSW_17 file was read; the nulls still pass. The narrower alternative (the plan's
    per-quantity remedy: tighten only the refit flag, to its defect effect / 1.5) was not taken."""
    p = min(p, 1.0 - p)
    if p >= 0.1:
        return 0.05, 0.15
    return max(0.002, min(0.05, p)), max(0.01, min(0.15, 4 * p))


# ---------------------------------------------------------------- declared sources (plan 3.4.9) -------------------
# 1 generator, 2 Geant4, 3 (removed: conditions), 4 truth threshold (inexact remap), 5 calorimeter/timing/muon
# geometry, 6 payload carry-over (never blocking), 7 pileup library, 8 luminous region, 9 inputs stored by 14_0_9,
# 10 the 1 GeV cut on the emulated angles. "control" = gate; () = no declared source.
SOURCES = {
    "baseline": ("report",),
    "control": ("control",),
    "mult_tracker": (2, 7, 9),
    "mult_pf_jets": (1, 2, 5, 7),
    "mult_particles": (1, 7),
    "eff_ot": (2, 9),
    "eff_sp": (2, 6, 10),
    "refit_quality": (1, 2, 6, 7, 10),
    "hit_truth": (1, 2, 4, 6, 7, 10),
    "res_ot_width": (2, 9),
    "res_ot_mean": (),
    "res_sp_width": (1, 2, 6, 10),
    "res_sp_mean": (6, 10),
    "angle_classC": (2, 6),
    "emu_class": (4, 10),
    "heads": (1, 2, 6, 10),
    "tagger_track": (1, 2, 6, 10),
    "tagger_neutral": (1, 5, 7),
    "ng_scores": (1, 2, 5, 6, 10),
    "jet_response": (5,),
    "report": ("report",),
}
PAYLOAD_SOURCE = 6


def classify(group: str, disagrees: bool, beyond_gross: bool, mixed_payload_rule: str = "sole") -> str:
    """MATERIAL / MINOR / ok for a disagreeing cell (plan 3.4.9). mixed_payload_rule: "sole" (recommended,
    decision 2 (a)): only quantities whose sole source is the payload carry-over are never blocking; "any"
    (decision 2 (b)): every quantity listing the payload carry-over is never blocking."""
    src = SOURCES[group]
    if not disagrees:
        return "ok"
    if src == ("report",):
        return "report"
    if src == ("control",):
        return "MATERIAL"
    if not src:
        return "MATERIAL"
    payload_only = set(src) == {PAYLOAD_SOURCE}
    if payload_only or (mixed_payload_rule == "any" and PAYLOAD_SOURCE in src):
        return "MINOR"
    return "MATERIAL" if beyond_gross else "MINOR"


def bin_index(x, edges):
    """Bin index of x in edges (len(edges) - 1 bins); -1 outside."""
    x = np.asarray(x, np.float64)
    i = np.searchsorted(edges, x, side="right") - 1
    return np.where((x >= edges[0]) & (x < edges[-1]) | ((x == edges[-1]) & np.isfinite(edges[-1])),
                    np.minimum(i, len(edges) - 2), -1)
