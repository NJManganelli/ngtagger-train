#!/usr/bin/env python
"""IT-vs-OT tracklet cost, with the REAL OT topology and windows.

Supersedes tracklet_cost_model.py, which modelled one seed pair projecting to one
layer on both sides. That was comparable but it was not what the OT does, so it
understated the OT baseline by ~7x in projection paths.

WHAT IS DIFFERENT HERE
  * OT seeds, projections, Tracklet-Calculator counts and match windows are read
    from L1Trigger/TrackFindingTracklet/interface/Settings.h, not invented.
  * Two OT study points. "full" is the OT's real phase space. "matched" is
    |eta| < ETA_MATCHED with EVERY disc-inclusive seed and EVERY disk projection
    removed, because a track inside that range stays in the barrel -- so
    including D1D2/D3D4/L1D1/L2D1 would charge the OT for phase space the
    comparison has excluded.
  * IT topology per 3-layer configuration: all THREE pair seeds (so a missing
    layer does not kill the seed) each projecting to the remaining layer, plus
    ONE 3-layer displaced seed that needs no projection. Mirrors the OT's
    prompt-pairs-plus-displaced-triplets structure.
  * The sensor angle words are QUANTIZED as a stage, so the bit-budget
    benchmarks are measured through the pipeline instead of estimated
    analytically.

WHAT THIS STILL CANNOT SAY
  * The OT model runs the same gate logic on stubs as the IT model runs on
    clusters. Real OT firmware differs in detail (its TE uses precomputed
    lookup tables on binned r/phi/z/bend, not arithmetic gates), so treat the
    OT numbers as "the same algorithm on OT data", not as an OT emulation.
  * sigma(z0) on the IT side is the section-4a OPTIMISTIC BOUND (the PixelAV
    payload's own smear). If the true resolution is worse, every quantization
    penalty here shrinks and the case for more bits weakens.
"""
from __future__ import annotations
import argparse, json, os, resource
import sys as _sys
import awkward as ak
import numpy as np
import uproot

IT_TABLE, OT_TABLE = "L1TSmartPixelsCluster", "L1TOTStub"
C_BEND = 0.29979246 * 3.8 / 2.0 / 100.0
Z_LUMI, NSIG = 15.0, 3.0
SIG_PULL_INFLATE = 1.21          # sigGlobalClusterCotTheta measured 21% optimistic
THETA_MS_MRAD = 1.36             # per ~1% X0 layer at 1 GeV  [ASSUMED]
ETA_MATCHED = 1.4
PT_MINS = (2.0, 1.5, 1.0)

# ---- prompt d0 allowance -----------------------------------------------------
# The pair phi window must admit a track whose production point is NOT on the
# beamline: a d0 displaces the azimuth by ~d0/r, worst at the innermost layer.
# An earlier revision omitted this entirely ("d0-blind"), which cost 42-61% of
# findable tracks because a d0 of only 100 um is 3.3 mrad at r = 2.9 cm against
# a ~11 mrad total window.
#
# MEASURED on PU200 ttbar, TPs above 2 GeV: the |d0| distribution is a 9.2 um
# robust-sigma CORE (the beamspot) plus a heavy tail that is REAL PHYSICS -- B
# and D decays, material secondaries -- not resolution.
#
# AN EARLIER REVISION CONCLUDED "SIZE THE WINDOW ON THE CORE". THAT WAS WRONG,
# and wrong in the direction that guts the target physics. The tail being real
# is the reason to COVER it. B hadrons have c*tau ~ 450-500 um and D hadrons
# 100-300 um, and a decay product's impact parameter is O(c*tau) largely
# independent of the parent boost, so heavy-flavour tracks sit at d0 ~ 100 um to
# a few mm -- exactly the band a core-sized window throws away. A seeder that
# only reaches 37 um finds only the tracks nobody needed help with.
#
# THREE REGIMES, not two. The prompt path must cover the first TWO:
#   prompt core      ~10 um   (beamspot)
#   heavy flavour    100 um - few mm   <-- THE TARGET
#   long-lived       cm       (separate displaced treatment)
# Sizing "prompt" at the core and "displaced" at 5 mm leaves heavy flavour
# served by neither, which is the gap this scan exists to close.
D0_CORE_SIGMA_CM = 9.2e-4
# TWO TARGETS, costed separately because they need different mechanisms.
#
# HEAVY FLAVOUR (100 um - 1 mm) rides the PROMPT pair seeds via curvature
# widening. This is the OT-equivalent path: VERIFIED that the Hnpar=5 prompt
# collection seeds from pairs only (nbitsseed_ = 3 -> 8 seeds, while the four
# displaced triplets sit behind nbitsseedextended_ = 4), and recovers d0 in the
# 5-parameter fit afterwards rather than at seeding.
#
# BSM DISPLACED (cm) cannot ride that path: matching a 1 cm d0 by curvature
# widening needs kappa to +-10, i.e. no pT floor at all. It requires the triplet,
# where three azimuths determine (kappa, phi0, d0) exactly.
HF_D0_SETTINGS = {"prompt_only": 0.0, "100um": 100e-4, "500um": 500e-4, "1mm": 1000e-4}
BSM_D0_SETTINGS = {"1mm": 0.1, "5mm": 0.5, "10mm": 1.0}
D0_DISPLACED_CM = 0.5

# ---- IT topology: 3 pair seeds + 1 displaced triplet, per configuration ----
# The SPANNING pair (L1L3 in the first config, L2L4 in the second) is dropped: it
# was the most expensive of the three (3594 tracklets/event against 1945 and 727)
# and it projects to a layer the two adjacent pairs already cover, so it bought
# messier seeds and no acceptance.
IT_CONFIGS = {
    "L1L2L3": {"pairs": [((1, 2), 3), ((2, 3), 1)], "triplet": (1, 2, 3)},
    "L2L3L4": {"pairs": [((2, 3), 4), ((3, 4), 2)], "triplet": (2, 3, 4)},
}

# ---- OT topology, from Settings.h --------------------------------------------
# seedlayers_ / projlayers_ / projdisks_ / ntc_ .  Barrel layers 1-6, disks 1-5.
OT_SEEDS = {
    "L1L2":   {"pair": (1, 2), "proj_l": (3, 4, 5, 6), "proj_d": (1, 2, 3, 4), "ntc": 12, "disc": False},
    "L2L3":   {"pair": (2, 3), "proj_l": (1, 4, 5, 6), "proj_d": (1, 2, 3, 4), "ntc": 4,  "disc": False},
    "L3L4":   {"pair": (3, 4), "proj_l": (1, 2, 5, 6), "proj_d": (1, 2),       "ntc": 4,  "disc": False},
    "L5L6":   {"pair": (5, 6), "proj_l": (1, 2, 3, 4), "proj_d": (),           "ntc": 4,  "disc": False},
    "D1D2":   {"pair": None,   "proj_l": (1, 2),       "proj_d": (3, 4, 5),    "ntc": 4,  "disc": True},
    "D3D4":   {"pair": None,   "proj_l": (1,),         "proj_d": (1, 2, 5),    "ntc": 4,  "disc": True},
    "L1D1":   {"pair": None,   "proj_l": (),           "proj_d": (2, 3, 4, 5), "ntc": 8,  "disc": True},
    "L2D1":   {"pair": None,   "proj_l": (1,),         "proj_d": (2, 3, 4),    "ntc": 4,  "disc": True},
    # displaced triplets; the pair is the FIRST TWO entries of seedlayers_
    "L2L3L4": {"pair": (3, 4), "third": 2, "proj_l": (1, 5, 6), "proj_d": (1, 2, 3), "ntc": 10, "disc": False},
    "L4L5L6": {"pair": (5, 6), "third": 4, "proj_l": (1, 2, 3), "proj_d": (),        "ntc": 10, "disc": False},
    "L2L3D1": {"pair": None,   "proj_l": (1,),         "proj_d": (2, 3, 4),    "ntc": 10, "disc": True},
    "D1D2L2": {"pair": None,   "proj_l": (1,),         "proj_d": (3, 4),       "ntc": 10, "disc": True},
}
# rphimatchcut_ / zmatchcut_ [cm], indexed [layer][seed]; 0.0 means "not used".
OT_SEED_ORDER = ["L1L2", "L2L3", "L3L4", "L5L6", "D1D2", "D3D4", "L1D1", "L2D1",
                 "L2L3L4", "L4L5L6", "L2L3D1", "D1D2L2"]
OT_RPHI_CUT = np.array([
    [0.0, 0.1, 0.07, 0.08, 0.07, 0.05, 0.0, 0.05, 0.08, 0.15, 0.125, 0.15],
    [0.0, 0.0, 0.06, 0.08, 0.05, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0],
    [0.19, 0.19, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.4, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]])
OT_Z_CUT = np.array([
    [0.0, 0.7, 5.5, 15.0, 1.5, 2.0, 0.0, 1.5, 1.0, 8.0, 1.0, 1.5],
    [0.0, 0.0, 3.5, 15.0, 1.25, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
    [0.7, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
    [3.0, 3.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0, 3.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0],
    [4.0, 0.0, 9.5, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0]])

# ---- sensor word quantization benchmarks -------------------------------------
# Reserve 3 codes: two SIGNED overflow (sign carries charge for alpha, and
# forward/backward for z0) and one invalid. The two overflows are complementary
# and jointly form the displaced tag: alpha overflow means transversely
# displaced (|d0| beyond ~5 mm at L1) or below the pT floor, z0 overflow means
# longitudinally displaced. Neither alone would cover it.
QUANT_RESERVE = 3
ALPHA_RANGE = 0.34               # +-0.17 in cot(alpha): prompt to 0.5 GeV at L4
Z0_RANGE = 30.0                  # +-15 cm: the luminous region, by physics
BENCHMARKS = {"native": None, "A_a3b5": (3, 5), "B_a4b6": (4, 6), "C_a3b7": (3, 7)}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quantize(v, bits, rng):
    """Mid-tread uniform quantizer over +-rng/2 with signed overflow.

    Returns (value, overflow_mask). Overflow keeps the SIGN, so a saturated word
    still says which way -- the thing a naive clamp destroys.
    """
    n = 2 ** bits - QUANT_RESERVE
    lsb = rng / n
    ovf = np.abs(v) > rng / 2.0
    qv = np.clip(np.round(v / lsb) * lsb, -rng / 2.0, rng / 2.0)
    return np.where(ovf, np.sign(v) * rng / 2.0, qv), ovf, lsb


def quant_sigma(bits, rng):
    return (rng / (2 ** bits - QUANT_RESERVE)) / np.sqrt(12.0)


# Event stride for the composite sort key below. Must exceed the 6*pi band that
# the wrap-around duplication occupies, plus twice the widest search window, so
# one event's band can never reach its neighbour's.
EVENT_STRIDE = 64.0

# ---- candidate-triplet budget ------------------------------------------------
# A projection stage expands to CANDIDATE TRIPLETS: one surviving cluster pair
# tupled with one layer-C cluster inside the projection window. That count -- not
# the event count -- sets peak memory, at ~128 bytes per candidate triplet.
#
# THE 40 GB INCIDENT (2026-09-09), and why --batch-events could not prevent it.
# The projection window carries the d0 allowance, NSIG * (... + d0/r_inner). At a
# displaced point (d0 = 0.5-1 cm against r_inner ~ 2.9 cm) that window is
# ~0.5-1 rad, so each surviving cluster pair matches ~1e3 layer-C clusters
# instead of the ~54 measured on the prompt path. Candidate triplets per event go
# from ~2.2e5 to ~1e8, i.e. tens of GB INSIDE ONE EVENT -- below batch
# granularity, so no batch size can bound it. It exhausted a 24 GB machine.
#
# Two limits, because the failure has two independent halves:
#   PAIR_BUDGET  bounds ONE slice, so memory is flat in the candidate-triplet
#                count however large that count becomes.
#   ROW_CEILING  refuses a seed whose total is absurd, because bounding memory
#                alone converts a crash into a run that never finishes.
# Both are read from the per-cluster-pair candidate counts, which searchsorted
# yields with NO expansion. The previous guard polled RSS between file chunks:
# that can only autopsy an allocation that has already completed.
TRIPLET_BYTES = 8 * 16
PHYS_RAM_GB = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024.0 ** 3
SAFE_RSS_FRAC = 0.40
DEFAULT_PAIR_BUDGET_GB = 1.5
DEFAULT_TRIPLET_CEILING = 300_000_000


def triplets_for_budget(gb):
    return max(1 << 16, int(gb * (1024.0 ** 3) / TRIPLET_BYTES))


_MAX_SLICE = triplets_for_budget(DEFAULT_PAIR_BUDGET_GB)
_RSS_LIMIT_GB = None
_RSS_DIV = 1024.0 ** 3 if _sys.platform == "darwin" else 1024.0 ** 2


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIV


def set_limits(slice_triplets, rss_gb):
    global _MAX_SLICE, _RSS_LIMIT_GB
    _MAX_SLICE, _RSS_LIMIT_GB = slice_triplets, rss_gb


def rss_check():
    """Called per slice, so it can interrupt work IN PROGRESS."""
    if _RSS_LIMIT_GB is not None and peak_gb() > _RSS_LIMIT_GB:
        raise SystemExit(f"ABORT: peak RSS {peak_gb():.1f} GB exceeded ceiling "
                         f"{_RSS_LIMIT_GB:.1f} GB. Lower --pair-budget-gb.")


class TooWide(Exception):
    """A seed's candidate-triplet count exceeds the ceiling.

    Not a crash and not something to raise the ceiling for: it means the
    configuration is combinatorially infeasible as formulated, which is a finding
    to record and move past.
    """

    def __init__(self, n, ceiling):
        super().__init__(f"{n:.3g} candidate triplets exceeds ceiling {ceiling:.3g}")
        self.n, self.ceiling = int(n), int(ceiling)


def _window_index(evA, phiA, evB, phiB, half):
    """Per-A candidate ranges, with NO expansion. O((NA + NB) log NB).

    The event is folded into a composite key
        k = event * EVENT_STRIDE + (phi + pi)
    so a phi-window search is a plain interval search in key space, and because
    EVENT_STRIDE dwarfs the 6*pi band the duplication below occupies, a window
    can never leak into a neighbouring event.

    Wrap-around is handled by duplicating B at k-2pi, k, k+2pi. Both copies of a
    pair can only be inside one window if that window were >= 2pi wide, which it
    never is, so nothing is double counted.

    An earlier revision looped over events and called searchsorted per event.
    That is O(N) per event to build the mask alone, i.e. O(N*E) -- tolerable at
    40 events and hopeless at 1e6. This is O(N log N) once.
    """
    half = np.asarray(half)
    if half.size and float(np.max(half)) > (EVENT_STRIDE - 6.0 * np.pi) / 2.0:
        raise SystemExit("search window too wide for EVENT_STRIDE; raise the stride")
    kb = evB.astype(np.float64) * EVENT_STRIDE + (phiB + np.pi)
    kb3 = np.concatenate([kb - 2.0 * np.pi, kb, kb + 2.0 * np.pi])
    src3 = np.tile(np.arange(len(kb), dtype=np.int64), 3)
    o = np.argsort(kb3, kind="stable")
    kb3, src3 = kb3[o], src3[o]
    ka = evA.astype(np.float64) * EVENT_STRIDE + (phiA + np.pi)
    lo = np.searchsorted(kb3, ka - half, "left")
    hi = np.searchsorted(kb3, ka + half, "right")
    return src3, lo, (hi - lo).astype(np.int64)


def _expand(src3, lo, cnt, a0):
    """Expand one contiguous A-range into explicit (ia, ib); ia is absolute."""
    tot = int(cnt.sum())
    ia = np.repeat(np.arange(a0, a0 + len(cnt), dtype=np.int64), cnt)
    ramp = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return ia, src3[np.repeat(lo, cnt) + ramp]


def pairs_in_window_slices(evA, phiA, evB, phiB, half, max_slice=None,
                           ceiling=DEFAULT_TRIPLET_CEILING):
    """Yield (ia, ib) in CONTIGUOUS SLICES OF A, each at most max_slice wide.

    Slicing on A is what keeps the downstream arbitration exact: every candidate
    sharing an A index lands in the same slice, so a per-A argmin over a slice is
    the same answer as over the whole set. A single A whose own count exceeds
    max_slice is emitted whole rather than split, because splitting it WOULD
    break that property; its count is bounded by 3 * N_B, which is small.

    Raises TooWide, before yielding anything, if the total exceeds `ceiling`.
    """
    if max_slice is None:
        max_slice = _MAX_SLICE
    src3, lo, cnt = _window_index(evA, phiA, evB, phiB, half)
    tot = int(cnt.sum())
    if tot > ceiling:
        raise TooWide(tot, ceiling)
    if not tot:
        return
    edges = np.r_[0, np.cumsum(cnt)]
    a0 = 0
    while a0 < len(cnt):
        a1 = int(np.searchsorted(edges, edges[a0] + max_slice, "right")) - 1
        a1 = max(a1, a0 + 1)                       # always make progress
        yield _expand(src3, lo[a0:a1], cnt[a0:a1], a0)
        a0 = a1


def pairs_in_window(evA, phiA, evB, phiB, half, ceiling=DEFAULT_TRIPLET_CEILING):
    """All (A,B) pairs within +-half in phi, for all events at once.

    Materialises the whole expansion, so it is only for stages whose count is
    known small (the OT stub seeds). Anything driven by a d0-inflated window must
    use pairs_in_window_slices instead.
    """
    src3, lo, cnt = _window_index(evA, phiA, evB, phiB, half)
    tot = int(cnt.sum())
    if tot > ceiling:
        raise TooWide(tot, ceiling)
    if not tot:
        return (np.empty(0, dtype=np.int64),) * 2
    return _expand(src3, lo, cnt, 0)


IT_COLS = ["layer", "globalR", "globalZ", "globalPhi", "globalClusterPhi",
           "globalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta",
           "sigY", "tpIdx", "tpPt", "tpVx", "tpVy", "tpVz", "tpPhi", "tpEta"]


def it_chunks(path, nev, step=16):
    """Stream the cluster table in chunks, yielding one flat dict per chunk.

    THE BATCH SIZE IS THE MEMORY CONTROL. The stages are vectorised across every
    THE CHUNK SIZE IS NOT THE MEMORY CONTROL -- see --pair-budget-gb. An earlier
    revision claimed here that peak RSS was "~0.2 GB/event, so 16 ~ 3 GB". That
    was wrong by ~30x: measured, the displaced configuration reached 12.4 GB on
    TWO events, because the blow-up is in the candidate triplets of a SINGLE
    event and no chunk size can reach below one event. The chunk size only trades
    I/O calls against the per-cluster arrays, which are small.
    """
    seen = 0
    for A in uproot.iterate(f"{path}:Events", [f"{IT_TABLE}_{c}" for c in IT_COLS],
                            step_size=step, entry_stop=nev):
        n = ak.to_numpy(ak.num(A[f"{IT_TABLE}_layer"]))
        D = {c: ak.to_numpy(ak.flatten(A[f"{IT_TABLE}_{c}"])) for c in IT_COLS}
        D["event"] = np.repeat(np.arange(seen, seen + len(n)), n)
        D["_events"] = np.arange(seen, seen + len(n))
        seen += len(n)
        yield D
        del A, D


def load_ot(path, nev):
    t = uproot.open(f"{path}:Events")
    cols = ["layer", "isBarrel", "r", "phi", "z", "bend", "tpIdx", "tpPt"]
    have = [c for c in cols if f"{OT_TABLE}_{c}" in t.keys()]
    A = t.arrays([f"{OT_TABLE}_{c}" for c in have], entry_stop=nev)
    n = ak.to_numpy(ak.num(A[f"{OT_TABLE}_layer"]))
    D = {c: ak.to_numpy(ak.flatten(A[f"{OT_TABLE}_{c}"])) for c in have}
    D["event"] = np.repeat(np.arange(len(n)), n)
    D["eta"] = np.abs(np.arcsinh(D["z"] / np.maximum(D["r"], 1e-6)))
    return D, len(n), have


# ==========================================================================
# IT stages
# ==========================================================================
# WHY THE DISPLACED TRIPLET IS A DIFFERENT PROBLEM, NOT A HARDER ONE.
# A 3-hit seed with d0 FREE has 5 parameters (kappa, phi0, d0, z0, cotTheta).
# The three r-phi measurements determine (kappa, phi0, d0) EXACTLY -- zero
# degrees of freedom -- so there is NO r-phi chi2 to reject on. The three r-z
# measurements give 2 parameters and 1 DOF. So a displaced triplet's only
# rejection comes from that single r-z constraint plus the per-cluster sensor
# angles, which supply 6 further constraints on an otherwise exactly-determined
# system. For prompt seeding the angles are a convenience worth ~3x; for
# displaced seeding they are the ONLY discrimination there is.
def it_prepare(D, bench):
    """Per-cluster kappa_alpha, z0 and their effective sigmas, after quantization."""
    half = wrap(D["globalPhi"] - D["globalClusterPhi"])
    cot_a = np.sin(half) / (C_BEND * D["globalR"])          # kappa implied by alpha
    z0 = D["globalZ"] - D["globalR"] * D["globalClusterCotTheta"]
    s_kap = D["sigGlobalClusterPhi"] * np.abs(np.cos(half)) / (C_BEND * D["globalR"])
    s_z0 = D["globalR"] * D["sigGlobalClusterCotTheta"] * SIG_PULL_INFLATE
    ovf_a = np.zeros(len(z0), bool)
    ovf_z = np.zeros(len(z0), bool)
    if bench is not None:
        ab, bb = bench
        # alpha is encoded as cot(alpha); convert the range through c*r
        cota_raw = np.sin(half)
        q, ovf_a, lsb_a = quantize(cota_raw, ab, ALPHA_RANGE)
        cot_a = q / (C_BEND * D["globalR"])
        s_kap = np.hypot(s_kap, quant_sigma(ab, ALPHA_RANGE) / (C_BEND * D["globalR"]))
        z0, ovf_z, lsb_z = quantize(z0, bb, Z0_RANGE)
        s_z0 = np.hypot(s_z0, quant_sigma(bb, Z0_RANGE))
    return {"kap_a": cot_a, "s_kap": s_kap, "z0": z0, "s_z0": s_z0,
            "ovf_a": ovf_a, "ovf_z": ovf_z}


# ---- projecting to the third layer: SEARCH ON z, CONFIRM ON phi -----------
# MEASURED (projection_residuals.py, PU200 ttbar, 100 events, pT > 2 GeV, correct
# cluster triples only): between the prompt core and d0 = 1-5 mm, sigma(dphi) at
# layer C widens 40.4x (0.671 -> 27.1 mrad) while sigma(dz) is FLAT within 10%
# (77.8 -> 68.8 um). L2L3->L4 gives 24.2x on phi and flat on z.
# An earlier 40-event pass read sigma(dz) as slightly NARROWING at large d0
# (0.7x); that was 17 entries of noise. It is flat. A
# pair-derived curvature assumes d0 = 0, so phi carries the full d0 bias; the z
# prediction z0 + r*cot(theta) is untouched by a TRANSVERSE impact parameter.
#
# Searching phi therefore pays for heavy-flavour coverage twice: the window must
# open to ~58 mrad to keep d0 ~ 1 mm tracks, and every prompt seed pays it too.
# Searching z reaches the same tracks in a 464 um window that never widens.
#
# THE INTRA-LAYER RADIAL SPREAD IS WHAT MAKES THIS NON-TRIVIAL, and an earlier
# estimate here ignored it. A layer is not a thin shell: MEASURED ptp is ~1.1 cm
# (staggered/tilted ladders, and the outer shell alone is ~5.7 mm thick and
# continuous, so splitting into two shells does not help -- equal-count binning
# saturates near 4.4 mm even at 16 bins). Searching z at one reference radius
# needs padding (spread/2)*|cot|, which at |cot| = 1.16 is ~5000 um against a
# 232 um resolution window: that is WORSE than the phi search it replaces.
#
# So bin the layer in radius finely enough that geometry stops dominating:
# N_RBINS = 48 over ~1.1 cm gives 229 um bins, whose padding (bin/2)*|cot| is
# ~133 um at the median |cot| -- under the resolution window, so the search is
# resolution-limited as intended.
Z_STRIDE = 200.0        # z spans ~40 cm; windows are mm-scale
N_RBINS = 48


def build_layer_z_index(cidx, event, r, z, sig, nrbins=N_RBINS):
    """Per radial bin of one layer: cluster indices sorted by (event, z).

    Returned per bin: the sorted composite key, the GLOBAL cluster indices in
    that order, the bin's mid radius and half thickness, and the worst sigma in
    the bin (so a search window can bound the per-candidate sigma without
    gathering). The indices must be global: an earlier revision stored positions
    within the layer-C subset, which the caller then used to index D directly --
    the selftest caught it as "missed 947 of 947".
    """
    if not len(r):
        return []
    edges = np.linspace(r.min(), r.max() + 1e-9, nrbins + 1)
    b = np.clip(np.searchsorted(edges, r, "right") - 1, 0, nrbins - 1)
    out = []
    for ib in range(nrbins):
        sel = np.flatnonzero(b == ib)
        if not len(sel):
            continue
        k = event[sel].astype(np.float64) * Z_STRIDE + z[sel]
        o = np.argsort(k, kind="stable")
        out.append((k[o], cidx[sel[o]], 0.5 * (edges[ib] + edges[ib + 1]),
                    0.5 * (edges[ib + 1] - edges[ib]), float(sig[sel].max())))
    return out


def _slices(cnt, max_slice):
    """Contiguous A-ranges whose candidate counts sum to at most max_slice."""
    edges = np.r_[0, np.cumsum(cnt)]
    a0 = 0
    while a0 < len(cnt):
        a1 = int(np.searchsorted(edges, edges[a0] + max_slice, "right")) - 1
        yield a0, max(a1, a0 + 1)
        a0 = max(a1, a0 + 1)


def candidates_by_z(ev_p, z0p, cot, sct, ms, zidx, ceiling=DEFAULT_TRIPLET_CEILING):
    """Yield (pair index, layer-C cluster index) for z-consistent candidates.

    One interval search per radial bin. A given cluster pair therefore appears in
    several yields -- once per bin that has a candidate -- so the caller must
    arbitrate AFTER collecting every bin, not per yield.
    """
    total = 0
    for kz, cidx, rmid, rhalf, sgmax in zidx:
        zpred = z0p + rmid * cot
        half = (NSIG * (np.hypot(sgmax, rmid * sct) + rmid * ms)
                + rhalf * np.abs(cot))
        ka = ev_p.astype(np.float64) * Z_STRIDE + zpred
        lo = np.searchsorted(kz, ka - half, "left")
        hi = np.searchsorted(kz, ka + half, "right")
        cnt = (hi - lo).astype(np.int64)
        total += int(cnt.sum())
        if total > ceiling:
            raise TooWide(total, ceiling)
        if not cnt.sum():
            continue
        for a0, a1 in _slices(cnt, _MAX_SLICE):
            ja, jb = _expand(cidx, lo[a0:a1], cnt[a0:a1], a0)
            yield ja, jb


_EXHAUSTIVE = False


def candidates_exhaustive(ev_p, ev_c, cidx, ceiling=DEFAULT_TRIPLET_CEILING):
    """EVERY layer-C cluster in the pair's own event. The --selftest oracle.

    Deliberately does no geometric reasoning, so comparing against it isolates
    exactly one thing: whether the z-binned search drops a candidate the exact
    test would have kept. Only tractable on a couple of events.
    """
    o = np.argsort(ev_c, kind="stable")
    evs, cs = ev_c[o], cidx[o]
    lo = np.searchsorted(evs, ev_p, "left")
    cnt = (np.searchsorted(evs, ev_p, "right") - lo).astype(np.int64)
    if int(cnt.sum()) > ceiling:
        raise TooWide(int(cnt.sum()), ceiling)
    for a0, a1 in _slices(cnt, _MAX_SLICE):
        yield _expand(cs, lo[a0:a1], cnt[a0:a1], a0)


# ---- stage 1: pair on (z0, phi) JOINTLY, not phi-then-gate -----------------
# Measured selectivities against the 61.6M possible L1xL2 cluster pairs in one
# PU200 event: phi window alone 0.886%, z0 consistency alone 3.97%, the two
# together 0.035%. The old order formed 116,829 cluster pairs on phi and then
# discarded 89% at the z0 gate, so pairing on both at once forms far fewer
# cluster pairs for an identical accepted set. Same mistake the projection stage
# had, one stage earlier.
#
# THE sigma(z0) TAIL IS WHAT MAKES THIS NON-TRIVIAL. Per-cluster sigma(z0)
# measured: median 0.140 cm, q99 0.580, q99.9 2.56, MAX 21.77 cm. Sizing buckets
# on the median (0.596 cm) but the offset loop on the maximum needs K = 156, i.e.
# 313 buckets scanned per cluster -- slower than the phi search it replaces. So
# clusters with sigma above Z0_SPLIT_Q are WILDCARDS searched on phi alone, on
# BOTH sides of the pair: they carry no usable z0, and pretending otherwise is
# what blows up the loop. Capping sigma instead would be cheaper and would
# silently drop real clusters, so it is not done.
PHI_STRIDE = 24.0       # > 6*pi, the span the +-2pi duplication occupies
Z0_SPLIT_Q = 99.0       # percentile of sigma(z0) above which a cluster is wild


def build_phi_z0_index(z0, phi, s_z0, event):
    """Sharp clusters keyed by (event, z0 bucket, phi); wild ones by phi alone.

    Every key duplicates phi at +-2pi so a wrap-around window is a plain
    interval search. PHI_STRIDE exceeds the 6*pi span those copies occupy, so one
    bucket's band can never reach its neighbour's.
    """
    if not len(z0):
        return None
    s_split = float(np.percentile(s_z0, Z0_SPLIT_Q))
    wild = s_z0 > s_split
    sharp = ~wild
    med = float(np.median(s_z0[sharp])) if sharp.any() else 1.0
    w = max(2.0 * NSIG * np.sqrt(2.0) * med, 0.05)
    z_lo, z_hi = float(z0.min()), float(z0.max())
    nb = max(int(np.ceil((z_hi - z_lo) / w)) + 1, 1)
    kmax_off = int(np.ceil(NSIG * np.sqrt(2.0) * s_split / w)) + 1

    def _key(pos, bucket):
        if not len(pos):
            return np.empty(0), np.empty(0, np.int64)
        base = (event[pos].astype(np.float64) * nb + bucket) * PHI_STRIDE
        p = phi[pos] + np.pi
        k = np.concatenate([base + p - 2 * np.pi, base + p, base + p + 2 * np.pi])
        s = np.tile(pos, 3)
        o = np.argsort(k, kind="stable")
        return k[o], s[o]

    ps = np.flatnonzero(sharp)
    bs = np.clip(((z0[ps] - z_lo) / w).astype(np.int64), 0, nb - 1)
    ksharp, ssharp = _key(ps, bs)
    bmax = np.zeros(nb)
    if len(ps):
        np.maximum.at(bmax, bs, s_z0[ps])
    zeros = lambda n: np.zeros(n, np.int64)
    pw = np.flatnonzero(wild)
    kwild, swild = _key(pw, zeros(len(pw)))
    # A-side wildcards need every B cluster, sharp or not, on phi alone
    pall = np.arange(len(z0))
    kall, sall = _key(pall, zeros(len(pall)))
    return dict(ksharp=ksharp, ssharp=ssharp, kwild=kwild, swild=swild,
                kall=kall, sall=sall, w=w, z_lo=z_lo, nb=nb, bmax=bmax,
                kmax_off=kmax_off, s_split=s_split)


def pairs_joint_z0_phi(evA, phiA, z0A, sA, B, half_phi,
                       ceiling=DEFAULT_TRIPLET_CEILING):
    """Yield (ia, ib) for cluster pairs consistent in BOTH z0 and phi.

    ib indexes the array handed to build_phi_z0_index, matching
    pairs_in_window_slices so the caller is unchanged.
    """
    nb, w, z_lo = B["nb"], B["w"], B["z_lo"]
    total = 0
    bA = np.clip(((z0A - z_lo) / w).astype(np.int64), 0, nb - 1)
    pa = phiA + np.pi
    # A-SIDE WILDCARDS. A cluster whose own sigma(z0) exceeds the split has a z0
    # reach wider than the offset loop spans, so the bucketed search would
    # under-reach and silently DROP real cluster pairs. Measured on a first
    # revision that only split the B side: 0.2-1.4 lost cluster pairs/event, and
    # only on seeds with layer 1 as the inner layer, where the small radius
    # amplifies z0. They get phi alone, which is what their z0 is worth.
    wildA = sA > B["s_split"]
    sharpA = ~wildA
    for off in range(-B["kmax_off"], B["kmax_off"] + 1):
        bt = bA + off
        ok = sharpA & (bt >= 0) & (bt < nb)
        if not ok.any():
            continue
        bt_c = np.clip(bt, 0, nb - 1)
        # does the pair's z0 window actually reach this bucket's z0 span?
        reach = NSIG * np.hypot(sA, B["bmax"][bt_c])
        blo = z_lo + bt_c * w
        ok &= (z0A + reach >= blo) & (z0A - reach <= blo + w)
        if not ok.any():
            continue
        key = (evA.astype(np.float64) * nb + bt_c) * PHI_STRIDE + pa
        lo = np.searchsorted(B["ksharp"], key - half_phi, "left")
        hi = np.searchsorted(B["ksharp"], key + half_phi, "right")
        cnt = np.where(ok, hi - lo, 0).astype(np.int64)
        total += int(cnt.sum())
        if total > ceiling:
            raise TooWide(total, ceiling)
        if not cnt.sum():
            continue
        for a0, a1 in _slices(cnt, _MAX_SLICE):
            yield _expand(B["ssharp"], lo[a0:a1], cnt[a0:a1], a0)
    # two remaining families, disjoint from the above and from each other, so no
    # cluster pair is counted twice: sharp A x wild B, and wild A x every B.
    key = evA.astype(np.float64) * nb * PHI_STRIDE + pa
    for kk, ss, rows in ((B["kwild"], B["swild"], sharpA),
                         (B["kall"], B["sall"], wildA)):
        if not len(kk) or not rows.any():
            continue
        lo = np.searchsorted(kk, key - half_phi, "left")
        hi = np.searchsorted(kk, key + half_phi, "right")
        cnt = np.where(rows, hi - lo, 0).astype(np.int64)
        total += int(cnt.sum())
        if total > ceiling:
            raise TooWide(total, ceiling)
        if not cnt.sum():
            continue
        for a0, a1 in _slices(cnt, _MAX_SLICE):
            yield _expand(ss, lo[a0:a1], cnt[a0:a1], a0)


_JOINT_PAIRING = True


def it_pair_seed(D, Q, ev_idx, la, lb, lc, ptmin, use_angles, displaced=False, d0_cm=0.0):
    """One seed type over an ENTIRE CHUNK -- all events at once, no event loop.
    ev_idx is the cluster index set to consider (normally the whole chunk); the
    event is folded into the pair-matching sort key, so cross-event pairs are
    impossible by construction rather than by iteration."""
    idx = {L: ev_idx[D["layer"][ev_idx] == L] for L in (la, lb, lc)}
    out = {"in_A": len(idx[la]), "in_B": len(idx[lb]), "in_C": len(idx[lc])}
    if min(len(v) for v in idx.values()) < 2:
        return out
    kmax = 1.0 / ptmin
    if use_angles:                        # single-cluster alpha veto
        for L in (la, lb, lc):
            k = np.abs(Q["kap_a"][idx[L]]) <= kmax + NSIG * Q["s_kap"][idx[L]]
            k |= Q["ovf_a"][idx[L]] & displaced      # displaced KEEPS alpha overflow
            idx[L] = idx[L][k]
        out["after_alpha_veto"] = sum(len(idx[L]) for L in (la, lb, lc))
    if min(len(v) for v in idx.values()) < 2:
        return out
    rA = np.median(D["globalR"][idx[la]]); rB = np.median(D["globalR"][idx[lb]])
    dr = abs(rB - rA)
    ms = (THETA_MS_MRAD * 1e-3 / ptmin) * np.sqrt(2.0)
    # d0 allowance: prompt seeds assume d0=0; displaced must open the window by
    # d0/r at the inner layer, which is where it hurts most.
    d0_eff = D0_DISPLACED_CM if displaced else d0_cm
    d0_allow = d0_eff / min(rA, rB)
    # THE CURVATURE ACCEPTANCE, not the phi window, is what gates d0 tolerance.
    # A pair-derived kappa assumes d0 = 0; a real d0 biases it by
    #     kappa_bias = d0 * (1/rA - 1/rB) / (c * dr)
    # = 10.1 per cm for IT L1L2 against 0.21 per cm for OT L1L2, so the IT pair
    # seed is ~48x more d0-sensitive purely from radius. An earlier revision
    # widened only the phi window: the extra pairs entered and were then killed
    # by |kappa| <= kmax, so efficiency stayed flat at 0.707 while pair work grew
    # 4.4x. Widening BOTH is the fix, and it costs the same as lowering the pT
    # floor by the same factor.
    kap_bias_per_cm = abs(1.0 / rA - 1.0 / rB) / (C_BEND * max(dr, 0.1))
    kap_slack = kap_bias_per_cm * d0_eff
    half_w = C_BEND * dr * kmax + ms + 5e-4 + d0_allow
    # ---- stage 1: CLUSTER PAIRS, streamed in slices ------------------------
    # Only the SURVIVING cluster pairs are retained; the full expansion of
    # layer-A x layer-B never exists at once.
    keepA, keepB, keepK, keepC, keepZ = [], [], [], [], []
    n_phi = n_kap = n_z0l = n_aok = n_zok = 0
    Bx = (build_phi_z0_index(Q["z0"][idx[lb]], D["globalPhi"][idx[lb]],
                             Q["s_z0"][idx[lb]], D["event"][idx[lb]])
          if _JOINT_PAIRING else None)
    if Bx is not None:
        pair_gen = pairs_joint_z0_phi(
            D["event"][idx[la]], D["globalPhi"][idx[la]], Q["z0"][idx[la]],
            Q["s_z0"][idx[la]], Bx, half_w)
    else:
        pair_gen = pairs_in_window_slices(
            D["event"][idx[la]], D["globalPhi"][idx[la]],
            D["event"][idx[lb]], D["globalPhi"][idx[lb]],
            np.full(len(idx[la]), half_w))
    for ia, ib in pair_gen:
        rss_check()
        n_phi += len(ia)
        gA, gB = idx[la][ia], idx[lb][ib]
        drp = D["globalR"][gB] - D["globalR"][gA]
        ok = np.abs(drp) > 0.5
        kap = wrap(D["globalPhi"][gA] - D["globalPhi"][gB]) / np.where(ok, C_BEND * drp, 1e9)
        cot = (D["globalZ"][gB] - D["globalZ"][gA]) / np.where(ok, drp, 1e9)
        z0p = D["globalZ"][gA] - D["globalR"][gA] * cot
        ok &= np.abs(kap) <= kmax + kap_slack
        n_kap += int(ok.sum())
        if not displaced:                      # beamspot / luminous-region gate
            ok &= np.abs(z0p) <= Z_LUMI
            n_z0l += int(ok.sum())
        if use_angles:
            # OVERFLOW SEMANTICS, and they are opposite for the two targets. The
            # code ranges are set AT the physics boundary (+-0.17 in cot alpha is
            # the 0.5 GeV floor; +-15 cm in z0 is the luminous region), so for a
            # PROMPT seed an overflowed word says "inconsistent with the prompt
            # hypothesis" and must REJECT. Treating overflow as a gate bypass --
            # which an earlier revision did -- lets 40% of clusters through
            # ungated and makes coarser quantization look BETTER than none, which
            # is how the bug announced itself. For a DISPLACED seed the same
            # overflow carries no information against the hypothesis, so it is
            # permissive.
            for g_ in (gA, gB):
                cons = np.abs(kap - Q["kap_a"][g_]) <= NSIG * np.maximum(Q["s_kap"][g_], 1e-9)
                ok &= (cons | Q["ovf_a"][g_]) if displaced else (cons & ~Q["ovf_a"][g_])
            n_aok += int(ok.sum())
            sz = np.hypot(Q["s_z0"][gA], Q["s_z0"][gB])
            consz = np.abs(Q["z0"][gA] - Q["z0"][gB]) <= NSIG * sz
            anyovf = Q["ovf_z"][gA] | Q["ovf_z"][gB]
            ok &= (consz | anyovf) if displaced else (consz & ~anyovf)
            n_zok += int(ok.sum())
        if ok.any():
            keepA.append(gA[ok]); keepB.append(gB[ok])
            keepK.append(kap[ok]); keepC.append(cot[ok]); keepZ.append(z0p[ok])
    out["pairs_phi"] = n_phi
    out["pairs_kappa"] = n_kap
    out["kappa_slack"] = float(kap_slack)
    if not displaced:
        out["pairs_z0lumi"] = n_z0l
    if use_angles:
        out["pairs_alpha_ok"] = n_aok
        out["pairs_z0_ok"] = n_zok
    out["tracklets"] = int(sum(len(x) for x in keepA))
    if not keepA:
        return out
    gA, gB = np.concatenate(keepA), np.concatenate(keepB)
    kap, cot, z0p = (np.concatenate(keepK), np.concatenate(keepC),
                     np.concatenate(keepZ))
    del keepA, keepB, keepK, keepC, keepZ

    # ---- per-cluster layer-C arrays, gathered ONCE --------------------------
    # NOT D["globalR"][idx[lc]][jb] inside the loop, which the previous revision
    # did: that copies the whole layer-C column and THEN indexes it, rebuilding
    # the copy on every slice, and it replicates each layer-C cluster's radius
    # once per cluster pair that might match it.
    cC = idx[lc]
    rC, phC, zC = D["globalR"][cC], D["globalPhi"][cC], D["globalZ"][cC]
    sgC = np.maximum(D["sigY"][cC], 1e-6)

    phi0 = wrap(D["globalPhi"][gA] + C_BEND * D["globalR"][gA] * kap)
    # PROJECT TO EACH CANDIDATE'S OWN RADIUS, not the layer median. Comparing a
    # median-radius projection against a candidate's actual z carries an error
    #     dz = (r_actual - r_median) * cot(theta)
    # which for |cot| ~ 2 and the mm-scale radius spread within a layer is ~1 cm
    # against an ~800 um window. That one approximation rejected 52.7% of
    # findable prompt-core tracks at the z step, while an independent residual
    # measurement showed the window contains 93.4% of correct triples -- i.e. the
    # entire "core efficiency loss" was this bug, not physics.
    rCmed = float(np.median(rC))
    skap = np.sqrt(2.0) * 1e-3 / (C_BEND * max(dr, 0.1))
    sph = np.hypot(5e-4, C_BEND * rCmed * skap) + ms + d0_allow
    # sigma(cot theta) from the pair's two z measurements, using the MEASURED
    # per-cluster CPE sigma (sigY ~ 12.5 um) rather than an assumed 30 um.
    sct = np.hypot(np.maximum(D["sigY"][gA], 1e-6),
                   np.maximum(D["sigY"][gB], 1e-6)) / max(dr, 0.1)
    out["projections"] = int(len(gA))

    # ---- stage 2: CANDIDATE TRIPLETS, found by z, confirmed by phi ---------
    zidx = build_layer_z_index(cC, D["event"][cC], rC, zC, sgC)
    n_cand = 0
    sv_p, sv_c, sv_res = [], [], []
    gen = (candidates_exhaustive(D["event"][gA], D["event"][cC], cC) if _EXHAUSTIVE
           else candidates_by_z(D["event"][gA], z0p, cot, sct, ms, zidx))
    for ja, gC in gen:
        rss_check()
        n_cand += len(ja)
        rc_i = D["globalR"][gC]
        zC_i = z0p[ja] + rc_i * cot[ja]
        szp_i = np.hypot(np.maximum(D["sigY"][gC], 1e-6), rc_i * sct[ja]) + rc_i * ms
        phiC_i = wrap(phi0[ja] - C_BEND * rc_i * kap[ja])
        dphi = np.abs(wrap(D["globalPhi"][gC] - phiC_i))
        good = (np.abs(D["globalZ"][gC] - zC_i) <= NSIG * szp_i) & (dphi <= NSIG * sph)
        if not good.any():
            continue
        sv_p.append(ja[good]); sv_c.append(gC[good]); sv_res.append(dphi[good])
    out["match_cand"] = n_cand
    if not sv_p:
        out["match_cand_z"] = 0
        out["tracks_to_fit"] = 0
        return out
    pa, pc, pr = np.concatenate(sv_p), np.concatenate(sv_c), np.concatenate(sv_res)
    out["match_cand_z"] = int(len(pa))
    # ARBITRATE ONCE, over every radial bin together: a cluster pair's candidates
    # are spread across bins, so a per-yield argmin would pick a different winner.
    o = np.lexsort((pr, pa))
    first = np.r_[True, pa[o][1:] != pa[o][:-1]]
    pk = o[first]
    out["tracks_to_fit"] = int(len(pk))
    out["_trip"] = (gA[pa[pk]], gB[pa[pk]], pc[pk])
    return out


def ot_seed_cost(D, ev_idx, name, ptmin, use_bend, cal, eta_max):
    """One OT seed: pair formation + projections, with the REAL match windows."""
    S = OT_SEEDS[name]
    if S["pair"] is None:
        return None
    si = OT_SEED_ORDER.index(name)
    keep = D["isBarrel"][ev_idx] > 0
    if eta_max is not None:
        keep &= D["eta"][ev_idx] < eta_max
    base = ev_idx[keep]
    la, lb = S["pair"]
    idx = {L: base[D["layer"][base] == L] for L in (la, lb)}
    out = {"in_A": len(idx[la]), "in_B": len(idx[lb]),
           "ntc": S["ntc"], "n_proj_paths": len(S["proj_l"]) + (0 if eta_max else len(S["proj_d"]))}
    if min(len(v) for v in idx.values()) < 2:
        return out
    kmax = 1.0 / ptmin
    if use_bend:
        for L in (la, lb):
            if L in cal:
                s, sg, _ = cal[L]
                idx[L] = idx[L][np.abs(s * D["bend"][idx[L]]) <= kmax + NSIG * sg]
        out["after_bend"] = sum(len(idx[L]) for L in (la, lb))
    if min(len(v) for v in idx.values()) < 2:
        return out
    rA = np.median(D["r"][idx[la]]); rB = np.median(D["r"][idx[lb]])
    dr = abs(rB - rA)
    ms = (THETA_MS_MRAD * 1e-3 / ptmin) * np.sqrt(2.0)
    ia, ib = pairs_in_window(D["event"][idx[la]], D["phi"][idx[la]],
                             D["event"][idx[lb]], D["phi"][idx[lb]],
                             np.full(len(idx[la]), C_BEND * dr * kmax + ms + 5e-4))
    out["pairs_phi"] = int(len(ia))
    if not len(ia):
        return out
    gA, gB = idx[la][ia], idx[lb][ib]
    drp = D["r"][gB] - D["r"][gA]
    ok = np.abs(drp) > 0.5
    kap = wrap(D["phi"][gA] - D["phi"][gB]) / np.where(ok, C_BEND * drp, 1e9)
    cot = (D["z"][gB] - D["z"][gA]) / np.where(ok, drp, 1e9)
    z0 = D["z"][gA] - D["r"][gA] * cot
    ok &= (np.abs(kap) <= kmax) & (np.abs(z0) <= Z_LUMI)
    out["pairs_kappa_z0"] = int(ok.sum())
    if use_bend:
        for g_, L in ((gA, la), (gB, lb)):
            if L in cal:
                s, sg, _ = cal[L]
                ok &= np.abs(kap - s * D["bend"][g_]) <= NSIG * sg
        out["pairs_bend_ok"] = int(ok.sum())
    out["tracklets"] = int(ok.sum())
    if not ok.sum():
        return out
    kap, cot, z0 = kap[ok], cot[ok], z0[ok]
    phi0 = wrap(D["phi"][gA[ok]] + C_BEND * D["r"][gA[ok]] * kap)
    tot_c = tot_cz = tot_fit = 0
    trips = []
    for lp in S["proj_l"]:                       # layer projections only in matched
        tgt = base[D["layer"][base] == lp]
        if len(tgt) < 2:
            continue
        rC = np.median(D["r"][tgt])
        wr = OT_RPHI_CUT[lp - 1][si]; wz = OT_Z_CUT[lp - 1][si]
        if wr <= 0:                              # not a configured projection
            continue
        phiC = wrap(phi0 - C_BEND * rC * kap); zC = z0 + rC * cot
        ja, jb = pairs_in_window(D["event"][gA[ok]], phiC,
                                 D["event"][tgt], D["phi"][tgt],
                                 np.full(len(phiC), wr / rC))
        tot_c += len(ja)
        if len(ja):
            good = np.abs(D["z"][tgt][jb] - zC[ja]) <= wz
            tot_cz += int(good.sum())
            mm = np.zeros(len(phiC), bool); mm[ja[good]] = True
            tot_fit += int(mm.sum())
    out["projections"] = int(len(phi0)) * max(len(S["proj_l"]), 1)
    out["match_cand"] = tot_c
    out["match_cand_z"] = tot_cz
    out["tracks_to_fit"] = tot_fit
    # ---- truth, so the OT reports the SAME quality metrics as the IT --------
    # Without this the OT could only be compared on combinatorics while the IT
    # had efficiency and fake rate, and that asymmetry silently biases any
    # conclusion drawn from the pair.
    if "tpIdx" in D and trips:
        na = nt_ = nu = 0
        for ga_, gb_, gc_ in trips:
            ta, tb, tc = D["tpIdx"][ga_], D["tpIdx"][gb_], D["tpIdx"][gc_]
            real = (ta >= 0) & (ta == tb) & (tb == tc)
            na += len(ta); nt_ += int(real.sum()); nu += len(np.unique(ta[real]))
        out["cand_truth_matched"] = nt_
        out["unique_true_found"] = nu
        out["cand_total"] = na
    return out


def measure_d0_core(D, ptmin=2.0):
    """Robust sigma of the TP transverse impact parameter above ptmin, MEASURED
    from this file rather than assumed. One row per (event, TP).

    Reported alongside quantiles because the distribution is NOT Gaussian: a
    beamspot-sized core plus a real displaced tail. Sizing a prompt window on a
    quantile of the whole thing gives millimetres and is wrong; size it on the
    core and give the tail to the displaced path.
    """
    m = (D["tpIdx"] >= 0) & (D["tpPt"] > ptmin)
    key = D["event"][m].astype(np.int64) * (1 << 20) + D["tpIdx"][m].astype(np.int64)
    _, first = np.unique(key, return_index=True)
    vx, vy, ph = D["tpVx"][m][first], D["tpVy"][m][first], D["tpPhi"][m][first]
    d0 = -vx * np.sin(ph) + vy * np.cos(ph)
    core = 1.4826 * np.median(np.abs(d0 - np.median(d0)))
    q = np.percentile(np.abs(d0), [50, 68.3, 95.4, 99.73, 99.994])
    return {"n_tp": int(len(d0)), "core_sigma_cm": float(core),
            "q50_cm": float(q[0]), "q68_cm": float(q[1]), "q95_cm": float(q[2]),
            "q99p73_cm": float(q[3]), "q99p994_cm": float(q[4]),
            "frac_within_100um": float((np.abs(d0) < 100e-4).mean()),
            "frac_within_200um": float((np.abs(d0) < 200e-4).mean())}


# ==========================================================================
# Truth bookkeeping -- ARRAY BASED
# ==========================================================================
# An earlier revision did this with Python dicts and sets of (event, tpIdx)
# tuples, rebuilt per benchmark x configuration x pT floor. On 40 events that
# reached 9.7 GB resident and was growing ~1 GB/min: ~1.07 M boxed tuple keys
# plus per-combination set rebuilds. It was killed before finishing.
#
# Everything below is int64 arrays and sorted-array set algebra, so cost is
# O(N log N) in clusters with no Python-level per-track work. The only remaining
# per-event Python loop is pair formation, which is inherently per-event and is
# itself vectorised inside; input is streamed in chunks so nothing scales with
# total event count.
TP_KEY_SHIFT = 20          # tpIdx must fit in 2^20; asserted at build time


def tp_key(event, tpidx):
    """Pack (event, tpIdx) into one int64. Event is the HIGH word so that
    sorting by key groups by event, which the pair stages rely on."""
    return event.astype(np.int64) * (1 << TP_KEY_SHIFT) + tpidx.astype(np.int64)


def build_tp_index(D):
    """One row per (event, TP): sorted key array plus aligned |d0| and pT.

    Sorted, so every later lookup is a searchsorted rather than a hash.
    """
    m = D["tpIdx"] >= 0
    if m.sum() and D["tpIdx"][m].max() >= (1 << TP_KEY_SHIFT):
        raise SystemExit(f"tpIdx exceeds 2^{TP_KEY_SHIFT}; widen TP_KEY_SHIFT")
    k = tp_key(D["event"][m], D["tpIdx"][m])
    order = np.argsort(k, kind="stable")
    k = k[order]
    first = np.r_[True, k[1:] != k[:-1]]
    uk = k[first]
    sel = np.flatnonzero(m)[order][first]
    d0 = np.abs(-D["tpVx"][sel] * np.sin(D["tpPhi"][sel])
                + D["tpVy"][sel] * np.cos(D["tpPhi"][sel]))
    return {"key": uk, "d0": d0, "pt": D["tpPt"][sel]}


def findable_keys(D, la, lb, lc, ptmin):
    """Sorted keys of TPs above ptmin with a cluster on ALL THREE layers.

    Fully vectorised: the event is inside the key, so there is no event loop.
    """
    base = (D["tpIdx"] >= 0) & (D["tpPt"] >= ptmin)
    sets = []
    for L in (la, lb, lc):
        m = base & (D["layer"] == L)
        sets.append(np.unique(tp_key(D["event"][m], D["tpIdx"][m])))
    out = sets[0]
    for nxt in sets[1:]:
        out = np.intersect1d(out, nxt, assume_unique=True)
    return out


def recovered_keys(D, ga, gb, gc):
    """Sorted keys of TPs recovered by a truth-consistent candidate."""
    ta, tb, tc = D["tpIdx"][ga], D["tpIdx"][gb], D["tpIdx"][gc]
    real = (ta >= 0) & (ta == tb) & (tb == tc)
    if not real.any():
        return np.empty(0, dtype=np.int64)
    return np.unique(tp_key(D["event"][ga][real], ta[real]))


def cand_purity(D, ga, gb, gc):
    """(n_candidates, n_truth_matched) for one batch of candidate triples."""
    ta, tb, tc = D["tpIdx"][ga], D["tpIdx"][gb], D["tpIdx"][gc]
    return len(ta), int(((ta >= 0) & (ta == tb) & (tb == tc)).sum())


def eff_by_d0(rec, find, tpi, edges=(0.0, 100e-4, 500e-4, 1e9)):
    """Efficiency split by |d0| band, array based.

    The INCLUSIVE number cannot judge heavy flavour: only ~5% of pT>2 TPs sit
    above 100 um, so it can move by at most ~3.5 points however well the HF path
    works. The band split is the quantity that decides whether b-tagging-relevant
    tracks are seeded.
    """
    pos = np.searchsorted(tpi["key"], find)
    ok = (pos < len(tpi["key"])) & (tpi["key"][np.minimum(pos, len(tpi["key"]) - 1)] == find)
    find, d0 = find[ok], tpi["d0"][pos[ok]]
    got = np.isin(find, rec, assume_unique=True)
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        b = (d0 >= lo) & (d0 < hi)
        if not b.any():
            continue
        lab = (f"d0_{lo*1e4:.0f}to{hi*1e4:.0f}um" if hi < 1e8 else f"d0_gt{lo*1e4:.0f}um")
        out[lab] = {"n_findable": int(b.sum()), "efficiency": float(got[b].mean())}
    return out


def render(R):
    """Print every table this study reports. Kept in the script so the tables
    are reproducible from the JSON instead of being retyped by hand."""
    d0 = R.get("d0_measured", {})
    if d0:
        print("\n=== TP transverse impact parameter, pT>2 (MEASURED) ===")
        print(f"  n={d0['n_tp']}  core sigma={d0['core_sigma_cm']*1e4:.1f} um")
        print(f"  |d0| q50={d0['q50_cm']*1e4:.1f}  q68={d0['q68_cm']*1e4:.1f}  "
              f"q95={d0['q95_cm']*1e4:.1f}  q99.73={d0['q99p73_cm']*1e4:.0f}  "
              f"q99.994={d0['q99p994_cm']*1e4:.0f} um")
        print(f"  within 100um: {d0['frac_within_100um']:.4f}   200um: {d0['frac_within_200um']:.4f}")
    if R.get("hf_scan"):
        print("\n=== HEAVY FLAVOUR on prompt pair seeds (4a+6b, pT>2) ===")
        print(f"{'setting':20s}{'d0[um]':>8}{'pairs_phi':>11}{'tracklets':>11}"
              f"{'hits tested':>13}{'per proj':>10}{'efficiency':>12}{'fake':>8}")
        for k, v in R["hf_scan"].items():
            print(f"{k:20s}{v['d0_um']:8.0f}{v['pairs_phi']:11.0f}{v['tracklets']:11.0f}"
                  f"{v['hits_tested']:13.0f}{v['per_projection']:10.1f}"
                  f"{v['efficiency']:12.3f}{v['fake_fraction']:8.3f}")
    if R.get("proj_cost"):
        print("\n=== projection cost (per event) ===")
        print(f"{'side / seed':26s}{'tracklets':>10}{'projections':>12}{'hits tested':>12}"
              f"{'per proj':>10}{'pass z':>8}")
        for k, v in R["proj_cost"].items():
            print(f"{k:26s}{v['tracklets']:10.0f}{v['projections']:12.0f}"
                  f"{v['hits_tested']:12.0f}{v['per_projection']:10.1f}{v['pass_z']:8.0f}")


def calibrate_ot_bend(path, nev):
    """bend -> kappa per OT layer, MEASURED from on-track stubs vs track rInv,
    so no strip pitch or sensor spacing has to be assumed."""
    t = uproot.open(f"{path}:Events")
    S = t.arrays(["L1TTrackStub_bend", "L1TTrackStub_layer", "L1TTrackStub_trackIdx"], entry_stop=nev)
    K = t.arrays(["L1TTrack_rInv"], entry_stop=nev)
    ns = ak.to_numpy(ak.num(S["L1TTrackStub_bend"]))
    nt = ak.to_numpy(ak.num(K["L1TTrack_rInv"]))
    evs = np.repeat(np.arange(len(ns)), ns)
    bend = ak.to_numpy(ak.flatten(S["L1TTrackStub_bend"]))
    lay = ak.to_numpy(ak.flatten(S["L1TTrackStub_layer"])).astype(np.int64)
    ti = ak.to_numpy(ak.flatten(S["L1TTrackStub_trackIdx"])).astype(np.int64)
    rinv = ak.to_numpy(ak.flatten(K["L1TTrack_rInv"]))
    off = np.concatenate([[0], np.cumsum(nt)])
    kap = np.full(len(bend), np.nan)
    ok = ti >= 0
    kap[ok] = rinv[off[evs[ok]] + ti[ok]] / (2.0 * C_BEND)
    cal = {}
    for L in range(1, 7):
        m = (lay == L) & np.isfinite(kap) & (np.abs(bend) > 1e-6)
        if m.sum() < 200:
            continue
        s = float(np.sum(bend[m] * kap[m]) / np.sum(bend[m] ** 2))
        r = kap[m] - s * bend[m]
        cal[L] = (s, float(1.4826 * np.median(np.abs(r - np.median(r)))), int(m.sum()))
    return cal


def acc_add(acc, o):
    for k, v in o.items():
        if k.startswith("_") or isinstance(v, str):
            continue
        acc[k] = acc.get(k, 0) + v


def selftest(a):
    """Does the z-binned search accept exactly what an exhaustive search does?"""
    global _EXHAUSTIVE
    set_limits(triplets_for_budget(a.pair_budget_gb),
               min(a.rss_ceiling_gb, SAFE_RSS_FRAC * PHYS_RAM_GB))
    nbad = ncmp = 0
    for D in it_chunks(a.input, a.nev or 2, a.batch_events):
        allidx = np.arange(len(D["layer"]))
        for bname, bench in BENCHMARKS.items():
            Q = it_prepare(D, bench)
            for cname, cfg in IT_CONFIGS.items():
                jobs = [((la, lb), lc, False, a.d0) for (la, lb), lc in cfg["pairs"]]
                jobs.append((cfg["triplet"][:2], cfg["triplet"][2], True, 0.0))
                for (la, lb), lc, disp, d0c in jobs:
                    got = {}
                    for mode in (False, True):
                        _EXHAUSTIVE = mode
                        globals()["_JOINT_PAIRING"] = not mode
                        try:
                            o = it_pair_seed(D, Q, allidx, la, lb, lc, 2.0,
                                             True, disp, d0c)
                        except TooWide:
                            got = None
                            break
                        t = o.get("_trip")
                        # NOT just the triples. A first revision compared only those, and a
                        # joint-pairing bug that dropped 0.2-1.4 cluster pairs/event passed
                        # clean, because the lost pairs never reached a triple. pairs_phi and
                        # match_cand are WORK counters and MUST differ, so they stay out.
                        got[mode] = (
                            (set() if t is None else
                             set(zip(t[0].tolist(), t[1].tolist(), t[2].tolist()))),
                            tuple(round(o.get(c, -1), 6) for c in
                                  ("tracklets", "match_cand_z", "tracks_to_fit")))
                    _EXHAUSTIVE = False
                    if got is None:
                        continue
                    ncmp += 1
                    tag = (f"{bname}|{cname}|L{la}L{lb}->L{lc}"
                           + ("|displaced" if disp else ""))
                    if got[False] != got[True]:
                        nbad += 1
                        (sF, cF), (sT, cT) = got[False], got[True]
                        print(f"  MISMATCH {tag}: missed {len(sT - sF)}, invented "
                              f"{len(sF - sT)}, of {len(sT)} reference triples; "
                              f"counters {cF} vs {cT}")
        break
    print(f"selftest: {ncmp} seed configurations compared, {nbad} mismatched")
    raise SystemExit(1 if nbad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None,
                    help="stop after this many events; default all")
    ap.add_argument("--batch-events", type=int, default=16,
                    help="events per streamed chunk. NOT the memory knob -- see "
                         "--pair-budget-gb. Peak memory is set by candidate triplets "
                         "held at once, which for a d0-inflated projection window "
                         "explodes WITHIN a single event. Does not change results.")
    ap.add_argument("--pair-budget-gb", type=float, default=DEFAULT_PAIR_BUDGET_GB,
                    help="THE MEMORY KNOB: bytes per candidate-triplet slice. Peak "
                         "memory is flat in the total candidate-triplet count.")
    ap.add_argument("--rss-ceiling-gb", type=float,
                    default=round(min(6.0, SAFE_RSS_FRAC * PHYS_RAM_GB), 1),
                    help=f"abort above this. HARD-CLAMPED to {SAFE_RSS_FRAC:.0%} of "
                         f"physical RAM ({PHYS_RAM_GB:.1f} GB here); a ceiling above "
                         "RAM cannot protect anything.")
    ap.add_argument("--benchmark", default=None, choices=sorted(BENCHMARKS),
                    help="run ONE angle-bit benchmark; default all")
    ap.add_argument("--d0", type=float, default=500e-4,
                    help="prompt d0 allowance [cm] for the benchmark pass")
    ap.add_argument("--skip-hf-scan", action="store_true")
    ap.add_argument("--skip-ot", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="compare the z-binned projection search against an "
                         "exhaustive one on a few events; exits nonzero on any "
                         "difference in the accepted cluster triples")
    ap.add_argument("-o", "--out", default="tracklet_topology_cost.json")
    a = ap.parse_args()
    benches = ({a.benchmark: BENCHMARKS[a.benchmark]} if a.benchmark else dict(BENCHMARKS))
    if a.selftest:
        return selftest(a)

    acc, purity, reck, findk = {}, {}, {}, {}
    tp_k, tp_d0 = [], []
    nev = ncl = 0

    # A CEILING ABOVE PHYSICAL RAM IS NOT A CEILING. Passing
    # --rss-ceiling-gb 40 on a 24 GB machine is what turned a diagnosable bug
    # into a hard lockup, so the option is clamped here rather than trusted.
    ceiling = min(a.rss_ceiling_gb, SAFE_RSS_FRAC * PHYS_RAM_GB)
    if ceiling < a.rss_ceiling_gb:
        print(f"  --rss-ceiling-gb {a.rss_ceiling_gb:.1f} clamped to {ceiling:.1f} "
              f"({SAFE_RSS_FRAC:.0%} of {PHYS_RAM_GB:.1f} GB physical)")
    set_limits(triplets_for_budget(a.pair_budget_gb), ceiling)

    for D in it_chunks(a.input, a.nev, a.batch_events):
        rss_check()
        nev += len(D["_events"])
        ncl += len(D["layer"])
        ti = build_tp_index(D)
        tp_k.append(ti["key"]); tp_d0.append(ti["d0"])
        allidx = np.arange(len(D["layer"]))

        for cname, cfg in IT_CONFIGS.items():
            for ptmin in PT_MINS:
                findk.setdefault((cname, ptmin), []).append(
                    findable_keys(D, *cfg["triplet"], ptmin))

        for bname, bench in benches.items():
            Q = it_prepare(D, bench)
            for cname, cfg in IT_CONFIGS.items():
                for ptmin in PT_MINS:
                    jobs = [((la, lb), lc, False, a.d0) for (la, lb), lc in cfg["pairs"]]
                    jobs.append((cfg["triplet"][:2], cfg["triplet"][2], True, 0.0))
                    for (la, lb), lc, disp, d0c in jobs:
                        tag = f"{bname}|{cname}|pt{ptmin:g}|" + (
                            "displaced" if disp else f"L{la}L{lb}->L{lc}")
                        A = acc.setdefault(tag, {})
                        P = purity.setdefault(tag, [0, 0])
                        # ONE call for the whole chunk -- no event loop
                        try:
                            o = it_pair_seed(D, Q, allidx, la, lb, lc, ptmin,
                                             True, disp, d0c)
                        except TooWide as e:
                            # A RESULT, not a failure: this design point needs
                            # more candidate triplets than any sane machine can
                            # hold, so it is infeasible as formulated.
                            A["infeasible_candidate_triplets"] = e.n
                            print(f"  INFEASIBLE {tag}: {e.n:.3g} candidate "
                                  f"triplets (ceiling {e.ceiling:.3g})")
                            continue
                        acc_add(A, o)
                        keys = []
                        if "_trip" in o:
                            ga, gb, gc = o["_trip"]
                            keys.append(recovered_keys(D, ga, gb, gc))
                            nc, nt = cand_purity(D, ga, gb, gc)
                            P[0] += nc; P[1] += nt
                        if keys:
                            # collapse per chunk so the key list cannot grow with
                            # the sample -- this is what blew up at 9.7 GB before
                            prev = reck.get(tag)
                            merged = np.concatenate(keys + ([prev] if prev is not None else []))
                            reck[tag] = np.unique(merged)
        del D

    tpi = {"key": np.concatenate(tp_k), "d0": np.concatenate(tp_d0)}
    o = np.argsort(tpi["key"], kind="stable")
    tpi = {"key": tpi["key"][o], "d0": tpi["d0"][o]}
    find = {k: np.unique(np.concatenate(v)) for k, v in findk.items()}

    R = {"n_events": nev, "n_clusters": ncl, "clusters_per_event": ncl / max(nev, 1),
         "nsig": NSIG, "z_lumi_cm": Z_LUMI, "eta_matched": ETA_MATCHED,
         "d0_allowance_cm": a.d0, "benchmarks_run": sorted(benches), "it": {}}
    print(f"IT: {nev} events, {ncl/max(nev,1):.0f} clusters/event")
    for tag in sorted(acc):
        bname, cname, ptl, seed = tag.split("|")
        ptmin = float(ptl[2:])
        per = {k: v / max(nev, 1) for k, v in acc[tag].items() if not isinstance(v, dict)}
        fk = find[(cname, ptmin)]
        rk = reck.get(tag, np.empty(0, dtype=np.int64))
        hit = np.isin(fk, rk, assume_unique=True)
        per["n_findable"] = int(len(fk))
        per["efficiency"] = float(hit.mean()) if len(fk) else 0.0
        per["fake_fraction"] = 1.0 - purity[tag][1] / max(purity[tag][0], 1)
        per["by_d0"] = eff_by_d0(rk, fk, tpi)
        pj = per.get("projections", 0)
        per["hits_per_projection"] = (per.get("match_cand", 0) / pj) if pj else 0.0
        R["it"][tag] = per
        print(f"  {tag:44s} trk={per.get('tracklets',0):9.0f} "
              f"hits/proj={per['hits_per_projection']:6.1f} "
              f"eff={per['efficiency']:.3f} fake={per['fake_fraction']:.3f}")
        for bk, bv in per["by_d0"].items():
            print(f"      {bk:20s} eff={bv['efficiency']:.3f} (n={bv['n_findable']})")

    if not a.skip_ot:
        try:
            O, onev, have = load_ot(a.input, a.nev or 10 ** 9)
        except Exception as ex:
            print(f"OT SKIPPED: {type(ex).__name__}: {ex}")
            O = None
        if O is not None:
            eta = O["eta"]
            R["ot_input"] = {"stubs_per_event_full": len(O["layer"]) / onev,
                             "stubs_per_event_matched": float((eta < ETA_MATCHED).sum() / onev),
                             "has_truth": "tpIdx" in have}
            print(f"OT: {R['ot_input']['stubs_per_event_full']:.0f} stubs/event full, "
                  f"{R['ot_input']['stubs_per_event_matched']:.0f} matched")
            cal = calibrate_ot_bend(a.input, a.nev or 10 ** 9)
            R["ot"] = {}
            oall = np.arange(len(O["layer"]))
            for ename, emax in (("barrel_alleta", None), ("barrel_matched", ETA_MATCHED)):
                seeds = [k for k, v in OT_SEEDS.items()
                         if v["pair"] is not None and (emax is None or not v["disc"])]
                R["ot"][f"{ename}_topology"] = {
                    "seeds": seeds, "n_seeds": len(seeds),
                    "n_projection_paths": sum(len(OT_SEEDS[k]["proj_l"])
                                              + (0 if emax else len(OT_SEEDS[k]["proj_d"]))
                                              for k in seeds),
                    "n_tc": sum(OT_SEEDS[k]["ntc"] for k in seeds)}
                for ptmin in PT_MINS:
                    for scheme in ("position", "bend"):
                        tot = {}
                        for sname in seeds:
                            A = {}
                            r = ot_seed_cost(O, oall, sname, ptmin,
                                             scheme == "bend", cal, emax)
                            if r:
                                acc_add(A, r)
                            tot[sname] = {k: v / onev for k, v in A.items()}
                        tot["_sum_tracklets"] = sum(v.get("tracklets", 0)
                                                    for v in tot.values() if isinstance(v, dict))
                        R["ot"][f"{ename}_pt{ptmin:g}_{scheme}"] = tot
                        print(f"  OT {ename:14s} pT>{ptmin:<4g} {scheme:8s} "
                              f"trk={tot['_sum_tracklets']:9.0f}")
    R["peak_rss_gb"] = round(peak_gb(), 2)
    R["batch_events"] = a.batch_events
    print(f"peak RSS {R['peak_rss_gb']:.2f} GB at {a.batch_events} events/batch")
    json.dump(R, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
