#!/usr/bin/env python
"""OMNIBUS combinatorics study for the SmartPixels refit, on the Clusters tier.

*** THIS SCRIPT IS DELIBERATELY AN OMNIBUS. *** It is expected to grow many more
figures over time. Each study is a `study_*` function registered in STUDIES,
takes the same prepared arrays, and returns a dict of numbers for the JSON
sidecar. Add a function, add it to STUDIES, done -- do not fork this script.

THE QUESTION. A refit crossing must test the clusters on the module it crosses.
At PU200 that module carries 20-40 clusters (measured 40.6/32.7/31.6/19.7 on
L1-L4), the current static window admits ~2 and truncates at 8. Nothing so far
measures what the window discards, whether a covariance-derived cone would do
better, or what other handles (angle, charge) could cut the pool down.

WHAT IT NEEDS. The Clusters tier (L1PFTrkNanoSmartPixClusters[withGen]):
untruncated per-cluster table joined to the per-crossing refit records on
(event, detId). Both are produced by the same job, so the clusters are exactly
the ones the refit was offered.

THE CONE. Built from the SEED covariance: projSeedSigX/Y = sqrt(diag(H C H^T))
projected to the module, with NO Kalman updates -- the single-shot cold start.
q68 -> 1.00 sigma, q95 -> 1.96 sigma per coordinate. This is the track's own
uncertainty and excludes the measurement term, which is the right choice here:
we are asking how big the search region must be, not how well a hit fits.

    pixi run python eval_refitq/combinatorics/spix_combinatorics_omnibus.py \
        -i <clusters-tier nano.root> -o eval_refitq/combinatorics/
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import textwrap

import numpy as np
from statistics import NormalDist
import uproot

SENTINEL = -900.0
LAYERS = (1, 2, 3, 4)
CONES = {"q68": 1.0, "q95": 1.959964}
PT_EDGES = np.array([2, 3, 4, 6, 10, 20, 1e9])
ETA_EDGES = np.array([-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4])
CLUSTER_TABLE = "L1TSmartPixelsCluster"
REF = "L1TTrack"
# eta bin EDGES for the cot(theta) resolution study; cot(theta) = sinh(eta)
ETA_RES_EDGES = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
Z_HALF_RANGE_CM = 15.0   # +-z0 span a seeding stage would have to slice


# --------------------------------------------------------------------------
# GLOSSARY -- every term this script prints or plots, defined
# --------------------------------------------------------------------------
# Written out as spix_glossary.txt on every run, printed at startup, and rendered
# into the spare panel of the sweep figure. Assume the reader has NOT been in the
# conversation these numbers came from: several of these terms look like each other
# (qX vs pXX), several are conditional in ways that change their meaning
# (containment), and one is a trap (max). None of that is guessable from a column
# heading.
GLOSSARY = [
    ("crossing",
     "One track's PREDICTED intersection with one instrumented module. The unit "
     "most per-layer quantities are counted in -- not a track, not a cluster. One "
     "track normally has one crossing per instrumented layer, occasionally two "
     "where it clips overlapping modules."),
    ("cluster",
     "A reconstructed group of adjacent fired pixels: one measurement. NOT a single "
     "pixel -- an inclined track fires several pixels and they form one cluster."),
    ("cone / search window",
     "The region around the predicted position on a module inside which clusters are "
     "accepted as CANDIDATES for the refit. Half-width is k times the projected "
     "track uncertainty, applied per axis (local x and local y independently)."),
    ("qX  (cone quantile)",
     "HOW WIDE THE CONE IS -- an INPUT, a design choice. Expressed as the central "
     "Gaussian probability per axis: k = Phi^-1((1+X/100)/2), so q68->k=1.00, "
     "q95->k=1.96, q99->k=2.58, q99.9->k=3.29, q99.99->k=3.89. NOMINAL ONLY: it "
     "assumes unit-width pulls. Ours are 0.81-1.52, so qX is NOT the achieved "
     "efficiency. Use the measured containment for that."),
    ("pXX  (percentile)",
     "WHERE A TRACK SITS in a distribution over tracks -- an OUTPUT. p99 = the "
     "99th-percentile track, i.e. 1 track in 100 is busier. NOT a cone width. "
     "qX and pXX look alike and are unrelated: qX is the window you choose, pXX "
     "describes the spread of what that window returns."),
    ("k  /  k_sigma",
     "Cone half-width in units of the projected track uncertainty. The quantity "
     "actually used in the cut |residual| < k*sigma."),
    ("containment",
     "MEASURED hit-finding efficiency, per crossing, CONDITIONAL ON THE HIT "
     "EXISTING. Denominator: crossings where the correct cluster exists on that "
     "module. Numerator: those where it also falls inside the cone. So it does not "
     "penalise a hit that was never produced, and it is NOT the probability the "
     "refit then PICKS that cluster (that is selection, measured separately). "
     "Averaged over crossings, so builds instrumenting different layers average "
     "different layer mixes -- L1 crossings are intrinsically harder, which is why "
     "an L1-instrumented build scores lower without the algorithm being worse."),
    ("local frame  /  local x, local y",
     "The module's own coordinate system. LOCAL X is the FINE-pitch axis (25 um) and "
     "points along r-phi, i.e. the BENDING plane; LOCAL Y is the coarse axis (100 um) "
     "and points along z. Verified from the CPE resolutions: median sigX = 3.0 um "
     "against sigY = 12.5 um. Local z is the module NORMAL."),
    ("cotAlpha  (ALPHA = BENDING)",
     "p_x_local / p_z_local, PixelAV convention -- the track's angle in the local x "
     "direction, hence the BENDING-PLANE (r-phi) angle. NOTE it carries almost no pT "
     "information by itself: median |cotAlpha| is flat against pT at every layer "
     "(L1: 0.147 / 0.148 / 0.148 / 0.146 over pT 2-3, 3-5, 5-10, 10+ GeV), because "
     "its value at a barrel module is set by geometry and module tilt rather than by "
     "curvature. Its use is COMPATIBILITY -- does this cluster's angle match what the "
     "track predicts here -- not measuring pT."),
    ("cotBeta  (BETA = LONGITUDINAL)",
     "p_y_local / p_z_local -- the angle in the local y (z) direction, hence the "
     "LONGITUDINAL (r-z) angle, essentially cot(theta) at the module. Verified: "
     "corr(|cotBeta|, |global cot(theta)|) = +0.956 with median difference 0.0089. "
     "The correlation WITHOUT absolute values is only +0.19, because the sign of "
     "local y relative to global z flips module to module -- the same orientation "
     "ambiguity that produced the global-direction sense bug."),
    ("cone is 2D, not phi-only",
     "The candidate cut is a BOX in both local coordinates: |dx| < k*sigma_x AND "
     "|dy| < k*sigma_y, applied independently per axis. So it constrains r-phi and z "
     "together. It uses POSITION ONLY -- no angle information enters the candidate "
     "count, so every combination number here is the no-angle case, and applying "
     "alpha/beta compatibility on top would reduce it further."),
    ("module boundary  (KNOWN BIAS)",
     "Candidates are joined to a crossing by EXACT detId, so a cluster just across a "
     "module edge is invisible even when it lies geometrically inside the cone. "
     "Modules are 3.38 x 4.33 cm. Measured fraction of crossings whose cone reaches "
     "past an edge: 5.0% / 6.7% / 8.2% at q99 / q99.9 / q99.99 for the refit-order "
     "cone, and 8.6% / 11.6% / 14.2% for the seed cone; worst at L4 (14.9%), which is "
     "first-visited and so has the widest cone, mildest at L1 (2.4%). Combination "
     "counts are therefore biased LOW by roughly that much. Separately, the real "
     "system does not project to one module at all -- it sees the aggregate clusters "
     "of an eta x phi sector board -- so these are a FLOOR that assumes perfect "
     "module targeting."),
    ("correct / true cluster",
     "The cluster whose DOMINANT TrackingParticle is the TP the track was matched "
     "to. Dominance is by charge, winner-takes-pixel."),
    ("combinations / track",
     "The number of (L1,L2,L3,L4) hit tuples one refit must test = PRODUCT of the "
     "candidate counts over INSTRUMENTED layers. A layer with no candidate "
     "contributes a factor of 1, not 0, because the refit skips it and carries on. "
     "Worked: 2,1,0,2 -> 2*1*1*2 = 4;  3,2,1,1 -> 6. Pinned by "
     "tests/test_combination_counting.py."),
    ("activeSP  /  A  /  I",
     "Four-character mask, one character per TBPX layer L1,L2,L3,L4. A = that layer "
     "IS instrumented with smart pixels; I = it is not. An uninstrumented layer "
     "delivers NO data at L1 latency at all -- conventional pixels are read out only "
     "after an L1 accept -- so it contributes nothing to L1 track building, not even "
     "a position. Each mask is therefore a candidate DETECTOR BUILD, not an "
     "algorithm setting. Example: AAII = L1 and L2 instrumented, L3 and L4 not."),
    ("nSP", "Number of instrumented layers = count of 'A' in the mask."),
    (">=2hit / >=3hit  (viability)",
     "Fraction of tracks to which the refit actually attached that many hits. 0% for "
     "every two-layer build is STRUCTURAL (only two layers exist to be hit), not a "
     "performance statement."),
    ("projSeedSig{X,Y}",
     "Projected track uncertainty from the OT SEED covariance alone -- single shot, "
     "no refit updates. The naive cone."),
    ("projSig{X,Y}",
     "RUNNING projected track uncertainty: after multiple-scattering Q has been "
     "added and before this layer's own update. The refit-order cone. Tightens as "
     "the refit walks inward, so the same layer has different widths in different "
     "builds depending on how deep it is visited."),
    ("outsideIn",
     "The refit's layer visit order: outermost instrumented layer first, working "
     "inward. The FIRST-visited layer gets no update and so carries the raw seed "
     "cone."),
    ("pull",
     "(measured - predicted) / claimed uncertainty. Width 1.0 means the claimed "
     "uncertainty is honest; 4.1 means it claims 4x better precision than it has."),
    ("Q  (process noise)",
     "Multiple-scattering term added to the covariance before projecting, so the "
     "cone grows to reflect material the track passed through. Without it the "
     "covariance only ever shrank."),
    ("nonant",
     "One ninth of the azimuth, 2*pi/9 = 698 mrad. The L1 track finder's phi sector, "
     "carried in the nano as phiSector 0-8. Sectors OVERLAP: each spans about 1031 "
     "mrad, so roughly 167 mrad of shared region per side, which is how tracks near a "
     "boundary are still found. Note etaSector exists in the nano but is an unfilled "
     "sentinel (constant 99), so there is no eta sectoring to inherit."),
    ("fixed eta-phi grid  /  bin",
     "In a refit-only design the pre-processing has NO TRACK KNOWLEDGE -- it runs in "
     "parallel with OT track finding and must finish before any track exists. So "
     "clusters are binned onto an ABSOLUTE eta-phi grid, and the only track-dependent "
     "step is a projection choosing which bin to read. THE BIN IS THE SELECTION: no "
     "cone cut is applied, and whatever landed in the bin is what a refit tests."),
    ("f  (bin width multiple)",
     "Bin width expressed as a multiple of that layer's q99 offset between the desired "
     "cluster and the NAIVE projection. f=1 means the bin is as wide as the 99th "
     "percentile offset. NAIVE because binning precedes the refit, so only the OT seed "
     "projection exists (308 um at L1, not the refit-order 37 um) -- sizing against the "
     "refit cone would be circular."),
    ("offset grids  (1grid / 2grid_phi / 4grid)",
     "A single grid leaves the projection at an arbitrary PHASE inside its bin, so a "
     "one-bin read guarantees NOTHING: a cluster just past the edge sits in the "
     "neighbour. Storing the same clusters in offset grids -- edges of one passing "
     "through centres of another -- lets the track pick the grid whose bin centre its "
     "projection lands nearest. Measured at matched containment (~90%, one read): "
     "1grid_1bin needs f=4 and gives 2293 combinations per track; 4grid_1bin needs f=2 and "
     "gives 55, a 42x improvement for 4x the storage. 2grid_phi_1bin buys almost nothing "
     "(2367) because the ETA phase is still arbitrary -- offsetting one dimension while "
     "the other stays misaligned does not help. Both offsets are needed."),
    ("reads  /  neighbour radius",
     "READS is how many bins are fetched per track per layer. The projection lands in "
     "exactly ONE bin; the NEIGHBOUR RADIUS says how many bins outward are fetched as "
     "well, so reads = (2*radius + 1)^2 -- radius 0 is a single bin, radius 1 is a 3x3 "
     "block, i.e. 9 reads. The ring exists because one grid guarantees nothing: a "
     "projection near a bin edge leaves the desired cluster in the neighbour, and "
     "reading the ring recovers it at 9x the bandwidth. Offset grids are the "
     "alternative -- they buy the same protection with 1 read and more storage. Which "
     "is preferable depends on whether bandwidth or memory binds."),
    ("max  (AVOID)",
     "Largest single-track value in the sample. An extreme-value statistic on a "
     "heavy tail: it DOES NOT CONVERGE and grows with the number of events "
     "processed (AIAI: 54 at 100 events, 550 at 500). Rankings built on it reverse "
     "when events are added. Use p99."),
    # ---- OT-only projection study (ot_projection.py) ---------------------
    ("OT-only projection",
     "The OT track's own helix and fit covariance projected onto the inner-tracker "
     "barrel with NO Kalman update and NO multiple-scattering term: what a refit "
     "knows before it has included any IT cluster. Each layer is projected "
     "independently from the OT fit, so IL1 is not informed by IL4. Computed "
     "outside the refit (ot_projection.py) and cross-checked against the "
     "producer's own seed projection (projSeed*) on every shared crossing."),
    ("layer projection",
     "One OT track projected to one IT barrel layer. It may span SEVERAL modules: "
     "the ellipse is evaluated on every module it overlaps, not only on the one the "
     "helix lands in, which removes the 'module boundary' bias above."),
    ("4-sigma ellipse",
     "Mahalanobis distance d < 4 in the module's local (x, y), with the 2x2 "
     "covariance J C J^T: C is the OT fit's helix covariance (L1TTrack_cov_*), J the "
     "numerical Jacobian of the local position with respect to (rInv, phi, tanL, "
     "z0, d0). Evaluated in the frame of the module each cluster is on. The OT "
     "covariance is OPTIMISTIC (see the pull widths printed with the study), so "
     "'4 sigma' of the claimed covariance is fewer true sigma."),
    ("majority-owner TP",
     "The TrackingParticle owning the most GENUINE stubs of the OT track: the "
     "particle the track is for. Its IT clusters are the TARGETS of the refit."),
    ("foreign stub",
     "A GENUINE stub (both clusters from one TP) whose TP is NOT the majority "
     "owner: a real piece of another particle on this track."),
    ("fake stub",
     "A stub that is not genuine: stub-level combinatoric (its two clusters come "
     "from different TPs) or unknown (no TP: noise, out-of-time pileup). Its TPs are "
     "not recorded, so a fake stub cannot name a co-owner. 'Combinatoric' is "
     "deliberately NOT used as a track class: CMSSW uses it for tracks, L1TOTStub for "
     "stubs, and they mean different things."),
    ("track classes  (perfect / foreign-1 / foreign-2+ / fake-1 / fake-2+ / foreign-fake)",
     "By the stubs the OT track is built from. perfect: every stub genuine and owned "
     "by the majority TP. foreign-1 / foreign-2+: 1 / 2 or more foreign stubs, no "
     "fake stub. fake-1 / fake-2+: 1 / 2 or more fake stubs, no foreign stub. "
     "foreign-fake: at least one of each. NB CMSSW's L1TTrack_genuine is NOT "
     "'perfect': it can carry ONE fake stub (1,224 of 17,882 genuine-flagged tracks "
     "on PU200 ttbar)."),
    ("excluded tracks  (tie / no-owner / unjoined / covariance absent)",
     "Not in any class, counted and printed with every result. tie: two TPs own the "
     "same, largest number of genuine stubs, so there is no target. no-owner: no "
     "genuine stub. unjoined: a stub whose truth could not be joined. covariance "
     "absent: L1TTrack_cov_* all zero."),
    ("A  (target)  /  A-in  /  A-out",
     "A cluster of the majority-owner TP anywhere on the layer. A-in lies inside the "
     "4-sigma ellipse, A-out outside; A-out residuals are taken on the target's own "
     "module, so they are frame-consistent too. Containment = fraction of layer "
     "projections with at least one A-in, among those where an A cluster exists."),
    ("B1 / B2 / B3  (background inside the ellipse)",
     "Every other cluster inside the 4-sigma ellipse. B1: owned by a TP that also "
     "owns a foreign stub of this track (the trap a contaminated track sets for "
     "itself). B2: owned by any other TP. B3: no TP (unlinked: mostly soft "
     "sub-0.1 GeV deposits, delta rays, curlers). CAVEAT: in these nanos (no "
     "noise-angle payload) B3 clusters carry NO angle estimate, so angle-using "
     "metrics never charge them an angle mismatch. Once angles are attached to "
     "them (a sensor-ADC angle regressor, or a smart-pixels estimate), B3 is a "
     "potential contaminant that these purities do not yet represent."),
    ("residual  (du, dv, dcotAlpha, dcotBeta)",
     "Cluster minus OT-only projection, in the local frame of the module the "
     "cluster is on, after projecting the track onto THAT module's plane: du, dv in "
     "um; dcotAlpha, dcotBeta unitless, cluster = the sensor's angle estimate "
     "(localCotAlpha/Beta), track = the helix direction taken through the same "
     "GeomDet::toLocal. Clusters without an angle estimate enter the position "
     "panels only."),
]


def _glossary_text(width=94, terms=None):
    out = []
    for term, body in GLOSSARY:
        if terms is not None and term not in terms:
            continue
        out.append(f"{term}")
        out.extend("    " + l for l in textwrap.wrap(body, width - 4))
        out.append("")
    return "\n".join(out).rstrip()


def write_glossary(outdir):
    txt = ("SmartPixels combinatorics omnibus -- GLOSSARY\n"
           "Definitions for every term this script prints or plots.\n"
           + "=" * 74 + "\n\n" + _glossary_text())
    p = os.path.join(outdir, "spix_glossary.txt")
    with open(p, "w") as fh:
        fh.write(txt + "\n")
    return p


def robust_sigma(a):
    """MAD-scaled width and the half 16-84 interval, as (mad, q68).

    Both are quoted because they disagree exactly when it matters: the
    cot(theta) residual has tails, and a plain std would chase them.
    """
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return float("nan"), float("nan")
    mad = 1.4826 * np.median(np.abs(a - np.median(a)))
    lo, hi = np.percentile(a, [16, 84])
    return float(mad), float(0.5 * (hi - lo))


# --------------------------------------------------------------------------
# loading + the (event, detId) join
# --------------------------------------------------------------------------
def _discover(path):
    with uproot.open(f"{path}:Events") as t:
        keys = set(t.keys())
    cfgs = sorted({m.group(1) for m in
                   (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys) if m})
    if not cfgs:
        raise SystemExit("no L1TSmartPixelsRefitHitDigiRefit* table in the input")
    # PICK THE MOST-INSTRUMENTED CONFIG, not the alphabetically last one. On a
    # multi-variant file (an activeSP sweep) sorted()[-1] is "IIIA" -- the build with
    # only L4 instrumented -- so every single-config study silently ran on the least
    # informative configuration in the file and reported three empty layers as
    # though that were the result. Sorting by count of "A" fixes it; ties break
    # alphabetically for reproducibility.
    chosen = sorted(cfgs, key=lambda c: (c.count("A"), c))[-1]
    if len(cfgs) > 1:
        print(f"  {len(cfgs)} activeSP variants present {cfgs}; single-config studies "
              f"use {chosen} (most instrumented). Study (10) covers all of them.")
    hit = f"L1TSmartPixelsRefitHitDigiRefit{chosen}"
    need_hit = ["trackIdx", "layer", "detId", "hitAccepted", "selHitClass",
                "projSeedLocalX", "projSeedLocalY", "projSeedSigX", "projSeedSigY",
                "projSeedCotAlpha", "projSeedCotBeta", "projLocalX", "projLocalY",
                "projCotAlpha", "projCotBeta", "projSigX", "projSigY",
                "recoLocalX", "recoLocalY", "selClusterIdx"]
    miss_h = [c for c in need_hit if f"{hit}_{c}" not in keys]
    miss_c = ([CLUSTER_TABLE] if not any(k.startswith(CLUSTER_TABLE + "_") for k in keys)
              else [c for c in ("layer", "detId", "localX", "localY", "charge",
                                "tpPt", "tpLocalCotAlpha", "tpLocalCotBeta", "tpIdx")
                    if f"{CLUSTER_TABLE}_{c}" not in keys])
    if miss_h or miss_c:
        raise SystemExit(
            "input cannot support this study.\n"
            f"  missing from {hit}: {miss_h or 'none'}\n"
            f"  missing from {CLUSTER_TABLE}: {miss_c or 'none'}\n\n"
            "Produce a Clusters-tier file:\n"
            "  test/makeSpixConfig.py --pu 200 --tier clusters-truth "
            "--variant digiRefit:1111 --needs-truth -o <out>.py")
    return hit, chosen, need_hit


def _ttc():
    """tracklet_topology_cost, loaded by path (the studies below do the same)."""
    import importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracklet_topology_cost.py")
    _sp = _ilu.spec_from_file_location("_ttc", _p)
    M = _ilu.module_from_spec(_sp); _sp.loader.exec_module(M)
    return M


def load(paths):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    hit, cfg, need_hit = _discover(files[0])
    ccols = ["layer", "detId", "localX", "localY", "charge", "tpPt",
             "tpLocalCotAlpha", "tpLocalCotBeta", "sizeX", "sizeY", "tpIdx"]
    # sensor angle + CPE sigma, present once the cluster table reads SmartPixelsRecHit
    optional = {"localCotAlpha": "clLocalCotAlpha", "localCotBeta": "clLocalCotBeta",
                "sigAlpha": "clSigAlpha", "sigBeta": "clSigBeta",
                "sigX": "sigX", "sigY": "sigY", "hasAlpha": "clHasAlpha",
                # global frame: cluster POSITION (globalR/Z/Phi) and the sensor's
                # estimate of the track DIRECTION there (gClPhi/gClCotTheta), with
                # rotated uncertainties. Study (7); absent in older files.
                "globalR": "globalR", "globalZ": "globalZ", "globalPhi": "globalPhi",
                "globalClusterPhi": "gClPhi",
                "globalClusterCotTheta": "gClCotTheta",
                "sigGlobalClusterPhi": "gSigPhi",
                "sigGlobalClusterCotTheta": "gSigCotTheta",
                "tpGlobalClusterPhi": "tpGClPhi",
                "tpGlobalClusterCotTheta": "tpGClCotTheta",
                "hasBeta": "clHasBeta"}
    rcols = ["pt", "eta"]

    H = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{hit}_{c}" for c in need_hit])
    C = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{CLUSTER_TABLE}_{c}" for c in ccols])
    R = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{REF}_{c}" for c in rcols])
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)

    # flatten crossings, tagging event index and the owning track's pt/eta
    ncross = ak.to_numpy(ak.num(H[f"{hit}_layer"]))
    ev_x = np.repeat(np.arange(len(ncross)), ncross)
    X = {c: ak.to_numpy(ak.flatten(H[f"{hit}_{c}"])) for c in need_hit}
    X["event"] = ev_x
    ntrk = ak.to_numpy(ak.num(R[f"{REF}_pt"]))
    off = np.concatenate([[0], np.cumsum(ntrk)])
    for c in rcols:
        flat = ak.to_numpy(ak.flatten(R[f"{REF}_{c}"]))
        X[f"trk_{c}"] = flat[off[ev_x] + X["trackIdx"].astype(np.int64)]

    vtrk = uproot.concatenate([f"{f}:Events" for f in files],
                              filter_name=[f"{hit.replace('RefitHit', 'Track')}_spixMatchedTpIdx"])
    vname = f"{hit.replace('RefitHit', 'Track')}_spixMatchedTpIdx"
    mtp = ak.to_numpy(ak.flatten(vtrk[vname]))
    nmt = ak.to_numpy(ak.num(vtrk[vname]))
    offm = np.concatenate([[0], np.cumsum(nmt)])
    X["trk_tpIdx"] = mtp[offm[ev_x] + X["trackIdx"].astype(np.int64)]

    ncl = ak.to_numpy(ak.num(C[f"{CLUSTER_TABLE}_layer"]))
    ev_c = np.repeat(np.arange(len(ncl)), ncl)
    K = {c: ak.to_numpy(ak.flatten(C[f"{CLUSTER_TABLE}_{c}"])) for c in ccols}
    with uproot.open(f"{files[0]}:Events") as t:
        avail = set(t.keys())
    got = [c for c in optional if f"{CLUSTER_TABLE}_{c}" in avail]
    if got:
        O = uproot.concatenate([f"{f}:Events" for f in files],
                               filter_name=[f"{CLUSTER_TABLE}_{c}" for c in got])
        for c in got:
            K[optional[c]] = ak.to_numpy(ak.flatten(O[f"{CLUSTER_TABLE}_{c}"]))
    K["event"] = ev_c
    K["_evt_base"] = np.concatenate([[0], np.cumsum(ncl)])
    # per-cluster TP truth at the POCA (tp_d0, tp_z0, ...) from the L1TTP table;
    # study (12) facets on tp_d0 and skips without it
    M = _ttc()
    if all(b in avail for b in M.tp_branches()):
        TPA = uproot.concatenate([f"{f}:Events" for f in files], filter_name=M.tp_branches())
        M.attach_tp_truth(K, M.tp_table(TPA, np.arange(len(ncl))))

    print(f"files={len(files)} config={cfg} events={n_ev} "
          f"crossings={len(X['layer'])} clusters={len(K['layer'])}")
    return hit, cfg, X, K, n_ev


def join_on_module(X, K):
    """Expand to (crossing, cluster) PAIRS sharing (event, detId).

    Fully vectorized: sort clusters by the packed key, searchsorted the crossing
    keys, then expand with repeat + a ramp. A python loop over ~70k crossings x
    ~40 clusters is avoidable and would dominate the runtime.
    """
    ckey = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    order = np.argsort(ckey, kind="stable")
    cs = ckey[order]
    xkey = X["event"].astype(np.int64) * (1 << 32) + X["detId"].astype(np.int64)
    lo = np.searchsorted(cs, xkey, "left")
    hi = np.searchsorted(cs, xkey, "right")
    n = hi - lo
    xi = np.repeat(np.arange(len(xkey)), n)                       # crossing index
    ramp = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)     # 0..n-1 per crossing
    ci = order[np.repeat(lo, n) + ramp]                            # cluster index
    return xi, ci, n


def prepare(X, K):
    xi, ci, n_on_module = join_on_module(X, K)
    P = {}
    P["xi"], P["ci"] = xi, ci
    P["n_on_module"] = n_on_module
    # displacement of each cluster from the SEED projection, in sigma
    dx = K["localX"][ci] - X["projSeedLocalX"][xi]
    dy = K["localY"][ci] - X["projSeedLocalY"][xi]
    sx = X["projSeedSigX"][xi]
    sy = X["projSeedSigY"][xi]
    good = (sx > 0) & (sy > 0) & (X["projSeedLocalX"][xi] > SENTINEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        P["nsigx"] = np.where(good, dx / sx, np.inf)
        P["nsigy"] = np.where(good, dy / sy, np.inf)
    P["good"] = good
    # Same displacement against the REFIT-ORDER projection: the running covariance
    # after multiple-scattering Q, before this layer's update. projSig* is present on
    # every crossing, including the ones whose window came up empty, so this arm is
    # not silently restricted to crossings that already found a hit.
    dxr = K["localX"][ci] - X["projLocalX"][xi]
    dyr = K["localY"][ci] - X["projLocalY"][xi]
    sxr, syr = X["projSigX"][xi], X["projSigY"][xi]
    good_rf = (sxr > 0) & (syr > 0) & (X["projLocalX"][xi] > SENTINEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        P["nsigx_rf"] = np.where(good_rf, dxr / sxr, np.inf)
        P["nsigy_rf"] = np.where(good_rf, dyr / syr, np.inf)
    P["good_rf"] = good_rf
    # angle mismatch vs the track's expectation at this module
    P["dCotA"] = K["tpLocalCotAlpha"][ci] - X["projSeedCotAlpha"][xi]
    P["dCotB"] = K["tpLocalCotBeta"][ci] - X["projSeedCotBeta"][xi]
    P["ang_ok"] = (K["tpLocalCotAlpha"][ci] > SENTINEL) & (X["projSeedCotAlpha"][xi] > SENTINEL)
    P["sel_gidx"] = selected_global_index(X, K)
    # over PAIRS: is this pair the crossing's selected cluster?
    P["is_selected"] = (P["sel_gidx"][P["xi"]] >= 0) & (P["ci"] == P["sel_gidx"][P["xi"]])
    return P


def in_cone(P, k):
    return P["good"] & (np.abs(P["nsigx"]) < k) & (np.abs(P["nsigy"]) < k)


def in_cone_refit(P, k):
    return P["good_rf"] & (np.abs(P["nsigx_rf"]) < k) & (np.abs(P["nsigy_rf"]) < k)


def selected_global_index(X, K):
    """Global cluster-table row of each crossing's SELECTED cluster, or -1.

    Uses the EXACT selClusterIdx link. Position matching was tried first and is
    unsafe: both tables store coordinates at 10-bit nano mantissa precision, which
    disagrees by up to 9.7 um against a 25 um pitch, so it can silently pick a
    neighbouring cluster.
    """
    g = np.where(X["selClusterIdx"] >= 0,
                 K["_evt_base"][X["event"]] + X["selClusterIdx"].astype(np.int64), -1)
    ok = g >= 0
    if ok.any():
        bad = int((K["detId"][g[ok]] != X["detId"][ok]).sum())
        if bad:
            raise SystemExit(
                f"selClusterIdx integrity check FAILED on {bad} crossings: the cluster it "
                "points at is on a different module. The refit and cluster tables have "
                "diverged in filter or iteration order; do not trust any result from this file.")
    return g


# --------------------------------------------------------------------------
# studies
# --------------------------------------------------------------------------
def study_cone_occupancy(X, K, P, ax_row, out):
    """(1) clusters inside the seed cone, per layer and vs pT / eta."""
    xlay = X["layer"]
    res = {}
    ax = ax_row[0]
    for cname, k in CONES.items():
        m = in_cone(P, k)
        cnt = np.bincount(P["xi"][m], minlength=len(xlay)).astype(float)
        per_layer = [cnt[xlay == L].mean() if (xlay == L).any() else np.nan for L in LAYERS]
        res[cname] = {"per_layer_mean": [float(v) for v in per_layer]}
        ax.plot(LAYERS, per_layer, marker="o", label=f"seed cone {cname}")
        P[f"cnt_{cname}"] = cnt
    tot = np.bincount(P["xi"], minlength=len(xlay)).astype(float)
    ax.plot(LAYERS, [tot[xlay == L].mean() for L in LAYERS], marker="s", ls="--",
            color="k", label="all on module")
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer [index]"); ax.set_ylabel("clusters per crossing [count]")
    ax.set_title("(1) candidates per crossing"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    res["all_on_module_per_layer"] = [float(tot[xlay == L].mean()) for L in LAYERS]

    for ax, var, edges, lab in ((ax_row[1], "trk_pt", PT_EDGES, r"track $p_T$ [GeV]"),
                                (ax_row[2], "trk_eta", ETA_EDGES, r"track $\eta$")):
        v = X[var]
        idx = np.digitize(v, edges) - 1
        for cname in CONES:
            cnt = P[f"cnt_{cname}"]
            ys = [cnt[idx == i].mean() if (idx == i).any() else np.nan
                  for i in range(len(edges) - 1)]
            ctr = [0.5 * (edges[i] + min(edges[i + 1], 40)) for i in range(len(edges) - 1)]
            ax.plot(ctr, ys, marker="o", label=cname)
            res.setdefault(f"vs_{var}", {})[cname] = {
                "bin_centres": [float(c) for c in ctr], "mean": [float(y) for y in ys]}
        ax.set_xlabel(lab); ax.set_ylabel("clusters in cone per crossing [count]")
        ax.set_title(f"(1) cone occupancy vs {lab}"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["cone_occupancy"] = res


def study_cone_containment(X, K, P, ax_row, out):
    """(2) does the cone contain the cluster the track actually needs?

    CONDITIONING, stated because it biases the number upward: the only clusters
    known to belong to this track are the ones the refit SELECTED and truth
    labelled selHitClass==0. A truth cluster that the current static window never
    offered cannot appear here. An unbiased efficiency needs a TP identifier on
    every cluster row, which the tier does not yet carry -- listed in the writeup.
    """
    ok = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0) & (X["recoLocalX"] > SENTINEL) \
         & (X["projSeedSigX"] > 0) & (X["projSeedLocalX"] > SENTINEL)
    res = {"n_reference_hits": int(ok.sum()), "conditioning":
           "selected-and-truth-correct hits only; biased upward, see docstring"}
    ax = ax_row[0]
    for cname, k in CONES.items():
        nx = np.abs(X["recoLocalX"] - X["projSeedLocalX"]) / np.where(X["projSeedSigX"] > 0, X["projSeedSigX"], np.nan)
        ny = np.abs(X["recoLocalY"] - X["projSeedLocalY"]) / np.where(X["projSeedSigY"] > 0, X["projSeedSigY"], np.nan)
        inside = ok & (nx < k) & (ny < k)
        eff = [inside[ok & (X["layer"] == L)].mean() if (ok & (X["layer"] == L)).any() else np.nan
               for L in LAYERS]
        ax.plot(LAYERS, eff, marker="o", label=cname)
        res[cname] = {"per_layer_eff": [float(e) for e in eff],
                      "overall_eff": float(inside[ok].mean()) if ok.any() else float("nan")}
    # The cone is a per-coordinate BOX cut at k sigma, so for an ideal Gaussian
    # projection the JOINT containment is (2*Phi(k)-1)^2, not 2*Phi(k)-1. Drawing
    # 0.68/0.95 here would make a correctly-sized cone look broken.
    from math import erf, sqrt
    for k, c in ((CONES["q68"], "grey"), (CONES["q95"], "grey")):
        ideal = erf(k / sqrt(2.0)) ** 2
        ax.axhline(ideal, color=c, ls=":", lw=1)
        ax.text(4.05, ideal, f" ideal {ideal:.3f}", fontsize=6, va="center", color=c)
        res.setdefault("ideal_box_containment", {})[f"k={k:.2f}"] = float(ideal)
    ax.set_xlabel("TBPX layer [index]"); ax.set_ylabel("containment efficiency [fraction]")
    ax.set_title("(2) correct hit inside the seed cone"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_ylim(0, 1.05)

    # cone size itself, the thing that drives (1) and (2) together
    ax = ax_row[1]
    for nm, key in (("$\\sigma_x$", "projSeedSigX"), ("$\\sigma_y$", "projSeedSigY")):
        v = [np.median(X[key][(X["layer"] == L) & (X[key] > 0)]) * 1e4 for L in LAYERS]
        ax.plot(LAYERS, v, marker="o", label=nm)
        res.setdefault("cone_sigma_um", {})[key] = [float(x) for x in v]
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer [index]"); ax.set_ylabel(r"median cone $\sigma$ [$\mu$m]")
    ax.set_title("(2) seed cone size"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["cone_containment"] = res


def study_angle_discrimination(X, K, P, ax_row, out):
    """(3) how much of the in-cone pool could an alpha/beta cut remove?

    UPPER BOUND, not a capability: the cluster angles here are UNSMEARED truth
    (tpLocalCotAlpha/Beta). A real smart-pixel angle carries the PixelAV response
    smear, so achievable rejection is strictly worse than this.
    """
    m = in_cone(P, CONES["q95"]) & P["ang_ok"]
    correct = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0)
    same = P["is_selected"] & correct[P["xi"]]
    res = {"note": "unsmeared truth angles: an UPPER BOUND on angle rejection"}
    for ax, d, nm in ((ax_row[0], P["dCotA"], r"$\Delta\cot\alpha$"),
                      (ax_row[1], P["dCotB"], r"$\Delta\cot\beta$")):
        a_sig, a_bkg = d[m & same], d[m & ~same]
        rng = np.nanpercentile(np.abs(d[m]), 99) if m.any() else 1.0
        bins = np.linspace(-rng, rng, 80)
        for arr, lab in ((a_bkg, "other clusters in cone"), (a_sig, "the track's own cluster")):
            if len(arr) > 10:
                ax.hist(arr, bins=bins, histtype="step", density=True, label=f"{lab} (n={len(arr)})")
        ax.set_xlabel(nm); ax.set_ylabel("density [1/bin]"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title(f"(3) {nm} vs track expectation")
        if len(a_sig) > 10 and len(a_bkg) > 10:
            for q in (0.68, 0.95):
                cut = np.quantile(np.abs(a_sig), q)
                res.setdefault(nm, {})[f"keep{int(q*100)}_cut"] = float(cut)
                res[nm][f"keep{int(q*100)}_bkg_rejected"] = float(np.mean(np.abs(a_bkg) > cut))
    out["angle_discrimination"] = res


def study_charge_gate(X, K, P, ax_row, out):
    """(4) the READOUT gate: only the N highest-charge clusters per module can be
    formed into L1 outputs; the rest wait for a Level-1 Accept and are useless to
    the trigger. So: what charge rank does the track's own cluster sit at?"""
    ci, xi = P["ci"], P["xi"]
    # rank each cluster within its module by descending charge
    key = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    order = np.lexsort((-K["charge"].astype(np.float64), key))
    sk = key[order]
    starts = np.r_[True, sk[1:] != sk[:-1]]
    grp = np.cumsum(starts) - 1
    pos = np.arange(len(sk)) - np.flatnonzero(starts)[grp]
    rank = np.empty(len(key), dtype=np.int64)
    rank[order] = pos

    correct = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0)
    match = P["is_selected"] & correct[xi]
    r = rank[ci][match]
    res = {"n_matched": int(match.sum())}
    ax = ax_row[0]
    if len(r) > 10:
        maxr = int(np.quantile(r, 0.99)) + 1
        ax.hist(r, bins=np.arange(0, max(maxr, 8) + 1) - 0.5, histtype="stepfilled", alpha=.7)
        ax.set_xlabel("charge rank within module [index, 0 = highest]")
        ax.set_ylabel("truth-correct hits [count]")
        ax.set_title("(4) where the needed cluster sits in charge order")
        ax.grid(alpha=.3)
        keep = {N: float(np.mean(r < N)) for N in (1, 2, 4, 8, 16, 32)}
        res["survival_vs_topN"] = keep
        ax2 = ax_row[1]
        Ns = sorted(keep)
        ax2.plot(Ns, [keep[N] for N in Ns], marker="o")
        ax2.set_xscale("log", base=2); ax2.set_xlabel("clusters read out per module, top-N by charge [count]")
        ax2.set_ylabel("needed clusters kept [fraction]")
        ax2.set_title("(4) readout gate efficiency"); ax2.grid(alpha=.3); ax2.set_ylim(0, 1.05)
    out["charge_gate"] = res


def study_true_containment(X, K, P, ax_row, out):
    """(5) UNBIASED containment: of the clusters that genuinely belong to this
    track's TrackingParticle and sit on the module it crosses, how many does the
    cone hold?

    This is the study that selClusterIdx and tpIdx were added for. Study (2)
    can only ever see clusters the current static window already offered, so it
    cannot detect a true cluster the window never showed the fit -- exactly the
    failure a cone redesign is meant to fix. Here the truth clusters are found by
    joining tpIdx to the track's spixMatchedTpIdx, independently of what the
    window did, so a cluster the window missed still counts against the cone.
    """
    xi, ci = P["xi"], P["ci"]
    have = (X["trk_tpIdx"][xi] >= 0) & (K["tpIdx"][ci] >= 0)
    is_true = have & (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    res = {"n_true_pairs": int(is_true.sum())}
    ax = ax_row[0]
    for cname, k in CONES.items():
        m = is_true & P["good"]
        if m.sum() < 20:
            continue
        inside = m & in_cone(P, k)
        eff = []
        for L in LAYERS:
            sel = m & (X["layer"][xi] == L)
            eff.append(float(inside[sel].mean()) if sel.any() else np.nan)
        ax.plot(LAYERS, eff, marker="o", label=f"{cname} (unbiased)")
        res[cname] = {"per_layer_eff": eff, "overall_eff": float(inside[m].mean())}
    # the biased version, for direct contrast on the same axes
    bc = out.get("cone_containment", {})
    for cname in CONES:
        if cname in bc:
            ax.plot(LAYERS, bc[cname]["per_layer_eff"], marker="s", ls="--", alpha=.6,
                    label=f"{cname} (window-conditioned)")
    ax.set_xlabel("TBPX layer [index]"); ax.set_ylabel("containment efficiency [fraction]")
    ax.set_title("(5) TRUE containment vs window-conditioned")
    ax.legend(fontsize=6); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)

    # how many true clusters does a track even have on the module it crosses?
    ax = ax_row[1]
    n_true = np.bincount(xi[is_true], minlength=len(X["layer"]))
    for L in LAYERS:
        sel = X["layer"] == L
        if sel.any():
            ax.plot(L, n_true[sel].mean(), marker="o", color="C0")
    res["mean_true_clusters_on_module"] = [
        float(n_true[X["layer"] == L].mean()) if (X["layer"] == L).any() else float("nan")
        for L in LAYERS]
    ax.plot(LAYERS, res["mean_true_clusters_on_module"], color="C0")
    ax.set_xlabel("TBPX layer"); ax.set_ylabel("true clusters on crossed module [count]")
    ax.set_title("(5) how many are there to find"); ax.grid(alpha=.3)
    out["true_containment"] = res


def study_chi2_weight_scan(X, K, P, ax_row, out):
    """(6) WHICH selection chi2 picks the cleanest hits?

    The refit currently selects the candidate minimising

        sel = (dx/sigx)^2 + (dy/sigy)^2 + [alpha] (dcotA/sigA)^2 + [beta] (dcotB/sigB)^2

    i.e. all four terms at unit weight. That is a choice, not a derivation: the
    angle terms come from a sensor estimator whose resolution is not commensurate
    with the CPE position resolution, and the r-phi and r-z terms are not equally
    informative either. This scans the weights and asks which combination picks
    the TRUTH-CORRECT cluster most often.

    THE FIGURE OF MERIT is per-crossing hit-selection purity: of the crossings
    where the correct cluster is present in the window at all, how often does the
    weighted metric rank it first. That is the quantity the refit's parameter
    resolution is downstream of -- measured earlier, one wrong hit annihilates the
    refit gain, and the whole outsideIn win came from selection rather than from
    fitting.

    Scans, as requested:
      * coarse over (w_rphi, w_rz) applied to the POSITION terms, from the
        physics-informed default (1, 1);
      * 1D over w_alpha alone (bending angle only, the beta term off);
      * 2D over (w_alpha, w_beta).

    The angle terms use the SENSOR estimate and its sigma, not truth, so the rule
    itself is deployable -- unlike anything scanned on truth angles.

    HISTORY WORTH KEEPING. This scan was meaningless until 2026-09-05, because
    SmartPixelsRecHitProducer gave NO angle to clusters with no simlink: hasAlpha
    was 99.5% for TP-linked clusters and 0.0% for unlinked ones, so "reports an
    angle" was a perfect proxy for "is real" and any angle weight bought that
    proxy rather than angle information. The scan now runs a LEAK SELF-CHECK on
    every invocation and says so if the two rates diverge again -- a measured
    guard, not a comment that can go stale.
    """
    xi, ci = P["xi"], P["ci"]
    need = ("clLocalCotAlpha", "clLocalCotBeta", "clSigAlpha", "clSigBeta")
    if not all(k in K for k in need):
        print("   (6) SKIPPED: cluster table lacks the sensor angle columns "
              "(localCotAlpha/localCotBeta/sigAlpha/sigBeta)")
        out["chi2_weight_scan"] = {"skipped": "cluster table has no sensor angle columns"}
        for a in ax_row:
            a.axis("off")
        return

    # per-pair residuals against the RUNNING projection (what the refit compares to)
    dx = (K["localX"][ci] - X["projLocalX"][xi]) / np.maximum(K["sigX"][ci], 1e-6)
    dy = (K["localY"][ci] - X["projLocalY"][xi]) / np.maximum(K["sigY"][ci], 1e-6)
    okA = (K["clLocalCotAlpha"][ci] > SENTINEL) & (K["clSigAlpha"][ci] > 0) \
          & (X["projCotAlpha"][xi] > SENTINEL)
    okB = (K["clLocalCotBeta"][ci] > SENTINEL) & (K["clSigBeta"][ci] > 0) \
          & (X["projCotBeta"][xi] > SENTINEL)
    # A cluster whose sensor reports NO angle must be neither rewarded nor punished
    # for it. Filling its normalized residual with 0 (the naive choice) makes it
    # cost-free, so any large angle weight simply selects angle-less clusters -- an
    # artefact that made angle-only selection look catastrophic (0.064) and made
    # huge weights look beneficial. The unbiased fill is the EXPECTATION of a
    # normalized residual squared, i.e. 1, so a missing term contributes its mean
    # and the comparison stays fair across candidates with different term counts.
    NEUTRAL = 1.0
    da2 = np.where(okA, ((K["clLocalCotAlpha"][ci] - X["projCotAlpha"][xi])
                         / np.maximum(K["clSigAlpha"][ci], 1e-9)) ** 2, NEUTRAL)
    db2 = np.where(okB, ((K["clLocalCotBeta"][ci] - X["projCotBeta"][xi])
                         / np.maximum(K["clSigBeta"][ci], 1e-9)) ** 2, NEUTRAL)

    # the correct cluster for each crossing, from the TP join (window-independent)
    have = (X["trk_tpIdx"][xi] >= 0) & (K["tpIdx"][ci] >= 0)
    is_true = have & (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    ncross = len(X["layer"])
    has_true = np.zeros(ncross, dtype=bool)
    has_true[xi[is_true]] = True
    inwin = P["good"]

    def purity(wrphi, wrz, wa, wb):
        """Fraction of crossings whose lowest-cost candidate is the correct one."""
        cost = wrphi * dx * dx + wrz * dy * dy + wa * da2 + wb * db2
        cost = np.where(inwin, cost, np.inf)
        best = np.full(ncross, np.inf)
        np.minimum.at(best, xi, cost)
        picked_true = np.zeros(ncross, dtype=bool)
        sel = np.isfinite(cost) & (cost <= best[xi]) & is_true
        picked_true[xi[sel]] = True
        d = has_true & np.isfinite(best)
        return float(picked_true[d].mean()) if d.any() else float("nan"), int(d.sum())

    # SCALE INVARIANCE: the argmin is unchanged by a global rescaling, so only
    # RATIOS matter and w_rphi is pinned to 1. Four weights therefore have THREE
    # free parameters, and they are scanned JOINTLY.
    #
    # The earlier version of this scan was wrong and is worth recording. It
    # optimised GREEDILY: first w_rz at unit angle weights, then the angle weights
    # at that frozen w_rz. That silently assumes the best position balance does not
    # depend on how much the angles are trusted, which is false -- once the angles
    # dominate, the position terms are a tiebreak and their optimal ratio changes.
    # It also mislabelled "position + alpha" as an "alpha-only scan" while the
    # LIMITS block used "alpha only" for alpha with NO position, so two different
    # things carried the same name.
    res = {"note": "w_rphi pinned to 1 (scale invariance); the remaining three weights "
                   "are scanned JOINTLY, not greedily"}
    base, n_den = purity(1, 1, 1, 1)
    res["baseline_all_unit_weights"] = {"purity": base, "n_crossings": n_den}
    print(f"   (6) baseline, all four terms at unit weight: purity {base:.4f} "
          f"over {n_den} crossings")

    rzgrid = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    # The angle-weight grid runs to 1e6 rather than stopping at 4096 because it
    # RAILED there: the earlier scan reported its optimum sitting exactly on the
    # ceiling, which means the optimum was outside the range and the reported
    # weights were an artefact of where the grid stopped. A scan that ends on its
    # own boundary has not found a maximum, it has found an edge. Extending far
    # enough to see the purity PLATEAU is what distinguishes "the angles should be
    # weighted very heavily" from "we never looked far enough".
    wgrid = [0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0,
             16384.0, 65536.0, 262144.0, 1048576.0]

    # 3D scan: (w_rphi=1, w_rz, w_alpha), beta OFF. The bending angle only, but
    # the POSITION terms are scanned with it rather than frozen.
    s3 = {}
    for wz in rzgrid:
        for wa in wgrid:
            s3[f"{wz}_{wa}"] = purity(1.0, wz, wa, 0.0)[0]
    b3 = max(s3, key=s3.get)
    res["scan3D_rphi_rz_alpha"] = s3
    res["best3D"] = {"w_rz": float(b3.split("_")[0]), "w_alpha": float(b3.split("_")[1]),
                     "purity": s3[b3]}
    print(f"       3D (w_rz, w_alpha; beta off) best {b3} -> {s3[b3]:.4f}")

    # 4D scan: (w_rphi=1, w_rz, w_alpha, w_beta). Fully joint.
    s4 = {}
    for wz in rzgrid:
        for wa in wgrid:
            for wb in wgrid:
                s4[f"{wz}_{wa}_{wb}"] = purity(1.0, wz, wa, wb)[0]
    b4 = max(s4, key=s4.get)
    wz4, wa4, wb4 = (float(v) for v in b4.split("_"))
    res["scan4D_rphi_rz_alpha_beta"] = s4
    res["best4D"] = {"w_rphi": 1.0, "w_rz": wz4, "w_alpha": wa4, "w_beta": wb4,
                     "purity": s4[b4]}
    print(f"       4D (w_rz, w_alpha, w_beta) best {b4} -> {s4[b4]:.4f}   "
          f"(baseline {base:.4f}, gain {s4[b4]-base:+.4f})")
    # Does the 4D optimum contain the 3D one? If the best w_rz differs between the
    # two, the greedy version could not have found this point.
    if res["best3D"]["w_rz"] != wz4:
        print(f"       NOTE best w_rz differs between the 3D ({res['best3D']['w_rz']}) and "
              f"4D ({wz4}) scans, which is exactly what the greedy version could not see")
    wr0, wz0 = 1.0, wz4
    a1 = {str(w): s3[f"{wz4}_{w}"] for w in wgrid}   # alpha slice at the 4D-best w_rz
    a2 = {f"{wa}_{wb}": s4[f"{wz4}_{wa}_{wb}"] for wa in wgrid for wb in wgrid}

    # ABLATIONS: each row uses a STRICT SUBSET of the four terms, so a lower purity
    # means that subset carries less information -- NOT that adding information hurt.
    # Ordered smallest subset first so the monotone build-up is visible. These are a
    # different thing from the SCANS above, which always keep the position terms and
    # vary only the weights.
    lim = [("alpha alone              (0,0,1,0)", purity(0.0, 0.0, 1.0, 0.0)[0]),
           ("beta alone               (0,0,0,1)", purity(0.0, 0.0, 0.0, 1.0)[0]),
           ("both angles, no position (0,0,1,1)", purity(0.0, 0.0, 1.0, 1.0)[0]),
           ("position alone           (1,wz,0,0)", purity(1.0, wz0, 0.0, 0.0)[0]),
           ("ALL four, unit weights   (1,1,1,1)", base),
           ("ALL four, weights tuned  (4D best)", s4[b4])]
    res["ablations"] = {k.strip(): v for k, v in lim}
    print("       ABLATIONS -- each row is a STRICT SUBSET of the four terms, so purity")
    print("       RISES as information is added. Subsets, not regressions:")
    for k, v in lim:
        print(f"         {k:<38} {v:.4f}")

    # LEAK SELF-CHECK, measured not asserted. Before the noise payload existed,
    # unlinked clusters carried NO angle (hasAlpha 99.5% TP-linked vs 0.0%
    # unlinked), so "reports an angle" was a perfect proxy for "is real" and any
    # angle weight bought that proxy. Recomputed every run so it cannot go stale.
    if "clHasAlpha" in K:
        lk = K["tpIdx"] >= 0
        ra, rb = float(K["clHasAlpha"][lk].mean()), float(K["clHasAlpha"][~lk].mean())
        res["angle_presence_TPlinked"], res["angle_presence_unlinked"] = ra, rb
        if abs(ra - rb) > 0.05:
            print(f"       *** CONFOUNDED: hasAlpha {100*ra:.1f}% TP-linked vs {100*rb:.1f}% "
                  "unlinked -- angle weight partly buys that proxy. Do not tune on this.")
            res["CONFOUNDED"] = f"angle presence differs by {abs(ra-rb):.3f} between classes"
        else:
            print(f"       leak self-check OK: hasAlpha {100*ra:.1f}% TP-linked vs "
                  f"{100*rb:.1f}% unlinked, so angle presence carries no class information")
    out["chi2_weight_scan"] = res

    ax = ax_row[0]
    for wz in rzgrid:
        ax.plot(wgrid, [s3[f"{wz}_{w}"] for w in wgrid], marker=".",
                label=rf"$w_{{rz}}$={wz}")
    ax.axhline(base, color="k", ls=":", label="baseline (1,1,1,1)")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"$w_\alpha$ [unitless]   ($w_{r\phi}\equiv1$, $w_\beta=0$)")
    ax.set_ylabel("hit-selection purity [fraction]")
    ax.set_title("(6) 3D scan: position terms NOT frozen")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

    ax = ax_row[1]
    M2 = np.array([[a2[f"{wa}_{wb}"] for wb in wgrid] for wa in wgrid])
    im2 = ax.imshow(M2, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(wgrid))); ax.set_xticklabels(wgrid, rotation=45, fontsize=7)
    ax.set_yticks(range(len(wgrid))); ax.set_yticklabels(wgrid, fontsize=7)
    ax.set_xlabel(r"$w_\beta$ [unitless]"); ax.set_ylabel(r"$w_\alpha$ [unitless]")
    ax.set_title("(6) angle-weight scan (1D = bottom row)"); plt.colorbar(im2, ax=ax)



def study_refit_cone_occupancy(X, K, P, ax_row, out):
    """(9) Candidates per crossing: naive SEED cone vs outsideIn REFIT-ORDER cone.

    THIS IS THE HALF OF THE COMBINATORICS QUESTION THAT WAS NEVER BUILT. Study (1)
    projects with the OT seed covariance -- one shot, no updates -- which answers
    "how many clusters would a naive projection have to test". The actual refit
    walks outsideIn and tightens its covariance at every layer, so the number it
    must test is smaller, and by how much is the thing that decides whether the
    per-track combinatorics are affordable. Nothing measured it until projSig*
    existed.

    IT ONLY BECAME MEANINGFUL AFTER THE Q TERM. Before process noise the running
    covariance was up to 2.7x too tight (correct-hit pull width 4.13 at L1), so a
    refit-order cone would have looked spectacularly better than the seed cone for
    the worst possible reason -- it was lying about its own precision and would
    have thrown away real hits. Any number from this study taken before Q is not a
    physics result, it is the bug.

    The reduction reported here is therefore an HONEST one: the covariance now
    passes a two-sided pull test (see eval_refitq/windows/q_acceptance.py).
    """
    xlay = X["layer"]
    res = {}
    ax = ax_row[0]
    for cname, k in CONES.items():
        ms, mr = in_cone(P, k), in_cone_refit(P, k)
        cs = np.bincount(P["xi"][ms], minlength=len(xlay)).astype(float)
        cr = np.bincount(P["xi"][mr], minlength=len(xlay)).astype(float)
        P[f"cnt_rf_{cname}"] = cr
        ys = [cs[xlay == L].mean() if (xlay == L).any() else np.nan for L in LAYERS]
        yr = [cr[xlay == L].mean() if (xlay == L).any() else np.nan for L in LAYERS]
        ax.plot(LAYERS, ys, marker="o", ls="--", label=f"seed {cname}")
        ax.plot(LAYERS, yr, marker="s", label=f"refit-order {cname}")
        res[cname] = {"seed_per_layer": [float(v) for v in ys],
                      "refit_per_layer": [float(v) for v in yr],
                      "reduction_per_layer": [float(a / b) if b > 0 else float("nan")
                                              for a, b in zip(ys, yr)]}
    tot = np.bincount(P["xi"], minlength=len(xlay)).astype(float)
    ax.plot(LAYERS, [tot[xlay == L].mean() for L in LAYERS], marker="^", ls=":",
            color="k", label="all on module")
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer [index]")
    ax.set_ylabel("candidates per crossing [count]")
    ax.set_title("(9) seed cone vs refit-order cone")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # cone half-width itself, in microns: the mechanism behind the count above
    ax = ax_row[1]
    hs, hr = [], []
    for L in LAYERS:
        m = xlay == L
        a = X["projSeedSigX"][m]; b = X["projSigX"][m]
        hs.append(float(np.median(a[a > 0]) * 1e4) if (a > 0).any() else np.nan)
        hr.append(float(np.median(b[b > 0]) * 1e4) if (b > 0).any() else np.nan)
    ax.plot(LAYERS, hs, marker="o", ls="--", label="seed sigma_x")
    ax.plot(LAYERS, hr, marker="s", label="refit-order sigma_x")
    ax.set_xlabel("TBPX layer [index]"); ax.set_ylabel(r"median projected $\sigma_x$ [$\mu$m]")
    ax.set_title("(9) projection cone half-width"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    res["median_sigma_x_um"] = {"seed": hs, "refit": hr}

    # does the gain survive at low pT, where scattering is worst?
    ax = ax_row[2]
    idx = np.digitize(X["trk_pt"], PT_EDGES) - 1
    ctr = [0.5 * (PT_EDGES[i] + min(PT_EDGES[i + 1], 40)) for i in range(len(PT_EDGES) - 1)]
    for cname in CONES:
        cs = P[f"cnt_{cname}"] if f"cnt_{cname}" in P else None
        cr = P[f"cnt_rf_{cname}"]
        if cs is None:
            continue
        ratio = []
        for i in range(len(PT_EDGES) - 1):
            m = idx == i
            if not m.any():
                ratio.append(np.nan); continue
            a, b = cs[m].mean(), cr[m].mean()
            ratio.append(float(a / b) if b > 0 else np.nan)
        ax.plot(ctr, ratio, marker="o", label=cname)
        res.setdefault("reduction_vs_pt", {})[cname] = {
            "bin_centres": [float(c) for c in ctr], "ratio": ratio}
    ax.axhline(1.0, color="k", lw=.8, ls=":")
    ax.set_xlabel(r"track $p_T$ [GeV]"); ax.set_ylabel("seed / refit-order candidates [ratio, unitless]")
    ax.set_title("(9) combinatorics reduction vs $p_T$"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    print("\n=== (9) refit-order cone vs naive seed cone ===")
    print(f"  median projected sigma_x [um] by layer")
    print(f"    seed        : " + "  ".join(f"{v:7.1f}" for v in hs))
    print(f"    refit-order : " + "  ".join(f"{v:7.1f}" for v in hr))
    for cname in CONES:
        r = res[cname]
        print(f"  {cname}: candidates/crossing seed "
              + "/".join(f"{v:.2f}" for v in r["seed_per_layer"])
              + "  refit " + "/".join(f"{v:.2f}" for v in r["refit_per_layer"])
              + "  reduction " + "/".join(f"{v:.2f}x" for v in r["reduction_per_layer"]))
    out["refit_cone_occupancy"] = res


def study_z0_resolution(X, K, P, ax_row, out):
    """(7) sigma(cot theta), and the z0 resolution it implies.

    THE QUESTION. A cluster measures both a POSITION (globalR, globalZ) and a
    DIRECTION (globalClusterCotTheta), so on its own it determines a longitudinal
    impact parameter:

        z0 = z - r * cot(theta)         =>   sigma(z0) = r * sigma(cot theta)

    If that z0 is sharp, a seeding Hough transform gains a third nearly free
    dimension: clusters can be sliced by z0 BEFORE the (phi0, q/pT) transform
    runs, and since random k-layer coincidences scale as the per-cell density to
    the k-th power, the fake rate falls as the CUBE of the slicing factor for a
    3-layer seed. `doc/SmartPixelsSeedingAndFitting.md` (sections 6, 11, 12)
    makes this the single largest factor in the design and flags it as ASSUMED.
    This measures it.

    THE SUPPRESSION FACTOR is Z_range / (4 sigma), not Z_range / (slice width).
    A hit must vote into every slice its z0 could belong to, so it occupies
    ~4 sigma / w of the w-wide slices and the w cancels. Choosing a fine slicing
    buys nothing on its own -- only a small sigma does.

    *** THIS RESIDUAL IS CIRCULAR. IT IS NOT A PHYSICAL RESOLUTION. ***
    SmartPixelsRecHitProducer builds the truth angle as the dominant TP's helix
    PROPAGATED TO THE HIT, with no multiple scattering (:292-319, and the file's
    own header at :32-40), then forms the reco angle as
    `cotB = trueCotB + corrBetaShift_->evaluate(...)` (:330-331). The nano truth
    column is that same trueCotBeta() rotated to global
    (L1SmartPixelsClusterTableProducer.cc:199-202). So (reco - truth) is
    IDENTICALLY the PixelAV payload draw, and what this study measures is the
    payload's own width, not how well a sensor knows a real track's direction.
    The tell is in the output below: sigma comes out equal for pT > 2 and for
    0.5-2 GeV to 0.3%, where a physical resolution would degrade toward low pT.

    What is missing is the scattering between the vertex and the module. It is
    not fatal for z0 -- a kink at radius r_s moves the EXTRAPOLATED z0 by
    r_s * delta(cot theta), not r * delta, so material outside the measurement
    radius does not bias it, and L1 has almost nothing inside it. Estimated
    inflation is +0% (L1, eta 0) to +27% (L4, eta 2, 0.5 GeV). See
    doc/SmartPixelsSeedingAndFitting.md section 4a.

    THE NEXT STEP is a closure test: with the dominant TP's production vz on the
    cluster truth block, compare z0_pred = globalZ - globalR * globalClusterCotTheta
    against the TP's actual z0. That is non-circular, and it is UNIFORM across PU
    and signal (TrackingParticles are post-mixing; 86.5% of clusters carry one).
    It gives a PESSIMISTIC BOUND rather than the true resolution, because the
    simulated cluster pairs a scattered position with an unscattered angle: the
    residual carries +(r - r_s)*delta where the physical term is -r_s*delta. Taken
    together with the number below it brackets the answer, which is enough to
    design against.

    NOT PSimHit. It is the only scattering-aware angle available, but it is
    signal-only at ~1.7% of PU200 clusters, so any resolution derived from it
    describes a population with different pT and eta spectra from pileup and would
    carry that asymmetry into everything downstream. SmartPixelsRecHitProducer
    already refuses PSimHit as a production input for this reason (:36-39); the
    same rule applies to using it as a validation reference.

    Resolving this EXACTLY needs the angle to come from the simulated cluster
    SHAPE for every cluster, rather than from helix-truth plus a payload draw.
    Until then, read every number below as the payload smear.

    METHOD. Robust widths only (MAD*1.4826 and the half 16-84 interval) -- the
    residual has tails a plain std would chase.

    BINNING IN ETA IS NOT OPTIONAL. cot(theta) is read out of cluster LENGTH,
    which is shortest at eta ~ 0, so the resolution is expected to be worst
    exactly where most tracks are. A single global number would be a fiction.
    The binning variable is the TRUE angle, eta = asinh(tpGlobalClusterCotTheta),
    so it is not the quantity being measured.

    PULLS. sigGlobalClusterCotTheta is the estimator's own claimed uncertainty.
    Every weighted vote in the seeding design inherits it, so its pull width is
    checked here rather than trusted; a width far from 1.0 means the segment
    lengths would be set from a miscalibrated sigma.
    """
    need = ("gClCotTheta", "tpGClCotTheta", "globalR", "globalZ")
    if not all(k in K for k in need):
        print("   (7) SKIPPED: cluster table lacks the global-frame angle columns "
              "(globalClusterCotTheta / tpGlobalClusterCotTheta / globalR / globalZ)")
        out["z0_resolution"] = {"skipped": "no global-frame angle columns"}
        for a in ax_row:
            a.axis("off")
        return

    ok = (K["tpIdx"] >= 0) & (K["gClCotTheta"] > SENTINEL) & (K["tpGClCotTheta"] > SENTINEL)
    if "clHasBeta" in K:
        ok &= K["clHasBeta"] > 0
    d = K["gClCotTheta"] - K["tpGClCotTheta"]          # cot(theta) residual
    dz0 = K["globalR"] * d                              # cm; z0 = z - r cot(theta)
    eta = np.arcsinh(K["tpGClCotTheta"])                # cot(theta) = sinh(eta)
    lay = K["layer"]
    Zspan = 2.0 * Z_HALF_RANGE_CM

    res = {"n_clusters_used": int(ok.sum()),
           "definition": "sigma = robust width of (reco - truth) on the same cluster",
           "suppression_formula": "Z_range / (4 sigma_z0), w cancels",
           "cotTheta_validated": (
               "globalClusterCotTheta IS the global polar slope dz/dr, verified after the "
               "module-flip fix: per-TP RMS of (z - r*cot) is 0.0087 cm against a 3.18 cm "
               "do-nothing baseline, a 365x collapse. An earlier revision wrongly also "
               "required |cot| to be layer-independent across the POPULATION; that is not a "
               "valid test, because TBPX is a barrel and high-|eta| tracks only reach the "
               "inner layers. Restricted to TPs that reach L4 the medians are flat "
               "(0.577/0.532/0.494/0.477 on L1-L4)."),
           "CIRCULAR": (
               "NOT A PHYSICAL RESOLUTION. SmartPixelsRecHitProducer builds the truth "
               "angle as the TP's helix propagated to the hit with NO multiple scattering "
               "(:292-319), then sets reco = truth + PixelAV draw (:330-331); the nano "
               "truth column is that same value rotated (L1SmartPixelsClusterTableProducer"
               ".cc:199-202). So (reco - truth) IS the payload draw and every sigma below "
               "is the payload's own width. Tell: sigma is equal for pT>2 and 0.5-2 GeV to "
               "0.3%, where a physical resolution would degrade toward low pT. Treat these "
               "as an OPTIMISTIC BOUND. See doc/SmartPixelsSeedingAndFitting.md section 4a.")}

    # ---- per layer, and per layer x |eta| -----------------------------------
    per_layer = {}
    for L in LAYERS:
        m = ok & (lay == L)
        s_cot = robust_sigma(d[m])
        s_z0 = robust_sigma(dz0[m])
        per_layer[f"L{L}"] = {
            "n": int(m.sum()),
            "median_r_cm": float(np.median(K["globalR"][m])) if m.any() else float("nan"),
            "sigma_cotTheta_mad": s_cot[0], "sigma_cotTheta_q68": s_cot[1],
            "sigma_z0_cm_mad": s_z0[0], "sigma_z0_cm_q68": s_z0[1],
            "z0_slices": float(Zspan / (4.0 * s_z0[0])) if s_z0[0] == s_z0[0] else float("nan"),
        }
    res["per_layer"] = per_layer

    eta_tab = {}
    for i in range(len(ETA_RES_EDGES) - 1):
        lo, hi = ETA_RES_EDGES[i], ETA_RES_EDGES[i + 1]
        me = ok & (np.abs(eta) >= lo) & (np.abs(eta) < hi)
        row = {}
        for L in LAYERS:
            s = robust_sigma(dz0[me & (lay == L)])
            row[f"L{L}"] = s[0]
        row["n"] = int(me.sum())
        row["sigma_cotTheta_mad"] = robust_sigma(d[me])[0]
        eta_tab[f"{lo:.1f}-{hi:.1f}"] = row
    res["vs_abs_eta"] = eta_tab

    # ---- the two design targets separately ---------------------------------
    if "tpPt" in K:
        for nm, sel in (("A_pt_gt_2", ok & (K["tpPt"] > 2.0)),
                        ("B_pt_0p5_to_2", ok & (K["tpPt"] > 0.5) & (K["tpPt"] <= 2.0))):
            s = robust_sigma(dz0[sel])
            res.setdefault("by_target", {})[nm] = {
                "n": int(sel.sum()), "sigma_z0_cm_mad": s[0],
                "z0_slices": float(Zspan / (4.0 * s[0])) if s[0] == s[0] else float("nan")}

    # ---- cluster-weighted headline + the pull check -------------------------
    s_all = robust_sigma(dz0[ok])
    res["overall"] = {"sigma_z0_cm_mad": s_all[0], "sigma_z0_cm_q68": s_all[1],
                      "z0_slices": float(Zspan / (4.0 * s_all[0])) if s_all[0] == s_all[0]
                      else float("nan"),
                      "sigma_for_30_slices_cm": float(Zspan / 120.0)}
    if "gSigCotTheta" in K:
        pm = ok & (K["gSigCotTheta"] > 0)
        pull = d[pm] / K["gSigCotTheta"][pm]
        pw = robust_sigma(pull)
        res["pull"] = {"n": int(pm.sum()), "width_mad": pw[0], "width_q68": pw[1],
                       "median": float(np.median(pull)) if pm.any() else float("nan"),
                       "note": "width far from 1.0 => the stored sigma is miscalibrated"}

    print(f"   (7) sigma(z0) overall {s_all[0]*1e4:.0f} um "
          f"-> {res['overall']['z0_slices']:.1f} usable z0 slices "
          f"(design assumed 30, which needs {Zspan/120.0*1e4:.0f} um)")
    for L in LAYERS:
        p = per_layer[f"L{L}"]
        print(f"       L{L}: r={p['median_r_cm']:.1f} cm  sigma(cot)={p['sigma_cotTheta_mad']:.4f}"
              f"  sigma(z0)={p['sigma_z0_cm_mad']*1e4:.0f} um  slices={p['z0_slices']:.1f}")
    if "pull" in res:
        print(f"       pull width {res['pull']['width_mad']:.3f} (want ~1.0)")

    # ---- figures -------------------------------------------------------------
    ax = ax_row[0]
    for k_eta, row in eta_tab.items():
        ys = [row[f"L{L}"] * 1e4 for L in LAYERS]
        ax.plot(LAYERS, ys, marker="o", label=rf"$|\eta|$ {k_eta}")
    ax.axhline(Zspan / 120.0 * 1e4, color="k", ls=":", lw=1)
    ax.text(4.05, Zspan / 120.0 * 1e4, " 30 slices", fontsize=6, va="center")
    ax.set_yscale("log"); ax.set_xlabel("TBPX layer [index]")
    ax.set_ylabel(r"robust $\sigma(z_0)$ [$\mu$m]")
    ax.set_title(r"(7) per-cluster $z_0$ resolution"); ax.legend(fontsize=6); ax.grid(alpha=.3)

    ax = ax_row[1]
    if "pull" in res:
        pm = ok & (K["gSigCotTheta"] > 0)
        ax.hist(np.clip(d[pm] / K["gSigCotTheta"][pm], -5, 5), bins=80,
                histtype="step", density=True, label=f"pull (w={res['pull']['width_mad']:.2f})")
    rng = np.nanpercentile(np.abs(d[ok]), 99) if ok.any() else 1.0
    ax.hist(np.clip(d[ok] / max(rng, 1e-9), -5, 5), bins=80, histtype="step", density=True,
            label=r"$\Delta\cot\theta$ / q99")
    ax.set_xlabel("normalised residual [unitless, pull]"); ax.set_ylabel("density [1/bin]")
    ax.set_title(r"(7) $\cot\theta$ residual and pull"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["z0_resolution"] = res


# --------------------------------------------------------------------------
# (8) Hough example panels
# --------------------------------------------------------------------------
B_FIELD_T = 3.8
# phi_pos = phi0 - C_BEND * r[cm] * kappa[1/GeV]. The DIRECTION turns twice as
# fast as the position azimuth, which is what makes a cluster a SEGMENT.
C_BEND = 0.29979246 * B_FIELD_T / 2.0 / 100.0
HOUGH_PT_POINTS = (1.0, 2.0, 5.0, 10.0, 20.0)
HOUGH_ETA_POINTS = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
LAYER_COLOR = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4", 4: "#2ca02c"}  # red/orange/blue/green
BAND_ALPHA = (0.95, 0.40, 0.10)      # |d| < 1 sigma, 1-3 sigma, > 3 sigma
CONE_DPHI, CONE_DETA = 0.20, 0.20
SECTOR_DPHI = 2.0 * np.pi / 9.0
Z0_SLICE_NSIG = 2.0
SECTOR_DRAW_CAP = 3000


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _half_turn(phi_pos, phi_dir):
    """asin(C_BEND * r * kappa): the half turn from the beamline out to the hit.

    EXACT, not the small-angle form. For a helix from the origin the position
    azimuth lags phi0 by asin(c*r*kappa) and the DIRECTION by twice that, so
    phi_p - phi_d = asin(c*r*kappa) and kappa = sin(phi_p - phi_d) / (c*r).

    HISTORY, because the failure was silent and cost a full round of wrong
    results: before the module-flip fix, `globalClusterPhi` was propagated
    without accounting for modules being physically flipped within a ladder (a
    real feature of the detector, not a code invention). The stored direction
    then pointed inward on flipped modules, so phi_p - phi_d came out as
    pi - asin(...) on some clusters and as +/-asin(...) on others -- correct
    magnitude, scrambled sign. sin() absorbs the pi but NOT the sign, so kappa
    came out with a random sign per cluster and any segment-shortening built on
    it was meaningless.

    Post-fix on spix_postq_500.root: median |phi_p - phi_d| = 0.098 with ZERO
    clusters beyond 2.0 rad (the pi branch is gone), |kappa|*pT = 1.0035, and
    the per-cluster sign agrees with truth 98.1% on clean >=3-layer topologies
    (98.3/98.3/98.1/97.5 on L1-L4).
    """
    return _wrap(phi_pos - phi_dir)


def _hough_bands(centre, sigma, slope, intercept, xlo, xhi, has_angle):
    """Split y = intercept + slope*x into |x-centre| < 1s, 1-3s, >3s.

    Returns three (N,2,2) arrays for LineCollection. A cluster whose sensor
    reports NO angle has centre=sigma=0, which makes bands 1 and 2 empty and
    band 3 span the full range in two pieces -- i.e. it constrains nothing and
    is drawn faintest across the whole line. That is the correct behaviour and
    needs no special case.
    """
    ok = has_angle & np.isfinite(sigma) & (sigma > 0) & np.isfinite(centre)
    c = np.where(ok, centre, 0.0)
    s = np.where(ok, sigma, 0.0)

    def seg(a, b):
        a, b = np.clip(a, xlo, xhi), np.clip(b, xlo, xhi)
        m = b > a
        if not m.any():
            return None
        x0, x1 = a[m], b[m]
        return np.stack([np.stack([x0, intercept[m] + slope[m] * x0], -1),
                         np.stack([x1, intercept[m] + slope[m] * x1], -1)], 1)

    full = np.full_like(c, xlo), np.full_like(c, xhi)
    spans = [[(c - s, c + s)],
             [(c - 3 * s, c - s), (c + s, c + 3 * s)],
             [(full[0], c - 3 * s), (c + 3 * s, full[1])]]
    out = []
    for band in spans:
        parts = [p for p in (seg(a, b) for a, b in band) if p is not None]
        out.append(np.concatenate(parts) if parts else np.empty((0, 2, 2)))
    return out


def _draw_hough(ax, sel, K, xlo, xhi, mode, truth, rasterize, shade=True):
    """One Hough panel. mode='rphi' -> (kappa, phi0); mode='rz' -> (cotTheta, z0).

    shade=False draws the classic POSITION-ONLY Hough line: full range, one shade
    per layer, no angle information used. That is the honest fallback while the
    direction columns carry a per-module orientation convention (see the module
    docstring of study 8).
    """
    from matplotlib.collections import LineCollection
    r = K["globalR"][sel]
    if not shade and mode == "rphi":
        phi_p = K["globalPhi"][sel]
        slope, intercept = C_BEND * r, _wrap(phi_p - truth["phi0"])
        for L in LAYERS:
            m = K["layer"][sel] == L
            if not m.any():
                continue
            x0 = np.full(int(m.sum()), xlo); x1 = np.full(int(m.sum()), xhi)
            segs = np.stack([np.stack([x0, intercept[m] + slope[m] * x0], -1),
                             np.stack([x1, intercept[m] + slope[m] * x1], -1)], 1)
            ax.add_collection(LineCollection(segs, colors=LAYER_COLOR[L], linewidths=0.6,
                                             alpha=0.45, rasterized=rasterize))
        ax.set_xlim(xlo, xhi)
        return
    if mode == "rphi":
        phi_p = K["globalPhi"][sel]
        slope = C_BEND * r
        intercept = _wrap(phi_p - truth["phi0"])          # plot relative to truth
        half = _half_turn(phi_p, K["gClPhi"][sel])
        centre = np.sin(half) / np.maximum(slope, 1e-12)
        sigma = K["gSigPhi"][sel] * np.abs(np.cos(half)) / np.maximum(slope, 1e-12)
        has = K["gClPhi"][sel] > SENTINEL
    else:
        slope = -r
        intercept = K["globalZ"][sel]
        centre = K["gClCotTheta"][sel]
        sigma = K["gSigCotTheta"][sel]
        has = K["gClCotTheta"][sel] > SENTINEL
    for L in LAYERS:
        m = K["layer"][sel] == L
        if not m.any():
            continue
        bands = _hough_bands(centre[m], sigma[m], slope[m], intercept[m], xlo, xhi, has[m])
        for b, segs in enumerate(bands):
            if len(segs):
                ax.add_collection(LineCollection(
                    segs, colors=LAYER_COLOR[L], linewidths=0.6, alpha=BAND_ALPHA[b],
                    rasterized=rasterize))
    ax.set_xlim(xlo, xhi)



# --------------------------------------------------------------------------
# (10) combination sweep across activeSP configurations
# --------------------------------------------------------------------------
# Two-sided per-axis Gaussian quantiles, the SAME convention as CONES above
# (q68 -> 1.0, q95 -> 1.96): a "qX cone" means each axis is within k sigma where
# k = Phi^-1((1+X)/2). Joint containment of a 2D box is the square of that, so
# these are per-axis levels and not the probability of keeping the true hit.
SWEEP_Q = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
SWEEP_COMBO_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 1 << 30],
                             dtype=float)

# Report order: grouped by how many layers are INSTRUMENTED, and within a group by
# which ones. Grouping this way is the point -- the interesting comparisons are
# between builds of equal cost in ASICs, where the only difference is WHICH layers
# were chosen, and those differ by more than an order of magnitude in search cost.
SWEEP_ORDER = ["AIII", "IAII", "IIAI", "IIIA",
               "AAII", "AIAI", "AIIA", "IAAI", "IAIA", "IIAA",
               "AAAI", "AAIA", "AIAA", "IAAA",
               "AAAA"]
# CONE WIDTHS the design table is evaluated at. These are the realistic knob: the
# cone quantile IS how wide the search window is, so it sets both how many true
# hits enter the refit and how much combinatorics comes with them. One total-work
# number cannot express that trade, so every cone gets its own column.
#
# NOMINAL, NOT MEASURED. k = Phi^-1((1+q)/2) assumes unit-width pulls, and ours are
# 0.81 to 1.52 depending on layer and visit depth (see q_acceptance.py), so a
# "q99.99 cone" does NOT deliver 99.99% true-hit containment. That is exactly why
# the table also carries MEASURED containment beside the cost.
SWEEP_TABLE_CONES = [0.99, 0.999, 0.9999]


def _variant_crossings(files, suffix):
    """Crossings for ONE activeSP variant. Only the columns the sweep needs."""
    hit = f"L1TSmartPixelsRefitHitDigiRefit{suffix}"
    cols = ["trackIdx", "layer", "detId", "projLocalX", "projLocalY", "projSigX", "projSigY",
            "hitAccepted"]
    H = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{hit}_{c}" for c in cols])
    ncross = ak.to_numpy(ak.num(H[f"{hit}_layer"]))
    X = {c: ak.to_numpy(ak.flatten(H[f"{hit}_{c}"])) for c in cols}
    ev = np.repeat(np.arange(len(ncross)), ncross)
    X["event"] = ev
    # The matched TP is per TRACK and per VARIANT (each variant refits separately),
    # so it has to come from THIS variant's track table, not the one load() picked.
    tname = f"{hit.replace('RefitHit', 'Track')}_spixMatchedTpIdx"
    V = uproot.concatenate([f"{f}:Events" for f in files], filter_name=[tname])
    mtp = ak.to_numpy(ak.flatten(V[tname]))
    offm = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(V[tname])))])
    X["trk_tpIdx"] = mtp[offm[ev] + X["trackIdx"].astype(np.int64)]
    return X


def _combinations_per_track(X, K, kvals):
    """For each k, the number of L1xL2xL3xL4 hit combinations each track must search.

    Counts are SUMMED over crossings within a layer before the product is taken:
    a track that clips two overlapping modules at one layer has a single candidate
    POOL there, so those crossings must add. Multiplying them would invent
    combinations that the refit never considers, since it takes one hit per layer.

    A layer with no candidate contributes a factor of 1, not 0. The refit skips it
    and carries on; treating it as 0 would erase the track from the distribution
    entirely and would bias the result toward the busy tracks.
    """
    xi, ci, _ = join_on_module(X, K)
    dx = K["localX"][ci] - X["projLocalX"][xi]
    dy = K["localY"][ci] - X["projLocalY"][xi]
    sx, sy = X["projSigX"][xi], X["projSigY"][xi]
    ok = (sx > 0) & (sy > 0) & (X["projLocalX"][xi] > SENTINEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = np.where(ok, np.abs(dx / sx), np.inf)
        ny = np.where(ok, np.abs(dy / sy), np.inf)

    tkey = X["event"].astype(np.int64) * (1 << 20) + X["trackIdx"].astype(np.int64)
    key2 = tkey * 8 + X["layer"].astype(np.int64)
    u2, inv2 = np.unique(key2, return_inverse=True)
    tk2 = u2 // 8
    ut, invt = np.unique(tk2, return_inverse=True)

    # TRUE-hit containment, the benefit side of widening the cone. A pair is the
    # true one when the cluster's dominant TP is the TP the track was matched to.
    # Denominator counts only crossings where such a cluster EXISTS on the module,
    # so a layer that simply had no true cluster does not count as an inefficiency.
    is_true = (K["tpIdx"][ci] >= 0) & (X["trk_tpIdx"][xi] >= 0) & \
              (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    have_true = np.zeros(len(X["layer"]), bool)
    have_true[xi[is_true]] = True
    n_have = int(have_true.sum())

    out, cont = {}, {}
    for k in kvals:
        m = (nx < k) & (ny < k)
        cnt = np.bincount(xi[m], minlength=len(X["layer"])).astype(np.float64)
        n_tl = np.bincount(inv2, weights=cnt, minlength=len(u2))
        prod = np.ones(len(ut), dtype=np.float64)
        np.multiply.at(prod, invt, np.maximum(n_tl, 1.0))
        out[k] = prod
        found = np.zeros(len(X["layer"]), bool)
        found[xi[is_true & m]] = True
        cont[k] = float((found & have_true).sum() / n_have) if n_have else float("nan")
    return out, len(ut), cont, n_have



def _write_sweep_table(res, sfx, out, n_ev):
    """Persistent activeSP design table: cost AND containment at each cone width.

    THE CONE WIDTH IS THE DESIGN KNOB, so it gets a column rather than a single
    chosen value. Widening the window admits more true hits into the refit and more
    combinatorics with them; a lone total-work number cannot express that trade and
    would let a build look cheap purely because it was quoted at a tight cone.
    Every cone therefore carries both its cost (TOTAL work) and what that cost
    bought (measured true-hit containment).

    CONTAINMENT IS MEASURED, NOT ASSUMED. The quantile is nominal: k =
    Phi^-1((1+q)/2) presumes unit-width pulls and ours run 0.81 to 1.52 depending
    on layer and visit depth, so a "q99.99 cone" does not deliver 99.99%. The
    denominator counts only crossings where a truth-matched cluster actually exists
    on the module, so a layer that had no true cluster is not scored as an
    inefficiency.

    Emitted on every run rather than pasted into a message: every number here moved
    at least once during development.
    """
    order = [s for s in SWEEP_ORDER if s in sfx] + [s for s in sfx if s not in SWEEP_ORDER]
    cones = res["table_cones"]
    ck = [f"{q:.4f}" for q in cones]
    ntrk_ref = max((res[s]["n_tracks"] for s in order), default=0)

    def qlab(q):
        return ("q%g" % (q * 100)).rstrip("0").rstrip(".") if q * 100 % 1 else "q%d" % (q * 100)

    labs = [qlab(q) for q in cones]
    trk_per_ev = ntrk_ref / max(n_ev, 1)
    W = 8
    blk = W * len(labs)
    grp = (f"{'':<10}  " + f"{'combinations / track':^{blk}}" + " | "
           + f"{'p99 (busy track)':^{blk}}" + " | " + f"{'containment':^{blk}}"
           + " | " + f"{'viability':^16}")
    sub = (f"{'cfg':<6}{'nSP':>4}  " + "".join(f"{l:>{W}}" for l in labs) + " | "
           + "".join(f"{l:>{W}}" for l in labs) + " | "
           + "".join(f"{l:>{W}}" for l in labs) + " | "
           + f"{'>=2hit':>8}{'>=3hit':>8}")
    hdr = grp + "\n" + sub
    lines = [
        "activeSP refit search cost vs CONE WIDTH",
        f"  sample        : {n_ev} events, PU200, {ntrk_ref:,} tracks per config",
        "  cone          : nominal per-axis two-sided Gaussian, k = Phi^-1((1+q)/2)",
        "                  " + " | ".join(
            f"{l} k={res['table_k_sigma'][i]:.3f}" for i, l in enumerate(labs)),
        f"                  {trk_per_ev:.0f} refit-able tracks per event",
        "  combinations  : PER TRACK, the number of (L1,L2,L3,L4) hit tuples a refit must",
        "                  test = product of the candidate counts over INSTRUMENTED layers.",
        "                  A layer with NO candidate contributes 1, not 0 -- the refit skips",
        "                  it and carries on. Worked: 2,1,0,2 -> 2*1*1*2 = 4;  3,2,1,1 -> 6.",
        "                  (pinned by tests/test_combination_counting.py)",
        "  combinations  : MEAN number per track refit at that cone width (see below for",
        "                  what a combination is).",
        f"                  Multiply by {trk_per_ev:.0f} for per-event throughput.",
        "  p99 <cone>    : 99th-percentile track, i.e. the busy-track cost. Stable: 100 ->",
        "                  500 events moves it by at most one unit.",
        "  NOT SHOWN     : the CSV also carries combos_per_track_max, which DOES NOT",
        "                  CONVERGE -- it is an extreme-value statistic on a heavy tail and",
        "                  grows with sample size (AIAI: 54 at 100 events, 550 at 500). It is",
        "                  not a design number without an explicit per-N-events framing, and",
        "                  ranking builds by it produces conclusions that reverse when more",
        "                  events are added. Use p99.",
        "  cfg / nSP     : the activeSP mask, and how many layers it instruments",
        "  <hits>        : mean number of hits the refit attached to a track",
        "  containment   : MEASURED fraction of crossings whose true cluster falls in the",
        "                  cone. The nominal quantile is NOT the achieved containment --",
        "                  pull widths are 0.81-1.52, so this is measured, not assumed.",
        "  >=2hit/>=3hit : fraction of tracks the refit gave that many hits. 0% for every",
        "                  two-layer build is structural, not performance.",
        "",
        "  qX is the cone WIDTH you choose (an input); pXX is where a track sits in the",
        "  resulting distribution (an output). They look alike and are unrelated.",
        "  Full definitions of every term: spix_glossary.txt, written beside this file.",
        "",
        hdr, "-" * len(sub)]
    group = {1: "1 instrumented layer", 2: "2 instrumented layers",
             3: "3 instrumented layers", 4: "4 instrumented layers"}
    last_n = None
    for st in order:
        n = st.count("A")
        if n != last_n:
            lines.append(f"--- {group.get(n, str(n))} ---")
            last_n = n
        r, f = res[st], res[st]["frac_tracks_with_hits"]
        bc = r["by_cone"]
        lines.append(
            f"{st:<6}{n:>4}  "
            + "".join(f"{bc[c]['mean']:>{W}.2f}" for c in ck) + " | "
            + "".join(f"{bc[c]['p99']:>{W}.0f}" for c in ck) + " | "
            + "".join(f"{100 * bc[c]['true_hit_containment']:>{W - 1}.1f}%" for c in ck)
            + " | " + f"{100 * f['ge2']:>7.0f}%{100 * f['ge3']:>7.0f}%")
    txt = "\n".join(lines)
    tp = os.path.join(out["_outdir"], "spix_activesp_table.txt")
    with open(tp, "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"\n   (10) wrote {tp}")

    cp = os.path.join(out["_outdir"], "spix_activesp_table.csv")
    with open(cp, "w") as fh:
        fh.write("activeSP,n_sp_layers,cone_q,k_sigma,combos_per_track_mean,"
                 "combos_per_track_median,combos_per_track_p99,combos_per_track_max,"
                 "combos_per_event,total_work_raw,true_hit_containment,"
                 "mean_accepted_hits,frac_ge2hit,frac_ge3hit\n")
        for st in order:
            r, f = res[st], res[st]["frac_tracks_with_hits"]
            for c, d in sorted(r["by_cone"].items(), key=lambda kv: float(kv[0])):
                fh.write(f"{st},{st.count('A')},{float(c):.4f},{d['k_sigma']:.4f},"
                         f"{d['mean']:.4f},{d['median']:.0f},{d['p99']:.0f},{d['max']:.0f},"
                         f"{d['combos_per_event']:.2f},{d['total_work']:.0f},"
                         f"{d['true_hit_containment']:.5f},"
                         f"{r['mean_accepted_hits']:.4f},{f['ge2']:.4f},{f['ge3']:.4f}\n")
    print(f"   (10) wrote {cp}")
    res["_table"] = os.path.basename(tp)
    res["_csv"] = os.path.basename(cp)


def study_combination_sweep(X, K, P, ax_row, out):
    """(10) Search-space size vs cone quantile, for every non-trivial activeSP mask.

    THE QUESTION THIS ANSWERS. Widening the cone buys hit-finding efficiency and
    costs combinatorics, and adding smart-pixel layers tightens the cone for free.
    Neither trade is readable from a covariance: the number that matters is how
    many L1xL2xL3xL4 hit combinations a refit must actually search, and how its
    DISTRIBUTION over tracks moves. A mean is not enough here, because the cost of
    the tail is what sets the hardware budget -- hence a 2D histogram per
    configuration rather than a curve.

    EACH MASK IS A DETECTOR BUILD, NOT AN ALGORITHM SETTING. activeSP is which IT
    layers are INSTRUMENTED with smart pixels. Smart pixels emit cluster data at L1
    latency; a conventional pixel layer is not read out until after an L1 accept. So
    an uninstrumented layer contributes NOTHING to L1 track building of any form --
    not "position but no angle", nothing at all. Measured crossing counts by layer:
    AAAA gives 436/374/296/227, AAII gives 435/373/0/0, IIIA gives 0/0/0/219.

    That means a mask with fewer A's searching fewer layers is not an unfairness in
    the comparison, it IS the trade: instrument less, get less information AND less
    combinatorics. These 15 panels are a cost curve across candidate builds.

    WHICH layers matters as much as HOW MANY, and asymmetrically. L1 is both the
    busiest layer (seed-cone occupancy 1.98 vs 0.82 at L4) and the smallest radius,
    so a build instrumenting only inner layers enters the densest region carrying
    the full untightened OT-seed cone (~308 um). An outer-first build gets
    progressive tightening before it arrives there. Two builds with the same number
    of A's are not interchangeable.

    COMBINATIONS ARE THE COST SIDE ONLY. A build can look cheap because it cannot do
    the job. The third panel therefore carries the viability counterpart -- what
    fraction of tracks even collect enough hits to refit -- and resolution/purity
    per build is still missing and must not be inferred from these histograms.

    The cone widths corroborate the outsideIn ordering: AAAA tightens monotonically
    133.9 / 84.8 / 61.9 / 35.2 um from L4 in to L1, while AAII visits L2 FIRST
    (235.8 um, no update yet) and then L1 (32.6 um). IIIA's only layer, L4, sits at
    133.9 um -- identical to AAAA's L4, which is the check that a first-visited
    layer receives no update in either configuration.
    """
    files = out.get("_inputs")
    if not files:
        return
    with uproot.open(f"{files[0]}:Events") as t:
        keys = set(t.keys())
    sfx = sorted({m.group(1) for m in
                  (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([AI]{4})_", k) for k in keys) if m})
    if len(sfx) < 2:
        print(f"   (10) SKIPPED: input has {len(sfx)} activeSP variant(s); the sweep needs the "
              "15 non-trivial masks. Regenerate with a repeated --variant digiRefit:XXXX.")
        for a in ax_row:
            a.axis("off")
        return

    kvals = [NormalDist().inv_cdf(0.5 * (1.0 + q)) for q in SWEEP_Q]
    tabq = [q for q in SWEEP_TABLE_CONES]
    ktab = [NormalDist().inv_cdf(0.5 * (1.0 + q)) for q in tabq]
    # one pass over the join covers plot cones and the wider design cones
    allq = list(SWEEP_Q) + [q for q in tabq if q not in SWEEP_Q]
    allk = kvals + [k for q, k in zip(tabq, ktab) if q not in SWEEP_Q]
    res, per_cfg = {"quantiles": SWEEP_Q, "k_sigma": [float(k) for k in kvals],
                    "table_cones": tabq,
                    "table_k_sigma": [float(k) for k in ktab]}, {}
    n_ev_local = max(int(out.get("n_events", 0)), 1)
    for s in sfx:
        Xv = _variant_crossings(files, s)
        combos, ntrk, cont, n_have = _combinations_per_track(Xv, K, allk)
        per_cfg[s] = combos
        # Viability: an uninstrumented layer yields no L1 hit at all, so a build can
        # simply run out of hits. Counted from the refit's OWN acceptance rather than
        # from cone occupancy, since that is what it actually kept.
        tk = Xv["event"].astype(np.int64) * (1 << 20) + Xv["trackIdx"].astype(np.int64)
        ut2, inv3 = np.unique(tk, return_inverse=True)
        nacc = np.bincount(inv3, weights=(Xv["hitAccepted"] > 0).astype(float),
                           minlength=len(ut2))
        frac = {f"ge{j}": float((nacc >= j).mean()) for j in (1, 2, 3, 4)}
        res[s] = {"n_tracks": int(ntrk), "frac_tracks_with_hits": frac,
                  "mean_accepted_hits": float(nacc.mean()),
                  "n_crossings_with_true_cluster": int(n_have),
                  # keyed by CONE quantile: each is a different design point
                  "by_cone": {f"{q:.4f}": {
                      "k_sigma": float(k),
                      # RAW SUM over whatever sample ran -- not interpretable on its
                      # own, kept only so the normalised numbers can be rederived.
                      "total_work": float(combos[k].sum()),
                      "combos_per_event": float(combos[k].sum() / max(n_ev_local, 1)),
                      "mean": float(combos[k].mean()),
                      "median": float(np.median(combos[k])),
                      "p99": float(np.percentile(combos[k], 99)),
                      "max": float(combos[k].max()),
                      "true_hit_containment": float(cont[k]),
                  } for q, k in zip(allq, allk)},
                  "median": [float(np.median(combos[k])) for k in kvals],
                  "mean": [float(combos[k].mean()) for k in kvals],
                  "p99": [float(np.percentile(combos[k], 99)) for k in kvals],
                  # p99 ALONE INVERTS THE RANKING and must not be quoted by itself.
                  # At q99 AAAA beats AIII on p99 (12 vs 15) and loses badly on the
                  # deep tail (672 vs 48), because multiplying four layers lets rare
                  # busy tracks produce enormous products. Throughput budget and
                  # worst-case budget disagree here, so both are reported.
                  "p999": [float(np.percentile(combos[k], 99.9)) for k in kvals],
                  "p9999": [float(np.percentile(combos[k], 99.99)) for k in kvals],
                  "max": [float(combos[k].max()) for k in kvals],
                  "total_work": [float(combos[k].sum()) for k in kvals]}
        print(f"   (10) {s}: median combos "
              + "/".join(f"{v:.0f}" for v in res[s]["median"])
              + f"   q99: p99={res[s]['p99'][-1]:.0f} p99.9={res[s]['p999'][-1]:.0f} "
                f"p99.99={res[s]['p9999'][-1]:.0f} max={res[s]['max'][-1]:.0f} "
                f"tot={res[s]['total_work'][-1]:,.0f}"
              + f"   <hits>={res[s]['mean_accepted_hits']:.2f}"
              + f"  >=2hit {100 * frac['ge2']:.0f}%  >=3hit {100 * frac['ge3']:.0f}%")

    # ---- per-configuration 2D histograms -----------------------------------
    ncol = 4
    nrow = int(np.ceil(len(sfx) / ncol))
    f2, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
    vmax = 1
    grids = {}
    for s in sfx:
        g = np.zeros((len(SWEEP_COMBO_EDGES) - 1, len(kvals)))
        for j, k in enumerate(kvals):
            g[:, j] = np.histogram(per_cfg[s][k], bins=SWEEP_COMBO_EDGES)[0]
        grids[s] = g
        vmax = max(vmax, g.max())
    # Clip the shared y-range to the highest OCCUPIED bin. The top edge exists to
    # catch an unbounded tail, but drawn literally it hands most of a log axis to
    # empty space and makes every panel look identical.
    occupied = max((np.flatnonzero(g.sum(axis=1)).max() for g in grids.values()
                    if g.sum() > 0), default=len(SWEEP_COMBO_EDGES) - 2)
    ytop = SWEEP_COMBO_EDGES[min(occupied + 1, len(SWEEP_COMBO_EDGES) - 1)]
    for i, s in enumerate(sfx):
        a = axs[i // ncol][i % ncol]
        mesh = a.pcolormesh(np.arange(len(kvals) + 1), SWEEP_COMBO_EDGES, grids[s],
                            norm=LogNorm(vmin=1, vmax=vmax), cmap="viridis")
        a.set_yscale("log")
        a.set_ylim(1, ytop)
        a.set_xticks(np.arange(len(kvals)) + 0.5)
        a.set_xticklabels([f"{int(q * 100)}" for q in SWEEP_Q], fontsize=7)
        a.set_title(f"activeSP {s}  ({s.count('A')} SP layer"
                    f"{'s' if s.count('A') != 1 else ''})", fontsize=9)
        a.set_xlabel("cone quantile qX [unitless]", fontsize=8)
        a.set_ylabel("combinations per track [count]", fontsize=8)
        f2.colorbar(mesh, ax=a, label="tracks")
    spare = [(i // ncol, i % ncol) for i in range(len(sfx), nrow * ncol)]
    for r_, c_ in spare:
        axs[r_][c_].axis("off")
    if spare:
        # A colour scale and two axis labels do not tell a reader what any of this
        # means, so the legend travels WITH the figure rather than living in a file
        # they may never open.
        a = axs[spare[0][0]][spare[0][1]]
        a.text(0.0, 1.0, _glossary_text(width=52, terms=[
            "activeSP  /  A  /  I", "qX  (cone quantile)", "combinations / track"]),
            transform=a.transAxes, va="top", ha="left", fontsize=5.4, family="monospace")
    f2.suptitle("(10) refit search space: L1xL2xL3xL4 combinations vs cone quantile, "
                "per activeSP configuration", y=1.002)
    f2.tight_layout()
    p2 = os.path.join(out["_outdir"], "spix_combination_sweep.png")
    f2.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(f2)
    print(f"   (10) wrote {p2}")
    res["_figure"] = os.path.basename(p2)

    # ---- summary panels in the omnibus figure -------------------------------
    order = [s for s in SWEEP_ORDER if s in sfx] + [s for s in sfx if s not in SWEEP_ORDER]
    cmap = plt.get_cmap("turbo")
    for a, stat, lab in ((ax_row[0], "total_work", "TOTAL combinations"),
                         (ax_row[1], "max", "worst-case (max)")):
        for i, s in enumerate(order):
            a.plot([q * 100 for q in SWEEP_Q], res[s][stat], marker="o", ms=3,
                   color=cmap(i / max(len(order) - 1, 1)), label=s)
        a.set_yscale("log"); a.set_xlabel("cone quantile qX [unitless]")
        a.set_ylabel(f"{lab} combinations per track [count]")
        a.set_title(f"(10) {lab} search space vs cone")
        a.grid(alpha=.3)
        a.legend(fontsize=5, ncol=3)
    a = ax_row[2]
    xs = np.arange(len(order))
    for j, mk in ((2, "s"), (3, "^"), (4, "v")):
        a.plot(xs, [res[s]["frac_tracks_with_hits"][f"ge{j}"] for s in order],
               marker=mk, ms=4, label=f"$\\geq${j} hits")
    a.set_xticks(xs); a.set_xticklabels(order, rotation=90, fontsize=6)
    a.set_ylabel("tracks [fraction]"); a.set_ylim(0, 1.02)
    a.set_title("(10) can this build even refit?")
    a.grid(alpha=.3); a.legend(fontsize=7)
    _write_sweep_table(res, sfx, out, out.get("n_events", -1))
    out["combination_sweep"] = res



# --------------------------------------------------------------------------
# (11) fixed eta-phi bin sizing for a refit-only design
# --------------------------------------------------------------------------
# WHY THE PER-MODULE CONE IS THE WRONG UNIT. A refit-only system never projects to
# a single module. Each eta x phi sector board holds the aggregate clusters its
# region produced, pre-binned in phi and eta, and a track's projection selects
# which bins to read. So the candidate pool is set by the BIN, not by the cone, and
# study (10) is a floor that assumes perfect module targeting.
#
# WHY THE BIN MUST EXCEED THE CONE. The cone is centred on the projection; a fixed
# grid is not. A projection lands at an arbitrary phase within its bin, so the
# desired cluster -- offset from the projection by up to the cone width -- falls in
# a NEIGHBOURING bin whenever the projection sits near an edge. The bin therefore
# has to be inflated relative to the cone, and by how much is an empirical
# question, which is what this study answers.
#
# AND THE CONE IT MUST EXCEED IS THE SEED CONE, not the refit-order one. Binning
# happens BEFORE any refit, so the only uncertainty available is the OT seed
# projection: 308 um at L1 against the refit-order 37 um. Sizing bins against the
# refit cone would be circular and would undersize them by an order of magnitude.
SECTOR_TARGET = 0.99          # required containment of desired clusters
# Bin width as a multiple of the per-layer q99 desired-cluster offset. Capped at 2:
# q99 already fills a large part of a nonant at L1 (92 mrad against 698), so larger
# multiples stop being a grid and become "read the whole sector".
SECTOR_F = [0.25, 0.5, 1.0, 2.0, 4.0]
SECTOR_ANG_NSIG = 3.0         # angle-compatibility cut, in sigma of the ML estimate
# Grid layouts. A single grid gives the projection an arbitrary phase inside its
# bin, so a one-bin read GUARANTEES NOTHING -- a cluster just past the edge sits in
# the neighbour. Storing the same clusters in offset grids (edges of one passing
# through centres of another) lets the track pick the grid whose bin centre its
# projection lands nearest, which is what makes a one-bin read viable at all.
# (label, phi offsets, eta offsets, bins read either side)
# Named for what they DO rather than by an abbreviation. The last field is the
# neighbour RADIUS in bins: 0 reads only the bin the projection landed in, 1 also
# reads every bin touching it, which in two dimensions is a 3x3 block. So
# reads = (2*radius + 1)^2.
SECTOR_GRIDS = [
    ("1grid_1bin",      1, 1, 0),   # baseline: one grid, one bin. Guarantees nothing.
    ("1grid_3x3",       1, 1, 1),   # one grid, read the surrounding ring too: 9 reads
    ("2grid_phi_1bin",  2, 1, 0),   # grids offset in phi, 2x storage, 1 read
    ("4grid_1bin",      2, 2, 0),   # grids offset in phi AND eta, 4x storage, 1 read
]


def _helix_project(rinv, phi0, tanl, z0, r):
    """Naive single-shot projection of an L1 track to radius r.

    Validated against 256,876 truth-matched track-cluster pairs: position phi
    residual 2.0 mrad, position z residual 910 um, direction phi residual 14.7 mrad
    and direction cot(theta) residual 0.0190 -- the last two sitting at the sensor's
    own angle resolution (sigma_beta ~ 0.0225), which is what confirms the formulas
    rather than a bug. Note the DIRECTION azimuth turns by TWICE the position
    azimuth (phi0 - 2*asin vs phi0 - asin); using the position form for the
    direction gives 19.3 mrad instead of 14.7.
    """
    half = np.arcsin(np.clip(0.5 * rinv * r, -1.0, 1.0))
    phi_pos = phi0 - half
    small = np.abs(rinv) < 1e-9
    s = np.where(small, r, 2.0 * half / np.where(small, 1.0, rinv))
    z = z0 + tanl * s
    eta = np.arcsinh(z / np.maximum(r, 1e-9))
    return phi_pos, z, eta, phi0 - 2.0 * half


def _phase_pick(x, w, ngrid):
    """Choose the offset grid whose bin CENTRE the value lands nearest, and its bin.

    Grid g has edges at (n + g/ngrid)*w. With ngrid=1 the phase is uniform and the
    projection can sit arbitrarily close to an edge, which is exactly why a one-bin
    read cannot guarantee capturing anything. With ngrid=2 the best grid always puts
    the projection within w/4 of a centre.
    """
    ph = x / w
    best_g = np.zeros(len(x), dtype=np.int64)
    best_d = np.full(len(x), np.inf)
    for g in range(ngrid):
        d = np.abs(((ph - g / ngrid) % 1.0) - 0.5)
        take = d < best_d
        best_d = np.where(take, d, best_d)
        best_g = np.where(take, g, best_g)
    ib = np.floor(ph - best_g / ngrid).astype(np.int64)
    return best_g, ib


def _sector_counts(tr, cl, tev, cev, layer, r_nom, wphi, weta, gspec, want_c,
                   xl, gil, have_ang):
    """Candidates per crossing, and whether the desired cluster was among them.

    Hash join on the fixed absolute grid rather than a track x cluster outer
    product, which inside one nonant would be ~250M pairs on this sample.

    NO CONE CUT IS APPLIED. The pre-processing runs with zero track knowledge, in
    parallel with OT track finding, so the bin IS the selection: whatever landed in
    the bin is what the refit must test. The only track-dependent step is choosing
    which bin to read.
    """
    _, ngp, nge, ring = gspec   # ring = neighbour radius in bins; reads = (2*ring+1)^2
    csel = cl["layer"] == layer
    if not csel.any():
        return None
    ca = np.flatnonzero(csel)
    phi_p, _, eta_p, phi_dir = _helix_project(tr["rInv"], tr["phi"], tr["tanL"],
                                              tr["z0"], r_nom)
    gp_t, ip_t = _phase_pick(phi_p, wphi, ngp)
    ge_t, ie_t = _phase_pick(eta_p, weta, nge)

    n_x = len(want_c)
    cand = np.zeros(n_x)
    cand_a = np.zeros(n_x)
    cand_ab = np.zeros(n_x)
    found = np.zeros(n_x, bool)
    found_a = np.zeros(n_x, bool)
    found_ab = np.zeros(n_x, bool)

    for gp in range(ngp):
        ipc = np.floor(cl["phi"][ca] / wphi - gp / ngp).astype(np.int64)
        for ge in range(nge):
            iec = np.floor(cl["eta"][ca] / weta - ge / nge).astype(np.int64)
            ck = (cev[ca].astype(np.int64) * 1000003 + ipc) * 1000003 + iec
            o = np.argsort(ck, kind="stable")
            cks, cis = ck[o], ca[o]
            tsel = np.flatnonzero((gp_t == gp) & (ge_t == ge))
            if not len(tsel):
                continue
            for dp in range(-ring, ring + 1):
                for de in range(-ring, ring + 1):
                    tk = ((tev[tsel].astype(np.int64) * 1000003 + ip_t[tsel] + dp)
                          * 1000003 + ie_t[tsel] + de)
                    lo = np.searchsorted(cks, tk, "left")
                    hi = np.searchsorted(cks, tk, "right")
                    n = hi - lo
                    if n.sum() == 0:
                        continue
                    ti = np.repeat(tsel[np.arange(len(tk))], n)
                    ramp = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
                    cj = cis[np.repeat(lo, n) + ramp]
                    # expand track -> that track's crossings on this layer
                    order = np.argsort(gil, kind="stable")
                    gs, xs = gil[order], xl[order]
                    l2 = np.searchsorted(gs, ti, "left")
                    h2 = np.searchsorted(gs, ti, "right")
                    n2 = h2 - l2
                    keep = n2 > 0
                    if not keep.any():
                        continue
                    idx = np.repeat(np.arange(len(ti))[keep], n2[keep])
                    r2 = np.arange(n2[keep].sum()) - np.repeat(
                        np.cumsum(n2[keep]) - n2[keep], n2[keep])
                    xidx = xs[np.repeat(l2[keep], n2[keep]) + r2]
                    cidx = cj[idx]
                    ok_a = np.ones(len(cidx), bool)
                    ok_ab = np.ones(len(cidx), bool)
                    if have_ang:
                        dph = np.abs(_wrap(cl["gphi"][cidx] - phi_dir[ti[idx]]))
                        useA = (cl["sphi"][cidx] > 0) & (cl["gphi"][cidx] > SENTINEL)
                        ok_a = ~useA | (dph < SECTOR_ANG_NSIG * cl["sphi"][cidx])
                        dct = np.abs(cl["gcot"][cidx] - tr["tanL"][ti[idx]])
                        useB = (cl["scot"][cidx] > 0) & (cl["gcot"][cidx] > SENTINEL)
                        ok_ab = ok_a & (~useB | (dct < SECTOR_ANG_NSIG * cl["scot"][cidx]))
                    cand += np.bincount(xidx, minlength=n_x)
                    cand_a += np.bincount(xidx[ok_a], minlength=n_x)
                    cand_ab += np.bincount(xidx[ok_ab], minlength=n_x)
                    hitv = cidx == want_c[xidx]
                    found[xidx[hitv]] = True
                    found_a[xidx[hitv & ok_a]] = True
                    found_ab[xidx[hitv & ok_ab]] = True
    return {"pos": (cand, found), "pos+alpha": (cand_a, found_a),
            "pos+alpha+beta": (cand_ab, found_ab)}


def study_sector_binning(X, K, P, ax_row, out):
    """(11) Fixed eta-phi bin sizing for a refit-only design.

    THE PRE-PROCESSING HAS NO TRACK KNOWLEDGE. It runs in parallel with OT track
    finding and must be finished before any track exists, so the grid is absolute
    and track-independent. The ONLY track-dependent step is a projection choosing
    which bin to read. That means THE BIN IS THE SELECTION -- there is no cone cut
    anywhere in this study, and whatever landed in the bin is what a refit tests.

    BINS MUST EXCEED THE CONE, because the cone is centred on the projection and a
    fixed grid is not. And the relevant cone is the SEED one: binning precedes the
    refit, so only the OT projection exists (308 um at L1, against the refit-order
    37 um). Sizing against the refit cone would be circular.

    DESIRED CLUSTER = the true cluster on the crossing's OWN module, which is what
    the refit wants. An earlier version scored ANY cluster sharing the track's TP,
    which admitted clusters elsewhere in the detector, inflated the L1 q99 offset
    from 92 to 439 mrad, and produced 8e8 combinations per track -- pure artefact.
    """
    files = out.get("_inputs")
    if not files:
        return
    tcols = ["rInv", "phi", "tanL", "z0", "pt"]
    T = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{REF}_{c}" for c in tcols])
    hitname = out.get("_hit_table", "")
    tpname = f"{hitname.replace('RefitHit', 'Track')}_spixMatchedTpIdx"
    MT = uproot.concatenate([f"{f}:Events" for f in files], filter_name=[tpname])
    ntr = ak.to_numpy(ak.num(T[f"{REF}_rInv"]))
    tr = {c: ak.to_numpy(ak.flatten(T[f"{REF}_{c}"])) for c in tcols}
    tr["tp"] = ak.to_numpy(ak.flatten(MT[tpname]))
    tev = np.repeat(np.arange(len(ntr)), ntr)
    off = np.concatenate([[0], np.cumsum(ntr)])

    cl = {"layer": K["layer"], "phi": K["globalPhi"],
          "eta": np.arcsinh(K["globalZ"] / np.maximum(K["globalR"], 1e-9)),
          "r": K["globalR"], "tpIdx": K["tpIdx"],
          "gphi": K.get("gClPhi"), "gcot": K.get("gClCotTheta"),
          "sphi": K.get("gSigPhi"), "scot": K.get("gSigCotTheta")}
    cev = K["event"]
    have_ang = cl["gphi"] is not None and cl["sphi"] is not None

    xi, ci, _ = join_on_module(X, K)
    is_true = (K["tpIdx"][ci] >= 0) & (X["trk_tpIdx"][xi] >= 0) & \
              (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    want_c = np.full(len(X["layer"]), -1, dtype=np.int64)
    want_c[xi[is_true]] = ci[is_true]
    gi = off[X["event"]] + X["trackIdx"].astype(np.int64)

    res = {"target_containment": SECTOR_TARGET, "f_scan": SECTOR_F,
           "grids": [g[0] for g in SECTOR_GRIDS], "layers": {}}
    for L in LAYERS:
        xl = np.flatnonzero((X["layer"] == L) & (want_c >= 0))
        if len(xl) < 200:
            continue
        cj = want_c[xl]
        g = gi[xl]
        pp, _, pe, _ = _helix_project(tr["rInv"][g], tr["phi"][g], tr["tanL"][g],
                                      tr["z0"][g], cl["r"][cj])
        dphi = np.abs(_wrap(cl["phi"][cj] - pp))
        deta = np.abs(cl["eta"][cj] - pe)
        res["layers"][L] = {
            "r_nom_cm": float(np.median(cl["r"][cl["layer"] == L])),
            "n_desired": int(len(cj)),
            "dphi_q68_mrad": float(np.percentile(dphi, 68) * 1e3),
            "dphi_q95_mrad": float(np.percentile(dphi, 95) * 1e3),
            "dphi_q99_mrad": float(np.percentile(dphi, 99) * 1e3),
            "deta_q95": float(np.percentile(deta, 95)),
            "deta_q99": float(np.percentile(deta, 99)),
        }
    if not res["layers"]:
        return

    variants = ["pos", "pos+alpha", "pos+alpha+beta"] if have_ang else ["pos"]
    tk = X["event"].astype(np.int64) * (1 << 20) + X["trackIdx"].astype(np.int64)
    k2 = tk * 8 + X["layer"].astype(np.int64)
    u2, i2 = np.unique(k2, return_inverse=True)
    ut, it_ = np.unique(u2 // 8, return_inverse=True)
    L1k = min(res["layers"])

    scan = []
    for gspec in SECTOR_GRIDS:
        for f in SECTOR_F:
            cand = {v: np.zeros(len(X["layer"])) for v in variants}
            num = {v: 0 for v in variants}
            den = 0
            for L, d in res["layers"].items():
                wphi = f * d["dphi_q99_mrad"] * 1e-3
                weta = f * d["deta_q99"]
                xl = np.flatnonzero(X["layer"] == L)
                got = _sector_counts(tr, cl, tev, cev, L, d["r_nom_cm"], wphi, weta,
                                     gspec, want_c, xl, gi[xl], have_ang)
                if got is None:
                    continue
                dsel = want_c[xl] >= 0
                den += int(dsel.sum())
                for v in variants:
                    c_, fnd = got[v]
                    cand[v] += c_
                    num[v] += int((fnd[xl] & dsel).sum())
            row = {"grid": gspec[0], "n_grids": gspec[1] * gspec[2],
                   "bins_read": (2 * gspec[3] + 1) ** 2, "f": f,
                   "bins_per_nonant_L1": float(
                       2 * np.pi / 9 / (f * res["layers"][L1k]["dphi_q99_mrad"] * 1e-3))}
            for v in variants:
                n_tl = np.bincount(i2, weights=cand[v], minlength=len(u2))
                prod = np.ones(len(ut))
                np.multiply.at(prod, it_, np.maximum(n_tl, 1.0))
                row[f"combos_{v}"] = float(prod.mean())
                row[f"cont_{v}"] = float(num[v] / max(den, 1))
            scan.append(row)
    res["scan"] = scan

    # Pareto front: the only comparison that matters is at MATCHED containment,
    # because any layout can look cheap by being undersized.
    if ax_row is not None:
        styles = {"1grid_1bin": ("o", "-"), "1grid_3x3": ("s", "-"),
                  "2grid_phi_1bin": ("^", "--"), "4grid_1bin": ("D", "--")}
        for a_, v, lab in ((ax_row[0], "pos", "position bin only"),
                           (ax_row[1], "pos+alpha+beta", "bin + both angles")):
            for gname in [g[0] for g in SECTOR_GRIDS]:
                rows = [r for r in scan if r["grid"] == gname]
                if not rows:
                    continue
                mk, ls = styles.get(gname, ("o", "-"))
                a_.plot([100 * r[f"cont_{v}"] for r in rows],
                        [r[f"combos_{v}"] for r in rows],
                        marker=mk, ls=ls, ms=4,
                        label=f"{gname} ({rows[0]['n_grids']}x store, {rows[0]['bins_read']} read)")
            a_.set_yscale("log")
            a_.set_xlabel("containment of the desired cluster [%]")
            a_.set_ylabel("combinations per track refit [count]")
            a_.set_title(f"(11) {lab}")
            a_.grid(alpha=.3)
            a_.legend(fontsize=6)
    _write_sector_table(res, out)
    out["sector_binning"] = res
    return res


def _write_sector_table(res, out):
    L1k = min(res["layers"])
    lines = [
        "FIXED eta-phi BIN SIZING for a refit-only design  (study 11)",
        "",
        "  The pre-processing has NO TRACK KNOWLEDGE: it runs in parallel with OT track",
        "  finding and must finish before any track exists. The grid is therefore",
        "  absolute, and THE BIN IS THE SELECTION -- no cone cut is applied anywhere",
        "  here. The only track-dependent step is a projection choosing which bin to",
        "  read, so whatever landed in that bin is what a refit must test.",
        "",
        "  Bins must EXCEED the cone, because the cone is centred on the projection and",
        "  a fixed grid is not. The relevant cone is the SEED one (308 um at L1, not the",
        "  refit-order 37 um): binning precedes the refit, so sizing against the refit",
        "  cone would be circular.",
        "",
        "  desired cluster = the true cluster on the crossing's OWN module, i.e. what the",
        "  refit wants. Its offset from the naive projection:",
        "",
        f"    {'L':2} {'r[cm]':>7} {'n':>8} {'dphi q68':>10} {'q95':>9} {'q99':>9}"
        f" {'deta q95':>10} {'q99':>9}",
    ]
    for L, d in sorted(res["layers"].items()):
        lines.append(f"    L{L} {d['r_nom_cm']:>7.2f} {d['n_desired']:>8} "
                     f"{d['dphi_q68_mrad']:>9.2f}m {d['dphi_q95_mrad']:>8.2f}m "
                     f"{d['dphi_q99_mrad']:>8.2f}m {d['deta_q95']:>10.5f} {d['deta_q99']:>9.5f}")
    d1 = res["layers"][L1k]
    lines += [
        "",
        f"  HEAVY TAIL, and it sets the design: at L{L1k} dphi q95 is "
        f"{d1['dphi_q95_mrad']:.1f} mrad but q99 is {d1['dphi_q99_mrad']:.1f} mrad,",
        f"  a factor {d1['dphi_q99_mrad'] / max(d1['dphi_q95_mrad'], 1e-9):.1f}. "
        "Demanding 99% containment costs that many times the width",
        "  95% would need, and the bin count per nonant falls accordingly.",
        "",
        "  GRID LAYOUTS. One grid gives the projection an arbitrary phase inside its",
        "  bin, so a one-bin read GUARANTEES NOTHING -- a cluster just past the edge is",
        "  in the neighbour. Offset grids (edges of one through centres of another) let",
        "  the track pick the grid whose bin centre it lands nearest, which is what makes",
        "  a one-bin read viable. The cost is storage: clusters are binned n_grids times.",
        "    reads = how many bins are fetched per track per layer. The projection lands",
        "    in ONE bin; a 3x3 layout also reads every bin touching it, recovering the",
        "    case where the projection sat near an edge, at 9x the bandwidth.",
        "",
        "    1grid_1bin       1 grid,  1 read   -- baseline. Guarantees nothing.",
        "    1grid_3x3        1 grid,  9 reads  -- the surrounding ring as well",
        "    2grid_phi_1bin   2 grids, 1 read   -- grids offset in phi only",
        "    4grid_1bin       4 grids, 1 read   -- grids offset in phi AND eta",
        "",
        "  f          = bin width as a multiple of that layer's q99 offset",
        "  bins/nonant= how many bins the L1 grid divides one nonant (698 mrad) into",
        "  ngrid      = how many offset copies of the grid are stored",
        "  reads      = bins fetched per track per layer; see neighbour radius below",
        "  combinations per track = MEAN number a refit must test at that setting",
        "  contain    = MEASURED containment of the desired cluster",
        "  pos        = position bin only.  +a = plus bending-angle (alpha) cut.",
        "  +a+b       = plus longitudinal-angle (beta) cut as well.",
        f"  Angle cuts at {SECTOR_ANG_NSIG:g} sigma of the ML estimate; they are free once",
        "  the cluster has been read out, and they cut candidates without costing bins.",
        "",
        f"  {'grid':<16}{'ngrid':>6}{'reads':>6}{'f':>6}{'bins/nonant':>12} | "
        + "".join(f"{'combinations ' + v:>19}" for v in ("pos", "+a", "+a+b"))
        + " | " + "".join(f"{'contain ' + v:>15}" for v in ("pos", "+a", "+a+b")),
    ]
    vs = ["pos", "pos+alpha", "pos+alpha+beta"]
    for r in res["scan"]:
        lines.append(
            f"  {r['grid']:<16}{r['n_grids']:>6}{r['bins_read']:>6}{r['f']:>6.2f}"
            f"{r['bins_per_nonant_L1']:>12.1f} | "
            + "".join(f"{r.get('combos_' + v, float('nan')):>19.1f}" for v in vs)
            + " | " + "".join(f"{100 * r.get('cont_' + v, float('nan')):>14.1f}%" for v in vs))
    txt = "\n".join(lines)
    p = os.path.join(out["_outdir"], "spix_sector_binning.txt")
    with open(p, "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"\n   (11) wrote {p}")



def study_hough_examples(X, K, P, ax_row, out):
    """(8) worked Hough transforms for example TrackingParticles.

    WHAT IS DRAWN. For each (pT, eta) cell one real TP is picked from the file
    and two populations are shown around it: a CONE (legible, shows the signal
    structure) and the full phi SECTOR one processing node would see (honest,
    shows the PU200 background it sits in). Each gets three panels: the r-z
    plane, the r-phi plane, and the r-phi plane again after z0 slicing.

    WHY A CLUSTER IS A SEGMENT, NOT A LINE. Position and direction turn at
    different rates -- phi_p = phi0 - c*r*kappa but phi_d = phi0 - 2*c*r*kappa --
    so a cluster's own angle fixes kappa_hat = (phi_p - phi_d)/(c*r) with
    sigma = sigma(phi_d)/(c*r). The Hough line is drawn dark within 1 sigma of
    that, mid to 3 sigma, faint beyond. Layers: L1 red, L2 orange, L3 blue,
    L4 green.

    THE r-z PLANE IS THE SAME CONSTRUCTION. z0 = z - r*cot(theta), so a cluster
    is a segment of slope -r there too, and the slope encodes the layer: L1's
    short lever arm makes a nearly flat, tightly-determined z0, L4's a steep and
    loose one. That is the whole reason z0 slicing works, drawn rather than
    asserted.

    ETA COVERAGE IS A RESULT, NOT A PLOTTING PROBLEM. TBPX is a barrel: measured
    on this file, the fraction of TPs lighting >=3 layers is 0.87 at |eta| 0.2-1.0
    but 0.21 at 1.4-1.8 and 0.03 above 1.8. That is the acceptance limit of
    barrel-only seeding (|eta| <~ 1.4). NOTE this is a BLOCKED DEPENDENCY, not a
    scope decision: forward coverage is wanted, but there is no PixelAV angle
    parametrisation for the disc sensors in their B-field configuration (a disc
    sits perpendicular to B where a barrel sits parallel), so a disc cluster has
    no usable angle yet. See doc/SmartPixelsSeedingAndFitting.md section 1.

    THE TP PER CELL IS THE TYPICAL ONE, NOT THE BEST ONE -- the median-ranked
    candidate by (n_layers, n_clusters). An earlier version took the argmax, which
    cherry-picked the rare 4-layer survivor at high |eta| and made the panels look
    as though seeding worked there. It also admitted 1-layer TPs to the pool, since
    requiring >=2 layers pre-selects away the very loss these panels exist to show.
    Each page states its pool's acceptance (fraction reaching >=3 and 4 layers)
    independently of which TP was drawn, so the acceptance claim does not rest on
    the single example.

    THE ANGLE SIGMAS ARE THE STORED ONES and sigGlobalClusterCotTheta is known to
    be ~21% optimistic (study 7), so the bands here are correspondingly tight.
    They are also drawn from a truth angle that neglects multiple scattering --
    see the CIRCULAR note in study (7).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    need = ("globalR", "globalZ", "globalPhi", "gClPhi", "gClCotTheta",
            "gSigPhi", "gSigCotTheta", "tpGClPhi", "tpGClCotTheta")
    if not all(k in K for k in need):
        print("   (8) SKIPPED: cluster table lacks the global-frame columns")
        out["hough_examples"] = {"skipped": "no global-frame columns"}
        for a in ax_row:
            a.axis("off")
        return

    # ---- per-TP aggregates, fully vectorised -------------------------------
    ok = (K["tpIdx"] >= 0) & (K["tpGClPhi"] > SENTINEL) & (K["tpGClCotTheta"] > SENTINEL)
    key = K["event"].astype(np.int64) * (1 << 20) + K["tpIdx"].astype(np.int64)
    ukey, inv = np.unique(key[ok], return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    laymask = np.zeros(len(ukey), dtype=np.int64)
    np.bitwise_or.at(laymask, inv, (1 << K["layer"][ok].astype(np.int64)))
    nlay = sum(((laymask >> L) & 1) for L in LAYERS)

    r_ok = K["globalR"][ok]
    half = _half_turn(K["globalPhi"][ok], K["tpGClPhi"][ok])
    kap = np.sin(half) / (C_BEND * r_ok)
    phi0 = _wrap(K["globalPhi"][ok] + half)
    # NOTE z0 here is NOT trustworthy: globalClusterCotTheta is -localCotBeta,
    # not dz/dr (see doc section 4a retraction). Kept only so the r-z panels can
    # be drawn as BLOCKED rather than silently wrong.
    z0 = K["globalZ"][ok] - r_ok * K["tpGClCotTheta"][ok]

    def mean_by(v):
        return np.bincount(inv, weights=v) / cnt

    def median_by(v):
        """Per-TP median. A MEAN is unusable for kappa: a cluster at small r has
        kappa bounded only by 1/(C*r) ~ 60, so a single bad one (secondary,
        merged cluster, broken pi convention) drags the mean by an order of
        magnitude -- which is exactly what put the truth marker at kappa = -2.7
        for a 2.5 GeV track before this was fixed."""
        order = np.lexsort((v, inv))
        vs, gs = v[order], inv[order]
        g = np.arange(len(cnt))
        lo = np.searchsorted(gs, g, "left")
        return vs[lo + (np.searchsorted(gs, g, "right") - lo) // 2]
    tp_pt = median_by(K["tpPt"][ok])
    tp_cot = median_by(K["tpGClCotTheta"][ok])
    tp_eta = np.arcsinh(tp_cot)
    tp_z0 = median_by(z0)
    # kappa's SIGN cannot be taken from the direction columns: their orientation
    # convention varies per module, so (phi_p - phi_d) is sometimes the small bend
    # and sometimes pi minus it, with either sign. |kappa| = 1/tpPt is exact from
    # truth; take the sign from how the POSITION azimuth turns with radius, since
    # phi_p = phi0 - asin(C*r*kappa) means dphi_p/dr < 0 for kappa > 0.
    rc, phi_c = K["globalR"][ok], K["globalPhi"][ok]
    ref = np.arctan2(np.bincount(inv, weights=np.sin(phi_c)),
                     np.bincount(inv, weights=np.cos(phi_c)))     # circular mean
    phi_rel = _wrap(phi_c - ref[inv])                             # wrap-safe residual
    cov = (np.bincount(inv, weights=rc * phi_rel)
           - np.bincount(inv, weights=rc) * np.bincount(inv, weights=phi_rel) / cnt)
    tp_kap = np.where(cov > 0, -1.0, 1.0) / np.maximum(tp_pt, 1e-6)
    # phi0 is circular: average via unit vectors
    tp_phi0 = np.arctan2(mean_by(np.sin(phi0)), mean_by(np.cos(phi0)))
    tp_ev = (ukey >> 20).astype(np.int64)

    clu_eta = np.arcsinh(K["globalZ"] / np.maximum(K["globalR"], 1e-6))
    res, pages = {"cells": {}}, []

    for pt_t in HOUGH_PT_POINTS:
        for eta_t in HOUGH_ETA_POINTS:
            cell = f"pt{pt_t:g}_eta{eta_t:g}"
            # nlay >= 1, NOT >= 2: requiring two layers already pre-selects away
            # the acceptance loss these panels exist to show.
            cand = (np.abs(np.abs(tp_eta) - eta_t) < 0.15) & \
                   (np.abs(np.log(np.maximum(tp_pt, 1e-6) / pt_t)) < np.log(1.3)) & (nlay >= 1)
            if not cand.any():
                res["cells"][cell] = {"found": False,
                                      "reason": "no TP within (dEta<0.15, pT within 30%)"}
                pages.append((cell, None))
                continue
            # TYPICAL, not best. Taking the argmax over (nlay, cnt) cherry-picks the
            # rare 4-layer survivor at high |eta| and hides the barrel acceptance
            # collapse -- the panel then shows an algorithm working on a track that
            # almost no track in that cell resembles. The median-ranked candidate is
            # what a track at this (pT, eta) actually looks like.
            order = np.lexsort((cnt[cand], nlay[cand]))
            idx = np.flatnonzero(cand)[order[len(order) // 2]]
            pool_ge3 = float((nlay[cand] >= 3).mean())
            pool_eq4 = float((nlay[cand] == 4).mean())
            truth = {"phi0": tp_phi0[idx], "kap": tp_kap[idx],
                     "z0": tp_z0[idx], "cot": tp_cot[idx]}
            ev = tp_ev[idx]
            same_ev = K["event"] == ev
            dphi = _wrap(K["globalPhi"] - tp_phi0[idx])
            cone = same_ev & (np.abs(dphi) < CONE_DPHI) & \
                   (np.abs(clu_eta - tp_eta[idx]) < CONE_DETA)
            sect = same_ev & (np.abs(dphi) < 0.5 * SECTOR_DPHI)
            if sect.sum() > SECTOR_DRAW_CAP:      # keep the PDF finite
                keep = np.zeros(sect.sum(), bool)
                keep[np.random.default_rng(0).choice(sect.sum(), SECTOR_DRAW_CAP, False)] = True
                si = np.flatnonzero(sect); sect = np.zeros_like(sect); sect[si[keep]] = True
            res["cells"][cell] = {
                "found": True, "event": int(ev), "tpIdx": int(ukey[idx] & ((1 << 20) - 1)),
                "tp_pt": float(tp_pt[idx]), "tp_eta": float(tp_eta[idx]),
                "n_layers": int(nlay[idx]), "n_clusters_tp": int(cnt[idx]),
                "n_cone": int(cone.sum()), "n_sector": int(sect.sum()),
                "n_candidate_tps": int(cand.sum()),
                "selection": "median-ranked by (n_layers, n_clusters) -- TYPICAL, not best",
                # the acceptance statement for this cell, independent of which TP
                # happened to be drawn
                "pool_frac_ge3_layers": pool_ge3, "pool_frac_eq4_layers": pool_eq4}
            pages.append((cell, (idx, truth, cone, sect)))

    # ---- render -------------------------------------------------------------
    pdf_path = os.path.join(out["_outdir"], "spix_hough_examples.pdf")
    with PdfPages(pdf_path) as pdf:
        for cell, payload in pages:
            fig, axs = plt.subplots(2, 3, figsize=(16.5, 9))
            if payload is None:
                for a in axs.ravel():
                    a.axis("off")
                axs[0][1].text(0.5, 0.5, f"{cell}\n\nno TrackingParticle found\n"
                               "barrel-only acceptance ends near |eta| 1.4",
                               ha="center", va="center", fontsize=13)
            else:
                idx, truth, cone, sect = payload
                kmax = max(0.6, 1.6 * abs(truth["kap"]))
                for row, (sel, nm, rast) in enumerate(
                        ((cone, "cone", False), (sect, "phi sector", True))):
                    zpred = K["globalZ"][sel] - K["globalR"][sel] * K["gClCotTheta"][sel]
                    zsig = K["globalR"][sel] * np.maximum(K["gSigCotTheta"][sel], 1e-9)
                    sel_sliced = np.zeros_like(sel)
                    sel_sliced[np.flatnonzero(sel)[
                        np.abs(zpred - truth["z0"]) < Z0_SLICE_NSIG * zsig]] = True

                    a = axs[row][0]
                    _draw_hough(a, sel, K, truth["cot"] - 0.5, truth["cot"] + 0.5, "rz", truth, rast)
                    a.axhline(truth["z0"], color="k", lw=0.8, ls="--")
                    a.plot(truth["cot"], truth["z0"], "k*", ms=11, zorder=5)
                    a.set_ylim(truth["z0"] - 15, truth["z0"] + 15)
                    a.set_xlabel(r"$\cot\theta$ [unitless]"); a.set_ylabel(r"$z_0$ [cm]")
                    a.set_title(f"{nm}: r-z plane ({int(sel.sum())} clusters)", fontsize=9)

                    for col, (s, lab) in enumerate(((sel, "all"),
                                                    (sel_sliced, "in $z_0$ slice")), 1):
                        a = axs[row][col]
                        _draw_hough(a, s, K, -kmax, kmax, "rphi", truth, rast)
                        if nlay[idx] >= 2:
                            a.plot(truth["kap"], 0.0, "k*", ms=11, zorder=5)
                            a.axvline(truth["kap"], color="k", lw=0.6, ls=":")
                        else:
                            # one layer: the curvature SIGN comes from how the position
                            # azimuth turns with radius, which a single cluster cannot
                            # give. Show |kappa| = 1/pT on both sides instead of
                            # planting a star at an arbitrary sign.
                            for sgn in (-1.0, 1.0):
                                a.axvline(sgn * abs(truth["kap"]), color="k", lw=0.6, ls=":")
                        a.set_ylim(-0.30, 0.30)
                        a.set_xlabel(r"$q/p_T$ [GeV$^{-1}$]")
                        a.set_ylabel(r"$\phi_0 - \phi_0^{\rm true}$ [rad]")
                        a.set_title(f"{nm}: r-$\\phi$, {lab} ({int(s.sum())})", fontsize=9)
                    for a in axs[row]:
                        a.grid(alpha=.25)
                c = res["cells"][cell]
                fig.suptitle(f"{cell}   |   TYPICAL TP: $p_T$={c['tp_pt']:.2f} GeV, "
                             f"$\\eta$={c['tp_eta']:+.2f}, {c['n_layers']} TBPX layers   |   "
                             f"pool of {c['n_candidate_tps']}: "
                             f"{c['pool_frac_ge3_layers']*100:.0f}% reach $\\geq$3 layers, "
                             f"{c['pool_frac_eq4_layers']*100:.0f}% reach 4   |   "
                             f"L1 red, L2 orange, L3 blue, L4 green; "
                             f"shade = 1/3$\\sigma$ of the cluster's own angle", fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    res["pdf"] = pdf_path
    res["n_pages"] = len(pages)
    res["n_cells_found"] = sum(1 for v in res["cells"].values() if v.get("found"))
    print(f"   (8) wrote {pdf_path} ({len(pages)} cells, "
          f"{res['n_cells_found']} with a TP)")

    # inline: one representative cell, before and after z0 slicing
    shown = next((c for c in ("pt2_eta0.4", "pt2_eta0", "pt5_eta0.4")
                  if res["cells"].get(c, {}).get("found")), None)
    if shown:
        idx, truth, cone, sect = dict(pages)[shown]
        kmax = max(0.6, 1.6 * abs(truth["kap"]))
        zp = K["globalZ"][cone] - K["globalR"][cone] * K["gClCotTheta"][cone]
        zs = K["globalR"][cone] * np.maximum(K["gSigCotTheta"][cone], 1e-9)
        sl = np.zeros_like(cone)
        sl[np.flatnonzero(cone)[np.abs(zp - truth["z0"]) < Z0_SLICE_NSIG * zs]] = True
        for a, (s, lab, rast) in zip(ax_row, ((cone, "cone, all", False),
                                              (sl, "cone, in $z_0$ slice", False))):
            _draw_hough(a, s, K, -kmax, kmax, "rphi", truth, rast)
            a.plot(truth["kap"], 0.0, "k*", ms=10, zorder=5)
            a.set_ylim(-0.30, 0.30); a.grid(alpha=.25)
            a.set_xlabel(r"$q/p_T$ [GeV$^{-1}$]"); a.set_ylabel(r"$\phi_0-\phi_0^{\rm true}$ [rad]")
            a.set_title(f"(8) {shown} {lab} ({int(s.sum())})", fontsize=9)
    else:
        for a in ax_row:
            a.axis("off")
    out["hough_examples"] = res


def study_seed_mode_confusion(X, K, P, ax_row, out):
    """(12) What does each seeding mode LOSE relative to the others, IT and OT?

    A per-seed efficiency cannot answer this. Two seeds that recover the SAME
    TrackingParticles and two that are perfectly complementary give identical
    per-seed numbers; only the overlap of the recovered SETS separates them. That
    is exactly how the union of the two IT pair seeds (0.907) beats the IT
    triplet alone (0.882) even though the triplet beats EITHER pair in EVERY d0
    band -- the pairs fail on different tracks.

    Reported as   M[i][j] = |found_i AND found_j| / |found_j|,
    the fraction of what column j finds that row i also finds. Diagonal is 1 by
    construction; a low off-diagonal says row i misses much of what column j
    carries. Absolute lost counts go to the side table, because a fraction
    without a count hides whether 5 or 5000 tracks are at stake.

    Each column is normalised by ITS OWN mode's finds, so IT and OT sit in one
    matrix despite different notions of findable (IT: a cluster on all three
    layers; OT: a stub on both seed layers and one projection layer).
    """
    M = _ttc()

    need = ("globalR", "globalZ", "globalPhi", "gClPhi", "gClCotTheta",
            "gSigPhi", "gSigCotTheta", "sigY", "tpIdx", "tpPt", "tp_d0")
    if any(c not in K for c in need):
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.5, 0.5, "seed-mode confusion: input lacks\n"
                       + ", ".join(c for c in need if c not in K),
                       ha="center", va="center", fontsize=9)
        out["seed_mode_confusion"] = {"skipped": [c for c in need if c not in K]}
        return

    D = {"layer": K["layer"], "globalR": K["globalR"], "globalZ": K["globalZ"],
         "globalPhi": K["globalPhi"], "globalClusterPhi": K["gClPhi"],
         "globalClusterCotTheta": K["gClCotTheta"],
         "sigGlobalClusterPhi": K["gSigPhi"],
         "sigGlobalClusterCotTheta": K["gSigCotTheta"], "sigY": K["sigY"],
         "tpIdx": K["tpIdx"], "tpPt": K["tpPt"], "tp_d0": K["tp_d0"], "event": K["event"]}
    M.set_limits(M.triplets_for_budget(M.DEFAULT_PAIR_BUDGET_GB),
                 min(6.0, M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
    PTMIN = 2.0
    ti = M.build_tp_index(D)
    Q = M.it_prepare(D, None)
    allidx = np.arange(len(D["layer"]))
    found, findable = {}, {}
    for cname, cfg in M.IT_CONFIGS.items():
        jobs = [((la, lb), lc, False) for (la, lb), lc in cfg["pairs"]]
        jobs.append((cfg["triplet"][:2], cfg["triplet"][2], True))
        for (la, lb), lc, disp in jobs:
            lbl = f"IT {cname} " + ("triplet" if disp else f"L{la}L{lb}>L{lc}")
            try:
                o = M.it_pair_seed(D, Q, allidx, la, lb, lc, PTMIN, True, disp, 0.0)
            except M.TooWide:
                continue
            if "_trip" in o:
                ga, gb, gc = o["_trip"]
                found[lbl] = M.recovered_keys(D, ga, gb, gc)
                # IT findable: a TP above ptmin with a cluster on all three
                # layers of this configuration. Same for all seeds of a config.
                findable[lbl] = M.findable_keys(D, la, lb, lc, PTMIN)
    # OT, from the same files
    try:
        srcs = out.get("_inputs") or []
        if srcs:
            O, onev, have = M.load_ot(",".join(srcs), None)
            if "tpIdx" in have:
                cal = M.calibrate_ot_bend(",".join(srcs), None)
                base = np.arange(len(O["layer"]))
                base = base[O["eta"][base] <= M.ETA_MATCHED]
                for sname, S in M.OT_SEEDS.items():
                    if S["disc"]:
                        continue
                    oo = M.ot_seed_cost(O, base, sname, PTMIN, False, cal, M.ETA_MATCHED)
                    fk = oo.get("_found_keys")
                    if fk is not None and len(fk):
                        found[f"OT {sname}"] = fk
                        findable[f"OT {sname}"] = oo.get(
                            "_findable_keys", np.empty(0, np.int64))
    except Exception as exc:                      # OT is a bonus, not a blocker
        print(f"    seed-mode confusion: OT skipped ({type(exc).__name__}: {exc})")

    labels = sorted(found, key=lambda t: (not t.startswith("IT"), t))
    if len(labels) < 2:
        for a in ax_row:
            a.axis("off")
        return
    # Build the per-TP d0/pT table from build_tp_index, which selects tpIdx >= 0.
    # An earlier revision keyed on np.maximum(tpIdx, 0), which folds EVERY noise
    # cluster onto real TP index 0 and corrupts that TP's d0 and pT in every
    # event -- it inflated the |d0| > 500 um count for one seed from ~96 to 455.
    uk, tpd0, tppt = ti["key"], ti["d0"], ti["pt"]
    # RESTRICT EVERY MODE TO pT >= PTMIN. A seed recovers sub-threshold
    # TrackingParticles too -- the gates are not sharp -- and measured, 8,037 of
    # one seed's 81,870 recovered TPs are outside its own findable set, 3,438 of
    # them sitting above d0 = 500 um. Left in, they inflate the inclusive and d0
    # facets while falling outside every pT facet, so the two sets of facets stop
    # describing the same population and neither reconciles with the cost model's
    # efficiency, which counts findable TPs above threshold only.
    inband = uk[tppt >= PTMIN]
    found = {k: np.intersect1d(v, inband, assume_unique=True)
             for k, v in found.items()}
    findable = {k: np.intersect1d(np.unique(v), inband, assume_unique=True)
                for k, v in findable.items()}

    def restrict(lo, hi, arr):
        sel = uk[(arr >= lo) & (arr < hi)]
        return ({k: np.intersect1d(v, sel, assume_unique=True) for k, v in found.items()},
                {k: np.intersect1d(v, sel, assume_unique=True) for k, v in findable.items()})

    facets = {"inclusive": (found, findable)}
    for nm, lo, hi in (("|d0|<100um", 0.0, 100e-4), ("|d0| 100-500um", 100e-4, 500e-4),
                       ("|d0|>500um", 500e-4, 1e9)):
        facets[nm] = restrict(lo, hi, tpd0)
    for nm, lo, hi in (("pT 2-5", 2.0, 5.0), ("pT 5-10", 5.0, 10.0),
                       ("pT 10-20", 10.0, 20.0), ("pT 20+", 20.0, 1e9)):
        facets[nm] = restrict(lo, hi, tppt)

    res, mats = {}, {}
    for fn, (fd, fa) in facets.items():
        n = len(labels)
        frac = np.full((n, n), np.nan); lost = np.zeros((n, n), np.int64)
        for j, lj in enumerate(labels):
            fj = fd[lj]
            for i, li in enumerate(labels):
                both = len(np.intersect1d(fd[li], fj, assume_unique=True))
                frac[i, j] = both / len(fj) if len(fj) else np.nan
                lost[i, j] = len(fj) - both
        mats[fn] = (frac, lost)
        res[fn] = {"n_found": {l: int(len(fd[l])) for l in labels},
                   # DENOMINATOR, per mode and per facet. A recovered count
                   # without the findable count it came from cannot be compared
                   # between modes, because IT and OT do not share a definition
                   # of findable and the bands hold different populations.
                   "n_findable": {l: int(len(fa.get(l, ()))) for l in labels},
                   "efficiency": {l: (len(fd[l]) / len(fa[l])
                                      if len(fa.get(l, ())) else None)
                                  for l in labels},
                   "frac": frac.tolist(), "lost": lost.tolist()}
    out["seed_mode_confusion"] = {"labels": labels, "pt_min": PTMIN, "facets": res}

    short = [l.replace("IT ", "").replace("OT ", "OT:") for l in labels]
    for ax, fn in zip(ax_row, ("inclusive", "|d0|>500um", "pT 20+")):
        _draw_confusion(ax, mats[fn][0], short, fn,
                        max(res[fn]["n_found"].values()) if res[fn]["n_found"] else 0)
    _confusion_side_artifacts(out, labels, mats, res)


def _draw_confusion(ax, frac, short, title, nmax):
    """Sequential single hue, light->dark: this encodes magnitude, not identity."""
    from matplotlib.colors import LinearSegmentedColormap
    BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE)
    n = len(short)
    ax.imshow(frac, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(f"{title}   (largest mode: {nmax:,} TPs)", fontsize=9, pad=6)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for i in range(n):
        for j in range(n):
            if np.isnan(frac[i, j]):
                continue
            ax.text(j, i, f"{frac[i,j]:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=5.4, color=("#fcfcfb" if frac[i, j] > 0.62 else "#0b0b0b"))
    ax.set_xlabel("found by column mode [TPs]", fontsize=7)
    ax.set_ylabel("also found by row mode [TPs]", fontsize=7)


def _confusion_side_artifacts(out, labels, mats, res):
    """All eight facets as their own figure, plus the absolute lost counts."""
    import matplotlib.pyplot as _plt
    od = out.get("_outdir", ".")
    short = [l.replace("IT ", "").replace("OT ", "OT:") for l in labels]
    names = list(mats)
    ncol = 4; nrow = int(np.ceil(len(names) / ncol))
    f2, axs = _plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.1 * nrow),
                            facecolor="#fcfcfb", squeeze=False)
    for ax, fn in zip(axs.ravel(), names):
        _draw_confusion(ax, mats[fn][0], short, fn,
                        max(res[fn]["n_found"].values()) if res[fn]["n_found"] else 0)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    f2.suptitle("Seed-mode complementarity: fraction of each column mode's "
                "TrackingParticles also recovered by the row mode", fontsize=11)
    f2.tight_layout()
    p = os.path.join(od, "spix_seed_mode_confusion.png")
    f2.savefig(p, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    _plt.close(f2)
    t = os.path.join(od, "spix_seed_mode_confusion.txt")
    with open(t, "w") as fh:
        for fn in names:
            frac, lost = mats[fn]
            fh.write(f"=== {fn}\n")
            fh.write(f"{'mode':<28}{'found':>11}{'findable':>11}{'eff':>8}\n")
            for l in labels:
                nfo = res[fn]["n_found"][l]; nfa = res[fn]["n_findable"][l]
                e = res[fn]["efficiency"][l]
                fh.write(f"{l:<28}{nfo:>11,d}{nfa:>11,d}"
                         + (f"{e:>8.3f}\n" if e is not None else f"{'-':>8}\n"))
            fh.write(f"\nABSOLUTE TPs found by COLUMN but missed by ROW\n")
            fh.write(f"{'':<28}" + "".join(f"{s[:10]:>11}" for s in short) + "\n")
            for i, l in enumerate(labels):
                fh.write(f"{l:<28}" + "".join(f"{lost[i,k]:>11,d}"
                                              for k in range(len(labels))) + "\n")
            fh.write("\n")
    print(f"    wrote {p}\n    wrote {t}")


def study_seed_composition(X, K, P, ax_row, out):
    """(13) Can the doublets be aimed at what the TRIPLET misses, in parallel?

    In L1L2L3 the triplet alone reaches 0.882 and the doublets 0.861 / 0.860,
    but all three together reach 0.923 -- so the doublets are largely REDUNDANT
    with the triplet while paying full combinatorial cost. The obvious fix,
    running the triplet first and letting the doublets see only the clusters it
    did not consume, SERIALISES the two and spends latency a trigger does not
    have. So the restriction must be evaluable from the doublet's OWN pair.

    MEASURED, where the triplet actually fails. Its misses concentrate at
    |kappa| in 0.40-0.50, i.e. pT just above threshold, where the three-point
    solve's own sigma(kappa) ~ 0.022 scatters tracks across its |kappa| <=
    kappa_max cut. The doublets recover 74.8% of the misses there against
    0.34-0.43 in every other kappa band, so 76% of their entire non-redundant
    value sits in one band holding 34% of the findable population.

    The regions where the triplet fails for OTHER reasons are not targetable:
    at |cot theta| > 3 or |z0| > 15 cm the doublets fail alongside it, recovering
    0.05 and 0.008 of its misses.

    So the variant scanned here is a LOWER gate on the doublet's own |kappa| --
    equivalently an UPPER pT bound -- which turns the doublets into pure
    threshold-recovery seeds that run concurrently with the triplet.
    """
    import importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "tracklet_topology_cost.py")
    _sp = _ilu.spec_from_file_location("_ttc2", _p)
    M = _ilu.module_from_spec(_sp); _sp.loader.exec_module(M)
    need = ("globalR", "globalZ", "globalPhi", "gClPhi", "gClCotTheta",
            "gSigPhi", "gSigCotTheta", "sigY", "tpIdx", "tpPt")
    if any(c not in K for c in need):
        for a in ax_row:
            a.axis("off")
        out["seed_composition"] = {"skipped": [c for c in need if c not in K]}
        return
    D = {"layer": K["layer"], "globalR": K["globalR"], "globalZ": K["globalZ"],
         "globalPhi": K["globalPhi"], "globalClusterPhi": K["gClPhi"],
         "globalClusterCotTheta": K["gClCotTheta"],
         "sigGlobalClusterPhi": K["gSigPhi"],
         "sigGlobalClusterCotTheta": K["gSigCotTheta"], "sigY": K["sigY"],
         "tpIdx": K["tpIdx"], "tpPt": K["tpPt"], "event": K["event"]}
    M.set_limits(M.triplets_for_budget(M.DEFAULT_PAIR_BUDGET_GB),
                 min(6.0, M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
    PTMIN = 2.0
    KMINS = (0.0, 0.25, 0.30, 0.35, 0.40, 0.45)
    Q = M.it_prepare(D, None)
    allidx = np.arange(len(D["layer"]))
    nev = int(D["event"].max()) + 1 if len(D["event"]) else 1
    COST = ("tracklets", "match_cand", "tracks_to_fit")
    res = {}
    for cname, cfg in M.IT_CONFIGS.items():
        (pa, la_), (pb, lb_) = cfg["pairs"]
        tri = cfg["triplet"]
        fk = M.findable_keys(D, tri[0], tri[1], tri[2], PTMIN)
        ot = M.it_pair_seed(D, Q, allidx, tri[0], tri[1], tri[2], PTMIN, True, True, 0.0)
        kt = (M.recovered_keys(D, *ot["_trip"]) if "_trip" in ot
              else np.empty(0, np.int64))
        ct = {c: float(ot.get(c, 0)) / nev for c in COST}
        scan = []
        for kmin in KMINS:
            oa = M.it_pair_seed(D, Q, allidx, pa[0], pa[1], la_, PTMIN, True,
                                False, 0.0, kap_min=kmin)
            ob = M.it_pair_seed(D, Q, allidx, pb[0], pb[1], lb_, PTMIN, True,
                                False, 0.0, kap_min=kmin)
            ka = M.recovered_keys(D, *oa["_trip"]) if "_trip" in oa else np.empty(0, np.int64)
            kb = M.recovered_keys(D, *ob["_trip"]) if "_trip" in ob else np.empty(0, np.int64)
            uni = np.unique(np.concatenate([kt, ka, kb]))
            scan.append({"kap_min": kmin,
                         "pt_max": (None if kmin == 0 else 1.0 / kmin),
                         "eff_union": float(np.isin(fk, uni).mean()),
                         "beyond_triplet": int(len(np.setdiff1d(np.union1d(ka, kb), kt))),
                         "cost": {c: (float(oa.get(c, 0)) + float(ob.get(c, 0))) / nev
                                  for c in COST}})
        res[cname] = {"n_findable": int(len(fk)), "n_events": nev,
                      "eff_triplet": float(np.isin(fk, kt).mean()),
                      "triplet_cost": ct, "scan": scan}
    out["seed_composition"] = {"pt_min": PTMIN, "configs": res}
    _draw_composition(ax_row, res)
    _composition_side_table(out, res)


def _draw_composition(ax_row, res):
    C = ["#2a78d6", "#e8833a"]
    INK2 = "#52514e"
    names = list(res)
    ax = ax_row[0]
    for ci, cn in enumerate(names):
        e = res[cn]
        x = [r["cost"]["match_cand"] / 1e3 for r in e["scan"]]
        y = [r["eff_union"] for r in e["scan"]]
        ax.plot(x, y, "-o", color=C[ci], ms=5, lw=2, label=cn, zorder=3)
        for r, xx, yy in zip(e["scan"], x, y):
            if r["kap_min"] in (0.0, 0.35, 0.45):
                ax.annotate(f"{r['kap_min']:.2f}", (xx, yy), fontsize=6,
                            xytext=(3, -9), textcoords="offset points", color=INK2)
        ax.axhline(e["eff_triplet"], color=C[ci], lw=0.9, ls=":")
    ax.set_xlabel("doublet candidate triplets per event  [thousands]", fontsize=8)
    ax.set_ylabel("union efficiency, triplet + doublets [fraction]", fontsize=8)
    ax.set_title("cost bought back by the |kappa| gate\n(dotted = triplet alone; "
                 "labels = |kappa| min)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=7)
    ax.tick_params(labelsize=7, colors=INK2)

    ax = ax_row[1]
    for ci, cn in enumerate(names):
        e = res[cn]
        base = e["scan"][0]
        ax.plot([r["cost"]["match_cand"] / base["cost"]["match_cand"] for r in e["scan"]],
                [r["eff_union"] / base["eff_union"] for r in e["scan"]],
                "-o", color=C[ci], ms=5, lw=2, label=cn, zorder=3)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("doublet cost relative to no gate [ratio, unitless]", fontsize=8)
    ax.set_ylabel("union efficiency relative to no gate [ratio, unitless]", fontsize=8)
    ax.set_title("what the gate keeps per unit saved", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=7)
    ax.tick_params(labelsize=7, colors=INK2)

    ax = ax_row[2]
    w = 0.38
    xs = np.arange(len(res[names[0]]["scan"]))
    for ci, cn in enumerate(names):
        ax.bar(xs + ci * w, [r["beyond_triplet"] for r in res[cn]["scan"]], w,
               color=C[ci], label=cn)
    ax.set_xticks(xs + w / 2)
    ax.set_xticklabels([("none" if r["kap_min"] == 0 else f"{r['kap_min']:.2f}")
                        for r in res[names[0]]["scan"]], fontsize=7)
    ax.set_xlabel("doublet |kappa| lower gate [GeV$^{-1}$]", fontsize=8)
    ax.set_ylabel("TPs recovered beyond the triplet [count]", fontsize=8)
    ax.set_title("non-redundant recovery retained", fontsize=9)
    ax.grid(alpha=0.25, axis="y", lw=0.6); ax.legend(fontsize=7)
    ax.tick_params(labelsize=7, colors=INK2)


def _composition_side_table(out, res):
    p = os.path.join(out.get("_outdir", "."), "spix_seed_composition.txt")
    with open(p, "w") as fh:
        fh.write("Doublets aimed at the triplet's failure band, running in PARALLEL.\n"
                 "The gate is a lower bound on the doublet's own |kappa|, i.e. an\n"
                 "UPPER pT bound, so it needs nothing from the triplet.\n\n")
        for cn, e in res.items():
            fh.write(f"=== {cn}  findable {e['n_findable']:,} over {e['n_events']} events\n")
            fh.write(f"  triplet alone: eff {e['eff_triplet']:.4f}, "
                     f"to-fit/ev {e['triplet_cost']['tracks_to_fit']:,.0f}\n")
            fh.write(f"  {'|kap|min':>9}{'pT<=':>8}{'union eff':>11}{'beyond':>9}"
                     f"{'pairs/ev':>11}{'cand/ev':>11}{'fit/ev':>9}{'cost rel':>10}\n")
            b = e["scan"][0]["cost"]["match_cand"]
            for r in e["scan"]:
                pt = "none" if r["pt_max"] is None else f"{r['pt_max']:.1f}"
                fh.write(f"  {r['kap_min']:>9.2f}{pt:>8}{r['eff_union']:>11.4f}"
                         f"{r['beyond_triplet']:>9,d}{r['cost']['tracklets']:>11,.0f}"
                         f"{r['cost']['match_cand']:>11,.0f}"
                         f"{r['cost']['tracks_to_fit']:>9,.0f}"
                         f"{r['cost']['match_cand']/b:>9.2f}x\n")
            fh.write("\n")
    print(f"    wrote {p}")


# ---- (15) seed menus per SmartPixels build ---------------------------------
SEED_MENU_MASKS = ["AAAA", "AAAI", "AAIA", "AIAA", "IAAA", "AAII", "AIAI",
                   "AIIA", "IAAI", "IAIA", "IIAA", "AIII", "IAII", "IIAI", "IIIA"]
IL_OF = {0: 1, 1: 2, 2: 3, 3: 4}
OT_BARREL = (11, 12, 13, 14, 15, 16)


def study_seed_menu_by_build(X, K, P, ax_row, out):
    """(15) Which seed menu does each SmartPixels build support, and how good is it?

    activeSP says which IT layers are INSTRUMENTED. An uninstrumented layer emits
    nothing at L1 -- not position without angle, nothing -- so it cannot appear in
    a seed at all. A 1010 build (AIAI) therefore has exactly one IT-only doublet,
    IL1+IL3, and it can only project to OT layers, because IL2 and IL4 do not
    exist for it. The OT is always the standard full barrel.

    Ranking seeds by cost answers the wrong question: the cheapest eight may all
    recover the same TrackingParticles. So each build's menu is composed GREEDILY
    BY MARGINAL GAIN -- best single seed, then whichever raises the union most --
    and the table reports what each entry uniquely adds, not merely what it finds.

    A seed's recovered set depends only on its own three layers, never on what
    else is instrumented, so every distinct seed is run ONCE and each build's menu
    is composed from that cache. Without it this would be fifteen full passes.
    """
    import importlib.util as _ilu
    _here = os.path.dirname(os.path.abspath(__file__))

    def _mod(name):
        sp = _ilu.spec_from_file_location("_" + name,
                                          os.path.join(_here, name + ".py"))
        m = _ilu.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m
    TF = _mod("tp_findability")
    SA = _mod("seed_arity")
    M = TF.M
    # THIS STUDY STREAMS; the rest of the omnibus does not. load() concatenates
    # every input up front with no event cap, which is fine for one 100-event
    # file and tens of GB across ten, so the census gets its own input spec and
    # the full ttbar set is passed here rather than to -i.
    spec = out.get("_menu_inputs") or ",".join(out.get("_inputs") or [])
    if not spec:
        for a in ax_row:
            a.axis("off")
        out["seed_menu_by_build"] = {"skipped": "needs _inputs"}
        return
    PTMIN = float(out.get("_menu_ptmin", 2.0))
    NEV = int(out.get("_menu_nev", 100))
    # CHUNKED AND CACHED. Every seed is run once per chunk of events and reduced
    # immediately to a per-TrackingParticle bitmap, so the thousand-event set
    # never needs the whole unified IT+OT table (5.2 GB) or a cache of every
    # seed's recovered key set in memory at once. The shards are keyed on the
    # inputs, so a rerun of this study is a load, not a recomputation.
    try:
        C = TF.build(spec, NEV, PTMIN, list(TF.ALL_LAYERS), 2,
                     int(out.get("_menu_chunk", 16)),
                     out.get("_menu_cache", os.path.join(_here, "cache")),
                     out.get("_menu_cache_mode", "auto"),
                     masks=SEED_MENU_MASKS,
                     budget_gb=float(out.get("_menu_budget_gb", 0.4)),
                     rss_gb=float(out.get("_menu_rss_gb", 8.0)),
                     export_tracks=int(out.get("_menu_export_tracks", 0)))
    except Exception as exc:
        for a in ax_row:
            a.axis("off")
        out["seed_menu_by_build"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        return
    nev = max(C["n_events"], 1)
    rmed = {int(k): float(v) for k, v in C["calibration"]["r_median"].items()}
    idx_of = {t: i for i, t in enumerate(C["seed_tags"])}
    above = C["pt"] >= PTMIN

    def on_layer(L):
        return (C["hit_it"] if L <= 4 else C["hit_ot"]) & np.uint8(
            1 << ((L - 1) if L <= 4 else (L - 11))) != 0
    ON = {L: on_layer(L) for L in (1, 2, 3, 4) + OT_BARREL}
    # Per-seed recovered sets, as boolean masks over the census rather than key
    # arrays: the greedy composition is then a popcount of (mask & ~have) instead
    # of an np.setdiff1d over 1e5-element int64 arrays, which is what let the
    # composition stay cheap once the census grew to millions of TPs.
    FOUND, COST = {}, {}
    for tag, i in idx_of.items():
        c = C["counters"].get(i, {})
        if c.get("infeasible"):
            continue
        sd = C["seeds"][i]
        m = TF.seed_mask(C, i)
        if not m.any():
            continue
        q = C["qual"].get(i, np.zeros((0, 5), np.float32))
        FOUND[tag] = m
        COST[tag] = {"cand": c.get("cand", 0.0) / nev,
                     "arity": sd.arity, "note": C["seed_notes"][i],
                     "nlayer": c.get("nlayer_sum", 0.0) / max(c.get("fit", 1.0), 1.0),
                     "fit": c.get("fit", 0.0) / nev,
                     "fake": 1.0 - c.get("trip_true", 0.0) / max(c.get("trip", 0.0), 1.0),
                     # POST-FIT, from the KF emulation: columns are
                     # (dkappa, dd0, dcot, dz0, nhit), not the old
                     # seed-level (dkappa, dcot, dz0).
                     "sig_kappa": TF.robust_sigma(q[:, 0]),
                     "sig_d0_cm": TF.robust_sigma(q[:, 1]),
                     "sig_cot": TF.robust_sigma(q[:, 2]),
                     "sig_z0_cm": TF.robust_sigma(q[:, 3]),
                     "mean_nhit": float(q[:, 4].mean()) if len(q) else float("nan")}
    if not FOUND:
        for a in ax_row:
            a.axis("off")
        out["seed_menu_by_build"] = {"skipped": "no viable seeds"}
        return
    # TWO PASSES. The common denominator is every TP that ANY seed of ANY build
    # recovers. A per-build denominator -- "TPs some seed of this build found" --
    # is self-referential and makes builds incomparable: it scored a one-layer
    # build (AIII, 0.915) above the fully instrumented one (AAAA, 0.909), an
    # artefact of the crippled build being graded on its own reduced reach.
    DEN = np.zeros(len(C["key"]), bool)
    for m in FOUND.values():
        DEN |= m
    n_den = int(DEN.sum())

    builds = {}
    for mask in SEED_MENU_MASKS:
        il = [IL_OF[i] for i, ch in enumerate(mask) if ch == "A"]
        layers = il + list(OT_BARREL)
        found, cost = {}, {}
        # PER BUILD, including arity: which seeds exist depends on which IT
        # layers are instrumented, and the locality rule that governs mixed
        # seeds is anchored on the OUTERMOST instrumented IT layer, so a build's
        # seed list is not a subset of a richer build's.
        for sd in SA.enumerate_seeds(il):
            la, lb = sd.layers[0], sd.layers[1]
            if float((above & ON[la] & ON[lb]).sum()) / nev < 20.0:
                continue
            if sd.tag in FOUND:
                found[sd.tag], cost[sd.tag] = FOUND[sd.tag], COST[sd.tag]
        if not found:
            builds[mask] = {"n_seeds": 0}
            continue

        def compose(weighted):
            """Greedy menu. weighted=False maximises marginal TPs; True maximises
            marginal TPs PER CANDIDATE, which is what a cost-limited system wants
            and which demotes seeds that buy a few hundred tracks for 1e6
            candidates."""
            m, hv, pl = [], np.zeros(len(C["key"]), bool), dict(found)
            while pl and len(m) < 8:
                gains = {t: int((pl[t] & ~hv).sum()) for t in pl}

                def score(t):
                    return (gains[t] / max(cost[t]["cand"], 1.0) if weighted
                            else gains[t])
                best = max(pl, key=score)
                if gains[best] <= 0:
                    break
                hv |= pl.pop(best)
                m.append({"seed": best, "marginal_tps": gains[best],
                          "cum_eff": float(hv.sum() / max(n_den, 1)),
                          "marg_per_kcand": 1e3 * gains[best]
                                            / max(cost[best]["cand"], 1.0),
                          **cost[best]})
            return m
        menu = compose(False)
        menu_cost = compose(True)
        builds[mask] = {"n_seeds": len(found), "denominator": n_den,
                        "n_it_layers": len(il), "it_layers": [SA.NAME[L] for L in il],
                        "menu": menu, "menu_cost_weighted": menu_cost,
                        "eff_3_cost_weighted": (menu_cost[2]["cum_eff"]
                                                if len(menu_cost) > 2 else 0.0),
                        "cand_3_cost_weighted": sum(m["cand"] for m in menu_cost[:3]),
                        "eff_3": menu[2]["cum_eff"] if len(menu) > 2 else
                                 (menu[-1]["cum_eff"] if menu else 0.0),
                        "cand_3": sum(m["cand"] for m in menu[:3])}
    out["seed_menu_by_build"] = {"n_events": nev, "pt_min": PTMIN,
                                 "inputs": spec,
                                 "n_tp_census": int(len(C["key"])),
                                 "n_seeds_run": len(C["seed_tags"]),
                                 "cache_dir": C["cache_dir"], "builds": builds}
    _draw_menu_by_build(ax_row, builds)
    _menu_by_build_table(out, builds)


def _draw_menu_by_build(ax_row, builds):
    ok = {k: v for k, v in builds.items() if v.get("menu")}
    if not ok:
        for a in ax_row:
            a.axis("off")
        return
    order = sorted(ok, key=lambda k: (-ok[k]["n_it_layers"], k))
    C = {4: "#2a78d6", 3: "#4b9f6e", 2: "#e8833a", 1: "#b1524f"}
    ax = ax_row[0]
    for k in order:
        b = ok[k]
        y = [m["cum_eff"] for m in b["menu"]]
        ax.plot(range(1, len(y) + 1), y, "-o", ms=3.5, lw=1.6,
                color=C[b["n_it_layers"]], alpha=0.85,
                label=k if b["n_it_layers"] in (1, 4) else None)
    ax.set_xlabel("seeds in the menu [count]", fontsize=8)
    ax.set_ylabel("cumulative efficiency [fraction]", fontsize=8)
    ax.set_title("what each build's menu reaches\n(colour = instrumented IT layers)",
                 fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    ax = ax_row[1]
    xs = np.arange(len(order))
    ax.bar(xs, [ok[k]["eff_3"] for k in order],
           color=[C[ok[k]["n_it_layers"]] for k in order])
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=90, fontsize=6.5)
    ax.set_ylabel("efficiency of the best three seeds [fraction]", fontsize=8)
    ax.set_title("three-seed menu, per build", fontsize=9)
    ax.grid(alpha=0.25, axis="y", lw=0.6); ax.tick_params(labelsize=7)
    ax = ax_row[2]
    for k in order:
        b = ok[k]
        ax.scatter(b["cand_3"], b["eff_3"], s=55, color=C[b["n_it_layers"]],
                   edgecolor="#fcfcfb", lw=1.0, zorder=3)
        ax.annotate(k, (b["cand_3"], b["eff_3"]), fontsize=5.5,
                    xytext=(4, 3), textcoords="offset points", color="#52514e")
    ax.set_xscale("log")
    ax.set_xlabel("candidate triplets per event, best three seeds [count]", fontsize=8)
    ax.set_ylabel("efficiency of those three [fraction]", fontsize=8)
    ax.set_title("what each build costs for what it reaches", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.tick_params(labelsize=7)


def _menu_by_build_table(out, builds):
    p = os.path.join(out.get("_outdir", "."), "spix_seed_menu_by_build.txt")
    with open(p, "w") as fh:
        fh.write("Seed menu per SmartPixels build. activeSP mask: A = instrumented,\n"
                 "I = not. An uninstrumented IT layer emits nothing at L1 and cannot\n"
                 "appear in a seed. The OT is always the standard full barrel.\n"
                 "Menus are composed greedily by MARGINAL gain, not by cost.\n\n")
        for mask, b in builds.items():
            if not b.get("menu"):
                fh.write(f"=== {mask}: no viable seeds\n\n")
                continue
            fh.write(f"=== {mask}   IT layers {','.join(b['it_layers']) or 'none'}"
                     f"   {b['n_seeds']} viable seeds   denominator {b['denominator']:,}\n")
            for lbl, key in (("greedy by MARGINAL GAIN", "menu"),
                             ("greedy by MARGINAL GAIN PER CANDIDATE", "menu_cost_weighted")):
                fh.write(f"  -- {lbl}\n")
                fh.write(f"  {'#':>2} {'seed':<18}{'marginal':>10}{'cum eff':>9}"
                         f"{'marg/kcand':>12}{'cand/ev':>11}{'fit/ev':>8}{'fake':>7}"
                         f"{'ar':>4}{'nlay':>6}{'sig(kap)':>10}{'sig(d0)um':>11}"
                         f"{'sig(z0)um':>11}{'sig(cot)':>10}\n")
                for i, m in enumerate(b.get(key, []), 1):
                    um = lambda k: (m.get(k, float("nan")) * 1e4
                                    if m.get(k, float("nan")) == m.get(k, float("nan"))
                                    else float("nan"))
                    fh.write(f"  {i:>2} {m['seed']:<18}{m['marginal_tps']:>10,d}"
                             f"{m['cum_eff']:>9.3f}"
                             f"{m.get('marg_per_kcand', float('nan')):>12.1f}"
                             f"{m['cand']:>11,.0f}{m['fit']:>8,.0f}{m['fake']:>7.3f}"
                             f"{m.get('arity', 0):>4d}{m.get('nlayer', float('nan')):>6.1f}"
                             f"{m.get('sig_kappa', float('nan')):>10.4f}"
                             f"{um('sig_d0_cm'):>11,.0f}{um('sig_z0_cm'):>11,.0f}"
                             f"{m.get('sig_cot', float('nan')):>10.4f}\n")
                fh.write("\n")
            fh.write("  marg/kcand = marginal TPs per 1000 candidate triplets per event.\n"
                     "  nhit and the sigmas are POST-FIT, from the 5-parameter KF\n"
                     "  emulation of the OT track finder run on the seed's clusters\n"
                     "  plus one projected hit per remaining layer -- not the seed's\n"
                     "  own three points. Following the projections is worth ~2x on\n"
                     "  sigma(kappa), so the seed-level number was pessimistic.\n\n")
    print(f"    wrote {p}")


# ---------------------------------------------------------------------------
# SECTIONS. The studies fall into four questions, and the figure is now grouped
# and banner-labelled by them rather than being one undifferentiated stack. Each
# entry is (title, function, n_panels); a section is a title plus a one-line
# statement of what the section is for, printed on the figure and in the log.
# ---------------------------------------------------------------------------
# ---- the two IT-only L1L2L3 designs, named so they cannot be conflated -------
# BASELINE DOUBLETS   triplet + L1L2->L3 + L2L3->L1, both doublets ungated.
#                     Union 0.9231, doublet cost 17,683 candidate triplets/event.
# RECOVERY DOUBLETS   the same three seeds, but each doublet carries a lower gate
#                     on its OWN |kappa| (>= 0.35, i.e. pT <= 2.9 GeV), aiming it
#                     at the band where the triplet fails. Union 0.9185 for 0.35x
#                     the doublet cost. Runs in PARALLEL with the triplet.
# L2L3L4 is retained in the JSON for the record but the design focus is L1L2L3.
IT_DESIGNS = {"baseline doublets": 0.0, "recovery doublets": 0.35}


def study_combined_it_ot(X, K, P, ax_row, out):
    """(14) A combined IT+OT tracker: which seeds, and what does projection cost?

    Everything so far treated IT and OT as separate systems joined only at the
    fit. If instead one system seeds, projects and fits across both, the seed
    choice changes, because the two subsystems are complementary in a very
    specific way. MEASURED on truth-matched pairs, 1000 PU200 ttbar events:

      seed pair        dr [cm]   sigma(kappa)   sigma(z0)
      IT L1 + IT L2       3.1       0.0306         73 um     <- best z0
      IT L2 + IT L3       4.3       0.0170        109 um
      IT L3 + OT L1      14.4       0.0093        550 um
      IT L4 + OT L2      22.7       0.0074        550 um
      OT L1 + OT L2      12.5       0.0085      2,483 um     <- best kappa, worst z0

    The IT buys LONGITUDINAL precision -- z0 extrapolates from a short radius and
    its clusters are finely segmented in z -- while a long lever arm buys
    CURVATURE precision. A mixed IT+OT pair also nearly eliminates the d0
    problem: kappa_bias falls from 10.34 per cm (IT L1L2) to 0.49 (IT L4+OT L1),
    so at d0 = 500 um the curvature slack is 5% of kappa_max instead of 103%.

    The projection cost then separates into the WORK (objects inside the
    3-sigma z road, which is what the search actually scans) and the BACKGROUND
    that survives the joint phi-and-z window. The real hit is present by
    construction, so the second number is fakes per projection, not track length.
    """
    M = _ttc()
    srcs = out.get("_inputs") or []
    need = ("globalR", "globalZ", "globalPhi", "tpIdx", "tpPt")
    if not srcs or any(c not in K for c in need):
        for a in ax_row:
            a.axis("off")
        out["combined_it_ot"] = {"skipped": "needs _inputs and IT truth columns"}
        return
    PT, NSIG = 2.0, 3.0
    src = ",".join(srcs)
    try:
        I, nev, _ = M.load_flat(src, M.IT_TABLE,
                                ["layer", "globalR", "globalZ", "globalPhi",
                                 "tpIdx", "tpPt"], None, tp=("tp_z0", "tp_tanL"))
        O, onev, _ = M.load_ot(src, None)
    except Exception as exc:
        for a in ax_row:
            a.axis("off")
        out["combined_it_ot"] = {"skipped": f"{type(exc).__name__}: {exc}"}
        return
    bar = (O["isBarrel"] > 0) & (O["eta"] <= M.ETA_MATCHED)
    U = {"layer": np.r_[I["layer"], O["layer"][bar] + 10],
         "r": np.r_[I["globalR"], O["r"][bar]], "z": np.r_[I["globalZ"], O["z"][bar]],
         "phi": np.r_[I["globalPhi"], O["phi"][bar]],
         "tpIdx": np.r_[I["tpIdx"], O["tpIdx"][bar]],
         "tpPt": np.r_[I["tpPt"], O["tpPt"][bar]],
         "event": np.r_[I["event"], O["event"][bar]]}
    mi = (I["tpIdx"] >= 0) & np.isfinite(I["tp_z0"])
    kv = M.tp_key(I["event"][mi], I["tpIdx"][mi])
    o = np.argsort(kv, kind="stable")
    kv, z0t, tlt = kv[o], I["tp_z0"][mi][o], I["tp_tanL"][mi][o]
    f = np.r_[True, kv[1:] != kv[:-1]]
    KV, Z0T, TLT = kv[f], z0t[f], tlt[f]

    def firstpertp(L):
        m = (U["layer"] == L) & (U["tpIdx"] >= 0) & (U["tpPt"] >= PT)
        idx = np.flatnonzero(m)
        k = M.tp_key(U["event"][idx], U["tpIdx"][idx])
        oo = np.argsort(k, kind="stable")
        k, idx = k[oo], idx[oo]
        ff = np.r_[True, k[1:] != k[:-1]]
        return k[ff], idx[ff]

    def rs(x):
        qq = np.percentile(x, [15.865, 84.135])
        return float(0.5 * (qq[1] - qq[0]))

    NAME = {1: "IT L1", 2: "IT L2", 3: "IT L3", 4: "IT L4",
            11: "OT L1", 12: "OT L2", 13: "OT L3", 14: "OT L4",
            15: "OT L5", 16: "OT L6"}
    SEEDS = [("IT L1+L2", 1, 2), ("IT L2+L3", 2, 3), ("IT L3+OT L1", 3, 11),
             ("IT L4+OT L2", 4, 12), ("OT L1+L2", 11, 12)]
    FP = {}
    qual = {}
    for nm, la, lb in SEEDS:
        for L in (la, lb):
            if L not in FP:
                FP[L] = firstpertp(L)
        ka, ia = FP[la]; kb, ib = FP[lb]
        com = np.intersect1d(np.intersect1d(ka, kb), KV)
        if len(com) < 500:
            continue
        ga = ia[np.searchsorted(ka, com)]; gb = ib[np.searchsorted(kb, com)]
        p = np.searchsorted(KV, com)
        dr = U["r"][gb] - U["r"][ga]
        ok = np.abs(dr) > 0.5
        kap = M.wrap(U["phi"][ga] - U["phi"][gb]) / (M.C_BEND * np.where(ok, dr, 1e9))
        cot = (U["z"][gb] - U["z"][ga]) / np.where(ok, dr, 1e9)
        z0 = U["z"][ga] - U["r"][ga] * cot
        g = ok & (np.abs(kap) < 2) & (np.abs(z0 - Z0T[p]) < 30)
        if g.sum() < 500:
            continue
        qual[nm] = {"dr_cm": float(np.median(dr)),
                    "sig_kappa": rs(np.abs(kap[g]) - 1.0 / U["tpPt"][ga][g]),
                    "sig_z0_cm": rs(z0[g] - Z0T[p][g]),
                    "sig_cot": rs(cot[g] - TLT[p][g]),
                    "n": int(g.sum())}
    # TARGET-LAYER z RESOLUTION, measured against each TP's own helix
    # z = z0 + r*tanL (L1TTP, at the POCA). This MUST enter the z road: an earlier revision used
    # only the seed's z uncertainty and understated IT-seed projection work by
    # 2.9x, because OT 2S modules resolve z to ~1.8 cm and completely dominate
    # the window there.
    def zres(rr, zz, ev, tpi):
        k = M.tp_key(ev, tpi)
        p = np.clip(np.searchsorted(KV, k), 0, len(KV) - 1)
        g = KV[p] == k
        if g.sum() < 200:
            return 0.0
        return rs(zz[g] - (Z0T[p[g]] + rr[g] * TLT[p[g]]))
    layers = {}
    for L in (1, 2, 3, 4):
        m = (I["layer"] == L)
        mt = m & (I["tpIdx"] >= 0) & (I["tpPt"] >= PT)
        layers[NAME[L]] = (float(np.median(I["globalR"][m])), m.sum() / nev,
                           float(np.ptp(I["globalZ"][m])),
                           zres(I["globalR"][mt], I["globalZ"][mt],
                                I["event"][mt], I["tpIdx"][mt]))
    for L in range(1, 7):
        m = bar & (O["layer"] == L)
        if m.sum() < 100:
            continue
        mt = m & (O["tpIdx"] >= 0) & (O["tpPt"] >= PT)
        layers[NAME[L + 10]] = (float(np.median(O["r"][m])), m.sum() / onev,
                                float(np.ptp(O["z"][m])),
                                zres(O["r"][mt], O["z"][mt], O["event"][mt],
                                     O["tpIdx"][mt]))
    ms = M.THETA_MS_MRAD * 1e-3 / PT
    # WHICH COORDINATE TO SEARCH ON, per target layer. The IT-only study found
    # z-first beat phi-first by 28x -- but that phi window carried the d0
    # allowance (17.5 mrad) that beamline-constrained pairs no longer need, and
    # it projected only into finely-z-segmented IT layers. With the d0 term gone
    # and the target's own z resolution in, the answer REVERSES on nearly every
    # layer, and the seed ranking reverses with it: phi cost scales with
    # sigma(kappa), which is exactly where the long-lever-arm mixed seeds win.
    cost = {}
    for nm, qq in qual.items():
        per = {}
        for tn, (r, occ, zs, stz) in layers.items():
            wz = 2 * NSIG * np.sqrt(qq["sig_z0_cm"] ** 2 + (r * qq["sig_cot"]) ** 2
                                    + stz ** 2)
            wphi = 2 * NSIG * (M.C_BEND * r * qq["sig_kappa"] + ms)
            zc = occ * min(wz / zs, 1.0)
            pc = occ * min(wphi / (2 * np.pi), 1.0)
            per[tn] = {"z_search": float(zc), "phi_search": float(pc),
                       "best": float(min(zc, pc)),
                       "use": ("z" if zc <= pc else "phi"),
                       "background": float(zc * min(wphi / (2 * np.pi), 1.0))}
        sub = [v for k, v in per.items() if k != "ALL"]
        per["ALL"] = {"z_search": float(sum(v["z_search"] for v in sub)),
                      "phi_search": float(sum(v["phi_search"] for v in sub)),
                      "best": float(sum(v["best"] for v in sub)),
                      "background": float(sum(v["background"] for v in sub))}
        cost[nm] = per
    out["combined_it_ot"] = {"pt_min": PT, "n_events": nev, "seed_quality": qual,
                             "layers": {k: {"r": v[0], "obj_per_event": v[1],
                                            "z_span": v[2], "sigma_z_cm": v[3]}
                                        for k, v in layers.items()},
                             "projection_cost": cost,
                             "it_designs": IT_DESIGNS}
    _draw_combined(ax_row, qual, layers, cost)


def _draw_combined(ax_row, qual, layers, cost):
    C = ["#2a78d6", "#e8833a", "#4b9f6e", "#b1524f", "#7a6ff0"]
    INK2 = "#52514e"
    names = list(qual)
    ax = ax_row[0]
    for i, nm in enumerate(names):
        ax.scatter(qual[nm]["sig_kappa"], qual[nm]["sig_z0_cm"] * 1e4, s=90,
                   color=C[i % len(C)], edgecolor="#fcfcfb", lw=1.2, zorder=3)
        ax.annotate(nm, (qual[nm]["sig_kappa"], qual[nm]["sig_z0_cm"] * 1e4),
                    fontsize=6.5, xytext=(5, 4), textcoords="offset points", color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("sigma(kappa)  [GeV^-1]", fontsize=8)
    ax.set_ylabel("sigma(z0)  [um]", fontsize=8)
    ax.set_title("the IT/OT trade: longitudinal vs curvature\n(lower-left is better, "
                 "nothing is there)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.tick_params(labelsize=7, colors=INK2)

    ax = ax_row[1]
    tn = [t for t in layers]
    for i, nm in enumerate(names):
        ax.plot(range(len(tn)), [cost[nm][t]["z_search"] for t in tn], "-o",
                color=C[i % len(C)], ms=4, lw=1.8, label=nm)
    ax.set_yscale("log")
    ax.set_xticks(range(len(tn))); ax.set_xticklabels(tn, rotation=90, fontsize=6.5)
    ax.set_ylabel("objects scanned in the z road per projection [count]", fontsize=8)
    ax.set_title("projection WORK, per seed per target layer", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=6); ax.tick_params(labelsize=7, colors=INK2)

    ax = ax_row[2]
    xs = np.arange(len(names))
    ax.bar(xs - 0.25, [cost[n]["ALL"]["z_search"] for n in names], 0.25,
           color="#9ec5f4", label="always z-first")
    ax.bar(xs, [cost[n]["ALL"]["phi_search"] for n in names], 0.25,
           color="#2a78d6", label="always phi-first")
    ax.bar(xs + 0.25, [cost[n]["ALL"]["best"] for n in names], 0.25,
           color="#e8833a", label="best per layer")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylabel("objects scanned, summed over all ten IT+OT layers [count]", fontsize=8)
    ax.set_title("total projection cost of one seed\n(real hit excluded: it is there "
                 "by construction)", fontsize=9)
    ax.grid(alpha=0.25, axis="y", lw=0.6); ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=7, colors=INK2)



def study_cluster_angles_by_layer(X, K, P, ax_row, out):
    """(17) per-layer cluster alpha/beta, and the residual against the track.

    WHY PER LAYER. A cluster's incidence angle is set by where it sits: at IL1's
    radius a track crosses at a much steeper angle than at IL4, so a pooled
    alpha distribution is a mixture of four different things and its width means
    nothing. MEASURED, 1 to 99 percentile of the cluster's own alpha term:
    IL1 +-0.026, IL2 +-0.014, IL3 +-0.010, IL4 +-0.008.

    WHY TWO BREAKDOWNS ON THE RESIDUAL. Whether the cluster belongs to the track
    it was attached to is the cluster-level question, and it is not the same as
    whether the TRACK is clean: a correctly assigned cluster on a track carrying
    three wrong hits still has a good angle residual, and its own residual is
    what a per-cluster gate would cut on. Both splits are applied at once --
    assignment as line style, the track's wrong-hit count as colour -- so
    neither hides the other.

    Reads the angle rows from the menu census rather than re-fitting: they are
    produced there by KF.angle_rows for one emitted track in ANGLE_SAMPLE, and
    re-deriving them here would be a second implementation of the same thing.
    """
    import importlib.util as _ilu
    _here = os.path.dirname(os.path.abspath(__file__))

    def _mod(name):
        sp = _ilu.spec_from_file_location("_" + name,
                                          os.path.join(_here, name + ".py"))
        m = _ilu.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m

    def _skip(msg):
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.02, 0.5, msg, fontsize=7, wrap=True, va="center")
        out["cluster_angles_by_layer"] = {"skipped": msg}

    TF = _mod("tp_findability")
    KF = TF.KF
    cdir = out.get("_menu_cache") or os.path.join(_here, "cache")
    ds = sorted(_glob.glob(os.path.join(cdir, "tpcensus_*")))
    if not ds:
        return _skip("no census under %s; run the menu study first, or point "
                     "--menu-cache at one" % cdir)
    C = TF.load(ds[-1], with_tracks=True)
    per_seed = [v for v in C.get("angles", {}).values() if len(v)]
    if not per_seed:
        return _skip("that census carries no angle rows; re-run it with "
                     "--menu-export-tracks so KF.angle_rows is stored")
    A = np.concatenate(per_seed, axis=0)
    ci = {n: i for i, n in enumerate(KF.ANGLE_COLS)}
    lay = A[:, ci["layer"]].astype(int)
    res = {"a": A[:, ci["res_a"]].astype(float),
           "b": A[:, ci["res_b"]].astype(float)}
    meas = {"a": res["a"] + A[:, ci["pred_a"]].astype(float),
            "b": res["b"] + A[:, ci["pred_b"]].astype(float)}
    wrong_cluster = A[:, ci["hit_wrong"]].astype(float) > 0
    nwc = np.clip(A[:, ci["n_wrong"]].astype(float), 0, 3).astype(int)
    LAYERS = [1, 2, 3, 4]
    NWLAB = ["0 wrong", "1 wrong", "2 wrong", r"$\geq$3 wrong"]
    COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    summary = {"n_clusters": int(len(A)), "per_layer": {}}
    for ax, side, nm in ((ax_row[0], "a",
                          r"cluster $d\phi/dr$ [rad/cm]"),
                         (ax_row[1], "b",
                          r"cluster $\cot\theta$ [unitless]")):
        v = meas[side]
        rng = float(np.nanpercentile(np.abs(v), 99)) if len(v) else 1.0
        bins = np.linspace(-rng, rng, 90)
        for L in LAYERS:
            s = lay == L
            if s.sum() < 20:
                continue
            ax.hist(v[s], bins=bins, histtype="step", density=True,
                    color=COLS[L - 1], label="IL%d (n=%s)" % (L, f"{int(s.sum()):,}"))
            summary["per_layer"].setdefault("IL%d" % L, {})["meas_%s_p1_p99" % side] = \
                [float(np.percentile(v[s], 1)), float(np.percentile(v[s], 99))]
        ax.set_yscale("log")
        ax.set_xlabel(nm)
        ax.set_ylabel("density [1/bin]")
        ax.legend(fontsize=6)
        ax.grid(alpha=.3)
        ax.set_title("(17) %s per layer" % nm)

    f2, axs = plt.subplots(2, len(LAYERS), figsize=(4.3 * len(LAYERS), 7.4),
                           squeeze=False)
    for r, side in enumerate(("a", "b")):
        v = res[side]
        rng = float(np.nanpercentile(np.abs(v), 99)) if len(v) else 1.0
        bins = np.linspace(-rng, rng, 70)
        for c, L in enumerate(LAYERS):
            ax = axs[r][c]
            base = lay == L
            for k in range(4):
                for wrongc, style in ((False, "-"), (True, "--")):
                    s = base & (nwc == k) & (wrong_cluster == wrongc)
                    if s.sum() < 30:
                        continue
                    ax.hist(v[s], bins=bins, histtype="step", density=True,
                            color=COLS[k], linestyle=style, lw=1.1,
                            label="%s, %s (n=%s)" % (
                                NWLAB[k], "other TP" if wrongc else "same TP",
                                f"{int(s.sum()):,}"))
                    # robust_sigma returns (mad, half 16-84) and the two
                    # disagree exactly where it matters here: a wrongly
                    # assigned cluster's residual is tail-heavy, so keep both
                    mad, q68 = robust_sigma(v[s])
                    summary.setdefault("residual_sigma", {})[
                        "%s:IL%d/%sTP/%dwrong" % (
                            side, L, "other" if wrongc else "same", k)] = \
                        {"mad": mad, "q68": q68, "n": int(s.sum())}
            ax.set_yscale("log")
            ax.set_xlabel(r"$\Delta(d\phi/dr)$ [rad/cm]" if side == "a"
                          else r"$\Delta\cot\theta$ [unitless]")
            if c == 0:
                ax.set_ylabel("density [1/bin]")
            ax.grid(alpha=.3)
            ax.legend(fontsize=5)
            ax.set_title("IL%d: %s residual" % (L, "alpha" if side == "a"
                                                else "beta"), fontsize=9)
    f2.suptitle("(17) cluster angle residual per layer -- colour is the TRACK's "
                "wrong-hit count, dashed is a cluster from a different "
                "TrackingParticle", fontsize=10)
    f2.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = os.path.join(out["_outdir"], "spix_cluster_angles_by_layer.png")
    f2.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(f2)
    summary["figure"] = p2
    out["cluster_angles_by_layer"] = summary


def study_mva_score_correlation(X, K, P, ax_row, out):
    """(18) rank correlation between the shipped discriminants, per activeSP build.

    WHY PER BUILD. The scores are trained ONCE on the whole census, so their
    correlation structure is not guaranteed to be the same in every
    configuration: a build with fewer instrumented IT layers forms different
    seeds, and its tracks are a different population. If two scores are
    redundant in AAAA but separate in IAAA, keeping both is justified by the
    build that needs them, and a single pooled matrix would hide that.

    SPEARMAN, NOT PEARSON. These are selection scores used by thresholding, so
    what matters is whether they ORDER tracks the same way; a monotone
    re-scaling of one score would move Pearson and not change any cut.

    Reads the exported site rather than retraining: the scores are produced by
    export_site (one model per discriminant, trained on event-disjoint folds),
    and retraining here would be a second implementation whose numbers could
    drift from the ones the page actually shows.
    """
    od = out.get("_outdir", ".")
    site = out.get("_site") or os.path.join(od, "site")
    mpath = os.path.join(site, "manifest.json")

    def _skip(msg):
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.02, 0.5, msg, fontsize=7, wrap=True, va="center")
        out["mva_score_correlation"] = {"skipped": msg}

    if not os.path.exists(mpath):
        return _skip("no exported site at %s; run export_site first or pass "
                     "--site" % site)
    m = json.load(open(mpath))
    if "score" not in m or "track" not in m:
        return _skip("that site carries no score table")
    SC, TR = m["score"], m["track"]
    names = SC["cols"]
    nsc, ntc = len(names), len(TR["cols"])
    N = TR["rows"]
    S = np.memmap(os.path.join(site, SC["file"]), dtype=np.uint8,
                  mode="r").reshape(N, nsc)
    T = np.memmap(os.path.join(site, TR["file"]), dtype=np.float32,
                  mode="r").reshape(N, ntc)
    j_seed = TR["cols"].index("seed_idx")
    tag_of = {s["idx"]: s["tag"] for s in m["seeds"]}

    rng = np.random.default_rng(0)
    take = min(N, 1_500_000)
    sub = np.sort(rng.choice(N, take, replace=False))
    seed_idx = np.asarray(T[sub, j_seed], np.int64)
    SS = np.asarray(S[sub, :], np.float64)

    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        return r

    builds = list(m["builds"])
    res, mats = {}, {}
    for b in builds:
        allowed = set(m["builds"][b])
        keep = np.fromiter((tag_of.get(i) in allowed for i in seed_idx),
                           bool, len(seed_idx))
        n = int(keep.sum())
        if n < 5000:
            res[b] = {"n": n, "skipped": "too few tracks"}
            continue
        R = np.column_stack([rank(SS[keep, k]) for k in range(nsc)])
        Cm = np.corrcoef(R, rowvar=False)
        mats[b] = Cm
        iu = np.triu_indices(nsc, 1)
        off = np.abs(Cm[iu])
        res[b] = {"n": n, "max_abs_offdiag": float(off.max()),
                  "mean_abs_offdiag": float(off.mean()),
                  "most_correlated": [names[iu[0][off.argmax()]],
                                      names[iu[1][off.argmax()]]],
                  "matrix": {names[i]: {names[j]: float(Cm[i, j])
                                        for j in range(nsc)}
                             for i in range(nsc)}}

    # main row: how redundant the set is, build by build
    ax = ax_row[0]
    ok = [b for b in builds if b in mats]
    ax.plot(range(len(ok)), [res[b]["max_abs_offdiag"] for b in ok],
            marker="o", ms=4, label="most correlated pair")
    ax.plot(range(len(ok)), [res[b]["mean_abs_offdiag"] for b in ok],
            marker="s", ms=4, label="mean over pairs")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(ok, rotation=90, fontsize=6)
    ax.set_ylabel(r"$|\rho_{\rm Spearman}|$ [unitless]")
    ax.set_ylim(0, 1)
    ax.axhline(0.95, color="crimson", ls=":", lw=1)
    ax.legend(fontsize=6)
    ax.grid(alpha=.3)
    ax.set_title("(18) score redundancy per activeSP build", fontsize=9)

    # and the spread of each pair across builds: a pair that is redundant
    # everywhere is a drop candidate, one that varies is not
    ax = ax_row[1]
    iu = np.triu_indices(nsc, 1)
    pair_lab, pair_lo, pair_hi = [], [], []
    for a_, b_ in zip(*iu):
        v = [abs(mats[b][a_, b_]) for b in ok]
        pair_lab.append(f"{names[a_].replace('mva_', '')}/"
                        f"{names[b_].replace('mva_', '')}")
        pair_lo.append(min(v))
        pair_hi.append(max(v))
    order = np.argsort(pair_hi)[::-1][:14]
    y = np.arange(len(order))
    ax.hlines(y, [pair_lo[i] for i in order], [pair_hi[i] for i in order],
              lw=3, alpha=.6)
    ax.plot([pair_hi[i] for i in order], y, "o", ms=3)
    ax.set_yticks(y)
    ax.set_yticklabels([pair_lab[i] for i in order], fontsize=5)
    ax.set_xlabel(r"$|\rho_{\rm Spearman}|$ range over builds [unitless]")
    ax.set_xlim(0, 1)
    ax.axvline(0.95, color="crimson", ls=":", lw=1)
    ax.grid(alpha=.3)
    ax.set_title("(18) most correlated pairs, min-max over builds", fontsize=9)

    # extra figure: the full matrix for every build
    nb = len(ok)
    nc = 4
    nr = int(np.ceil(nb / nc))
    f2, axs = plt.subplots(nr, nc, figsize=(3.5 * nc, 3.3 * nr), squeeze=False)
    short = [n.replace("mva_", "") for n in names]
    for i, b in enumerate(ok):
        a = axs[i // nc][i % nc]
        im = a.imshow(mats[b], vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(nsc)); a.set_yticks(range(nsc))
        a.set_xticklabels(short, rotation=90, fontsize=5)
        a.set_yticklabels(short, fontsize=5)
        a.set_title(f"{b}  (n={res[b]['n']:,})", fontsize=8)
        for p in range(nsc):
            for q in range(nsc):
                a.text(q, p, f"{mats[b][p, q]:.2f}", ha="center", va="center",
                       fontsize=3.6,
                       color="white" if abs(mats[b][p, q]) > 0.6 else "black")
    for i in range(nb, nr * nc):
        axs[i // nc][i % nc].axis("off")
    f2.colorbar(im, ax=axs, shrink=0.5,
                label=r"$\rho_{\rm Spearman}$ [unitless]")
    f2.suptitle("(18) full rank-correlation matrix of the shipped "
                "discriminants, per activeSP build", fontsize=10)
    p2 = os.path.join(od, "spix_mva_score_correlation.png")
    f2.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(f2)
    out["mva_score_correlation"] = {"scores": names, "n_sampled": int(take),
                                    "figure": p2, "per_build": res}

# ---- (19) OT-only projection: targets vs background, per track class --------
PROJ_TERMS = ["OT-only projection", "layer projection", "4-sigma ellipse",
              "majority-owner TP", "foreign stub", "fake stub",
              "track classes  (perfect / foreign-1 / foreign-2+ / fake-1 / fake-2+ / foreign-fake)",
              "excluded tracks  (tie / no-owner / unjoined / covariance absent)",
              "A  (target)  /  A-in  /  A-out",
              "B1 / B2 / B3  (background inside the ellipse)",
              "residual  (du, dv, dcotAlpha, dcotBeta)"]
PROJ_CAT_STYLE = {0: dict(color="#111111", ls="-", lw=1.6),     # A-in
                  1: dict(color="#111111", ls="--", lw=1.1),    # A-out
                  2: dict(color="#d62728", ls="-", lw=1.2),     # B1
                  3: dict(color="#1f77b4", ls="-", lw=1.2),     # B2
                  4: dict(color="#8c8c8c", ls="-", lw=1.2)}     # B3
PROJ_QTY = [("du", "du  [um]", 1e4), ("dv", "dv  [um]", 1e4),
            ("d", "Mahalanobis d  [sigma of the OT fit]", 1.0),
            ("dcotA", "dcotAlpha  [unitless]", 1.0), ("dcotB", "dcotBeta  [unitless]", 1.0)]
# The cross-check against the producer's own seed projection must hold at nano
# precision, or every number below is a statement about a bug here. Measured on
# PU200: |du|,|dv| p99 0.01 um; |dcotAlpha| p99 <= 3e-5, |dcotBeta| <= 4.4e-4;
# cone width ratio median 1.000 (central vs one-sided differences).
PROJ_VALIDATION_LIMITS = {"du_p99_cm": 1e-4, "dv_p99_cm": 1e-4, "dcotA_p99": 2e-3,
                          "dcotB_p99": 5e-3, "sig_ratio_median_tol": 0.02}


def _proj_mad_rms(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {"n": int(len(x)), "median": float("nan"), "mad": float("nan"), "rms": float("nan")}
    med = float(np.median(x))
    return {"n": int(len(x)), "median": med, "mad": float(1.4826 * np.median(np.abs(x - med))),
            "rms": float(np.std(x))}


# |other angle residual| kept on the cross-cut page, unitless: comfortably wider
# than the target's own core (MAD ~0.02-0.03) and far inside B2's (0.1-2.1)
PROJ_CROSS_CUT = 1.0


def _draw_proj_crosscut(R, Ls, nsig, n_events, n_perfect):
    """Perfect tracks: dcotAlpha with |dcotBeta| <= cut, and the converse, per layer.

    Same common normalisation as the per-layer pages: every category in a panel
    is divided by the total AFTER the cut, so relative proportions survive.
    Clusters without an angle estimate (all of B3) cannot enter either panel.
    """
    import ot_projection as OPJ
    cut = PROJ_CROSS_CUT
    fig = plt.figure(figsize=(14, 3.0 * len(Ls) + 1.6))
    gs = fig.add_gridspec(len(Ls), 3, width_ratios=[1, 1, 0.8])
    table = {}
    for li, L in enumerate(Ls):
        base = (R["layer"] == L) & (R["cls"] == 0)
        both = base & np.isfinite(R["dcotA"]) & np.isfinite(R["dcotB"])
        table[f"IL{L}"] = {}
        for pi, (q, other, qlab, olab) in enumerate(
                (("dcotA", "dcotB", "dcotAlpha", "dcotBeta"),
                 ("dcotB", "dcotA", "dcotBeta", "dcotAlpha"))):
            ax = fig.add_subplot(gs[li, pi])
            keep = both & (np.abs(R[other]) <= cut)
            n_all = int(keep.sum())
            v = R[q][keep & np.isin(R["cat"], (0, 2, 3, 4))].astype(np.float64)
            hi = np.nanpercentile(np.abs(v), 99.5) if len(v) else 1.0
            edges = np.linspace(-hi, hi, 81)
            rows = {}
            for c in range(5):
                m_all = both & (R["cat"] == c)
                m = keep & (R["cat"] == c)
                x = R[q][m].astype(np.float64)
                rows[OPJ.CATEGORIES[c]] = {
                    "n_with_both_angles": int(m_all.sum()), "n_after_cut": int(m.sum()),
                    "kept_fraction": float(m.sum() / m_all.sum()) if m_all.any() else float("nan"),
                    **{k: v_ for k, v_ in _proj_mad_rms(x).items() if k != "n"}}
                if not len(x):
                    continue
                h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), edges)
                ax.stairs(h / max(n_all, 1), edges, **PROJ_CAT_STYLE[c],
                          label=f"{OPJ.CATEGORIES[c]} n={len(x):,} "
                                f"(kept {100 * rows[OPJ.CATEGORIES[c]]['kept_fraction']:.1f}%)")
            table[f"IL{L}"][f"{qlab} | |{olab}|<={cut:g}"] = rows
            ax.set_yscale("log")
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_xlabel(f"{qlab}  [unitless]   (|{olab}| <= {cut:g}; edge bins hold overflow)",
                          fontsize=7)
            if pi == 0:
                ax.set_ylabel(f"IL{L}\nfraction of all clusters\nin panel / bin", fontsize=8)
            if li == 0:
                ax.set_title(f"{qlab}, with |{olab}| <= {cut:g}", fontsize=9)
    tax = fig.add_subplot(gs[:, -1])
    tax.axis("off")
    tax.text(0.0, 1.0,
             f"PERFECT tracks only ({n_perfect:,} tracks, {n_events:,} events).\n"
             f"OT-only projection, no KF update between layers,\n"
             f"{nsig:g}-sigma ellipse as on the per-layer pages.\n\n"
             f"Each panel: one angle residual, the OTHER\n"
             f"restricted to |d| <= {cut:g} (unitless) -- a 1D view of\n"
             f"the 2D (dcotAlpha, dcotBeta) distribution.\n"
             f"'kept' = fraction of that category's clusters\n"
             f"(with both angles) surviving the other-angle cut.\n\n"
             f"Normalised to ALL clusters in the panel after\n"
             f"the cut, so category proportions are preserved.\n\n"
             "A-in / A-out = majority-owner TP's cluster inside /\n"
             "  outside the ellipse; B2 = other TP in ellipse.\n"
             "B1 is empty by definition (a perfect track has no\n"
             "  foreign stub); B3 (no TP) carries no angle\n"
             "  estimate, so it cannot appear here.",
             fontsize=7, va="top", ha="left", family="monospace", transform=tax.transAxes)
    fig.suptitle(f"Perfect tracks: each angle residual with the other angle restricted to "
                 f"|d| <= {cut:g}, per layer", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return {"fig": fig, "table": table}


def study_ot_projection_targets(X, K, P, ax_row, out):
    """(19) OT-only projection: what the refit's TARGET clusters and the
    BACKGROUND clusters inside a 4-sigma ellipse look like, by track class.

    WHY THIS IS SEPARATE FROM THE REST OF THE OMNIBUS. Every other study here
    reads the refit's per-crossing tables, which keep ONE module per layer and
    record the seed projection only where the seed and the KF-updated state share
    a module. This one needs neither restriction, so it projects the OT track
    itself (ot_projection.py) onto every module its ellipse overlaps, and uses
    the refit tables only as a cross-check of its helix, frame and Jacobian. It
    ignores X/K/P and streams its own inputs (--proj-inputs, default -i).

    WHY PER LAYER. There is no Kalman update between layers here, so IL1 and IL4
    are two independent extrapolations of the same OT fit with different lever
    arms; pooling them would mix four different cone sizes. The KF-updated
    version is a later study.

    WHY BY TRACK CLASS. The target is the majority-owner TP's cluster. On a
    contaminated OT track the projection itself is pulled, and the cluster of a
    TP that owns a foreign stub (B1) is the trap it sets for itself; separating
    classes says whether that trap is visible in position or angle.
    """
    import ot_projection as OPJ

    def _skip(msg):
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.02, 0.5, "OT-only projection study SKIPPED:\n" + msg, fontsize=7,
                       wrap=True, va="center")
        out["ot_projection_targets"] = {"skipped": msg}
        print(f"    SKIPPED: {msg}")

    files = out.get("_proj_inputs") or out["_inputs"]
    geom = out.get("_geometry")
    if not geom or not os.path.exists(geom):
        return _skip("no module geometry JSON (--geometry); produce it with "
                     "L1Trigger/Phase3SmartPixels/test/dumpModuleGeometry_cfg.py")
    try:
        OPJ.check_inputs(files[0], with_refit=True)
    except SystemExit as e:
        return _skip(str(e))
    nsig = 4.0
    step = [0]

    def _log(msg):
        step[0] += 1
        if step[0] % 10 == 0:
            print(msg)
    r = OPJ.run(files, geom, nsig=nsig, chunk=out.get("_proj_chunk", 10),
                nev=out.get("_proj_nev"), validate=True, log=_log)
    pulls = OPJ.param_pulls(files, nev=out.get("_proj_nev"))
    R, PR, V = r["rows"], r["proj"], r["validation"]
    classes = OPJ.TRACK_CLASSES
    tbc = r["tracks_by_class"]

    # ---- the cross-check gate -------------------------------------------
    lim = PROJ_VALIDATION_LIMITS
    val = {"n_refit_seed_crossings": V["n_refit"], "n_compared": V["n_compared"],
           "du_p99_cm": float(np.percentile(np.abs(V["du"]), 99)),
           "dv_p99_cm": float(np.percentile(np.abs(V["dv"]), 99)),
           "dcotA_p99": float(np.percentile(np.abs(V["dcotA"]), 99)),
           "dcotB_p99": float(np.percentile(np.abs(V["dcotB"]), 99)),
           "sigU_ratio_median": float(np.median(V["rsu"])),
           "sigV_ratio_median": float(np.median(V["rsv"])),
           "sig_ratio_p1_p99": [float(np.percentile(np.r_[V["rsu"], V["rsv"]], q)) for q in (1, 99)]}
    bad = [k for k in ("du_p99_cm", "dv_p99_cm", "dcotA_p99", "dcotB_p99") if val[k] > lim[k]]
    bad += [k for k in ("sigU_ratio_median", "sigV_ratio_median")
            if abs(val[k] - 1.0) > lim["sig_ratio_median_tol"]]
    if V["n_compared"] < 0.99 * V["n_refit"]:
        bad.append("n_compared")
    print(f"    cross-check vs refit projSeed*: {V['n_compared']}/{V['n_refit']} crossings, "
          f"|du| p99 {val['du_p99_cm'] * 1e4:.3f} um, |dcotB| p99 {val['dcotB_p99']:.2e}, "
          f"width ratio {val['sigU_ratio_median']:.4f}/{val['sigV_ratio_median']:.4f}")
    if bad:
        raise SystemExit(f"OT-only projection DISAGREES with the producer's seed projection on "
                         f"{bad}: {val}. Do not trust any result of this study.")
    if r["bound_ratio_max"] > 0.9:
        print(f"    WARNING: an overlapping module sat at {r['bound_ratio_max']:.2f} of the "
              f"candidate bound; widen BOUND_* in ot_projection.py")

    # ---- per class x layer tables ---------------------------------------
    table = {}
    for ci, cname in enumerate(classes):
        table[cname] = {}
        for L in OPJ.IT_LAYERS:
            m = (PR["cls"] == ci) & (PR["layer"] == L)
            cv = m & PR["covered"]
            ae = m & PR["a_exists"]
            e = {"n_layer_projections": int(m.sum()), "n_covered": int(cv.sum()),
                 "n_lands": int((m & PR["lands"]).sum()), "n_target_exists": int(ae.sum()),
                 "n_target_inside": int((ae & PR["a_in"]).sum()),
                 "n_target_unprojectable": int((ae & PR["a_unprojectable"]).sum()),
                 "containment": float((ae & PR["a_in"]).sum() / max(ae.sum(), 1))
                 if ae.any() else float("nan")}
            for b in ("nB1", "nB2", "nB3"):
                e[f"mean_{b}_per_covered"] = float(PR[b][cv].mean()) if cv.any() else float("nan")
            ld = m & PR["lands"]
            for s_ in ("su", "sv", "sCotA", "sCotB"):
                e[f"median_{s_}"] = float(np.median(PR[s_][ld])) if ld.any() else float("nan")
            rm = (R["cls"] == ci) & (R["layer"] == L)
            e["residuals"] = {OPJ.CATEGORIES[c]: {q: _proj_mad_rms(R[q][rm & (R["cat"] == c)].astype(np.float64))
                                                  for q, _, _ in PROJ_QTY} for c in range(5)}
            table[cname][f"IL{L}"] = e

    excl = {k: tbc[k] for k in ("tie", "no-owner", "unjoined", "covariance-absent")}
    pull_txt = "  ".join(f"{p} {pulls[p]['mad']:.2f}" for p in OPJ.HPAR)
    excl_txt = ", ".join(f"{k} {v:,}" for k, v in excl.items())
    cls_txt = "  ".join(f"{c} {tbc[c]:,}" for c in classes)

    # ---- summary row on the omnibus grid ---------------------------------
    ax = ax_row[0]
    Ls = list(OPJ.IT_LAYERS)
    cols_ = plt.cm.tab10(np.arange(len(classes)))
    for ci, cname in enumerate(classes):
        y = [table[cname][f"IL{L}"]["containment"] for L in Ls]
        ax.plot(Ls, y, "o-", color=cols_[ci], label=f"{cname} ({tbc[cname]:,} trk)")
    ax.set_xticks(Ls, [f"IL{L}" for L in Ls])
    ax.set_ylabel("target containment  [fraction of layer projections]")
    ax.set_title("A-in / (A exists): majority-owner cluster\ninside the 4-sigma OT-only ellipse",
                 fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=6, loc="lower left")
    ax = ax_row[1]
    for ci, cname in enumerate(classes):
        tot = [sum(table[cname][f"IL{L}"][f"mean_nB{b}_per_covered"] for b in (1, 2, 3)) for L in Ls]
        b1 = [table[cname][f"IL{L}"]["mean_nB1_per_covered"] for L in Ls]
        ax.plot(Ls, tot, "o-", color=cols_[ci], label=cname)
        if cname.startswith("foreign"):
            ax.plot(Ls, b1, "s:", color=cols_[ci], ms=4)
    ax.set_xticks(Ls, [f"IL{L}" for L in Ls])
    ax.set_yscale("log")
    ax.set_ylabel("background clusters per covered layer projection  [count]")
    ax.set_title("B1+B2+B3 inside the ellipse (solid)\nB1 alone, foreign classes (dotted)", fontsize=9)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=6)
    ax = ax_row[2]
    ax.axis("off")
    legend_txt = (
        "DEFINITIONS (full text: spix_glossary.txt)\n"
        "Track classes, by the OT track's stubs vs its majority-owner TP:\n"
        + "\n".join(f"  {c}: {OPJ.TRACK_CLASS_DEF[c]}" for c in classes)
        + "\n  foreign stub = genuine stub of ANOTHER TP;  fake stub = not genuine\n"
          "  (stub-combinatoric or unknown).  CMSSW L1TTrack_genuine can carry\n"
          "  ONE fake stub: it is NOT 'perfect'.\n"
        f"EXCLUDED (no class): {excl_txt}\n"
        "A = majority-owner TP's cluster on the layer (A-in / A-out of ellipse)\n"
        "B1 = in ellipse, TP owns a foreign stub of this track\n"
        "B2 = in ellipse, other TP;  B3 = in ellipse, no TP\n"
        "Ellipse: d < 4 of J C J^T, OT fit covariance, no MS, no KF update;\n"
        "  every module it overlaps, residuals in the cluster's module frame.\n"
        f"OT fit pull widths (MAD; 1 = honest): {pull_txt}\n"
        f"Cross-check vs refit projSeed*: {V['n_compared']:,} crossings, |du| p99\n"
        f"  {val['du_p99_cm'] * 1e4:.3f} um, width ratio {val['sigU_ratio_median']:.3f}\n"
        f"Per-layer residual figures: spix_ot_projection_IL1..4.png / .pdf")
    ax.text(0.0, 1.0, legend_txt, fontsize=5.6, va="top", ha="left", family="monospace",
            transform=ax.transAxes)

    # ---- per-layer residual figures --------------------------------------
    from matplotlib.backends.backend_pdf import PdfPages
    od = out["_outdir"]
    pdf_path = os.path.join(od, "spix_ot_projection.pdf")
    defs = _glossary_text(width=58, terms=PROJ_TERMS)
    pngs = []
    with PdfPages(pdf_path) as pdf:
        for L in Ls:
            fig = plt.figure(figsize=(26, 3.0 * len(classes) + 1.6))
            gs = fig.add_gridspec(len(classes), len(PROJ_QTY) + 1,
                                  width_ratios=[1] * len(PROJ_QTY) + [1.25])
            inl = (R["layer"] == L) & np.isin(R["cat"], (0, 2, 3, 4))
            for qi, (q, qlab, sc) in enumerate(PROJ_QTY):
                v = R[q][inl].astype(np.float64) * sc
                if q == "d":
                    edges = np.linspace(0, 10, 81)
                else:
                    hi = np.nanpercentile(np.abs(v), 99.5) if np.isfinite(v).any() else 1.0
                    edges = np.linspace(-hi, hi, 81)
                for ci, cname in enumerate(classes):
                    ax = fig.add_subplot(gs[ci, qi])
                    base = (R["layer"] == L) & (R["cls"] == ci)
                    # ONE common normalisation for every category in the panel:
                    # the total of A-in + A-out + B1 + B2 + B3. That keeps the
                    # relative proportions (how much background per target) and
                    # makes A-in and A-out -- two halves of one distribution split
                    # at d = nsig -- join continuously at the ellipse boundary.
                    # Per-category normalisation made that curve jump ~6x there.
                    xall = R[q][base].astype(np.float64)
                    n_all = int(np.isfinite(xall).sum())
                    for c in range(5):
                        x = R[q][base & (R["cat"] == c)].astype(np.float64) * sc
                        x = x[np.isfinite(x)]
                        if not len(x):
                            continue
                        h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), edges)
                        ax.stairs(h / n_all, edges, **PROJ_CAT_STYLE[c],
                                  label=f"{OPJ.CATEGORIES[c]} n={len(x):,}")
                    if q == "d":
                        ax.axvline(nsig, color="#555", lw=0.8, ls=":")
                    ax.set_yscale("log")
                    ax.tick_params(labelsize=6)
                    ax.grid(alpha=.25)
                    ax.legend(fontsize=5.2, loc="upper right")
                    if ci == len(classes) - 1:
                        ax.set_xlabel(qlab + "   (edge bins hold overflow)", fontsize=7)
                    if qi == 0:
                        e = table[cname][f"IL{L}"]
                        ax.set_ylabel(f"{cname}\nfraction of all clusters\nin panel / bin\n"
                                      f"containment {e['containment']:.3f}\n"
                                      f"({e['n_target_inside']:,}/{e['n_target_exists']:,})", fontsize=7)
                    if ci == 0:
                        ax.set_title(qlab, fontsize=8)
            tax = fig.add_subplot(gs[:, -1])
            tax.axis("off")
            tax.text(0.0, 1.0,
                     f"IL{L}: OT-only projection, no KF update between layers\n"
                     f"{r['n_events']:,} events.  Tracks per class: {cls_txt}\n"
                     f"EXCLUDED (no class): {excl_txt}\n"
                     f"OT fit pull widths (MAD): {pull_txt}\n\n" + defs,
                     fontsize=5.4, va="top", ha="left", family="monospace",
                     transform=tax.transAxes, wrap=True)
            fig.suptitle(f"SmartPixels OT-only projection onto IL{L}: target (A) vs background "
                         f"(B1/B2/B3) clusters inside the {nsig:g}-sigma ellipse, by track class",
                         fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            p_ = os.path.join(od, f"spix_ot_projection_IL{L}.png")
            fig.savefig(p_, dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
            pngs.append(p_)
        # ---- perfect tracks, one angle cut on the other angle -------------
        # A 1D look at the 2D (dcotAlpha, dcotBeta) distribution: each angle's
        # residual with the OTHER restricted to |d| <= PROJ_CROSS_CUT, per
        # layer. Overlaying five categories as surfaces in (alpha, beta,
        # count) is unreadable; this shows how much of each category survives
        # the second angle and what its first-angle shape is once it has.
        crosscut = _draw_proj_crosscut(R, Ls, nsig, r["n_events"], tbc["perfect"])
        pdf.savefig(crosscut["fig"])
        p_ = os.path.join(od, "spix_ot_projection_perfect_crosscut.png")
        crosscut["fig"].savefig(p_, dpi=110)
        plt.close(crosscut["fig"])
        pngs.append(p_)
    for cname in classes:
        print(f"    {cname:13s} " + "  ".join(
            f"IL{L} cont {table[cname][f'IL{L}']['containment']:.3f} "
            f"B/cov {sum(table[cname][f'IL{L}'][f'mean_nB{b}_per_covered'] for b in (1, 2, 3)):.2f}"
            for L in Ls))
    out["ot_projection_targets"] = {
        "n_events": r["n_events"], "nsig": nsig, "tracks_by_class": tbc, "excluded": excl,
        "ot_fit_pulls": pulls, "validation_vs_refit": val,
        "candidate_bound_ratio_max": r["bound_ratio_max"], "per_class_layer": table,
        "perfect_crosscut": crosscut["table"],
        "figures": pngs + [pdf_path]}
SECTIONS = [
    ("A. Cluster cones: how much is in reach of a track?",
     "Occupancy and containment around a projected track -- the raw material every"
     " later stage draws on, and the cost floor nothing can go below.",
     [("cone occupancy", study_cone_occupancy, 3),
      ("refit-order cone", study_refit_cone_occupancy, 3),
      ("cone containment + size", study_cone_containment, 2)]),

    ("B. What the SmartPixels angles buy",
     "Whether the per-cluster alpha/beta angles and the charge readout actually"
     " discriminate, and what the longitudinal angle delivers as a z0 estimate.",
     [("angle discrimination", study_angle_discrimination, 2),
      ("charge readout gate", study_charge_gate, 2),
      ("unbiased containment", study_true_containment, 2),
      ("z0 resolution (seeding)", study_z0_resolution, 2),
      ("cluster angles per layer", study_cluster_angles_by_layer, 2),
      ("MVA score correlation per build", study_mva_score_correlation, 2)]),

    ("C. Fitting and cluster-combination choices",
     "How hits should be weighted in the refit, and which activeSP combinations"
     " are worth reading out.",
     [("chi2 weight scan", study_chi2_weight_scan, 2),
      ("combination sweep (activeSP)", study_combination_sweep, 3)]),

    ("D. Seeding: binning, transforms, and which seeds to build",
     "The seeding design itself -- sector sizing, the Hough picture, what each"
     " seed mode recovers that the others do not, and whether the doublets can be"
     " aimed at the triplet's failures instead of duplicating its successes.",
     [("sector bin sizing", study_sector_binning, 2),
      ("hough examples", study_hough_examples, 2),
      ("seed-mode confusion", study_seed_mode_confusion, 3),
      ("seed composition", study_seed_composition, 3),
      ("combined IT+OT seeding", study_combined_it_ot, 3),
      ("seed menu per build", study_seed_menu_by_build, 3)]),

    ("E. OT-only projection: what a refit is offered, by track class",
     "The OT track's own helix and covariance projected onto every IT barrel module"
     " its 4-sigma ellipse overlaps, with no KF update: the majority-owner TP's"
     " cluster (target) against every other cluster in reach, per layer.",
     [("OT-only projection targets vs background", study_ot_projection_targets, 3)]),
]

# Flat view, kept because the figure is still one grid and several studies index
# their row directly.
STUDIES = [t for _, _, group in SECTIONS for t in group]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--outdir", default="eval_refitq/combinatorics")
    ap.add_argument("--only", default=None,
                    help="run only studies whose name contains this substring")
    ap.add_argument("--menu-inputs", default=None,
                    help="comma-separated files/globs for study 15's streamed "
                         "census; defaults to -i. The other studies load their "
                         "inputs whole, so the full ttbar set belongs here.")
    ap.add_argument("--menu-nev", type=int, default=100,
                    help="events for the per-build seed menu study (15)")
    ap.add_argument("--menu-chunk", type=int, default=4,
                    help="events per chunk for study 15's cached census; smaller "
                         "is FASTER (measured 27.8 vs 62.9 s/event at 4 vs 16)")
    ap.add_argument("--menu-budget-gb", type=float, default=0.4)
    ap.add_argument("--menu-rss-gb", type=float, default=8.0)
    ap.add_argument("--site", default=None,
                    help="exported site directory, for the MVA score "
                         "correlation study; defaults to <outdir>/site")
    ap.add_argument("--menu-cache",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "cache"))
    ap.add_argument("--menu-cache-mode", default="auto",
                    choices=["auto", "rebuild", "off", "require"])
    ap.add_argument("--menu-export-tracks", type=int, default=0, nargs="?",
                    const=-1,
                    help="write per-track rows into the census shards for the "
                         "interactive page and the quality MVA; bare flag "
                         "exports every track. This is part of the CACHE KEY, "
                         "so a census built without it cannot be extended with "
                         "it -- set it on the run that is meant to serve both.")
    ap.add_argument("--geometry", default=None,
                    help="module geometry JSON from SmartPixelsModuleGeometryDumper, "
                         "for the OT-only projection study (19)")
    ap.add_argument("--proj-inputs", default=None,
                    help="comma-separated files/globs for study 19, which streams "
                         "its inputs; defaults to -i")
    ap.add_argument("--proj-nev", type=int, default=None,
                    help="event cap for study 19 (default: all)")
    ap.add_argument("--proj-chunk", type=int, default=10,
                    help="events per chunk for study 19 (~0.35 GB RSS at 10)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    hit, cfg, X, K, n_ev = load(args.inputs)
    P = prepare(X, K)
    out = {"config": cfg, "n_events": n_ev, "n_crossings": int(len(X["layer"])),
           "n_clusters": int(len(K["layer"])),
           # side-artefact directory for studies that write their own files;
           # stripped before the JSON is dumped
           "_outdir": args.outdir,
           # studies that need to re-read the file for OTHER activeSP variants
           "_inputs": [f for p in args.inputs for f in (sorted(_glob.glob(p)) or [p])],
           "_hit_table": hit,
           "_menu_inputs": args.menu_inputs, "_menu_nev": args.menu_nev, "_menu_chunk": args.menu_chunk,
           "_menu_budget_gb": args.menu_budget_gb,
           "_menu_rss_gb": args.menu_rss_gb, "_menu_cache": args.menu_cache,
           "_site": args.site,
           "_menu_cache_mode": args.menu_cache_mode,
           "_menu_export_tracks": args.menu_export_tracks,
           "_geometry": args.geometry, "_proj_nev": args.proj_nev,
           "_proj_chunk": args.proj_chunk,
           "_proj_inputs": ([f for g in args.proj_inputs.split(",") for f in
                             (sorted(_glob.glob(g)) or [g])] if args.proj_inputs else None)}

    gp = write_glossary(args.outdir)
    print(f"  glossary: {gp}  (defines crossing, cone, qX vs pXX, containment, max)")

    SEL = [(t, d, [g for g in grp if not args.only or args.only in g[0]])
           for t, d, grp in SECTIONS]
    SEL = [(t, d, g) for t, d, g in SEL if g]
    if not SEL:
        raise SystemExit(f"--only {args.only!r} matched no study")
    ncols = max(n for _, _, grp in SEL for _, _, n in grp)
    # one extra row per section for its banner
    nrows = sum(len(g) for _, _, g in SEL) + len(SEL)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_2d(axes)
    row = 0
    for stitle, sdesc, group in SEL:
        print(f"\n{stitle}\n    {sdesc}")
        for c in range(ncols):
            axes[row][c].axis("off")
        axes[row][0].text(0.0, 0.45, stitle, fontsize=15, fontweight="bold",
                          va="center", ha="left", transform=axes[row][0].transAxes)
        axes[row][0].text(0.0, 0.12, sdesc, fontsize=9, color="#52514e", va="center",
                          ha="left", wrap=True, transform=axes[row][0].transAxes)
        axes[row][0].axhline(0.78, color="#0b0b0b", lw=1.4,
                             xmin=0.0, xmax=ncols * 0.98)
        row += 1
        for (name, fn, n) in group:
            print(f"  study: {name}")
            fn(X, K, P, axes[row], out)
            for c in range(n, ncols):
                axes[row][c].axis("off")
            row += 1
    fig.suptitle(f"SmartPixels combinatorics omnibus — {cfg}, {n_ev} events", y=1.005)
    # qX vs pXX is the confusion most likely to survive into a slide, so it is
    # stamped on the figure itself rather than only in the glossary file.
    fig.text(0.5, -0.004,
             "qX = cone WIDTH chosen (input; k = Phi^-1((1+X/100)/2), nominal, assumes unit pulls)   |   "
             "pXX = percentile over TRACKS (output)   |   containment = MEASURED, conditional on the "
             "hit existing   |   see spix_glossary.txt",
             ha="center", va="top", fontsize=7.5)
    fig.tight_layout()
    png = os.path.join(args.outdir, "spix_combinatorics_omnibus.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    js = os.path.join(args.outdir, "spix_combinatorics_omnibus.json")
    out.pop("_outdir", None)
    out.pop("_inputs", None)
    out.pop("_hit_table", None)
    with open(js, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {png}\nwrote {js}")


if __name__ == "__main__":
    main()
