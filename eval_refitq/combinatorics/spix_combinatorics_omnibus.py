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

THE CONE. Built from the OT-track covariance: projOTSigX/Y = sqrt(diag(H C H^T))
projected to the module, with NO Kalman updates -- the single-shot cold start.
q68 -> 1.00 sigma, q95 -> 1.96 sigma per coordinate. This is the track's own
uncertainty and excludes the measurement term, which is the right choice here:
we are asking how big the search region must be, not how well a hit fits.

    pixi run python eval_refitq/combinatorics/spix_combinatorics_omnibus.py \
        -i <clusters-tier nano.root> -o eval_refitq/combinatorics/ [--variants ...]

WHICH REFIT. --variants picks the refit variants to study: activeSP masks of a
COEXIST nano (tables ...SeedlessRefit<mask>), "canonical" for the unsuffixed
tables of a one-build nano (mask from its ParameterSets, or --canonical-mask),
or "all". Omitted, the input must carry exactly one. Several variants: the
per-variant studies write one page each into <outdir>/<variant>/, the shared
ones (incl. the activeSP sweep (10) over the selected set) one page in <outdir>.

SURFACES. The refit visits the barrel layers IL1-IL4 AND the discs ID1-ID4 (TFPX
disks 1-4, both sides; build 92bad91b2ba onward). Every per-layer quantity here is
per SURFACE, with the nano's codes: 1-4 = IL1-IL4, 11-14 = ID1-ID4 (the refit hit
table's `layer` column and the cluster table's `surface` column carry them). An
activeSP mask character k instruments ILk AND IDk ("AIAI" = IL1, IL3, ID1, ID3).
On a disc module local x is r-phi and local y the RADIUS. The seeding studies
(12-15, 17, 18) are BARREL-ONLY BY DESIGN: the seeders they run
(tracklet_topology_cost, tp_findability) implement the barrel IT seeds only (the
disc seed menu is designed, not implemented), so they get the barrel clusters.

REFIT MONITOR (section F). R1 track-parameter resolution and pulls, R2 hit
selection per surface, R3 the angle-estimate monitor against the emulator's sigma
table, R4 the refit-covariance calibration check, R5 the track MVAs (optional).
Every run writes spix_monitor_summary.json (key numbers + provenance) into
<outdir>; compare two runs with monitor_diff.py.

FIGURES are vector PDF (legends off the data, statistics on table pages).
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import resource
import time

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import textwrap
from typing import NamedTuple

import numpy as np
from statistics import NormalDist
import uproot

from ngtagger.data import spix_nano as SPN

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omnibus_support as OSU   # noqa: E402

SENTINEL = -900.0
LAYERS = SPN.BARREL                    # IL1-IL4: the barrel-only (seeding) studies
SURFACES = SPN.SP_SURFACES             # IL1-IL4, ID1-ID4: every per-surface study
SNAME = {s: SPN.surface_name(s) for s in SURFACES}
CONES = {"q68": 1.0, "q95": 1.959964}
PT_EDGES = np.array([2, 3, 4, 6, 10, 20, 1e9])
ETA_EDGES = np.array([-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4])
CLUSTER_TABLE = SPN.CLUSTER_TABLE
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
     "most per-surface quantities are counted in -- not a track, not a cluster. One "
     "track normally has one crossing per instrumented surface it reaches."),
    ("surface  (IL1-IL4, ID1-ID4)",
     "An inner-tracker surface the refit visits: the barrel layers IL1-IL4 (codes "
     "1-4) and the discs ID1-ID4 (codes 11-14; TFPX disks 1-4, both sides). The "
     "refit visits the surfaces in PATH order along the OT track, outermost first: "
     "e.g. IL4, IL3, IL2, IL1 for a central track, ID4, ID3, ID2, ID1, IL1 for a "
     "forward one. On a disc module local x is r-phi and local y the RADIUS."),
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
     "and points along z on a barrel module, along the RADIUS (inward) on a disc "
     "module. Verified from the CPE resolutions: median sigX = 3.0 um against sigY "
     "= 12.5 um (barrel). Local z is the module NORMAL."),
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
     "The number of hit tuples (one cluster per crossed surface) one refit must "
     "test = PRODUCT of the candidate counts over the INSTRUMENTED surfaces it "
     "crosses (IL1-IL4 and ID1-ID4). A surface with no candidate contributes a "
     "factor of 1, not 0, because the refit skips it and carries on. "
     "Worked: 2,1,0,2 -> 2*1*1*2 = 4;  3,2,1,1 -> 6. Pinned by "
     "tests/test_combination_counting.py."),
    ("activeSP  /  A  /  I",
     "Four-character mask; character k describes the barrel layer ILk AND the disc "
     "IDk together. A = that pair IS instrumented with smart pixels; I = it is not. "
     "An uninstrumented surface delivers NO data at L1 latency at all -- "
     "conventional pixels are read out only after an L1 accept -- so it contributes "
     "nothing to L1 track building, not even a position. Each mask is therefore a "
     "candidate DETECTOR BUILD, not an algorithm setting. Example: AAII = IL1, IL2, "
     "ID1, ID2 instrumented, IL3, IL4, ID3, ID4 not."),
    ("nSP", "Number of instrumented layers = count of 'A' in the mask."),
    (">=2hit / >=3hit  (viability)",
     "Fraction of tracks to which the refit actually attached that many hits. 0% for "
     "every two-layer build is STRUCTURAL (only two layers exist to be hit), not a "
     "performance statement."),
    ("projOTSig{X,Y}",
     "Projected track uncertainty from the OT-track covariance alone -- single shot, "
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
     "surfaces (IL1-IL4, ID1-ID4) with NO Kalman update and NO multiple-scattering "
     "term: what a refit knows before it has included any IT cluster. Each surface "
     "is projected independently from the OT fit, so IL1 is not informed by IL4. Computed "
     "outside the refit (ot_projection.py) and cross-checked against the "
     "producer's own OT-track projection (projOT*) on every shared crossing."),
    ("surface projection",
     "One OT track projected to one IT surface (a barrel layer, or the disc on the "
     "side it heads to). It may span SEVERAL modules: the ellipse is evaluated on "
     "every module it overlaps, not only on the one the helix lands in, which "
     "removes the 'module boundary' bias above."),
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
     "A cluster of the majority-owner TP anywhere on the surface. A-in lies inside the "
     "4-sigma ellipse, A-out outside; A-out residuals are taken on the target's own "
     "module, so they are frame-consistent too. Containment = fraction of surface "
     "projections with at least one A-in, among those where an A cluster exists."),
    ("B1 / B2 / B3  (background inside the ellipse)",
     "Every other cluster inside the 4-sigma ellipse. B1: owned by a TP that also "
     "owns a foreign stub of this track (the trap a contaminated track sets for "
     "itself). B2: owned by any other TP. B3: no TP (unlinked: mostly soft "
     "sub-0.1 GeV deposits, delta rays, curlers). With the emulator angle source "
     "(angleSource emu: every cluster gets an angle, class C for those without a "
     "trustworthy truth helix) B3 clusters carry an angle like any other, so the "
     "angle-using metrics charge them their mismatch; on payload-era nanos they "
     "carried none, and purities against B3 were not representative there."),
    # ---- the refit monitor (section F) ------------------------------------
    ("core RMS / full RMS",
     "The project's resolution convention: the core RMS is the RMS about ZERO of "
     "the residuals kept by an iterative 3-sigma clip about the mean (so a bias "
     "counts), quoted with the fraction outside the core; the full RMS is the plain "
     "RMS of all residuals."),
    ("pull / coverage",
     "Residual over the uncertainty the fit or table claims for it; width 1 = honest. "
     "Coverage = fraction of |pull| < 1, 2, 3 (Gaussian 0.683, 0.954, 0.997)."),
    ("width68",
     "(q84 - q16) / 2 of a distribution: the emulator sigma table's own definition of "
     "an angle resolution, used wherever a measured width is compared with it."),
    ("class T / class C  (tpEmuClass)",
     "The emulator's two routes. T: the cluster has a TP whose helix passes within "
     "1 cm (truth-consistent) and its angle is the SmartPixels response on the truth. "
     "C: everything else (no TP, delta rays, loopers) -- angle from a CNN plus a donor "
     "residual; its 'true' angle (the dominant TP's helix) is unreliable."),
    ("sigma table",
     "The emulator's angle-resolution lookup (emuSigmaSet, e.g. spix_emu_sigma_v2.json): "
     "sigma per surface and |reported cot|, the error the refit gives every emitted "
     "angle. Read from the nano's ParameterSets, its sha256 recorded in the monitor."),
    ("correct hit / wrong hit",
     "An accepted refit hit whose cluster belongs to the OT track's TP "
     "(L1TTrack_tpIdx; the producer's selHitClass 0) / to another TP or none (1, 2)."),
    ("innovation pull",
     "(cluster - running projection) / sqrt(track projected variance + cluster "
     "variance): the KF's own consistency test on a surface (pullX/Y/Alpha/Beta)."),
    ("residual  (du, dv, dcotAlpha, dcotBeta)",
     "Cluster minus OT-only projection, in the local frame of the module the "
     "cluster is on, after projecting the track onto THAT module's plane: du, dv in "
     "um (on a disc: r-phi, radius); dcotAlpha, dcotBeta unitless, cluster = the "
     "sensor's angle estimate "
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
# ANGLE-CAP SENTINEL -- run on every invocation, before any study
# --------------------------------------------------------------------------
# A measured cluster angle is used only when |cot| <= SPN.MEAS_ANGLE_MAX_ABS (12,
# the refit's seedlessRefitMeasAngleMaxAbs; ~85 deg from normal incidence). Payload-era
# producers cleared hasAlpha/hasBeta beyond it; the emu producers set them on every
# cluster, so the loader's cap is now the only protection. This counts, per angle
# (alpha, beta, and the pair that globalClusterPhi/CotTheta are built from) and per
# surface / TP link / tpEmuClass, how many valid angles pass and fail the cap, and
# WARNs when a cell's failing fraction exceeds FACTOR x its reference (a band
# measured once per producer era, stored in ANGLE_CAP_REFERENCE with provenance).
ANGLE_CAP_REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "angle_cap_reference.json")
ANGLE_CAP_FACTOR = 2.0
_SENTINEL_COLS = ["surface", "hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta", "tpIdx"]


def angle_cap_census_files(files, step=10):
    """SPN.angle_cap_census over every cluster (barrel + ID1-ID4) of `files`,
    with the STORED flags. Returns (census, era, n_events): era 'emu' when every
    cluster has both flags set (the emu producers), else 'payload'."""
    acc, n_ev = None, 0
    with uproot.open(f"{files[0]}:Events") as t:
        keys = t.keys()
    cols = _SENTINEL_COLS + (["tpEmuClass"] if SPN.has_column(keys, CLUSTER_TABLE, "tpEmuClass") else [])
    br = SPN.cluster_branches(keys, cols, disc_scope="sp", cap_angles=False)
    for A in uproot.iterate([f"{f}:Events" for f in files], br, step_size=step):
        if MAX_EVENTS is not None and n_ev >= MAX_EVENTS:
            break
        A = A[:MAX_EVENTS - n_ev] if MAX_EVENTS is not None else A
        D, n = SPN.flat_clusters(A, cols, disc_scope="sp", cap_angles=False)
        n_ev += len(n)
        g = SPN.angle_cap_groups(D["surface"], D["tpIdx"], D.get("tpEmuClass"))
        acc = SPN.angle_cap_census_add(acc, SPN.angle_cap_census(D, g))
    a = acc["angles"]
    emu = all(a[x]["all"]["n_flag_set"] == a[x]["all"]["n_total"] for x in ("alpha", "beta"))
    return acc, ("emu" if emu else "payload"), n_ev


def _angle_cap_compare(census, era, ref_path=ANGLE_CAP_REFERENCE):
    """Attach reference band and status to every cell; returns (cells, n_warn, ref_meta).
    Band = FACTOR x max(ref frac_fail, 1 / ref n_valid): the floor keeps a cell whose
    reference saw no failure from warning on the first one."""
    try:
        with open(ref_path) as fh:
            R = json.load(fh)
    except FileNotFoundError:
        R = {}
    E = R.get("eras", {}).get(era)
    factor = R.get("factor", ANGLE_CAP_FACTOR)
    rows, n_warn = [], 0
    for ang, cells in census["angles"].items():
        for lab, c in cells.items():
            r = (E or {}).get("cells", {}).get(ang, {}).get(lab)
            band, status = None, "no reference"
            if r is not None and r["n_valid"] > 0:
                band = factor * max(r["frac_fail"] or 0.0, 1.0 / r["n_valid"])
                if c["frac_fail"] is None:
                    status = "no valid angle"
                elif c["frac_fail"] > band:
                    status, n_warn = "WARNING", n_warn + 1
                else:
                    status = "ok"
            rows.append(dict(c, angle=ang, group=lab, ref_frac_fail=None if r is None else r["frac_fail"],
                             band=band, status=status))
    meta = ({k: v for k, v in E.items() if k != "cells"} if E else None)
    return rows, n_warn, {"reference_file": ref_path, "era": era, "factor": factor,
                          "rule": R.get("rule"), "reference": meta}


def _fmt_frac(f):
    return "-" if f is None else f"{100 * f:.3f}%"


def _angle_cap_pdf(rows, head, pdf_path, per_page=34):
    """Vector table pages: one block of rows per angle."""
    from matplotlib.backends.backend_pdf import PdfPages
    cols = ["group", "N total", "N valid", "N pass", "N fail", "frac fail",
            "N flag set", "N dropped by cap", "ref frac fail", "band", "status"]
    with PdfPages(pdf_path) as pdf:
        for ang in dict.fromkeys(r["angle"] for r in rows):
            sel = [r for r in rows if r["angle"] == ang]
            for p0 in range(0, len(sel), per_page):
                part = sel[p0:p0 + per_page]
                fig = plt.figure(figsize=(11.7, 8.3))
                fig.text(0.03, 0.97, f"Angle-cap sentinel: {ang}  (|cot| <= {SPN.MEAS_ANGLE_MAX_ABS:g})",
                         fontsize=13, fontweight="bold", va="top")
                fig.text(0.03, 0.935, head, fontsize=7.5, va="top", family="monospace")
                cell = [[r["group"], f"{r['n_total']:,}", f"{r['n_valid']:,}", f"{r['n_pass']:,}",
                         f"{r['n_fail']:,}", _fmt_frac(r["frac_fail"]),
                         f"{r.get('n_flag_set', 0):,}", f"{r.get('n_dropped_by_cap', 0):,}",
                         _fmt_frac(r["ref_frac_fail"]), _fmt_frac(r["band"]), r["status"]]
                        for r in part]
                ax = fig.add_axes([0.03, 0.03, 0.94, 0.80])
                ax.axis("off")
                tb = ax.table(cellText=cell, colLabels=cols, loc="upper left", cellLoc="right")
                tb.auto_set_font_size(False)
                tb.set_fontsize(7)
                tb.scale(1.0, 1.15)
                for i, r in enumerate(part, start=1):
                    tb[i, 0].set_text_props(ha="left")
                    if r["status"] == "WARNING":
                        for j in range(len(cols)):
                            tb[i, j].set_facecolor("#f6d2cc")
                pdf.savefig(fig)
                plt.close(fig)


def angle_cap_sentinel(files, outdir, ref_path=ANGLE_CAP_REFERENCE):
    """Count, print, compare with the reference band, and write the table PDF.
    Returns the JSON block."""
    census, era, n_ev = angle_cap_census_files(files)
    rows, n_warn, meta = _angle_cap_compare(census, era, ref_path)
    print(f"\nANGLE-CAP SENTINEL  |cot| <= {SPN.MEAS_ANGLE_MAX_ABS:g}   era={era}  events={n_ev}  "
          f"reference={os.path.basename(ref_path)}"
          + ("" if meta["reference"] else "  (NO REFERENCE FOR THIS ERA: counts only)"))
    print(f"  {'angle':9s} {'group':18s} {'N total':>10s} {'N valid':>10s} {'N pass':>10s} "
          f"{'N fail':>8s} {'frac fail':>10s} {'dropped':>8s} {'band':>9s}  status")
    for r in rows:
        print(f"  {r['angle']:9s} {r['group']:18s} {r['n_total']:10d} {r['n_valid']:10d} "
              f"{r['n_pass']:10d} {r['n_fail']:8d} {_fmt_frac(r['frac_fail']):>10s} "
              f"{r.get('n_dropped_by_cap', 0):8d} {_fmt_frac(r['band']):>9s}  {r['status']}")
    for r in rows:
        if r["status"] == "WARNING":
            print(f"WARNING *** ANGLE-CAP SENTINEL: {r['angle']} {r['group']}: "
                  f"{_fmt_frac(r['frac_fail'])} of valid angles fail |cot| <= "
                  f"{SPN.MEAS_ANGLE_MAX_ABS:g}, above the band {_fmt_frac(r['band'])} "
                  f"({meta['factor']:g} x the {era} reference {_fmt_frac(r['ref_frac_fail'])}). "
                  "The input's angle population differs from the reference: check the producer.")
    if n_warn:
        print(f"WARNING *** ANGLE-CAP SENTINEL: {n_warn} cell(s) outside the band")
    pdf_path = os.path.join(outdir, "spix_angle_cap_sentinel.pdf")
    ref = meta["reference"] or {}
    head = (f"input: {', '.join(os.path.basename(f) for f in files)}   events {n_ev}   era {era} "
            f"(emu = every cluster has both flags set)\n"
            f"valid = not the -999 fill; frac fail = N fail / N valid; dropped by cap = stored flag set "
            f"but not usable (what the loader removes; a -999 under a set flag included)\n"
            f"direction = the (alpha, beta) pair globalClusterPhi/CotTheta are built from: valid when "
            f"both are, passing when both pass\n"
            f"band = {meta['factor']:g} x max(ref frac fail, 1/ref N valid);  reference: "
            f"{ref.get('source', 'none for this era')}  ({ref.get('n_events', '-')} events)\n"
            f"cells outside the band: {n_warn}")
    _angle_cap_pdf(rows, head, pdf_path)
    print(f"  wrote {pdf_path}")
    return {"max_abs": census["max_abs"], "era": era, "n_events": n_ev, "n_warnings": n_warn,
            "band": meta, "cells": rows, "pdf": pdf_path}


def write_angle_cap_reference(files, ref_path=ANGLE_CAP_REFERENCE):
    """Measure the census on `files` and store it as the reference of their era."""
    import datetime
    import subprocess
    census, era, n_ev = angle_cap_census_files(files)
    try:
        with open(ref_path) as fh:
            R = json.load(fh)
    except FileNotFoundError:
        R = {}
    R.setdefault("description",
                 "Reference failing fractions of the measured-angle cap |cot| <= max_abs, per "
                 "angle x cluster group, one entry per producer era. Read by the omnibus's "
                 "angle-cap sentinel; regenerate with spix_combinatorics_omnibus.py "
                 "--write-angle-cap-reference -i <file>.")
    R["rule"] = ("WARNING when a cell's frac_fail > factor x max(reference frac_fail, "
                 "1 / reference n_valid), using the reference of the input's era (emu when "
                 "every cluster has hasAlpha = hasBeta = 1, else payload)")
    R["factor"], R["max_abs"] = ANGLE_CAP_FACTOR, census["max_abs"]
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        if subprocess.run(["git", "-C", here, "status", "--porcelain", "--",
                           os.path.abspath(__file__)], capture_output=True, text=True).stdout.strip():
            commit += " + uncommitted changes to this script"
    except OSError:
        commit = None
    R.setdefault("eras", {})[era] = {
        "source": ", ".join(os.path.abspath(f) for f in files), "n_events": n_ev,
        "created": datetime.date.today().isoformat(), "code_commit": commit,
        "cells": {a: {lab: {"frac_fail": c["frac_fail"], "n_valid": c["n_valid"],
                            "n_fail": c["n_fail"]} for lab, c in cells.items()}
                  for a, cells in census["angles"].items()}}
    with open(ref_path, "w") as fh:
        json.dump(R, fh, indent=1)
    print(f"wrote the {era} reference ({n_ev} events) into {ref_path}")


# --------------------------------------------------------------------------
# loading + the (event, detId) join
# --------------------------------------------------------------------------
# A nano carries one or more SmartPixels REFIT VARIANTS: a prompt SP track table
# plus its per-crossing refit hit table.
#   COEXIST   several activeSP builds refitted side by side in one job; tables
#             L1TSmartPixels{RefitHit,Track}SeedlessRefit<mask>, variant named by
#             its mask (AAIA = IL1, IL2, IL4 instrumented).
#   CANONICAL one build per job (L1TSPStudyNano + @SPActiveLayers<mask>); the
#             unsuffixed L1TSmartPixels{RefitHit,Track}, variant named "canonical".
#             Its mask comes from the job's ParameterSets (the SP track producer's
#             smartPixelsActiveLayers) or from --canonical-mask.
# The study runs on the variants chosen with --variants, never on an implicit
# pick. Only the PROMPT tables are read; the Ext (extended) twins are not used.
CANONICAL = "canonical"
CANONICAL_HIT = "L1TSmartPixelsRefitHit"
COEXIST_HIT_PREFIX = "L1TSmartPixelsRefitHitSeedlessRefit"
MASK_RE = re.compile(r"^[AI]{4}$")
NEED_HIT = ["trackIdx", "layer", "detId", "hitAccepted", "selHitClass",
            "projOTLocalX", "projOTLocalY", "projOTSigX", "projOTSigY",
            "projOTCotAlpha", "projOTCotBeta", "projLocalX", "projLocalY",
            "projCotAlpha", "projCotBeta", "projSigX", "projSigY",
            "recoLocalX", "recoLocalY", "selClusterIdx", "windowMult", "windowTruncated"]


class Variant(NamedTuple):
    name: str   # the mask (coexist) or CANONICAL
    mask: str   # activeSP mask; character i describes TBPX layer i + 1
    hit: str    # the refit hit table

    @property
    def track(self):
        """The variant's SP track table (L1TSmartPixels[...]Track[<suffix>])."""
        return self.hit.replace("RefitHit", "Track")

    @property
    def label(self):
        return self.name if self.name == self.mask else f"{self.name} ({self.mask})"


# --max-events: a DEVELOPMENT cap on the events every in-memory read takes (the
# first N events over the inputs, in order). The streamed studies (19, 20) take
# --proj-nev (defaulted to it); the seeding studies that read their own inputs
# whole (12's OT part, 14) refuse to run under it rather than mix event ranges.
MAX_EVENTS = None


def _concat(files, filter_name):
    """uproot.concatenate of the Events trees, capped at MAX_EVENTS."""
    if MAX_EVENTS is None:
        return uproot.concatenate([f"{f}:Events" for f in files], filter_name=filter_name)
    parts, left = [], MAX_EVENTS
    for f in files:
        if left <= 0:
            break
        with uproot.open(f"{f}:Events") as t:
            n = min(t.num_entries, left)
            parts.append(t.arrays(filter_name=filter_name, entry_stop=n))
        left -= n
    return ak.concatenate(parts)


def _keys(path):
    with uproot.open(f"{path}:Events") as t:
        return set(t.keys())


def _variant_names(keys):
    """Refit variants present in a file's branch set: coexist masks, then CANONICAL."""
    sfx = {k[len(COEXIST_HIT_PREFIX):].split("_", 1)[0] for k in keys
           if k.startswith(COEXIST_HIT_PREFIX)}
    odd = sorted(s for s in sfx if not MASK_RE.match(s))
    if odd:
        print(f"  WARNING: ignoring refit tables with a non-mask suffix: {odd}")
    names = [m for m in SWEEP_ORDER if m in sfx]
    if any(k.startswith(CANONICAL_HIT + "_") for k in keys):
        names.append(CANONICAL)
    return names


def canonical_mask_from_psets(path):
    """activeSP mask of the producer behind the unsuffixed L1TSmartPixelsTrack table.

    Read from the file's ParameterSets tree: the L1TTTrackFlatTableProducer
    writing name=L1TSmartPixelsTrack names its src producer, whose
    smartPixelsActiveLayers ("1110" = IL1-IL3 on) is the mask. None when the
    tree is absent or does not give exactly one answer.
    """
    with uproot.open(path) as F:
        if "ParameterSets" not in F:
            return None
        br = F["ParameterSets"]["IdToParameterSetsBlobs"]
        blob = b"".join(bytes(br.basket(i).data) for i in range(br.num_baskets))
    end = rb"[\x00 ]?\);"          # a string parameter's terminator in the blob
    srcs = set(re.findall(rb"@module_type=\+Z\(L1TTTrackFlatTableProducer" + end
                          + rb"[^>]*?name=\+Z\(L1TSmartPixelsTrack" + end
                          + rb"[^>]*?src=\+t\(([^:)]+)", blob))
    masks = {m for lab in srcs for m in re.findall(
        rb"@module_label=\+Z\(" + re.escape(lab) + end
        + rb"@module_type=\+Z\(L1SmartPixelsTrackProducer" + end
        + rb"[^>]*?smartPixelsActiveLayers=\+Z\(([01]{4})" + end, blob)}
    if len(masks) != 1:
        return None
    return masks.pop().decode().replace("1", "A").replace("0", "I")


def select_variants(files, spec=None, canonical_mask=None):
    """The refit variants to study, as Variant objects, in SWEEP_ORDER.

    spec: None (the input must then carry exactly one variant), "all", or a
    comma list of masks and/or "canonical". Every file must carry the same
    variants, and canonical files the same mask; a requested variant that is
    absent is an error, not a fallback.
    """
    per_file = {f: _variant_names(_keys(f)) for f in files}
    names = per_file[files[0]]
    bad = [f for f, n in per_file.items() if n != names]
    if bad:
        raise SystemExit(f"inputs disagree on their refit variants: {files[0]} has {names}, "
                         f"{bad[0]} has {per_file[bad[0]]}")
    if not names:
        raise SystemExit(f"no SmartPixels refit hit table ({CANONICAL_HIT} or "
                         f"{COEXIST_HIT_PREFIX}<mask>) in {files[0]}")
    if spec is None:
        if len(names) > 1:
            raise SystemExit(f"{len(names)} refit variants in the input: {','.join(names)}.\n"
                             "Choose with --variants: a comma list of them, 'all', or 'canonical'.")
        want = names
    elif spec.strip().lower() == "all":
        want = names
    else:
        want = [CANONICAL if s.strip().lower() == CANONICAL else s.strip()
                for s in spec.split(",") if s.strip()]
        odd = [s for s in want if s != CANONICAL and not MASK_RE.match(s)]
        if odd:
            raise SystemExit(f"--variants: {odd} are neither 'all', 'canonical' nor a "
                             "4-character A/I mask")
        if len(set(want)) != len(want):
            raise SystemExit(f"--variants lists a variant twice: {spec}")
        miss = [s for s in want if s not in names]
        if miss:
            raise SystemExit(f"--variants: {','.join(miss)} not in the input "
                             f"(it carries {','.join(names)})")
        want = [n for n in names if n in want]
    if canonical_mask is not None and not MASK_RE.match(canonical_mask):
        raise SystemExit(f"--canonical-mask must be 4 characters of A/I, got {canonical_mask!r}")
    out = []
    for n in want:
        if n != CANONICAL:
            out.append(Variant(n, n, COEXIST_HIT_PREFIX + n))
            continue
        meta = {f: canonical_mask_from_psets(f) for f in files}
        found = set(meta.values()) - {None}
        if len(found) > 1:
            raise SystemExit(f"canonical inputs mix activeSP builds: {meta}")
        mask = found.pop() if found else None
        if canonical_mask and mask and canonical_mask != mask:
            raise SystemExit(f"--canonical-mask {canonical_mask} contradicts the input's "
                             f"ParameterSets ({mask})")
        mask = mask or canonical_mask
        if mask is None:
            raise SystemExit("cannot read the canonical refit's activeSP mask from the "
                             "input's ParameterSets; give it with --canonical-mask")
        out.append(Variant(CANONICAL, mask, CANONICAL_HIT))
    if canonical_mask and CANONICAL not in want:
        print("  NOTE: --canonical-mask ignored (canonical not selected)")
    return out


def _check_hit_columns(keys, v):
    miss_h = [c for c in NEED_HIT if not SPN.has_column(keys, v.hit, c)]
    miss_c = ([CLUSTER_TABLE] if not any(k.startswith(CLUSTER_TABLE + "_") for k in keys)
              else [c for c in ("layer", "detId", "localX", "localY", "charge",
                                "tpPt", "tpLocalCotAlpha", "tpLocalCotBeta", "tpIdx")
                    if not SPN.has_column(keys, CLUSTER_TABLE, c)])
    if miss_h or miss_c:
        raise SystemExit(
            "input cannot support this study.\n"
            f"  missing from {v.hit}: {miss_h or 'none'}\n"
            f"  missing from {CLUSTER_TABLE}: {miss_c or 'none'}\n\n"
            "Produce a Clusters-tier file:\n"
            "  test/makeSpixConfig.py --pu 200 --tier clusters-truth "
            "--variant seedlessRefit:1111 --needs-truth -o <out>.py")


def _matched_tp_branch(files, v):
    """Variant v's track TP link branch, L1TSmartPixels[Ext]Track<suffix>_tpIdx:
    the track's weighted cluster-majority TP (strict majority of 1/2 per stub
    cluster + 1 per accepted IT cluster; -1 unmatched or tied, <= -3 a TP not in
    L1TTP)."""
    return SPN.branch_name(_keys(files[0]), v.track, "tpIdx")


def _ttc():
    """tracklet_topology_cost, loaded by path (the studies below do the same)."""
    import importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracklet_topology_cost.py")
    _sp = _ilu.spec_from_file_location("_ttc", _p)
    M = _ilu.module_from_spec(_sp); _sp.loader.exec_module(M)
    return M


def load_clusters(files):
    """Variant-independent inputs: the cluster table K (barrel AND discs), the OT
    tracks' pt/eta (flat, with per-event offsets) and the event count."""
    ccols = ["surface", "layer", "detId", "localX", "localY", "charge", "tpPt",
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
                "hasBeta": "clHasBeta", "tpEmuClass": "tpEmuClass"}
    rcols = ["pt", "eta", "tpIdx"]

    avail = _keys(files[0])
    got = [c for c in optional if SPN.has_column(avail, CLUSTER_TABLE, c)]
    # The WHOLE table (disc_scope "all": nothing filtered), so selClusterIdx -- the
    # refit's row of the file's table, barrel or disc -- indexes K directly. The
    # surfaces are IL1-IL4 then ID1-ID4 on a discScope-sp nano; layer is 0 on a disc.
    C = _concat(files, SPN.cluster_branches(avail, ccols + got, disc_scope="all"))
    C = SPN.cluster_columns(C, ccols + got, disc_scope="all")
    R = _concat(files, [f"{REF}_{c}" for c in rcols])
    n_ev = len(R)
    ntrk = ak.to_numpy(ak.num(R[f"{REF}_pt"]))
    T = {"off": np.concatenate([[0], np.cumsum(ntrk)])}
    for c in rcols:
        T[c] = ak.to_numpy(ak.flatten(R[f"{REF}_{c}"]))

    ncl = ak.to_numpy(ak.num(C["layer"]))
    ev_c = np.repeat(np.arange(len(ncl)), ncl)
    K = {c: ak.to_numpy(ak.flatten(C[c])) for c in ccols}
    for c in got:
        K[optional[c]] = ak.to_numpy(ak.flatten(C[c]))
    # THE ANGLE CAP ON THE GLOBAL DIRECTION. gClPhi/gClCotTheta (and their sigmas)
    # are built by the producer from BOTH module-frame angles whatever the flags
    # say, so a cluster whose alpha or beta fails the cap (or is the -999 fill,
    # which on payload-era files becomes a constant gClCotTheta ~ -1) carried a
    # direction every study below used. Set those to the -999 fill, so each
    # study's existing "> SENTINEL" gate drops them; studies 13/14 hand the flags
    # to the seeder, which gates on SPN.direction_usable itself.
    pair = {"hasAlpha": "clHasAlpha", "localCotAlpha": "clLocalCotAlpha",
            "hasBeta": "clHasBeta", "localCotBeta": "clLocalCotBeta"}
    if all(v in K for v in pair.values()):
        dir_ok = SPN.direction_usable({k: K[v] for k, v in pair.items()})
        for c in SPN.DIRECTION_COLUMNS:
            if optional.get(c) in K:
                K[optional[c]] = np.where(dir_ok, K[optional[c]], np.float32(-999.0))
        print(f"  angle cap: global direction unusable on {int((~dir_ok).sum())} of "
              f"{len(dir_ok)} clusters (alpha or beta fails |cot| <= "
              f"{SPN.MEAS_ANGLE_MAX_ABS:g} or is absent); set to -999")
    K["event"] = ev_c
    K["_evt_base"] = np.concatenate([[0], np.cumsum(ncl)])
    # per-cluster TP truth at the POCA (tp_d0, tp_z0, ...) from the L1TTP table;
    # study (12) facets on tp_d0 and skips without it
    M = _ttc()
    if all(SPN.has_column(avail, M.TP_TABLE, c) for c in M.TP_COLS):
        TPA = _concat(files, M.tp_branches(files[0]))
        M.attach_tp_truth(K, M.tp_table(TPA, np.arange(len(ncl))))
    sc = np.bincount(K["surface"].astype(np.int64), minlength=23)
    print(f"files={len(files)} events={n_ev} clusters={len(K['layer'])} ("
          + ", ".join(f"{SPN.surface_name(s)} {sc[s]}" for s in np.flatnonzero(sc)) + ")")
    return K, T, n_ev


def barrel_view(K):
    """The barrel rows of K (IL1-IL4), every column but the row-index bookkeeping:
    the input of the barrel-only seeding studies, whose seeders key on the layer
    (0 on a disc) and implement the barrel IT seeds only."""
    m = ~SPN.is_disc_surface(K["surface"])
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) and not k.startswith("_") else v)
            for k, v in K.items()}


def surface_axis(ax, surfaces=SURFACES, label="surface"):
    """Categorical x axis IL1..IL4, ID1..ID4 (values are drawn at 0..n-1)."""
    ax.set_xticks(range(len(surfaces)))
    ax.set_xticklabels([SPN.surface_name(s) for s in surfaces], fontsize=7)
    ax.set_xlabel(label)


def per_surface(values, surfaces=SURFACES):
    """{IL1: v, ..., ID4: v} for a list aligned with `surfaces` (NaN -> None in JSON)."""
    return {SPN.surface_name(s): (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                                  else float(v)) for s, v in zip(surfaces, values)}


def load_crossings(files, v, T):
    """Variant v's refit crossings, tagged with event index, the owning OT
    track's pt/eta (T from load_clusters) and the SP track's matched TP."""
    avail = _keys(files[0])
    _check_hit_columns(avail, v)
    H = _concat(files, SPN.branch_names(avail, v.hit, NEED_HIT))
    H = SPN.table_columns(H, v.hit, NEED_HIT)
    # NB the hit table's `layer` is the SURFACE code: 1-4 IL1-IL4, 11-14 ID1-ID4
    ncross = ak.to_numpy(ak.num(H["layer"]))
    ev_x = np.repeat(np.arange(len(ncross)), ncross)
    X = {c: ak.to_numpy(ak.flatten(H[c])) for c in NEED_HIT}
    X["event"] = ev_x
    for c in ("pt", "eta"):
        X[f"trk_{c}"] = T[c][T["off"][ev_x] + X["trackIdx"].astype(np.int64)]
    # the OT track's own truth link (TTTrackAssociator, loosely genuine): the TP the
    # producer's selHitClass is relative to
    X["ot_tpIdx"] = T["tpIdx"][T["off"][ev_x] + X["trackIdx"].astype(np.int64)]

    vname = _matched_tp_branch(files, v)
    vtrk = _concat(files, [vname])
    mtp = ak.to_numpy(ak.flatten(vtrk[vname]))
    nmt = ak.to_numpy(ak.num(vtrk[vname]))
    offm = np.concatenate([[0], np.cumsum(nmt)])
    X["trk_tpIdx"] = mtp[offm[ev_x] + X["trackIdx"].astype(np.int64)]
    print(f"variant={v.label} table={v.hit} crossings={len(X['layer'])}")
    return X


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
    dx = K["localX"][ci] - X["projOTLocalX"][xi]
    dy = K["localY"][ci] - X["projOTLocalY"][xi]
    sx = X["projOTSigX"][xi]
    sy = X["projOTSigY"][xi]
    good = (sx > 0) & (sy > 0) & (X["projOTLocalX"][xi] > SENTINEL)
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
    P["dCotA"] = K["tpLocalCotAlpha"][ci] - X["projOTCotAlpha"][xi]
    P["dCotB"] = K["tpLocalCotBeta"][ci] - X["projOTCotBeta"][xi]
    P["ang_ok"] = (K["tpLocalCotAlpha"][ci] > SENTINEL) & (X["projOTCotAlpha"][xi] > SENTINEL)
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
REGIONS = ("barrel", "disc")


def _region(surf, region):
    """Crossing/cluster mask of a region: barrel (IL1-IL4), disc (ID1-ID4) or all."""
    d = SPN.is_disc_surface(surf)
    return ~d if region == "barrel" else d if region == "disc" else np.ones(len(surf), bool)


def _surface_means(vals, surf, surfaces=SURFACES):
    return [float(vals[surf == S].mean()) if (surf == S).any() else float("nan") for S in surfaces]


def study_cone_occupancy(X, K, P, ax_row, out):
    """(1) clusters inside the seed cone, per surface and vs pT / eta (barrel and
    disc crossings separately)."""
    xs = X["layer"]
    xpos = np.arange(len(SURFACES))
    res = {}
    ax = ax_row[0]
    for cname, k in CONES.items():
        m = in_cone(P, k)
        cnt = np.bincount(P["xi"][m], minlength=len(xs)).astype(float)
        per = _surface_means(cnt, xs)
        res[cname] = {"per_surface_mean": per_surface(per)}
        ax.plot(xpos, per, marker="o", label=f"seed cone {cname}")
        P[f"cnt_{cname}"] = cnt
    tot = np.bincount(P["xi"], minlength=len(xs)).astype(float)
    ax.plot(xpos, _surface_means(tot, xs), marker="s", ls="--", color="k", label="all on module")
    ax.set_yscale("log"); surface_axis(ax); ax.set_ylabel("clusters per crossing [count]")
    ax.set_title("(1) candidates per crossing"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    res["all_on_module_per_surface"] = per_surface(_surface_means(tot, xs))

    for ax, var, edges, lab in ((ax_row[1], "trk_pt", PT_EDGES, r"track $p_T$ [GeV]"),
                                (ax_row[2], "trk_eta", ETA_EDGES, r"track $\eta$")):
        v = X[var]
        idx = np.digitize(v, edges) - 1
        ctr = [0.5 * (edges[i] + min(edges[i + 1], 40)) for i in range(len(edges) - 1)]
        for cname in CONES:
            cnt = P[f"cnt_{cname}"]
            for reg, ls in (("barrel", "-"), ("disc", "--")):
                r = _region(xs, reg)
                ys = [cnt[r & (idx == i)].mean() if (r & (idx == i)).any() else np.nan
                      for i in range(len(edges) - 1)]
                ax.plot(ctr, ys, marker="o", ls=ls, label=f"{cname} {reg}")
                res.setdefault(f"vs_{var}", {}).setdefault(cname, {"bin_centres": [float(c) for c in ctr]})[reg] = \
                    [float(y) for y in ys]
        ax.set_xlabel(lab); ax.set_ylabel("clusters in cone per crossing [count]")
        ax.set_title(f"(1) cone occupancy vs {lab}"); ax.legend(fontsize=6); ax.grid(alpha=.3)
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
         & (X["projOTSigX"] > 0) & (X["projOTLocalX"] > SENTINEL)
    res = {"n_reference_hits": int(ok.sum()), "conditioning":
           "selected-and-truth-correct hits only; biased upward, see docstring"}
    ax = ax_row[0]
    for cname, k in CONES.items():
        nx = np.abs(X["recoLocalX"] - X["projOTLocalX"]) / np.where(X["projOTSigX"] > 0, X["projOTSigX"], np.nan)
        ny = np.abs(X["recoLocalY"] - X["projOTLocalY"]) / np.where(X["projOTSigY"] > 0, X["projOTSigY"], np.nan)
        inside = ok & (nx < k) & (ny < k)
        eff = [inside[ok & (X["layer"] == S)].mean() if (ok & (X["layer"] == S)).any() else np.nan
               for S in SURFACES]
        ax.plot(range(len(SURFACES)), eff, marker="o", label=cname)
        res[cname] = {"per_surface_eff": per_surface(eff),
                      "overall_eff": float(inside[ok].mean()) if ok.any() else float("nan")}
    # The cone is a per-coordinate BOX cut at k sigma, so for an ideal Gaussian
    # projection the JOINT containment is (2*Phi(k)-1)^2, not 2*Phi(k)-1. Drawing
    # 0.68/0.95 here would make a correctly-sized cone look broken.
    from math import erf, sqrt
    for k, c in ((CONES["q68"], "grey"), (CONES["q95"], "grey")):
        ideal = erf(k / sqrt(2.0)) ** 2
        ax.axhline(ideal, color=c, ls=":", lw=1, label=f"ideal box {ideal:.3f} (k={k:.2f})")
        res.setdefault("ideal_box_containment", {})[f"k={k:.2f}"] = float(ideal)
    surface_axis(ax); ax.set_ylabel("containment efficiency [fraction]")
    ax.set_title("(2) correct hit inside the seed cone"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_ylim(0, 1.05)

    # cone size itself, the thing that drives (1) and (2) together
    ax = ax_row[1]
    for nm, key in (("$\\sigma_x$ (r-phi)", "projOTSigX"), ("$\\sigma_y$ (z | disc: r)", "projOTSigY")):
        v = [np.median(X[key][(X["layer"] == S) & (X[key] > 0)]) * 1e4
             if ((X["layer"] == S) & (X[key] > 0)).any() else np.nan for S in SURFACES]
        ax.plot(range(len(SURFACES)), v, marker="o", label=nm)
        res.setdefault("cone_sigma_um", {})[key] = per_surface(v)
    ax.set_yscale("log"); surface_axis(ax); ax.set_ylabel(r"median cone $\sigma$ [$\mu$m]")
    ax.set_title("(2) seed cone size"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    out["cone_containment"] = res


def study_angle_discrimination(X, K, P, ax_row, out):
    """(3) how much of the in-cone pool could an alpha/beta cut remove?

    UPPER BOUND, not a capability: the cluster angles here are UNSMEARED truth
    (tpLocalCotAlpha/Beta). A real smart-pixel angle carries the PixelAV response
    smear, so achievable rejection is strictly worse than this. Barrel (solid) and
    disc (dashed) crossings separately: on a disc local y is the radius, so
    cotBeta is the radial angle there.
    """
    m = in_cone(P, CONES["q95"]) & P["ang_ok"]
    correct = (X["hitAccepted"] > 0) & (X["selHitClass"] == 0)
    same = P["is_selected"] & correct[P["xi"]]
    res = {"note": "unsmeared truth angles: an UPPER BOUND on angle rejection"}
    xs = X["layer"][P["xi"]]
    for ax, d, nm in ((ax_row[0], P["dCotA"], r"$\Delta\cot\alpha$"),
                      (ax_row[1], P["dCotB"], r"$\Delta\cot\beta$")):
        rng = np.nanpercentile(np.abs(d[m]), 99) if m.any() else 1.0
        bins = np.linspace(-rng, rng, 80)
        key = "dCotAlpha" if "alpha" in nm else "dCotBeta"
        for reg, ls in (("barrel", "-"), ("disc", "--")):
            r = _region(xs, reg)
            a_sig, a_bkg = d[m & same & r], d[m & ~same & r]
            for arr, lab in ((a_bkg, "other clusters in cone"), (a_sig, "the track's own cluster")):
                if len(arr) > 10:
                    ax.hist(arr, bins=bins, histtype="step", density=True, ls=ls,
                            label=f"{reg}: {lab} (n={len(arr)})")
            if len(a_sig) > 10 and len(a_bkg) > 10:
                for q in (0.68, 0.95):
                    cut = np.quantile(np.abs(a_sig), q)
                    e = res.setdefault(reg, {}).setdefault(key, {})
                    e[f"keep{int(q*100)}_cut"] = float(cut)
                    e[f"keep{int(q*100)}_bkg_rejected"] = float(np.mean(np.abs(a_bkg) > cut))
        ax.set_xlabel(nm + " [unitless]"); ax.set_ylabel("density [1/bin]"); ax.legend(fontsize=6); ax.grid(alpha=.3)
        ax.set_title(f"(3) {nm} vs track expectation")
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
        ax2.plot(Ns, [keep[N] for N in Ns], marker="o", color="k", label="all surfaces")
        xs = X["layer"][xi][match]
        for reg, ls in (("barrel", "-"), ("disc", "--")):
            rr = r[_region(xs, reg)]
            if len(rr) > 10:
                kr = {N: float(np.mean(rr < N)) for N in Ns}
                res.setdefault("survival_vs_topN_by_region", {})[reg] = kr
                ax2.plot(Ns, [kr[N] for N in Ns], marker=".", ls=ls, label=reg)
        ax2.set_xscale("log", base=2); ax2.set_xlabel("clusters read out per module, top-N by charge [count]")
        ax2.set_ylabel("needed clusters kept [fraction]"); ax2.legend(fontsize=6)
        ax2.set_title("(4) readout gate efficiency"); ax2.grid(alpha=.3); ax2.set_ylim(0, 1.05)
    out["charge_gate"] = res


def study_true_containment(X, K, P, ax_row, out):
    """(5) UNBIASED containment: of the clusters that genuinely belong to this
    track's TrackingParticle and sit on the module it crosses, how many does the
    cone hold?

    This is the study that selClusterIdx and the cluster TP link were added for. Study (2)
    can only ever see clusters the current static window already offered, so it
    cannot detect a true cluster the window never showed the fit -- exactly the
    failure a cone redesign is meant to fix. Here the truth clusters are found by
    joining the cluster's tpIdx to the track's tpIdx, independently of what the
    window did, so a cluster the window missed still counts against the cone.
    """
    xi, ci = P["xi"], P["ci"]
    have = SPN.tp_matched(X["trk_tpIdx"][xi]) & SPN.tp_matched(K["tpIdx"][ci])
    is_true = have & (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    res = {"n_true_pairs": int(is_true.sum())}
    ax = ax_row[0]
    for cname, k in CONES.items():
        m = is_true & P["good"]
        if m.sum() < 20:
            continue
        inside = m & in_cone(P, k)
        eff = []
        for S in SURFACES:
            sel = m & (X["layer"][xi] == S)
            eff.append(float(inside[sel].mean()) if sel.any() else np.nan)
        ax.plot(range(len(SURFACES)), eff, marker="o", label=f"{cname} (unbiased)")
        res[cname] = {"per_surface_eff": per_surface(eff), "overall_eff": float(inside[m].mean())}
    # the biased version, for direct contrast on the same axes
    bc = out.get("cone_containment", {})
    for cname in CONES:
        if cname in bc:
            ax.plot(range(len(SURFACES)), [bc[cname]["per_surface_eff"][SNAME[S]] or np.nan for S in SURFACES],
                    marker="s", ls="--", alpha=.6, label=f"{cname} (window-conditioned)")
    surface_axis(ax); ax.set_ylabel("containment efficiency [fraction]")
    ax.set_title("(5) TRUE containment vs window-conditioned")
    ax.legend(fontsize=6); ax.grid(alpha=.3); ax.set_ylim(0, 1.05)

    # how many true clusters does a track even have on the module it crosses?
    ax = ax_row[1]
    n_true = np.bincount(xi[is_true], minlength=len(X["layer"]))
    mt = _surface_means(n_true.astype(float), X["layer"])
    res["mean_true_clusters_on_module"] = per_surface(mt)
    ax.plot(range(len(SURFACES)), mt, marker="o", color="C0")
    surface_axis(ax); ax.set_ylabel("true clusters on crossed module [count]")
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

    SURFACES. The scan pools the crossings of every surface, barrel and disc,
    because the producer applies ONE metric everywhere; the purity of the
    baseline and of the 4D optimum is also reported per surface (IL1-ID4), so a
    weight set that helps the barrel at the discs' expense is visible.

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
    # An angle enters only where the cap allows it (SPN.angle_usable: flag set and
    # |cot| <= MEAS_ANGLE_MAX_ABS), as in the refit. Before 2026-09-27 only the -999
    # fill was excluded, so |cot| > 12 angles entered even on payload-era files,
    # whose flags were already cleared there.
    hasA = K.get("clHasAlpha", np.ones(len(K["clLocalCotAlpha"]), np.uint8))
    hasB = K.get("clHasBeta", np.ones(len(K["clLocalCotBeta"]), np.uint8))
    okA = SPN.angle_usable(hasA[ci], K["clLocalCotAlpha"][ci]) & (K["clSigAlpha"][ci] > 0) \
          & (X["projCotAlpha"][xi] > SENTINEL)
    okB = SPN.angle_usable(hasB[ci], K["clLocalCotBeta"][ci]) & (K["clSigBeta"][ci] > 0) \
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
    have = SPN.tp_matched(X["trk_tpIdx"][xi]) & SPN.tp_matched(K["tpIdx"][ci])
    is_true = have & (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    ncross = len(X["layer"])
    has_true = np.zeros(ncross, dtype=bool)
    has_true[xi[is_true]] = True
    inwin = P["good"]

    def purity(wrphi, wrz, wa, wb, per_surface_out=False):
        """Fraction of crossings whose lowest-cost candidate is the correct one
        (per_surface_out: also {surface name: (purity, n)})."""
        cost = wrphi * dx * dx + wrz * dy * dy + wa * da2 + wb * db2
        cost = np.where(inwin, cost, np.inf)
        best = np.full(ncross, np.inf)
        np.minimum.at(best, xi, cost)
        picked_true = np.zeros(ncross, dtype=bool)
        sel = np.isfinite(cost) & (cost <= best[xi]) & is_true
        picked_true[xi[sel]] = True
        d = has_true & np.isfinite(best)
        tot = (float(picked_true[d].mean()) if d.any() else float("nan"), int(d.sum()))
        if not per_surface_out:
            return tot
        ps = {SNAME[S]: {"purity": float(picked_true[d & (X["layer"] == S)].mean())
                         if (d & (X["layer"] == S)).any() else None,
                         "n": int((d & (X["layer"] == S)).sum())} for S in SURFACES}
        return tot, ps

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
    (base, n_den), base_ps = purity(1, 1, 1, 1, per_surface_out=True)
    res["baseline_all_unit_weights"] = {"purity": base, "n_crossings": n_den, "per_surface": base_ps}
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
                     "purity": s4[b4], "per_surface": purity(1.0, wz4, wa4, wb4, per_surface_out=True)[1]}
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
        lk = SPN.tp_matched(K["tpIdx"])
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
    xpos = np.arange(len(SURFACES))
    res = {}
    ax = ax_row[0]
    for cname, k in CONES.items():
        ms, mr = in_cone(P, k), in_cone_refit(P, k)
        cs = np.bincount(P["xi"][ms], minlength=len(xlay)).astype(float)
        cr = np.bincount(P["xi"][mr], minlength=len(xlay)).astype(float)
        P[f"cnt_rf_{cname}"] = cr
        ys = _surface_means(cs, xlay)
        yr = _surface_means(cr, xlay)
        ax.plot(xpos, ys, marker="o", ls="--", label=f"seed {cname}")
        ax.plot(xpos, yr, marker="s", label=f"refit-order {cname}")
        res[cname] = {"seed_per_surface": per_surface(ys),
                      "refit_per_surface": per_surface(yr),
                      "reduction_per_surface": per_surface([a / b if b > 0 else float("nan")
                                                            for a, b in zip(ys, yr)])}
    ax.plot(xpos, _surface_means(np.bincount(P["xi"], minlength=len(xlay)).astype(float), xlay),
            marker="^", ls=":", color="k", label="all on module")
    ax.set_yscale("log"); surface_axis(ax)
    ax.set_ylabel("candidates per crossing [count]")
    ax.set_title("(9) seed cone vs refit-order cone")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # cone half-width itself, in microns: the mechanism behind the count above
    ax = ax_row[1]
    hs, hr = [], []
    for S in SURFACES:
        m = xlay == S
        a = X["projOTSigX"][m]; b = X["projSigX"][m]
        hs.append(float(np.median(a[a > 0]) * 1e4) if (a > 0).any() else np.nan)
        hr.append(float(np.median(b[b > 0]) * 1e4) if (b > 0).any() else np.nan)
    ax.plot(xpos, hs, marker="o", ls="--", label="seed sigma_x")
    ax.plot(xpos, hr, marker="s", label="refit-order sigma_x")
    surface_axis(ax); ax.set_ylabel(r"median projected $\sigma_x$ (r-phi) [$\mu$m]")
    ax.set_title("(9) projection cone half-width"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    res["median_sigma_x_um"] = {"seed": per_surface(hs), "refit": per_surface(hr)}

    # does the gain survive at low pT, where scattering is worst?
    ax = ax_row[2]
    idx = np.digitize(X["trk_pt"], PT_EDGES) - 1
    ctr = [0.5 * (PT_EDGES[i] + min(PT_EDGES[i + 1], 40)) for i in range(len(PT_EDGES) - 1)]
    for cname in CONES:
        cs = P[f"cnt_{cname}"] if f"cnt_{cname}" in P else None
        cr = P[f"cnt_rf_{cname}"]
        if cs is None:
            continue
        for reg, ls in (("barrel", "-"), ("disc", "--")):
            rg = _region(xlay, reg)
            ratio = []
            for i in range(len(PT_EDGES) - 1):
                m = rg & (idx == i)
                if not m.any():
                    ratio.append(np.nan); continue
                a, b = cs[m].mean(), cr[m].mean()
                ratio.append(float(a / b) if b > 0 else np.nan)
            ax.plot(ctr, ratio, marker="o", ls=ls, label=f"{cname} {reg}")
            res.setdefault("reduction_vs_pt", {}).setdefault(cname, {"bin_centres": [float(c) for c in ctr]})[reg] = ratio
    ax.axhline(1.0, color="k", lw=.8, ls=":")
    ax.set_xlabel(r"track $p_T$ [GeV]"); ax.set_ylabel("seed / refit-order candidates [ratio, unitless]")
    ax.set_title("(9) combinatorics reduction vs $p_T$"); ax.legend(fontsize=6); ax.grid(alpha=.3)

    print("\n=== (9) refit-order cone vs naive seed cone ===")
    print("  median projected sigma_x [um] by surface " + " ".join(f"{SNAME[S]:>7}" for S in SURFACES))
    print(f"    seed        : " + "  ".join(f"{v:7.1f}" for v in hs))
    print(f"    refit-order : " + "  ".join(f"{v:7.1f}" for v in hr))
    for cname in CONES:
        r = res[cname]
        print(f"  {cname}: candidates/crossing seed "
              + "/".join(f"{(v if v is not None else np.nan):.2f}" for v in r["seed_per_surface"].values())
              + "  refit " + "/".join(f"{(v if v is not None else np.nan):.2f}" for v in r["refit_per_surface"].values())
              + "  reduction " + "/".join(f"{(v if v is not None else np.nan):.2f}x" for v in r["reduction_per_surface"].values()))
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

    ok = SPN.tp_matched(K["tpIdx"]) & (K["gClCotTheta"] > SENTINEL) & (K["tpGClCotTheta"] > SENTINEL)
    if "clHasBeta" in K:
        ok &= K["clHasBeta"] > 0
    d = K["gClCotTheta"] - K["tpGClCotTheta"]          # cot(theta) residual
    dz0 = K["globalR"] * d                              # cm; z0 = z - r cot(theta)
    eta = np.arcsinh(K["tpGClCotTheta"])                # cot(theta) = sinh(eta)
    lay = K["surface"]                                  # IL1-IL4, ID1-ID4 (z0 = z - r cot on both)
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

    # ---- per surface, and per surface x |eta| -------------------------------
    per_layer = {}
    for L in SURFACES:
        m = ok & (lay == L)
        s_cot = robust_sigma(d[m])
        s_z0 = robust_sigma(dz0[m])
        per_layer[SNAME[L]] = {
            "n": int(m.sum()),
            "median_r_cm": float(np.median(K["globalR"][m])) if m.any() else float("nan"),
            "sigma_cotTheta_mad": s_cot[0], "sigma_cotTheta_q68": s_cot[1],
            "sigma_z0_cm_mad": s_z0[0], "sigma_z0_cm_q68": s_z0[1],
            "z0_slices": float(Zspan / (4.0 * s_z0[0])) if s_z0[0] == s_z0[0] else float("nan"),
        }
    res["per_surface"] = per_layer

    eta_tab = {}
    for i in range(len(ETA_RES_EDGES) - 1):
        lo, hi = ETA_RES_EDGES[i], ETA_RES_EDGES[i + 1]
        me = ok & (np.abs(eta) >= lo) & (np.abs(eta) < hi)
        row = {}
        for L in SURFACES:
            s = robust_sigma(dz0[me & (lay == L)])
            row[SNAME[L]] = s[0]
        row["n"] = int(me.sum())
        row["sigma_cotTheta_mad"] = robust_sigma(d[me])[0]
        eta_tab[f"{lo:.1f}-{hi:.1f}"] = row
    res["vs_abs_eta"] = eta_tab

    # ---- the two design targets separately (barrel and disc clusters) -------
    if "tpPt" in K:
        for reg in REGIONS:
            rg = _region(lay, reg)
            for nm, sel in (("A_pt_gt_2", ok & rg & (K["tpPt"] > 2.0)),
                            ("B_pt_0p5_to_2", ok & rg & (K["tpPt"] > 0.5) & (K["tpPt"] <= 2.0))):
                s = robust_sigma(dz0[sel])
                res.setdefault("by_target", {}).setdefault(reg, {})[nm] = {
                    "n": int(sel.sum()), "sigma_z0_cm_mad": s[0],
                    "z0_slices": float(Zspan / (4.0 * s[0])) if s[0] == s[0] else float("nan")}

    # ---- cluster-weighted headline + the pull check, barrel and disc --------
    res["overall"], res["pull"] = {}, {}
    for reg in REGIONS:
        rg = ok & _region(lay, reg)
        s_all = robust_sigma(dz0[rg])
        res["overall"][reg] = {"sigma_z0_cm_mad": s_all[0], "sigma_z0_cm_q68": s_all[1],
                               "z0_slices": float(Zspan / (4.0 * s_all[0])) if s_all[0] == s_all[0]
                               else float("nan"),
                               "sigma_for_30_slices_cm": float(Zspan / 120.0)}
        if "gSigCotTheta" in K:
            pm = rg & (K["gSigCotTheta"] > 0)
            pull = d[pm] / K["gSigCotTheta"][pm]
            pw = robust_sigma(pull)
            res["pull"][reg] = {"n": int(pm.sum()), "width_mad": pw[0], "width_q68": pw[1],
                                "median": float(np.median(pull)) if pm.any() else float("nan"),
                                "note": "width far from 1.0 => the stored sigma is miscalibrated"}
        print(f"   (7) {reg}: sigma(z0) {s_all[0]*1e4:.0f} um "
              f"-> {res['overall'][reg]['z0_slices']:.1f} usable z0 slices "
              f"(design assumed 30, which needs {Zspan/120.0*1e4:.0f} um)"
              + (f", pull width {res['pull'][reg]['width_mad']:.3f} (want ~1.0)" if reg in res["pull"] else ""))
    for L in SURFACES:
        p = per_layer[SNAME[L]]
        print(f"       {SNAME[L]}: r={p['median_r_cm']:.1f} cm  sigma(cot)={p['sigma_cotTheta_mad']:.4f}"
              f"  sigma(z0)={p['sigma_z0_cm_mad']*1e4:.0f} um  slices={p['z0_slices']:.1f}")

    # ---- figures -------------------------------------------------------------
    ax = ax_row[0]
    for k_eta, row in eta_tab.items():
        ys = [row[SNAME[L]] * 1e4 for L in SURFACES]
        ax.plot(range(len(SURFACES)), ys, marker="o", label=rf"$|\eta|$ {k_eta}")
    ax.axhline(Zspan / 120.0 * 1e4, color="k", ls=":", lw=1, label="30 slices")
    ax.set_yscale("log"); surface_axis(ax)
    ax.set_ylabel(r"robust $\sigma(z_0)$ [$\mu$m]")
    ax.set_title(r"(7) per-cluster $z_0$ resolution"); ax.legend(fontsize=6); ax.grid(alpha=.3)

    ax = ax_row[1]
    for reg, ls in (("barrel", "-"), ("disc", "--")):
        rg = ok & _region(lay, reg)
        if reg in res["pull"]:
            pm = rg & (K["gSigCotTheta"] > 0)
            ax.hist(np.clip(d[pm] / K["gSigCotTheta"][pm], -5, 5), bins=80, ls=ls,
                    histtype="step", density=True, label=f"{reg} pull (w={res['pull'][reg]['width_mad']:.2f})")
    rng = np.nanpercentile(np.abs(d[ok]), 99) if ok.any() else 1.0
    ax.hist(np.clip(d[ok] / max(rng, 1e-9), -5, 5), bins=80, histtype="step", density=True,
            label=r"$\Delta\cot\theta$ / q99 (all)")
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
# per surface: IL1-IL4 red/orange/blue/green, ID1-ID4 purple/brown/pink/olive
SURFACE_COLOR = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4", 4: "#2ca02c",
                 11: "#9467bd", 12: "#8c564b", 13: "#e377c2", 14: "#bcbd22"}
SURFACE_COLOR_TEXT = ("IL1 red, IL2 orange, IL3 blue, IL4 green; "
                      "ID1 purple, ID2 brown, ID3 pink, ID4 olive")
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
        for L in SURFACES:
            m = K["surface"][sel] == L
            if not m.any():
                continue
            x0 = np.full(int(m.sum()), xlo); x1 = np.full(int(m.sum()), xhi)
            segs = np.stack([np.stack([x0, intercept[m] + slope[m] * x0], -1),
                             np.stack([x1, intercept[m] + slope[m] * x1], -1)], 1)
            ax.add_collection(LineCollection(segs, colors=SURFACE_COLOR[L], linewidths=0.6,
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
    for L in SURFACES:
        m = K["surface"][sel] == L
        if not m.any():
            continue
        bands = _hough_bands(centre[m], sigma[m], slope[m], intercept[m], xlo, xhi, has[m])
        for b, segs in enumerate(bands):
            if len(segs):
                ax.add_collection(LineCollection(
                    segs, colors=SURFACE_COLOR[L], linewidths=0.6, alpha=BAND_ALPHA[b],
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


def _variant_crossings(files, v):
    """Crossings for ONE activeSP variant. Only the columns the sweep needs."""
    hit = v.hit
    cols = ["trackIdx", "layer", "detId", "projLocalX", "projLocalY", "projSigX", "projSigY",
            "hitAccepted"]
    H = _concat(files, [f"{hit}_{c}" for c in cols])
    ncross = ak.to_numpy(ak.num(H[f"{hit}_layer"]))
    X = {c: ak.to_numpy(ak.flatten(H[f"{hit}_{c}"])) for c in cols}
    ev = np.repeat(np.arange(len(ncross)), ncross)
    X["event"] = ev
    # The matched TP is per TRACK and per VARIANT (each variant refits separately),
    # so it has to come from THIS variant's track table.
    tname = _matched_tp_branch(files, v)
    V = _concat(files, [tname])
    mtp = ak.to_numpy(ak.flatten(V[tname]))
    offm = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(V[tname])))])
    X["trk_tpIdx"] = mtp[offm[ev] + X["trackIdx"].astype(np.int64)]
    return X


def _combinations_per_track(X, K, kvals):
    """For each k, the number of hit combinations (one cluster per crossed surface,
    IL1-IL4 and ID1-ID4) each track must search.

    Counts are SUMMED over crossings within a surface before the product is taken:
    a track that clips two overlapping modules at one surface has a single
    candidate POOL there, so those crossings must add. Multiplying them would
    invent combinations that the refit never considers, since it takes one hit per
    surface.

    A surface with no candidate contributes a factor of 1, not 0. The refit skips
    it and carries on; treating it as 0 would erase the track from the
    distribution entirely and would bias the result toward the busy tracks.
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
    key2 = tkey * 32 + X["layer"].astype(np.int64)      # surface codes 1-4, 11-14
    u2, inv2 = np.unique(key2, return_inverse=True)
    tk2 = u2 // 32
    ut, invt = np.unique(tk2, return_inverse=True)

    # TRUE-hit containment, the benefit side of widening the cone. A pair is the
    # true one when the cluster's dominant TP is the TP the track was matched to.
    # Denominator counts only crossings where such a cluster EXISTS on the module,
    # so a layer that simply had no true cluster does not count as an inefficiency.
    is_true = SPN.tp_matched(K["tpIdx"][ci]) & SPN.tp_matched(X["trk_tpIdx"][xi]) & \
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
        *([f"  {res['partial']}"] if res.get("partial") else []),
        "  cone          : nominal per-axis two-sided Gaussian, k = Phi^-1((1+q)/2)",
        "                  " + " | ".join(
            f"{l} k={res['table_k_sigma'][i]:.3f}" for i, l in enumerate(labs)),
        f"                  {trk_per_ev:.0f} refit-able tracks per event",
        "  combinations  : PER TRACK, the number of hit tuples (one cluster per crossed",
        "                  surface) a refit must test = product of the candidate counts over",
        "                  the INSTRUMENTED surfaces it crosses, IL1-IL4 AND ID1-ID4 (mask",
        "                  character k = ILk + IDk). A surface with NO candidate contributes",
        "                  1, not 0 -- the refit skips it and carries on. Worked: 2,1,0,2 ->",
        "                  2*1*1*2 = 4;  3,2,1,1 -> 6. (tests/test_combination_counting.py)",
        "                  The barrel-only (IL) and disc-only (ID) counts are in the JSON",
        "                  (by_region).",
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
    many hit combinations (one cluster per instrumented surface crossed, IL1-IL4
    and, since the disc refit, ID1-ID4) a refit must actually search, and how its
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
    # THE SELECTED VARIANTS (--variants), keyed by mask. The full sweep is all 15
    # non-trivial builds; a subset of >= 2 still compares like with like and runs,
    # flagged PARTIAL; one build has nothing to compare against and is skipped.
    vmap = {v.mask: v for v in out["_variants"]}
    if len(vmap) != len(out["_variants"]):
        raise SystemExit("(10) two selected variants share an activeSP mask: "
                         f"{[v.label for v in out['_variants']]}")
    sfx = sorted(vmap)
    if len(sfx) < 2:
        msg = (f"{len(sfx)} activeSP variant selected ({', '.join(v.label for v in vmap.values())}); "
               "the sweep compares builds and needs >= 2 (all 15 non-trivial masks for the full "
               "sweep): pass --variants all on a coexist nano")
        print(f"   (10) SKIPPED: {msg}")
        out["combination_sweep"] = {"skipped": msg}
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.02, 0.5, "(10) combination sweep SKIPPED:\n" + textwrap.fill(msg, 60),
                       fontsize=7, va="center")
        return
    missing = [m for m in SWEEP_ORDER if m not in vmap]
    partial = (f"PARTIAL sweep: {len(sfx)} of {len(SWEEP_ORDER)} builds (missing "
               f"{','.join(missing)})") if missing else ""
    if partial:
        print(f"   (10) {partial}")

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
        Xv = _variant_crossings(files, vmap[s])
        combos, ntrk, cont, n_have = _combinations_per_track(Xv, K, allk)
        per_cfg[s] = combos
        # the barrel-only count (the pre-disc definition: IL crossings only) and the
        # discs' alone, at the design-table cones
        region = {}
        for reg in REGIONS:
            rm = _region(Xv["layer"], reg)
            if rm.any():
                cr, nr, contr, _ = _combinations_per_track({k_: v_[rm] for k_, v_ in Xv.items()}, K, ktab)
                region[reg] = {"n_tracks": int(nr), "by_cone": {
                    f"{q:.4f}": {"mean": float(cr[k].mean()), "p99": float(np.percentile(cr[k], 99)),
                                 "true_hit_containment": float(contr[k])} for q, k in zip(tabq, ktab)}}
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
                  "total_work": [float(combos[k].sum()) for k in kvals],
                  "by_region": region}
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
    f2.suptitle("(10) refit search space: combinations over the instrumented surfaces (IL + ID) "
                "vs cone quantile, per activeSP configuration" + (f"\n{partial}" if partial else ""), y=1.002)
    f2.tight_layout()
    p2 = os.path.join(out["_outdir"], "spix_combination_sweep.pdf")
    f2.savefig(p2, bbox_inches="tight")
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
    a.set_title("(10) can this build even refit?" + ("\n" + partial if partial else ""),
                **({"fontsize": 7} if partial else {}))
    a.grid(alpha=.3); a.legend(fontsize=7)
    res["variants"] = {s: vmap[s].hit for s in sfx}
    if partial:
        res["partial"] = partial
    _write_sweep_table(res, sfx, out, out.get("n_events", -1))
    out["combination_sweep"] = res
    out["_monitor"]["study10_combination_sweep"] = {
        s: {f"q{100 * float(c):g}": {"mean": d["mean"], "p99": d["p99"],
                                     "containment": d["true_hit_containment"]}
            for c, d in res[s]["by_cone"].items() if float(c) in SWEEP_TABLE_CONES}
        for s in sfx}



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


def _helix_project_disc(rinv, phi0, tanl, z0, z):
    """The same naive single-shot projection onto the PLANE z (signed): transverse
    arc s = (z - z0) / tanL, chord radius r = 2 sin(rInv s / 2) / rInv, position
    azimuth phi0 - rInv s / 2, direction azimuth phi0 - rInv s (d0 ignored, as
    _helix_project). Returns (phi_pos, z, eta, phi_dir) like _helix_project."""
    tl = np.where(np.abs(tanl) > 1e-9, tanl, 1e-9)
    s = np.maximum((z - z0) / tl, 0.0)
    half = 0.5 * rinv * s
    small = np.abs(rinv) < 1e-9
    r = np.where(small, s, 2.0 * np.sin(half) / np.where(small, 1.0, rinv))
    eta = np.arcsinh(z / np.maximum(r, 1e-9))
    return phi0 - half, z, eta, phi0 - 2.0 * half


def _project_surface(tr, g, S, nominal):
    """_helix_project (layer: radius `nominal`) or _helix_project_disc (disc: the
    plane at |z| = `nominal` on the side each track heads to) for tracks g."""
    if SPN.is_disc_surface(S):
        z = np.where(tr["tanL"][g] > 0, 1.0, -1.0) * nominal
        return _helix_project_disc(tr["rInv"][g], tr["phi"][g], tr["tanL"][g], tr["z0"][g], z)
    return _helix_project(tr["rInv"][g], tr["phi"][g], tr["tanL"][g], tr["z0"][g], nominal)


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


def _sector_counts(tr, cl, tev, cev, layer, nominal, wphi, weta, gspec, want_c,
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
    csel = cl["surface"] == layer
    if not csel.any():
        return None
    ca = np.flatnonzero(csel)
    phi_p, _, eta_p, phi_dir = _project_surface(tr, slice(None), layer, nominal)
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

    DISCS. The same absolute eta-phi grid holds for ID1-ID4: the projection lands
    on the disc's PLANE (the median |z| of its clusters, on the side the track
    heads to) instead of a layer's cylinder, and eta-phi bins are cut the same way
    (eta's sign separates the two sides). The combination count multiplies over
    every instrumented surface the track crosses.
    """
    files = out.get("_inputs")
    if not files:
        return
    tcols = ["rInv", "phi", "tanL", "z0", "pt"]
    T = _concat(files, [f"{REF}_{c}" for c in tcols])
    tpname = _matched_tp_branch(files, out["_variant"])
    MT = _concat(files, [tpname])
    ntr = ak.to_numpy(ak.num(T[f"{REF}_rInv"]))
    tr = {c: ak.to_numpy(ak.flatten(T[f"{REF}_{c}"])) for c in tcols}
    tr["tp"] = ak.to_numpy(ak.flatten(MT[tpname]))
    tev = np.repeat(np.arange(len(ntr)), ntr)
    off = np.concatenate([[0], np.cumsum(ntr)])

    cl = {"surface": K["surface"], "phi": K["globalPhi"],
          "eta": np.arcsinh(K["globalZ"] / np.maximum(K["globalR"], 1e-9)),
          "r": K["globalR"], "z": K["globalZ"], "tpIdx": K["tpIdx"],
          "gphi": K.get("gClPhi"), "gcot": K.get("gClCotTheta"),
          "sphi": K.get("gSigPhi"), "scot": K.get("gSigCotTheta")}
    cev = K["event"]
    have_ang = cl["gphi"] is not None and cl["sphi"] is not None

    xi, ci, _ = join_on_module(X, K)
    is_true = SPN.tp_matched(K["tpIdx"][ci]) & SPN.tp_matched(X["trk_tpIdx"][xi]) & \
              (K["tpIdx"][ci] == X["trk_tpIdx"][xi])
    want_c = np.full(len(X["layer"]), -1, dtype=np.int64)
    want_c[xi[is_true]] = ci[is_true]
    gi = off[X["event"]] + X["trackIdx"].astype(np.int64)

    res = {"target_containment": SECTOR_TARGET, "f_scan": SECTOR_F,
           "grids": [g[0] for g in SECTOR_GRIDS], "layers": {}}
    for L in SURFACES:
        xl = np.flatnonzero((X["layer"] == L) & (want_c >= 0))
        if len(xl) < 200:
            continue
        cj = want_c[xl]
        g = gi[xl]
        disc = SPN.is_disc_surface(L)
        if disc:      # the desired cluster's own plane (signed z), as the cylinder r below
            pp, _, pe, _ = _helix_project_disc(tr["rInv"][g], tr["phi"][g], tr["tanL"][g],
                                               tr["z0"][g], cl["z"][cj])
        else:
            pp, _, pe, _ = _helix_project(tr["rInv"][g], tr["phi"][g], tr["tanL"][g],
                                          tr["z0"][g], cl["r"][cj])
        dphi = np.abs(_wrap(cl["phi"][cj] - pp))
        deta = np.abs(cl["eta"][cj] - pe)
        on = cl["surface"] == L
        res["layers"][L] = {
            "surface": SNAME[L],
            # the nominal surface the grid projection uses: median cluster radius
            # (layer) or median cluster |z| (disc, both sides)
            "nominal": "abs_z_cm" if disc else "r_cm",
            "nominal_cm": float(np.median(np.abs(cl["z"][on]) if disc else cl["r"][on])),
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
    k2 = tk * 32 + X["layer"].astype(np.int64)          # surface codes reach 14
    u2, i2 = np.unique(k2, return_inverse=True)
    ut, it_ = np.unique(u2 // 32, return_inverse=True)
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
                got = _sector_counts(tr, cl, tev, cev, L, d["nominal_cm"], wphi, weta,
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
        f"    {'surf':4} {'nominal':>12} {'n':>8} {'dphi q68':>10} {'q95':>9} {'q99':>9}"
        f" {'deta q95':>10} {'q99':>9}",
        "    (nominal = the layer radius r or the disc plane |z| the grid projection uses)",
    ]
    for L, d in sorted(res["layers"].items()):
        nom = f"{'|z|' if d['nominal'] == 'abs_z_cm' else 'r'} {d['nominal_cm']:.2f}cm"
        lines.append(f"    {d['surface']:4} {nom:>12} {d['n_desired']:>8} "
                     f"{d['dphi_q68_mrad']:>9.2f}m {d['dphi_q95_mrad']:>8.2f}m "
                     f"{d['dphi_q99_mrad']:>8.2f}m {d['deta_q95']:>10.5f} {d['deta_q99']:>9.5f}")
    d1 = res["layers"][L1k]
    lines += [
        "",
        f"  HEAVY TAIL, and it sets the design: at {SNAME[L1k]} dphi q95 is "
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
    that, mid to 3 sigma, faint beyond. Surfaces: IL1 red, IL2 orange, IL3 blue,
    IL4 green; ID1 purple, ID2 brown, ID3 pink, ID4 olive.

    THE r-z PLANE IS THE SAME CONSTRUCTION. z0 = z - r*cot(theta), so a cluster
    is a segment of slope -r there too, and the slope encodes the layer: L1's
    short lever arm makes a nearly flat, tightly-determined z0, L4's a steep and
    loose one. That is the whole reason z0 slicing works, drawn rather than
    asserted.

    ETA COVERAGE IS A RESULT, NOT A PLOTTING PROBLEM. TBPX alone is a barrel:
    measured (barrel-only nanos), the fraction of TPs lighting >=3 layers was
    0.87 at |eta| 0.2-1.0 but 0.21 at 1.4-1.8 and 0.03 above 1.8. The disc
    clusters ID1-ID4 are drawn as well, so the forward cells show what the discs
    add; each page states the pool's fraction reaching >= 3 and >= 4 SURFACES
    (barrel + disc). The disc angles are the emulator's (an effective-incidence
    mapping of a BARREL PixelAV response, no disc-orientation simulation): their
    bands are provisional.

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
    M = _ttc()
    ok = SPN.tp_matched(K["tpIdx"]) & (K["tpGClPhi"] > SENTINEL) & (K["tpGClCotTheta"] > SENTINEL)
    ukey, inv = np.unique(M.tp_key(K["event"][ok], K["tpIdx"][ok]), return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    # surfaces reached per TP (bit i = SURFACES[i])
    sbit = np.zeros(32, np.int64)
    sbit[list(SURFACES)] = 1 << np.arange(len(SURFACES))
    laymask = np.zeros(len(ukey), dtype=np.int64)
    np.bitwise_or.at(laymask, inv, sbit[K["surface"][ok].astype(np.int64)])
    nlay = sum(((laymask >> i) & 1) for i in range(len(SURFACES)))

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
    tp_ev, tp_idx = M.tp_key_parts(ukey)

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
            pool_ge4 = float((nlay[cand] >= 4).mean())
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
                "found": True, "event": int(ev), "tpIdx": int(tp_idx[idx]),
                "tp_pt": float(tp_pt[idx]), "tp_eta": float(tp_eta[idx]),
                "n_surfaces": int(nlay[idx]), "n_clusters_tp": int(cnt[idx]),
                "n_cone": int(cone.sum()), "n_sector": int(sect.sum()),
                "n_candidate_tps": int(cand.sum()),
                "selection": "median-ranked by (n_surfaces, n_clusters) -- TYPICAL, not best",
                # the acceptance statement for this cell, independent of which TP
                # happened to be drawn; surfaces = IL1-IL4 and ID1-ID4
                "pool_frac_ge3_surfaces": pool_ge3, "pool_frac_ge4_surfaces": pool_ge4}
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
                               "(no TP with a cluster on IL1-IL4 / ID1-ID4 in this cell)",
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
                             f"$\\eta$={c['tp_eta']:+.2f}, {c['n_surfaces']} surfaces   |   "
                             f"pool of {c['n_candidate_tps']}: "
                             f"{c['pool_frac_ge3_surfaces']*100:.0f}% reach $\\geq$3 surfaces, "
                             f"{c['pool_frac_ge4_surfaces']*100:.0f}% $\\geq$4\n"
                             f"{SURFACE_COLOR_TEXT}; "
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

    BARREL-ONLY BY DESIGN: the seeders it runs (tracklet_topology_cost IT_CONFIGS:
    the L1L2L3 / L2L3L4 barrel triplets and doublets; OT barrel seeds) implement
    no disc seed (the disc seed menu, mem smartpixels-disc-seeding-design, is
    designed, not implemented), so it is given the barrel clusters only
    (barrel_view: the seeders key on the layer, which is 0 on a disc).
    """
    M = _ttc()
    K = barrel_view(K)

    need = ("globalR", "globalZ", "globalPhi", "gClPhi", "gClCotTheta",
            "gSigPhi", "gSigCotTheta", "sigY", "tpIdx", "tpPt", "tp_d0",
            "clHasAlpha", "clHasBeta", "clLocalCotAlpha", "clLocalCotBeta")
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
         "tpIdx": K["tpIdx"], "tpPt": K["tpPt"], "tp_d0": K["tp_d0"], "event": K["event"],
         # the flags + angles the seeder's direction_usable gate reads
         "hasAlpha": K["clHasAlpha"], "hasBeta": K["clHasBeta"],
         "localCotAlpha": K["clLocalCotAlpha"], "localCotBeta": K["clLocalCotBeta"]}
    M.set_limits(M.triplets_for_budget(out.get("_seed_budget_gb", M.DEFAULT_PAIR_BUDGET_GB)),
                 min(out.get("_seed_rss_gb", 6.0), M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
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
    # OT, from the same files (not under --max-events: the OT loader reads them whole)
    try:
        srcs = out.get("_inputs") or []
        if srcs and MAX_EVENTS is None:
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
    # Build the per-TP d0/pT table from build_tp_index, which selects matched TPs.
    # An earlier revision keyed on np.maximum(TP link, 0), which folded EVERY noise
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
    p = os.path.join(od, "spix_seed_mode_confusion.pdf")
    f2.savefig(p, facecolor="#fcfcfb", bbox_inches="tight")
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


SEED_COMPOSITION_CHUNK_EV = 10   # events per seeding pass in study (13)


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

    BARREL-ONLY BY DESIGN: the seeders it runs (tracklet_topology_cost IT_CONFIGS:
    the L1L2L3 / L2L3L4 barrel triplets and doublets; OT barrel seeds) implement
    no disc seed (the disc seed menu, mem smartpixels-disc-seeding-design, is
    designed, not implemented), so it is given the barrel clusters only
    (barrel_view: the seeders key on the layer, which is 0 on a disc).
    """
    K = barrel_view(K)
    import importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "tracklet_topology_cost.py")
    _sp = _ilu.spec_from_file_location("_ttc2", _p)
    M = _ilu.module_from_spec(_sp); _sp.loader.exec_module(M)
    need = ("globalR", "globalZ", "globalPhi", "gClPhi", "gClCotTheta",
            "gSigPhi", "gSigCotTheta", "sigY", "tpIdx", "tpPt",
            "clHasAlpha", "clHasBeta", "clLocalCotAlpha", "clLocalCotBeta")
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
         "tpIdx": K["tpIdx"], "tpPt": K["tpPt"], "event": K["event"],
         # the flags + angles the seeder's direction_usable gate reads
         "hasAlpha": K["clHasAlpha"], "hasBeta": K["clHasBeta"],
         "localCotAlpha": K["clLocalCotAlpha"], "localCotBeta": K["clLocalCotBeta"]}
    M.set_limits(M.triplets_for_budget(out.get("_seed_budget_gb", M.DEFAULT_PAIR_BUDGET_GB)),
                 min(out.get("_seed_rss_gb", 6.0), M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
    PTMIN = 2.0
    KMINS = (0.0, 0.25, 0.30, 0.35, 0.40, 0.45)
    Q = M.it_prepare(D, None)
    nev = int(D["event"].max()) + 1 if len(D["event"]) else 1
    COST = ("tracklets", "match_cand", "tracks_to_fit")
    # EVENT BATCHES. One pass over 100 PU200 events exceeds the seeder's candidate
    # ceiling (TooWide), so seed SEED_COMPOSITION_CHUNK_EV events at a time, summing
    # costs and uniting recovered TP keys. Candidates never pair across events (the
    # pair keys carry the event), so efficiencies are unchanged; costs move at the
    # 1e-5 level (measured, 10 events, batches of 3 vs one pass) because the
    # seeder's z0 "wildcard" routing depends on the batch's widest sigma(z0).
    chunks = [np.flatnonzero((D["event"] >= e0) & (D["event"] < e0 + SEED_COMPOSITION_CHUNK_EV))
              for e0 in range(0, nev, SEED_COMPOSITION_CHUNK_EV)]

    def seed(la, lb, lc, displaced, kmin=0.0):
        cost, keys = dict.fromkeys(COST, 0.0), [np.empty(0, np.int64)]
        for idx in chunks:
            o = M.it_pair_seed(D, Q, idx, la, lb, lc, PTMIN, True, displaced, 0.0, kap_min=kmin)
            for c in COST:
                cost[c] += float(o.get(c, 0))
            if "_trip" in o:
                keys.append(M.recovered_keys(D, *o["_trip"]))
        return cost, np.unique(np.concatenate(keys))
    res = {}
    for cname, cfg in M.IT_CONFIGS.items():
        (pa, la_), (pb, lb_) = cfg["pairs"]
        tri = cfg["triplet"]
        fk = M.findable_keys(D, tri[0], tri[1], tri[2], PTMIN)
        cost_t, kt = seed(tri[0], tri[1], tri[2], True)
        ct = {c: cost_t[c] / nev for c in COST}
        scan = []
        for kmin in KMINS:
            cost_a, ka = seed(pa[0], pa[1], la_, False, kmin)
            cost_b, kb = seed(pb[0], pb[1], lb_, False, kmin)
            uni = np.unique(np.concatenate([kt, ka, kb]))
            scan.append({"kap_min": kmin,
                         "pt_max": (None if kmin == 0 else 1.0 / kmin),
                         "eff_union": float(np.isin(fk, uni).mean()),
                         "beyond_triplet": int(len(np.setdiff1d(np.union1d(ka, kb), kt))),
                         "cost": {c: (cost_a[c] + cost_b[c]) / nev for c in COST}})
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

    BARREL-ONLY BY DESIGN: the census (tp_findability) enumerates the barrel IT +
    OT barrel seeds only (seed_arity.enumerate_seeds over IL layers); a mask's
    disc IDk is instrumented with ILk but no disc seed exists to use it yet.
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
# entry is (title, function, n_panels, scope); a section is a title plus a one-line
# statement of what the section is for, printed on the figure and in the log.
# SCOPE: VARIANT studies read one refit variant's crossings (X, P, its tables) and
# run once per selected variant; SHARED studies read only the variant-independent
# inputs (clusters, OT tracks, TPs) or, like the sweep (10), all selected variants
# at once, and run once.
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

    BARREL-ONLY BY DESIGN: the seed pairs and target layers are the IT barrel
    layers and the OT barrel (M.load_flat + it_barrel_codes keep IL1-IL4 only); a
    disc seed pair (ID + OT disk) is designed, not implemented.
    """
    M = _ttc()
    srcs = out.get("_inputs") or []
    need = ("globalR", "globalZ", "globalPhi", "tpIdx", "tpPt")
    if MAX_EVENTS is not None:
        for a in ax_row:
            a.axis("off")
        out["combined_it_ot"] = {"skipped": "--max-events set: this study reads its inputs whole"}
        return
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
    U = {"layer": np.r_[M.it_barrel_codes(I["layer"], "omnibus.study_combined_it_ot"),
                        O["layer"][bar] + 10],
         "r": np.r_[I["globalR"], O["r"][bar]], "z": np.r_[I["globalZ"], O["z"][bar]],
         "phi": np.r_[I["globalPhi"], O["phi"][bar]],
         "tpIdx": np.r_[I["tpIdx"], O["tpIdx"][bar]],
         "tpPt": np.r_[I["tpPt"], O["tpPt"][bar]],
         "event": np.r_[I["event"], O["event"][bar]]}
    mi = SPN.tp_matched(I["tpIdx"]) & np.isfinite(I["tp_z0"])
    kv = M.tp_key(I["event"][mi], I["tpIdx"][mi])
    o = np.argsort(kv, kind="stable")
    kv, z0t, tlt = kv[o], I["tp_z0"][mi][o], I["tp_tanL"][mi][o]
    f = np.r_[True, kv[1:] != kv[:-1]]
    KV, Z0T, TLT = kv[f], z0t[f], tlt[f]

    def firstpertp(L):
        m = (U["layer"] == L) & SPN.tp_matched(U["tpIdx"]) & (U["tpPt"] >= PT)
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
        mt = m & SPN.tp_matched(I["tpIdx"]) & (I["tpPt"] >= PT)
        layers[NAME[L]] = (float(np.median(I["globalR"][m])), m.sum() / nev,
                           float(np.ptp(I["globalZ"][m])),
                           zres(I["globalR"][mt], I["globalZ"][mt],
                                I["event"][mt], I["tpIdx"][mt]))
    for L in range(1, 7):
        m = bar & (O["layer"] == L)
        if m.sum() < 100:
            continue
        mt = m & SPN.tp_matched(O["tpIdx"]) & (O["tpPt"] >= PT)
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

    BARREL-ONLY BY DESIGN: those rows come from the barrel seed census (study 15).
    The disc (and barrel) cluster angles against the sigma table the refit uses
    are monitored per surface in R3 (section F).
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
    p2 = os.path.join(out["_outdir"], "spix_cluster_angles_by_layer.pdf")
    f2.savefig(p2, bbox_inches="tight")
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

    BARREL-ONLY BY DESIGN: the site's scores are built on the barrel seed census.
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
    p2 = os.path.join(od, "spix_mva_score_correlation.pdf")
    f2.savefig(p2, bbox_inches="tight")
    plt.close(f2)
    out["mva_score_correlation"] = {"scores": names, "n_sampled": int(take),
                                    "figure": p2, "per_build": res}

# ---- (19) OT-only projection: targets vs background, per track class --------
PROJ_TERMS = ["OT-only projection", "surface projection", "4-sigma ellipse",
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


def _cat_legend(fig, y=0.005):
    """ONE legend for the cluster categories, outside every axes (figure bottom)."""
    import ot_projection as OPJ
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], **PROJ_CAT_STYLE[c]) for c in range(5)]
    fig.legend(h, list(OPJ.CATEGORIES), loc="lower center", bbox_to_anchor=(0.5, y), ncol=5,
               fontsize=9, frameon=False)


def _draw_proj_crosscut(R, Ls, nsig, n_events, n_perfect, title_extra=""):
    """Perfect tracks: dcotAlpha with |dcotBeta| <= cut, and the converse, per surface.

    Same common normalisation as the per-surface pages: every category in a panel
    is divided by the total AFTER the cut, so relative proportions survive.
    Clusters without an angle estimate cannot enter either panel.
    """
    import ot_projection as OPJ
    cut = PROJ_CROSS_CUT
    fig = plt.figure(figsize=(14, 3.0 * len(Ls) + 2.0))
    gs = fig.add_gridspec(len(Ls), 3, width_ratios=[1, 1, 0.8])
    table = {}
    for li, L in enumerate(Ls):
        base = (R["surface"] == L) & (R["cls"] == 0)
        both = base & np.isfinite(R["dcotA"]) & np.isfinite(R["dcotB"])
        table[SNAME[L]] = {}
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
                ax.stairs(h / max(n_all, 1), edges, **PROJ_CAT_STYLE[c])
            table[SNAME[L]][f"{qlab} | |{olab}|<={cut:g}"] = rows
            ax.set_yscale("log")
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(f"{qlab}  [unitless]   (|{olab}| <= {cut:g}; edge bins hold overflow)",
                          fontsize=7)
            if pi == 0:
                ax.set_ylabel(f"{SNAME[L]}\nfraction of all clusters\nin panel / bin", fontsize=8)
            if li == 0:
                ax.set_title(f"{qlab}, with |{olab}| <= {cut:g}", fontsize=9)
    tax = fig.add_subplot(gs[:, -1])
    tax.axis("off")
    tax.text(0.0, 1.0,
             f"PERFECT tracks only ({n_perfect:,} tracks, {n_events:,} events).\n"
             f"OT-only projection, no KF update between surfaces,\n"
             f"{nsig:g}-sigma ellipse as on the per-surface pages.\n\n"
             f"Each panel: one angle residual, the OTHER\n"
             f"restricted to |d| <= {cut:g} (unitless) -- a 1D view of\n"
             f"the 2D (dcotAlpha, dcotBeta) distribution.\n"
             f"Kept fractions and widths per category: the\n"
             f"table page after this one.\n\n"
             f"Normalised to ALL clusters in the panel after\n"
             f"the cut, so category proportions are preserved.\n\n"
             "A-in / A-out = majority-owner TP's cluster inside /\n"
             "  outside the ellipse; B2 = other TP in ellipse.\n"
             "B1 is empty by definition (a perfect track has no\n"
             "  foreign stub); B3 = no TP.\n"
             "On a disc cotBeta is the RADIAL angle.",
             fontsize=7, va="top", ha="left", family="monospace", transform=tax.transAxes)
    fig.suptitle(f"Perfect tracks: each angle residual with the other angle restricted to "
                 f"|d| <= {cut:g}, per surface{title_extra}", fontsize=12)
    _cat_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return {"fig": fig, "table": table}


def _proj_validation(V, lim):
    """The study-19 cross-check against the producer's projOT*, per surface."""
    per, bad = {}, []
    for S in np.unique(V["surface"]):
        m = V["surface"] == S
        e = {"n": int(m.sum()),
             "du_p99_um": float(np.percentile(np.abs(V["du"][m]), 99) * 1e4),
             "dv_p99_um": float(np.percentile(np.abs(V["dv"][m]), 99) * 1e4),
             "dcotA_p99": float(np.percentile(np.abs(V["dcotA"][m]), 99)),
             "dcotB_p99": float(np.percentile(np.abs(V["dcotB"][m]), 99)),
             "sigU_ratio_median": float(np.median(V["rsu"][m])),
             "sigV_ratio_median": float(np.median(V["rsv"][m])),
             "sig_ratio_p1_p99": [float(np.percentile(np.r_[V["rsu"][m], V["rsv"][m]], q)) for q in (1, 99)]}
        per[SNAME[int(S)]] = e
        if (e["du_p99_um"] > lim["du_p99_cm"] * 1e4 or e["dv_p99_um"] > lim["dv_p99_cm"] * 1e4
                or e["dcotA_p99"] > lim["dcotA_p99"] or e["dcotB_p99"] > lim["dcotB_p99"]
                or abs(e["sigU_ratio_median"] - 1) > lim["sig_ratio_median_tol"]
                or abs(e["sigV_ratio_median"] - 1) > lim["sig_ratio_median_tol"]):
            bad.append(SNAME[int(S)])
    return per, bad


def study_ot_projection_targets(X, K, P, ax_row, out):
    """(19) OT-only projection: what the refit's TARGET clusters and the
    BACKGROUND clusters inside a 4-sigma ellipse look like, by track class, on
    every surface (IL1-IL4 and ID1-ID4).

    WHY THIS IS SEPARATE FROM THE REST OF THE OMNIBUS. Every other study here
    reads the refit's per-crossing tables, which keep ONE module per surface and
    record the seed projection only where the seed and the KF-updated state share
    a module. This one needs neither restriction, so it projects the OT track
    itself (ot_projection.py) onto every module its ellipse overlaps, and uses
    the refit tables only as a cross-check of its helix, frame and Jacobian. It
    ignores X/K/P and streams its own inputs (--proj-inputs, default -i).

    DISCS. The helix meets a disc's mean-|z| plane on the side it heads to, then
    each candidate module's face (ot_projection.disc_s / surface_s); a disc
    module's local (u, v) is (r-phi, radius). The cross-check against the
    producer's projOT* holds on the disc crossings as on the barrel ones (per
    surface below).

    WHY PER SURFACE. There is no Kalman update between surfaces here, so IL1 and
    IL4 are two independent extrapolations of the same OT fit with different
    lever arms; pooling them would mix different cone sizes.

    WHY BY TRACK CLASS. The target is the majority-owner TP's cluster. On a
    contaminated OT track the projection itself is pulled, and the cluster of a
    TP that owns a foreign stub (B1) is the trap it sets for itself; separating
    classes says whether that trap is visible in position or angle.
    """
    import ot_projection as OPJ
    from matplotlib.backends.backend_pdf import PdfPages

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
    # THE VALIDATION GATE'S REFIT TABLE. projOT* is the OT track's own projection,
    # identical in every variant, so any refit table of the (possibly separate)
    # projection inputs validates it: a selected variant first, and among those the
    # most instrumented (projOT* is stored on more surfaces there).
    have = _variant_names(_keys(files[0]))
    sel = {v.name: v for v in out["_variants"]}
    cand = sorted(have, key=lambda n: (n not in sel, -(sel[n].mask if n in sel else n).count("A")))
    if not cand:
        return _skip(f"no SmartPixels refit table in {files[0]} to validate against")
    c = cand[0]
    refit = sel[c].hit if c in sel else CANONICAL_HIT if c == CANONICAL else COEXIST_HIT_PREFIX + c
    try:
        OPJ.check_inputs(files[0], with_refit=True, refit=refit)
    except SystemExit as e:
        return _skip(str(e))
    print(f"    validating against {refit}")
    nsig = 4.0
    step = [0]

    def _log(msg):
        step[0] += 1
        if step[0] % 10 == 0:
            print(msg)
    r = OPJ.run(files, geom, nsig=nsig, chunk=out.get("_proj_chunk", 10),
                nev=out.get("_proj_nev"), validate=True, log=_log, refit=refit)
    pulls = OPJ.param_pulls(files, nev=out.get("_proj_nev"))
    R, PR, V = r["rows"], r["proj"], r["validation"]
    classes = OPJ.TRACK_CLASSES
    tbc = r["tracks_by_class"]
    Ls = [S for S in r["surfaces"] if (PR["surface"] == S).any()]

    # ---- the cross-check gate, per surface ------------------------------
    lim = PROJ_VALIDATION_LIMITS
    per_val, bad = _proj_validation(V, lim)
    val = {"n_refit_seed_crossings": V["n_refit"], "n_compared": V["n_compared"],
           "per_surface": per_val}
    if V["n_compared"] < 0.99 * V["n_refit"]:
        bad.append("n_compared")
    for nm, e in per_val.items():
        print(f"    cross-check vs refit projOT* {nm}: {e['n']} crossings, |du| p99 {e['du_p99_um']:.3f} um, "
              f"|dv| p99 {e['dv_p99_um']:.3f} um, |dcotB| p99 {e['dcotB_p99']:.2e}, "
              f"width ratio {e['sigU_ratio_median']:.4f}/{e['sigV_ratio_median']:.4f}")
    if bad:
        raise SystemExit(f"OT-only projection DISAGREES with the producer's seed projection on "
                         f"{bad}: {val}. Do not trust any result of this study.")
    if r["bound_ratio_max"] > 0.9:
        print(f"    WARNING: an overlapping module sat at {r['bound_ratio_max']:.2f} of the "
              f"candidate bound; widen BOUND_* in ot_projection.py")

    # ---- per class x surface tables --------------------------------------
    table = {}
    for ci, cname in enumerate(classes):
        table[cname] = {}
        for L in Ls:
            m = (PR["cls"] == ci) & (PR["surface"] == L)
            cv = m & PR["covered"]
            ae = m & PR["a_exists"]
            e = {"n_surface_projections": int(m.sum()), "n_covered": int(cv.sum()),
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
            rm = (R["cls"] == ci) & (R["surface"] == L)
            e["residuals"] = {OPJ.CATEGORIES[c]: {q: _proj_mad_rms(R[q][rm & (R["cat"] == c)].astype(np.float64))
                                                  for q, _, _ in PROJ_QTY} for c in range(5)}
            table[cname][SNAME[L]] = e

    excl = {k: tbc[k] for k in ("tie", "no-owner", "unjoined", "covariance-absent")}
    pull_txt = "  ".join(f"{p} {pulls[p]['mad']:.2f}" for p in OPJ.HPAR)
    excl_txt = ", ".join(f"{k} {v:,}" for k, v in excl.items())
    cls_txt = "  ".join(f"{c} {tbc[c]:,}" for c in classes)

    # ---- summary row on the omnibus grid ---------------------------------
    ax = ax_row[0]
    xp = np.arange(len(Ls))
    cols_ = plt.cm.tab10(np.arange(len(classes)))
    for ci, cname in enumerate(classes):
        y = [table[cname][SNAME[L]]["containment"] for L in Ls]
        ax.plot(xp, y, "o-", color=cols_[ci], label=f"{cname} ({tbc[cname]:,} trk)")
    surface_axis(ax, Ls)
    ax.set_ylabel("target containment  [fraction of surface projections]")
    ax.set_title("A-in / (A exists): majority-owner cluster\ninside the 4-sigma OT-only ellipse",
                 fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=6)
    ax = ax_row[1]
    for ci, cname in enumerate(classes):
        tot = [sum(table[cname][SNAME[L]][f"mean_nB{b}_per_covered"] for b in (1, 2, 3)) for L in Ls]
        b1 = [table[cname][SNAME[L]]["mean_nB1_per_covered"] for L in Ls]
        ax.plot(xp, tot, "o-", color=cols_[ci], label=cname)
        if cname.startswith("foreign"):
            ax.plot(xp, b1, "s:", color=cols_[ci], ms=4)
    surface_axis(ax, Ls)
    ax.set_yscale("log")
    ax.set_ylabel("background clusters per covered surface projection  [count]")
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
        "A = majority-owner TP's cluster on the surface (A-in / A-out of ellipse)\n"
        "B1 = in ellipse, TP owns a foreign stub of this track\n"
        "B2 = in ellipse, other TP;  B3 = in ellipse, no TP\n"
        "Ellipse: d < 4 of J C J^T, OT fit covariance, no MS, no KF update;\n"
        "  every module it overlaps, residuals in the cluster's module frame\n"
        "  (disc: u = r-phi, v = radius).\n"
        f"OT fit pull widths (MAD; 1 = honest): {pull_txt}\n"
        f"Cross-check vs refit projOT*: {V['n_compared']:,} crossings, per surface\n"
        f"  on page 1 of spix_ot_projection.pdf")
    ax.text(0.0, 1.0, legend_txt, fontsize=5.6, va="top", ha="left", family="monospace",
            transform=ax.transAxes)

    # ---- the PDF: validation + class counts, then per surface ------------
    od = out["_outdir"]
    pdf_path = os.path.join(od, "spix_ot_projection.pdf")
    defs = _glossary_text(width=58, terms=PROJ_TERMS)
    with PdfPages(pdf_path) as pdf:
        OSU.table_pages(
            pdf, "Study 19: cross-check of the OT-only projection against the producer's projOT*",
            ["surface", "crossings", "|du| p99 [um]", "|dv| p99 [um]", "|dcotA| p99", "|dcotB| p99",
             "sigma_u ratio med", "sigma_v ratio med", "ratio p1", "ratio p99"],
            [[nm, f"{e['n']:,}", OSU.fmt(e["du_p99_um"]), OSU.fmt(e["dv_p99_um"]),
              OSU.fmt(e["dcotA_p99"], "{:.2e}"), OSU.fmt(e["dcotB_p99"], "{:.2e}"),
              OSU.fmt(e["sigU_ratio_median"], "{:.4f}"), OSU.fmt(e["sigV_ratio_median"], "{:.4f}"),
              OSU.fmt(e["sig_ratio_p1_p99"][0], "{:.3f}"), OSU.fmt(e["sig_ratio_p1_p99"][1], "{:.3f}")]
             for nm, e in per_val.items()],
            notes=[f"{r['n_events']:,} events; validation refit table {refit}; {V['n_compared']:,} of "
                   f"{V['n_refit']:,} crossings with projOT* compared (same module as the producer).",
                   "Limits: |du|,|dv| p99 < 1 um, |dcotA| p99 < 2e-3, |dcotB| p99 < 5e-3, width-ratio "
                   "medians within 2%. Widths: ours central differences (disc perturbations refined",
                   "past the 1 um Newton stop), the producer's one-sided. Disc u = r-phi, v = radius.",
                   f"Tracks per class: {cls_txt}",
                   f"EXCLUDED (no class): {excl_txt}",
                   f"OT fit pull widths (MAD; 1 = honest): {pull_txt}"])
        crows = []
        for cname in classes:
            for L in Ls:
                e = table[cname][SNAME[L]]
                crows.append([cname, SNAME[L], f"{e['n_surface_projections']:,}", f"{e['n_target_exists']:,}",
                              OSU.fmt(e["containment"]),
                              OSU.fmt(sum(e[f"mean_nB{b}_per_covered"] for b in (1, 2, 3)), "{:.2f}"),
                              OSU.fmt(e["mean_nB1_per_covered"], "{:.2f}"), OSU.fmt(e["mean_nB2_per_covered"], "{:.2f}"),
                              OSU.fmt(e["mean_nB3_per_covered"], "{:.2f}"),
                              OSU.fmt(e["median_su"] * 1e4, "{:.0f}"), OSU.fmt(e["median_sv"] * 1e4, "{:.0f}")])
        OSU.table_pages(pdf, "Study 19: containment and background per class and surface",
                        ["class", "surface", "projections", "A exists", "containment", "B per covered",
                         "B1", "B2", "B3", "median sigma_u [um]", "median sigma_v [um]"], crows,
                        notes=["containment = A-in / A exists (majority-owner cluster inside the 4-sigma "
                               "ellipse); B per covered projection (ellipse overlaps a module).",
                               "sigma_u/v: projected OT-fit width on the module(s) the helix lands on "
                               "(disc v = radius)."])
        for L in Ls:
            fig = plt.figure(figsize=(26, 3.0 * len(classes) + 2.0))
            gs = fig.add_gridspec(len(classes), len(PROJ_QTY) + 1,
                                  width_ratios=[1] * len(PROJ_QTY) + [1.25])
            inl = (R["surface"] == L) & np.isin(R["cat"], (0, 2, 3, 4))
            disc = SPN.is_disc_surface(L)
            for qi, (q, qlab, sc) in enumerate(PROJ_QTY):
                v = R[q][inl].astype(np.float64) * sc
                if q == "d":
                    edges = np.linspace(0, 10, 81)
                else:
                    hi = np.nanpercentile(np.abs(v), 99.5) if np.isfinite(v).any() else 1.0
                    edges = np.linspace(-hi, hi, 81)
                if disc:
                    qlab = qlab.replace("du  [um]", "du (r-phi)  [um]").replace("dv  [um]", "dv (radius)  [um]")
                for ci, cname in enumerate(classes):
                    ax = fig.add_subplot(gs[ci, qi])
                    base = (R["surface"] == L) & (R["cls"] == ci)
                    # ONE common normalisation for every category in the panel:
                    # the total of A-in + A-out + B1 + B2 + B3. That keeps the
                    # relative proportions (how much background per target) and
                    # makes A-in and A-out -- two halves of one distribution split
                    # at d = nsig -- join continuously at the ellipse boundary.
                    xall = R[q][base].astype(np.float64)
                    n_all = int(np.isfinite(xall).sum())
                    for c in range(5):
                        x = R[q][base & (R["cat"] == c)].astype(np.float64) * sc
                        x = x[np.isfinite(x)]
                        if not len(x):
                            continue
                        h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), edges)
                        ax.stairs(h / n_all, edges, **PROJ_CAT_STYLE[c])
                    if q == "d":
                        ax.axvline(nsig, color="#555", lw=0.8, ls=":")
                    ax.set_yscale("log")
                    ax.tick_params(labelsize=6)
                    ax.grid(alpha=.25)
                    if ci == len(classes) - 1:
                        ax.set_xlabel(qlab + "   (edge bins hold overflow)", fontsize=7)
                    if qi == 0:
                        e = table[cname][SNAME[L]]
                        ax.set_ylabel(f"{cname}\nfraction of all clusters\nin panel / bin\n"
                                      f"containment {e['containment']:.3f}\n"
                                      f"({e['n_target_inside']:,}/{e['n_target_exists']:,})", fontsize=7)
                    if ci == 0:
                        ax.set_title(qlab, fontsize=8)
            tax = fig.add_subplot(gs[:, -1])
            tax.axis("off")
            tax.text(0.0, 1.0,
                     f"{SNAME[L]}: OT-only projection, no KF update between surfaces\n"
                     f"{r['n_events']:,} events.  Tracks per class: {cls_txt}\n"
                     f"EXCLUDED (no class): {excl_txt}\n"
                     f"OT fit pull widths (MAD): {pull_txt}\n"
                     f"Counts and widths per category: next page.\n\n" + defs,
                     fontsize=5.4, va="top", ha="left", family="monospace",
                     transform=tax.transAxes, wrap=True)
            fig.suptitle(f"SmartPixels OT-only projection onto {SNAME[L]}: target (A) vs background "
                         f"(B1/B2/B3) clusters inside the {nsig:g}-sigma ellipse, by track class",
                         fontsize=12)
            _cat_legend(fig)
            fig.tight_layout(rect=(0, 0.025, 1, 0.975))
            pdf.savefig(fig)
            plt.close(fig)
            rows_t = []
            for cname in classes:
                for c in range(5):
                    rr = table[cname][SNAME[L]]["residuals"][OPJ.CATEGORIES[c]]
                    rows_t.append([cname, OPJ.CATEGORIES[c], f"{rr['du']['n']:,}"]
                                  + [OSU.fmt(rr[q]["mad"] * sc, "{:.1f}" if sc > 1 else "{:.3f}")
                                     for q, _, sc in PROJ_QTY])
            OSU.table_pages(pdf, f"Study 19, {SNAME[L]}: clusters and robust widths (MAD x 1.4826) "
                                 f"per class and category",
                            ["class", "category", "n clusters", "du [um]", "dv [um]", "d [sigma]",
                             "dcotA", "dcotB"], rows_t,
                            notes=["Widths of the histograms on the previous page; du, dv in the cluster's "
                                   "module frame" + (" (disc: r-phi, radius)." if disc else ".")])
        # ---- perfect tracks, one angle cut on the other angle -------------
        # A 1D look at the 2D (dcotAlpha, dcotBeta) distribution: each angle's
        # residual with the OTHER restricted to |d| <= PROJ_CROSS_CUT, per
        # surface, barrel and disc on separate pages.
        crosscut = {}
        for grp, title in (([L for L in Ls if not SPN.is_disc_surface(L)], " (barrel)"),
                           ([L for L in Ls if SPN.is_disc_surface(L)], " (discs)")):
            if not grp:
                continue
            cc = _draw_proj_crosscut(R, grp, nsig, r["n_events"], tbc["perfect"], title)
            pdf.savefig(cc["fig"])
            plt.close(cc["fig"])
            crosscut.update(cc["table"])
            rows_c = []
            for nm, blocks in cc["table"].items():
                for blk, cats in blocks.items():
                    for cat, e in cats.items():
                        rows_c.append([nm, blk, cat, f"{e['n_with_both_angles']:,}", f"{e['n_after_cut']:,}",
                                       OSU.fmt(e["kept_fraction"]), OSU.fmt(e["mad"]), OSU.fmt(e["rms"])])
            OSU.table_pages(pdf, "Study 19: perfect tracks, angle residuals after the other-angle cut"
                            + title, ["surface", "panel", "category", "n (both angles)", "n after cut",
                                      "kept", "MAD width", "RMS"], rows_c)
    for cname in classes:
        print(f"    {cname:13s} " + "  ".join(
            f"{SNAME[L]} cont {table[cname][SNAME[L]]['containment']:.3f} "
            f"B/cov {sum(table[cname][SNAME[L]][f'mean_nB{b}_per_covered'] for b in (1, 2, 3)):.2f}"
            for L in Ls))
    out["ot_projection_targets"] = {
        "n_events": r["n_events"], "nsig": nsig, "surfaces": [SNAME[L] for L in Ls],
        "tracks_by_class": tbc, "excluded": excl,
        "ot_fit_pulls": pulls, "validation_vs_refit": val, "validation_refit_table": refit,
        "candidate_bound_ratio_max": r["bound_ratio_max"], "per_class_surface": table,
        "perfect_crosscut": crosscut, "figures": [pdf_path]}
    mon = out["_monitor"].setdefault("study19_ot_projection", {})
    mon.update({"n_events": r["n_events"],
                "containment_perfect": {SNAME[L]: table["perfect"][SNAME[L]]["containment"] for L in Ls},
                "background_per_covered_perfect": {
                    SNAME[L]: sum(table["perfect"][SNAME[L]][f"mean_nB{b}_per_covered"] for b in (1, 2, 3))
                    for L in Ls},
                "median_sigma_u_um_perfect": {SNAME[L]: table["perfect"][SNAME[L]]["median_su"] * 1e4 for L in Ls},
                "validation": {nm: {k: e[k] for k in ("du_p99_um", "sigU_ratio_median", "sigV_ratio_median")}
                               for nm, e in per_val.items()},
                "ot_fit_pull_mad": {p: pulls[p]["mad"] for p in OPJ.HPAR}})


# ---- (20) refit-update projection: A vs B under the projection a refit uses --
REFIT_VAR_STYLE = {"none": dict(color="#8c8c8c", ls="-", lw=1.4),
                   "innovation": dict(color="#1f77b4", ls="-", lw=1.4),
                   "producer": dict(color="#d62728", ls="-", lw=1.2),
                   "tuned": dict(color="#9467bd", ls="--", lw=1.2),
                   "oracle": dict(color="#111111", ls=":", lw=1.6)}
# The replay must reproduce the producer's running projection, or the update here
# is not the refit's. WHAT CAN AND CANNOT CLOSE: the producer's crossings are
# single-precision (GlobalPoint / LocalPoint are float), and its Jacobian is a
# one-sided difference over steps of 1e-6 (rInv) .. 1e-3 cm (z0, d0), so every
# column carries ~1% float noise (measured: emulating float32 crossing points in
# this replay moves its own widths by +-0.7-0.9% p1/p99 at the first visited
# surface, the same spread as against the producer). The FIRST visited surface
# (the OT state) closes exactly in position (p99 0.01 um); after each update the
# noise is carried into the state (larger where the innovation is large), so the
# later surfaces close in the MEDIAN (position and width) with a float-noise band.
# Gated: the path (every recorded crossing on the emulated visit sequence, in
# order), the first visits, the per-surface medians and the width band q05-q95,
# and the selection closure (box window + metric re-derive the producer's pick).
# Re-baselined 2026-10-01 (canon_v2 AAAA, alphaBeta, discs, 10 events): width-ratio
# medians 0.9995-1.0005; q05/q95 0.98/1.02 on IL1-IL3, 0.950-0.957/1.049-1.054 on
# ID1-ID3 (one to three disc updates carry the noise); |du|/sigma p50 <= 0.04;
# first visits p99 <= 0.015 um; selection 0.999-1.000; path 0 off / 0 out of order.
REFIT_REPLAY_LIMITS = {"pos_p50_sig": 0.1, "sig_ratio_median_tol": 0.01, "sig_ratio_q05_min": 0.93,
                       "sig_ratio_q95_max": 1.07, "first_visit_pos_p99_cm": 1e-5,
                       "selection_agreement_min": 0.99, "path_bad_frac_max": 1e-3}


def _refit_tables(res, classes):
    """per variant x class x surface: containment, widths, background, selection."""
    tab = {}
    for v, d in res["variants"].items():
        P, S = d["proj"], d["sel"]
        tab[v] = {}
        for ci, cname in enumerate(classes):
            tab[v][cname] = {}
            for L in res["surfaces"]:
                m = (P["cls"] == ci) & (P["surface"] == L)
                ae, cv, ld = m & P["a_exists"], m & P["covered"], m & P["lands"]
                ms = (S["cls"] == ci) & (S["surface"] == L)
                sa = ms & S["a_in"]
                sn = ms & ~S["a_in"]
                e = {"n_surface_projections": int(m.sum()),
                     "containment": float((ae & P["a_in"]).sum() / ae.sum()) if ae.any() else float("nan"),
                     "median_su_um": float(np.median(P["su"][ld]) * 1e4) if ld.any() else float("nan"),
                     "median_sv_um": float(np.median(P["sv"][ld]) * 1e4) if ld.any() else float("nan"),
                     "B_per_covered": float((P["nB1"] + P["nB2"] + P["nB3"])[cv].mean()) if cv.any() else float("nan"),
                     "n_A_in_exists": int(sa.sum())}
                if v != "none":
                    for lab, code in (("A-in", 0), ("B1", 2), ("B2", 3), ("B3", 4), ("none", -1)):
                        e[f"chosen_{lab}_given_A_in"] = float((S["chosen"][sa] == code).mean()) if sa.any() else float("nan")
                    e["update_without_target"] = float((S["chosen"][sn] >= 0).mean()) if sn.any() else float("nan")
                tab[v][cname][SNAME[L]] = e
    return tab


def _replay_closure(V, lim):
    """Per-surface closure of the replay against the producer (see REFIT_REPLAY_LIMITS)."""
    rows = V["rows"]
    row = V["row"]
    # rows["first"]: a track's first recorded crossing, compared at the OT state
    per, bad = {}, []
    for S in np.unique(V["surface"]):
        m = V["surface"] == S
        sx, sy = rows["sigX_proj"][row[m]], rows["sigY_proj"][row[m]]
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = np.where(sx > 0, np.abs(V["du"][m]) / sx, np.nan)
            nv = np.where(sy > 0, np.abs(V["dv"][m]) / sy, np.nan)
        nu, nv = nu[np.isfinite(nu)], nv[np.isfinite(nv)]
        fv = rows["first"][row[m]]
        acc = (rows["surface"] == S) & rows["accepted"]
        allr = rows["surface"] == S
        e = {"n": int(m.sum()),
             "du_p50_um": float(np.median(np.abs(V["du"][m])) * 1e4),
             "du_p99_um": float(np.percentile(np.abs(V["du"][m]), 99) * 1e4),
             "dv_p50_um": float(np.median(np.abs(V["dv"][m])) * 1e4),
             "dv_p99_um": float(np.percentile(np.abs(V["dv"][m]), 99) * 1e4),
             "du_over_sigma_p50": float(np.median(nu)), "du_over_sigma_p99": float(np.percentile(nu, 99)),
             "dv_over_sigma_p50": float(np.median(nv)), "dv_over_sigma_p99": float(np.percentile(nv, 99)),
             "sigu_ratio_median": float(np.median(V["rsu"][m])),
             "sigv_ratio_median": float(np.median(V["rsv"][m])),
             "sig_ratio_q05": float(min(np.percentile(V["rsu"][m], 5), np.percentile(V["rsv"][m], 5))),
             "sig_ratio_q95": float(max(np.percentile(V["rsu"][m], 95), np.percentile(V["rsv"][m], 95))),
             "within_2pct": float(np.mean((np.abs(V["rsu"][m] - 1) < 0.02) & (np.abs(V["rsv"][m] - 1) < 0.02))),
             "within_5pct": float(np.mean((np.abs(V["rsu"][m] - 1) < 0.05) & (np.abs(V["rsv"][m] - 1) < 0.05))),
             "n_first_visit": int(fv.sum()),
             "first_visit_pos_p99_um": (float(np.percentile(np.r_[np.abs(V["du"][m][fv]), np.abs(V["dv"][m][fv])], 99) * 1e4)
                                        if fv.any() else float("nan")),
             "selection_agreement": float(rows["same_pick"][acc].mean()) if acc.any() else float("nan"),
             "window_mult_agreement": float((rows["emu_mult"][allr] == rows["windowMult"][allr]).mean()) if allr.any() else float("nan"),
             "truncation_agreement": float((rows["emu_truncated"][allr] == rows["windowTruncated"][allr]).mean()) if allr.any() else float("nan")}
        per[SNAME[int(S)]] = e
        if (e["du_over_sigma_p50"] > lim["pos_p50_sig"] or e["dv_over_sigma_p50"] > lim["pos_p50_sig"]
                or abs(e["sigu_ratio_median"] - 1) > lim["sig_ratio_median_tol"]
                or abs(e["sigv_ratio_median"] - 1) > lim["sig_ratio_median_tol"]
                or e["sig_ratio_q05"] < lim["sig_ratio_q05_min"] or e["sig_ratio_q95"] > lim["sig_ratio_q95_max"]
                or (fv.any() and e["first_visit_pos_p99_um"] > lim["first_visit_pos_p99_cm"] * 1e4)
                or (acc.any() and e["selection_agreement"] < lim["selection_agreement_min"])
                or (allr.any() and e["window_mult_agreement"] < lim["selection_agreement_min"])):
            bad.append(SNAME[int(S)])
    n = len(rows["surface"])
    path = {"n_rows": int(n), "off_path": int(rows["off_path"].sum()), "order_bad": int(rows["order_bad"].sum())}
    if n and (path["off_path"] + path["order_bad"]) > lim["path_bad_frac_max"] * n:
        bad.append("path")
    return {"per_surface": per, "path": path, "n_bad_det": int(V["n_bad_det"])}, bad


def _replay_decomposition(V):
    """On the CORRECT accepted hits (selected cluster of the OT track's TP), per
    surface: the KF innovation pull in u, v from the replayed prediction, and the
    three-way split of each angle's innovation -- the measurement against truth
    over the cluster sigma (the sigma table's share), the PREDICTED track angle
    against truth over the track's own projected sigma (the projection
    covariance's share), and the innovation over their quadrature sum."""
    D, rows = V["decomposition"], V["rows"]
    r = D["row"]
    good = (rows["selHitClass"][r] == 0)
    out = {}
    for S in np.unique(D["surface"]):
        m = good & (D["surface"] == S)
        if m.sum() < 20:
            continue
        rr = r[m]
        e = {"n": int(m.sum())}
        for ax_, meas, sigc, pred, strk in (("u", "recoLocalX", "sigX", "u_pred", "su_trk"),
                                            ("v", "recoLocalY", "sigY", "v_pred", "sv_trk")):
            inn = (rows[meas][rr] - D[pred][m]) / np.sqrt(D[strk][m] ** 2 + rows[sigc][rr] ** 2)
            c, fo = OSU.core_rms(inn)
            e[f"innovation_pull_{ax_}"] = {"core_rms": c, "frac_out": fo, "full_rms": OSU.full_rms(inn),
                                           **OSU.coverage(inn)}
        for ang, meas, sig, truth, pred, strk, has in (
                ("alpha", "recoCotAlpha", "sigAlpha", "tpLocalCotAlpha", "a_pred", "sa_trk", "hasAlpha"),
                ("beta", "recoCotBeta", "sigBeta", "tpLocalCotBeta", "b_pred", "sb_trk", "hasBeta")):
            ok = rows[has][rr] & (rows[truth][rr] > SENTINEL) & (rows[sig][rr] > 0)
            if ok.sum() < 20:
                continue
            mm, sg, t = rows[meas][rr][ok], rows[sig][rr][ok], rows[truth][rr][ok]
            h0, st = D[pred][m][ok], D[strk][m][ok]
            parts = {"measurement_vs_truth": (mm - t) / sg,
                     "track_prediction_vs_truth": (h0 - t) / np.maximum(st, 1e-12),
                     "innovation": (mm - h0) / np.sqrt(st ** 2 + sg ** 2)}
            e[ang] = {k: {"core_rms": OSU.core_rms(x)[0], "full_rms": OSU.full_rms(x),
                          "width68": OSU.width68(x)} for k, x in parts.items()}
            e[ang]["median_sigma_meas"] = float(np.median(sg))
            e[ang]["median_sigma_track"] = float(np.median(st))
        out[SNAME[int(S)]] = e
    return out


def _refit_explain_page(res, rep, mask, nsig, cfg):
    fig = plt.figure(figsize=(11, 14))
    fig.text(0.04, 0.975, f"Refit-update projection ({mask}): what the A / B populations look like\n"
             f"under the projection a refit actually uses", fontsize=13, fontweight="bold", va="top")
    surf_txt = ", ".join(SNAME[s] for s in res["surfaces"])
    txt = (
        "WHAT IS DONE\n"
        f"Every OT track (L1TTrack helix + fit covariance) visits the active surfaces ({surf_txt}) in\n"
        "PATH order along the OT helix, OUTERMOST FIRST, as the producer does (e.g. IL4, IL3, IL2, IL1 for a\n"
        "central track; ID4, ID3, ID2, ID1, IL1 for a forward one; a disc is on the path when the helix\n"
        "meets its plane within its radial extent). At each surface the track is projected onto every\n"
        f"module its {nsig:g}-sigma ellipse overlaps (study 19's code; a disc: the side the track heads to),\n"
        "every cluster is labelled A-in / A-out / B1 / B2 / B3, the variant's metric picks the best\n"
        "in-ellipse cluster, and the track is UPDATED on it before the next surface is projected. The first\n"
        f"visited surface has had no update, so it equals study 19. At most {cfg['maxKFUpdates']} updates.\n\n"
        "UPDATE (mirrors the seedlessRefit producer term for term)\n"
        "Sequential scalar Kalman updates in local x, y, then cotAlpha, cotBeta, linearised about the\n"
        "pre-update state (the projection, and so the Jacobian, is recomputed at every surface); a scalar\n"
        "with r^2/S > 2e6 is skipped; Jacobian columns with |H| > 1e4 are zeroed; no update where the\n"
        f"predicted |cot| > 12. Angles used by the emulated variants: {res['use_angles']} (the producer's:\n"
        f"{cfg['useAngles']}).\n"
        + ("COVARIANCE MODEL: CALIBRATED (" + os.path.basename(res["calibration"]) + "), BARREL SURFACES\n"
           "ONLY (the v3 calibration has no disc material or disc seed groups). Seed = reported covariance +\n"
           "a correlated deficit per (OT-track group, stub count, pT bin); Highland scattering on the tkLayout\n"
           "Phase-2 planes x1; the cluster size/angle R lookup.\n\n"
           if res.get("calibration") else
           "COVARIANCE MODEL: the PRODUCER'S. The cluster's own errors (CPE sigX/sigY, sensor sigAlpha/sigBeta,\n"
           "the emulator's sigma table); at every visited surface after the first update, process noise\n"
           "Q = theta0^2 nCross (J_T J_T^T + J_L J_L^T), theta0 = 0.00075 / p, nCross = traversed surfaces\n"
           "since the last constraint (layers within their |z|, discs within their radius, +-1 cm), kink at\n"
           "the last constraint's radius (the same for a disc); the seed is the reported covariance.\n\n")
        + "THE FOUR RANKING METRICS, and the baseline\n"
        "  none        no update at all: study 19's OT-only projection to every surface.\n"
        "  innovation  d^2 = r^T (J C J^T + R)^-1 r over (du, dv, dcotAlpha, dcotBeta), nothing tuned;\n"
        "              a missing angle is dropped from the form and replaced by its mean, 1.\n"
        "  producer    seedlessRefit's selection: (du/sigX)^2 + (dv/sigY)^2 + (dcotA/sigA)^2 + (dcotB/sigB)^2,\n"
        "              each term over the CLUSTER'S error only, unit weights; a missing angle adds 0.\n"
        "  tuned       the same terms weighted (1, 2, 16384, 16384) (study 6, STALE: pre-Q covariance).\n"
        "  oracle      truth: the majority-owner TP's cluster when inside the ellipse, else no update.\n"
        "All variants choose among the SAME candidates: every cluster inside the ellipse on every\n"
        "overlapping module (the producer: a box on ONE module, the first 8 in readout order).\n\n"
        "VALIDATION: REPLAY AGAINST CMSSW (table on the next page)\n"
        "The update is fed exactly the clusters the producer selected, along its visit sequence (rebuilt\n"
        "from the OT helix; Q added at every visited surface after the first update, as the producer does,\n"
        f"also where it recorded no crossing), with ITS useAngles ({cfg['useAngles']}); it must reproduce the\n"
        "recorded running projection (projLocal*, projSig*, projCot*) on every surface. The first visited\n"
        "surface closes exactly; later ones in the median, within a float-noise band: the producer's\n"
        "crossing points are single precision and its one-sided Jacobian steps are 1e-6..1e-3, so each\n"
        "column carries ~1% noise that later updates carry on (emulating float32 crossings in this replay\n"
        "reproduces the spread). SELECTION closure: the producer's box windows (barrel "
        f"{'/'.join(f'{x:g}' for x in cfg['windowRPhi'])} x\n"
        f"{'/'.join(f'{x:g}' for x in cfg['windowZ'])} cm, discs "
        f"{'/'.join(f'{x:g}' for x in cfg['windowDiscRPhi'])} x {'/'.join(f'{x:g}' for x in cfg['windowDiscR'])} cm), "
        f"the first {cfg['maxHitsPerWindow']} in readout order and its metric re-derive\n"
        "its pick, window multiplicity and truncation from the cluster table.\n\n"
        f"path: {rep['path']['n_rows']:,} recorded crossings, {rep['path']['off_path']} off the emulated path, "
        f"{rep['path']['order_bad']} out of order; cluster/module mismatches {rep['n_bad_det']}.\n\n"
        "EXCLUDED TRACKS are as in study 19 (tie / no-owner / unjoined / covariance absent); track\n"
        "classes, A and B categories are defined on the study-19 pages and in spix_glossary.txt.")
    fig.text(0.04, 0.93, txt, fontsize=7.8, va="top", family="monospace")
    return fig


def _refit_summary_page(tab, res, classes, variants):
    Ls = res["surfaces"]
    qs = [("containment", "target containment  [fraction]", False),
          ("median_su_um", "median projected sigma_u  [um]", True),
          ("B_per_covered", "background clusters per covered projection  [count]", True),
          ("chosen_A-in_given_A_in", "P(chosen = A-in | A-in exists)  [fraction]", False)]
    fig, axs = plt.subplots(len(classes), len(qs), figsize=(19, 3.0 * len(classes) + 1.8), squeeze=False)
    from matplotlib.lines import Line2D
    for ci, cname in enumerate(classes):
        for qi, (key, lab, logy) in enumerate(qs):
            ax = axs[ci][qi]
            for v in variants:
                if key.startswith("chosen") and v == "none":
                    continue
                y = [tab[v][cname][SNAME[L]].get(key, np.nan) for L in Ls]
                ax.plot(range(len(Ls)), y, "o-", ms=3, **REFIT_VAR_STYLE[v])
            ax.set_xticks(range(len(Ls)), [SNAME[L] for L in Ls])
            ax.grid(alpha=.3, which="both")
            ax.tick_params(labelsize=7)
            if logy:
                ax.set_yscale("log")
            if ci == 0:
                ax.set_title(lab, fontsize=9)
            if qi == 0:
                ax.set_ylabel(cname, fontsize=8)
    fig.legend([Line2D([0], [0], **REFIT_VAR_STYLE[v]) for v in variants], variants,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(variants), fontsize=9, frameon=False)
    fig.suptitle("Refit-update projection, per track class and surface: containment, window width, "
                 "background in reach, and how often the metric chose the target", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def _refit_surface_page(res, L, cls_sel, title, variants, nsig):
    fig, axs = plt.subplots(len(variants), len(PROJ_QTY), figsize=(24, 2.8 * len(variants) + 1.8),
                            squeeze=False)
    ref = res["variants"]["none"]["rows"]
    inl = (ref["surface"] == L) & np.isin(ref["cls"], cls_sel) & np.isin(ref["cat"], (0, 2, 3, 4))
    for qi, (q, qlab, sc) in enumerate(PROJ_QTY):
        vv = ref[q][inl].astype(np.float64) * sc
        if q == "d":
            edges = np.linspace(0, 10, 81)
        else:
            hi = np.nanpercentile(np.abs(vv), 99.5) if np.isfinite(vv).any() else 1.0
            edges = np.linspace(-hi, hi, 81)
        for vi, v in enumerate(variants):
            ax = axs[vi][qi]
            R = res["variants"][v]["rows"]
            base = (R["surface"] == L) & np.isin(R["cls"], cls_sel)
            n_all = int(np.isfinite(R[q][base].astype(np.float64)).sum())
            for c in range(5):
                x = R[q][base & (R["cat"] == c)].astype(np.float64) * sc
                x = x[np.isfinite(x)]
                if not len(x):
                    continue
                h, _ = np.histogram(np.clip(x, edges[0], edges[-1]), edges)
                ax.stairs(h / max(n_all, 1), edges, **PROJ_CAT_STYLE[c])
            if q == "d":
                ax.axvline(nsig, color="#555", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=6)
            if vi == 0:
                ax.set_title(qlab, fontsize=8)
            if qi == 0:
                ax.set_ylabel(f"{v}\nfraction of all\nclusters in panel / bin", fontsize=7)
            if vi == len(variants) - 1:
                ax.set_xlabel(qlab + "   (edge bins hold overflow)", fontsize=7)
    fig.suptitle(f"{SNAME[L]}, {title}: A / B residuals against the projection each variant makes here "
                 f"(the first visited surface of a track is identical for all variants)"
                 + ("; disc: du = r-phi, dv = radius" if SPN.is_disc_surface(L) else ""), fontsize=11)
    _cat_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def study_ot_refit_projection(X, K, P, ax_row, out):
    """(20) Refit-update projection: the A vs B picture under the projection a
    refit actually uses -- outside-in along the OT track's path over the active
    surfaces (IL1-IL4 and ID1-ID4), choosing and updating at each -- for four
    ranking metrics plus the no-update baseline. See ot_refit_projection.

    WHY FOUR METRICS. Which cluster a refit updates on decides everything after
    it: a wrong choice pulls the state, and the next surface's window is centred
    on the pulled state. The producer's metric never uses the track's own
    uncertainty; the innovation chi2 does; the tuned weights are an empirical
    approximation of it; the oracle is the ceiling. Running all four on the same
    candidates shows how much of the gap to the ceiling each closes.

    WHY THE REPLAY GATE. The update is a reimplementation; replaying the
    producer's own selections must reproduce its recorded running projection
    (REFIT_REPLAY_LIMITS, per surface, discs included), and its windows + metric
    must re-derive the clusters it picked, or the study measures a different refit.
    The replay uses the producer's useAngles (read from the nano's ParameterSets);
    --refit-use-angles sets the emulated variants' (alphaBeta default, alpha option).
    """
    import ot_refit_projection as RPJ
    import ot_projection as OPJ
    from matplotlib.backends.backend_pdf import PdfPages

    def _skip(msg):
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.02, 0.5, "refit-update projection SKIPPED:\n" + msg, fontsize=7,
                       wrap=True, va="center")
        out["ot_refit_projection"] = {"skipped": msg}
        print(f"    SKIPPED: {msg}")

    files = out.get("_proj_inputs") or out["_inputs"]
    geom = out.get("_geometry")
    if not geom or not os.path.exists(geom):
        return _skip("no module geometry JSON (--geometry)")
    # this page's variant: its mask drives the emulated visit order, its refit
    # table the replay of the producer's own selections
    v = out["_variant"]
    mask = v.mask
    try:
        OPJ.check_inputs(files[0], with_refit=True, refit=v.hit)
    except SystemExit as e:
        return _skip(str(e))
    cfg = dict(RPJ.PRODUCER_DEFAULTS, **out["_refit_config"].get(v.name, {}))
    nsig = 4.0
    step = [0]

    def _log(msg):
        step[0] += 1
        if step[0] % 10 == 0:
            print(msg)
    res = RPJ.run(files, geom, mask=mask, use_angles=out.get("_refit_use_angles", "alphaBeta"), nsig=nsig,
                  chunk=out.get("_proj_chunk", 10), nev=out.get("_proj_nev"), replay=True, log=_log,
                  calibration=out.get("_refit_calibration"), refit=v.hit, producer=cfg)
    variants = list(res["variants"])
    classes = OPJ.TRACK_CLASSES

    # ---- replay gate --------------------------------------------------------
    V = res["replay"]
    rep, bad = _replay_closure(V, REFIT_REPLAY_LIMITS)
    for nm, e in rep["per_surface"].items():
        print(f"    replay {nm}: |du| p50 {e['du_p50_um']:.2f} p99 {e['du_p99_um']:.1f} um ({e['du_over_sigma_p50']:.3f}/"
              f"{e['du_over_sigma_p99']:.2f} sigma), width ratio {e['sigu_ratio_median']:.4f}/{e['sigv_ratio_median']:.4f} "
              f"(q05 {e['sig_ratio_q05']:.3f}, q95 {e['sig_ratio_q95']:.3f}; within 2% {100 * e['within_2pct']:.1f}%), "
              f"selection {e['selection_agreement']:.4f}, mult {e['window_mult_agreement']:.4f}")
    print(f"    replay path: {rep['path']}, bad-det {rep['n_bad_det']}")
    if bad or rep["n_bad_det"]:
        raise SystemExit(f"refit replay DISAGREES with the producer on {bad} "
                         f"(bad-det {rep['n_bad_det']}): {rep}. Do not trust study 20.")
    decomp = _replay_decomposition(V)

    tab = _refit_tables(res, classes)
    tbc = res["tracks_by_class"]

    # ---- summary row on the omnibus grid -------------------------------------
    Ls = res["surfaces"]
    for ai, (key, lab, logy) in enumerate((
            ("containment", "target containment, perfect tracks  [fraction]", False),
            ("median_su_um", "median projected sigma_u, perfect  [um]", True),
            ("chosen_A-in_given_A_in", "P(chosen = A-in | A-in exists), perfect", False))):
        ax = ax_row[ai]
        for vv in variants:
            if key.startswith("chosen") and vv == "none":
                continue
            y = [tab[vv]["perfect"][SNAME[L]].get(key, np.nan) for L in Ls]
            ax.plot(range(len(Ls)), y, "o-", label=vv, **REFIT_VAR_STYLE[vv])
        ax.set_xticks(range(len(Ls)), [SNAME[L] for L in Ls])
        ax.set_title(lab + f"\n{mask}", fontsize=9)
        ax.grid(alpha=.3, which="both")
        if logy:
            ax.set_yscale("log")
        if ai == 0:
            ax.legend(fontsize=7)
    if len(ax_row) > 3:
        ax = ax_row[3]
        ax.axis("off")
        ax.text(0.0, 1.0, "VARIANTS (full text: page 1 of spix_ot_refit_projection_" + mask + ".pdf)\n"
                + "\n".join(f"  {vv}: {RPJ.VARIANT_DEF[vv]}" for vv in variants)
                + f"\nUpdate: sequential x, y, cotA, cotB (angles: {res['use_angles']}),"
                  "\n  producer Q (nTraversed rule), candidates = the 4-sigma ellipse, all modules."
                  f"\nReplay vs CMSSW (producer's selections, useAngles {cfg['useAngles']}):\n"
                + "\n".join(f"  {L}: |du|/sig p50 {e['du_over_sigma_p50']:.3f}, width {e['sigu_ratio_median']:.3f}, "
                            f"sel {e['selection_agreement']:.3f}"
                            for L, e in rep["per_surface"].items()),
                fontsize=5.6, va="top", family="monospace", transform=ax.transAxes)

    # ---- the PDF ---------------------------------------------------------------
    od = out["_outdir"]
    pdf_path = os.path.join(od, f"spix_ot_refit_projection_{mask}.pdf")
    contaminated = [ci for ci, c in enumerate(classes) if c != "perfect"]
    with PdfPages(pdf_path) as pdf:
        fig = _refit_explain_page(res, rep, mask, nsig, cfg)
        pdf.savefig(fig)
        plt.close(fig)
        OSU.table_pages(
            pdf, f"Study 20 ({mask}): replay of the producer's refit, closure per surface",
            ["surface", "crossings", "|du| p50/p99 [um]", "|dv| p50/p99 [um]", "|du|/sig p50/p99",
             "|dv|/sig p50/p99", "width ratio med u/v", "ratio q05/q95", "within 2%/5%", "first visits",
             "first p99 [um]", "selection", "mult / trunc"],
            [[nm, f"{e['n']:,}", f"{e['du_p50_um']:.2f}/{e['du_p99_um']:.1f}", f"{e['dv_p50_um']:.2f}/{e['dv_p99_um']:.1f}",
              f"{e['du_over_sigma_p50']:.3f}/{e['du_over_sigma_p99']:.2f}",
              f"{e['dv_over_sigma_p50']:.3f}/{e['dv_over_sigma_p99']:.2f}",
              f"{e['sigu_ratio_median']:.4f}/{e['sigv_ratio_median']:.4f}",
              f"{e['sig_ratio_q05']:.3f}/{e['sig_ratio_q95']:.3f}",
              f"{100 * e['within_2pct']:.1f}/{100 * e['within_5pct']:.1f}%", f"{e['n_first_visit']:,}",
              OSU.fmt(e["first_visit_pos_p99_um"], "{:.3f}"), OSU.fmt(e["selection_agreement"], "{:.4f}"),
              f"{OSU.fmt(e['window_mult_agreement'], '{:.4f}')} / {OSU.fmt(e['truncation_agreement'], '{:.4f}')}"]
             for nm, e in rep["per_surface"].items()],
            notes=[f"{res['n_events']:,} events; refit table {v.hit}; producer useAngles {cfg['useAngles']}, "
                   f"maxKFUpdates {cfg['maxKFUpdates']}, maxHitsPerWindow {cfg['maxHitsPerWindow']}.",
                   f"path: {rep['path']['n_rows']:,} recorded crossings, {rep['path']['off_path']} off the "
                   f"emulated visit sequence, {rep['path']['order_bad']} out of order; bad-det {rep['n_bad_det']}.",
                   "Gate: |du|/sigma and |dv|/sigma p50 < 0.1 (sigma = the recorded projSigX/Y); width-ratio medians",
                   "within 1%, q05 >= 0.93, q95 <= 1.07; first visits p99 < 0.1 um; selection and window "
                   "multiplicity >= 99%.",
                   "Later surfaces carry the producer's float32 crossing noise (see the explain page)."])
        drows = []
        for nm, e in decomp.items():
            for ang in ("alpha", "beta"):
                if ang not in e:
                    continue
                d = e[ang]
                drows.append([nm, ang, f"{e['n']:,}",
                              OSU.fmt(d["measurement_vs_truth"]["core_rms"]), OSU.fmt(d["measurement_vs_truth"]["width68"]),
                              OSU.fmt(d["track_prediction_vs_truth"]["core_rms"]),
                              OSU.fmt(d["track_prediction_vs_truth"]["width68"]),
                              OSU.fmt(d["innovation"]["core_rms"]), OSU.fmt(d["innovation"]["width68"]),
                              OSU.fmt(d["median_sigma_meas"], "{:.4f}"), OSU.fmt(d["median_sigma_track"], "{:.4f}")])
            drows.append([nm, "u / v", f"{e['n']:,}", "-", "-", "-", "-",
                          f"{OSU.fmt(e['innovation_pull_u']['core_rms'])} / {OSU.fmt(e['innovation_pull_v']['core_rms'])}",
                          "-", "-", "-"])
        OSU.table_pages(
            pdf, f"Study 20 ({mask}): which part of the KF pull is short -- the cluster sigma or the track's",
            ["surface", "angle", "correct hits", "meas-truth / sigma_table core", "w68",
             "pred-truth / sigma_track core", "w68", "innovation pull core", "w68",
             "median sigma_table", "median sigma_track"], drows,
            notes=["On the producer's CORRECT accepted hits (cluster of the OT track's TP), with the replayed "
                   "prediction (= the producer's to float precision).",
                   "meas-truth / sigma_table: emitted angle against the TP's true local angle (helix, no "
                   "scattering) over the cluster's sigma = the sigma table's honesty;",
                   "pred-truth / sigma_track: the track's predicted angle against the same truth over its own "
                   "projected sigma sqrt(J C J^T) = the projection covariance's honesty;",
                   "innovation: (emitted - predicted) / sqrt(sigma_track^2 + sigma_table^2) = the KF pull. "
                   "core = the project's 3-sigma-clipped RMS; w68 = (q84 - q16)/2.",
                   "u / v: position innovation pulls (CPE sigma + track sigma)."])
        fig = _refit_summary_page(tab, res, classes, variants)
        pdf.savefig(fig)
        plt.close(fig)
        for L in Ls:
            for sel_cls, title in (([0], "perfect tracks"),
                                   (contaminated, "contaminated tracks (all non-perfect classes)")):
                fig = _refit_surface_page(res, L, sel_cls, title, variants, nsig)
                pdf.savefig(fig)
                plt.close(fig)
    for vv in variants:
        print(f"    {vv:10s} " + "  ".join(
            f"{SNAME[L]} cont {tab[vv]['perfect'][SNAME[L]]['containment']:.3f} "
            f"su {tab[vv]['perfect'][SNAME[L]]['median_su_um']:.0f}um"
            + (f" pur {tab[vv]['perfect'][SNAME[L]]['chosen_A-in_given_A_in']:.3f}" if vv != "none" else "")
            for L in Ls))
    out["ot_refit_projection"] = {
        "mask": mask, "surfaces": [SNAME[L] for L in Ls], "use_angles": res["use_angles"],
        "producer_config": {k: (list(x) if isinstance(x, tuple) else x) for k, x in cfg.items()},
        "calibration": res.get("calibration"), "n_events": res["n_events"],
        "nsig": nsig, "tracks_by_class": tbc, "replay_vs_cmssw": rep, "pull_decomposition": decomp,
        "gated_scalar_updates": {vv: res["variants"][vv]["gated"] for vv in variants},
        "variant_definitions": RPJ.VARIANT_DEF, "per_variant_class_surface": tab,
        "figures": [pdf_path]}
    mon = out["_monitor"].setdefault("study20_refit_projection", {}).setdefault(v.name, {})
    mon.update({"n_events": res["n_events"],
                "replay": {nm: {k: e[k] for k in ("du_over_sigma_p50", "sigu_ratio_median", "sig_ratio_q05",
                                                    "sig_ratio_q95", "within_2pct", "selection_agreement")}
                           for nm, e in rep["per_surface"].items()},
                "containment_perfect": {vv: {SNAME[L]: tab[vv]["perfect"][SNAME[L]]["containment"] for L in Ls}
                                        for vv in variants},
                "purity_perfect": {vv: {SNAME[L]: tab[vv]["perfect"][SNAME[L]].get("chosen_A-in_given_A_in")
                                        for L in Ls} for vv in variants if vv != "none"},
                "pull_decomposition": decomp})


# ---- (F) the refit monitor: R1-R5 -------------------------------------------
# Studies meant to run CONTINUOUSLY as the SmartPixels cluster estimate (today the
# emulator's angles and sigma table) is replaced by better simulation: each reads
# its inputs whole, is parametrised by the emulator/sigma-table inputs it finds in
# the nano (or on the command line), and writes trend-friendly numbers into the
# run's monitor summary (spix_monitor_summary.json; compare runs with monitor_diff.py).
R_ETA_EDGES = (0.0, 1.1, 1.8, 2.4)                 # |eta| bins of the monitor
R_ETA_LABELS = ("|eta|<1.1", "1.1-1.8", "1.8-2.4")
R_PT_EDGES = (2.0, 3.0, 5.0, 10.0, 20.0, 1e9)      # true pT bins [GeV]
R_PT_LABELS = ("2-3", "3-5", "5-10", "10-20", ">20")
# (key, label, unit, scale): residual reco - truth; pT as (reco - true) / true
R1_PARAMS = (("d0", "d0", "um", 1e4), ("z0", "z0", "um", 1e4), ("pt", "pT", "rel.", 1.0),
             ("phi", "phi0", "mrad", 1e3), ("tanL", "cotTheta", "unitless", 1.0))
# the covariance entry each parameter's pull divides by (pT: q/pT via rInv)
R1_COV = {"d0": "cov_d0_d0", "z0": "cov_z0_z0", "pt": "cov_rInv_rInv", "phi": "cov_phi_phi",
          "tanL": "cov_tanL_tanL"}
R_TRK = ["rInv", "phi", "tanL", "z0", "d0", "pt", "tpIdx", "cov_rInv_rInv", "cov_phi_phi",
         "cov_tanL_tanL", "cov_z0_z0", "cov_d0_d0"]
R_SP_EXTRA = ["l1TrackIdx", "spixRefitPerformed", "spixNAcceptedHits", "spixLayerHitMask"]
R_TP = ["pt", "eta", "phi0", "d0", "z0", "charge"]
R_FLAG_BAND = (0.85, 1.15)                          # monitor flag: a width ratio / pull outside
R_PLATEAU_COT = float(np.tan(np.radians(25.0)))     # |true cot| of the emulator's beta plateau


def _flat_tables(files, spec):
    """{table: {col: flat np array, "_ev": event index}, "_n_events": n} for
    spec {table: [cols]}, read whole (track/TP tables are small)."""
    keys = _keys(files[0])
    br = [b for t, cols in spec.items() for b in SPN.branch_names(keys, t, cols)]
    A = _concat(files, br)
    out = {}
    for t, cols in spec.items():
        J = SPN.table_columns(A, t, cols)
        n = ak.to_numpy(ak.num(J[cols[0]]))
        D = {c: ak.to_numpy(ak.flatten(J[c])) for c in cols}
        D["_ev"] = np.repeat(np.arange(len(n)), n)
        D["_off"] = np.concatenate([[0], np.cumsum(n)])
        out[t] = D
    out["_n_events"] = len(A)
    return out


def _truth_rows(TP, ev, tp_idx):
    """Flat L1TTP row of each object's TP (-1 where tpIdx is not a written row)."""
    ok = tp_idx >= 0
    return np.where(ok, TP["_off"][ev] + np.where(ok, tp_idx, 0), -1)


def _param_residuals(trk, TP, row, kpt):
    """{param: (residual, pull)} of tracks against L1TTP rows `row` (>= 0)."""
    pt_t = TP["pt"][row].astype(np.float64)
    tru = {"d0": TP["d0"][row], "z0": TP["z0"][row], "phi": TP["phi0"][row],
           "tanL": np.sinh(TP["eta"][row].astype(np.float64)),
           "rInv": TP["charge"][row] * kpt / np.maximum(pt_t, 1e-9)}
    out = {}
    for k, _, _, _ in R1_PARAMS:
        if k == "pt":
            res = trk["pt"].astype(np.float64) / pt_t - 1.0
            pull = (trk["rInv"] - tru["rInv"]) / np.sqrt(np.abs(trk["cov_rInv_rInv"]))
        else:
            res = trk[k].astype(np.float64) - tru[k]
            if k == "phi":
                res = np.angle(np.exp(1j * res))
            pull = res / np.sqrt(np.abs(trk[R1_COV[k]]))
        out[k] = (res, pull)
    return out


def _res_stats(res, pull, scale):
    c, fo = OSU.core_rms(res * scale) if len(res) >= 20 else (float("nan"), float("nan"))
    pc, _ = OSU.core_rms(pull) if len(pull) >= 20 else (float("nan"), float("nan"))
    return {"n": int(len(res)), "core_rms": c, "frac_out": fo,
            "full_rms": OSU.full_rms(res * scale) if len(res) else float("nan"),
            "pull_core": pc, "pull_full": OSU.full_rms(pull) if len(pull) else float("nan"),
            **({f"pull_{k}": v for k, v in OSU.coverage(pull).items()} if len(pull) >= 20 else {})}


def _binned_stats(R, eta, pt, sel):
    """{eta label: {pt label | "all": {param: stats}}} for the residuals R over `sel`."""
    out = {}
    for ie, el in enumerate(R_ETA_LABELS):
        me = sel & (np.abs(eta) >= R_ETA_EDGES[ie]) & (np.abs(eta) < R_ETA_EDGES[ie + 1])
        row = {}
        for ip, pl in enumerate(R_PT_LABELS + ("all",)):
            m = me & ((pt >= R_PT_EDGES[ip]) & (pt < R_PT_EDGES[ip + 1]) if pl != "all" else (pt >= R_PT_EDGES[0]))
            row[pl] = {k: _res_stats(R[k][0][m], R[k][1][m][np.isfinite(R[k][1][m])], sc)
                       for k, _, _, sc in R1_PARAMS}
        out[el] = row
    return out


def study_track_parameters(X, K, P, ax_row, out):
    """(R1) Final track-parameter performance: resolution and pulls of d0, z0, pT,
    phi0 and cotTheta, refit (SmartPixels) vs OT, per variant, against true pT
    and |eta| (< 1.1, 1.1-1.8, 1.8-2.4).

    TRUTH LINKS (the project's conventions): an OT track (L1TTrack) against its
    own tpIdx (TTTrackAssociator, loosely genuine); a SmartPixels track against
    BOTH its own tpIdx (the strict majority owner of its clusters, 1/2 per OT stub
    cluster and 1 per accepted IT cluster) AND the OT link (its OT track's
    tpIdx, via l1TrackIdx). Truth parameters are the L1TTP row's (pT, eta ->
    cotTheta, phi0, d0, z0 at the POCA, charge); a TP outside the L1TTP selection
    (tpIdx <= -3, |p| <= 1 GeV) has no row and is not counted. Binning variables
    are the TRUE pT and |eta|. Every SmartPixels track is counted, refitted or
    passed through (the collection downstream sees); the refitted fraction is
    reported per bin.

    RESOLUTION: core RMS (the project convention: iterative 3-sigma clip about the
    mean, RMS about zero of the kept core, so a bias counts; plus the fraction
    outside the core) and the full RMS. pT is relative, (pT - pT_true) / pT_true;
    its PULL is that of q/pT (rInv / cov_rInv_rInv). PULLS use each track's own
    covariance (L1TTrack_cov_*, the OT KF's; the refit's KF posterior).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    files = out["_inputs"]
    variants = out["_variants"]
    spec = {REF: R_TRK, SPN.TP_TABLE: R_TP}
    for v in variants:
        spec[v.track] = R_TRK + R_SP_EXTRA
    D = _flat_tables(files, spec)
    TP, OT = D[SPN.TP_TABLE], D[REF]
    kpt = float(np.median(OT["pt"] * np.abs(OT["rInv"])))
    res = {"definitions": {
        "resolution": "core RMS = RMS about zero of the iterative 3-sigma-clipped core; full RMS; "
                      "d0, z0 [um], pT relative, phi0 [mrad], cotTheta [unitless]",
        "pull": "residual / sqrt(own covariance); pT pull = q/pT pull (rInv)",
        "truth": "OT: L1TTrack_tpIdx; SP: own tpIdx (cluster majority) and the OT link; "
                 "L1TTP rows only (pruned TPs not counted); binned in TRUE pT and |eta|"},
        "n_events": D["_n_events"], "populations": {}}
    # the OT tracks against their own link
    row_ot = _truth_rows(TP, OT["_ev"], OT["tpIdx"])
    pops = {}
    ok = row_ot >= 0
    eta_ot = np.where(ok, TP["eta"][np.maximum(row_ot, 0)], np.nan)
    pt_ot = np.where(ok, TP["pt"][np.maximum(row_ot, 0)], np.nan)
    R_ot = _param_residuals({k: OT[k][ok] for k in R_TRK}, TP, row_ot[ok], kpt)
    pops["OT"] = _binned_stats(R_ot, eta_ot[ok], pt_ot[ok], np.ones(int(ok.sum()), bool))
    res["populations"]["OT"] = {"tracks": int(len(row_ot)), "matched": int(ok.sum()), "bins": pops["OT"]}
    refit_frac = {}
    for v in variants:
        S = D[v.track]
        g = OT["_off"][S["_ev"]] + S["l1TrackIdx"].astype(np.int64)
        for lab, link in (("SP link", S["tpIdx"]), ("OT link", OT["tpIdx"][g])):
            row = _truth_rows(TP, S["_ev"], link)
            ok = row >= 0
            eta = TP["eta"][row[ok]]
            pt = TP["pt"][row[ok]]
            R = _param_residuals({k: S[k][ok] for k in R_TRK}, TP, row[ok], kpt)
            key = f"{v.name} SP ({lab})"
            pops[key] = _binned_stats(R, eta, pt, np.ones(int(ok.sum()), bool))
            res["populations"][key] = {"tracks": int(len(row)), "matched": int(ok.sum()), "bins": pops[key]}
            if lab == "OT link":
                rf = S["spixRefitPerformed"][ok] > 0
                refit_frac[v.name] = {el: float(rf[(np.abs(eta) >= R_ETA_EDGES[i]) & (np.abs(eta) < R_ETA_EDGES[i + 1])
                                                   & (pt >= 2)].mean())
                                      if ((np.abs(eta) >= R_ETA_EDGES[i]) & (np.abs(eta) < R_ETA_EDGES[i + 1])
                                          & (pt >= 2)).any() else None
                                      for i, el in enumerate(R_ETA_LABELS)}
    res["refit_fraction_pt_ge_2"] = refit_frac

    # ---- grid row: core d0 / z0 resolution vs |eta| (pT >= 2), OT vs SP ------
    xs = np.arange(len(R_ETA_LABELS))
    for ax, k, lab in ((ax_row[0], "d0", "d0 core RMS [um]"), (ax_row[1], "z0", "z0 core RMS [um]"),
                       (ax_row[2], "pt", "pT/pT core RMS [rel.]")):
        for i, (pn, b) in enumerate(pops.items()):
            if "SP link" in pn:
                continue
            y = [b[el]["all"][k]["core_rms"] for el in R_ETA_LABELS]
            ax.plot(xs, y, "o-" if pn == "OT" else "s--", color="k" if pn == "OT" else None, ms=4,
                    label=pn.replace(" SP (OT link)", " SP"))
        ax.set_xticks(xs, R_ETA_LABELS); ax.set_xlabel("true |eta|"); ax.set_yscale("log")
        ax.set_ylabel(lab); ax.grid(alpha=.3, which="both")
        ax.set_title(f"(R1) {lab}, true pT >= 2, OT vs refit", fontsize=9)
        ax.legend(fontsize=6, ncol=2)

    # ---- PDF: curves vs true pT per |eta| bin, pulls, tables ----------------
    od = out["_outdir"]
    pdf_path = os.path.join(od, "spix_refit_track_parameters.pdf")
    show = [p for p in pops if "SP link" not in p]
    cmap = plt.get_cmap("tab10" if len(show) <= 11 else "tab20")
    styles = {p: dict(color="k", ls="-", marker="o") if p == "OT" else
              dict(color=cmap((i - 1) % cmap.N), ls="--", marker="s") for i, p in enumerate(show)}
    from matplotlib.lines import Line2D
    with PdfPages(pdf_path) as pdf:
        for what, key, ylab in (("resolution", "core_rms", "core RMS"), ("resolution (full)", "full_rms", "full RMS"),
                                ("pull width", "pull_core", "pull core RMS")):
            fig, axs = plt.subplots(len(R1_PARAMS), len(R_ETA_LABELS), figsize=(14, 3.0 * len(R1_PARAMS) + 1.8),
                                    squeeze=False)
            xp = np.arange(len(R_PT_LABELS))
            for i, (k, nm, unit, _) in enumerate(R1_PARAMS):
                for j, el in enumerate(R_ETA_LABELS):
                    ax = axs[i][j]
                    for pn in show:
                        y = [pops[pn][el][pl][k][key] for pl in R_PT_LABELS]
                        ax.plot(xp, y, ms=3, **styles[pn])
                    ax.set_xticks(xp, R_PT_LABELS, fontsize=7)
                    if key != "pull_core":
                        ax.set_yscale("log")
                    else:
                        ax.axhline(1.0, color="#888", lw=0.8, ls=":")
                    ax.grid(alpha=.3, which="both")
                    if i == 0:
                        ax.set_title(f"true {el}", fontsize=9)
                    if j == 0:
                        ax.set_ylabel(f"{nm} {ylab}" + (f" [{unit}]" if key != "pull_core" else ""), fontsize=8)
                    if i == len(R1_PARAMS) - 1:
                        ax.set_xlabel("true pT [GeV]", fontsize=8)
            fig.legend([Line2D([0], [0], **styles[p]) for p in show], show, loc="lower center",
                       bbox_to_anchor=(0.5, 0.0), ncol=min(5, len(show)), fontsize=7, frameon=False)
            fig.suptitle(f"(R1) track-parameter {what} vs true pT and |eta|: OT tracks (own link) vs "
                         f"SmartPixels tracks (OT link); {D['_n_events']} events", fontsize=11)
            fig.tight_layout(rect=(0, 0.06, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
        rows = []
        for pn, b in pops.items():
            for el in R_ETA_LABELS:
                st = b[el]["all"]
                rows.append([pn, el, f"{st['d0']['n']:,}"]
                            + [f"{OSU.fmt(st[k]['core_rms'], '{:.1f}' if sc > 1 else '{:.4f}')} / "
                               f"{OSU.fmt(st[k]['full_rms'], '{:.1f}' if sc > 1 else '{:.4f}')}"
                               for k, _, _, sc in R1_PARAMS]
                            + [f"{OSU.fmt(st[k]['pull_core'], '{:.2f}')}" for k, _, _, _ in R1_PARAMS])
        OSU.table_pages(pdf, "(R1) resolution core / full RMS and pull core width, true pT >= 2 GeV",
                        ["population", "true |eta|", "N", "d0 [um]", "z0 [um]", "pT/pT", "phi0 [mrad]",
                         "cotTheta", "pull d0", "pull z0", "pull q/pT", "pull phi0", "pull cotTheta"], rows,
                        notes=["core RMS = RMS about zero of the iterative 3-sigma-clipped core (bias "
                               "counts); full = plain RMS. OT: own link; SP: own (cluster-majority) or OT link.",
                               "Refitted fraction of the SP collection, true pT >= 2: "
                               + "; ".join(f"{vn}: " + ", ".join(f"{el} {OSU.fmt(f, '{:.3f}')}" for el, f in fr.items())
                                           for vn, fr in list(refit_frac.items())[:4])
                               + (" ..." if len(refit_frac) > 4 else "")])
        for pn, b in pops.items():
            rows = []
            for el in R_ETA_LABELS:
                for pl in R_PT_LABELS:
                    st = b[el][pl]
                    rows.append([el, pl, f"{st['d0']['n']:,}"]
                                + [OSU.fmt(st[k]["core_rms"], "{:.1f}" if sc > 1 else "{:.4f}") for k, _, _, sc in R1_PARAMS]
                                + [OSU.fmt(st[k]["pull_core"], "{:.2f}") for k, _, _, _ in R1_PARAMS]
                                + [OSU.fmt(st["d0"].get("pull_lt1"), "{:.3f}"), OSU.fmt(st["z0"].get("pull_lt1"), "{:.3f}")])
            OSU.table_pages(pdf, f"(R1) {pn}: core RMS and pull core width per true |eta| x pT bin",
                            ["true |eta|", "true pT", "N", "d0 [um]", "z0 [um]", "pT/pT", "phi0 [mrad]",
                             "cotTheta", "pull d0", "pull z0", "pull q/pT", "pull phi0", "pull cotTh",
                             "d0 |pull|<1", "z0 |pull|<1"], rows,
                            notes=["Gaussian coverage |pull| < 1 = 0.683."])
    res["pdf"] = pdf_path
    out["track_parameters"] = res
    mon = out["_monitor"].setdefault("R1_track_parameters", {})
    for pn, b in pops.items():
        mon[pn] = {el: {k: {"core": b[el]["all"][k]["core_rms"], "full": b[el]["all"][k]["full_rms"],
                            "pull_core": b[el]["all"][k]["pull_core"]} for k, _, _, _ in R1_PARAMS}
                   for el in R_ETA_LABELS}
    mon["refit_fraction_pt_ge_2"] = refit_frac
    print("   (R1) core RMS d0/z0 [um], pT>=2, per |eta| bin:")
    for pn in show:
        print(f"       {pn:28s} " + "  ".join(
            f"{el}: {OSU.fmt(pops[pn][el]['all']['d0']['core_rms'], '{:.0f}')}/"
            f"{OSU.fmt(pops[pn][el]['all']['z0']['core_rms'], '{:.0f}')}" for el in R_ETA_LABELS))


def study_hit_selection(X, K, P, ax_row, out):
    """(R2) Hit selection per SURFACE (IL1-IL4, ID1-ID4), and per the OT track's
    |eta| and pT. Per recorded refit crossing (one per visited surface with a
    valid crossing; NB tracks the refit left without any accepted hit are passed
    through and record NO crossing, so these rates are over the refitted tracks):

      correct exists   a cluster of the OT track's TP (L1TTrack_tpIdx, the TP the
                       producer's selHitClass refers to) on the crossed module
      in window        that cluster inside the producer's box window around the
                       running projection (barrel / disc half-widths from the nano)
      acceptance       P(the accepted hit IS that TP's | it exists on the module)
      wrong-hit rate   P(the accepted hit is another TP's or no TP's | a hit was
                       accepted), OT-matched tracks only (selHitClass 1 or 2)
      empty window     P(no cluster in the window) = windowMult 0
      truncation       P(the window held more than maxHitsPerWindow clusters)

    Folded together: the cone containment of the refit-order cone (study 5,
    q95) and the selection purity of the chi2 metric (study 6 baseline) per
    surface, from this page's earlier studies.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    xi, ci = P["xi"], P["ci"]
    n = len(X["layer"])
    cfg = dict(out["_refit_config"].get(out["_variant"].name, {}))
    import ot_refit_projection as RPJ
    cfg = dict(RPJ.PRODUCER_DEFAULTS, **cfg)
    otm = SPN.tp_matched(X["ot_tpIdx"])
    is_c = otm[xi] & SPN.tp_matched(K["tpIdx"][ci]) & (K["tpIdx"][ci] == X["ot_tpIdx"][xi])
    exists = np.zeros(n, bool)
    exists[xi[is_c]] = True
    wr = np.array([RPJ.window(cfg, int(s))[0] for s in X["layer"]])
    wz = np.array([RPJ.window(cfg, int(s))[1] for s in X["layer"]])
    inw = is_c & (np.abs(K["localX"][ci] - X["projLocalX"][xi]) <= wr[xi]) \
        & (np.abs(K["localY"][ci] - X["projLocalY"][xi]) <= wz[xi])
    in_window = np.zeros(n, bool)
    in_window[xi[inw]] = True
    acc = X["hitAccepted"] > 0
    correct = acc & (X["selHitClass"] == 0)
    wrong = acc & otm & (X["selHitClass"] > 0)
    empty = X["windowMult"] == 0
    trunc = X["windowTruncated"] > 0
    eta, pt = np.abs(X["trk_eta"]), X["trk_pt"]
    s5 = out.get("true_containment", {}).get("q95", {}).get("per_surface_eff", {})
    s6 = out.get("chi2_weight_scan", {}).get("baseline_all_unit_weights", {}).get("per_surface", {})

    def rates(m):
        e = m & exists
        a = m & acc & otm
        return {"n_crossings": int(m.sum()), "n_correct_exists": int(e.sum()),
                "correct_in_window": float(in_window[e].mean()) if e.any() else None,
                "acceptance": float(correct[e].mean()) if e.any() else None,
                "wrong_hit_rate": float(wrong[a].mean()) if a.any() else None,
                "empty_window": float(empty[m].mean()) if m.any() else None,
                "truncation": float(trunc[m].mean()) if m.any() else None,
                "mean_window_mult": float(X["windowMult"][m].mean()) if m.any() else None}
    res = {"definitions": "see the docstring / PDF notes", "per_surface": {}}
    for S in SURFACES:
        ms = X["layer"] == S
        if not ms.any():
            continue
        e = {"all": rates(ms), "eta": {}, "pt": {}}
        for i, el in enumerate(R_ETA_LABELS):
            e["eta"][el] = rates(ms & (eta >= R_ETA_EDGES[i]) & (eta < R_ETA_EDGES[i + 1]))
        for i, pl in enumerate(R_PT_LABELS):
            e["pt"][pl] = rates(ms & (pt >= R_PT_EDGES[i]) & (pt < R_PT_EDGES[i + 1]))
        e["cone_containment_q95_study5"] = s5.get(SNAME[S])
        e["metric_purity_study6"] = (s6.get(SNAME[S]) or {}).get("purity")
        res["per_surface"][SNAME[S]] = e
    present = [S for S in SURFACES if SNAME[S] in res["per_surface"]]
    xp = np.arange(len(present))
    for ax, keys, title in ((ax_row[0], ("acceptance", "correct_in_window"), "correct hit accepted / in window"),
                            (ax_row[1], ("wrong_hit_rate", "empty_window", "truncation"), "wrong hits, empty, truncated")):
        for k in keys:
            ax.plot(xp, [res["per_surface"][SNAME[S]]["all"][k] or np.nan for S in present], "o-", label=k)
        surface_axis(ax, present); ax.set_ylabel("fraction"); ax.grid(alpha=.3)
        ax.set_title(f"(R2) {title}", fontsize=9); ax.legend(fontsize=6)
    ax = ax_row[2]
    for i, el in enumerate(R_ETA_LABELS):
        ax.plot(xp, [res["per_surface"][SNAME[S]]["eta"][el]["acceptance"] or np.nan for S in present],
                "o-", label=f"acceptance {el}")
    surface_axis(ax, present); ax.set_ylabel("fraction"); ax.grid(alpha=.3); ax.legend(fontsize=6)
    ax.set_title("(R2) acceptance per OT-track |eta|", fontsize=9)

    pdf_path = os.path.join(out["_outdir"], "spix_refit_hit_selection.pdf")
    cols = ["n_crossings", "n_correct_exists", "correct_in_window", "acceptance", "wrong_hit_rate",
            "empty_window", "truncation", "mean_window_mult"]
    hdr = ["surface", "bin", "crossings", "correct exists", "in window", "acceptance", "wrong hits",
           "empty", "truncated", "<mult>"]

    def cells(e):
        return [f"{e['n_crossings']:,}", f"{e['n_correct_exists']:,}"] + \
            [OSU.fmt(e[c]) for c in cols[2:7]] + [OSU.fmt(e["mean_window_mult"], "{:.2f}")]
    with PdfPages(pdf_path) as pdf:
        rows = []
        for S in present:
            e = res["per_surface"][SNAME[S]]
            rows.append([SNAME[S], "all"] + cells(e["all"]))
            rows += [[SNAME[S], el] + cells(e["eta"][el]) for el in R_ETA_LABELS]
            rows += [[SNAME[S], f"pT {pl}"] + cells(e["pt"][pl]) for pl in R_PT_LABELS]
        notes = [f"variant {out['_variant'].label}; windows barrel "
                 f"{'/'.join(f'{x:g}' for x in cfg['windowRPhi'])} x {'/'.join(f'{x:g}' for x in cfg['windowZ'])} cm, "
                 f"discs {'/'.join(f'{x:g}' for x in cfg['windowDiscRPhi'])} x "
                 f"{'/'.join(f'{x:g}' for x in cfg['windowDiscR'])} cm (r-phi x z|r), max {cfg['maxHitsPerWindow']} per window.",
                 "Per recorded crossing; tracks with NO accepted hit are passed through and record no crossing.",
                 "correct = a cluster of the OT track's TP (L1TTrack_tpIdx); acceptance = P(accepted hit is it | it exists"
                 " on the module); wrong hits = P(accepted hit of another / no TP | accepted), OT-matched tracks.",
                 "|eta| and pT bins: the OT track's."]
        OSU.table_pages(pdf, f"(R2) hit selection per surface, |eta| and pT ({out['_variant'].label})", hdr, rows, notes)
        OSU.table_pages(pdf, "(R2) folded in: cone containment (study 5, q95 refit-order cone) and metric "
                             "purity (study 6, unit weights)",
                        ["surface", "acceptance", "correct in window", "cone containment q95 (5)", "metric purity (6)"],
                        [[SNAME[S], OSU.fmt(res["per_surface"][SNAME[S]]["all"]["acceptance"]),
                          OSU.fmt(res["per_surface"][SNAME[S]]["all"]["correct_in_window"]),
                          OSU.fmt(res["per_surface"][SNAME[S]]["cone_containment_q95_study5"]),
                          OSU.fmt(res["per_surface"][SNAME[S]]["metric_purity_study6"])] for S in present],
                        notes=["Study 5 truth = the SP track's cluster-majority TP; study 6 purity = the unit-weight "
                               "chi2 picks the true cluster when it is in the seed cone."])
    res["pdf"] = pdf_path
    out["hit_selection"] = res
    mon = out["_monitor"].setdefault("R2_hit_selection", {}).setdefault(out["_variant"].name, {})
    for S in present:
        e = res["per_surface"][SNAME[S]]
        mon[SNAME[S]] = {k: e["all"][k] for k in ("acceptance", "correct_in_window", "wrong_hit_rate",
                                                  "empty_window", "truncation")}
        mon[SNAME[S]]["acceptance_by_eta"] = {el: e["eta"][el]["acceptance"] for el in R_ETA_LABELS}
    print("   (R2) acceptance / wrong / empty / truncated per surface: " + "  ".join(
        f"{SNAME[S]} {OSU.fmt(res['per_surface'][SNAME[S]]['all']['acceptance'])}/"
        f"{OSU.fmt(res['per_surface'][SNAME[S]]['all']['wrong_hit_rate'])}/"
        f"{OSU.fmt(res['per_surface'][SNAME[S]]['all']['empty_window'])}/"
        f"{OSU.fmt(res['per_surface'][SNAME[S]]['all']['truncation'])}" for S in present))


def _sigma_table(out):
    """(SigmaTable or None, provenance dict): the emulator sigma table the nano's
    rec-hit producer names (emuSigmaSet), resolved against --cmssw-src, or the
    --emu-sigma-table override."""
    prov = dict(out["_provenance"].get("rechit", {}))
    path = out.get("_emu_sigma_table") or OSU.resolve_data_path(prov.get("emuSigmaSet"), out.get("_cmssw_search", []))
    if not path:
        prov["table_status"] = "not found (pass --emu-sigma-table or --cmssw-src): compared to the per-cluster stored sigma"
        return None, prov
    prov.update({"table_path": os.path.abspath(path), "table_sha256": OSU.sha256(path)})
    return OSU.SigmaTable(path), prov


def study_angle_monitor(X, K, P, ax_row, out):
    """(R3) Angle-estimate monitor: the SmartPixels angles the refit is given,
    against the truth and against the sigma table it is told they have, per
    SURFACE (IL1-IL4 and ID1-ID4), angle and emulator class.

    CLUSTERS (variant-independent). For every truth-linked cluster with a valid
    true local angle (tpLocalCotAlpha/Beta: the TP's helix at the hit) and a
    usable emitted angle (|cot| <= 12): the residual emitted - true, binned in
    |emitted cot| on the sigma table's OWN bin edges, per class -- T (truth-
    consistent: TP helix within 1 cm; emitted = the SmartPixels response on the
    truth) and C (no trustworthy truth: emitted = CNN + donor route, its "true"
    angle is the dominant TP's helix and can be a delta ray's parent) -- and
    both together. Widths: (q84 - q16)/2 (the TABLE'S OWN DEFINITION), the core
    RMS and the full RMS; table = the sigma the refit uses in that bin (the
    emulator table named by the nano's emuSigmaSet; its sha256 is recorded).
    The table is built on the INCLUSIVE emulated population (both routes run on
    truth-consistent training clusters), which cannot be rebuilt on data: class
    T alone should sit at or below it, class C's residual is dominated by its
    unreliable truth. These ratios are reported, not flagged; a table bin far
    below both neighbours (a sparse-bin artefact) is flagged.

    REFIT CROSSINGS (per variant). On the CORRECT accepted hits (selHitClass 0):
    the producer's KF angle pulls (pullAlpha / pullBeta: innovation over
    sqrt(track + cluster variance); beta is the 4th sequential scalar), and the
    split that says WHICH variance is short -- (emitted - true) / sigma: the
    table's share; (predicted - true): the track prediction's error, whose
    quadrature with sigma_table must match the innovation. The full split with
    the track's own projected sigma is study 20's replay (pull decomposition).
    FLAGS (outside [0.85, 1.15]): "TABLE" when (emitted - true)/sigma on these
    hits is off (the sigma table does not describe the clusters the refit uses);
    "TRACK" when only the KF pull is off (the projection covariance).
    """
    from matplotlib.backends.backend_pdf import PdfPages
    need = ("clLocalCotAlpha", "clLocalCotBeta", "clSigAlpha", "clSigBeta", "clHasAlpha", "clHasBeta")
    if not all(c in K for c in need):
        out["angle_monitor"] = {"skipped": "cluster table lacks the sensor angle columns"}
        for a in ax_row:
            a.axis("off")
        return
    T, prov = _sigma_table(out)
    linked = SPN.tp_matched(K["tpIdx"])
    emu = K.get("tpEmuClass", np.full(len(K["tpIdx"]), -1))
    res = {"sigma_table": prov, "clusters": {}, "kf": {}, "flags": []}
    angles = (("alpha", "clLocalCotAlpha", "tpLocalCotAlpha", "clSigAlpha", "clHasAlpha"),
              ("beta", "clLocalCotBeta", "tpLocalCotBeta", "clSigBeta", "clHasBeta"))
    # does the stored per-cluster sigma follow the table? (provenance check)
    if T is not None:
        chk = {}
        for ang, m_, _, s_, h_ in angles:
            ok = (K[h_] > 0) & np.isin(K["surface"], list(T.tables[ang]))
            tab = T.evaluate(ang, K["surface"][ok], np.abs(K[m_][ok]))
            rel = np.abs(K[s_][ok] / tab - 1)
            chk[ang] = {"n": int(ok.sum()), "frac_rel_diff_gt_1e-3": float(np.mean(rel > 1e-3)) if ok.any() else None}
        res["stored_sigma_vs_table"] = chk
    rows_t = []
    for ang, m_, t_, s_, h_ in angles:
        res["clusters"][ang] = {}
        for S in SURFACES:
            base = (K["surface"] == S) & linked & (K[h_] > 0) & (K[t_] > SENTINEL)
            if base.sum() < 50:
                continue
            if T is not None and S in T.tables[ang]:
                edges, tabv = T.bins(ang, S)
            else:
                edges, tabv = np.array([0, 0.1, 0.3, 0.75, 1.5, 3.0, 12.0]), None
            e_s = {}
            for cl, cm in (("T", emu == 1), ("C", emu == 0), ("T+C", np.ones(len(emu), bool))):
                sel = base & cm
                r = K[m_][sel] - K[t_][sel]
                ac = np.abs(K[m_][sel])
                sg = K[s_][sel]
                bins = []
                for b in range(len(edges) - 1):
                    mb = (ac >= edges[b]) & ((ac < edges[b + 1]) if b < len(edges) - 2 else True)
                    if mb.sum() < 30:
                        continue
                    w68 = OSU.width68(r[mb])
                    ref = float(tabv[b]) if tabv is not None else float(np.median(sg[mb]))
                    bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(mb.sum()),
                                 "width68": w68, "core_rms": OSU.core_rms(r[mb])[0],
                                 "full_rms": OSU.full_rms(r[mb]), "table_sigma": ref,
                                 "ratio": w68 / ref if ref > 0 else float("nan"),
                                 "pull_width68": OSU.width68(r[mb] / sg[mb])})
                    rows_t.append([ang, SNAME[S], cl, f"{edges[b]:g}-{edges[b + 1]:g}", f"{mb.sum():,}",
                                   OSU.fmt(w68, "{:.4f}"), OSU.fmt(OSU.core_rms(r[mb])[0], "{:.4f}"),
                                   OSU.fmt(OSU.full_rms(r[mb]), "{:.3f}"), OSU.fmt(ref, "{:.4f}"),
                                   OSU.fmt(w68 / ref if ref > 0 else None), OSU.fmt(OSU.width68(r[mb] / sg[mb]))])
                n = np.array([b["n"] for b in bins], float)
                ratio = float(np.sum(n * np.array([b["ratio"] for b in bins])) / n.sum()) if len(bins) else float("nan")
                e_s[cl] = {"n": int(sel.sum()), "bins": bins, "ratio_weighted": ratio,
                           "pull_width68": OSU.width68(r / sg), "pull_core": OSU.core_rms(r / sg)[0],
                           "width68": OSU.width68(r), "full_rms": OSU.full_rms(r)}
            res["clusters"][ang][SNAME[S]] = e_s
            # a table bin far below BOTH neighbours is a sparse-bin artefact of the table
            if tabv is not None and len(tabv) >= 3:
                for b in range(1, len(tabv) - 1):
                    if tabv[b] < 0.5 * min(tabv[b - 1], tabv[b + 1]):
                        res["flags"].append(f"table {ang} {SNAME[S]}: bin |cot| {edges[b]:g}-{edges[b + 1]:g} "
                                            f"sigma {tabv[b]:.3f} < half its neighbours "
                                            f"({tabv[b - 1]:.3f}, {tabv[b + 1]:.3f})")
    # ---- the refit crossings, per variant -------------------------------------
    files = out["_inputs"]
    cols = ["layer", "hitAccepted", "selHitClass", "pullAlpha", "pullBeta", "recoCotAlpha", "recoCotBeta",
            "sigAlpha", "sigBeta", "tpLocalCotAlpha", "tpLocalCotBeta", "projCotAlpha", "projCotBeta",
            "hasAlpha", "hasBeta"]
    D = _flat_tables(files, {v.hit: cols for v in out["_variants"]})
    rows_k = []
    for v in out["_variants"]:
        H = D[v.hit]
        good = (H["hitAccepted"] > 0) & (H["selHitClass"] == 0)
        res["kf"][v.name] = {}
        for S in SURFACES:
            ms = good & (H["layer"] == S)
            if ms.sum() < 30:
                continue
            e = {"n": int(ms.sum())}
            for ang, pull, reco, sig, tru, pred, has in (
                    ("alpha", "pullAlpha", "recoCotAlpha", "sigAlpha", "tpLocalCotAlpha", "projCotAlpha", "hasAlpha"),
                    ("beta", "pullBeta", "recoCotBeta", "sigBeta", "tpLocalCotBeta", "projCotBeta", "hasBeta")):
                m = ms & (H[has] > 0) & (H[pull] > SENTINEL) & (H[tru] > SENTINEL)
                if m.sum() < 30:
                    continue
                kp = H[pull][m]
                mt = (H[reco][m] - H[tru][m]) / H[sig][m]
                pe = H[pred][m] - H[tru][m]
                inn = H[reco][m] - H[pred][m]
                # the track-prediction variance the innovation implies, given the table
                implied = OSU.width68(inn) ** 2 - np.median(H[sig][m]) ** 2
                # the SmartPixels model's beta PLATEAU: true |beta| < 25 deg is emitted ~0, so
                # the residual there is ~ -true, whatever |emitted cot| the table is keyed on
                small = np.abs(H[tru][m]) < R_PLATEAU_COT
                e[ang + "_by_true_angle"] = {
                    "lt25deg": {"n": int(small.sum()), "meas_vs_truth_pull_width68": OSU.width68(mt[small])},
                    "ge25deg": {"n": int((~small).sum()), "meas_vs_truth_pull_width68": OSU.width68(mt[~small])}}
                e[ang] = {"n": int(m.sum()), "kf_pull_core": OSU.core_rms(kp)[0], "kf_pull_width68": OSU.width68(kp),
                          "kf_pull_full": OSU.full_rms(kp),
                          "meas_vs_truth_pull_width68": OSU.width68(mt), "meas_vs_truth_pull_core": OSU.core_rms(mt)[0],
                          "pred_minus_truth_width68": OSU.width68(pe), "innovation_width68": OSU.width68(inn),
                          "median_sigma_table": float(np.median(H[sig][m])),
                          "implied_track_sigma": float(np.sqrt(implied)) if implied > 0 else 0.0}
                rows_k.append([v.name, SNAME[S], ang, f"{m.sum():,}", OSU.fmt(e[ang]["kf_pull_core"]),
                               OSU.fmt(e[ang]["kf_pull_width68"]), OSU.fmt(e[ang]["meas_vs_truth_pull_width68"]),
                               OSU.fmt(e[ang]["pred_minus_truth_width68"], "{:.4f}"),
                               OSU.fmt(e[ang]["innovation_width68"], "{:.4f}"),
                               OSU.fmt(e[ang]["median_sigma_table"], "{:.4f}"),
                               f"{OSU.fmt(e[ang + '_by_true_angle']['lt25deg']['meas_vs_truth_pull_width68'])} / "
                               f"{OSU.fmt(e[ang + '_by_true_angle']['ge25deg']['meas_vs_truth_pull_width68'])}"])
                w, mt68 = e[ang]["kf_pull_width68"], e[ang]["meas_vs_truth_pull_width68"]
                # which variance is off: the table's (the measurement against truth over
                # its sigma) or the track's (the KF pull off while the table is right)
                if np.isfinite(mt68) and not (R_FLAG_BAND[0] <= mt68 <= R_FLAG_BAND[1]):
                    res["flags"].append(f"{v.name} {ang} {SNAME[S]}: TABLE sigma {'short' if mt68 > 1 else 'long'}"
                                        f" -- (emitted-true)/sigma width68 {mt68:.2f} on the correct hits "
                                        f"(KF pull {w:.2f})")
                elif np.isfinite(w) and not (R_FLAG_BAND[0] <= w <= R_FLAG_BAND[1]):
                    res["flags"].append(f"{v.name} {ang} {SNAME[S]}: TRACK projection variance "
                                        f"{'short' if w > 1 else 'long'} -- KF pull width68 {w:.2f} while the "
                                        f"table closes ({mt68:.2f})")
            res["kf"][v.name][SNAME[S]] = e
    # ---- grid row -----------------------------------------------------------
    for ax, ang in ((ax_row[0], "alpha"), (ax_row[1], "beta")):
        present = [S for S in SURFACES if SNAME[S] in res["clusters"][ang]]
        xp = np.arange(len(present))
        # class C is left to the tables: its "true" angle is unreliable (ratios 2-11)
        for cl, mk in (("T", "o"), ("T+C", "^")):
            ax.plot(xp, [res["clusters"][ang][SNAME[S]][cl]["ratio_weighted"] for S in present], marker=mk,
                    label=f"cluster w68/table, class {cl}")
        v0 = out["_variants"][0].name
        ax.plot(xp, [res["kf"].get(v0, {}).get(SNAME[S], {}).get(ang, {}).get("kf_pull_width68", np.nan)
                     for S in present], "k*--", label=f"KF pull w68 ({v0})")
        ax.plot(xp, [res["kf"].get(v0, {}).get(SNAME[S], {}).get(ang, {}).get("meas_vs_truth_pull_width68", np.nan)
                     for S in present], "rs:", label=f"(emitted-true)/sigma w68, KF hits ({v0})")
        ax.axhspan(*R_FLAG_BAND, color="#dddddd", alpha=0.5, lw=0)
        surface_axis(ax, present); ax.set_ylabel("ratio / pull width"); ax.grid(alpha=.3)
        ax.set_title(f"(R3) cot{ang} estimate vs its sigma table", fontsize=9); ax.legend(fontsize=6)
    ax = ax_row[2]
    ax.axis("off")
    ax.text(0.0, 1.0, "R3 FLAGS (outside [0.85, 1.15]):\n" + ("\n".join(textwrap.fill(f, 80) for f in res["flags"][:20])
                                                         or "none")
            + f"\n\nsigma table: {prov.get('emuSigmaSet', '?')}\n  {prov.get('table_path', prov.get('table_status', ''))}"
            + (f"\n  sha256 {prov['table_sha256'][:16]}" if "table_sha256" in prov else ""),
            fontsize=5.6, va="top", family="monospace", transform=ax.transAxes)

    pdf_path = os.path.join(out["_outdir"], "spix_refit_angle_monitor.pdf")
    with PdfPages(pdf_path) as pdf:
        # residual-width / table curves per surface
        for ang in ("alpha", "beta"):
            present = [S for S in SURFACES if SNAME[S] in res["clusters"][ang]]
            fig, axs = plt.subplots(2, 4, figsize=(17, 8.5), squeeze=False)
            for i, S in enumerate(present[:8]):
                ax = axs[i // 4][i % 4]
                for cl, st in (("T", "o-"), ("C", "s--"), ("T+C", "^-")):
                    b = res["clusters"][ang][SNAME[S]][cl]["bins"]
                    x = [0.5 * (e["lo"] + min(e["hi"], 6)) for e in b]
                    ax.plot(x, [e["width68"] for e in b], st, ms=3, label=f"w68 {cl}")
                b = res["clusters"][ang][SNAME[S]]["T+C"]["bins"]
                ax.step([e["lo"] for e in b] + [min(b[-1]["hi"], 6)] if b else [],
                        [e["table_sigma"] for e in b] + [b[-1]["table_sigma"]] if b else [], where="post",
                        color="k", lw=1.4, label="table sigma")
                ax.set_xscale("symlog", linthresh=0.1); ax.set_yscale("log"); ax.grid(alpha=.3)
                ax.set_title(SNAME[S], fontsize=9)
                ax.set_xlabel(f"|emitted cot{ang}|", fontsize=8); ax.set_ylabel("(q84-q16)/2 of emitted - true", fontsize=8)
            for j in range(len(present), 8):
                axs[j // 4][j % 4].axis("off")
            from matplotlib.lines import Line2D
            fig.legend([Line2D([0], [0], marker=m, ls=l, color=c) for m, l, c in
                        (("o", "-", "C0"), ("s", "--", "C1"), ("^", "-", "C2"), ("", "-", "k"))],
                       ["class T", "class C", "T + C", "sigma table"], loc="lower center", ncol=4, frameon=False)
            fig.suptitle(f"(R3) emitted - true cot{ang}: width vs the emulator sigma table, per surface "
                         f"(clusters linked to a TP)", fontsize=11)
            fig.tight_layout(rect=(0, 0.04, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
        OSU.table_pages(pdf, "(R3) cluster angle residual widths vs the sigma table, per surface x class x |cot| bin",
                        ["angle", "surface", "class", "|cot| bin", "n", "width68", "core RMS", "full RMS",
                         "table sigma", "w68/table", "pull w68"], rows_t,
                        notes=[f"sigma table: {prov.get('emuSigmaSet', '?')} -> {prov.get('table_path', prov.get('table_status'))}",
                               "width68 = (q84-q16)/2 (the table's definition); class T = truth-consistent, C = CNN+donor "
                               "route (its 'true' angle unreliable); pull = residual / the cluster's stored sigma.",
                               f"stored sigma vs table: {res.get('stored_sigma_vs_table')}"],
                        highlight=lambda r: r[2] == "T+C" and r[9] not in ("-",) and
                        not (R_FLAG_BAND[0] <= float(r[9]) <= R_FLAG_BAND[1]))
        OSU.table_pages(pdf, "(R3) KF angle pulls on the refit's correct accepted hits",
                        ["variant", "surface", "angle", "n", "KF pull core", "KF pull w68", "meas-truth pull w68",
                         "pred-truth w68", "innovation w68", "median sigma table",
                         "meas-truth w68 |true|<25 / >=25 deg"], rows_k,
                        notes=["KF pull = the producer's (innovation / sqrt(track var + sigma^2)); meas-truth pull = "
                               "(emitted - true)/sigma: ~1 => the table is right;",
                               "pred-truth = the track's predicted angle minus the true one (cot units): with the "
                               "table sigma it should add in quadrature to the innovation width.",
                               "Track-side sigma (projection covariance) per crossing: study 20's pull decomposition.",
                               "Last column: the emulator's beta PLATEAU -- a true |angle| < 25 deg is emitted near 0, "
                               "so the residual there is ~ -true whatever the |emitted cot| key of the table says."])
        OSU.table_pages(pdf, "(R3) flags", ["flag"], [[f] for f in res["flags"]] or [["none"]])
    res["pdf"] = pdf_path
    out["angle_monitor"] = res
    mon = out["_monitor"].setdefault("R3_angle_monitor", {})
    mon["sigma_table"] = {k: prov.get(k) for k in ("emuSigmaSet", "table_sha256", "angleSource", "emuInflation",
                                                    "emuDiscAlphaOffset", "table_status")}
    mon["cluster_width68_over_table"] = {ang: {s: {cl: e[cl]["ratio_weighted"] for cl in e}
                                               for s, e in res["clusters"][ang].items()} for ang in ("alpha", "beta")}
    mon["cluster_pull_width68"] = {ang: {s: {cl: e[cl]["pull_width68"] for cl in e}
                                         for s, e in res["clusters"][ang].items()} for ang in ("alpha", "beta")}
    mon["kf_meas_vs_truth_by_true_angle"] = {
        vn: {s: {ang: {k: v["meas_vs_truth_pull_width68"] for k, v in e[ang + "_by_true_angle"].items()}
                 for ang in ("alpha", "beta") if ang + "_by_true_angle" in e}
             for s, e in d.items()} for vn, d in res["kf"].items()}
    mon["kf_meas_vs_truth_pull_width68"] = {
        vn: {s: {ang: e[ang]["meas_vs_truth_pull_width68"] for ang in ("alpha", "beta") if ang in e}
             for s, e in d.items()} for vn, d in res["kf"].items()}
    mon["kf_pull_width68"] = {vn: {s: {ang: e[ang]["kf_pull_width68"] for ang in ("alpha", "beta") if ang in e}
                                   for s, e in d.items()} for vn, d in res["kf"].items()}
    mon["flags"] = res["flags"]
    print(f"   (R3) {len(res['flags'])} flag(s): " + "; ".join(res["flags"][:12]))


def study_covariance_check(X, K, P, ax_row, out):
    """(R4) Refit covariance calibration check: the pulls and coverage of the
    quantities the refit-calibration scripts tune, per SURFACE and overall.

      innovation pulls  the producer's position pulls pullX / pullY on its correct
                        accepted hits: (cluster - running projection) / sqrt(track
                        variance + CPE variance) -- the seed + material + R model
                        in one number per surface (pullX is the first scalar
                        update; pullY follows it), core / full width and coverage
                        of |pull| < 1, 2, 3 against the Gaussian 0.683 / 0.954 /
                        0.997, per surface and true-pT bin of the OT track;
      final state       the refit track parameters' pulls against truth (from R1,
                        SP tracks with the OT link): coverage per parameter and
                        |eta| bin.

    NOT INVOKED: the calibration scripts (refit_calibration.py extraction +
    refit_calibration_fit.py joint fits) are a multi-hour, barrel-only
    pipeline; what a full recalibration with the discs needs is listed on the PDF.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    files = out["_inputs"]
    cols = ["layer", "trackIdx", "hitAccepted", "selHitClass", "pullX", "pullY"]
    spec = {v.hit: cols for v in out["_variants"]}
    spec[REF] = ["pt", "eta"]
    D = _flat_tables(files, spec)
    OT = D[REF]
    res = {"innovation": {}, "final_state": {}}
    rows = []
    for v in out["_variants"]:
        H = D[v.hit]
        g = OT["_off"][H["_ev"]] + H["trackIdx"].astype(np.int64)
        pt = OT["pt"][g]
        good = (H["hitAccepted"] > 0) & (H["selHitClass"] == 0) & (H["pullX"] > SENTINEL) & (H["pullY"] > SENTINEL)
        res["innovation"][v.name] = {}
        for S in SURFACES:
            ms = good & (H["layer"] == S)
            if ms.sum() < 30:
                continue
            e = {}
            for axn, col in (("u", "pullX"), ("v", "pullY")):
                p = H[col][ms]
                c, fo = OSU.core_rms(p)
                e[axn] = {"n": int(ms.sum()), "core": c, "frac_out": fo, "full": OSU.full_rms(p), **OSU.coverage(p),
                          "core_by_pt": {pl: (OSU.core_rms(H[col][ms & (pt >= R_PT_EDGES[i]) & (pt < R_PT_EDGES[i + 1])])[0]
                                              if (ms & (pt >= R_PT_EDGES[i]) & (pt < R_PT_EDGES[i + 1])).sum() >= 30
                                              else None) for i, pl in enumerate(R_PT_LABELS)}}
                rows.append([v.name, SNAME[S], axn, f"{ms.sum():,}", OSU.fmt(c), OSU.fmt(fo), OSU.fmt(e[axn]["full"]),
                             OSU.fmt(e[axn]["lt1"]), OSU.fmt(e[axn]["lt2"]), OSU.fmt(e[axn]["lt3"])]
                            + [OSU.fmt(e[axn]["core_by_pt"][pl]) for pl in R_PT_LABELS])
            res["innovation"][v.name][SNAME[S]] = e
    tp = out.get("track_parameters", {}).get("populations", {})
    frows = []
    for pn, b in tp.items():
        if "OT link" not in pn and pn != "OT":
            continue
        res["final_state"][pn] = {el: {k: {c: b["bins"][el]["all"][k].get(c) for c in ("pull_core", "pull_lt1", "pull_lt2", "pull_lt3")}
                                       for k, _, _, _ in R1_PARAMS} for el in R_ETA_LABELS}
        for el in R_ETA_LABELS:
            frows.append([pn, el] + [f"{OSU.fmt(b['bins'][el]['all'][k]['pull_core'], '{:.2f}')} / "
                                     f"{OSU.fmt(b['bins'][el]['all'][k].get('pull_lt1'), '{:.3f}')}"
                                     for k, _, _, _ in R1_PARAMS])
    v0 = out["_variants"][0].name
    for ax, axn in ((ax_row[0], "u"), (ax_row[1], "v")):
        present = [S for S in SURFACES if SNAME[S] in res["innovation"].get(v0, {})]
        xp = np.arange(len(present))
        for vn, d in list(res["innovation"].items())[:6]:
            ax.plot(xp, [d.get(SNAME[S], {}).get(axn, {}).get("core", np.nan) for S in present], "o-", ms=3, label=vn)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        surface_axis(ax, present); ax.set_ylabel(f"innovation pull {axn} core RMS"); ax.grid(alpha=.3)
        ax.set_title(f"(R4) position innovation pull {axn} (1 = calibrated)", fontsize=9); ax.legend(fontsize=6)
    ax = ax_row[2]
    ax.axis("off")
    ax.text(0.0, 1.0, "(R4) final-state pull core / coverage |pull|<1 (0.683)\n"
            + "\n".join("  ".join(str(c) for c in r) for r in frows[:18]),
            fontsize=5.4, va="top", family="monospace", transform=ax.transAxes)
    pdf_path = os.path.join(out["_outdir"], "spix_refit_covariance_check.pdf")
    with PdfPages(pdf_path) as pdf:
        OSU.table_pages(pdf, "(R4) innovation pulls on the correct accepted hits, per variant and surface",
                        ["variant", "surface", "axis", "n", "core RMS", "frac out", "full RMS", "|p|<1", "|p|<2",
                         "|p|<3"] + [f"core pT {pl}" for pl in R_PT_LABELS], rows,
                        notes=["Gaussian: |p|<1 0.683, |p|<2 0.954, |p|<3 0.997; core = 3-sigma-clipped RMS about 0.",
                               "Disc u = r-phi, v = radius. pT = the OT track's."])
        OSU.table_pages(pdf, "(R4) final-state pulls (core RMS / coverage |pull|<1) against truth",
                        ["population", "true |eta|"] + [f"{nm}" for _, nm, _, _ in R1_PARAMS], frows,
                        notes=["From R1 (true pT >= 2): OT tracks (own link) and SP tracks (OT link)."])
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.04, 0.95, "(R4) what a full refit-covariance recalibration with the discs needs", fontsize=12,
                 fontweight="bold", va="top")
        fig.text(0.04, 0.9, "\n".join([
            "The calibration pipeline (refit_calibration.py extraction + refit_calibration_fit.py joint fits; v3 JSON",
            "consumed by study 20 --refit-calibration) is BARREL-ONLY and was fitted on payload-era barrel refits:",
            "  1. extraction: run ot_projection.evaluate_surface over IL1-IL4 AND ID1-ID4 (the kernel now supports it),",
            "     with the TP truth helix projected onto the disc module planes (disc_s) -- refit_calibration.py loops",
            "     IT_LAYERS only;",
            "  2. seed deficit: re-measure cov(TTTrack - truth at the POCA) per (floor group incl. the OT disk-stub",
            "     groups, stub count, pT bin) on canon_v2 -- the forward OT tracks dominate the disc crossings;",
            "  3. material: disc planes (TFPX module 1.34% X0 at normal incidence, path factor 1/|cos|) and the",
            "     |eta| > 1.2 service shell (1.7-1.9x the barrel gap, tkLayout / D121 scan) in the 'planes' list,",
            "     keyed by z for discs;",
            "  4. R tables: rebuild the CPE / angle error lookup on the emulator clusters, per surface incl. discs",
            "     (disc local y = radius; the emu disc beta has no Gaussian core, see R3);",
            "  5. export a v4 JSON with disc planes and groups; then study 20's calibrated mode can visit discs",
            "     (today it is restricted to IL1-IL4 when a calibration is given).",
            "The monitoring here (innovation pulls per surface and pT, final-state coverage) is what such a",
            "recalibration would be judged on; it runs on every omnibus invocation."]),
                 fontsize=9, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
    res["pdf"] = pdf_path
    out["covariance_check"] = res
    mon = out["_monitor"].setdefault("R4_covariance_check", {})
    mon["innovation_pull_core"] = {vn: {s: {a: e[a]["core"] for a in e} for s, e in d.items()}
                                   for vn, d in res["innovation"].items()}
    mon["innovation_coverage_lt1"] = {vn: {s: {a: e[a]["lt1"] for a in e} for s, e in d.items()}
                                      for vn, d in res["innovation"].items()}
    mon["final_state"] = res["final_state"]


def study_track_mva(X, K, P, ax_row, out):
    """(R5) Track-MVA performance per variant (ngtagger.train.trackmva, the
    canonical/prompt family): the clean head's AUC and the sigma_hat(d0) head's
    68% coverage against |eta|. A frozen bundle is used when --trackmva-bundle
    holds the variant's (trackmva_<family>_<mask>), otherwise the heads are
    trained with event cross-fitting (outer, 5 folds; --train-only files never
    evaluated). OFF by default (--with-trackmva): it trains several HGB models
    per variant. NB a bundle trained on another build (e.g. the barrel-only
    canon_v1 refit) scores the disc-refit tracks out of its training domain."""
    if not out.get("_with_trackmva"):
        msg = "off by default; enable with --with-trackmva (trains/loads HGB track MVAs per variant)"
        out["track_mva"] = {"skipped": msg}
        for a in ax_row:
            a.axis("off")
        ax_row[0].text(0.02, 0.5, "(R5) track MVA SKIPPED:\n" + msg, fontsize=7, va="center")
        return
    from sklearn.metrics import roc_auc_score
    from ngtagger.train import trackmva as TM
    files = out["_inputs"]
    res = {}
    for v in out["_variants"]:
        family = "canonical" if v.name == CANONICAL else "prompt"
        T, E = TM.build_dataset(files, family, v.mask, step_events=25, max_events=MAX_EVENTS, log=lambda m: None)
        bdir = out.get("_trackmva_bundle")
        bpath = os.path.join(bdir, TM.bundle_stem(family, v.mask) + "_meta.json") if bdir else None
        if bpath and os.path.exists(bpath):
            M = TM.load_bundle(bpath)
            sc = M.predict(T, "refit", quantiles=True)
            outer = np.column_stack([sc[c] for c in TM.EVAL_COLUMNS])
            how = f"frozen bundle {bpath}"
            tested = np.ones(len(outer), bool)
        else:
            # the --train-only entries that name one of these inputs (f01/f02 by default)
            tro = [t for t in out.get("_train_only") or [] if any(t in os.path.basename(f) for f in files)]
            to = TM.train_only_mask(files, tro)
            folds = TM.event_folds(E["file"], to, TM.N_FOLDS, 0)[T["evt"]]
            scores = TM.crossfit(T, "refit", folds, "outer", TM.N_FOLDS, log=lambda m: None)
            outer = TM.outer_scores(scores, folds, "outer")
            tested = folds >= 0
            how = "event cross-fitting (outer, 5 folds)"
        matched, clean, rd0, rz0 = TM.truth(T, "refit", "ot")
        eta = np.abs(T["b_eta"])
        e = {"how": how, "per_eta": {}}
        for i, el in enumerate(R_ETA_LABELS):
            m = tested & (eta >= R_ETA_EDGES[i]) & (eta < R_ETA_EDGES[i + 1]) & np.isfinite(outer[:, 0])
            mm = m & matched
            cov = np.mean((rd0[mm] >= outer[mm, 5]) & (rd0[mm] <= outer[mm, 7])) if mm.any() else np.nan
            e["per_eta"][el] = {"n": int(m.sum()),
                                "auc_clean": float(roc_auc_score(clean[m], outer[m, 0])) if len(set(clean[m])) == 2 else None,
                                "sd0_coverage_68": float(cov)}
        res[v.name] = e
    out["track_mva"] = res
    out["_monitor"]["R5_track_mva"] = {vn: e["per_eta"] for vn, e in res.items()}
    ax = ax_row[0]
    xs = np.arange(len(R_ETA_LABELS))
    for vn, e in res.items():
        ax.plot(xs, [e["per_eta"][el]["auc_clean"] or np.nan for el in R_ETA_LABELS], "o-", label=vn)
    ax.set_xticks(xs, R_ETA_LABELS); ax.set_ylabel("AUC P(clean)"); ax.grid(alpha=.3); ax.legend(fontsize=6)
    ax.set_title("(R5) clean-head AUC vs |eta|", fontsize=9)
    ax = ax_row[1]
    for vn, e in res.items():
        ax.plot(xs, [e["per_eta"][el]["sd0_coverage_68"] for el in R_ETA_LABELS], "o-", label=vn)
    ax.axhline(0.683, color="k", ls=":"); ax.set_xticks(xs, R_ETA_LABELS)
    ax.set_ylabel("sigma_hat(d0) 68% coverage"); ax.grid(alpha=.3); ax.legend(fontsize=6)
    ax.set_title("(R5) d0 width-head coverage vs |eta|", fontsize=9)


VARIANT, SHARED = "variant", "shared"

SECTIONS = [
    ("A. Cluster cones: how much is in reach of a track?",
     "Occupancy and containment around a projected track, per surface (IL1-IL4, ID1-ID4)"
     " -- the raw material every later stage draws on, and the cost floor nothing can go below.",
     [("cone occupancy", study_cone_occupancy, 3, VARIANT),
      ("refit-order cone", study_refit_cone_occupancy, 3, VARIANT),
      ("cone containment + size", study_cone_containment, 2, VARIANT)]),

    ("B. What the SmartPixels angles buy",
     "Whether the per-cluster alpha/beta angles and the charge readout actually"
     " discriminate, and what the longitudinal angle delivers as a z0 estimate.",
     [("angle discrimination", study_angle_discrimination, 2, VARIANT),
      ("charge readout gate", study_charge_gate, 2, VARIANT),
      ("unbiased containment", study_true_containment, 2, VARIANT),
      ("z0 resolution (seeding)", study_z0_resolution, 2, SHARED),
      ("cluster angles per layer", study_cluster_angles_by_layer, 2, SHARED),
      ("MVA score correlation per build", study_mva_score_correlation, 2, SHARED)]),

    ("C. Fitting and cluster-combination choices",
     "How hits should be weighted in the refit, and which activeSP combinations"
     " are worth reading out.",
     [("chi2 weight scan", study_chi2_weight_scan, 2, VARIANT),
      ("combination sweep (activeSP)", study_combination_sweep, 3, SHARED)]),

    ("D. Seeding: binning, transforms, and which seeds to build",
     "The seeding design itself -- sector sizing (barrel + discs), the Hough picture (barrel +"
     " discs), and, BARREL-ONLY BY DESIGN (no disc seed is implemented), what each seed mode"
     " recovers, whether the doublets can aim at the triplet's failures, and the seed menus.",
     [("sector bin sizing", study_sector_binning, 2, VARIANT),
      ("hough examples", study_hough_examples, 2, SHARED),
      ("seed-mode confusion", study_seed_mode_confusion, 3, SHARED),
      ("seed composition", study_seed_composition, 3, SHARED),
      ("combined IT+OT seeding", study_combined_it_ot, 3, SHARED),
      ("seed menu per build", study_seed_menu_by_build, 3, SHARED)]),

    ("E. OT-only projection: what a refit is offered, by track class",
     "The OT track's own helix and covariance projected onto every IT module (layers and"
     " discs) its 4-sigma ellipse overlaps, with no KF update, and then updated surface by"
     " surface as the refit does (replayed against CMSSW), per surface.",
     [("OT-only projection targets vs background", study_ot_projection_targets, 3, SHARED),
      ("OT refit-update projection", study_ot_refit_projection, 4, VARIANT)]),

    ("F. Refit monitor: what the refit delivers, tracked run to run",
     "Track-parameter resolution and pulls (R1), hit selection per surface (R2), the angle"
     " estimates against their sigma table (R3), the covariance calibration (R4) and the"
     " track MVAs (R5, optional); key numbers into spix_monitor_summary.json.",
     [("R1 track-parameter resolution and pulls", study_track_parameters, 3, SHARED),
      ("R2 hit selection per surface", study_hit_selection, 3, VARIANT),
      ("R3 angle-estimate monitor", study_angle_monitor, 3, SHARED),
      ("R4 refit covariance check", study_covariance_check, 3, SHARED),
      ("R5 track MVA performance", study_track_mva, 2, SHARED)]),
]

# Flat view, kept because the figure is still one grid and several studies index
# their row directly.
STUDIES = [t for _, _, group in SECTIONS for t in group]


def run_page(sections, X, K, P, out, title):
    """One report: the studies of `sections` on one set of inputs, ONE PDF PAGE PER
    SECTION (vector; legends below the panels), with its JSON sidecar, into
    out["_outdir"]. Each study's outcome (ran / skipped, why / FAILED, why), time
    and memory go into out["_status"]; a failing study (an exception, or a
    validation gate's SystemExit) does not stop the others."""
    import traceback
    from matplotlib.backends.backend_pdf import PdfPages
    od = out["_outdir"]
    os.makedirs(od, exist_ok=True)
    page = os.path.relpath(od, out["_status_root"]) if out.get("_status_root") else od
    pdf_path = os.path.join(od, "spix_combinatorics_omnibus.pdf")
    # qX vs pXX is the confusion most likely to survive into a slide, so it is
    # stamped on every page rather than only in the glossary file.
    footer = ("qX = cone WIDTH chosen (input; k = Phi^-1((1+X/100)/2), nominal, assumes unit pulls)   |   "
              "pXX = percentile over TRACKS (output)   |   containment = MEASURED, conditional on the "
              "hit existing   |   surfaces IL1-IL4, ID1-ID4   |   see spix_glossary.txt")
    with PdfPages(pdf_path) as pdf:
        for stitle, sdesc, group in sections:
            print(f"\n{stitle}\n    {sdesc}")
            ncols = max(2, max(n for _, _, n, _ in group))
            fig, axes = plt.subplots(len(group), ncols, figsize=(5.4 * ncols, 4.9 * len(group) + 1.4),
                                     squeeze=False)
            for row, (name, fn, n, _) in enumerate(group):
                print(f"  study: {name}")
                keys0 = set(out)
                t0 = time.time()
                try:
                    fn(X, K, P, axes[row], out)
                    new = [out[k] for k in set(out) - keys0 if isinstance(out[k], dict)]
                    skip = next((d["skipped"] for d in new if "skipped" in d), None)
                    status = ("skipped: " + str(skip)) if skip else "ran"
                except (Exception, SystemExit) as exc:                 # noqa: BLE001 (reported, run goes on)
                    status = f"FAILED: {type(exc).__name__}: {exc}"
                    print(f"  *** study {name} FAILED:\n" + traceback.format_exc())
                    for a in axes[row]:
                        a.cla(); a.axis("off")
                    axes[row][0].text(0.02, 0.5, textwrap.fill(f"{name} FAILED: {exc}", 90)[:900],
                                      fontsize=7, color="#b1524f", va="center")
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30 if sys.platform == "darwin"
                                                                             else 1 << 20)
                out["_status"].append({"page": page, "study": name, "status": status,
                                       "seconds": round(time.time() - t0, 1), "max_rss_gb_so_far": round(rss, 2)})
                OSU.legends_outside(axes[row][:n])
                for c in range(n, ncols):
                    axes[row][c].axis("off")
            top = 1.0 - 0.9 / fig.get_figheight()          # inches -> figure fraction
            fig.suptitle(f"{title}\n{stitle}", fontsize=13, fontweight="bold", y=0.998, va="top")
            fig.text(0.5, top, textwrap.fill(sdesc, 160), ha="center", va="top", fontsize=9, color="#52514e")
            fig.text(0.5, 0.004, footer, ha="center", va="bottom", fontsize=7)
            fig.tight_layout(rect=(0, 0.02, 1, top - 0.5 / fig.get_figheight()))
            pdf.savefig(fig)
            plt.close(fig)
    js = os.path.join(od, "spix_combinatorics_omnibus.json")
    with open(js, "w") as fh:
        json.dump(OSU.jsonable({k: v for k, v in out.items() if not k.startswith("_")}), fh, indent=2)
    print(f"\nwrote {pdf_path}\nwrote {js}")


def provenance(files, variants, args):
    """(provenance dict for the monitor, {variant name: producer refit config}).

    From every input's ParameterSets: the refit producer behind each variant's
    track table (useAngles, windows, maxKFUpdates, ... and its activeSP mask), the
    rec-hit producer's angle source and emulator tables (emuSigmaSet, ...); the
    inputs (size, mtime); this code's git state; the CMSSW build only via
    --build-hash (the nano does not carry it)."""
    blobs = {f: OSU.pset_blob(f) for f in files}
    refit_config, prod = {}, {}
    for v in variants:
        per = {}
        for f, b in blobs.items():
            lab, params = OSU.producer_params(b, v.track)
            cfgv = OSU.refit_producer_config(params)
            if params:
                cfgv["activeLayers"] = params.get("smartPixelsActiveLayers")
                cfgv["discScope"] = params.get("seedlessRefitDiscScope")
                cfgv["label"] = lab
            per[f] = cfgv
        first = per[files[0]]
        odd = [os.path.basename(f) for f, c in per.items() if c != first]
        if odd:
            print(f"  WARNING: the refit configuration of {v.label} differs between inputs: {odd}")
        refit_config[v.name] = {k: c for k, c in first.items() if k not in ("activeLayers", "discScope", "label")}
        prod[v.name] = {k: (list(c) if isinstance(c, tuple) else c) for k, c in first.items()}
        if first.get("activeLayers") and first["activeLayers"].replace("1", "A").replace("0", "I") != v.mask:
            print(f"  WARNING: {v.label}: producer smartPixelsActiveLayers {first['activeLayers']} != mask {v.mask}")
    lab, rh = OSU.rechit_params(blobs[files[0]])
    keep = ("angleSource", "discScope", "emuSigmaSet", "emuLibrary", "emuBarrelModel", "emuDiscModel",
            "emuBarrelA1", "emuDiscA1", "emuInflation", "emuDiscAlphaOffset", "angleSet", "noiseSet")
    rechit = {k: rh[k] for k in keep if rh and k in rh}
    odd = [os.path.basename(f) for f in files[1:] if (OSU.rechit_params(blobs[f])[1] or {}).get("emuSigmaSet")
           != rechit.get("emuSigmaSet")]
    if odd:
        print(f"  WARNING: the rec-hit producer's emuSigmaSet differs between inputs: {odd}")
    prov = {"created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "inputs": OSU.file_info(files), "n_inputs": len(files),
            "cmssw_build": args.build_hash or "not recorded in the nano (pass --build-hash)",
            "refit_producers": prod, "rechit": dict(rechit, label=lab),
            "ngtagger": OSU.git_state(os.path.abspath(__file__)),
            "variants": [v.label for v in variants], "argv": sys.argv,
            "geometry": ({"path": os.path.abspath(args.geometry), "sha256": OSU.sha256(args.geometry)}
                         if args.geometry and os.path.exists(args.geometry) else None)}
    return prov, refit_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--outdir", default="eval_refitq/combinatorics")
    ap.add_argument("--variants", default=None,
                    help="refit variants to study: a comma list of activeSP masks (coexist "
                         "tables L1TSmartPixels{RefitHit,Track}SeedlessRefit<mask>) and/or "
                         "'canonical' (the unsuffixed tables), or 'all'. Default: the input's "
                         "only variant (an error if it carries several). With more than one, "
                         "per-variant studies write one page each into <outdir>/<variant>/ and "
                         "the shared studies (incl. the sweep over the selected set) one page "
                         "into <outdir>")
    ap.add_argument("--canonical-mask", default=None,
                    help="activeSP mask of the canonical tables (e.g. AAAI); read from the "
                         "input's ParameterSets when omitted, and checked against it when given")
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
    ap.add_argument("--seed-budget-gb", type=float, default=1.5,
                    help="studies 12/13: the seeder's candidate-slice memory budget "
                         "(tracklet_topology_cost DEFAULT_PAIR_BUDGET_GB)")
    ap.add_argument("--seed-rss-gb", type=float, default=6.0,
                    help="studies 12/13: the seeder's RSS ceiling (it raises above it)")
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
                    help="event cap for studies 19 and 20 (default: --max-events, else all)")
    ap.add_argument("--max-events", type=int, default=None,
                    help="DEVELOPMENT: read only the first N events of the inputs (in order) in "
                         "every in-memory study; studies 12 (OT part) and 14 do not run under it")
    ap.add_argument("--refit-calibration", default=None,
                    help="v3 refit calibration JSON (refit_calibration_fit.export_refit_calibration) "
                         "for study 20; default: the producer's covariance model. The v3 model is "
                         "barrel-only: with it study 20 visits IL1-IL4 only")
    ap.add_argument("--refit-use-angles", default="alphaBeta", choices=("alphaBeta", "alpha"),
                    help="angles the EMULATED refit variants of study 20 update on (the producer's "
                         "setting is alphaBeta). The replay against CMSSW always uses the producer's "
                         "own useAngles, read from the nano's ParameterSets")
    ap.add_argument("--build-hash", default=None,
                    help="CMSSW build (git) the inputs were produced with, for the monitor's "
                         "provenance: the nano does not record it")
    ap.add_argument("--cmssw-src", default=None,
                    help="CMSSW src directory to resolve the nano's data paths (the emulator sigma "
                         "table emuSigmaSet) against; default $CMSSW_BASE/src")
    ap.add_argument("--emu-sigma-table", default=None,
                    help="the emulator sigma-table JSON (R3), overriding the nano's emuSigmaSet path")
    ap.add_argument("--with-trackmva", action="store_true",
                    help="run R5 (track MVAs per variant: trains HGB heads with event cross-fitting "
                         "unless --trackmva-bundle holds the variant's frozen bundle)")
    ap.add_argument("--trackmva-bundle", default=None,
                    help="directory of frozen trackmva bundles (trackmva_<family>_<mask>) for R5")
    ap.add_argument("--train-only", nargs="*", default=["_f01_", "_f02_"],
                    help="R5: files (substring of the basename) whose events only train, never "
                         "evaluate (f01/f02 are in-sample for the cluster CNN)")
    ap.add_argument("--proj-chunk", type=int, default=10,
                    help="events per chunk for study 19 (~0.35 GB RSS at 10)")
    ap.add_argument("--angle-cap-reference", default=ANGLE_CAP_REFERENCE,
                    help="reference JSON for the angle-cap sentinel's WARNING band")
    ap.add_argument("--write-angle-cap-reference", action="store_true",
                    help="measure the angle-cap census on -i, store it as the reference "
                         "of the input's era in --angle-cap-reference, and exit")
    args = ap.parse_args()
    global MAX_EVENTS
    MAX_EVENTS = args.max_events
    if args.proj_nev is None:
        args.proj_nev = args.max_events
    files = [f for p in args.inputs for f in (sorted(_glob.glob(p)) or [p])]
    if args.write_angle_cap_reference:
        write_angle_cap_reference(files, args.angle_cap_reference)
        return
    os.makedirs(args.outdir, exist_ok=True)

    variants = select_variants(files, args.variants, args.canonical_mask)
    print(f"  refit variants: {', '.join(f'{v.label} [{v.hit}]' for v in variants)}")

    prov, refit_config = provenance(files, variants, args)
    monitor, status = {"provenance": prov}, []
    cmssw_search = [d for d in (args.cmssw_src, os.path.join(os.environ.get("CMSSW_BASE", ""), "src")
                                if os.environ.get("CMSSW_BASE") else None) if d]
    sentinel = angle_cap_sentinel(files, args.outdir, args.angle_cap_reference)
    monitor["angle_cap_sentinel"] = {"era": sentinel["era"], "n_warnings": sentinel["n_warnings"]}
    K, T, n_ev = load_clusters(files)
    base = {"n_events": n_ev, "n_clusters": int(len(K["layer"])),
            "angle_cap_sentinel": sentinel,
            # side-artefact directory for studies that write their own files;
            # stripped before the JSON is dumped
            "_outdir": args.outdir,
            # studies that re-read the file, and the variants the run selected
            "_inputs": files, "_variants": variants,
            "_menu_inputs": args.menu_inputs, "_menu_nev": args.menu_nev, "_menu_chunk": args.menu_chunk,
            "_menu_budget_gb": args.menu_budget_gb,
            "_seed_budget_gb": args.seed_budget_gb, "_seed_rss_gb": args.seed_rss_gb,
            "_menu_rss_gb": args.menu_rss_gb, "_menu_cache": args.menu_cache,
            "_site": args.site,
            "_menu_cache_mode": args.menu_cache_mode,
            "_menu_export_tracks": args.menu_export_tracks,
            "_geometry": args.geometry, "_proj_nev": args.proj_nev,
            "_proj_chunk": args.proj_chunk,
            "_refit_calibration": args.refit_calibration,
            "_refit_use_angles": args.refit_use_angles, "_refit_config": refit_config,
            "_provenance": prov, "_cmssw_search": cmssw_search, "_emu_sigma_table": args.emu_sigma_table,
            "_with_trackmva": args.with_trackmva, "_trackmva_bundle": args.trackmva_bundle,
            "_train_only": args.train_only,
            "_monitor": monitor, "_status": status, "_status_root": args.outdir,
            "_proj_inputs": ([f for g in args.proj_inputs.split(",") for f in
                              (sorted(_glob.glob(g)) or [g])] if args.proj_inputs else None)}

    gp = write_glossary(args.outdir)
    print(f"  glossary: {gp}  (defines crossing, cone, qX vs pXX, containment, max)")

    def pick(scopes):
        sel = [(t, d, [g for g in grp if g[3] in scopes and (not args.only or args.only in g[0])])
               for t, d, grp in SECTIONS]
        return [(t, d, g) for t, d, g in sel if g]
    if not pick((VARIANT, SHARED)):
        raise SystemExit(f"--only {args.only!r} matched no study")
    # PAGES. One variant: a single page with every study, as before. Several: the
    # shared studies once on <outdir>'s page, the variant studies on one page per
    # variant in <outdir>/<variant>/ -- one grid per build stays readable.
    if len(variants) == 1:
        pages = [(args.outdir, variants[0], pick((VARIANT, SHARED)))]
    else:
        pages = ([(args.outdir, None, pick((SHARED,)))]
                 + [(os.path.join(args.outdir, v.name), v, pick((VARIANT,))) for v in variants])
    for od, v, sections in pages:
        if not sections:
            continue
        out = dict(base, _outdir=od)
        if v is None:
            X = P = None
            out["variants"] = [v_.label for v_ in variants]
            title = (f"SmartPixels combinatorics omnibus — shared studies, variants "
                     f"{','.join(v_.name for v_ in variants)}, {n_ev} events")
        else:
            print(f"\n######## refit variant {v.label}")
            X = load_crossings(files, v, T)
            P = prepare(X, K)
            out.update({"config": v.mask, "n_crossings": int(len(X["layer"])), "_variant": v,
                        "variant": {"name": v.name, "mask": v.mask, "hit_table": v.hit,
                                    "track_table": v.track}})
            title = f"SmartPixels combinatorics omnibus — {v.label}, {n_ev} events"
        run_page(sections, X, K, P, out, title)
        del X, P
    monitor["status"] = status
    mp = os.path.join(args.outdir, "spix_monitor_summary.json")
    with open(mp, "w") as fh:
        json.dump(OSU.jsonable(monitor), fh, indent=1)
    print(f"\nwrote {mp}\n\nSTUDY STATUS")
    for e in status:
        print(f"  {e['page']:>10s}  {e['study']:42s} {e['seconds']:7.1f}s  rss<={e['max_rss_gb_so_far']:5.2f}GB  "
              f"{e['status'][:140]}")
    failed = [e for e in status if e["status"].startswith("FAILED")]
    if failed:
        raise SystemExit(f"{len(failed)} study(ies) FAILED (see above)")


if __name__ == "__main__":
    main()
