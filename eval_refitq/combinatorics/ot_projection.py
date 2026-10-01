"""OT-only track projection onto the inner-tracker surfaces, independent of the refit.

WHAT THIS IS FOR. A refit starts from an OT-only track and has to decide which
inner-tracker clusters to include. Before any Kalman update, the only thing it
knows is the OT fit: its helix and its covariance. This module projects that
helix, with that covariance, onto EVERY module of the SmartPixels surfaces --
the barrel layers IL1-IL4 and the discs ID1-ID4 (TFPX disks 1-4, both sides) --
it could reach within a k-sigma ellipse, and labels every cluster found there by
who made it. It deliberately does NOT use the refit's per-crossing tables:

  * the refit keeps ONE module per surface (the nearest containing one), so
    clusters on a neighbouring module inside the ellipse are never offered;
  * the refit records its seed projection only where the seed lands on the
    same module as the KF-UPDATED state, so under outsideIn the inner surfaces
    are conditioned on the outer updates;
  * only the diagonal of the projected covariance is stored there, and
    grazing predictions are dropped by the refit's angle clamp.

Those tables are used here only as an independent cross-check of the helix,
frame and Jacobian (validate_against_refit).

SURFACES, as SmartPixelsHelixProjector (L1Trigger/Phase3SmartPixels) builds them,
codes as the nano's (1-4 = IL1-IL4, 11-14 = ID1-ID4):
  barrel  the helix meets the layer CYLINDER at the mean module radius
          (cylinder_s), then each candidate module plane (Newton in s);
  disc    the helix meets the disc's mean-|z| PLANE on the side it heads to,
          z = sign(tanL) * Z_disc (s = (z - z0) / tanL, s > 0 and within half a
          turn: disc_s), then each candidate module plane. The disc modules sit
          on four faces within +-0.9 cm of that plane; the Newton step resolves it.
  surface_s dispatches on the code. A disc module's local x is r-phi, local y
  the RADIUS (inward), so the residuals (du, dv) are (r-phi, r) on a disc.

INPUTS (nano + one JSON):
  L1TTrack               helix (rInv, phi, tanL, z0, d0) and its fit covariance
                         cov_<p>_<q> (L1TrackHelixCovTableProducer)
  L1TTrackStub           the track's stubs (link-only: the stub columns are
                         L1TOTStub[otStubIdx]); L1TOTStub their truth (ot_oracle.
                         stub_truth, exact via otStubIdx)
  L1TSmartPixelsCluster  untruncated IT clusters: local position, sensor angle
                         estimate (localCotAlpha/Beta), dominant TP
  module geometry JSON   SmartPixelsModuleGeometryDumper: per module origin and
                         local axes AS GLOBAL VECTORS (see COORDINATES)

COORDINATES -- the point of the whole exercise. Every residual is computed in
the local frame of the module the CLUSTER is on, after projecting the track onto
THAT module's plane. A global direction d has local components
(d.ex, d.ey, d.ez), which is GeomDet::toLocal; cotAlpha = d.ex/d.ez,
cotBeta = d.ey/d.ez, exactly as the helix projector (Crossing::cotAlpha) and
SmartPixelsRecHitProducer compute them. So the track angle and the cluster angle
are always in the same frame, tilted and flipped modules included (about half
the barrel modules have ez pointing inward). The residual itself does not care
which frame it is in, only that both sides share one; this guarantees it.
Closure measured on PU200 (itot_truth_100ev): cluster local -> global position
<= 2 um; true global direction -> stored tpLocalCot* median 1e-5 (a swapped-axis
frame would give 1.07).

HELIX, in TTTrack's own conventions, exactly as SmartPixelsHelixProjector::crossLayer
(no sign flips): rInv > 0 turns CLOCKWISE, phi(s) = phi0 - rInv s;
  POCA (x0, y0) = (d0 sin phi0, -d0 cos phi0);
  x = x0 - (sin(phi0 - rInv s) - sin phi0)/rInv,  y = y0 + (cos(phi0 - rInv s) - cos phi0)/rInv,
  z = z0 + tanL s,  direction (cos(phi0 - rInv s), sin(phi0 - rInv s), tanL);
  s = transverse arc length, started at the layer-cylinder crossing and moved
  onto each module plane by Newton iteration.
The projected covariance is J C J^T, J the numerical Jacobian of
(u, v, cotAlpha, cotBeta) with respect to (rInv, phi, tanL, z0, d0), C the OT
fit covariance with NO multiple-scattering term: the single-shot cold start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from ngtagger.data import spix_nano as SPN
from ngtagger.truth_helix import KPT_CMSSW, tp_phi0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ot_oracle import OT_LINK, stub_truth   # noqa: E402
import tracklet_topology_cost as M          # noqa: E402

HPAR = ("rInv", "phi", "tanL", "z0", "d0")          # TTTrack::Hpar order
COV_COLS = [f"cov_{HPAR[i]}_{HPAR[j]}" for i in range(5) for j in range(i, 5)]
# Jacobian steps: the producer's own seed-cone steps (kEpsS), used here as
# central differences.
JAC_EPS = np.array([1e-6, 1e-5, 1e-5, 1e-3, 1e-3])
IT_LAYERS = SPN.BARREL                  # IL1-IL4 (the barrel-only calibration scripts)
IT_SURFACES = SPN.SP_SURFACES           # IL1-IL4, ID1-ID4
# Candidate modules per (track, surface): every module whose centre is within
#   BOUND_SIGMA_MARGIN * nsig * sqrt(sigma_u^2 + sigma_v^2)       (ellipse box corner)
#   + half-diagonal of the module
#   + BOUND_SLACK_CM * sqrt(1 + tanL^2)        barrel: cylinder -> module plane
#     BOUND_SLACK_CM * sqrt(1 + 1/tanL^2)      disc: mean plane -> module face (+-0.9 cm in z)
# of the nominal-surface crossing, sigmas taken on the nearest module. NOT
# k-nearest-by-centre: 4 sigma in v reaches ~6 cm on soft tracks and a long
# module overlaps while its centre is far away (k = 24 vs 48 changed the B
# counts). Measured by exhaustive projection onto every module of the layer:
# every overlapping module lies within |s - s0| <= 1.0 cm of transverse arc of
# the cylinder crossing, <= 9 modules per track at IL1. The trace (not the
# largest eigenvalue) is needed because the overlap test is a bounding box.
# Self-check reported: the largest dist/bound of any OVERLAPPING module.
BOUND_SIGMA_MARGIN = 1.5
BOUND_SLACK_CM = 2.0
NEWTON_ITERS = 8
ON_PLANE_CM = 1e-4        # |distance to plane| accepted as converged (1 um)
REFINE_STEPS = 2          # project_to_modules(refine=): Newton steps past convergence
# SmartPixelsHelixProjector / L1SmartPixelsTrackProducer constants
BARREL_CANDIDATES = 4     # crossLayer: the k nearest module centres
DISC_CAND_SLACK_CM = 1.5  # crossDisc: centres within the largest half-diagonal + this
TRAVERSED_MARGIN_CM = 1.0  # pathToSurface: a surface is traversed within its extent + this

TRK_COLS = list(HPAR) + ["pt", "eta", "nStubs"] + COV_COLS
TST_COLS = ["trackIdx", "detId", "x", "y", "z", "bend", "layer", "isBarrel"]
OTS_COLS = ["detId", "x", "y", "z", "bend", "layer", "isBarrel", "tpIdx",
            "tpGenuine", "tpCombinatoric", "tpUnknown"]
CL_COLS = ["detId", "surface", "localX", "localY", "localCotAlpha",
           "localCotBeta", "hasAlpha", "hasBeta", "tpIdx",
           "sigX", "sigY", "sigAlpha", "sigBeta",      # errors: the refit's measurement R
           "sizeX", "sizeY"]                             # cluster size: calibrated R lookup
CL_TABLE = SPN.CLUSTER_TABLE
# default refit hit table (a coexist nano's AAAA variant); every reader takes `refit=`
# so a caller can point it at another variant or at the canonical unsuffixed table
REFIT = "L1TSmartPixelsRefitHitSeedlessRefitAAAA"
REFIT_COLS = ["trackIdx", "detId", "layer", "projOTLocalX", "projOTLocalY",
              "projOTCotAlpha", "projOTCotBeta", "projOTSigX", "projOTSigY"]
SENTINEL = -900.0

# --------------------------------------------------------------------------
# track classes, by the stubs the OT track is built from
# --------------------------------------------------------------------------
# majority-owner TP: the TrackingParticle owning the most GENUINE stubs of the
#   track. It is the particle the track is "for", and its IT clusters are the
#   targets a refit should include.
# foreign stub: a GENUINE stub (both clusters from one TP) whose TP is NOT the
#   majority owner -- a real piece of another particle.
# fake stub: a stub that is not genuine at all -- stub-level combinatoric (its
#   two clusters come from different TPs) or unknown (no TP: noise, out-of-time
#   pileup). Its TPs are not recorded (L1TOTStub_tpIdx = -1), so a fake stub
#   cannot name a co-owner.
# NB "combinatoric" is avoided as a TRACK class name on purpose: CMSSW uses it
#   for tracks, L1TOTStub for stubs, and they mean different things.
# NB CMSSW's L1TTrack_genuine is NOT "every stub genuine": measured on PU200,
#   1,224 of 17,882 genuine-flagged tracks carry exactly one fake stub.
TRACK_CLASSES = ("perfect", "foreign-1", "foreign-2+", "fake-1", "fake-2+",
                 "foreign-fake")
TRACK_CLASS_DEF = {
    "perfect": "every stub genuine and owned by the majority TP",
    "foreign-1": "exactly 1 foreign stub, no fake stub",
    "foreign-2+": "2 or more foreign stubs, no fake stub",
    "fake-1": "exactly 1 fake stub, no foreign stub",
    "fake-2+": "2 or more fake stubs, no foreign stub",
    "foreign-fake": "at least 1 foreign AND at least 1 fake stub",
}
# EXCLUDED, counted and reported, never silently dropped:
#   tie       -- two TPs own the same, largest number of genuine stubs, so there
#                is no majority owner and no target
#   no-owner  -- no genuine stub at all
#   unjoined  -- a stub whose truth could not be joined (0 on PU200 ttbar)
EXCLUDED = ("tie", "no-owner", "unjoined")

# --------------------------------------------------------------------------
# cluster categories on a surface projection (one track onto one layer or disc)
# --------------------------------------------------------------------------
# A   -- a cluster of the majority-owner TP anywhere on the surface: the TARGET.
#        Split A-in / A-out by the k-sigma ellipse; A-out residuals are taken on
#        the target's own module, so they are still frame-consistent.
# B1  -- inside the ellipse, owned by a TP that also owns a foreign stub of
#        this track: the trap a contaminated track sets for itself.
# B2  -- inside the ellipse, owned by any other TP.
# B3  -- inside the ellipse, no TP (tpIdx -1/-2): noise / unlinked. A TP not
#        written to L1TTP (tpIdx <= -3) is a TP: B1/B2.
CATEGORIES = ("A-in", "A-out", "B1", "B2", "B3")


def _surface_code(m):
    """Surface code of a geometry-JSON module: PXB layer, or 10 + PXF disk."""
    return m["layer"] if m["subdet"] == "PXB" else SPN.DISC_SURFACE_OFFSET + m["disk"]


def load_geometry(path, surfaces=IT_SURFACES):
    """Modules of the requested surfaces, as arrays sorted by detId, plus the
    per-surface constants SmartPixelsHelixProjector::build derives:
      R[L], halfZ[L]   barrel: mean module-centre radius; largest |z| of the
                       module outlines (corners and edge midpoints)
      Z[D], rext[D]    disc: mean |z| of the module centres (both sides); radial
                       extent (min, max) of the module outlines
      maxHalfDiag[D]   disc: largest module half-diagonal
      by_surface[S]    module rows of surface S; by_side[(D, side)] the rows of
                       disc D on the side z < 0 (0) or z > 0 (1)."""
    G = json.load(open(path))
    ms = [m for m in G["modules"] if m["subdet"] in ("PXB", "PXF") and _surface_code(m) in surfaces]
    ms.sort(key=lambda m: m["detId"])
    g = {"detId": np.array([m["detId"] for m in ms], np.int64),
         "surface": np.array([_surface_code(m) for m in ms], np.int64)}
    for k in ("origin", "ex", "ey", "ez"):
        g[k] = np.array([m[k] for m in ms], np.float64)
    g["hw"] = np.array([m["halfWidth"] for m in ms], np.float64)
    g["hl"] = np.array([m["halfLength"] for m in ms], np.float64)
    # the outline the projector measures extents on: corners and edge midpoints
    f = np.array([(fx, fy) for fx in (-1., 0., 1.) for fy in (-1., 0., 1.) if fx or fy])
    outline = (g["origin"][:, None, :] + (f[None, :, 0] * g["hw"][:, None])[..., None] * g["ex"][:, None, :]
               + (f[None, :, 1] * g["hl"][:, None])[..., None] * g["ey"][:, None, :])
    g["surfaces"] = tuple(s for s in surfaces if (g["surface"] == s).any())
    g["by_surface"] = {s: np.flatnonzero(g["surface"] == s) for s in g["surfaces"]}
    g["R"], g["halfZ"], g["Z"], g["rext"], g["maxHalfDiag"], g["by_side"] = {}, {}, {}, {}, {}, {}
    for s in g["surfaces"]:
        rows = g["by_surface"][s]
        if not SPN.is_disc_surface(s):
            g["R"][s] = float(np.hypot(*g["origin"][rows][:, :2].T).mean())
            g["halfZ"][s] = float(np.abs(outline[rows][..., 2]).max())
            continue
        g["Z"][s] = float(np.abs(g["origin"][rows][:, 2]).mean())
        rr = np.hypot(outline[rows][..., 0], outline[rows][..., 1])
        g["rext"][s] = (float(rr.min()), float(rr.max()))
        g["maxHalfDiag"][s] = float(np.hypot(g["hw"][rows], g["hl"][rows]).max())
        for side in (0, 1):
            g["by_side"][(s, side)] = rows[(g["origin"][rows][:, 2] > 0) == bool(side)]
    return g


def module_index(geo, detid):
    """Row in geo for each detId, -1 where the module is not in geo."""
    detid = np.asarray(detid, np.int64)
    i = np.clip(np.searchsorted(geo["detId"], detid), 0, len(geo["detId"]) - 1)
    return np.where(geo["detId"][i] == detid, i, -1)


# --------------------------------------------------------------------------
# helix
# --------------------------------------------------------------------------
def helix_at(a, s):
    """Position and unit-free direction (cos, sin, tanL) at transverse arc s.
    TTTrack conventions throughout: phi(s) = phi0 - rInv s (clockwise for rInv > 0)."""
    rinv = a[..., 0]
    phi0, tanl, z0, d0 = a[..., 1], a[..., 2], a[..., 3], a[..., 4]
    x0, y0 = d0 * np.sin(phi0), -d0 * np.cos(phi0)
    straight = np.abs(rinv) < 1e-6
    rs_ = np.where(straight, 1.0, rinv)
    ph = phi0 - np.where(straight, 0.0, rinv * s)
    x = np.where(straight, x0 + s * np.cos(phi0), x0 - (np.sin(ph) - np.sin(phi0)) / rs_)
    y = np.where(straight, y0 + s * np.sin(phi0), y0 + (np.cos(ph) - np.cos(phi0)) / rs_)
    z = z0 + tanl * s
    return (np.stack([x, y, z], -1),
            np.stack([np.cos(ph), np.sin(ph), tanl], -1))


def cylinder_s(a, R):
    """Transverse arc length to radius R (NaN if never reached), as crossLayer.
    The circle centre is (x0 + sin(phi0)/rInv, y0 - cos(phi0)/rInv) in TTTrack's convention."""
    rinv = a[..., 0]
    phi0, d0 = a[..., 1], a[..., 4]
    x0, y0 = d0 * np.sin(phi0), -d0 * np.cos(phi0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = x0 * np.cos(phi0) + y0 * np.sin(phi0)
        c = x0 * x0 + y0 * y0 - R * R
        s_line = -b + np.sqrt(b * b - c)
        Rc = 1.0 / np.where(np.abs(rinv) < 1e-6, 1.0, rinv)
        cx, cy = x0 + Rc * np.sin(phi0), y0 - Rc * np.cos(phi0)
        dc, aR = np.hypot(cx, cy), np.abs(Rc)
        cosarg = (aR * aR + dc * dc - R * R) / (2.0 * aR * dc)
        ok = (R <= dc + aR) & (R >= np.abs(dc - aR)) & (np.abs(cosarg) <= 1.0)
        s_arc = np.where(ok, np.arccos(np.clip(cosarg, -1, 1)) * aR, np.nan)
    s = np.where(np.abs(rinv) < 1e-6, s_line, s_arc)
    return np.where(s > 0, s, np.nan)


def disc_s(a, Z):
    """Transverse arc length to the disc plane z = sign(tanL) * Z on the side the
    helix heads to (NaN if |tanL| < 1e-9, s <= 0 or beyond half a turn), as
    HelixProjector::crossDisc / pathToSurface."""
    rinv, tanl, z0 = a[..., 0], a[..., 2], a[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (np.where(tanl > 0, 1.0, -1.0) * Z - z0) / tanl
    ok = (np.abs(tanl) >= 1e-9) & (s > 0) & ~(np.abs(rinv) * s > np.pi)
    return np.where(ok, s, np.nan)


def surface_s(a, geo, S):
    """Arc length to the nominal surface S (cylinder for a layer, mean plane for a disc)."""
    return disc_s(a, geo["Z"][S]) if SPN.is_disc_surface(S) else cylinder_s(a, geo["R"][S])


def surface_path_s(a, geo, S, margin=TRAVERSED_MARGIN_CM):
    """HelixProjector::pathToSurface: (s, traversed). s as surface_s (NaN where the
    helix does not meet the surface); traversed when that point lies within the
    surface's module extent widened by `margin` (barrel |z| <= halfZ + margin,
    disc radius within rext +- margin): the surfaces the particle crosses, which
    order the visits and are charged scattering material."""
    s = surface_s(a, geo, S)
    ok = np.isfinite(s)
    p, _ = helix_at(a, np.where(ok, s, 0.0))
    if SPN.is_disc_surface(S):
        r = np.hypot(p[..., 0], p[..., 1])
        lo, hi = geo["rext"][S]
        crossed = ok & (r >= lo - margin) & (r <= hi + margin)
    else:
        crossed = ok & (np.abs(p[..., 2]) <= geo["halfZ"][S] + margin)
    return s, crossed


def _newton_cmssw(a, s0, o, n_):
    """SmartPixelsHelixProjector::crossLayer's Newton loop, verbatim semantics:
    at most 5 iterations, converged if |f| < 1 um at the START of an iteration,
    abandoned on a flat derivative or an unphysical step."""
    s = np.where(np.isfinite(s0), s0, 1.0)
    alive = np.isfinite(s0)
    conv = np.zeros(len(s), bool)
    for _ in range(5):
        p, d = helix_at(a, s)
        f = ((p - o) * n_).sum(-1)
        fp = (d * n_).sum(-1)
        now = alive & ~conv & (np.abs(f) < ON_PLANE_CM)
        conv |= now
        act = alive & ~conv
        flat = act & (np.abs(fp) < 1e-9)
        alive &= ~flat
        act &= ~flat
        s = np.where(act, s - f / np.where(act, fp, 1.0), s)
        bad = act & (~(s > 0) | (s > 3.0 * np.where(np.isfinite(s0), s0, 1.0) + 10.0))
        alive &= ~bad
    return s, conv


def _surface_candidates(a, S, geo, s0):
    """Candidate module rows (n, k) for HelixProjector::land, nearest first, -1
    padding: crossLayer's k = 4 module centres nearest the cylinder point, or
    crossDisc's every centre on the side the helix heads to within the largest
    half-diagonal + DISC_CAND_SLACK_CM of the mean-plane point."""
    pc, _ = helix_at(a, np.where(np.isfinite(s0), s0, 1.0))
    if not SPN.is_disc_surface(S):
        rows = geo["by_surface"][S]
        d2 = ((pc[:, None, :] - geo["origin"][rows][None, :, :]) ** 2).sum(-1)
        k = min(BARREL_CANDIDATES, len(rows))
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        o = np.argsort(np.take_along_axis(d2, part, 1), axis=1, kind="stable")
        return rows[np.take_along_axis(part, o, 1)]
    out = []
    rmax2 = (geo["maxHalfDiag"][S] + DISC_CAND_SLACK_CM) ** 2
    for side in (0, 1):
        rows = geo["by_side"][(S, side)]
        d2 = ((pc[:, None, :] - geo["origin"][rows][None, :, :]) ** 2).sum(-1)
        d2 = np.where(d2 < rmax2, d2, np.inf)
        o = np.argsort(d2, axis=1, kind="stable")
        out.append((np.where(np.isfinite(np.take_along_axis(d2, o, 1)), rows[o], -1)))
    k = max(int((out[0] >= 0).sum(1).max(initial=0)), int((out[1] >= 0).sum(1).max(initial=0)), 1)
    up = a[:, 2] > 0
    return np.where(up[:, None], out[1][:, :k], out[0][:, :k])


def cross_module(a, S, geo):
    """The module SmartPixelsHelixProjector::crossSurface resolves for helices a
    on surface S (crossLayer / crossDisc): each candidate (_surface_candidates)
    re-solved onto its plane by crossLayer's Newton loop from the nominal-surface
    crossing, kept if the on-plane point is inside the bounds, the one whose
    centre is nearest THAT point winning. -1 if none."""
    s0 = surface_s(a, geo, S)
    near = _surface_candidates(a, S, geo, s0)
    best = np.full(len(a), -1, np.int64)
    bestd = np.full(len(a), np.inf)
    for c in range(near.shape[1]):
        have = near[:, c] >= 0
        mi = np.where(have, near[:, c], 0)
        s, ok = _newton_cmssw(a, s0, geo["origin"][mi], geo["ez"][mi])
        p, _ = helix_at(a, s)
        rel = p - geo["origin"][mi]
        u, v = (rel * geo["ex"][mi]).sum(-1), (rel * geo["ey"][mi]).sum(-1)
        inside = have & ok & np.isfinite(s0) & (np.abs(u) <= geo["hw"][mi]) & (np.abs(v) <= geo["hl"][mi])
        dd = (rel ** 2).sum(-1)
        take = inside & (dd < bestd)
        best = np.where(take, mi, best)
        bestd = np.where(take, dd, bestd)
    return best


def project_to_modules(a, s0, geo, mi, newton="study", refine=None):
    """Project helices a[n] onto module rows mi[n], Newton from arc s0[n].

    Returns dict of u, v (local position, cm), cotA, cotB, s and ok, where ok
    means the helix reached the plane (converged, s > 0). u, v may lie OUTSIDE
    the module bounds; containment is the caller's question.

    refine (boolean per row): after convergence, REFINE_STEPS more Newton steps
    without the 1 um stop. For the Jacobian's perturbed solves on a DISC: a disc
    module's normal is along z, so a tanL or z0 step moves the crossing ALONG
    the normal by ~1 um -- the stop tolerance -- and an unrefined solve that
    starts on the nominal plane often never moves, losing the column (the
    anti-correlated tanL-z0 pair then inflated sigma_r by a median 14-17% on
    ID1/ID2 against the producer). Barrel perturbations move the crossing
    mostly in-plane; they are NOT refined, which keeps barrel results as they
    were (their width carries a +-1-6% p1/p99 noise from the same stop).
    """
    o, n_ = geo["origin"][mi], geo["ez"][mi]
    if newton == "cmssw":
        s, ok = _newton_cmssw(a, s0, o, n_)
        p, d = helix_at(a, s)
        rel = p - o
        lz = (d * n_).sum(-1)
        lz = np.where(np.abs(lz) > 1e-9, lz, 1e-9)
        return {"u": (rel * geo["ex"][mi]).sum(-1), "v": (rel * geo["ey"][mi]).sum(-1),
                "cotA": (d * geo["ex"][mi]).sum(-1) / lz,
                "cotB": (d * geo["ey"][mi]).sum(-1) / lz, "s": s, "ok": ok}
    s = s0.copy()
    ok = np.isfinite(s)
    s = np.where(ok, s, 1.0)
    conv = np.zeros(len(s), bool)
    for _ in range(NEWTON_ITERS):
        p, d = helix_at(a, s)
        f = ((p - o) * n_).sum(-1)
        fp = (d * n_).sum(-1)
        conv = np.abs(f) < ON_PLANE_CM
        step_ok = np.abs(fp) > 1e-9
        s = np.where(conv | ~step_ok, s, s - f / np.where(step_ok, fp, 1.0))
        ok &= step_ok | conv
        ok &= (s > 0) & (s < 3.0 * np.where(np.isfinite(s0), s0, 1.0) + 10.0)
    p, d = helix_at(a, s)
    conv = np.abs(((p - o) * n_).sum(-1)) < ON_PLANE_CM
    ok &= conv
    if refine is not None and refine.any():
        for _ in range(REFINE_STEPS):
            f = ((p - o) * n_).sum(-1)
            fp = (d * n_).sum(-1)
            go = refine & ok & (np.abs(fp) > 1e-9)
            s = np.where(go, s - f / np.where(go, fp, 1.0), s)
            p, d = helix_at(a, s)
    rel = p - o
    lz = (d * geo["ez"][mi]).sum(-1)
    lz = np.where(np.abs(lz) > 1e-9, lz, 1e-9)
    return {"u": (rel * geo["ex"][mi]).sum(-1), "v": (rel * geo["ey"][mi]).sum(-1),
            "cotA": (d * geo["ex"][mi]).sum(-1) / lz,
            "cotB": (d * geo["ey"][mi]).sum(-1) / lz, "s": s, "ok": ok}


OUT_KEYS = ("u", "v", "cotA", "cotB")


def project_with_cov(a, C, s0, geo, mi, jac="central", surface=None):
    """project_to_modules plus the 4x4 covariance of (u, v, cotA, cotB).

    jac="central": central differences (the default, more accurate).
    jac="cmssw": the producer's scheme exactly -- one-sided +eps falling back
    to -eps, each perturbed helix re-solved by crossSurface's own Newton loop
    started from ITS nominal-surface crossing and stopped at 1 um, and a
    perturbation that crossSurface would put on a DIFFERENT module is not usable
    (the producer's same-module guard; both sides unusable -> NaN column, which
    the update zeroes). `surface` is the surface code (1-4, 11-14). Used only
    where CMSSW must be reproduced (replay).

    NEWTON NOISE, why "central" does not simply reuse the producer's scheme:
    crossLayer stops at |distance to plane| < 1 um, and the rInv step (1e-6)
    moves an IL4 crossing by only ~1.3 um, so the producer's rInv column is a
    difference of two iterates each anywhere within 1 um of the plane. Here
    the perturbed solve starts from the converged nominal s, and the width
    agrees with the producer's to ~1% at p1/p99 (~5% before matching starts).
    """
    if jac == "cmssw":
        nom = project_to_modules(a, s0, geo, mi, newton="cmssw")
        J = np.full((len(mi), 4, 5), np.nan)
        for j in range(5):
            cols = []
            for sgn in (+1.0, -1.0):
                ap = a.copy()
                ap[:, j] += sgn * JAC_EPS[j]
                pp = project_to_modules(ap, surface_s(ap, geo, surface), geo, mi, newton="cmssw")
                pp["ok"] = pp["ok"] & (cross_module(ap, surface, geo) == mi)
                cols.append(pp)
            for r, key in enumerate(OUT_KEYS):
                fwd = (cols[0][key] - nom[key]) / JAC_EPS[j]
                bwd = (nom[key] - cols[1][key]) / JAC_EPS[j]
                J[:, r, j] = np.where(cols[0]["ok"], fwd, np.where(cols[1]["ok"], bwd, np.nan))
        nom["S"] = np.einsum("nij,njk,nlk->nil", J, C, J)
        nom["J"] = J
        nom["ok"] &= np.isfinite(nom["S"]).all((1, 2))
        return nom
    nom = project_to_modules(a, s0, geo, mi)
    disc = SPN.is_disc_surface(geo["surface"][mi])
    J = np.full((len(mi), 4, 5), np.nan)
    for j in range(5):
        cols = []
        for sgn in (+1.0, -1.0):
            ap = a.copy()
            ap[:, j] += sgn * JAC_EPS[j]
            cols.append(project_to_modules(ap, nom["s"], geo, mi, refine=disc))
        both = cols[0]["ok"] & cols[1]["ok"]
        for r, key in enumerate(OUT_KEYS):
            cen = (cols[0][key] - cols[1][key]) / (2 * JAC_EPS[j])
            fwd = (cols[0][key] - nom[key]) / JAC_EPS[j]
            bwd = (nom[key] - cols[1][key]) / JAC_EPS[j]
            J[:, r, j] = np.where(both, cen, np.where(cols[0]["ok"], fwd,
                                                     np.where(cols[1]["ok"], bwd, np.nan)))
    S = np.einsum("nij,njk,nlk->nil", J, C, J)
    nom["S"] = S
    nom["J"] = J              # d(u, v, cotA, cotB) / d(rInv, phi, tanL, z0, d0)
    nom["ok"] &= np.isfinite(S).all((1, 2))
    return nom


# --------------------------------------------------------------------------
# per-chunk flattening
# --------------------------------------------------------------------------
def _flat(A, pre, cols):
    """Flat numpy columns of table `pre`. The cluster table is read WHOLE
    (disc_scope "all": every row the producer wrote, barrel then discs), so
    selClusterIdx -- a row of the file's table, barrel or disc -- indexes it
    exactly as it does the file's. Clusters on modules outside the geometry
    (none on discScope-sp nanos) are ignored downstream (module_index -1)."""
    J = (SPN.cluster_columns(A, cols, disc_scope="all") if pre == CL_TABLE
         else SPN.table_columns(A, pre, cols))
    n = ak.to_numpy(ak.num(J[cols[0]]))
    D = {c: ak.to_numpy(ak.flatten(J[c])) for c in cols}
    D["event"] = np.repeat(np.arange(len(n)), n)
    D["_n"] = n
    return D


def _tables(with_refit, cl_extra=(), refit=REFIT):
    t = [("L1TTrack", TRK_COLS), ("L1TTrackStub", TST_COLS), ("L1TOTStub", OTS_COLS),
         (CL_TABLE, CL_COLS + [c for c in cl_extra if c not in CL_COLS])]
    return t + ([(refit, REFIT_COLS)] if with_refit else [])


def _keys(path):
    with uproot.open(f"{path}:Events") as t:
        return set(t.keys())


def branches(with_refit, path, cl_extra=(), refit=REFIT):
    """The branches the study reads from files shaped like `path`; `cl_extra`
    adds cluster columns (read with _flat(A, CL_TABLE, ...))."""
    keys = _keys(path)
    b = [f"L1TTrackStub_{OT_LINK}"]
    for tbl, cols in _tables(with_refit, cl_extra, refit):
        b += (SPN.cluster_branches(keys, cols, disc_scope="all") if tbl == CL_TABLE
              else SPN.branch_names(keys, tbl, cols))
    return b


def check_inputs(path, with_refit=False, refit=REFIT):
    keys = _keys(path)
    miss = [f"{tbl}_{c}" for tbl, cols in _tables(with_refit, refit=refit) for c in cols
            if not SPN.has_column(keys, tbl, c)]
    if miss:
        cov = [m for m in miss if "_cov_" in m]
        hint = ("\n  L1TTrack_cov_* are written by L1TrackHelixCovTableProducer "
                "(DPGAnalysis/Phase2L1TNanoAOD); regenerate the nano with it scheduled."
                if cov else "")
        raise SystemExit(f"{path} cannot support the OT-projection study; missing "
                         f"{len(miss)} branches, e.g. {miss[:6]}{hint}")


# --------------------------------------------------------------------------
# track classification
# --------------------------------------------------------------------------
def classify_tracks(T, S, O):
    """Per track: class label index (or -1..-3 for EXCLUDED), majority TP, and
    the packed (track, tp) keys of its foreign-stub TPs."""
    ntrk = len(T["pt"])
    off = np.concatenate([[0], np.cumsum(T["_n"])])
    gtrk = off[S["event"]] + S["trackIdx"].astype(np.int64)
    tp, flag, hit = stub_truth(S, O)
    n_unj = np.bincount(gtrk, weights=~hit, minlength=ntrk).astype(int)
    gen = hit & (flag == 1) & SPN.tp_matched(tp)
    # (global track, tpIdx) packed with M.tp_key (the track in the high word)
    key = M.tp_key(gtrk[gen], tp[gen])
    u, c = np.unique(key, return_counts=True)
    ut, utp = M.tp_key_parts(u)
    o = np.lexsort((-c, ut))
    ut, utp, c = ut[o], utp[o], c[o]
    first = np.r_[True, ut[1:] != ut[:-1]] if len(ut) else np.zeros(0, bool)
    second = np.r_[False, (~first[1:]) & first[:-1]] if len(ut) else np.zeros(0, bool)
    maj_tp = np.full(ntrk, -1, np.int64)
    maj_n = np.zeros(ntrk, np.int64)
    sec_n = np.zeros(ntrk, np.int64)
    maj_tp[ut[first]], maj_n[ut[first]] = utp[first], c[first]
    sec_n[ut[second]] = c[second]
    n_gen = np.bincount(gtrk, weights=gen, minlength=ntrk).astype(int)
    n_fake = np.bincount(gtrk, weights=hit & (flag >= 2), minlength=ntrk).astype(int)
    n_foreign = n_gen - maj_n
    cls = np.full(ntrk, -99, np.int64)
    cls[(n_foreign == 0) & (n_fake == 0)] = 0
    cls[(n_foreign == 1) & (n_fake == 0)] = 1
    cls[(n_foreign >= 2) & (n_fake == 0)] = 2
    cls[(n_fake == 1) & (n_foreign == 0)] = 3
    cls[(n_fake >= 2) & (n_foreign == 0)] = 4
    cls[(n_fake >= 1) & (n_foreign >= 1)] = 5
    cls[(maj_n > 0) & (sec_n == maj_n)] = -1               # tie
    cls[maj_n == 0] = -2                                   # no-owner
    cls[n_unj > 0] = -3                                    # unjoined
    assert (cls != -99).all()
    # foreign-stub TPs, as packed (global track, tp) keys, for the B1 test
    fr = gen & (tp != maj_tp[gtrk])
    foreign_keys = np.unique(M.tp_key(gtrk[fr], tp[fr]))
    return cls, maj_tp, foreign_keys


# --------------------------------------------------------------------------
# the study kernel, one chunk of events
# --------------------------------------------------------------------------
def track_state(T):
    a = np.stack([T[p].astype(np.float64) for p in HPAR], -1)
    C = np.zeros((len(a), 5, 5))
    for i in range(5):
        for j in range(i, 5):
            C[:, i, j] = C[:, j, i] = T[f"cov_{HPAR[i]}_{HPAR[j]}"]
    return a, C


def prepare_chunk(A, geo):
    """Everything about a chunk that does not depend on the track STATE:
    tables, track classes, and the (event, detId) cluster index."""
    T = _flat(A, "L1TTrack", TRK_COLS)
    S = _flat(A, "L1TTrackStub", TST_COLS + [OT_LINK])
    O = _flat(A, "L1TOTStub", OTS_COLS)
    K = _flat(A, CL_TABLE, CL_COLS)
    a, C = track_state(T)
    cls, maj_tp, foreign_keys = classify_tracks(T, S, O)
    covok = np.trace(C, axis1=1, axis2=2) > 0
    cls = np.where(covok, cls, -4)                         # covariance absent
    kmi = module_index(geo, K["detId"])                    # clusters on the geometry's surfaces
    kev = K["event"]
    kkey = kev.astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    toff = np.concatenate([[0], np.cumsum(T["_n"])])
    gst = toff[S["event"]] + S["trackIdx"].astype(np.int64)
    n_disk = np.bincount(gst, weights=S["isBarrel"] == 0, minlength=len(T["pt"])).astype(np.int64)
    return {"A": A, "T": T, "K": K, "ntrk": len(T["pt"]), "a": a, "C": C, "n_disk_stubs": n_disk,
            "tev": T["event"], "cls": cls, "maj_tp": maj_tp,
            "foreign_keys": foreign_keys, "kmi": kmi, "kin": kmi >= 0, "kev": kev,
            "ksurf": np.where(kmi >= 0, geo["surface"][np.maximum(kmi, 0)], -1), "kkey": kkey,
            "ko": np.argsort(kkey, kind="stable")}


def evaluate_surface(ctx, a, C, S, geo, nsig, use):
    """One surface S (1-4 IL, 11-14 ID), for the track STATE (a, C): candidate
    modules, the ellipse on each, and every cluster there labelled A-in / A-out /
    B1 / B2 / B3.

    Returns the study rows and per-projection bookkeeping (what study 19
    records), plus the candidate-level arrays a refit needs to choose a cluster
    and update on it (key "cand"). None if no track reaches the surface.
    """
    K, tev, ntrk = ctx["K"], ctx["tev"], ctx["ntrk"]
    maj_tp, foreign_keys, cls = ctx["maj_tp"], ctx["foreign_keys"], ctx["cls"]
    kmi, kin, kev, ksurf, kkey, ko = (ctx[k] for k in ("kmi", "kin", "kev", "ksurf", "kkey", "ko"))
    disc = bool(SPN.is_disc_surface(S))
    s0 = surface_s(a, geo, S)
    reach = use & np.isfinite(s0)
    ti = np.flatnonzero(reach)
    if not len(ti):
        return None
    pc, _ = helix_at(a[ti], s0[ti])
    rows = geo["by_surface"][S]
    dist = np.sqrt(((pc[:, None, :] - geo["origin"][rows][None, :, :]) ** 2).sum(-1))
    if disc:
        # only the disc on the side the helix heads to; the other side is ~2 Z away
        up = a[ti, 2] > 0
        dist = np.where((geo["origin"][rows][None, :, 2] > 0) == up[:, None], dist, np.inf)
    # ellipse size from the nearest module; a failed projection there gets
    # the surface-wide worst case so the track is not silently narrowed
    m0 = rows[np.argmin(dist, 1)]
    p0 = project_with_cov(a[ti], C[ti], s0[ti], geo, m0)
    tr = np.nan_to_num(p0["S"][:, 0, 0]) + np.nan_to_num(p0["S"][:, 1, 1])
    r_ell = BOUND_SIGMA_MARGIN * nsig * np.sqrt(np.clip(tr, 0, None))
    r_ell = np.where(p0["ok"], r_ell, np.nanmax(np.where(p0["ok"], r_ell, np.nan)) if p0["ok"].any() else 10.0)
    hdiag = np.hypot(geo["hw"][rows], geo["hl"][rows])
    tl2 = a[ti, 2] ** 2
    slack = BOUND_SLACK_CM * np.sqrt(1.0 + (1.0 / np.maximum(tl2, 1e-12) if disc else tl2))
    bound = (r_ell + slack)[:, None] + hdiag[None, :]
    it_, im_ = np.nonzero(dist <= bound)
    cand_t, cand_m = ti[it_], rows[im_]
    cand_ratio = (dist / bound)[it_, im_]
    # target modules: every IT module on this surface holding a cluster of
    # the majority-owner TP, even if it is not among the candidates
    tgt = kin & (ksurf == S) & SPN.tp_matched(K["tpIdx"])
    tk = M.tp_key(kev[tgt], K["tpIdx"][tgt])
    has_maj = SPN.tp_matched(maj_tp[ti])      # no owner: -1, matches no tk
    want = np.full(len(ti), -1, np.int64)
    want[has_maj] = M.tp_key(tev[ti][has_maj], maj_tp[ti][has_maj])
    order = np.argsort(tk, kind="stable")
    lo = np.searchsorted(tk[order], want, "left")
    hi = np.searchsorted(tk[order], want, "right")
    nA = hi - lo
    at = np.repeat(ti, nA)
    am = kmi[tgt][order][np.repeat(lo, nA) + np.arange(nA.sum())
                         - np.repeat(np.cumsum(nA) - nA, nA)]
    pt_ = np.concatenate([cand_t, at])
    pm_ = np.concatenate([cand_m, am])
    pr_ = np.concatenate([cand_ratio, np.zeros(len(at))])
    # unique (track, module) pairs; a module can be both a candidate and a
    # target module, and a target module is ALWAYS evaluated
    pk = pt_ * (1 << 20) + pm_
    uq, first = np.unique(pk, return_index=True)
    pt_, pm_, pr_ = pt_[first], pm_[first], pr_[first]
    is_tgt = np.isin(uq, at * (1 << 20) + am)
    pj = project_with_cov(a[pt_], C[pt_], s0[pt_], geo, pm_)
    su = np.sqrt(np.clip(pj["S"][:, 0, 0], 0, None))
    sv = np.sqrt(np.clip(pj["S"][:, 1, 1], 0, None))
    overlap = pj["ok"] & (np.abs(pj["u"]) <= geo["hw"][pm_] + nsig * su) \
        & (np.abs(pj["v"]) <= geo["hl"][pm_] + nsig * sv)
    lands = pj["ok"] & (np.abs(pj["u"]) <= geo["hw"][pm_]) & (np.abs(pj["v"]) <= geo["hl"][pm_])
    bound_ratio = float(pr_[overlap].max()) if overlap.any() else 0.0
    # surface projection = one track projected to one surface: lands if the
    # helix point is inside at least one module
    lp_lands = np.bincount(pt_[lands], minlength=ntrk) > 0
    # covered: the ellipse overlaps at least one module (it can do so with
    # the helix point itself in a gap between modules)
    lp_cov = np.bincount(pt_[overlap], minlength=ntrk) > 0
    # clusters on each kept (track, module): join on (event, detId)
    keep = pj["ok"] & (overlap | is_tgt)
    qi = np.flatnonzero(keep)
    qkey = tev[pt_[qi]].astype(np.int64) * (1 << 32) + geo["detId"][pm_[qi]]
    lo = np.searchsorted(kkey[ko], qkey, "left")
    hi = np.searchsorted(kkey[ko], qkey, "right")
    nq = hi - lo
    pq = np.repeat(qi, nq)                             # pair index per (pair, cluster)
    ci = ko[np.repeat(lo, nq) + np.arange(nq.sum()) - np.repeat(np.cumsum(nq) - nq, nq)]
    du = K["localX"][ci] - pj["u"][pq]
    dv = K["localY"][ci] - pj["v"][pq]
    Suv = pj["S"][pq][:, :2, :2]
    det = Suv[:, 0, 0] * Suv[:, 1, 1] - Suv[:, 0, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = (Suv[:, 1, 1] * du * du - 2 * Suv[:, 0, 1] * du * dv
              + Suv[:, 0, 0] * dv * dv) / det
    m2 = np.where(det > 0, m2, np.inf)
    inside = m2 < nsig * nsig
    t_of = pt_[pq]
    ctp = K["tpIdx"][ci].astype(np.int64)
    isA = ctp == maj_tp[t_of]
    cm = SPN.tp_matched(ctp)
    ck = np.full(len(ctp), -1, np.int64)
    ck[cm] = M.tp_key(t_of[cm], ctp[cm])
    isB1 = ~isA & cm & np.isin(ck, foreign_keys)
    cat = np.where(isA, np.where(inside, 0, 1),
                   np.where(~inside, -1, np.where(isB1, 2, np.where(cm, 3, 4))))
    sel = cat >= 0
    hasA = K["hasAlpha"][ci] > 0
    hasB = K["hasBeta"][ci] > 0
    dcA = np.where(hasA, K["localCotAlpha"][ci] - pj["cotA"][pq], np.nan)
    dcB = np.where(hasB, K["localCotBeta"][ci] - pj["cotB"][pq], np.nan)
    rows_out = {
        "surface": np.full(sel.sum(), S, np.int8), "cls": cls[t_of][sel].astype(np.int8),
        "cat": cat[sel].astype(np.int8), "du": du[sel].astype(np.float32),
        "dv": dv[sel].astype(np.float32), "d": np.sqrt(m2[sel]).astype(np.float32),
        "dcotA": dcA[sel].astype(np.float32), "dcotB": dcB[sel].astype(np.float32),
        "pt": ctx["T"]["pt"][t_of][sel].astype(np.float32)}
    # per surface projection bookkeeping: lands?, target exists?, target inside?
    a_any = np.bincount(t_of[isA], minlength=ntrk) > 0
    a_in = np.bincount(t_of[isA & inside], minlength=ntrk) > 0
    a_proj_fail = np.zeros(ntrk, bool)
    a_proj_fail[at] = True
    a_proj_fail &= ~a_any                               # target module never reached
    nB = np.zeros((3, ntrk), np.int64)
    for b in range(3):
        nB[b] = np.bincount(t_of[cat == 2 + b], minlength=ntrk)
    nl = np.maximum(np.bincount(pt_[lands], minlength=ntrk)[ti], 1)
    wmean = lambda w: np.bincount(pt_[lands], weights=w, minlength=ntrk)[ti] / nl
    proj_out = {
        "surface": np.full(len(ti), S, np.int8), "cls": cls[ti].astype(np.int8),
        "lands": lp_lands[ti], "covered": lp_cov[ti], "a_exists": (nA > 0), "a_reached": a_any[ti],
        "a_in": a_in[ti], "a_unprojectable": a_proj_fail[ti],
        "nB1": nB[0][ti], "nB2": nB[1][ti], "nB3": nB[2][ti],
        "su": wmean(su[lands]), "sv": wmean(sv[lands]),
        "sCotA": wmean(np.sqrt(np.clip(pj["S"][lands, 2, 2], 0, None))),
        "sCotB": wmean(np.sqrt(np.clip(pj["S"][lands, 3, 3], 0, None)))}
    # candidate-level view for a refit: every cluster that is inside the
    # ellipse or is a target, with the projection it was measured against
    cq = inside | isA
    cand = {"t": t_of[cq], "ci": ci[cq], "pair": pq[cq], "cat": cat[cq], "inside": inside[cq],
            "cotA_pred": pj["cotA"][pq][cq], "cotB_pred": pj["cotB"][pq][cq],
            "isA": isA[cq], "du": du[cq], "dv": dv[cq], "dcA": dcA[cq], "dcB": dcB[cq],
            "m2": m2[cq], "hasA": hasA[cq], "hasB": hasB[cq]}
    return {"rows": rows_out, "proj": proj_out, "bound_ratio": bound_ratio,
            "ti": ti, "a_in_track": a_in, "cand": cand, "pj": pj, "pm": pm_, "pt": pt_}


def process_chunk(A, geo, nsig=4.0, with_refit=False, refit=REFIT):
    ctx = prepare_chunk(A, geo)
    cls = ctx["cls"]
    res = {"n_tracks_by_class": np.bincount(cls + 4, minlength=10),
           "bound_ratio_max": 0.0, "rows": [], "proj": []}
    use = cls >= 0
    for S in geo["surfaces"]:
        ev = evaluate_surface(ctx, ctx["a"], ctx["C"], S, geo, nsig, use)
        if ev is None:
            continue
        res["bound_ratio_max"] = max(res["bound_ratio_max"], ev["bound_ratio"])
        res["rows"].append(ev["rows"])
        res["proj"].append(ev["proj"])
    if with_refit:
        res["validation"] = validate_against_refit(A, ctx["a"], ctx["C"], geo, refit)
    return res


def validate_against_refit(A, a, C, geo, refit=REFIT):
    """Our projection vs the producer's own OT-track projection, same track, same module.

    The refit's projOT* is the UNMODIFIED OT-track helix projected by the CMSSW
    projector, recorded wherever the OT track and the updated state share a module.
    On those crossings the two implementations must agree on position, angle
    and cone width; any disagreement is a helix, frame or Jacobian bug here.
    """
    H = _flat(A, refit, REFIT_COLS)
    ntrk_ev = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    off = np.concatenate([[0], np.cumsum(ntrk_ev)])
    has = H["projOTLocalX"] > SENTINEL
    gt = off[H["event"][has]] + H["trackIdx"][has].astype(np.int64)
    mi = module_index(geo, H["detId"][has])
    ok = mi >= 0
    gt, mi = gt[ok], mi[ok]
    surf = geo["surface"][mi]
    s0 = np.full(len(gt), np.nan)
    for S in np.unique(surf):
        m = surf == S
        s0[m] = surface_s(a[gt[m]], geo, int(S))
    pj = project_with_cov(a[gt], C[gt], s0, geo, mi)
    h = {c: H[c][has][ok].astype(np.float64) for c in REFIT_COLS}
    g = pj["ok"]
    return {"surface": surf[g], "du": (pj["u"] - h["projOTLocalX"])[g],
            "dv": (pj["v"] - h["projOTLocalY"])[g],
            "dcotA": (pj["cotA"] - h["projOTCotAlpha"])[g],
            "dcotB": (pj["cotB"] - h["projOTCotBeta"])[g],
            "rsu": (np.sqrt(pj["S"][g, 0, 0]) / h["projOTSigX"][g]),
            "rsv": (np.sqrt(pj["S"][g, 1, 1]) / h["projOTSigY"][g]),
            "n_refit": int(has.sum()), "n_compared": int(g.sum())}


def run(files, geometry, nsig=4.0, chunk=10, nev=None, validate=True, log=print, refit=REFIT,
        surfaces=IT_SURFACES):
    """Stream the inputs; return concatenated per-cluster rows and per-projection rows
    for the `surfaces` (default IL1-IL4 and ID1-ID4)."""
    geo = load_geometry(geometry, surfaces)
    check_inputs(files[0], with_refit=validate, refit=refit)
    rows, proj, vals = [], [], []
    ncls = np.zeros(10, np.int64)
    bound_ratio = 0.0
    done = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], branches(validate, files[0], refit=refit),
                            step_size=chunk):
        if nev is not None and done >= nev:
            break
        if nev is not None and done + len(A) > nev:
            A = A[:nev - done]
        r = process_chunk(A, geo, nsig=nsig, with_refit=validate, refit=refit)
        rows += r["rows"]
        proj += r["proj"]
        ncls += r["n_tracks_by_class"]
        bound_ratio = max(bound_ratio, r["bound_ratio_max"])
        if validate:
            vals.append(r["validation"])
        done += len(A)
        log(f"    ot-projection: {done} events")
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]} if L else {}
    V = {}
    if vals:
        V = {k: np.concatenate([v[k] for v in vals]) for k in vals[0] if k not in ("n_refit", "n_compared")}
        V["n_refit"] = sum(v["n_refit"] for v in vals)
        V["n_compared"] = sum(v["n_compared"] for v in vals)
    names = ["covariance-absent", "unjoined", "no-owner", "tie"] + list(TRACK_CLASSES)
    return {"rows": cat(rows), "proj": cat(proj), "validation": V, "n_events": done,
            "tracks_by_class": dict(zip(names, ncls[:len(names)].tolist())),
            "bound_ratio_max": bound_ratio, "nsig": nsig, "bound_slack_cm": BOUND_SLACK_CM,
            "surfaces": list(geo["surfaces"])}


def param_pulls(files, nev=None):
    """(track - matched TP) / sqrt(fit variance) for the five helix parameters,
    genuine tracks with tp_pt > 2 GeV. Robust width = 1.4826 MAD; RMS alongside.

    Measures how optimistic the OT covariance is on THESE inputs, so the
    "4 sigma" of the ellipse can be read in true sigma.
    """
    cols = [f"L1TTrack_{c}" for c in list(HPAR) + COV_COLS
            + ["genuine", "tp_pt", "tp_phi", "tp_tanL", "tp_z0", "tp_d0", "tp_charge", "tp_vx", "tp_vy"]]
    A = uproot.concatenate([f"{f}:Events" for f in files], cols)
    if nev is not None:
        A = A[:nev]
    f = {c[9:]: ak.to_numpy(ak.flatten(A[c])).astype(np.float64) for c in cols}
    m = (f["genuine"] > 0) & (f["tp_pt"] > 2.0)
    tru = {"rInv": f["tp_charge"] * KPT_CMSSW / f["tp_pt"], "phi": tp_phi0(f),
           "tanL": f["tp_tanL"], "z0": f["tp_z0"], "d0": f["tp_d0"]}
    out = {}
    for p in HPAR:
        d = f[p][m] - tru[p][m]
        if p == "phi":
            d = np.angle(np.exp(1j * d))
        pu = d / np.sqrt(f[f"cov_{p}_{p}"][m])
        pu = pu[np.isfinite(pu)]
        med = float(np.median(pu))
        out[p] = {"median": med, "mad": float(1.4826 * np.median(np.abs(pu - med))),
                  "rms": float(np.std(pu)), "n": int(len(pu))}
    return out
