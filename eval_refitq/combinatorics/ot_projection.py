"""OT-only track projection onto the inner-tracker barrel, independent of the refit.

WHAT THIS IS FOR. A refit starts from an OT-only track and has to decide which
inner-tracker clusters to include. Before any Kalman update, the only thing it
knows is the OT fit: its helix and its covariance. This module projects that
helix, with that covariance, onto EVERY barrel module of IL1-IL4 it could reach
within a k-sigma ellipse, and labels every cluster found there by who made it.
It deliberately does NOT use the refit's per-crossing tables:

  * the refit keeps ONE module per layer (the nearest containing one), so
    clusters on a neighbouring module inside the ellipse are never offered;
  * the refit records its seed projection only where the seed lands on the
    same module as the KF-UPDATED state, so under outsideIn IL1-IL3 are
    conditioned on the outer updates;
  * only the diagonal of the projected covariance is stored there, and
    grazing predictions are dropped by the refit's angle clamp.

Those tables are used here only as an independent cross-check of the helix,
frame and Jacobian (validate_against_refit).

INPUTS (nano + one JSON):
  L1TTrack               helix (rInv, phi, tanL, z0, d0) and its fit covariance
                         cov_<p>_<q> (L1TrackHelixCovTableProducer)
  L1TTrackStub           the track's stubs; L1TOTStub their truth (joined by
                         ot_oracle.join_stub_truth on the full stub identity)
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
from ngtagger.truth_helix import KPT_CMSSW, tp_phi0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ot_oracle import join_stub_truth   # noqa: E402

HPAR = ("rInv", "phi", "tanL", "z0", "d0")          # TTTrack::Hpar order
COV_COLS = [f"cov_{HPAR[i]}_{HPAR[j]}" for i in range(5) for j in range(i, 5)]
# Jacobian steps: the producer's own seed-cone steps (kEpsS), used here as
# central differences.
JAC_EPS = np.array([1e-6, 1e-5, 1e-5, 1e-3, 1e-3])
IT_LAYERS = (1, 2, 3, 4)
# Candidate modules per (track, layer): every module whose centre is within
#   BOUND_SIGMA_MARGIN * nsig * sqrt(sigma_u^2 + sigma_v^2)       (ellipse box corner)
#   + half-diagonal of the module
#   + BOUND_SLACK_CM * sqrt(1 + tanL^2)                           (cylinder -> plane)
# of the layer-cylinder crossing, sigmas taken on the nearest module. NOT
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

TRK_COLS = list(HPAR) + ["pt", "eta", "nStubs"] + COV_COLS
TST_COLS = ["trackIdx", "detId", "x", "y", "z", "bend", "layer", "isBarrel"]
OTS_COLS = ["detId", "x", "y", "z", "bend", "layer", "isBarrel", "tpIdx",
            "tpGenuine", "tpCombinatoric", "tpUnknown"]
CL_COLS = ["detId", "layer", "localX", "localY", "localCotAlpha",
           "localCotBeta", "hasAlpha", "hasBeta", "tpIdx",
           "sigX", "sigY", "sigAlpha", "sigBeta",      # errors: the refit's measurement R
           "sizeX", "sizeY"]                             # cluster size: calibrated R lookup
REFIT = "L1TSmartPixelsRefitHitDigiRefitAAAA"
REFIT_COLS = ["trackIdx", "detId", "layer", "projSeedLocalX", "projSeedLocalY",
              "projSeedCotAlpha", "projSeedCotBeta", "projSeedSigX", "projSeedSigY"]
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
# cluster categories on a layer projection
# --------------------------------------------------------------------------
# A   -- a cluster of the majority-owner TP anywhere on the layer: the TARGET.
#        Split A-in / A-out by the k-sigma ellipse; A-out residuals are taken on
#        the target's own module, so they are still frame-consistent.
# B1  -- inside the ellipse, owned by a TP that also owns a foreign stub of
#        this track: the trap a contaminated track sets for itself.
# B2  -- inside the ellipse, owned by any other TP.
# B3  -- inside the ellipse, no TP (tpIdx = -1): noise / unlinked.
CATEGORIES = ("A-in", "A-out", "B1", "B2", "B3")


def load_geometry(path, layers=IT_LAYERS):
    """Barrel modules of the requested layers, as arrays sorted by detId."""
    G = json.load(open(path))
    ms = [m for m in G["modules"] if m["subdet"] == "PXB" and m["layer"] in layers]
    ms.sort(key=lambda m: m["detId"])
    g = {"detId": np.array([m["detId"] for m in ms], np.int64),
         "layer": np.array([m["layer"] for m in ms], np.int64)}
    for k in ("origin", "ex", "ey", "ez"):
        g[k] = np.array([m[k] for m in ms], np.float64)
    g["hw"] = np.array([m["halfWidth"] for m in ms], np.float64)
    g["hl"] = np.array([m["halfLength"] for m in ms], np.float64)
    g["R"] = {L: float(np.hypot(*g["origin"][g["layer"] == L][:, :2].T).mean())
              for L in layers}
    g["by_layer"] = {L: np.flatnonzero(g["layer"] == L) for L in layers}
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


def crosslayer_module(a, L, geo):
    """The module SmartPixelsHelixProjector::crossLayer resolves for helices a
    on layer L: the 4 module centres nearest the mean-radius cylinder point,
    each re-solved onto its plane, kept if the on-plane point is inside the
    bounds, the one whose centre is nearest THAT point winning. -1 if none."""
    rows = geo["by_layer"][L]
    s0 = cylinder_s(a, geo["R"][L])
    pc, _ = helix_at(a, np.where(np.isfinite(s0), s0, 1.0))
    d2 = ((pc[:, None, :] - geo["origin"][rows][None, :, :]) ** 2).sum(-1)
    k = min(4, len(rows))
    near = rows[np.argpartition(d2, k - 1, axis=1)[:, :k]]
    best = np.full(len(a), -1, np.int64)
    bestd = np.full(len(a), np.inf)
    for c in range(k):
        mi = near[:, c]
        s, ok = _newton_cmssw(a, s0, geo["origin"][mi], geo["ez"][mi])
        p, _ = helix_at(a, s)
        rel = p - geo["origin"][mi]
        u, v = (rel * geo["ex"][mi]).sum(-1), (rel * geo["ey"][mi]).sum(-1)
        inside = ok & np.isfinite(s0) & (np.abs(u) <= geo["hw"][mi]) & (np.abs(v) <= geo["hl"][mi])
        dd = (rel ** 2).sum(-1)
        take = inside & (dd < bestd)
        best = np.where(take, mi, best)
        bestd = np.where(take, dd, bestd)
    return best


def project_to_modules(a, s0, geo, mi, newton="study"):
    """Project helices a[n] onto module rows mi[n], Newton from arc s0[n].

    Returns dict of u, v (local position, cm), cotA, cotB, s and ok, where ok
    means the helix reached the plane (converged, s > 0). u, v may lie OUTSIDE
    the module bounds; containment is the caller's question.
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
    rel = p - o
    lz = (d * geo["ez"][mi]).sum(-1)
    lz = np.where(np.abs(lz) > 1e-9, lz, 1e-9)
    return {"u": (rel * geo["ex"][mi]).sum(-1), "v": (rel * geo["ey"][mi]).sum(-1),
            "cotA": (d * geo["ex"][mi]).sum(-1) / lz,
            "cotB": (d * geo["ey"][mi]).sum(-1) / lz, "s": s, "ok": ok}


OUT_KEYS = ("u", "v", "cotA", "cotB")


def project_with_cov(a, C, s0, geo, mi, jac="central", R=None):
    """project_to_modules plus the 4x4 covariance of (u, v, cotA, cotB).

    jac="central": central differences (the default, more accurate).
    jac="cmssw": the producer's scheme exactly -- one-sided +eps falling back
    to -eps, each perturbed helix re-solved by crossLayer's own Newton loop
    started from ITS cylinder crossing and stopped at 1 um, and a perturbation
    that crossLayer would put on a DIFFERENT module is not usable (the
    producer's same-module guard; both sides unusable -> NaN column, which the
    update zeroes). R is the layer's number (1..4). Used only where CMSSW must
    be reproduced (replay).

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
                pp = project_to_modules(ap, cylinder_s(ap, geo["R"][R]), geo, mi, newton="cmssw")
                pp["ok"] = pp["ok"] & (crosslayer_module(ap, R, geo) == mi)
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
    J = np.full((len(mi), 4, 5), np.nan)
    for j in range(5):
        cols = []
        for sgn in (+1.0, -1.0):
            ap = a.copy()
            ap[:, j] += sgn * JAC_EPS[j]
            cols.append(project_to_modules(ap, nom["s"], geo, mi))
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
    n = ak.to_numpy(ak.num(A[f"{pre}_{cols[0]}"]))
    D = {c: ak.to_numpy(ak.flatten(A[f"{pre}_{c}"])) for c in cols}
    D["event"] = np.repeat(np.arange(len(n)), n)
    D["_n"] = n
    return D


def branches(with_refit):
    b = ([f"L1TTrack_{c}" for c in TRK_COLS] + [f"L1TTrackStub_{c}" for c in TST_COLS]
         + [f"L1TOTStub_{c}" for c in OTS_COLS]
         + [f"L1TSmartPixelsCluster_{c}" for c in CL_COLS])
    if with_refit:
        b += [f"{REFIT}_{c}" for c in REFIT_COLS]
    return b


def check_inputs(path, with_refit=False):
    with uproot.open(f"{path}:Events") as t:
        keys = set(t.keys())
    miss = [b for b in branches(with_refit) if b not in keys]
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
    tp, flag, hit = join_stub_truth(S, O)
    n_unj = np.bincount(gtrk, weights=~hit, minlength=ntrk).astype(int)
    gen = hit & (flag == 1) & (tp >= 0)
    key = gtrk[gen] * (1 << 24) + tp[gen]
    u, c = np.unique(key, return_counts=True)
    ut, utp = u >> 24, u & ((1 << 24) - 1)
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
    foreign_keys = np.unique(gtrk[fr] * (1 << 24) + tp[fr])
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
    S = _flat(A, "L1TTrackStub", TST_COLS)
    O = _flat(A, "L1TOTStub", OTS_COLS)
    K = _flat(A, "L1TSmartPixelsCluster", CL_COLS)
    a, C = track_state(T)
    cls, maj_tp, foreign_keys = classify_tracks(T, S, O)
    covok = np.trace(C, axis1=1, axis2=2) > 0
    cls = np.where(covok, cls, -4)                         # covariance absent
    kmi = module_index(geo, K["detId"])                    # IT barrel clusters only
    kev = K["event"]
    kkey = kev.astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    toff = np.concatenate([[0], np.cumsum(T["_n"])])
    gst = toff[S["event"]] + S["trackIdx"].astype(np.int64)
    n_disk = np.bincount(gst, weights=S["isBarrel"] == 0, minlength=len(T["pt"])).astype(np.int64)
    return {"A": A, "T": T, "K": K, "ntrk": len(T["pt"]), "a": a, "C": C, "n_disk_stubs": n_disk,
            "tev": T["event"], "cls": cls, "maj_tp": maj_tp,
            "foreign_keys": foreign_keys, "kmi": kmi, "kin": kmi >= 0, "kev": kev,
            "klay": geo["layer"][np.maximum(kmi, 0)], "kkey": kkey,
            "ko": np.argsort(kkey, kind="stable")}


def evaluate_layer(ctx, a, C, L, geo, nsig, use):
    """One layer, for the track STATE (a, C): candidate modules, the ellipse on
    each, and every cluster there labelled A-in / A-out / B1 / B2 / B3.

    Returns the study rows and per-projection bookkeeping (what study 19
    records), plus the candidate-level arrays a refit needs to choose a cluster
    and update on it (key "cand"). None if no track reaches the layer.
    """
    K, tev, ntrk = ctx["K"], ctx["tev"], ctx["ntrk"]
    maj_tp, foreign_keys, cls = ctx["maj_tp"], ctx["foreign_keys"], ctx["cls"]
    kmi, kin, kev, klay, kkey, ko = (ctx[k] for k in ("kmi", "kin", "kev", "klay", "kkey", "ko"))
    R = geo["R"][L]
    s0 = cylinder_s(a, R)
    reach = use & np.isfinite(s0)
    ti = np.flatnonzero(reach)
    if not len(ti):
        return None
    pc, _ = helix_at(a[ti], s0[ti])
    rows = geo["by_layer"][L]
    dist = np.sqrt(((pc[:, None, :] - geo["origin"][rows][None, :, :]) ** 2).sum(-1))
    # ellipse size from the nearest module; a failed projection there gets
    # the layer-wide worst case so the track is not silently narrowed
    m0 = rows[np.argmin(dist, 1)]
    p0 = project_with_cov(a[ti], C[ti], s0[ti], geo, m0)
    tr = np.nan_to_num(p0["S"][:, 0, 0]) + np.nan_to_num(p0["S"][:, 1, 1])
    r_ell = BOUND_SIGMA_MARGIN * nsig * np.sqrt(np.clip(tr, 0, None))
    r_ell = np.where(p0["ok"], r_ell, np.nanmax(np.where(p0["ok"], r_ell, np.nan)) if p0["ok"].any() else 10.0)
    hdiag = np.hypot(geo["hw"][rows], geo["hl"][rows])
    slack = BOUND_SLACK_CM * np.sqrt(1.0 + a[ti, 2] ** 2)
    bound = (r_ell + slack)[:, None] + hdiag[None, :]
    it_, im_ = np.nonzero(dist <= bound)
    cand_t, cand_m = ti[it_], rows[im_]
    cand_ratio = (dist / bound)[it_, im_]
    # target modules: every IT module on this layer holding a cluster of
    # the majority-owner TP, even if it is not among the candidates
    tgt = kin & (klay == L) & (K["tpIdx"] >= 0)
    tk = kev[tgt].astype(np.int64) * (1 << 24) + K["tpIdx"][tgt].astype(np.int64)
    want = tev[ti].astype(np.int64) * (1 << 24) + maj_tp[ti]
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
    # layer projection = one track projected to one layer: lands if the
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
    isB1 = ~isA & np.isin(t_of * (1 << 24) + ctp, foreign_keys)
    cat = np.where(isA, np.where(inside, 0, 1),
                   np.where(~inside, -1, np.where(isB1, 2, np.where(ctp >= 0, 3, 4))))
    sel = cat >= 0
    hasA = K["hasAlpha"][ci] > 0
    hasB = K["hasBeta"][ci] > 0
    dcA = np.where(hasA, K["localCotAlpha"][ci] - pj["cotA"][pq], np.nan)
    dcB = np.where(hasB, K["localCotBeta"][ci] - pj["cotB"][pq], np.nan)
    rows_out = {
        "layer": np.full(sel.sum(), L, np.int8), "cls": cls[t_of][sel].astype(np.int8),
        "cat": cat[sel].astype(np.int8), "du": du[sel].astype(np.float32),
        "dv": dv[sel].astype(np.float32), "d": np.sqrt(m2[sel]).astype(np.float32),
        "dcotA": dcA[sel].astype(np.float32), "dcotB": dcB[sel].astype(np.float32),
        "pt": ctx["T"]["pt"][t_of][sel].astype(np.float32)}
    # per layer projection bookkeeping: lands?, target exists?, target inside?
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
        "layer": np.full(len(ti), L, np.int8), "cls": cls[ti].astype(np.int8),
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


def process_chunk(A, geo, nsig=4.0, with_refit=False):
    ctx = prepare_chunk(A, geo)
    cls = ctx["cls"]
    res = {"n_tracks_by_class": np.bincount(cls + 4, minlength=10),
           "bound_ratio_max": 0.0, "rows": [], "proj": []}
    use = cls >= 0
    for L in IT_LAYERS:
        ev = evaluate_layer(ctx, ctx["a"], ctx["C"], L, geo, nsig, use)
        if ev is None:
            continue
        res["bound_ratio_max"] = max(res["bound_ratio_max"], ev["bound_ratio"])
        res["rows"].append(ev["rows"])
        res["proj"].append(ev["proj"])
    if with_refit:
        res["validation"] = validate_against_refit(A, ctx["a"], ctx["C"], geo)
    return res


def validate_against_refit(A, a, C, geo):
    """Our projection vs the producer's own seed projection, same track, same module.

    The refit's projSeed* is the UNMODIFIED seed helix projected by the CMSSW
    projector, recorded wherever the seed and the updated state share a module.
    On those crossings the two implementations must agree on position, angle
    and cone width; any disagreement is a helix, frame or Jacobian bug here.
    """
    H = _flat(A, REFIT, REFIT_COLS)
    ntrk_ev = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    off = np.concatenate([[0], np.cumsum(ntrk_ev)])
    has = H["projSeedLocalX"] > SENTINEL
    gt = off[H["event"][has]] + H["trackIdx"][has].astype(np.int64)
    mi = module_index(geo, H["detId"][has])
    ok = mi >= 0
    gt, mi = gt[ok], mi[ok]
    lay = geo["layer"][mi]
    s0 = cylinder_s(a[gt], np.array([geo["R"][L] for L in lay], np.float64))
    pj = project_with_cov(a[gt], C[gt], s0, geo, mi)
    h = {c: H[c][has][ok].astype(np.float64) for c in REFIT_COLS}
    g = pj["ok"]
    return {"layer": lay[g], "du": (pj["u"] - h["projSeedLocalX"])[g],
            "dv": (pj["v"] - h["projSeedLocalY"])[g],
            "dcotA": (pj["cotA"] - h["projSeedCotAlpha"])[g],
            "dcotB": (pj["cotB"] - h["projSeedCotBeta"])[g],
            "rsu": (np.sqrt(pj["S"][g, 0, 0]) / h["projSeedSigX"][g]),
            "rsv": (np.sqrt(pj["S"][g, 1, 1]) / h["projSeedSigY"][g]),
            "n_refit": int(has.sum()), "n_compared": int(g.sum())}


def run(files, geometry, nsig=4.0, chunk=10, nev=None, validate=True, log=print):
    """Stream the inputs; return concatenated per-cluster rows and per-projection rows."""
    geo = load_geometry(geometry)
    check_inputs(files[0], with_refit=validate)
    rows, proj, vals = [], [], []
    ncls = np.zeros(10, np.int64)
    bound_ratio = 0.0
    done = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], branches(validate),
                            step_size=chunk):
        if nev is not None and done >= nev:
            break
        if nev is not None and done + len(A) > nev:
            A = A[:nev - done]
        r = process_chunk(A, geo, nsig=nsig, with_refit=validate)
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
            "bound_ratio_max": bound_ratio, "nsig": nsig, "bound_slack_cm": BOUND_SLACK_CM}


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
