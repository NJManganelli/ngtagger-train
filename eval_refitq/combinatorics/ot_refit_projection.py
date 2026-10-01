"""Refit-update projection: the OT-only projection, updated surface by surface.

STUDY 19 projects the OT fit to every surface independently. A real refit does
not: it visits the active surfaces outside-in, picks a "best" cluster at each,
updates the track on it, and only THEN projects to the next surface. This module
does exactly that, so the target (A) and background (B1/B2/B3) populations can
be compared under the projection the refit actually uses.

  1. Start from the OT track: helix + fit covariance (L1TTrack_cov_*).
  2. Order the IT surfaces along the OT helix's PATH (surface_path), exactly as
     L1SmartPixelsTrackProducer (seedlessRefit) builds drPath: every TBPX layer
     (IL1-IL4) at its cylinder arc length (a layer the helix does not reach is
     placed first if inside |d0|, else last), and each disc ID1-ID4 whose mean
     plane the helix meets within the disc's radial extent (+- 1 cm). A barrel
     track gives IL1..IL4; a forward one e.g. IL1, ID1, ID2, ID3, ID4.
  3. Visit the ACTIVE surfaces of that path OUTSIDE-IN (the mask's character k
     activates ILk AND IDk), at most maxKFUpdates (8) updates per track:
       a. add multiple-scattering process noise Q for the material crossed
          since the last surface that constrained the track (none before the
          first update, so the first visited surface is IDENTICAL to study 19);
       b. project onto every module the 4-sigma ellipse overlaps and label
          every cluster there -- ot_projection.evaluate_surface, the same code
          study 19 runs, so the two are comparable by construction;
       c. choose the best candidate inside the ellipse with the variant's
          ranking metric (below);
       d. update the state on it with sequential scalar Kalman updates in
          local x, y, then cotAlpha, cotBeta (when the cluster has them and
          use_angles asks for them: "alphaBeta", the producer's setting, or
          "alpha").

THE UPDATE MIRRORS L1SmartPixelsTrackProducer (seedlessRefit) term for term:
  * measurement h = (u, v, cotAlpha, cotBeta) on the chosen cluster's module,
    H = its numerical Jacobian w.r.t. (rInv, phi, tanL, z0, d0); a column with
    any |H| > JACOBIAN_MAX_ABS (or non-finite) is zeroed;
  * R = the cluster's own errors (CPE sigX, sigY; sensor sigAlpha, sigBeta);
  * sequential scalar updates, linearised about the pre-update state:
    r = m - h0 - H (a - a_lin), S = H C H^T + sigma^2, K = C H^T / S,
    a += K r, C -= K (C H^T)^T; a scalar with r^2/S > CHI2_UPDATE_GATE is
    skipped (the producer's numerical-pathology gate);
  * no update where the predicted |cotAlpha| or |cotBeta| exceeds
    PRED_ANGLE_MAX_ABS (the producer invalidates such a crossing);
  * Q (add_process_noise), the producer's ONE physical material rule: at every
    visited active surface after the first update, theta0^2 * nCross * (J_T J_T^T
    + J_L J_L^T) with theta0 = MULT_SCATT_TERM / p (projected space angle),
    nCross = max(1, |nTraversed(this surface) - nTraversed(last constraint)|)
    counting the surfaces the OT helix TRAVERSES (layers within their |z|
    extent, discs within their radial extent, both +- 1 cm), the kink at the last
    constraint's transverse radius r_s: J_T = (0, sec, 0, 0, +r_s sec),
    J_L = (0, 0, sec^2, -r_s sec^2, 0). The same Jacobian holds for a kink on a
    disc (it depends only on r_s), and a disc is charged one theta0 like a
    layer. NB as in the producer, Q is added at EVERY visited surface after the
    first update, also where the crossing then fails or the window is empty, so
    the material since the last constraint is charged again at the next one.
  Checked against CMSSW by REPLAY (replay_against_refit): the producer's own
  selections, its visit sequence and its useAngles must reproduce its recorded
  running projection projLocalX/Y, projSigX/Y, projCotAlpha/Beta on every
  surface, and its box windows + selection metric (SELECTION closure) must
  re-derive the cluster it selected, the window multiplicity and truncation.

RANKING METRICS (which in-ellipse cluster is "best"):
  innovation  d^2 = r^T (J C J^T + R)^-1 r over (du, dv, dcotA, dcotB): the
              track's projected uncertainty PLUS the cluster's errors, with all
              correlations. Nothing tuned. A missing angle term is dropped from
              the quadratic form and replaced by its expectation, 1.
  producer    the seedlessRefit selection as it is today:
              (du/sigX)^2 + (dv/sigY)^2 + (dcotA/sigA)^2 + (dcotB/sigB)^2, every
              term over the CLUSTER'S error only; a missing angle adds 0, as
              in the producer. (Its window is a static box and one module; here
              the candidates are the ellipse, the same for every variant.)
  tuned       the same terms with the weights the chi2 scan (omnibus study 6)
              found best, (w_x, w_y, w_alpha, w_beta) = (1, 2, 16384, 16384),
              missing angle = 1 (the scan's convention). NB those weights were
              fitted through the pre-Q covariance and are STALE; the huge angle
              weight mostly undoes dividing a ~100 um projection residual by a
              ~3 um CPE error.
  oracle      the truth-best case: the majority-owner TP's cluster (A-in) if
              one is inside the ellipse (lowest innovation d^2 among several),
              otherwise no update. Not deployable; the reference the others
              are measured against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from ngtagger.data import spix_nano as SPN

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ot_projection as OP   # noqa: E402

# producer defaults (L1Trigger/Phase3SmartPixels/python/customizeSmartPixels_cff.py)
MULT_SCATT_TERM = 0.00075      # rad * GeV, TMTT KalmanMultiScattTerm
CHI2_UPDATE_GATE = 2.0e6
JACOBIAN_MAX_ABS = 1.0e4
PRED_ANGLE_MAX_ABS = 12.0
TUNED_WEIGHTS = np.array([1.0, 2.0, 16384.0, 16384.0])
NEUTRAL = 1.0
VARIANTS = ("none", "innovation", "producer", "tuned", "oracle")
VARIANT_DEF = {
    "none": "no update: study 19's OT-only projection to every surface (baseline)",
    "innovation": "innovation chi2 r^T (J C J^T + R)^-1 r, track + cluster errors",
    "producer": "seedlessRefit today: sum (residual / cluster error)^2, unit weights",
    "tuned": "study-6 weights (1, 2, 16384, 16384) on the producer terms (stale)",
    "oracle": "truth: the majority-owner cluster when inside the ellipse",
}
USE_ANGLES = ("alpha", "alphaBeta")
# The producer configuration the replay and the selection closure need: the
# customizeSmartPixels defaults, overridden by the nano's ParameterSets when the
# caller reads them (omnibus_support.refit_producer_config).
PRODUCER_DEFAULTS = {
    "useAngles": "alphaBeta",
    "maxKFUpdates": 8,
    "maxHitsPerWindow": 8,
    "windowRPhi": (0.05, 0.17, 0.5, 0.9),      # IL1-IL4 module-local x half-widths [cm]
    "windowZ": (0.45, 0.35, 0.25, 0.2),        # IL1-IL4 module-local y (z)
    "windowDiscRPhi": (0.2, 0.2, 0.2, 0.2),    # ID1-ID4 module-local x (r-phi)
    "windowDiscR": (0.3, 0.3, 0.3, 0.3),       # ID1-ID4 module-local y (radius)
    "multScattTerm": MULT_SCATT_TERM,
    "applyProcessNoise": True,
    "layerOrder": "outsideIn",
}


def active_surfaces(mask):
    """The surfaces an activeSP mask instruments: character k = ILk AND IDk."""
    try:
        return SPN.mask_surfaces(mask)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def window(cfg, S):
    """(r-phi, z-or-radius) half-widths [cm] of the producer's box on surface S."""
    if SPN.is_disc_surface(S):
        k = S - SPN.DISC_SURFACE_OFFSET - 1
        return cfg["windowDiscRPhi"][k], cfg["windowDiscR"][k]
    return cfg["windowRPhi"][S - 1], cfg["windowZ"][S - 1]


# ---- the visit order ------------------------------------------------------------
def surface_path(a, geo, surfaces=None):
    """drPath of L1SmartPixelsTrackProducer for helices a (the OT tracks).

    The surfaces in PATH ORDER (transverse arc length from the POCA): every
    barrel layer (s from the cylinder; a layer not reached sorts first when its
    radius is inside |d0|, else last: key -1e9 / +1e9 + layer), and each disc
    whose plane the helix meets (s > 0, within half a turn) within its radial
    extent +- TRAVERSED_MARGIN_CM. Ties keep the insertion order layers 1-4 then
    discs 1-4 (std::stable_sort). nT = nTraversed: the number of TRAVERSED
    surfaces up to and including this one along the path (a layer counts only
    when |z| at its crossing is within its module extent + 1 cm), the material
    count Q uses.

    Returns {"surfaces": the codes considered, in insertion order; "codes":
    (n, ns) path-ordered codes, -1 for discs off the path; "nT": (n, ns)
    nTraversed per surface in insertion order; "in_path": (n, ns)}."""
    surfaces = list(surfaces if surfaces is not None else geo["surfaces"])
    surfaces = [s for s in surfaces if not SPN.is_disc_surface(s)] + \
               [s for s in surfaces if SPN.is_disc_surface(s)]
    n, ns = len(a), len(surfaces)
    key = np.full((n, ns), np.inf)
    inpath = np.zeros((n, ns), bool)
    trav = np.zeros((n, ns), bool)
    d0 = np.abs(a[:, 4])
    for j, S in enumerate(surfaces):
        s, crossed = OP.surface_path_s(a, geo, S)
        ok = np.isfinite(s)
        if SPN.is_disc_surface(S):
            inpath[:, j] = ok & crossed
            key[:, j] = np.where(inpath[:, j], s, np.inf)
        else:
            inpath[:, j] = True
            key[:, j] = np.where(ok, s, np.where(geo["R"][S] < d0, -1e9, 1e9) + S)
        trav[:, j] = crossed & inpath[:, j]
    order = np.argsort(key, axis=1, kind="stable")
    codes = np.where(np.take_along_axis(inpath, order, 1), np.asarray(surfaces, np.int64)[order], -1)
    ntr = np.cumsum(np.take_along_axis(trav, order, 1), axis=1)
    nT = np.empty((n, ns), np.int64)
    np.put_along_axis(nT, order, ntr, 1)
    return {"surfaces": surfaces, "codes": codes, "nT": nT, "in_path": inpath}


def visit_sequence(path, active, order="outsideIn"):
    """(n, k) surface codes each track visits, in visit order (outsideIn = the path
    from its outer end), the inactive and off-path ones removed, -1 padded."""
    c = path["codes"][:, ::-1] if order == "outsideIn" else path["codes"]
    keep = np.isin(c, list(active)) & (c >= 0)
    o = np.argsort(~keep, axis=1, kind="stable")
    seq = np.where(np.take_along_axis(keep, o, 1), np.take_along_axis(c, o, 1), -1)
    k = max(int(keep.sum(1).max(initial=0)), 1)
    return seq[:, :k]


def n_traversed(path, t, S):
    """nTraversed of surface S for tracks t."""
    return path["nT"][t, path["surfaces"].index(S)]


# ---- the calibrated covariance model (refit_calibration_fit.export_refit_calibration) --
# Differences from the producer (record: memory smartpixels-scattering-excess-mystery):
#   * SEED: each track's reported TTTrack covariance PLUS a correlated deficit per
#     (OT-track group, stub count, pT bin) = cov(TTTrack - truth at the POCA) - the IT/gap
#     kinks - the bin-mean reported covariance. The reported covariance is too small
#     (more so with more stubs: OT-internal scattering treated as independent) and its
#     rInv-phi0 correlation near-degenerate (+0.975 vs +0.5..0.85 true);
#   * MATERIAL: Highland (full momentum, path length) on the tkLayout Phase-2 planes x1
#     (validated fit-free against the physical IT scattering and the CMSSW D121 scan),
#     the azimuthal kink carrying sec(lambda); every plane physically crossed between two
#     visited layers is charged, updated or not. A ~1.37x variance excess the innovation
#     likelihood asks for is UNEXPLAINED and deliberately not modelled;
#   * R: the cluster size/angle lookup, not the published CPE/payload errors.
# BARREL ONLY: the v3 calibration was fitted on barrel-only refits; it has no disc
# material planes and no disc seed groups, so the calibrated model visits IL1-IL4 only.
def load_calibration(path):
    """The v3 calibration JSON: seed_deficit {"g,ns,b": 5x5}, pt_bins, stub_min/max,
    floor_groups [[name, disk, |tanL| lo, hi], ...], planes [[r_cm or "ILj", x/X0,
    what], ...], ot_inner_r_cm, material_scale, R_tables."""
    import json
    with open(path) as fh:
        cal = json.load(fh)
    if cal.get("format") != "v3":
        raise SystemExit(f"{path}: not a v3 refit calibration (refit_calibration_fit.export_refit_calibration)")
    cal["seed_deficit"] = {tuple(int(x) for x in k.split(",")): np.asarray(v, np.float64)
                           for k, v in cal["seed_deficit"].items()}
    return cal


def seed_cov(cal, C, tanl, n_disk, n_stubs, pt):
    """Reported covariance + the per-bin deficit. Tracks outside every group keep C."""
    t = np.abs(tanl)
    ns = np.clip(n_stubs, cal["stub_min"], cal["stub_max"]) - cal["stub_min"]
    b = np.clip(np.searchsorted(cal["pt_bins"], pt, side="right") - 1, 0, len(cal["pt_bins"]) - 2)
    out = C.copy()
    for g, (_, dk, lo, hi) in enumerate(cal["floor_groups"]):
        mg = ((n_disk > 0) == bool(dk)) & (t >= lo) & (t < hi)
        for key, dC in cal["seed_deficit"].items():
            if key[0] == g:
                m = mg & (ns == key[1]) & (b == key[2])
                out[m] += dC
    return out


def _highland_theta2(xx0, p, tanl):
    X = np.maximum(xx0 * np.sqrt(1.0 + tanl ** 2), 1e-12)
    th = 0.0136 / p * np.sqrt(X) * (1.0 + 0.038 * np.log(X))
    return np.clip(th, 0, None) ** 2


def add_material(a, C, t, L, Lv, geo, cal, kpt):
    """Kinks for tracks t on every plane crossed between the previously visited layer
    Lv (5 = the OT) and layer L: radius in (R_L, R_Lv], R_5 = the OT's inner edge."""
    if not len(t):
        return
    tanl = a[t, 2]
    pt = kpt / np.maximum(np.abs(a[t, 0]), 1e-12)
    p = pt * np.sqrt(1.0 + tanl ** 2)
    sec2 = 1.0 + tanl ** 2
    sec = np.sqrt(sec2)
    r_prev = np.full(len(t), float(cal["ot_inner_r_cm"]))
    for j in (1, 2, 3, 4):
        r_prev[Lv[t] == j] = geo["R"][j]
    for r_spec, xx0, _ in cal["planes"]:
        r_s = geo["R"][int(r_spec[2:])] if isinstance(r_spec, str) else float(r_spec)
        m = (geo["R"][L] < r_s) & (r_s <= r_prev + 1e-6)
        if not m.any():
            continue
        th2 = np.where(m, _highland_theta2(cal["material_scale"] * xx0, p, tanl), 0.0)
        # +r_s: the producer's -r_s moves the crossing AT the kink (refit_calibration_fit.kink_cov)
        jT = np.zeros((len(t), 5)); jT[:, 1] = sec; jT[:, 4] = +r_s * sec
        jL = np.zeros((len(t), 5)); jL[:, 2] = sec2; jL[:, 3] = -r_s * sec2
        C[t] += th2[:, None, None] * (jT[:, :, None] * jT[:, None, :] + jL[:, :, None] * jL[:, None, :])


def pt_constant(T):
    """c * B(0) / 100, read off the tracks themselves: pT = k / |rInv|."""
    k = np.asarray(T["pt"], np.float64) * np.abs(np.asarray(T["rInv"], np.float64))
    return float(np.median(k[np.isfinite(k) & (k > 0)]))


def add_process_noise(a, C, t, n_cur, n_prev, prev_r, kpt, mult=MULT_SCATT_TERM):
    """The PRODUCER'S Q for tracks t (see the module docstring): n_cur / n_prev the
    nTraversed of the surface being predicted / of the last constraint, prev_r the
    transverse radius of the last constraint's crossing (the kink)."""
    if not len(t):
        return
    pt = kpt / np.maximum(np.abs(a[t, 0]), 1e-12)
    ncross = np.maximum(1, np.abs(n_cur - n_prev))
    sec2 = 1.0 + a[t, 2] ** 2
    sec = np.sqrt(sec2)
    var = np.where(pt > 1e-3, (mult / (pt * sec)) ** 2 * ncross, 0.0)
    jT = np.zeros((len(t), 5)); jT[:, 1] = sec; jT[:, 4] = +prev_r * sec
    jL = np.zeros((len(t), 5)); jL[:, 2] = sec2; jL[:, 3] = -prev_r * sec2
    C[t] += var[:, None, None] * (jT[:, :, None] * jT[:, None, :] + jL[:, :, None] * jL[:, None, :])


def kalman_update(a, C, t, J, h0, meas, sig, use_dim):
    """Sequential scalar updates for tracks t (one measurement each).

    J (n,4,5), h0/meas/sig (n,4), use_dim (n,4) bool. Returns the number of
    scalar updates skipped by the gate.
    """
    J = J.copy()
    bad = ~np.isfinite(J).all(1) | (np.abs(np.nan_to_num(J, nan=np.inf)) > JACOBIAN_MAX_ABS).any(1)
    J[np.repeat(bad[:, None, :], 4, 1)] = 0.0          # zero whole parameter columns
    a_lin = a[t].copy()
    gated = 0
    for k in range(4):
        H = J[:, k, :]
        Ct = C[t]
        v = np.einsum("nij,nj->ni", Ct, H)
        S = (H * v).sum(1) + sig[:, k] ** 2
        r = meas[:, k] - h0[:, k] - (H * (a[t] - a_lin)).sum(1)
        ok = use_dim[:, k] & (S > 0) & np.isfinite(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(ok, r * r / S, 0.0)
        g = ok & (chi2 > CHI2_UPDATE_GATE)
        gated += int(g.sum())
        ok &= ~g
        if not ok.any():
            continue
        tt, vv, SS, rr = t[ok], v[ok], S[ok], r[ok]
        Kg = vv / SS[:, None]
        a[tt] += Kg * rr[:, None]
        C[tt] -= Kg[:, :, None] * vv[:, None, :]
    return gated


def rank(variant, c, pj):
    """Ranking value per candidate (lower is better) for the in-ellipse ones."""
    p = c["pair"]
    r = np.stack([c["du"], c["dv"], c["dcA"], c["dcB"]], 1)
    sig = np.stack([c["sigX"], c["sigY"], c["sigA"], c["sigB"]], 1)
    have = np.isfinite(r) & (sig > 0) & c["use_dim"]
    if variant in ("innovation", "oracle"):
        Sm = pj["S"][p].copy()
        Sm[:, np.arange(4), np.arange(4)] += np.where(have, sig, 1.0) ** 2
        # drop a missing dimension from the quadratic form: decouple it, unit
        # variance, zero residual -- then add its expectation back
        miss = ~have
        Sm[np.repeat(miss[:, :, None], 4, 2)] = 0.0
        Sm[np.repeat(miss[:, None, :], 4, 1)] = 0.0
        Sm[:, np.arange(4), np.arange(4)] = np.where(miss, 1.0, Sm[:, np.arange(4), np.arange(4)])
        rz = np.where(have, r, 0.0)
        with np.errstate(invalid="ignore"):
            x = np.linalg.solve(Sm, rz[:, :, None])[:, :, 0]
        return (rz * x).sum(1) + NEUTRAL * miss.sum(1)
    q = np.where(have, (np.where(have, r, 0.0) / np.where(have, sig, 1.0)) ** 2,
                 0.0 if variant == "producer" else NEUTRAL)
    w = np.ones(4) if variant == "producer" else TUNED_WEIGHTS
    return (q * w).sum(1)


def choose(variant, c, pj):
    """Index into c of each track's chosen candidate, one per track (or none)."""
    elig = c["inside"] & (c["isA"] if variant == "oracle" else True)
    idx = np.flatnonzero(elig)
    if not len(idx):
        return idx
    val = rank(variant, {k: v[idx] for k, v in c.items()}, pj)
    val = np.where(np.isfinite(val), val, np.inf)
    o = np.lexsort((val, c["t"][idx]))
    first = np.r_[True, c["t"][idx][o][1:] != c["t"][idx][o][:-1]]
    return idx[o][first]


def _candidates(ev, K, use_angles, cal=None):
    c = dict(ev["cand"])
    ci = c["ci"]
    for src, dst in (("sigX", "sigX"), ("sigY", "sigY"), ("sigAlpha", "sigA"), ("sigBeta", "sigB")):
        c[dst] = K[src][ci].astype(np.float64)
    if cal is not None:
        import refit_calibration_fit as RCF
        pub = np.stack([c["sigX"], c["sigY"], c["sigA"], c["sigB"]], 1)
        s = RCF.apply_R(cal["R_tables"], K["sizeX"][ci], K["sizeY"][ci],
                        c["cotA_pred"], c["cotB_pred"], pub)
        c["sigX"], c["sigY"], c["sigA"], c["sigB"] = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    ua = use_angles in ("alpha", "alphaBeta")
    ub = use_angles == "alphaBeta"
    c["use_dim"] = np.stack([np.ones(len(ci), bool), np.ones(len(ci), bool),
                             c["hasA"] & ua, c["hasB"] & ub], 1)
    return c


def update_on(a, C, sel_c, c, ev, K, geo, state, n_trav):
    """Update the tracks of the chosen candidates; bookkeeping into state
    (n_trav: nTraversed of this surface per track, for the next Q)."""
    if not len(sel_c):
        return 0
    t = c["t"][sel_c]
    p = c["pair"][sel_c]
    pj, pm = ev["pj"], ev["pm"]
    h0 = np.stack([pj["u"][p], pj["v"][p], pj["cotA"][p], pj["cotB"][p]], 1)
    ok = (np.abs(h0[:, 2]) <= PRED_ANGLE_MAX_ABS) & (np.abs(h0[:, 3]) <= PRED_ANGLE_MAX_ABS)
    t, p, h0, sel_c = t[ok], p[ok], h0[ok], sel_c[ok]
    ci = c["ci"][sel_c]
    meas = np.stack([K["localX"][ci], K["localY"][ci], K["localCotAlpha"][ci], K["localCotBeta"][ci]], 1)
    sig = np.stack([c["sigX"][sel_c], c["sigY"][sel_c], c["sigA"][sel_c], c["sigB"][sel_c]], 1)
    gated = kalman_update(a, C, t, pj["J"][p], h0, meas.astype(np.float64), sig, c["use_dim"][sel_c])
    g = geo["origin"][pm[p]] + pj["u"][p][:, None] * geo["ex"][pm[p]] + pj["v"][p][:, None] * geo["ey"][pm[p]]
    state["prevR"][t] = np.hypot(g[:, 0], g[:, 1])
    state["nUpd"][t] += 1
    state["prevNT"][t] = n_trav[t]
    return gated


def run_chunk(A, geo, mask, use_angles, variants, nsig, replay, cal=None, refit=OP.REFIT,
              cfg=PRODUCER_DEFAULTS):
    ctx = OP.prepare_chunk(A, geo)
    K, cls, ntrk = ctx["K"], ctx["cls"], ctx["ntrk"]
    use = cls >= 0
    kpt = pt_constant(ctx["T"])
    if cal is not None:                                   # calibrated seed, once
        C0 = seed_cov(cal, ctx["C"], ctx["a"][:, 2], ctx["n_disk_stubs"], ctx["T"]["nStubs"],
                      kpt / np.maximum(np.abs(ctx["a"][:, 0]), 1e-12))
    else:
        C0 = ctx["C"]
    active = [s for s in active_surfaces(mask) if s in geo["surfaces"]
              and (cal is None or not SPN.is_disc_surface(s))]
    # the visit order is fixed by the OT helix (the producer builds drPath once)
    path = surface_path(ctx["a"], geo)
    seq = visit_sequence(path, active, cfg["layerOrder"])
    out = {"n_tracks_by_class": np.bincount(cls + 4, minlength=10), "by_variant": {}}
    first_ev = {}                                         # step 0: the OT state, every variant
    for v in variants:
        a, C = ctx["a"].copy(), C0.copy()
        state = {"nUpd": np.zeros(ntrk, np.int64), "prevR": np.full(ntrk, -1.0),
                 "prevNT": np.zeros(ntrk, np.int64)}
        Lv = np.full(ntrk, 5, np.int64)                      # last VISITED layer (5 = OT)
        rows, proj, selrec, gated = [], [], [], 0
        for j in range(seq.shape[1]):
            for S in np.unique(seq[:, j][seq[:, j] >= 0]):
                S = int(S)
                on = use & (seq[:, j] == S) & (state["nUpd"] < cfg["maxKFUpdates"])
                tq = np.flatnonzero(on)
                if not len(tq):
                    continue
                n_trav = np.zeros(ntrk, np.int64)
                n_trav[tq] = n_traversed(path, tq, S)
                if cal is not None:
                    add_material(a, C, tq, S, Lv, geo, cal, kpt)
                    Lv[tq] = S
                elif cfg["applyProcessNoise"]:
                    q = tq[(state["nUpd"][tq] > 0) & (state["prevR"][tq] > 0)]
                    add_process_noise(a, C, q, n_trav[q], state["prevNT"][q], state["prevR"][q], kpt,
                                      cfg["multScattTerm"])
                if j == 0:                                # identical state for every variant
                    if S not in first_ev:
                        first_ev[S] = OP.evaluate_surface(ctx, a, C, S, geo, nsig, on)
                    ev = first_ev[S]
                else:
                    ev = OP.evaluate_surface(ctx, a, C, S, geo, nsig, on)
                if ev is None:
                    continue
                ev["rows"]["step"] = np.full(len(ev["rows"]["cls"]), j, np.int8)
                ev["proj"]["step"] = np.full(len(ev["proj"]["cls"]), j, np.int8)
                rows.append(ev["rows"])
                proj.append(ev["proj"])
                c = _candidates(ev, K, use_angles, cal)
                sel = choose(v, c, ev["pj"]) if v != "none" else np.zeros(0, np.int64)
                # selection outcome per surface projection: which category was chosen
                chosen = np.full(ntrk, -1, np.int8)
                chosen[c["t"][sel]] = c["cat"][sel]
                ti = ev["ti"]
                selrec.append({"surface": np.full(len(ti), S, np.int8), "cls": cls[ti].astype(np.int8),
                               "chosen": chosen[ti], "a_in": ev["a_in_track"][ti],
                               "step": np.full(len(ti), j, np.int8)})
                gated += update_on(a, C, sel, c, ev, K, geo, state, n_trav)
        out["by_variant"][v] = {"rows": rows, "proj": proj, "sel": selrec, "gated": gated}
    if replay:
        out["replay"] = replay_against_refit(ctx, geo, kpt, refit, cfg, path, mask)
    return out


# ---- the replay against CMSSW ---------------------------------------------------
# hit-table columns the replay reads: the producer's crossing, its running
# projection, and the measurement it updated on (full-precision copies of the
# selected cluster, so the update does not inherit the cluster table's rounding)
REPLAY_COLS = ["selClusterIdx", "hitAccepted", "windowMult", "windowTruncated", "selHitClass",
               "hasAlpha", "hasBeta", "projLocalX", "projLocalY", "projSigX", "projSigY",
               "projCotAlpha", "projCotBeta", "recoLocalX", "recoLocalY", "sigX", "sigY",
               "recoCotAlpha", "recoCotBeta", "sigAlpha", "sigBeta", "tpLocalCotAlpha",
               "tpLocalCotBeta"]


def _row_steps(seq, gt, surf):
    """Visit step of every hit-table row in its track's emulated visit sequence
    (-1 when the producer recorded a crossing on a surface the emulated path
    does not visit)."""
    step = np.full(len(gt), -1, np.int64)
    for j in range(seq.shape[1]):
        m = (step < 0) & (seq[gt, j] == surf)
        step[m] = j
    return step


def selection_closure(H, K, coff, cfg):
    """The producer's window + selection, re-derived from the cluster table.

    For every recorded crossing (its module, its RECORDED running projection
    projLocal* / projCot*), the clusters on the module in readout order (the
    table's row order within a module) inside the box |x - x_pred| <= w_rphi,
    |y - y_pred| <= w_z|r of the surface, the first maxHitsPerWindow of them
    kept (truncated when another cluster of the module follows the last one
    kept), and the lowest selection chi2 (dx/ex)^2 + (dy/ey)^2 [+ (dcotA/sigA)^2
    if useAngles has alpha and the angle is usable] [+ beta likewise] wins (the
    first in readout order on a tie). Returns per row: windowMult, truncated and
    the selected cluster's table row (-1 none), from the cluster table's
    (reduced-precision) values."""
    use_a = cfg["useAngles"] in USE_ANGLES
    use_b = cfg["useAngles"] == "alphaBeta"
    ev = H["event"]
    det = H["detId"].astype(np.int64)
    ckey = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    o = np.argsort(ckey, kind="stable")                  # readout order kept within a module
    cs = ckey[o]
    xkey = ev.astype(np.int64) * (1 << 32) + det
    lo, hi = np.searchsorted(cs, xkey, "left"), np.searchsorted(cs, xkey, "right")
    n = hi - lo
    xi = np.repeat(np.arange(len(det)), n)
    ramp = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ci = o[np.repeat(lo, n) + ramp]
    surf = H["layer"].astype(np.int64)
    wr = np.zeros(len(det)); wz = np.zeros(len(det))
    for S in np.unique(surf):
        m = surf == S
        wr[m], wz[m] = window(cfg, int(S))
    dx = K["localX"][ci].astype(np.float64) - H["projLocalX"][xi]
    dy = K["localY"][ci].astype(np.float64) - H["projLocalY"][xi]
    inwin = (np.abs(dx) <= wr[xi]) & (np.abs(dy) <= wz[xi])
    # rank among the in-window ones of the crossing (readout order)
    cin = np.cumsum(inwin)
    base = np.repeat(np.r_[0, np.cumsum(np.bincount(xi, weights=inwin, minlength=len(det)))[:-1]], n)
    rk = cin - base - 1                                    # 0-based in-window rank (valid where inwin)
    nmax = cfg["maxHitsPerWindow"]
    kept = inwin & (rk < nmax)
    mult = np.bincount(xi[kept], minlength=len(det))
    # truncated: the nmax-th kept candidate is followed by another cluster of the module
    last = inwin & (rk == nmax - 1)
    pos_last = np.full(len(det), -1, np.int64)
    pos_last[xi[last]] = ramp[last]
    truncated = (pos_last >= 0) & (pos_last < n - 1)
    sel = (dx / np.maximum(K["sigX"][ci], 1e-12)) ** 2 + (dy / np.maximum(K["sigY"][ci], 1e-12)) ** 2
    if use_a:
        ok = K["hasAlpha"][ci] > 0
        sel = sel + np.where(ok, ((K["localCotAlpha"][ci] - H["projCotAlpha"][xi])
                                  / np.maximum(K["sigAlpha"][ci], 1e-12)) ** 2, 0.0)
    if use_b:
        ok = K["hasBeta"][ci] > 0
        sel = sel + np.where(ok, ((K["localCotBeta"][ci] - H["projCotBeta"][xi])
                                  / np.maximum(K["sigBeta"][ci], 1e-12)) ** 2, 0.0)
    sel = np.where(kept, sel, np.inf)
    oo = np.lexsort((ramp, sel, xi))                       # per crossing: lowest chi2, then readout order
    first = np.r_[True, xi[oo][1:] != xi[oo][:-1]] if len(oo) else np.zeros(0, bool)
    best = np.full(len(det), -1, np.int64)
    pick = oo[first]
    pick = pick[np.isfinite(sel[pick])]
    best[xi[pick]] = ci[pick]
    return {"mult": mult, "truncated": truncated, "best": best}


def replay_against_refit(ctx, geo, kpt, refit, cfg, path, mask):
    """Feed this update the producer's OWN selections (with its useAngles) along its
    visit sequence and compare against its recorded running projection.

    The visit sequence is rebuilt from the OT helix (surface_path, outside-in over
    the active surfaces of the producer's mask): at each step the producer's Q is added (after the first update,
    also where it recorded no crossing), the state is projected onto the module
    the producer recorded with the producer's Newton/Jacobian scheme, compared
    with projLocal*/projSig*/projCot*, and updated on the selected cluster's
    full-precision measurement. Also returned: the selection closure
    (selection_closure) and, on the accepted hits, the track-side predicted
    variances needed to split a pull into its track and cluster parts.
    """
    A, K = ctx["A"], ctx["K"]
    H = OP._flat(A, refit, OP.REFIT_COLS + REPLAY_COLS)
    ntrk_ev = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    toff = np.concatenate([[0], np.cumsum(ntrk_ev)])
    coff = np.concatenate([[0], np.cumsum(K["_n"])])     # K is the whole cluster table (OP._flat)
    gt = toff[H["event"]] + H["trackIdx"].astype(np.int64)
    surf = H["layer"].astype(np.int64)
    ntrk = ctx["ntrk"]
    active = [s for s in active_surfaces(mask) if s in geo["surfaces"]]
    seq = visit_sequence(path, active, cfg["layerOrder"])
    step = _row_steps(seq, gt, surf)
    # every recorded crossing must sit on the emulated path, and in its order
    off_path = step < 0
    order_bad = np.zeros(len(gt), bool)
    same = gt[1:] == gt[:-1]
    order_bad[1:] = same & ~off_path[1:] & ~off_path[:-1] & (step[1:] <= step[:-1])
    row_of = np.full((ntrk, seq.shape[1]), -1, np.int64)
    ok_rows = np.flatnonzero(~off_path)
    row_of[gt[ok_rows], step[ok_rows]] = ok_rows
    has_rows = np.zeros(ntrk, bool)
    has_rows[gt] = True                                  # the producer attempted a refit here
    use_dim_ab = (cfg["useAngles"] in USE_ANGLES, cfg["useAngles"] == "alphaBeta")

    a, C = ctx["a"].copy(), ctx["C"].copy()
    state = {"nUpd": np.zeros(ntrk, np.int64), "prevR": np.full(ntrk, -1.0),
             "prevNT": np.zeros(ntrk, np.int64)}
    cmp = {k: [] for k in ("surface", "row", "du", "dv", "dcotA", "dcotB", "rsu", "rsv")}
    dec = {k: [] for k in ("surface", "row", "su_trk", "sv_trk", "sa_trk", "sb_trk",
                           "u_pred", "v_pred", "a_pred", "b_pred")}
    n_bad_det = 0
    for j in range(seq.shape[1]):
        for S in np.unique(seq[:, j][seq[:, j] >= 0]):
            S = int(S)
            t = np.flatnonzero(has_rows & (seq[:, j] == S) & (state["nUpd"] < cfg["maxKFUpdates"]))
            if not len(t):
                continue
            if cfg["applyProcessNoise"]:
                q = t[(state["nUpd"][t] > 0) & (state["prevR"][t] > 0)]
                add_process_noise(a, C, q, n_traversed(path, q, S), state["prevNT"][q],
                                  state["prevR"][q], kpt, cfg["multScattTerm"])
            r = row_of[t, j]
            t, r = t[r >= 0], r[r >= 0]
            if not len(t):
                continue
            mi = OP.module_index(geo, H["detId"][r])
            good = mi >= 0
            t, r, mi = t[good], r[good], mi[good]
            if not len(t):
                continue
            s0 = OP.surface_s(a[t], geo, S)
            pj = OP.project_with_cov(a[t], C[t], s0, geo, mi, jac="cmssw", surface=S)  # the producer's H
            # the producer records its width through the guard-ZEROED H
            Jz = np.where(np.isfinite(pj["J"]).all(1, keepdims=True)
                          & (np.abs(np.nan_to_num(pj["J"], nan=np.inf)) <= JACOBIAN_MAX_ABS).all(1, keepdims=True),
                          np.nan_to_num(pj["J"]), 0.0)
            pj["S"] = np.einsum("nij,njk,nlk->nil", Jz, C[t], Jz)
            g = np.isfinite(pj["u"]) & (H["projLocalX"][r] > OP.SENTINEL) & (H["projSigX"][r] > 0) \
                & (H["projSigY"][r] > 0)
            cmp["surface"].append(np.full(g.sum(), S))
            cmp["row"].append(r[g])
            cmp["du"].append((pj["u"] - H["projLocalX"][r])[g])
            cmp["dv"].append((pj["v"] - H["projLocalY"][r])[g])
            cmp["dcotA"].append((pj["cotA"] - H["projCotAlpha"][r])[g])
            cmp["dcotB"].append((pj["cotB"] - H["projCotBeta"][r])[g])
            cmp["rsu"].append((np.sqrt(pj["S"][:, 0, 0]) / H["projSigX"][r])[g])
            cmp["rsv"].append((np.sqrt(pj["S"][:, 1, 1]) / H["projSigY"][r])[g])
            # update on the producer's selected cluster (its full-precision copy)
            acc = (H["hitAccepted"][r] > 0) & (H["selClusterIdx"][r] >= 0) & np.isfinite(pj["u"])
            cg = coff[H["event"][r]] + H["selClusterIdx"][r].astype(np.int64)
            cg = np.where(acc, cg, 0)
            n_bad_det += int((acc & (K["detId"][cg] != H["detId"][r])).sum())
            u = np.flatnonzero(acc)
            if not len(u):
                continue
            ru = r[u]
            dec["surface"].append(np.full(len(u), S))
            dec["row"].append(ru)
            for k_, i_ in (("su_trk", 0), ("sv_trk", 1), ("sa_trk", 2), ("sb_trk", 3)):
                dec[k_].append(np.sqrt(np.clip(pj["S"][u, i_, i_], 0, None)))
            for k_, q_ in (("u_pred", "u"), ("v_pred", "v"), ("a_pred", "cotA"), ("b_pred", "cotB")):
                dec[k_].append(pj[q_][u])
            h0 = np.stack([pj["u"][u], pj["v"][u], pj["cotA"][u], pj["cotB"][u]], 1)
            meas = np.stack([H["recoLocalX"][ru], H["recoLocalY"][ru], H["recoCotAlpha"][ru],
                             H["recoCotBeta"][ru]], 1).astype(np.float64)
            sig = np.stack([H["sigX"][ru], H["sigY"][ru], H["sigAlpha"][ru],
                            H["sigBeta"][ru]], 1).astype(np.float64)
            use_dim = np.stack([np.ones(len(u), bool), np.ones(len(u), bool),
                                (H["hasAlpha"][ru] > 0) & use_dim_ab[0],
                                (H["hasBeta"][ru] > 0) & use_dim_ab[1]], 1)
            kalman_update(a, C, t[u], pj["J"][u], h0, meas, sig, use_dim)
            gp = geo["origin"][mi[u]] + pj["u"][u][:, None] * geo["ex"][mi[u]] \
                + pj["v"][u][:, None] * geo["ey"][mi[u]]
            state["prevR"][t[u]] = np.hypot(gp[:, 0], gp[:, 1])
            state["nUpd"][t[u]] += 1
            state["prevNT"][t[u]] = n_traversed(path, t[u], S)
    outc = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cmp.items()}
    outc["n_bad_det"] = n_bad_det
    outc["decomposition"] = {k: np.concatenate(v) if v else np.zeros(0) for k, v in dec.items()}
    # the per-row record the decomposition and the closure tables are keyed on
    sc = selection_closure(H, K, coff, cfg)
    acc = (H["hitAccepted"] > 0) & (H["selClusterIdx"] >= 0)
    prod_best = np.where(acc, coff[H["event"]] + H["selClusterIdx"].astype(np.int64), -1)
    first = np.r_[True, gt[1:] != gt[:-1]] if len(gt) else np.zeros(0, bool)
    outc["rows"] = {
        "surface": surf.astype(np.int8), "off_path": off_path, "order_bad": order_bad,
        "first": first, "sigX_proj": H["projSigX"], "sigY_proj": H["projSigY"],
        "accepted": acc, "selHitClass": H["selHitClass"].astype(np.int8),
        "windowMult": H["windowMult"].astype(np.int16), "windowTruncated": H["windowTruncated"] > 0,
        "emu_mult": sc["mult"].astype(np.int16), "emu_truncated": sc["truncated"],
        "same_pick": sc["best"] == prod_best,
        "recoCotAlpha": H["recoCotAlpha"], "recoCotBeta": H["recoCotBeta"],
        "sigAlpha": H["sigAlpha"], "sigBeta": H["sigBeta"],
        "tpLocalCotAlpha": H["tpLocalCotAlpha"], "tpLocalCotBeta": H["tpLocalCotBeta"],
        "hasAlpha": H["hasAlpha"] > 0, "hasBeta": H["hasBeta"] > 0,
        "recoLocalX": H["recoLocalX"], "recoLocalY": H["recoLocalY"],
        "sigX": H["sigX"], "sigY": H["sigY"]}
    outc["mask"] = mask
    return outc


def run(files, geometry, mask="AAAA", use_angles="alphaBeta", variants=VARIANTS,
        nsig=4.0, chunk=10, nev=None, replay=True, log=print, calibration=None,
        refit=None, producer=None):
    """refit: the refit hit table the replay reads; default the coexist table of `mask`
    (the producer's selections are only meaningful for the build being replayed).
    producer: the producer configuration of that table (PRODUCER_DEFAULTS keys;
    omitted keys keep the defaults): the replay uses ITS useAngles, windows and
    update cap; `use_angles` is the emulated variants' choice."""
    if use_angles not in USE_ANGLES:
        raise SystemExit(f"use_angles {use_angles!r} not in {USE_ANGLES}")
    cfg = dict(PRODUCER_DEFAULTS, **(producer or {}))
    refit = refit or f"L1TSmartPixelsRefitHitSeedlessRefit{mask}"
    cal = load_calibration(calibration) if calibration else None
    geo = OP.load_geometry(geometry)
    OP.check_inputs(files[0], with_refit=True, refit=refit)
    extra = [f"{refit}_{c}" for c in REPLAY_COLS]
    acc = {v: {"rows": [], "proj": [], "sel": [], "gated": 0} for v in variants}
    rep = []
    ncls = np.zeros(10, np.int64)
    done = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], OP.branches(True, files[0], refit=refit) + extra,
                            step_size=chunk):
        if nev is not None and done >= nev:
            break
        if nev is not None and done + len(A) > nev:
            A = A[:nev - done]
        r = run_chunk(A, geo, mask, use_angles, variants, nsig, replay, cal, refit, cfg)
        ncls += r["n_tracks_by_class"]
        for v in variants:
            for k in ("rows", "proj", "sel"):
                acc[v][k] += r["by_variant"][v][k]
            acc[v]["gated"] += r["by_variant"][v]["gated"]
        if replay:
            rep.append(r["replay"])
        done += len(A)
        log(f"    refit-projection: {done} events")
    cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]} if L else {}
    names = ["covariance-absent", "unjoined", "no-owner", "tie"] + list(OP.TRACK_CLASSES)
    active = [s for s in active_surfaces(mask) if s in geo["surfaces"]
              and (cal is None or not SPN.is_disc_surface(s))]
    res = {"n_events": done, "mask": mask, "use_angles": use_angles, "surfaces": active,
           "calibration": calibration, "producer_config": cfg,
           "tracks_by_class": dict(zip(names, ncls[:len(names)].tolist())), "nsig": nsig,
           "variants": {v: {"rows": cat(acc[v]["rows"]), "proj": cat(acc[v]["proj"]),
                            "sel": cat(acc[v]["sel"]), "gated": acc[v]["gated"]} for v in variants}}
    if rep:
        res["replay"] = {
            **{k: np.concatenate([x[k] for x in rep]) for k in rep[0]
               if k not in ("n_bad_det", "decomposition", "rows", "mask")},
            "n_bad_det": sum(x["n_bad_det"] for x in rep),
            "decomposition": {k: np.concatenate([x["decomposition"][k] for x in rep])
                              for k in rep[0]["decomposition"]},
            "mask": rep[0]["mask"]}
        # rows: concatenated per chunk; the cmp/decomposition "row" indices are
        # chunk-local, so re-base them onto the concatenated rows
        base = np.cumsum([0] + [len(x["rows"]["surface"]) for x in rep[:-1]])
        res["replay"]["rows"] = {k: np.concatenate([x["rows"][k] for x in rep]) for k in rep[0]["rows"]}
        res["replay"]["row"] = np.concatenate([x["row"] + b for x, b in zip(rep, base)])
        res["replay"]["decomposition"]["row"] = np.concatenate(
            [x["decomposition"]["row"] + b for x, b in zip(rep, base)])
    return res
