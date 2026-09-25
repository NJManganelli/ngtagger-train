"""Refit-update projection: the OT-only projection, updated layer by layer.

STUDY 19 projects the OT fit to every layer independently. A real refit does
not: it visits the active layers outside-in, picks a "best" cluster at each,
updates the track on it, and only THEN projects to the next layer. This module
does exactly that, so the target (A) and background (B1/B2/B3) populations can
be compared under the projection the refit actually uses.

  1. Start from the OT track: helix + fit covariance (L1TTrack_cov_*).
  2. For each ACTIVE layer, outermost first (AAAA: IL4, IL3, IL2, IL1; AIAI:
     IL3 then IL1 -- mask character i is layer i+1):
       a. add multiple-scattering process noise Q for the material crossed
          since the last layer that constrained the track (none before the
          first update, so the first active layer is IDENTICAL to study 19);
       b. project onto every module the 4-sigma ellipse overlaps and label
          every cluster there -- ot_projection.evaluate_layer, the same code
          study 19 runs, so the two are comparable by construction;
       c. choose the best candidate inside the ellipse with the variant's
          ranking metric (below);
       d. update the state on it with sequential scalar Kalman updates in
          local x, y, then cotAlpha, cotBeta (when the cluster has them and
          USE_ANGLES asks for them).

THE UPDATE MIRRORS L1SmartPixelsTrackProducer (digiRefit) term for term:
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
  * Q = theta0^2 * n_layers_crossed * (J_T J_T^T + J_L J_L^T), theta0 =
    MULT_SCATT_TERM / pT, kink at the last constrained radius r_s:
    J_T = (0, 1, 0, 0, -r_s), J_L = (0, 0, sec^2, -r_s sec^2, 0).
  Checked against CMSSW by REPLAY: feeding this update exactly the clusters
  the producer selected (and its alpha-only setting) must reproduce the
  producer's recorded running projection projLocalX/Y, projSigX/Y,
  projCotAlpha/Beta at every layer (replay_against_refit).

RANKING METRICS (which in-ellipse cluster is "best"):
  innovation  d^2 = r^T (J C J^T + R)^-1 r over (du, dv, dcotA, dcotB): the
              track's projected uncertainty PLUS the cluster's errors, with all
              correlations. Nothing tuned. A missing angle term is dropped from
              the quadratic form and replaced by its expectation, 1.
  producer    the digiRefit selection as it is today:
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
    "none": "no update: study 19's OT-only projection to every layer (baseline)",
    "innovation": "innovation chi2 r^T (J C J^T + R)^-1 r, track + cluster errors",
    "producer": "digiRefit today: sum (residual / cluster error)^2, unit weights",
    "tuned": "study-6 weights (1, 2, 16384, 16384) on the producer terms (stale)",
    "oracle": "truth: the majority-owner cluster when inside the ellipse",
}


def active_layers(mask):
    """Active layers, OUTERMOST first. mask[i] describes layer i + 1."""
    if len(mask) != 4 or set(mask) - {"A", "I"}:
        raise SystemExit(f"activeSP mask must be 4 of A/I, got {mask!r}")
    return [L for L in (4, 3, 2, 1) if mask[L - 1] == "A"]


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


# Which producer Q to reproduce. The nano's refit tables were written by one of:
#   "f629627"  the first Q (2026-09-06 .. 2026-09-23): theta = k/pT, jT = (0, 1, 0, 0,
#              -r_s) -- the WRONG d0 sign for TTTrack d0 -- and dtanL = theta sec^2.
#              Every refit nano produced before 2026-09-24 (e.g. itot_htt, itot_nopu,
#              itot_pu200_60ev) predates the fix and needs this.
#   "natural"  the fixed producer: theta = k/p (projected space angle), jT = (0, sec,
#              0, 0, +r_s sec), jL = (0, 0, sec^2, -r_s sec^2, 0) -- the one
#              formulation used everywhere else in these codes. itot_tp_* (2026-09-24)
#              were written by it.
PRODUCER_Q_FORMS = ("f629627", "natural")


def add_process_noise(a, C, t, L, prevL, prevR, kpt, form="f629627"):
    """The PRODUCER'S Q for tracks t, between their last constrained layer and L,
    in the form of the producer that wrote the file (PRODUCER_Q_FORMS). Only for
    replaying CMSSW and the producer-covariance comparison; the calibrated model
    is add_material."""
    if not len(t):
        return
    if form not in PRODUCER_Q_FORMS:
        raise ValueError(f"producer Q form {form!r} not in {PRODUCER_Q_FORMS}")
    pt = kpt / np.maximum(np.abs(a[t, 0]), 1e-12)
    ncross = np.maximum(1, np.abs(L - prevL[t]))
    rs, sec2 = prevR[t], 1.0 + a[t, 2] ** 2
    if form == "f629627":
        var = (MULT_SCATT_TERM / pt) ** 2 * ncross
        jT = np.zeros((len(t), 5)); jT[:, 1] = 1.0; jT[:, 4] = -rs
        jL = np.zeros((len(t), 5)); jL[:, 2] = sec2; jL[:, 3] = -rs * sec2
    else:
        sec = np.sqrt(sec2)
        var = (MULT_SCATT_TERM / (pt * sec)) ** 2 * ncross
        jT = np.zeros((len(t), 5)); jT[:, 1] = sec; jT[:, 4] = +rs * sec
        jL = np.zeros((len(t), 5)); jL[:, 2] = sec2; jL[:, 3] = -rs * sec2
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


def update_on(a, C, sel_c, c, ev, K, geo, state, L):
    """Update the tracks of the chosen candidates; bookkeeping into state."""
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
    state["prevL"][t] = L
    return gated


def run_chunk(A, geo, mask, use_angles, variants, nsig, replay, cal=None, q_form="f629627"):
    ctx = OP.prepare_chunk(A, geo)
    K, cls, ntrk = ctx["K"], ctx["cls"], ctx["ntrk"]
    use = cls >= 0
    kpt = pt_constant(ctx["T"])
    if cal is not None:                                   # calibrated seed, once
        C0 = seed_cov(cal, ctx["C"], ctx["a"][:, 2], ctx["n_disk_stubs"], ctx["T"]["nStubs"],
                      kpt / np.maximum(np.abs(ctx["a"][:, 0]), 1e-12))
    else:
        C0 = ctx["C"]
    layers = active_layers(mask)
    out = {"n_tracks_by_class": np.bincount(cls + 4, minlength=10), "by_variant": {}}
    first_ev = None
    for v in variants:
        a, C = ctx["a"].copy(), C0.copy()
        state = {"nUpd": np.zeros(ntrk, np.int64), "prevR": np.full(ntrk, -1.0),
                 "prevL": np.zeros(ntrk, np.int64)}
        Lv = np.full(ntrk, 5, np.int64)                      # last VISITED layer (5 = OT)
        rows, proj, selrec, gated = [], [], [], 0
        for li, L in enumerate(layers):
            if cal is not None:
                add_material(a, C, np.flatnonzero(use), L, Lv, geo, cal, kpt)
                Lv[use] = L
            else:
                q = np.flatnonzero(use & (state["nUpd"] > 0) & (state["prevR"] > 0))
                add_process_noise(a, C, q, L, state["prevL"], state["prevR"], kpt, q_form)
            if li == 0:                               # identical state for every variant
                if first_ev is None:                      # (calibrated: gap + outer kinks too)
                    first_ev = OP.evaluate_layer(ctx, a, C, L, geo, nsig, use)
                ev = first_ev
            else:
                ev = OP.evaluate_layer(ctx, a, C, L, geo, nsig, use)
            if ev is None:
                continue
            rows.append(ev["rows"])
            proj.append(ev["proj"])
            c = _candidates(ev, K, use_angles, cal)
            sel = choose(v, c, ev["pj"]) if v != "none" else np.zeros(0, np.int64)
            # selection outcome per layer projection: which category was chosen
            chosen = np.full(ntrk, -1, np.int8)
            chosen[c["t"][sel]] = c["cat"][sel]
            ti = ev["ti"]
            selrec.append({"layer": np.full(len(ti), L, np.int8), "cls": cls[ti].astype(np.int8),
                           "chosen": chosen[ti], "a_in": ev["a_in_track"][ti]})
            gated += update_on(a, C, sel, c, ev, K, geo, state, L)
        out["by_variant"][v] = {"rows": rows, "proj": proj, "sel": selrec, "gated": gated}
    if replay:
        out["replay"] = replay_against_refit(ctx, geo, kpt, q_form)
    return out


def replay_against_refit(ctx, geo, kpt, q_form="f629627"):
    """Feed this update the producer's OWN selections (alpha-only, as the
    production config ran) and compare against its recorded running projection.

    Visits the refit hit table in layer order IL4 -> IL1 (outsideIn); at each
    crossing projects the current state onto the producer's module, compares,
    then updates on the producer's selected cluster.
    """
    A, K = ctx["A"], ctx["K"]
    H = OP._flat(A, OP.REFIT, OP.REFIT_COLS + ["selClusterIdx", "hitAccepted", "projLocalX", "projLocalY",
                                                "projSigX", "projSigY", "projCotAlpha",
                                                "projCotBeta"])
    ntrk_ev = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    toff = np.concatenate([[0], np.cumsum(ntrk_ev)])
    ncl = ak.to_numpy(ak.num(A["L1TSmartPixelsCluster_detId"]))
    coff = np.concatenate([[0], np.cumsum(ncl)])
    gt = toff[H["event"]] + H["trackIdx"].astype(np.int64)
    a, C = ctx["a"].copy(), ctx["C"].copy()
    ntrk = ctx["ntrk"]
    state = {"nUpd": np.zeros(ntrk, np.int64), "prevR": np.full(ntrk, -1.0),
             "prevL": np.zeros(ntrk, np.int64)}
    cmp = {k: [] for k in ("layer", "du", "dv", "dcotA", "dcotB", "rsu", "rsv")}
    n_bad_det = 0
    for L in (4, 3, 2, 1):
        m = (H["layer"] == L)
        mi = OP.module_index(geo, H["detId"][m])
        ok = mi >= 0
        idx = np.flatnonzero(m)[ok]
        t, mi = gt[idx], mi[ok]
        q = t[(state["nUpd"][t] > 0) & (state["prevR"][t] > 0)]
        add_process_noise(a, C, q, L, state["prevL"], state["prevR"], kpt, q_form)
        s0 = OP.cylinder_s(a[t], geo["R"][L])
        pj = OP.project_with_cov(a[t], C[t], s0, geo, mi, jac="cmssw", R=L)   # the producer's H
        # the producer records its width through the guard-ZEROED H
        Jz = np.where(np.isfinite(pj["J"]).all(1, keepdims=True)
                      & (np.abs(np.nan_to_num(pj["J"], nan=np.inf)) <= JACOBIAN_MAX_ABS).all(1, keepdims=True),
                      np.nan_to_num(pj["J"]), 0.0)
        pj["S"] = np.einsum("nij,njk,nlk->nil", Jz, C[t], Jz)
        g = np.isfinite(pj["u"]) & (H["projLocalX"][idx] > OP.SENTINEL)
        cmp["layer"].append(np.full(g.sum(), L))
        cmp["du"].append((pj["u"] - H["projLocalX"][idx])[g])
        cmp["dv"].append((pj["v"] - H["projLocalY"][idx])[g])
        cmp["dcotA"].append((pj["cotA"] - H["projCotAlpha"][idx])[g])
        cmp["dcotB"].append((pj["cotB"] - H["projCotBeta"][idx])[g])
        cmp["rsu"].append((np.sqrt(pj["S"][:, 0, 0]) / H["projSigX"][idx])[g])
        cmp["rsv"].append((np.sqrt(pj["S"][:, 1, 1]) / H["projSigY"][idx])[g])
        # update on the producer's selected cluster
        acc = (H["hitAccepted"][idx] > 0) & (H["selClusterIdx"][idx] >= 0) & np.isfinite(pj["u"])
        cg = coff[H["event"][idx]] + H["selClusterIdx"][idx].astype(np.int64)
        cg = np.where(acc, cg, 0)
        n_bad_det += int((acc & (K["detId"][cg] != H["detId"][idx])).sum())
        u = np.flatnonzero(acc)
        if len(u):
            ci = cg[u]
            h0 = np.stack([pj["u"][u], pj["v"][u], pj["cotA"][u], pj["cotB"][u]], 1)
            meas = np.stack([K["localX"][ci], K["localY"][ci], K["localCotAlpha"][ci],
                             K["localCotBeta"][ci]], 1).astype(np.float64)
            sig = np.stack([K["sigX"][ci], K["sigY"][ci], K["sigAlpha"][ci],
                            K["sigBeta"][ci]], 1).astype(np.float64)
            use_dim = np.stack([np.ones(len(u), bool), np.ones(len(u), bool),
                                K["hasAlpha"][ci] > 0, np.zeros(len(u), bool)], 1)   # alpha only
            kalman_update(a, C, t[u], pj["J"][u], h0, meas, sig, use_dim)
            gp = geo["origin"][mi[u]] + pj["u"][u][:, None] * geo["ex"][mi[u]] \
                + pj["v"][u][:, None] * geo["ey"][mi[u]]
            state["prevR"][t[u]] = np.hypot(gp[:, 0], gp[:, 1])
            state["nUpd"][t[u]] += 1
            state["prevL"][t[u]] = L
    outc = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cmp.items()}
    outc["n_bad_det"] = n_bad_det
    return outc


def run(files, geometry, mask="AAAA", use_angles="alphaBeta", variants=VARIANTS,
        nsig=4.0, chunk=10, nev=None, replay=True, log=print, calibration=None,
        producer_q="natural"):
    """producer_q: which producer wrote the file's refit tables (PRODUCER_Q_FORMS);
    it sets the replay AND the uncalibrated (producer-covariance) variants."""
    cal = load_calibration(calibration) if calibration else None
    geo = OP.load_geometry(geometry)
    OP.check_inputs(files[0], with_refit=True)
    extra = [f"{OP.REFIT}_{c}" for c in ("selClusterIdx", "hitAccepted", "projLocalX", "projLocalY", "projSigX",
                                         "projSigY", "projCotAlpha", "projCotBeta")]
    acc = {v: {"rows": [], "proj": [], "sel": [], "gated": 0} for v in variants}
    rep = []
    ncls = np.zeros(10, np.int64)
    done = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], OP.branches(True) + extra,
                            step_size=chunk):
        if nev is not None and done >= nev:
            break
        if nev is not None and done + len(A) > nev:
            A = A[:nev - done]
        r = run_chunk(A, geo, mask, use_angles, variants, nsig, replay, cal, producer_q)
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
    res = {"n_events": done, "mask": mask, "use_angles": use_angles, "layers": active_layers(mask),
           "calibration": calibration,
           "tracks_by_class": dict(zip(names, ncls[:len(names)].tolist())), "nsig": nsig,
           "variants": {v: {"rows": cat(acc[v]["rows"]), "proj": cat(acc[v]["proj"]),
                            "sel": cat(acc[v]["sel"]), "gated": acc[v]["gated"]} for v in variants}}
    if rep:
        res["replay"] = {k: (np.concatenate([x[k] for x in rep]) if k != "n_bad_det"
                             else sum(x[k] for x in rep)) for k in rep[0]}
    return res
