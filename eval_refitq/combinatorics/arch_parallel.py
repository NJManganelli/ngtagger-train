"""Parallel architecture: standalone IT and OT track finding, then a combined refit.

THE HARDWARE PREMISE. During the 4-5 us in which the existing OT track finder
runs, a SmartPixels track finder would be running standalone -- seeding and
fitting IT clusters into mini-tracks with no OT information whatsoever, not even
an OT stub to constrain z. Only afterwards are the two systems' OUTPUTS (tracks,
not hits) brought together and refit. That is what makes it a fair test of the
parallel design, and it is also the least favourable reading, since the IT side
gets no help.

Contrast the two designs this exists to compare:

  PARALLEL (here)   IT-only seeds -> IT layers only -> IT mini-tracks
                    OT-only seeds -> OT layers only -> OT tracks
                    DR within each system separately
                    match IT track <-> OT track, then refit on the union
                    MIXED seeds cannot exist: they cross systems at seed time

  COMBINED          every seed follows all ten layers, chi2 gating and DR act
                    on combined tracks, mixed seeds are first class
                    (this is what tp_findability already does)

  DIGIREFIT-LIKE    an OT track projected into the IT, attaching clusters, then
                    refit -- the IT is GUIDED by the OT rather than finding
                    anything itself. The existing SmartPixels digiRefit is a
                    simplified, hardware-ignorant precursor of the parallel
                    design, and this baseline is obtained for free as the
                    COMBINED mode restricted to OT-only seeds.

THE MINIMUM-LAYER RULE CANNOT BE INHERITED. The OT's MinLayers = 4 is defined
against six barrel layers. The IT has four in total, so demanding four means
demanding ALL of them, and in a 1110 build a doublet could reach at most three
and would be rejected outright. IT mini-tracks therefore get their own minimum,
default 3, and in a three-layer build that already means every layer.

THE MATCH IS THE OPEN QUESTION. The two tracks share ZERO hits by construction,
so the shared-hit criterion duplicate removal uses is unavailable. What is left
is agreement of the helix parameters, normalised by the two covariances the KF
already produces. Whether that is sharp enough is quantitative and not obvious:
an IT mini-track has at most a 13 cm lever arm, so its sigma(kappa) is poor and
the match may end up carried by phi0 and z0 alone. The matching efficiency and
purity is a headline result of the architecture comparison, not a detail.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import seed_arity as SA              # noqa: E402
import kf_emulation as KF            # noqa: E402

IT_MIN_LAYERS = 3
MATCH_PARS = ("phi0", "z0", "cot", "kappa")


def find_system(U, Q, seeds, ptmin, system, min_layers, kf_opts=None):
    """Standalone track finding inside ONE system.

    system "IT" keeps IT-only seeds and follows IT layers only; "OT" keeps
    OT-only seeds and follows OT layers only. Mixed seeds are dropped either
    way -- by definition they are not available to a parallel architecture.
    Returns a list of per-seed dicts, each already chi2-accepted.
    """
    kf_opts = kf_opts or {}
    d0_opt = {k: v for k, v in kf_opts.items() if k == "d0_prior_cm"}
    lo, hi = (1, 4) if system == "IT" else (11, 16)
    targets = [L for L in (list(SA.IL) + list(SA.OT_BARREL)) if lo <= L <= hi]
    out = []
    for s_i, sd in enumerate(seeds):
        if not all(lo <= L <= hi for L in sd.layers):
            continue
        try:
            o = SA.run_seed(U, Q, sd, ptmin, targets)
        except M.TooWide:
            continue
        if o is None or not o.get("tracks_to_fit"):
            continue
        # own minimum-layer rule, applied here rather than inside run_seed
        # because run_seed cannot know how many layers the system even has
        nlay = sd.arity + o["_nconf"]
        enough = nlay >= min_layers
        if not enough.any():
            continue
        o = SA.filter_tracks(o, enough)
        gidx = KF.hits_from_seed(U, o, targets)
        gc = o.get("_gC")
        trip = (o["_gA"], o["_gB"], gc if gc is not None else o["_gB"])
        fit = KF.fit_tracks(U, Q, trip, gidx=gidx, layers=targets,
                            use_angles=False, **d0_opt)
        keep, chi2s = KF.good_state(fit, ptmin)
        if not keep.any():
            continue
        w = 0.30 if sd.arity == 2 else 0.03
        score = KF.rank_score(fit, chi2s, w_angle=w)
        out.append({"seed": sd, "seed_idx": s_i,
                    "o": SA.filter_tracks(o, keep), "gidx": gidx[keep],
                    "fit": {k: (v[keep] if isinstance(v, np.ndarray)
                                and v.ndim == 1 and len(v) == len(keep) else v)
                            for k, v in fit.items() if k != "hits"},
                    "score": score[keep], "chi2": chi2s[keep],
                    "targets": targets})
    return out


def pool_and_dr(U, per_seed, do_dr=True):
    """Flatten one system's per-seed results and run DR WITHIN that system."""
    if not per_seed:
        return None
    G = np.concatenate([p["gidx"] for p in per_seed])
    SC = np.concatenate([p["score"] for p in per_seed])
    sidx = np.concatenate([np.full(len(p["score"]), p["seed_idx"])
                           for p in per_seed])
    ga = np.concatenate([p["o"]["_gA"] for p in per_seed])
    gb = np.concatenate([p["o"]["_gB"] for p in per_seed])
    keys = [k for k in ("kappa", "phi0", "d0", "cot", "z0", "var_kappa",
                        "var_phi0", "var_d0", "var_cot", "var_z0", "nhit")]
    F = {k: np.concatenate([p["fit"][k] for p in per_seed]) for k in keys}
    surv = SA.duplicate_removal(G, SC) if do_dr else np.ones(len(SC), bool)
    sel = np.flatnonzero(surv)
    return {"gidx": G[sel], "score": SC[sel], "seed_idx": sidx[sel],
            "gA": ga[sel], "gB": gb[sel], "event": U["event"][ga[sel]],
            "tpIdx": U["tpIdx"][ga[sel]],
            **{k: F[k][sel] for k in keys}}


def match_chi2(A, B, ia, ib, pars=MATCH_PARS):
    """Helix-agreement chi2 between IT track ia and OT track ib.

    Each term is a squared difference over the SUM of the two variances, so a
    parameter one system measures badly contributes little rather than
    dominating. pars selects which parameters take part, because it is not
    obvious that all four should: the IT mini-track's curvature is weak over a
    13 cm lever arm and may only add noise.
    """
    c2 = np.zeros(len(ia))
    for p in pars:
        da = A[p][ia] - B[p][ib]
        if p == "phi0":
            da = M.wrap(da)
        v = A["var_" + p][ia] + B["var_" + p][ib]
        c2 += da * da / np.maximum(v, 1e-30)
    return c2


def match_systems(IT, OT, pars=MATCH_PARS, max_chi2=50.0):
    """One-to-one IT<->OT assignment, best chi2 first, within an event.

    Greedy rather than globally optimal: a real system would not run the
    Hungarian algorithm in firmware, and greedy-by-best-chi2 is what a
    sequential matcher does.
    """
    if IT is None or OT is None:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    ia_l, ib_l, c2_l = [], [], []
    ev = np.unique(IT["event"])
    for e in ev:
        ai = np.flatnonzero(IT["event"] == e)
        bi = np.flatnonzero(OT["event"] == e)
        if not len(ai) or not len(bi):
            continue
        AA = np.repeat(ai, len(bi))
        BB = np.tile(bi, len(ai))
        c2 = match_chi2(IT, OT, AA, BB, pars)
        good = c2 <= max_chi2
        if not good.any():
            continue
        AA, BB, c2 = AA[good], BB[good], c2[good]
        order = np.argsort(c2)
        usedA, usedB = set(), set()
        for k in order:
            a, b = int(AA[k]), int(BB[k])
            if a in usedA or b in usedB:
                continue
            usedA.add(a)
            usedB.add(b)
            ia_l.append(a)
            ib_l.append(b)
            c2_l.append(c2[k])
    return (np.array(ia_l, int), np.array(ib_l, int), np.array(c2_l))


def refit_matched(U, Q, IT, OT, ia, ib, kf_opts=None,
                  ms_scale=None, it_ot_scale=None, reverse=False):
    """Refit each matched pair on the UNION of its two tracks' hits."""
    kf_opts = kf_opts or {}
    d0_opt = {k: v for k, v in kf_opts.items() if k == "d0_prior_cm"}
    L = list(KF.LAYER_ORDER)
    pos = {x: i for i, x in enumerate(L)}
    G = np.full((len(ia), len(L)), -1, np.int64)
    for src, idx in ((IT, ia), (OT, ib)):
        tg = src["gidx"]
        for j in range(tg.shape[1]):
            g = tg[idx, j]
            m = g >= 0
            if not m.any():
                continue
            lay = U["layer"][g[m]]
            for Lv in np.unique(lay):
                mm = lay == Lv
                rows = np.flatnonzero(m)[mm]
                G[rows, pos[int(Lv)]] = g[m][mm]
    # Truth is keyed on the IT track's seed cluster; the pT hint comes from the
    # OT track, whose curvature is the better of the two by a long way.
    trip = (IT["gA"][ia], IT["gB"][ia], IT["gA"][ia])
    pt = 1.0 / np.maximum(np.abs(OT["kappa"][ib]), 1e-3)
    extra = {}
    if ms_scale is not None:
        extra["ms_scale"] = ms_scale
    if it_ot_scale is not None:
        extra["it_ot_scale"] = it_ot_scale
    extra["reverse"] = reverse
    fit = KF.fit_tracks(U, Q, trip, gidx=G, use_angles=False,
                        pt_hint=pt, **d0_opt, **extra)
    return fit, G
