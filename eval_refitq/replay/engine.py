"""The WP-C replay engine (plan C1a items 2-6).

One visit loop over the IT surfaces of an OT track, outside-in along the OT helix's path, as
L1SmartPixelsTrackProducer (seedlessRefit) visits them. This file holds the FORCED mode: CMSSW's
own recorded selections are replayed (module, accepted cluster), and every quantity the producer
records per crossing is reproduced side by side with it:

  running projection  projLocalX/Y, projCotAlpha/Beta  (the state's crossing, after Q)
  running width       projSigX/Y = sqrt(H C H^T)       (C after Q, before the update)
  per-scalar pulls    pullX/Y/Alpha/Beta = r / sqrt(S)
  OT crossing / cone  projOTLocalX/Y, projOTSigX/Y      (the OT state: no drift)

The building blocks are ot_projection (OP) and ot_refit_projection (ORP), imported unchanged.
At the default options the forced mode reproduces ORP.replay_against_refit bit for bit
(tests/test_replay_kat.py). The options exist for the tail localisation (tail.py, plan §1
item 1); each one is a hypothesis about where the replay departs from CMSSW:

  crossing   "f64"     OP's float64 crossing arithmetic (ORP)
             "f32"     CMSSW's float types: the helix point is a float GlobalPoint, the local point and
                       the local momentum come from the module's float rotation and position in float
                       arithmetic (SmartPixelsHelixProjector::land), the geometry in float (H4)
             "f32fma"  "f32" with the rotation x*a + y*b + z*c contracted the way the canon_v3 (aarch64)
                       build evaluates it, fma(x, a, fma(y, b, z*c)): measured on file 4, it makes the OT
                       crossing bit-identical on ~90 % (u) / ~97 % (v) of crossings against ~72 / ~89 %
                       without it (tail.py reports the fractions)
  meas       "refit"   the refit table's copies (positions and their sigma as floats)
             "cluster" the cluster table (16-bit positions, 10-bit sigma): the mirror's input (H3)
  ncross_offset  nCross + offset (min 1) in the producer's Q (H5)
  kpt        "tracks" median pT |rInv| of the chunk (ORP); "cmssw" float32 MagConstant x b_field
  relin      False drops the relinearisation term -H (a - a_lin) (a planted bug: the attribution
             known-bad)
  noise_seed not None, with `noise` given to forced(): the H1 arithmetic-noise ensemble (PLAN §10
             PD6 (b)). Per crossing, the predicted u, v move by a draw from the measured OT-state
             residuals of the surface class, the Jacobian's position rows scale by (1 + a draw from the
             measured OT-cone width residuals), the predicted angles move by one float32 ulp and the
             angle measurements uniformly within their 12-bit storage quantum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import common as CM
from common import OP, ORP
from ngtagger.data import spix_nano as SPN

F32 = np.float32
MAG_CONSTANT_F32 = F32(0.299792458)          # L1SmartPixelsTrackProducer MagConstant (float)


@dataclass(frozen=True)
class ForcedOpts:
    crossing: str = "f64"
    meas: str = "refit"
    ncross_offset: int = 0
    kpt: str = "tracks"
    relin: bool = True
    noise_seed: int | None = None
    shift_il2_sigma: float = 0.0     # E4 known-bad: measured u shifted by this many sigma_x at IL2

    def label(self) -> str:
        parts = [self.crossing, self.meas]
        if self.ncross_offset:
            parts.append(f"ncross{self.ncross_offset:+d}")
        if self.kpt != "tracks":
            parts.append(f"kpt-{self.kpt}")
        if not self.relin:
            parts.append("no-relin")
        if self.noise_seed is not None:
            parts.append(f"noise-s{self.noise_seed}")
        if self.shift_il2_sigma:
            parts.append(f"shift-il2-{self.shift_il2_sigma:g}sig")
        return "/".join(parts)


DEFAULT = ForcedOpts()


# ---------------------------------------------------------------- the crossing (land) ----
def kpt_value(ctx, kind: str) -> float:
    """pT = kpt / |rInv|: the chunk median of pT |rInv| (ORP), or the producer's float32
    MagConstant x b_field with b_field recovered from that median."""
    k = ORP.pt_constant(ctx["T"])
    if kind == "tracks":
        return k
    b = F32(k * 100.0 / float(MAG_CONSTANT_F32))
    return float(F32(MAG_CONSTANT_F32 * b)) / 100.0


def _newton(a, s0, o, n_):
    """SmartPixelsHelixProjector::land's Newton loop (OP._newton_cmssw's semantics) with the
    module origin and normal given (float values for the f32 crossings)."""
    s = np.where(np.isfinite(s0), s0, 1.0)
    alive = np.isfinite(s0)
    lim = 3.0 * np.where(np.isfinite(s0), s0, 1.0) + 10.0
    conv = np.zeros(len(s), bool)
    for _ in range(5):
        p, d = OP.helix_at(a, s)
        f = ((p - o) * n_).sum(-1)
        fp = (d * n_).sum(-1)
        now = alive & ~conv & (np.abs(f) < OP.ON_PLANE_CM)
        conv |= now
        act = alive & ~conv
        flat = act & (np.abs(fp) < 1e-9)
        alive &= ~flat
        act &= ~flat
        s = np.where(act, s - f / np.where(act, fp, 1.0), s)
        bad = act & (~(s > 0) | (s > lim))
        alive &= ~bad
    return s, conv


def _fma32(a, b, c):
    """round_float32(a * b + c) for float32 a, b, c (the product is exact in float64)."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(F32)


def _dot32(v, w, fma):
    """TkRotation<float> x Basic3DVector<float>, x*a + y*b + z*c in float arithmetic; fma: the
    contraction fma(x, a, fma(y, b, z*c))."""
    if fma:
        return _fma32(v[..., 0], w[..., 0], _fma32(v[..., 1], w[..., 1], v[..., 2] * w[..., 2]))
    return (v[..., 0] * w[..., 0] + v[..., 1] * w[..., 1]) + v[..., 2] * w[..., 2]


def quantum12(x):
    """The storage quantum of a float kept with 12 mantissa bits (NanoAOD mantissaBits=12)."""
    x = np.abs(np.asarray(x, np.float64))
    e = np.floor(np.log2(np.where(x > 0, x, 1.0)))
    return np.where(x > 0, np.exp2(e - 12.0), 0.0)


def _land(a, s0, geo, mi, crossing, kpt):
    """(u, v, cotA, cotB, s, ok, gr) of helices a on modules mi in CMSSW's float types (crossing
    "f32" or "f32fma", see the module doc), Newton from the nominal-surface arc s0 with the float
    module origin and normal."""
    o = geo["origin"][mi].astype(F32).astype(np.float64)
    n_ = geo["ez"][mi].astype(F32).astype(np.float64)
    s, ok = _newton(a, s0, o, n_)
    p, d = OP.helix_at(a, s)
    fma = crossing == "f32fma"
    ex, ey, ez = (geo[k][mi].astype(F32) for k in ("ex", "ey", "ez"))
    p32 = p.astype(F32)
    rel = p32 - geo["origin"][mi].astype(F32)
    pt = np.abs(kpt / np.where(a[:, 0] != 0, a[:, 0], 1e-12))
    mom = (d * pt[:, None]).astype(F32)
    u, v = _dot32(rel, ex, fma), _dot32(rel, ey, fma)
    lx, ly, lz = _dot32(mom, ex, fma), _dot32(mom, ey, fma), _dot32(mom, ez, fma)
    lzd = lz.astype(np.float64)
    lzd = np.where(np.abs(lzd) > 1e-9, lzd, 1e-9)
    return {"u": u.astype(np.float64), "v": v.astype(np.float64),
            "cotA": lx.astype(np.float64) / lzd, "cotB": ly.astype(np.float64) / lzd,
            "s": s, "ok": ok, "gr": np.hypot(p32[:, 0].astype(np.float64), p32[:, 1].astype(np.float64))}


OUT_KEYS = OP.OUT_KEYS


def _draw(pool, n, rng):
    return pool[rng.integers(0, len(pool), n)] if len(pool) else np.zeros(n)


def project(a, C, s0, geo, mi, S, opts=DEFAULT, kpt=None, noise=None, rng=None):
    """The producer's crossing on modules mi and its numerical Jacobian (one-sided +eps, then
    -eps; a perturbation that crossSurface would put on another module is unusable), plus
    S = J C J^T. "f64" delegates to OP.project_with_cov(jac="cmssw") unchanged. `noise` (with
    rng): the H1 arithmetic-noise ensemble's perturbation of the prediction and the Jacobian."""
    if opts.crossing == "f64":
        pj = OP.project_with_cov(a, C, s0, geo, mi, jac="cmssw", surface=S)
        pj["gr"] = None
        return pj
    nom = _land(a, s0, geo, mi, opts.crossing, kpt)
    J = np.full((len(mi), 4, 5), np.nan)
    for j in range(5):
        cols = []
        for sgn in (+1.0, -1.0):
            ap = a.copy()
            ap[:, j] += sgn * OP.JAC_EPS[j]
            pp = _land(ap, OP.surface_s(ap, geo, S), geo, mi, opts.crossing, kpt)
            pp["ok"] = pp["ok"] & (OP.cross_module(ap, S, geo) == mi)
            cols.append(pp)
        for r, key in enumerate(OUT_KEYS):
            fwd = (cols[0][key] - nom[key]) / OP.JAC_EPS[j]
            bwd = (nom[key] - cols[1][key]) / OP.JAC_EPS[j]
            J[:, r, j] = np.where(cols[0]["ok"], fwd, np.where(cols[1]["ok"], bwd, np.nan))
    if noise is not None:
        cls = "disc" if SPN.is_disc_surface(S) else "barrel"
        n = len(mi)
        nom["u"] = nom["u"] + _draw(noise["du"][cls], n, rng)
        nom["v"] = nom["v"] + _draw(noise["dv"][cls], n, rng)
        for k in ("cotA", "cotB"):
            nom[k] = nom[k] * (1.0 + rng.choice((-1.0, 1.0), n) * 2.0 ** -24)
        J[:, 0, :] *= (1.0 + _draw(noise["wu"][cls], n, rng))[:, None]
        J[:, 1, :] *= (1.0 + _draw(noise["wv"][cls], n, rng))[:, None]
    nom["S"] = np.einsum("nij,njk,nlk->nil", J, C, J)
    nom["J"] = J
    nom["ok"] &= np.isfinite(nom["S"]).all((1, 2))
    return nom


def guard_zeroed(J, jmax=ORP.JACOBIAN_MAX_ABS):
    """The producer's jacobianMaxAbs guard: a column with any non-finite or |H| > jmax entry
    is zeroed (the parameter is locally unobservable)."""
    bad = ~np.isfinite(J).all(1, keepdims=True) | (np.abs(np.nan_to_num(J, nan=np.inf)) > jmax).any(1, keepdims=True)
    return np.where(bad, 0.0, np.nan_to_num(J))


# ---------------------------------------------------------------- the update ----
def kf_update(a, C, t, J, h0, meas, sig, use_dim, relin=True, gate=ORP.CHI2_UPDATE_GATE):
    """Sequential scalar updates (x, y, cotAlpha, cotBeta), ORP.kalman_update's arithmetic,
    returning per scalar the pull r / sqrt(S), r^2 / S and whether it was applied. relin=False
    drops the relinearisation term (the planted known-bad)."""
    J = J.copy()
    bad = ~np.isfinite(J).all(1) | (np.abs(np.nan_to_num(J, nan=np.inf)) > ORP.JACOBIAN_MAX_ABS).any(1)
    J[np.repeat(bad[:, None, :], 4, 1)] = 0.0
    a_lin = a[t].copy()
    n = len(t)
    pull = np.full((n, 4), np.nan)
    chi2 = np.zeros((n, 4))
    applied = np.zeros((n, 4), bool)
    for k in range(4):
        H = J[:, k, :]
        Ct = C[t]
        v = np.einsum("nij,nj->ni", Ct, H)
        S = (H * v).sum(1) + sig[:, k] ** 2
        r = meas[:, k] - h0[:, k] - (H * (a[t] - a_lin)).sum(1) if relin else meas[:, k] - h0[:, k]
        ok = use_dim[:, k] & (S > 0) & np.isfinite(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            c2 = np.where(ok, r * r / S, 0.0)
        ok &= ~(ok & (c2 > gate))
        if not ok.any():
            continue
        pull[ok, k] = r[ok] / np.sqrt(S[ok])
        chi2[ok, k] = c2[ok]
        applied[ok, k] = True
        tt, vv, SS, rr = t[ok], v[ok], S[ok], r[ok]
        Kg = vv / SS[:, None]
        a[tt] += Kg * rr[:, None]
        C[tt] -= Kg[:, :, None] * vv[:, None, :]
    return pull, chi2, applied


def d0_floor(C, cfg):
    """The producer's weak d0 prior: C(4,4) >= 0.25 cm^2 for 4-parameter tracks or a track
    without d0 variance (L1SmartPixelsTrackProducer.cc:1605-1607). Returns the floored copy and
    the number of tracks it acted on."""
    C = C.copy()
    m = (cfg.get("otNPar", 5) == 4) | ~(C[:, 4, 4] > 0)
    C[m, 4, 4] = np.maximum(C[m, 4, 4], 0.25)
    return C, int(m.sum())


# ---------------------------------------------------------------- forced mode ----
FORCED_HIT_COLS = ORP.REPLAY_COLS + ["projOTLocalX", "projOTLocalY", "projOTCotAlpha", "projOTCotBeta",
                                     "projOTSigX", "projOTSigY", "pullX", "pullY", "pullAlpha", "pullBeta"]


def hit_table(A):
    """The flat refit hit table of a chunk with the columns the forced mode reads."""
    cols = list(dict.fromkeys(OP.REFIT_COLS + FORCED_HIT_COLS))
    return OP._flat(A, CM.REFIT_HIT, cols)


def branches(path):
    keys = OP._keys(str(path))
    return OP.branches(True, str(path), refit=CM.REFIT_HIT) + \
        [b for c in FORCED_HIT_COLS for b in SPN.branch_names(keys, CM.REFIT_HIT, [c])]


def forced(ctx, H, geo, cfg, mask, opts=DEFAULT, track_mask=None, noise=None):
    """Replay CMSSW's own selections along the OT helix's visit sequence (see the module doc);
    track_mask (bool per track) restricts the replay to those tracks (rows of the others stay NaN);
    noise (with opts.noise_seed): the H1 ensemble's residual pools {"du", "dv", "wu", "wv"} x
    {"barrel", "disc"}.

    Returns per hit-table row the replay's running prediction (u, v, cotA, cotB, su, sv), its
    per-scalar pulls and applied flags, the OT-state crossing on the same module (u, v, cotA, cotB,
    su, sv at the OT state, C_OT after the d0 floor), the row's visit step; per track the final
    state (a, C) and the number of updates. NaN where the replay has no value."""
    K = ctx["K"]
    ntrk = ctx["ntrk"]
    kpt = kpt_value(ctx, opts.kpt)
    if opts.crossing not in ("f64", "f32", "f32fma") or opts.meas not in ("refit", "cluster"):
        raise ValueError(f"unknown forced options {opts}")
    if (opts.noise_seed is None) != (noise is None) or (noise is not None and opts.crossing == "f64"):
        raise ValueError("the H1 ensemble needs a float32 crossing, a noise_seed and the noise pools together")
    rng = np.random.default_rng(opts.noise_seed) if noise is not None else None
    a = ctx["a"].copy()
    C, n_floor = d0_floor(ctx["C"], cfg)
    a_ot, C_ot = a.copy(), C.copy()
    toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
    coff = np.concatenate([[0], np.cumsum(K["_n"])])
    gt = toff[H["event"]] + H["trackIdx"].astype(np.int64)
    surf = H["layer"].astype(np.int64)
    path = ORP.surface_path(a, geo)
    active = [s for s in ORP.active_surfaces(mask) if s in geo["surfaces"]]
    seq = ORP.visit_sequence(path, active, cfg["layerOrder"])
    step = ORP._row_steps(seq, gt, surf)
    row_of = np.full((ntrk, seq.shape[1]), -1, np.int64)
    okr = np.flatnonzero(step >= 0)
    row_of[gt[okr], step[okr]] = okr
    has_rows = np.zeros(ntrk, bool)
    has_rows[gt] = True
    if track_mask is not None:
        has_rows &= track_mask
    ua, ub = cfg["useAngles"] in ORP.USE_ANGLES, cfg["useAngles"] == "alphaBeta"
    n = len(gt)
    out = {k: np.full(n, np.nan) for k in ("u", "v", "cotA", "cotB", "su", "sv",
                                           "u_ot", "v_ot", "cotA_ot", "cotB_ot", "su_ot", "sv_ot")}
    out["pull"] = np.full((n, 4), np.nan)
    out["applied"] = np.zeros((n, 4), bool)
    out["step"] = step
    out["track"] = gt
    # OT-state crossing on every recorded module (no state drift: H4/H6 and the H rows)
    mi_all = OP.module_index(geo, H["detId"])
    for S in np.unique(surf):
        r = np.flatnonzero((surf == S) & (mi_all >= 0) & has_rows[gt])
        if not len(r):
            continue
        t = gt[r]
        pj = project(a_ot[t], C_ot[t], OP.surface_s(a_ot[t], geo, int(S)), geo, mi_all[r], int(S), opts, kpt)
        Jz = guard_zeroed(pj["J"])
        Sz = np.einsum("nij,njk,nlk->nil", Jz, C_ot[t], Jz)
        for k_, q_ in (("u_ot", "u"), ("v_ot", "v"), ("cotA_ot", "cotA"), ("cotB_ot", "cotB")):
            out[k_][r] = np.where(pj["ok"], pj[q_], np.nan)
        out["su_ot"][r] = np.sqrt(np.clip(Sz[:, 0, 0], 0, None))
        out["sv_ot"][r] = np.sqrt(np.clip(Sz[:, 1, 1], 0, None))
    nUpd = np.zeros(ntrk, np.int64)
    prevR = np.full(ntrk, -1.0)
    prevNT = np.zeros(ntrk, np.int64)
    for j in range(seq.shape[1]):
        for S in np.unique(seq[:, j][seq[:, j] >= 0]):
            S = int(S)
            t = np.flatnonzero(has_rows & (seq[:, j] == S) & (nUpd < cfg["maxKFUpdates"]))
            if not len(t):
                continue
            if cfg["applyProcessNoise"]:
                q = t[(nUpd[t] > 0) & (prevR[t] > 0)]
                nT = ORP.n_traversed(path, q, S)
                ncur = prevNT[q] + np.maximum(1, np.abs(nT - prevNT[q]) + opts.ncross_offset)
                ORP.add_process_noise(a, C, q, ncur, prevNT[q], prevR[q], kpt, cfg["multScattTerm"])
            r = row_of[t, j]
            t, r = t[r >= 0], r[r >= 0]
            mi = mi_all[r]
            t, r, mi = t[mi >= 0], r[mi >= 0], mi[mi >= 0]
            if not len(t):
                continue
            pj = project(a[t], C[t], OP.surface_s(a[t], geo, S), geo, mi, S, opts, kpt, noise, rng)
            Jz = guard_zeroed(pj["J"])
            Sz = np.einsum("nij,njk,nlk->nil", Jz, C[t], Jz)
            fin = np.isfinite(pj["u"])
            for k_ in ("u", "v", "cotA", "cotB"):
                out[k_][r] = np.where(fin, pj[k_], np.nan)
            out["su"][r] = np.sqrt(np.clip(Sz[:, 0, 0], 0, None))
            out["sv"][r] = np.sqrt(np.clip(Sz[:, 1, 1], 0, None))
            acc = (H["hitAccepted"][r] > 0) & (H["selClusterIdx"][r] >= 0) & fin
            u = np.flatnonzero(acc)
            if not len(u):
                continue
            ru = r[u]
            h0 = np.stack([pj["u"][u], pj["v"][u], pj["cotA"][u], pj["cotB"][u]], 1)
            if opts.meas == "refit":
                meas = np.stack([H["recoLocalX"][ru], H["recoLocalY"][ru], H["recoCotAlpha"][ru],
                                 H["recoCotBeta"][ru]], 1).astype(np.float64)
                sig = np.stack([H["sigX"][ru], H["sigY"][ru], H["sigAlpha"][ru],
                                H["sigBeta"][ru]], 1).astype(np.float64)
            else:
                cg = coff[H["event"][ru]] + H["selClusterIdx"][ru].astype(np.int64)
                meas = np.stack([K["localX"][cg], K["localY"][cg], K["localCotAlpha"][cg],
                                 K["localCotBeta"][cg]], 1).astype(np.float64)
                sig = np.stack([K["sigX"][cg], K["sigY"][cg], K["sigAlpha"][cg],
                                K["sigBeta"][cg]], 1).astype(np.float64)
            if opts.shift_il2_sigma:
                il2 = S == 2
                if il2:
                    meas[:, 0] += opts.shift_il2_sigma * sig[:, 0]
            if noise is not None:
                q12 = quantum12(meas[:, 2:4])
                meas[:, 2:4] += rng.uniform(-0.5, 0.5, q12.shape) * q12
            use_dim = np.stack([np.ones(len(u), bool), np.ones(len(u), bool),
                                (H["hasAlpha"][ru] > 0) & ua, (H["hasBeta"][ru] > 0) & ub], 1)
            pull, _, applied = kf_update(a, C, t[u], pj["J"][u], h0, meas, sig, use_dim, relin=opts.relin)
            out["pull"][ru] = pull
            out["applied"][ru] = applied
            if pj["gr"] is None:
                gp = geo["origin"][mi[u]] + pj["u"][u][:, None] * geo["ex"][mi[u]] \
                    + pj["v"][u][:, None] * geo["ey"][mi[u]]
                prevR[t[u]] = np.hypot(gp[:, 0], gp[:, 1])
            else:
                prevR[t[u]] = pj["gr"][u]
            nUpd[t[u]] += 1
            prevNT[t[u]] = ORP.n_traversed(path, t[u], S)
    return {"rows": out, "a": a, "C": C, "nUpd": nUpd, "has_rows": has_rows, "n_d0_floor": n_floor,
            "off_path": int((step < 0).sum()), "kpt": kpt}
