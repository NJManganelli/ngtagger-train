"""The free-running replay: arms, switches and the visit loop (PLAN §1 items 3-6, §2, §10 PD1-PD3).

An ARM is one configuration of the switches of PLAN §2. The named arms of the entry gate:

  PM    the mirror of refit v2 (L1SmartPixelsTrackProducer, seedlessRefit): the producer's module choice
        (crossSurface, with its predicted-angle clamp), its Jacobian and float32 crossing arithmetic
        (engine "f32fma", PLAN §10 PD5 (a)), one module, box windows in readout order truncated at
        maxHitsPerWindow, its selection metric (positions and angles over the CLUSTER sigma), its Q with
        the empty-window recharge, maxKFUpdates, the d0 floor, the measured-angle cap (the cluster loader
        applies it, spix_nano.cluster_columns), discs; the cluster table as input. PM-B: no discs.
  INN   the calibrated arm: ellipse candidates (ot_projection.evaluate_surface, central Jacobian), the
        innovation metric, the calibrated seed and Highland material (refit_calibration v3), the
        position R tables, the table's angle errors (angle_import, barrel), the inner material at the
        POCA. No discs at all (PLAN §7: C1b).
  ORP-cal  INN with ORP's payload-era angle scales and no inner material: ORP's calibrated `innovation`
        arm, which the engine reproduces bit for bit (E2 / E3, E6).

Angle use is a set of (consumer, angle, region): consumers window (a cut before ranking), rank, gate
(the forward hit gate), rr (reject-and-refit) and update.

Selections: FREE (run_arm: the arm chooses) or FORCED (refit_forced: (module, cluster) given per visit
step: reject-and-refit's leave-one-out and final refits, E5's first-flip refits). K = 2 + skip branches
at the first-visited surface into virtual tracks (best, runner-up, skip), each continued freely; the
branch with the lowest score wins (PLAN §10 PD2): the sum of the hit-gate statistic over its updates
plus, for each visited surface it leaves empty, that bin's hit-gate 99 % cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import awkward as ak
import numpy as np

import angle_import as AI
import common as CM
import engine as E
from common import OP, ORP
from ngtagger.data import spix_nano as SPN
from ngtagger.spdefs import angles as SA

ANGLE_NAMES = ("alpha", "beta")
CONSUMERS = ("window", "rank", "gate", "rr", "update")
REGIONS = ("barrel", "disc")
# inner material (PLAN §1 item 4): planes in the calibration's [r_cm or "ILj", x/X0, what] format
BEAM_PIPE = [2.3, 0.0022761, "beam pipe: tkLayout global_constants.hh:71 (x/X0), Analyzer.cc:224 (r = 23 mm)"]
IL1_MODULE = ["IL1", 0.018519, "TBPX L1 module: trackerRecoMaterial.xml (tkLayout RL)"]
EXTRA_CL_COLS = ("tpEmuClass", "tpLocalCotAlpha", "tpLocalCotBeta") + tuple(
    c for c in AI.GLOBAL_COLS if c != "tpEmuClass" and c not in OP.CL_COLS)      # beyond prepare_chunk's
TP_COLS = ("pt", "eta", "phi0", "d0", "z0", "charge", "vx", "vy", "genFromB", "genFromC", "evt", "bx")
CALIBRATION = str(CM.COMB / "results" / "refit_calibration" / "refit_calibration_v3.json")


def use_set(consumers=("rank", "update"), angles=ANGLE_NAMES, regions=REGIONS):
    return frozenset((c, a, r) for c in consumers for a in angles for r in regions)


def producer_use(cfg):
    """The producer's angle use: its useAngles in the selection metric and in the update."""
    ang = {"alphaBeta": ANGLE_NAMES, "alpha": ("alpha",)}.get(cfg["useAngles"], ())
    return use_set(("rank", "update"), ang)


@dataclass(frozen=True)
class Arm:
    name: str
    crossing: str = "f32fma"        # the producer-Jacobian arithmetic (engine.ForcedOpts.crossing)
    jac: str = "cmssw"              # "cmssw": the producer's one-sided + same-module guard; "central": OP's
    window: str = "box"             # "box": the producer's module and boxes; "ellipse": nsig, all modules
    nsig: float = 4.0
    metric: str = "producer"        # "producer" | "innovation"
    seed_cov: str = "reported"      # "reported" | "calibrated"
    scattering: str = "producer"    # "producer" | "highland"
    inner_material: bool = False
    pos_err: str = "published"      # "published" | "rtables"
    ang_err: str = "published"      # "published" | "payload" (ORP's apply_R) | "table"
    use: frozenset = field(default_factory=use_set)
    k_alpha: float = 1.0
    k_beta: float = 1.0
    k_mode: str = "fixed"
    K: int = 1                      # 1 | 2 (best, runner-up, skip at the first-visited surface)
    gate: bool = False
    rr: bool = False
    rr_max_iter: int = 2
    discs: bool = True
    cap: bool = True                # measured-angle cap |cot| <= 12 (False: the E5 known-bad)
    truncate: bool = True           # maxHitsPerWindow (False: the E5 known-bad)
    empty_recharge: bool = True     # producer Q also after an empty window (False: the E5 known-bad)
    k_score: str = "penalty"        # K's comparison: "penalty" (PD2) | "tailprob" (the PD2 known-bad)
    table_variant: str = ""         # E1(a) known-bads: "global_sign" | "angle_units" | "folded"
    material_scale: float = 1.0     # E3 known-bad: 0.5 (half-strength calibrated material)
    rtable_swap: bool = False       # E2 known-bad: the R-table lookup with |cotA| and |cotB| swapped

    @property
    def cov_model(self) -> str:
        return "calibrated" if self.seed_cov == "calibrated" else "producer"

    def uses(self, consumer, angle, region) -> bool:
        return (consumer, angle, region) in self.use

    def dims_key(self, consumer) -> str:
        """The cut-file key of a chi2 consumer: its angle dims on the barrel ("xy", "xya", "xyb", "xyab")."""
        return "xy" + "".join(a[0] for a in ANGLE_NAMES if self.uses(consumer, a, "barrel"))


def pm(cfg, **kw):
    return Arm(kw.pop("name", "PM"), use=producer_use(cfg), **kw)


def pm_b(cfg, **kw):
    return Arm(kw.pop("name", "PM-B"), use=producer_use(cfg), discs=False, **kw)


def inn(cfg, **kw):
    base = dict(crossing="f64", jac="central", window="ellipse", metric="innovation", seed_cov="calibrated",
                scattering="highland", inner_material=True, pos_err="rtables", ang_err="table",
                use=use_set(("rank", "update"), ANGLE_NAMES, ("barrel",)), discs=False)
    base.update(kw)
    return Arm(base.pop("name", "INN"), **base)


def orp_cal(cfg, **kw):
    """ORP's calibrated `innovation` arm (the E2 / E3 identity configuration)."""
    ang = {"alphaBeta": ANGLE_NAMES, "alpha": ("alpha",)}[cfg["useAngles"]]
    return inn(cfg, name=kw.pop("name", "ORP-cal"), ang_err="payload", inner_material=False,
               use=use_set(("rank", "update"), ang, ("barrel",)), **kw)


# ---------------------------------------------------------------- inputs ----
def branches(path):
    keys = OP._keys(str(path))
    b = list(E.branches(path))
    b += SPN.cluster_branches(keys, list(EXTRA_CL_COLS) + ["hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta"],
                              disc_scope="all", cap_angles=False)
    b += SPN.branch_names(keys, "L1TTrack", ["tpIdx"])
    b += SPN.branch_names(keys, SPN.TP_TABLE, list(TP_COLS))
    return list(dict.fromkeys(b))


def prepare(A, geo):
    """OP.prepare_chunk plus the global / truth cluster columns, the UNCAPPED has-flags (the cap
    known-bad), the OT track's TP (L1TTrack_tpIdx: SP-track truth, truth.sp_track) as a global L1TTP row
    and the L1TTP columns the labels and the helix-nearest report need."""
    ctx = OP.prepare_chunk(A, geo)
    K = ctx["K"]
    J = SPN.cluster_columns(A, list(EXTRA_CL_COLS), disc_scope="all")
    for c in EXTRA_CL_COLS:
        K[c] = ak.to_numpy(ak.flatten(J[c]))
    R = SPN.cluster_columns(A, ["hasAlpha", "hasBeta"], disc_scope="all", cap_angles=False)
    K["hasAlpha_raw"] = ak.to_numpy(ak.flatten(R["hasAlpha"]))
    K["hasBeta_raw"] = ak.to_numpy(ak.flatten(R["hasBeta"]))
    ntp = ak.to_numpy(ak.num(A[f"{SPN.TP_TABLE}_pt"]))
    tpoff = np.concatenate([[0], np.cumsum(ntp)])
    ctx["TP"] = {c: ak.to_numpy(ak.flatten(A[f"{SPN.TP_TABLE}_{c}"])).astype(np.float64) for c in TP_COLS}
    tp = ak.to_numpy(ak.flatten(A["L1TTrack_tpIdx"])).astype(np.int64)
    ctx["trk_tp"] = np.where(tp >= 0, tpoff[ctx["tev"]] + tp, -1)                 # global L1TTP row
    kt = K["tpIdx"].astype(np.int64)
    K["tp_global"] = np.where(kt >= 0, tpoff[K["event"]] + kt, -1)
    K["mi"] = OP.module_index(geo, K["detId"])
    ctx["coff"] = np.concatenate([[0], np.cumsum(K["_n"])])
    ctx["tpoff"] = tpoff
    return ctx


class Tools:
    """What the arms read besides the event: the producer configuration, the refit calibration (seed,
    material, R tables), the angle table and the cut file (cuts.Cuts)."""

    def __init__(self, cfg, calibration=CALIBRATION, angle_table=AI.TABLE, cuts=None):
        self.cfg = cfg
        self.cal = ORP.load_calibration(calibration) if calibration else None
        self.angles = AI.AngleImport(angle_table) if angle_table else None
        self.cuts = cuts
        self._folded = None

    def folded(self):
        if self._folded is None:
            self._folded = AI.folded(self.angles)
        return self._folded


# ---------------------------------------------------------------- state ----
REC_KEYS = {"mi": -1, "ci": -1}


class State:
    """Virtual tracks: o[v] the original track; per v the helix state, the visit bookkeeping and the
    per-step record (module, cluster, accepted, chi2 per scalar, the gate statistic, window size)."""

    def __init__(self, a, C, o, seq):
        n, ns = len(o), seq.shape[1]
        self.o, self.a, self.C, self.seq = o, a, C, seq
        self.nUpd = np.zeros(n, np.int64)
        self.prevR = np.full(n, -1.0)
        self.prevNT = np.zeros(n, np.int64)
        self.Lv = np.full(n, 5, np.int64)                  # calibrated material: last visited (5 = the OT)
        self.branch = np.zeros(n, np.int8)                 # K: 0 best, 1 runner-up, 2 skip
        self.rec = {"mi": np.full((n, ns), -1, np.int64), "ci": np.full((n, ns), -1, np.int64),
                    "acc": np.zeros((n, ns), bool), "visited": np.zeros((n, ns), bool),
                    "chi2": np.zeros((n, ns, 4)), "gstat": np.full((n, ns), np.nan),
                    "ncand": np.zeros((n, ns), np.int16), "trunc": np.zeros((n, ns), bool),
                    "gated": np.zeros((n, ns), bool), "surface": np.full((n, ns), -1, np.int64)}

    ARRAYS = ("o", "a", "C", "seq", "nUpd", "prevR", "prevNT", "Lv", "branch")

    def take(self, idx):
        s = State.__new__(State)
        for k in self.ARRAYS:
            setattr(s, k, getattr(self, k)[idx].copy())
        s.rec = {k: v[idx].copy() for k, v in self.rec.items()}
        return s

    def extend(self, other):
        for k in self.ARRAYS:
            setattr(self, k, np.concatenate([getattr(self, k), getattr(other, k)]))
        for k in self.rec:
            self.rec[k] = np.concatenate([self.rec[k], other.rec[k]])


def seed(ctx, cfg, arm, tools, o):
    """The starting state of tracks o: the OT helix; the reported covariance after the producer's d0
    floor, plus the calibrated deficit for the calibrated arms."""
    a = ctx["a"][o].copy()
    C, _ = E.d0_floor(ctx["C"][o], cfg)
    if arm.seed_cov == "calibrated":
        kpt = ORP.pt_constant(ctx["T"])
        C = ORP.seed_cov(tools.cal, C, a[:, 2], ctx["n_disk_stubs"][o], ctx["T"]["nStubs"][o],
                         kpt / np.maximum(np.abs(a[:, 0]), 1e-12))
    return a, C


def visit_plan(ctx, geo, cfg, mask, arm):
    """(path, seq): the OT helix's path and the visit sequence over the arm's active surfaces."""
    path = ORP.surface_path(ctx["a"], geo)
    active = [s for s in ORP.active_surfaces(mask) if s in geo["surfaces"]
              and (arm.discs or not SPN.is_disc_surface(s))]
    return path, ORP.visit_sequence(path, active, cfg["layerOrder"])


# ---------------------------------------------------------------- measurements ----
def angle_dims(arm, consumer, surface, has):
    """(n, 2) bool: the angle dims (alpha, beta) a consumer uses per cluster (has: usable angles)."""
    disc = SPN.is_disc_surface(np.asarray(surface))
    out = np.zeros(has.shape, bool)
    for i, ang in enumerate(ANGLE_NAMES):
        out[:, i] = has[:, i] & np.where(disc, arm.uses(consumer, ang, "disc"), arm.uses(consumer, ang, "barrel"))
    return out


def measurements(ctx, geo, ci, mi, cot_a_pred, cot_b_pred, arm, tools):
    """Per cluster: meas (n, 4) = (x, y, cotAlpha', cotBeta') after the knob and the table centre, sig
    (n, 4) per the arm's error switches, has (n, 2) the usable angles (capped unless arm.cap is False)."""
    K = ctx["K"]
    pub = np.stack([K["sigX"][ci], K["sigY"][ci], K["sigAlpha"][ci], K["sigBeta"][ci]], 1).astype(np.float64)
    meas = np.stack([K["localX"][ci], K["localY"][ci], K["localCotAlpha"][ci], K["localCotBeta"][ci]],
                    1).astype(np.float64)
    hk = ("hasAlpha", "hasBeta") if arm.cap else ("hasAlpha_raw", "hasBeta_raw")
    has = np.stack([K[hk[0]][ci] > 0, K[hk[1]][ci] > 0], 1)
    sig = pub.copy()
    if arm.pos_err == "rtables" or arm.ang_err == "payload":
        import refit_calibration_fit as RCF
        ca, cb = (cot_b_pred, cot_a_pred) if arm.rtable_swap else (cot_a_pred, cot_b_pred)
        Rt = RCF.apply_R(tools.cal["R_tables"], K["sizeX"][ci], K["sizeY"][ci], ca, cb, pub)
        if arm.pos_err == "rtables":
            sig[:, :2] = Rt[:, :2]
        if arm.ang_err == "payload":
            sig[:, 2:] = Rt[:, 2:]
    surface = geo["surface"][mi]
    knobbed = arm.k_alpha != 1.0 or arm.k_beta != 1.0
    if knobbed:
        cols = {c: K[c][ci] for c in AI.GLOBAL_COLS}
        meas[:, 2], meas[:, 3] = AI.knob(cols, geo["ex"][mi], geo["ey"][mi], geo["ez"][mi], surface,
                                         arm.k_alpha, arm.k_beta, meas[:, 2], meas[:, 3])
    if arm.ang_err == "table":
        imp = tools.angles if arm.table_variant != "folded" else tools.folded()
        sg = None
        if arm.table_variant == "angle_units":
            cols = {c: K[c][ci] for c in AI.GLOBAL_COLS}
            sg = (SA.sigma_of("alpha", cols), SA.sigma_of("beta", cols))
        ma, mb, sa, sb, _ = AI.calibrate(imp, surface, geo["detId"][mi], cot_a_pred, cot_b_pred,
                                         meas[:, 2], meas[:, 3], pub[:, 2], pub[:, 3], arm.k_alpha, arm.k_beta,
                                         arm.k_mode, variant=arm.table_variant if arm.table_variant != "folded" else "",
                                         sig_global=sg)
        meas[:, 2], meas[:, 3], sig[:, 2], sig[:, 3] = ma, mb, sa, sb
    elif knobbed:
        disc = SPN.is_disc_surface(surface)
        for i, k in ((2, arm.k_alpha), (3, arm.k_beta)):
            sig[:, i] = np.where(disc, sig[:, i], SA.sigma_k(sig[:, i], k, arm.k_mode))
    return meas, sig, has


def innovation_chi2(r, Sm, sig, dims):
    """r^T (S + diag sig^2)^-1 r over the used dims (n, 4) bool; unused dims are decoupled."""
    M = Sm.copy()
    M[:, np.arange(4), np.arange(4)] += np.where(dims, sig, 1.0) ** 2
    miss = ~dims
    M[np.repeat(miss[:, :, None], 4, 2)] = 0.0
    M[np.repeat(miss[:, None, :], 4, 1)] = 0.0
    M[:, np.arange(4), np.arange(4)] = np.where(miss, 1.0, M[:, np.arange(4), np.arange(4)])
    rz = np.where(dims, np.nan_to_num(r), 0.0)
    with np.errstate(invalid="ignore"):
        x = np.linalg.solve(M, rz[:, :, None])[:, :, 0]
    return (rz * x).sum(1)


def rank_values(metric, r, Sm, sig, dims):
    """The ranking value per candidate (lower is better). producer: sum (r / sig)^2 over the used dims
    (the selection metric of L1SmartPixelsTrackProducer.cc:2012-2019); innovation: ORP.rank's innovation
    chi2, a missing dim replaced by its expectation (1)."""
    if metric == "producer":
        return np.where(dims, (np.nan_to_num(r) / np.where(dims, sig, 1.0)) ** 2, 0.0).sum(1)
    have = np.isfinite(r) & (sig > 0) & dims
    return innovation_chi2(r, Sm, sig, have) + ORP.NEUTRAL * (~have).sum(1)


# ---------------------------------------------------------------- windows ----
def box_candidates(ctx, cfg, ev_of, det, surf, u, v, truncate=True):
    """The producer's window collect on one module per crossing (L1SmartPixelsTrackProducer.cc:1960-2010):
    the module's clusters in readout order (the cluster table's row order), those inside the box
    |x - u| <= w_rphi, |y - v| <= w_z|r, the first maxHitsPerWindow kept, truncated when another cluster
    of the module follows the last one kept. ORP.selection_closure's logic, returning every kept
    candidate as (crossing index, cluster row, readout position); and truncated per crossing."""
    K = ctx["K"]
    ckey = K["event"].astype(np.int64) * (1 << 32) + K["detId"].astype(np.int64)
    o = ctx["ko"]                                             # argsort of the same key (prepare_chunk)
    cs = ckey[o]
    xkey = ev_of.astype(np.int64) * (1 << 32) + det.astype(np.int64)
    lo, hi = np.searchsorted(cs, xkey, "left"), np.searchsorted(cs, xkey, "right")
    n = hi - lo
    xi = np.repeat(np.arange(len(det)), n)
    ramp = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ci = o[np.repeat(lo, n) + ramp]
    wr = np.zeros(len(det))
    wz = np.zeros(len(det))
    for S in np.unique(surf):
        m = surf == S
        wr[m], wz[m] = ORP.window(cfg, int(S))
    dx = K["localX"][ci].astype(np.float64) - u[xi]
    dy = K["localY"][ci].astype(np.float64) - v[xi]
    inwin = (np.abs(dx) <= wr[xi]) & (np.abs(dy) <= wz[xi])
    cin = np.cumsum(inwin)
    base = np.repeat(np.r_[0, np.cumsum(np.bincount(xi, weights=inwin, minlength=len(det)))[:-1]], n)
    rk = cin - base - 1
    if not truncate:
        return xi[inwin], ci[inwin], ramp[inwin], np.zeros(len(det), bool)
    nmax = cfg["maxHitsPerWindow"]
    kept = inwin & (rk < nmax)
    last = inwin & (rk == nmax - 1)
    pos_last = np.full(len(det), -1, np.int64)
    pos_last[xi[last]] = ramp[last]
    return xi[kept], ci[kept], ramp[kept], (pos_last >= 0) & (pos_last < n - 1)


def window_keep(z, dims, cut):
    """The angle window cut: a candidate survives unless an angle the consumer uses lies beyond its cut
    (z, dims, cut: (n, 2) per angle; an unusable angle never removes a candidate)."""
    return ~((dims & (np.nan_to_num(z, nan=0.0) > cut)).any(1))


def _vctx(ctx, o):
    """ctx re-indexed to virtual tracks o for OP.evaluate_surface (the B1 keys re-packed)."""
    import tracklet_topology_cost as M
    v = dict(ctx)
    v.update(tev=ctx["tev"][o], ntrk=len(o), maj_tp=ctx["maj_tp"][o], cls=ctx["cls"][o], T={"pt": ctx["T"]["pt"][o]})
    t, tp = M.tp_key_parts(ctx["foreign_keys"])
    pos = {}
    for vi, oi in enumerate(o):
        pos.setdefault(int(oi), []).append(vi)
    vt = [(vi, p) for ti, p in zip(t.tolist(), tp.tolist()) for vi in pos.get(int(ti), [])]
    v["foreign_keys"] = (np.unique(M.tp_key(np.array([x[0] for x in vt], np.int64),
                                            np.array([x[1] for x in vt], np.int64))) if vt
                         else np.zeros(0, np.int64))
    return v


def project(arm, a, C, S, geo, mi, kpt):
    """The arm's projection of states (a, C) onto modules mi: OP's central Jacobian, or the producer's
    (engine.project, crossing arithmetic per the arm)."""
    s0 = OP.surface_s(a, geo, S)
    if arm.jac == "central":
        pj = OP.project_with_cov(a, C, s0, geo, mi)
        pj["gr"] = None
        return pj
    return E.project(a, C, s0, geo, mi, S, E.ForcedOpts(crossing=arm.crossing), kpt)


# ---------------------------------------------------------------- the step ----
def _q_step(arm, st, t, S, path, geo, tools, kpt, cfg):
    """Process noise before predicting surface S: the producer's Q (at every visited surface after the
    first update, the empty-window recharge included), or the calibrated material."""
    if arm.scattering == "highland":
        cal = tools.cal if arm.material_scale == 1.0 else dict(tools.cal, material_scale=tools.cal["material_scale"]
                                                                * arm.material_scale)
        ORP.add_material(st.a, st.C, t, S, st.Lv, geo, cal, kpt)
        st.Lv[t] = S
    elif cfg["applyProcessNoise"]:
        q = t[(st.nUpd[t] > 0) & (st.prevR[t] > 0)]
        if not arm.empty_recharge and len(q):
            # known-bad: no recharge at a surface that follows an empty window
            last = np.array([np.flatnonzero(st.rec["visited"][x])[-1] if st.rec["visited"][x].any() else -1
                             for x in q])
            q = q[(last < 0) | st.rec["acc"][q, np.maximum(last, 0)]]
        ORP.add_process_noise(st.a, st.C, q, ORP.n_traversed(path, st.o[q], S), st.prevNT[q], st.prevR[q],
                              kpt, cfg["multScattTerm"])


def _apply(arm, st, j, t, mi, ci, h0, J, gr, ctx, geo, tools, path, S):
    """Update virtual tracks t on clusters ci (predictions h0, Jacobians J); bookkeeping and record."""
    if not len(t):
        return
    meas, sig, has = measurements(ctx, geo, ci, mi, h0[:, 2], h0[:, 3], arm, tools)
    ud = np.concatenate([np.ones((len(t), 2), bool), angle_dims(arm, "update", geo["surface"][mi], has)], 1)
    _, chi2, _ = E.kf_update(st.a, st.C, t, J, h0, meas, sig, ud,
                             gate=tools.cfg.get("chi2UpdateGate", ORP.CHI2_UPDATE_GATE))
    if gr is not None:
        st.prevR[t] = gr
    else:
        g = geo["origin"][mi] + h0[:, 0][:, None] * geo["ex"][mi] + h0[:, 1][:, None] * geo["ey"][mi]
        st.prevR[t] = np.hypot(g[:, 0], g[:, 1])
    st.nUpd[t] += 1
    st.prevNT[t] = ORP.n_traversed(path, st.o[t], S)
    st.rec["mi"][t, j], st.rec["ci"][t, j] = mi, ci
    st.rec["acc"][t, j] = True
    st.rec["chi2"][t, j] = chi2


def gate_stat(arm, consumer, ctx, geo, ci, mi, h0, Sm, tools):
    """A chi2 consumer's statistic per cluster: the innovation chi2 over positions and the consumer's
    angle dims, at the prediction (h0, Sm = J C J^T with the guard-zeroed J)."""
    meas, sig, has = measurements(ctx, geo, ci, mi, h0[:, 2], h0[:, 3], arm, tools)
    dims = np.concatenate([np.ones((len(ci), 2), bool), angle_dims(arm, consumer, geo["surface"][mi], has)], 1)
    return innovation_chi2(meas - h0, Sm, sig, dims)


def window_z(arm, ctx, geo, ci, mi, h0, Sm, tools):
    """|innovation-normalised angle pull| (n, 2) of the window-cut consumer (NaN where unusable)."""
    meas, sig, has = measurements(ctx, geo, ci, mi, h0[:, 2], h0[:, 3], arm, tools)
    z = np.full((len(ci), 2), np.nan)
    for i in (0, 1):
        with np.errstate(invalid="ignore"):
            z[:, i] = np.abs(meas[:, 2 + i] - h0[:, 2 + i]) / np.sqrt(sig[:, 2 + i] ** 2 + Sm[:, 2 + i, 2 + i])
    return np.where(has, z, np.nan)


def _cands(arm, st, j, t, S, ctx, geo, tools, kpt, cfg):
    """The crossings of virtual tracks t on surface S and their candidates.

    Returns (crossings, cands): crossings {t, mi, h0 (n,4), J (n,4,5), Sz (n,4,4), gr} for the valid
    crossings; cands {x (crossing index), ci, mi (the candidate's module), h0, J, Sz, gr (per
    candidate), val (ranking value), order (readout position / candidate order)}; plus per crossing
    the window size and truncation."""
    if arm.window == "box":
        mi = OP.cross_module(st.a[t], S, geo)
        good = mi >= 0
        t, mi = t[good], mi[good]
        if not len(t):
            return None
        pj = project(arm, st.a[t], st.C[t], S, geo, mi, kpt)
        pmax = tools.cfg.get("predAngleMaxAbs", ORP.PRED_ANGLE_MAX_ABS)
        ok = np.isfinite(pj["u"]) & (np.abs(pj["cotA"]) <= pmax) & (np.abs(pj["cotB"]) <= pmax)
        idx = np.flatnonzero(ok)
        t, mi = t[idx], mi[idx]
        if not len(t):
            return None
        h0 = np.stack([pj["u"][idx], pj["v"][idx], pj["cotA"][idx], pj["cotB"][idx]], 1)
        J = pj["J"][idx]
        Jz = E.guard_zeroed(J)
        Sz = np.einsum("nij,njk,nlk->nil", Jz, st.C[t], Jz)
        gr = pj["gr"][idx] if pj.get("gr") is not None else None
        xi, ci, order, trunc = box_candidates(ctx, cfg, ctx["tev"][st.o[t]], geo["detId"][mi], geo["surface"][mi],
                                              h0[:, 0], h0[:, 1], arm.truncate)
        cmi = mi[xi]
        ch0, cJ, cSz = h0[xi], J[xi], Sz[xi]
        cgr = gr[xi] if gr is not None else None
        S_rank = cSz
    else:
        on = np.zeros(len(st.o), bool)
        on[t] = True
        ev = OP.evaluate_surface(_vctx(ctx, st.o), st.a, st.C, S, geo, arm.nsig, on)
        if ev is None:
            return None
        c = ev["cand"]
        keep = c["inside"]
        pj = ev["pj"]
        p = c["pair"][keep]
        tt = c["t"][keep]
        ci = c["ci"][keep]
        cmi = ev["pm"][p]
        ch0 = np.stack([pj["u"][p], pj["v"][p], pj["cotA"][p], pj["cotB"][p]], 1)
        cJ = pj["J"][p]
        cSz = np.einsum("nij,njk,nlk->nil", E.guard_zeroed(cJ), st.C[tt], E.guard_zeroed(cJ))
        S_rank = pj["S"][p]                                       # ORP.rank uses the unguarded S
        cgr = None
        # crossings = the tracks with at least one in-ellipse candidate (one record per track)
        t_u, first = np.unique(tt, return_index=True)
        xmap = {int(v): i for i, v in enumerate(t_u)}
        xi = np.array([xmap[int(v)] for v in tt], np.int64)
        order = np.arange(len(tt))
        t, mi = t_u, cmi[first]
        h0 = ch0[first]
        trunc = np.zeros(len(t), bool)
    meas, sig, has = measurements(ctx, geo, ci, cmi, ch0[:, 2], ch0[:, 3], arm, tools)
    r = meas - ch0
    if arm.window == "ellipse":                                  # ORP's residuals: NaN for an unusable angle
        r[:, 2] = np.where(has[:, 0], r[:, 2], np.nan)
        r[:, 3] = np.where(has[:, 1], r[:, 3], np.nan)
    keep = np.ones(len(ci), bool)
    if any(arm.uses("window", a, rg) for a in ANGLE_NAMES for rg in REGIONS):
        z = window_z(arm, ctx, geo, ci, cmi, ch0, cSz, tools)
        dims = angle_dims(arm, "window", geo["surface"][cmi], has)
        surf_c = geo["surface"][cmi]
        orient = tools.angles.table.orientation(geo["detId"][cmi])
        kb = SA.fine_bin(surf_c, SA.key_from_local(surf_c, ch0[:, 2], ch0[:, 3], orient))
        cut = tools.cuts.window(arm.cov_model, surf_c, np.full(len(ci), j == 0), kb, kb)
        keep &= window_keep(z, dims, cut)
    rdims = np.concatenate([np.ones((len(ci), 2), bool), angle_dims(arm, "rank", geo["surface"][cmi], has)], 1)
    val = rank_values(arm.metric, r, S_rank, sig, rdims)
    val = np.where(np.isfinite(val), val, np.inf)
    ncand = np.bincount(xi[keep], minlength=len(t))
    return {"t": t, "mi": mi, "trunc": trunc, "ncand": ncand}, \
           {"x": xi[keep], "ci": ci[keep], "mi": cmi[keep], "h0": ch0[keep], "J": cJ[keep], "Sz": cSz[keep],
            "gr": (cgr[keep] if cgr is not None else None), "val": val[keep], "order": order[keep]}


def _ranked(cands, nx):
    """Per crossing: the indices into cands of the best and the runner-up (-1 none); ties by the
    candidate order (readout order in a box; ORP's track-sorted order in an ellipse)."""
    o = np.lexsort((cands["order"], cands["val"], cands["x"]))
    xs = cands["x"][o]
    first = np.r_[True, xs[1:] != xs[:-1]] if len(xs) else np.zeros(0, bool)
    best = np.full(nx, -1, np.int64)
    best[xs[first]] = o[first]
    second_mask = np.r_[False, ~first[1:] & first[:-1]] if len(xs) else np.zeros(0, bool)
    runner = np.full(nx, -1, np.int64)
    runner[xs[second_mask]] = o[second_mask]
    return best, runner


def _choose_apply(arm, st, j, X, cands, pick, ctx, geo, tools, path, S, kpt):
    """Apply the picked candidate per crossing (pick: index into cands or -1), with the forward gate."""
    rows = np.flatnonzero(pick >= 0)
    if not len(rows):
        return
    k = pick[rows]
    t = X["t"][rows]
    ci, mi = cands["ci"][k], cands["mi"][k]
    h0, J, Sz = cands["h0"][k], cands["J"][k], cands["Sz"][k]
    g = gate_stat(arm, "gate", ctx, geo, ci, mi, h0, Sz, tools)
    st.rec["gstat"][t, j] = g
    ok = np.ones(len(t), bool)
    if arm.gate:
        cut = tools.cuts.chi2("gate", arm.dims_key("gate"), arm.cov_model, geo["surface"][mi], np.full(len(t), j == 0))
        ok = ~(g > cut)
        st.rec["gated"][t[~ok], j] = True
        st.rec["mi"][t[~ok], j], st.rec["ci"][t[~ok], j] = mi[~ok], ci[~ok]
    gr = cands["gr"][k] if cands["gr"] is not None else None
    if arm.window == "ellipse":                                  # ORP.update_on's predicted-angle guard
        ok &= (np.abs(h0[:, 2]) <= ORP.PRED_ANGLE_MAX_ABS) & (np.abs(h0[:, 3]) <= ORP.PRED_ANGLE_MAX_ABS)
    _apply(arm, st, j, t[ok], mi[ok], ci[ok], h0[ok], J[ok], (gr[ok] if gr is not None else None),
           ctx, geo, tools, path, S)


def step(arm, st, j, t, S, ctx, geo, tools, kpt, cfg, path, own_rows=None):
    """One (visit step, surface): Q, crossings, candidates, (K), gate, update; with own_rows (a list),
    the own-cluster diagnostics are taken after Q and before the update. Returns the State (K adds
    virtual tracks)."""
    _q_step(arm, st, t, S, path, geo, tools, kpt, cfg)
    if own_rows is not None:
        import diagnostics as DG
        own_rows.append(DG.own_rows(arm, st, j, t, S, ctx, geo, tools, kpt))
    st.rec["visited"][t, j] = True
    st.rec["surface"][t, j] = S
    got = _cands(arm, st, j, t, S, ctx, geo, tools, kpt, cfg)
    if got is None:
        return st
    X, cands = got
    st.rec["ncand"][X["t"], j] = X["ncand"]
    st.rec["trunc"][X["t"], j] = X["trunc"]
    best, runner = _ranked(cands, len(X["t"]))
    if arm.K == 2 and j == 0:
        has1 = np.flatnonzero(best >= 0)
        has2 = np.flatnonzero(runner >= 0)
        n0 = len(st.o)
        ru = st.take(X["t"][has2])
        ru.branch[:] = 1
        sk = st.take(X["t"][has1])
        sk.branch[:] = 2
        st.extend(ru)
        st.extend(sk)
        _choose_apply(arm, st, j, {"t": np.arange(n0, n0 + len(has2))}, cands, runner[has2],
                      ctx, geo, tools, path, S, kpt)
    _choose_apply(arm, st, j, X, cands, best, ctx, geo, tools, path, S, kpt)
    return st


# ---------------------------------------------------------------- the run ----
def run_arm(ctx, geo, cfg, mask, arm, tools, tracks=None, own=False, loo=False):
    """One arm on one chunk (free selection). tracks: bool per track (default: a covariance, as the
    producer's otCovOK). own: also record the own-cluster diagnostics (diagnostics.own_rows; K = 1);
    loo: also the leave-one-out chi2 of every accepted hit for all angle-dims sets (the rr cut fit)."""
    if own and arm.K != 1:
        raise ValueError("the own-cluster diagnostics are recorded with K = 1")
    if tracks is None:
        tracks = np.trace(ctx["C"], axis1=1, axis2=2) > 0
    o = np.flatnonzero(tracks)
    kpt = ORP.pt_constant(ctx["T"])
    path, seq = visit_plan(ctx, geo, cfg, mask, arm)
    a, C = seed(ctx, cfg, arm, tools, o)
    st = State(a, C, o.copy(), seq[o])
    own_rows = [] if own else None
    for j in range(seq.shape[1]):
        for S in np.unique(st.seq[:, j][st.seq[:, j] >= 0]):
            S = int(S)
            t = np.flatnonzero((st.seq[:, j] == S) & (st.nUpd < cfg["maxKFUpdates"]))
            if len(t):
                st = step(arm, st, j, t, S, ctx, geo, tools, kpt, cfg, path, own_rows)
    return finish(arm, st, ctx, geo, cfg, mask, tools, kpt, path, seq, own_rows, loo)


def refit_forced(arm, ctx, geo, cfg, tools, o, seq, sel_mi, sel_ci, path, kpt):
    """Refit tracks o (may repeat) from their seeds on the given (module, cluster) per visit step
    (-1: no hit), with the arm's Q, projection and update; no gate. Returns the State."""
    a, C = seed(ctx, cfg, arm, tools, o)
    st = State(a, C, np.asarray(o).copy(), seq)
    for j in range(seq.shape[1]):
        for S in np.unique(seq[:, j][seq[:, j] >= 0]):
            S = int(S)
            t = np.flatnonzero((seq[:, j] == S) & (st.nUpd < cfg["maxKFUpdates"]))
            if not len(t):
                continue
            _q_step(arm, st, t, S, path, geo, tools, kpt, cfg)
            st.rec["visited"][t, j] = True
            st.rec["surface"][t, j] = S
            u = t[sel_ci[t, j] >= 0]
            if not len(u):
                continue
            mi = sel_mi[u, j]
            pj = project(arm, st.a[u], st.C[u], S, geo, mi, kpt)
            fin = np.isfinite(pj["u"])
            u, mi = u[fin], mi[fin]
            h0 = np.stack([pj["u"][fin], pj["v"][fin], pj["cotA"][fin], pj["cotB"][fin]], 1)
            gr = pj["gr"][fin] if pj.get("gr") is not None else None
            _apply(arm, st, j, u, mi, sel_ci[u, j], h0, pj["J"][fin], gr, ctx, geo, tools, path, S)
    return st


DIMS = {"xy": (False, False), "xya": (True, False), "xyb": (False, True), "xyab": (True, True)}


def loo_stats(arm, ctx, geo, cfg, tools, st, v, path, kpt, all_dims=False):
    """Leave-one-out chi2 of every accepted hit of virtual tracks v (PD4 (c)): the full refit from the OT
    seed without the hit, its final state projected onto the hit's module, the innovation chi2 over
    positions and the rr angle dims (all_dims: a dict over DIMS, for the cut fit). Returns
    (owner virtual track, step, stat)."""
    vv, jj = np.nonzero(st.rec["acc"][v])
    if not len(vv):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), ({k: np.zeros(0) for k in DIMS} if all_dims else np.zeros(0))
    owner = v[vv]
    sel_mi = st.rec["mi"][owner].copy()
    sel_ci = np.where(st.rec["acc"][owner], st.rec["ci"][owner], -1)
    sel_ci[np.arange(len(vv)), jj] = -1
    R = refit_forced(arm, ctx, geo, cfg, tools, st.o[owner], st.seq[owner], sel_mi, sel_ci, path, kpt)
    mi = st.rec["mi"][owner, jj]
    ci = st.rec["ci"][owner, jj]
    keys = list(DIMS) if all_dims else [arm.dims_key("rr")]
    stat = {k: np.full(len(vv), np.nan) for k in keys}
    for S in np.unique(geo["surface"][mi]):
        m = np.flatnonzero(geo["surface"][mi] == S)
        pj = project(arm, R.a[m], R.C[m], int(S), geo, mi[m], kpt)
        h0 = np.stack([pj["u"], pj["v"], pj["cotA"], pj["cotB"]], 1)
        Jz = E.guard_zeroed(pj["J"])
        Sz = np.einsum("nij,njk,nlk->nil", Jz, R.C[m], Jz)
        meas, sig, has = measurements(ctx, geo, ci[m], mi[m], h0[:, 2], h0[:, 3], arm, tools)
        for k in keys:
            if all_dims:
                dm = np.concatenate([np.ones((len(m), 2), bool), has & np.array(DIMS[k])[None, :]], 1)
            else:
                dm = np.concatenate([np.ones((len(m), 2), bool), angle_dims(arm, "rr", geo["surface"][mi[m]], has)], 1)
            stat[k][m] = innovation_chi2(meas - h0, Sz, sig, dm)
    return owner, jj, (stat if all_dims else stat[keys[0]])


def finish(arm, st, ctx, geo, cfg, mask, tools, kpt, path, seq, own_rows, loo=False):
    """K resolution, reject-and-refit (with the scenario mask's rr cuts, PD15), inner material,
    passthrough; the per-track output."""
    ntrk = ctx["ntrk"]
    if arm.K == 2:
        st = st.take(resolve_k(arm, st, geo, tools))
    loo_out = None
    if loo:
        owner, jj, stat = loo_stats(arm, ctx, geo, cfg, tools, st, np.arange(len(st.o)), path, kpt, all_dims=True)
        loo_out = {"o": st.o[owner], "step": jj, "ci": st.rec["ci"][owner, jj], "mi": st.rec["mi"][owner, jj],
                   "surface": st.rec["surface"][owner, jj], "first": jj == 0, "dims": stat}
    removed = np.zeros((len(st.o), seq.shape[1]), bool)
    if arm.rr:
        for _ in range(arm.rr_max_iter):
            owner, jj, stat = loo_stats(arm, ctx, geo, cfg, tools, st, np.arange(len(st.o)), path, kpt)
            if not len(owner):
                break
            mi = st.rec["mi"][owner, jj]
            cut = tools.cuts.chi2("rr", arm.dims_key("rr"), arm.cov_model, geo["surface"][mi], jj == 0, scen=mask)
            ratio = stat / cut
            worst = np.full(len(st.o), -1, np.int64)
            o_ = np.lexsort((-ratio, owner))
            first = np.r_[True, owner[o_][1:] != owner[o_][:-1]]
            w = o_[first]
            w = w[ratio[w] > 1.0]
            if not len(w):
                break
            worst[owner[w]] = jj[w]
            hit = np.flatnonzero(worst >= 0)
            sel_mi = st.rec["mi"][hit].copy()
            sel_ci = np.where(st.rec["acc"][hit], st.rec["ci"][hit], -1)
            sel_ci[np.arange(len(hit)), worst[hit]] = -1
            removed[hit, worst[hit]] = True
            R = refit_forced(arm, ctx, geo, cfg, tools, st.o[hit], st.seq[hit], sel_mi, sel_ci, path, kpt)
            for k in ("a", "C", "nUpd"):
                getattr(st, k)[hit] = getattr(R, k)
            for k in ("acc", "chi2"):
                st.rec[k][hit] = R.rec[k]
    if arm.inner_material:
        inner_material(st, geo, tools, kpt)
    nacc = st.rec["acc"].sum(1)
    fill = {"mi": -1, "ci": -1, "surface": -1, "gstat": np.nan}
    out = {"a": ctx["a"].copy(), "C": ctx["C"].copy(), "nacc": np.zeros(ntrk, np.int64),
           "refit": np.zeros(ntrk, bool), "branch": np.full(ntrk, -1, np.int8),
           "rec": {k: np.full((ntrk,) + v.shape[1:], fill.get(k, 0), v.dtype) for k, v in st.rec.items()},
           "removed": np.zeros((ntrk, seq.shape[1]), bool), "seq": seq}
    ok = nacc >= cfg.get("minHits", 1)
    o = st.o
    out["a"][o[ok]], out["C"][o[ok]] = st.a[ok], st.C[ok]
    out["nacc"][o] = nacc
    out["refit"][o[ok]] = True
    out["branch"][o] = st.branch
    for k, v in st.rec.items():
        out["rec"][k][o] = v
    out["removed"][o] = removed
    if own_rows is not None:
        out["own"] = CM.concat([r for r in own_rows if r])
    if loo_out is not None:
        out["loo"] = loo_out
    return out


def resolve_k(arm, st, geo, tools):
    """Per original track, the virtual track with the lowest K score (PD2): the sum of the gate
    statistic over its updates plus, per visited surface without an update, the hit-gate cut of that
    bin. "tailprob" (the PD2 known-bad) instead maximises the chi2 tail probability of the gate
    statistic sum for its own number of degrees of freedom."""
    acc = st.rec["acc"]
    g = np.where(acc, np.nan_to_num(st.rec["gstat"]), 0.0).sum(1)
    if arm.k_score == "tailprob":
        from scipy import stats
        ndf = 2 * acc.sum(1)
        score = -stats.chi2.logsf(g, np.maximum(ndf, 1))
    else:
        vis = st.rec["visited"] & ~acc
        S = st.rec["surface"]
        first = np.zeros_like(vis)
        first[:, 0] = True
        cut = np.zeros(vis.shape)
        m = vis & (S >= 0)
        cut[m] = tools.cuts.chi2("gate", arm.dims_key("gate"), arm.cov_model, S[m], first[m])
        score = g + cut.sum(1)
    o = np.lexsort((st.branch, score, st.o))
    first = np.r_[True, st.o[o][1:] != st.o[o][:-1]]
    return o[first]


def inner_material(st, geo, tools, kpt):
    """The material between the innermost update and the beam line, charged to C only (PLAN §1 item 4):
    every calibration plane at r <= R of the innermost updated layer (its own module included: the
    kink there separates the inner segment from the outer measurements, as the forward pass charges a
    layer's module when predicting the next one), the IL1 module (absent from the calibration's list)
    and the beam pipe. A helix state is unchanged by material."""
    acc = st.rec["acc"]
    t = np.flatnonzero(acc.any(1))
    if not len(t):
        return
    last_j = np.array([np.flatnonzero(acc[x])[-1] for x in t])
    Lv = np.full(len(st.o), 5, np.int64)
    Lv[t] = st.rec["surface"][t, last_j]
    geo0 = dict(geo)
    geo0["R"] = dict(geo["R"])
    geo0["R"][0] = 0.0
    cal = dict(tools.cal)
    cal["planes"] = list(tools.cal["planes"]) + [IL1_MODULE, BEAM_PIPE]
    ORP.add_material(st.a, st.C, t, 0, Lv, geo0, cal, kpt)
