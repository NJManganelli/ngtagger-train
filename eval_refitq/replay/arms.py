"""The free-running replay: arms, switches and the visit loop (PLAN §1 items 3-6, §2, §10 PD1-PD3, PD17, PD18).

An ARM is one configuration of the switches of PLAN §2. The named arms of the entry gate:

  PM    the mirror of refit v2 (L1SmartPixelsTrackProducer, seedlessRefit): the producer's module choice
        (crossSurface, with its predicted-angle clamp), its Jacobian and float32 crossing arithmetic
        (engine "f32fma", PLAN §10 PD5 (a)), one module, box windows in readout order truncated at
        maxHitsPerWindow, its selection metric (positions and angles over the CLUSTER sigma), its Q with
        the empty-window recharge, maxKFUpdates, the d0 floor, the measured-angle cap (the cluster loader
        applies it, spix_nano.cluster_columns), discs; the cluster table as input. PM-B: no discs.
  INN   the calibrated arm: ellipse candidates (ot_projection.evaluate_surface, central Jacobian) within
        4 sigma (PD20: PD18's window as the 99 % hit cut failed its validation; it stays as the switch
        window_cut="gate99"), the innovation metric, the calibrated seed and Highland material
        (refit_calibration v3), the position R tables, the table's angle errors (angle_import, barrel), the inner material at the
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

from dataclasses import dataclass, field, replace

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
BEAM_PIPE_PLANE = {"kind": "cyl", "r": BEAM_PIPE[0], "half_z": np.inf, "xx0": BEAM_PIPE[1], "what": BEAM_PIPE[2]}
                                    # disc_material's form; every track from the beam line crosses it
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


WINDOW_SEARCH_NSIG = 25.0           # PD18: the search bound of the "gate99" window (prediction-covariance sigma);
                                    # the largest 99 % gate cut is 148 (12.2 sigma), the cluster error only adds


@dataclass(frozen=True)
class Arm:
    name: str
    crossing: str = "f32fma"        # the producer-Jacobian arithmetic (engine.ForcedOpts.crossing)
    jac: str = "cmssw"              # "cmssw": the producer's one-sided + same-module guard; "central": OP's
    window: str = "box"             # "box": the producer's module and boxes; "ellipse": nsig, all modules
    nsig: float = 4.0
    window_cut: str = "nsig"        # the ellipse's acceptance: "nsig" (m^2 < nsig^2 in the prediction covariance;
                                    # INN's, PD20) | "gate99" (the hit-gate statistic <= its 99 % cut, PD18, which
                                    # failed its validation; searched within WINDOW_SEARCH_NSIG)
    metric: str = "producer"        # "producer" | "innovation"
    seed_cov: str = "reported"      # "reported" | "calibrated" (v3's binned deficits) | "smooth" (seedmodel, A16)
    scattering: str = "producer"    # "producer" | "highland"
    inner_material: bool = False
    pos_err: str = "published"      # "published" | "rtables"
    ang_err: str = "published"      # "published" | "payload" (ORP's apply_R) | "table"
    use: frozenset = field(default_factory=use_set)
    k_alpha: float = 1.0
    k_beta: float = 1.0
    k_mode: str = "fixed"
    K: int | str = 1                # 1 | 2 (PD2: best, runner-up, skip at the first-visited surface) | "tree2" (PD14:
                                    # the two-layer branching hit choice with rolling commitment, scan plan A1)
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
    loo_pred: str = "smoothed"      # the leave-one-out prediction (PD17): "smoothed" (two-sided) | "final" (the
                                    # final refit state, the PD17 known-bad)
    disc_model: bool = False        # C1b: disc-going tracks get the path-ordered material and the disc rows
    disc_material: str = "path"     # C1b known-bads: "radius" (the barrel planes by radius only on disc-going
                                    # tracks), "barrel_factor" (the barrel path factor on disc planes),
                                    # "all_tracks" (the disc model's charging on barrel-only tracks too)
    inner_material_rule: str = "path"   # scan plan B-pre: "path" (disc-going tracks along their path) | "radius" (the
                                        # pre-fix rule, every track by radius: the known-bad)
    angle_steps: str = "all"        # scan plan B1: the visit steps whose consumers use angles: "all" | "first" (the
                                    # first visited surface only) | "later" (every visit but the first)
    angle_term: str = "gauss"       # scan plan B1: "gauss" | "t9" (the tail-robust term: each angle's sigma scaled by
                                    # sqrt(1 + z^2 / 9), z its residual over sigma, so its chi2 saturates at 9)
    truth_modules: bool = False     # the ellipse window also searches the modules holding the track's majority-owner
                                    # TP clusters (OP.evaluate_surface's study-19 target bookkeeping: TRUTH). False
                                    # (v3, since 2026-10-20): the geometric search only. The entry gate, C1b and the
                                    # 2026-10-19 scan ran with True

    @property
    def cov_model(self) -> str:
        return "calibrated" if self.seed_cov in ("calibrated", "smooth") else "producer"

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
    base = dict(crossing="f64", jac="central", window="ellipse", window_cut="nsig", metric="innovation",
                seed_cov="smooth",
                scattering="highland", inner_material=True, pos_err="rtables", ang_err="table",
                use=use_set(("rank", "update"), ANGLE_NAMES, ("barrel",)), discs=False)
    base.update(kw)
    return Arm(base.pop("name", "INN"), **base)


def inn_disc(cfg, **kw):
    """C1b's calibrated arm with discs (c1b/PLAN.md rev 4): INN plus, for disc-going tracks, the path-ordered
    extent-aware material (disc_material), the disc angle rows (disc_angles) and the disc position errors."""
    base = dict(name="INN-D", discs=True, disc_model=True, use=use_set(("rank", "update"), ANGLE_NAMES, REGIONS))
    base.update(kw)
    return inn(cfg, **base)


def orp_cal(cfg, **kw):
    """ORP's calibrated `innovation` arm (the E2 / E3 identity configuration: v3's binned seed deficits)."""
    ang = {"alphaBeta": ANGLE_NAMES, "alpha": ("alpha",)}[cfg["useAngles"]]
    return inn(cfg, name=kw.pop("name", "ORP-cal"), ang_err="payload", inner_material=False, seed_cov="calibrated",
               use=use_set(("rank", "update"), ang, ("barrel",)), truth_modules=True, **kw)


# ---------------------------------------------------------------- inputs ----
def branches(path):
    keys = OP._keys(str(path))
    b = list(E.branches(path))
    b += SPN.cluster_branches(keys, list(EXTRA_CL_COLS) + ["hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta"],
                              disc_scope="all", cap_angles=False)
    b += SPN.branch_names(keys, "L1TTrack", ["tpIdx"])
    b += SPN.branch_names(keys, SPN.TP_TABLE, list(TP_COLS))
    b += [c for c in STUB_COLS if c in keys]
    return list(dict.fromkeys(b))


STUB_COLS = ("L1TTrackStub_trackIdx", "L1TTrackStub_otStubIdx", "L1TOTStub_isPS", "L1TOTStub_isBarrel")


def stub_features(A, ntrk):
    """Per flattened OT track: the number of PS and 2S stubs and of barrel and disc stubs (the smooth seed's keys)."""
    nt = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    off = np.concatenate([[0], np.cumsum(nt)[:-1]])
    ti, si = A["L1TTrackStub_trackIdx"], A["L1TTrackStub_otStubIdx"]
    gi = ak.to_numpy(ak.flatten(ti + ak.Array(off))).astype(np.int64)
    ps = ak.to_numpy(ak.flatten(A["L1TOTStub_isPS"][si])).astype(np.float64)
    br = ak.to_numpy(ak.flatten(A["L1TOTStub_isBarrel"][si])).astype(np.float64)
    out = {"n_ps": np.bincount(gi, ps, ntrk), "n_2s": np.bincount(gi, 1.0 - ps, ntrk),
           "n_barrel_stubs": np.bincount(gi, br, ntrk), "n_disc_stubs": np.bincount(gi, 1.0 - br, ntrk)}
    return {k: v.astype(np.int16) for k, v in out.items()}


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
    if all(c in A.fields for c in STUB_COLS):
        ctx.update(stub_features(A, ctx["ntrk"]))
    return ctx


class Tools:
    """What the arms read besides the event: the producer configuration, the refit calibration (seed,
    material, R tables), the angle table and the cut file (cuts.Cuts)."""

    def __init__(self, cfg, calibration=CALIBRATION, angle_table=AI.TABLE, cuts=None, disc=None, seed_model=None):
        self.cfg = cfg
        self.cal = ORP.load_calibration(calibration) if calibration else None
        self.angles = AI.AngleImport(angle_table) if angle_table else None
        self.cuts = cuts
        self.disc = disc                 # C1b's DiscTools (the disc_model arms)
        self._seed_model = seed_model    # seed_cov="smooth": seedmodel.SeedModel (default: its model file, loaded once)
        self._folded = None

    @property
    def seed_model(self):
        if self._seed_model is None:
            import seedmodel as SMOD
            self._seed_model = SMOD.SeedModel()
        return self._seed_model

    def folded(self):
        if self._folded is None:
            self._folded = AI.folded(self.angles)
        return self._folded


# ---------------------------------------------------------------- state ----
REC_KEYS = {"mi": -1, "ci": -1}


class DiscTools:
    """C1b: what the disc_model arms read for disc-going tracks: the disc angle rows and the material planes (with
    the disc material scale). The position errors are v3's tables (c1b/PLAN.md §8 A13)."""

    def __init__(self, table=None, material=None, disc_scale=1.0):
        import disc_angles as DA
        import disc_material as DM
        self.imp = DA.DiscImport(table or DA.DISC_TABLE)
        import json
        with open(material or DM.TABLE) as fh:
            self.material = json.load(fh)
        self.planes = self.material["planes"]
        self.disc_scale = disc_scale


BARREL_ON_DISC_GOING = 100         # C1b: a barrel visit of a disc-going track is binned as surface + 100


def cut_surface(arm, st, v, surface):
    """The cut bins' surface code (c1b/PLAN.md rev 4 §1): for the disc_model arms, a barrel visit of a
    disc-going track has its own bins (surface + 100); everything else keeps the surface."""
    surface = np.asarray(surface)
    if not arm.disc_model:
        return surface
    return np.where(st.dg[v] & (surface <= 4), surface + BARREL_ON_DISC_GOING, surface)


def disc_going(path):
    """C1b's population: the arm's own path (the producer's drPath: a disc within its radial extent +- 1 cm,
    before the scenario mask) contains a disc."""
    cols = [j for j, S in enumerate(path["surfaces"]) if SPN.is_disc_surface(S)]
    return path["in_path"][:, cols].any(1) if cols else np.zeros(len(path["in_path"]), bool)


class State:
    """Virtual tracks: o[v] the original track; per v the helix state, the visit bookkeeping and the
    per-step record (module, cluster, accepted, chi2 per scalar, the gate statistic, window size).
    dg: disc-going (C1b's disc_model arms; False otherwise)."""

    def __init__(self, a, C, o, seq, dg=None):
        n, ns = len(o), seq.shape[1]
        self.o, self.a, self.C, self.seq = o, a, C, seq
        self.dg = np.zeros(n, bool) if dg is None else np.asarray(dg, bool)
        self.nUpd = np.zeros(n, np.int64)
        self.prevR = np.full(n, -1.0)
        self.prevNT = np.zeros(n, np.int64)
        self.Lv = np.full(n, 5, np.int64)                  # calibrated material: last visited (5 = the OT)
        self.branch = np.zeros(n, np.int8)                 # K: 0 best, 1 runner-up, 2 skip
        self.pend = np.zeros(n, bool)                      # tree2: a first-layer choice awaiting its lookahead
        self.s1 = np.zeros(n)                              # tree2: its first-layer chi2 (0 for the skip)
        self.P1 = np.zeros(n)                              # tree2: the window's skip penalty
        self.j1 = np.zeros(n, np.int64)                    # tree2: the window's first visit step
        self.rec = {"mi": np.full((n, ns), -1, np.int64), "ci": np.full((n, ns), -1, np.int64),
                    "acc": np.zeros((n, ns), bool), "visited": np.zeros((n, ns), bool),
                    "chi2": np.zeros((n, ns, 4)), "gstat": np.full((n, ns), np.nan),
                    "ncand": np.zeros((n, ns), np.int16), "trunc": np.zeros((n, ns), bool),
                    "gated": np.zeros((n, ns), bool), "surface": np.full((n, ns), -1, np.int64),
                    "tgt_only": np.zeros((n, ns), bool),       # the accepted cluster's module came from truth alone
                    "kchoice": np.full((n, ns), -1, np.int8)}      # tree2: the committed choice per visit step

    ARRAYS = ("o", "a", "C", "seq", "nUpd", "prevR", "prevNT", "Lv", "branch", "dg", "pend", "s1", "P1", "j1")

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
    floor, plus the calibrated arms' correction: v3's binned deficit or the smooth model (A16)."""
    a = ctx["a"][o].copy()
    C, _ = E.d0_floor(ctx["C"][o], cfg)
    if arm.seed_cov == "calibrated":
        kpt = ORP.pt_constant(ctx["T"])
        C = ORP.seed_cov(tools.cal, C, a[:, 2], ctx["n_disk_stubs"][o], ctx["T"]["nStubs"][o],
                         kpt / np.maximum(np.abs(a[:, 0]), 1e-12))
    elif arm.seed_cov == "smooth":
        C = C + tools.seed_model.dC(ctx, a, o)
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
    if arm.disc_model:
        _disc_measurements(ctx, ci, surface, cot_a_pred, cot_b_pred, arm, tools, pub, meas, sig)
    if arm.angle_term == "t9":
        for i, pred in ((2, cot_a_pred), (3, cot_b_pred)):
            sig[:, i] = sig[:, i] * tail_robust_scale((meas[:, i] - pred) / sig[:, i])
    return meas, sig, has


T9_NU = 9.0


def tail_robust_scale(z, nu=T9_NU):
    """The "t9" angle term (scan plan B1): the factor on an angle's sigma, sqrt(1 + z^2 / nu), so that its chi2
    (z^2 / (1 + z^2 / nu)) stays z^2 in the core and saturates at nu. The prediction's angle variance is left out of
    z: it is ~1e-6 of the cluster's (C1b's decomposition). Non-finite z: unchanged."""
    z = np.asarray(z, np.float64)
    return np.where(np.isfinite(z), np.sqrt(1.0 + z * z / nu), 1.0)


def _disc_measurements(ctx, ci, surface, cot_a_pred, cot_b_pred, arm, tools, pub, meas, sig):
    """C1b (c1b/PLAN.md rev 4 §1 item 1), in place on the disc clusters: the cot-linear knob on the local columns and
    the disc rows (the barrel import refuses discs)."""
    import disc_angles as DA
    d = SPN.is_disc_surface(surface)
    if not d.any():
        return
    K, dk = ctx["K"], tools.disc
    cd = ci[d]
    T = SA.class_T(K["tpEmuClass"][cd])
    rows = dk.imp.rows(surface[d], cot_a_pred[d], cot_b_pred[d]) if arm.ang_err == "table" else None
    for i, ang, k in ((2, "alpha", arm.k_alpha), (3, "beta", arm.k_beta)):
        raw, tp = SA.LOCAL_COLS[ang][0], SA.LOCAL_COLS[ang][1]
        m = DA.knob(K[raw][cd], K[tp][cd], k, T)
        if rows is not None:
            mm, ss = dk.imp.calibrate(ang, m, pub[d, i], rows, k=k, mode=arm.k_mode)
            ok = rows >= 0
            meas[d, i] = np.where(ok, mm, m)
            sig[d, i] = np.where(ok, ss, SA.sigma_k(pub[d, i], k, arm.k_mode))
        else:
            meas[d, i] = m
            sig[d, i] = SA.sigma_k(pub[d, i], k, arm.k_mode)


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
        dg = (st.dg[t] | (arm.disc_material == "all_tracks")) if arm.disc_model else np.zeros(len(t), bool)
        dgt = t[dg]
        ORP.add_material(st.a, st.C, t[~dg], S, st.Lv, geo, cal, kpt)
        if len(dgt):                                   # C1b: path-ordered, extent-aware, each plane's path factor
            import disc_material as DM
            s_cur = OP.surface_s(st.a[dgt], geo, S)
            s_prev = np.full(len(dgt), np.inf)
            for L in np.unique(st.Lv[dgt]):
                if L != 5:
                    k = st.Lv[dgt] == L
                    s_prev[k] = OP.surface_s(st.a[dgt[k]], geo, int(L))
            DM.charge(st.a, st.C, dgt, s_cur, s_prev, tools.disc.planes, tools.disc.disc_scale, cal["material_scale"], kpt,
                      disc_factor="barrel" if arm.disc_material == "barrel_factor" else "disc",
                      cyl_extent=arm.disc_material != "radius")
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
        tgo = np.zeros(len(ci), bool)                              # the producer's box: geometry only
    else:
        on = np.zeros(len(st.o), bool)
        on[t] = True
        nsig = {"nsig": arm.nsig, "gate99": WINDOW_SEARCH_NSIG}[arm.window_cut]
        ev = OP.evaluate_surface(_vctx(ctx, st.o), st.a, st.C, S, geo, nsig, on, truth_modules=arm.truth_modules)
        if ev is None:
            return None
        c = ev["cand"]
        keep = c["inside"]
        pj = ev["pj"]
        p = c["pair"][keep]
        tt = c["t"][keep]
        ci = c["ci"][keep]
        tgo = c["tgt_only"][keep]
        cmi = ev["pm"][p]
        ch0 = np.stack([pj["u"][p], pj["v"][p], pj["cotA"][p], pj["cotB"][p]], 1)
        cJ = pj["J"][p]
        cSz = np.einsum("nij,njk,nlk->nil", E.guard_zeroed(cJ), st.C[tt], E.guard_zeroed(cJ))
        if arm.window_cut == "gate99":                            # PD18: the window is the 99 % hit cut
            g = gate_stat(arm, "gate", ctx, geo, ci, cmi, ch0, cSz, tools)
            w = g <= tools.cuts.chi2("gate", arm.dims_key("gate"), arm.cov_model,
                                     cut_surface(arm, st, tt, geo["surface"][cmi]), np.full(len(ci), j == 0))
            if not w.any():
                return None
            p, tt, ci, cmi, ch0, cJ, cSz, tgo = p[w], tt[w], ci[w], cmi[w], ch0[w], cJ[w], cSz[w], tgo[w]
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
            "gr": (cgr[keep] if cgr is not None else None), "val": val[keep], "order": order[keep],
            "tgt_only": tgo[keep]}


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
        cut = tools.cuts.chi2("gate", arm.dims_key("gate"), arm.cov_model, cut_surface(arm, st, t, geo["surface"][mi]),
                              np.full(len(t), j == 0))
        ok = ~(g > cut)
        st.rec["gated"][t[~ok], j] = True
        st.rec["mi"][t[~ok], j], st.rec["ci"][t[~ok], j] = mi[~ok], ci[~ok]
    gr = cands["gr"][k] if cands["gr"] is not None else None
    if arm.window == "ellipse":                                  # ORP.update_on's predicted-angle guard
        ok &= (np.abs(h0[:, 2]) <= ORP.PRED_ANGLE_MAX_ABS) & (np.abs(h0[:, 3]) <= ORP.PRED_ANGLE_MAX_ABS)
    _apply(arm, st, j, t[ok], mi[ok], ci[ok], h0[ok], J[ok], (gr[ok] if gr is not None else None),
           ctx, geo, tools, path, S)
    st.rec["tgt_only"][t[ok], j] = cands["tgt_only"][k][ok]


def arm_at_step(arm, j):
    """The arm as it acts at visit step j: angle_steps "first" drops every angle use after the first step, "later" at
    the first step."""
    if (arm.angle_steps == "first" and j > 0) or (arm.angle_steps == "later" and j == 0):
        return replace(arm, use=frozenset())
    return arm


def step(arm, st, j, t, S, ctx, geo, tools, kpt, cfg, path, own_rows=None):
    """One (visit step, surface): Q, crossings, candidates, (K), gate, update; with own_rows (a list),
    the own-cluster diagnostics are taken after Q and before the update. Returns the State (K adds
    virtual tracks)."""
    arm = arm_at_step(arm, j)
    _q_step(arm, st, t, S, path, geo, tools, kpt, cfg)
    if own_rows is not None:
        import diagnostics as DG
        own_rows.append(DG.own_rows(arm, st, j, t, S, ctx, geo, tools, kpt))
    st.rec["visited"][t, j] = True
    st.rec["surface"][t, j] = S
    if arm.K == "tree2":
        return _tree2_step(arm, st, j, t, S, ctx, geo, tools, kpt, cfg, path)
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


# ---------------------------------------------------------------- tree2 (PD14) ----
def tree2_scores(x1, x2, P, k_score="penalty", p_first=None):
    """Scan plan A1: the score of each first-layer choice (best, runner-up, skip) of n windows, the lower the better.
    x1 (n, 2): the first layer's chi2 of best and runner-up (inf: none); x2 (n, 3, 2): the second layer's chi2 of best
    and runner-up from each first choice's own prediction (inf: none), or None for a one-layer window (no next visit);
    P (n,): the window's skip penalty, the same at both layers. "penalty": chi2 sum plus P per skip, at most one
    chosen skip; an empty second layer costs P but is not a chosen skip. "tailprob" (the PD2 known-bad): -log of the
    chi2 tail probability of the pair's sum for 2 dof per hit. p_first: a different first-layer skip penalty (PD2's
    asymmetric known-bad)."""
    P = np.asarray(P, np.float64)
    zero = np.zeros(len(P))
    chi1 = np.stack([x1[:, 0], x1[:, 1], zero], 1)
    h1 = np.array([1, 1, 0])
    if k_score == "tailprob":
        from scipy import stats

        def tp(chi, hits):
            with np.errstate(invalid="ignore"):
                return np.where(np.isfinite(chi), -stats.chi2.logsf(chi, np.maximum(2 * hits, 1)), np.inf)
        if x2 is None:
            return np.stack([tp(chi1[:, c], h1[c]) for c in range(3)], 1)
        out = np.full((len(P), 3), np.inf)
        for c in range(3):
            empty = ~np.isfinite(x2[:, c]).any(1)
            opts = [tp(chi1[:, c] + x2[:, c, 0], h1[c] + 1), tp(chi1[:, c] + x2[:, c, 1], h1[c] + 1)]
            if c < 2:
                opts.append(tp(chi1[:, c], h1[c]))
            else:
                opts.append(np.where(empty, tp(zero, 0), np.inf))
            out[:, c] = np.min(opts, 0)
        return out
    s1 = np.stack([x1[:, 0], x1[:, 1], P if p_first is None else np.asarray(p_first, np.float64)], 1)
    if x2 is None:
        return s1
    cand = np.minimum(x2[:, :, 0], x2[:, :, 1])
    empty = ~np.isfinite(cand)
    L2 = np.where(empty, P[:, None], cand)
    L2[:, :2] = np.where(empty[:, :2], P[:, None], np.minimum(cand[:, :2], P[:, None]))
    return s1 + L2


def _tree2_commit(arm, st, idx, x2):
    """Commit the open windows of the virtual tracks idx (every copy of their OT tracks): per OT track the copy of the
    lowest-scoring first choice (ties: best, runner-up, skip). x2 (len(idx), 2): each copy's second-layer chi2 of best
    and runner-up (inf: none), or None (one layer). Records the choice at the window's first step; returns the kept
    mask over idx."""
    ug, gi = np.unique(st.o[idx], return_inverse=True)
    c = st.branch[idx].astype(np.int64)
    X1 = np.full((len(ug), 2), np.inf)
    hit = c < 2
    X1[gi[hit], c[hit]] = st.s1[idx[hit]]
    P = np.zeros(len(ug))
    P[gi] = st.P1[idx]
    X2 = None
    if x2 is not None:
        X2 = np.full((len(ug), 3, 2), np.inf)
        X2[gi, c] = x2
    win = np.argmin(tree2_scores(X1, X2, P, arm.k_score), 1)
    keep = c == win[gi]
    st.rec["kchoice"][idx[keep], st.j1[idx[keep]]] = c[keep]
    st.pend[idx] = False
    return keep


def _tree2_step(arm, st, j, t, S, ctx, geo, tools, kpt, cfg, path):
    """One (visit step, surface) of tree2: Q; the candidates and their chi2 (the rank consumer's dims); P is the
    calibrated model's 99 % gate cut in those dims for every arm, so an element's leave-one-out keeps the skip price; the windows
    opened at the tracks' previous visit are committed with this surface as their second layer; then a window opens
    here on every surviving track with a candidate: best (applied to the track), runner-up (applied to a copy) and
    skip (a copy), pending until the next visit."""
    _q_step(arm, st, t, S, path, geo, tools, kpt, cfg)
    st.rec["visited"][t, j] = True
    st.rec["surface"][t, j] = S
    got = _cands(arm, st, j, t, S, ctx, geo, tools, kpt, cfg)
    kb = np.full(len(t), -1, np.int64)
    kr = np.full(len(t), -1, np.int64)
    xb = np.full(len(t), np.inf)
    xr = np.full(len(t), np.inf)
    if got is not None:
        X, cands = got
        st.rec["ncand"][X["t"], j] = X["ncand"]
        st.rec["trunc"][X["t"], j] = X["trunc"]
        best, runner = _ranked(cands, len(X["t"]))
        pos = np.searchsorted(t, X["t"])
        kb[pos], kr[pos] = best, runner
        for k_, x_ in ((kb, xb), (kr, xr)):
            m = k_ >= 0
            if m.any():
                q = k_[m]
                x_[m] = gate_stat(arm, "rank", ctx, geo, cands["ci"][q], cands["mi"][q], cands["h0"][q], cands["Sz"][q],
                                  tools)
    drop = np.zeros(0, np.int64)
    if st.pend[t].any():
        allp = np.flatnonzero(st.pend & np.isin(st.o, st.o[t[st.pend[t]]]))
        x2 = np.full((len(allp), 2), np.inf)                      # a copy not at this visit: an empty layer
        pos = np.minimum(np.searchsorted(t, allp), len(t) - 1)
        here = t[pos] == allp
        x2[here, 0], x2[here, 1] = xb[pos[here]], xr[pos[here]]
        drop = allp[~_tree2_commit(arm, st, allp, x2)]
    alive = ~np.isin(t, drop)
    h1, h2 = alive & (kb >= 0), alive & (kr >= 0)
    if h1.any():
        P = tools.cuts.chi2("gate", arm.dims_key("rank"), "calibrated",     # one skip price for every arm (A1)
                            cut_surface(arm, st, t[h1], np.full(h1.sum(), S, np.int64)), np.full(h1.sum(), j == 0))
        P2 = P[np.isin(t[h1], t[h2])]
        n0 = len(st.o)
        ru = st.take(t[h2])
        ru.branch[:] = 1
        sk = st.take(t[h1])
        sk.branch[:] = 2
        st.extend(ru)
        st.extend(sk)
        ir = np.arange(n0, n0 + h2.sum())
        isk = np.arange(n0 + h2.sum(), n0 + h2.sum() + h1.sum())
        st.branch[t[h1]] = 0
        for idx, s1, p in ((t[h1], xb[h1], P), (ir, xr[h2], P2), (isk, np.zeros(h1.sum()), P)):
            st.pend[idx], st.s1[idx], st.P1[idx], st.j1[idx] = True, s1, p, j
        _choose_apply(arm, st, j, {"t": ir}, cands, kr[h2], ctx, geo, tools, path, S, kpt)
        _choose_apply(arm, st, j, {"t": t[h1]}, cands, kb[h1], ctx, geo, tools, path, S, kpt)
    if len(drop):
        st = st.take(np.setdiff1d(np.arange(len(st.o)), drop))
    return st


# ---------------------------------------------------------------- the run ----
def run_arm(ctx, geo, cfg, mask, arm, tools, tracks=None, own=False, loo=False):
    """One arm on one chunk (free selection). tracks: bool per track (default: a covariance, as the
    producer's otCovOK). own: also record the own-cluster diagnostics (diagnostics.own_rows; K = 1);
    loo: also the leave-one-out chi2 of every accepted hit for all angle-dims sets (the rr cut fit)."""
    if own and arm.K != 1:
        raise ValueError("the own-cluster diagnostics are recorded with K = 1")
    if arm.K == "tree2" and arm.gate:
        raise ValueError("tree2 prices a skip at the 99 % gate cut itself; the forward gate is off (scan plan A1)")
    if arm.angle_steps != "all" and (arm.rr or loo):
        raise ValueError("per-step angle use is defined for the forward pass only (no rr / leave-one-out)")
    if tracks is None:
        tracks = np.trace(ctx["C"], axis1=1, axis2=2) > 0
    o = np.flatnonzero(tracks)
    kpt = ORP.pt_constant(ctx["T"])
    path, seq = visit_plan(ctx, geo, cfg, mask, arm)
    a, C = seed(ctx, cfg, arm, tools, o)
    st = State(a, C, o.copy(), seq[o], dg=disc_going(path)[o] if arm.disc_model else None)
    own_rows = [] if own else None
    for j in range(seq.shape[1]):
        for S in np.unique(st.seq[:, j][st.seq[:, j] >= 0]):
            S = int(S)
            t = np.flatnonzero((st.seq[:, j] == S) & (st.nUpd < cfg["maxKFUpdates"]))
            if len(t):
                st = step(arm, st, j, t, S, ctx, geo, tools, kpt, cfg, path, own_rows)
    return finish(arm, st, ctx, geo, cfg, mask, tools, kpt, path, seq, own_rows, loo)


def refit_forced(arm, ctx, geo, cfg, tools, o, seq, sel_mi, sel_ci, path, kpt, history=False):
    """Refit tracks o (may repeat) from their seeds on the given (module, cluster) per visit step
    (-1: no hit), with the arm's Q, projection and update; no gate. Returns the State; with history,
    State.hist holds per step the predicted (after Q, before the update) and the filtered state
    ("pa", "pC", "fa", "fC"), for smooth_back. A step a track does not visit keeps its previous state."""
    a, C = seed(ctx, cfg, arm, tools, o)
    st = State(a, C, np.asarray(o).copy(), seq, dg=disc_going(path)[np.asarray(o)] if arm.disc_model else None)
    if history:
        n, ns = len(st.o), seq.shape[1]
        st.hist = {k: np.zeros((n, ns) + sh) for k, sh in (("pa", (5,)), ("pC", (5, 5)), ("fa", (5,)), ("fC", (5, 5)))}
    for j in range(seq.shape[1]):
        if history:
            st.hist["pa"][:, j], st.hist["pC"][:, j] = st.a, st.C
        for S in np.unique(seq[:, j][seq[:, j] >= 0]):
            S = int(S)
            t = np.flatnonzero((seq[:, j] == S) & (st.nUpd < cfg["maxKFUpdates"]))
            if not len(t):
                continue
            _q_step(arm, st, t, S, path, geo, tools, kpt, cfg)
            if history:
                st.hist["pC"][t, j] = st.C[t]
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
        if history:
            st.hist["fa"][:, j], st.hist["fC"][:, j] = st.a, st.C
    return st


def smooth_back(hist, steps):
    """Rauch-Tung-Striebel smoothing of a refit (refit_forced with history) back to the visit step
    steps[i] of each row: the two-sided state there, combining the steps before it (the forward
    prediction) with the steps after it (the inner hits), the process noise between them included
    (PD17). The helix parameters carry over between steps unchanged (F = I; the material adds to C)."""
    ns = hist["fa"].shape[1]
    aS, CS = hist["fa"][:, ns - 1].copy(), hist["fC"][:, ns - 1].copy()
    for k in range(ns - 2, -1, -1):
        m = np.flatnonzero(steps <= k)
        if not len(m):
            continue
        Cf, af, Cp, ap = hist["fC"][m, k], hist["fa"][m, k], hist["pC"][m, k + 1], hist["pa"][m, k + 1]
        A = np.transpose(np.linalg.solve(Cp, Cf), (0, 2, 1))          # Cf Cp^-1 (both symmetric)
        aS[m] = af + np.einsum("nij,nj->ni", A, aS[m] - ap)
        CS[m] = Cf + np.einsum("nij,njk,nlk->nil", A, CS[m] - Cp, A)
    return aS, CS


DIMS = {"xy": (False, False), "xya": (True, False), "xyb": (False, True), "xyab": (True, True)}


def loo_stats(arm, ctx, geo, cfg, tools, st, v, path, kpt, all_dims=False):
    """Leave-one-out chi2 of every accepted hit of virtual tracks v (PD4 (c), PD17): the full refit from
    the OT seed without the hit; with arm.loo_pred "smoothed", its state smoothed back to the hit's visit
    step (two-sided: the outer hits forward, the inner hits backward), with "final" (the known-bad) its
    final state; projected onto the hit's module; the innovation chi2 over positions and the rr angle
    dims (all_dims: a dict over DIMS, for the cut fit). Returns (owner virtual track, step, stat)."""
    vv, jj = np.nonzero(st.rec["acc"][v])
    if not len(vv):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), ({k: np.zeros(0) for k in DIMS} if all_dims else np.zeros(0))
    owner = v[vv]
    sel_mi = st.rec["mi"][owner].copy()
    sel_ci = np.where(st.rec["acc"][owner], st.rec["ci"][owner], -1)
    sel_ci[np.arange(len(vv)), jj] = -1
    smoothed = {"smoothed": True, "final": False}[arm.loo_pred]
    R = refit_forced(arm, ctx, geo, cfg, tools, st.o[owner], st.seq[owner], sel_mi, sel_ci, path, kpt, history=smoothed)
    aL, CL = smooth_back(R.hist, jj) if smoothed else (R.a, R.C)
    mi = st.rec["mi"][owner, jj]
    ci = st.rec["ci"][owner, jj]
    keys = list(DIMS) if all_dims else [arm.dims_key("rr")]
    stat = {k: np.full(len(vv), np.nan) for k in keys}
    for S in np.unique(geo["surface"][mi]):
        m = np.flatnonzero(geo["surface"][mi] == S)
        pj = project(arm, aL[m], CL[m], int(S), geo, mi[m], kpt)
        h0 = np.stack([pj["u"], pj["v"], pj["cotA"], pj["cotB"]], 1)
        Jz = E.guard_zeroed(pj["J"])
        Sz = np.einsum("nij,njk,nlk->nil", Jz, CL[m], Jz)
        meas, sig, has = measurements(ctx, geo, ci[m], mi[m], h0[:, 2], h0[:, 3], arm, tools)
        for k in keys:
            if all_dims:
                dm = np.concatenate([np.ones((len(m), 2), bool), has & np.array(DIMS[k])[None, :]], 1)
            else:
                dm = np.concatenate([np.ones((len(m), 2), bool), angle_dims(arm, "rr", geo["surface"][mi[m]], has)], 1)
            stat[k][m] = innovation_chi2(meas - h0, Sz, sig, dm)
    return owner, jj, (stat if all_dims else stat[keys[0]])


def finish(arm, st, ctx, geo, cfg, mask, tools, kpt, path, seq, own_rows, loo=False):
    """K resolution, reject-and-refit (with the scenario mask's rr cuts, PD15; never removing a track's
    only remaining SmartPixels cluster, PD19), inner material, passthrough; the per-track output."""
    ntrk = ctx["ntrk"]
    if arm.K == "tree2":                                         # windows still open: one layer
        p = np.flatnonzero(st.pend)
        if len(p):
            keep = _tree2_commit(arm, st, p, None)
            st = st.take(np.setdiff1d(np.arange(len(st.o)), p[~keep]))
    if arm.K == 2:
        st = st.take(resolve_k(arm, st, geo, tools))
    loo_out = None
    if loo:
        owner, jj, stat = loo_stats(arm, ctx, geo, cfg, tools, st, np.arange(len(st.o)), path, kpt, all_dims=True)
        inner = st.rec["acc"][owner] & (np.arange(seq.shape[1])[None, :] > jj[:, None])
        loo_out = {"o": st.o[owner], "step": jj, "ci": st.rec["ci"][owner, jj], "mi": st.rec["mi"][owner, jj],
                   "surface": st.rec["surface"][owner, jj], "first": jj == 0, "n_inner": inner.sum(1),
                   "csurface": cut_surface(arm, st, owner, st.rec["surface"][owner, jj]),
                   "n_acc": st.rec["acc"][owner].sum(1), "dims": stat}
    removed = np.zeros((len(st.o), seq.shape[1]), bool)
    protected = np.zeros(len(st.o), bool)
    if arm.rr and tools.cuts.rr_statistic != arm.loo_pred:
        raise ValueError(f"the cut file's rr statistic is {tools.cuts.rr_statistic!r}, the arm's {arm.loo_pred!r} (PD17)")
    if arm.rr:
        for _ in range(arm.rr_max_iter):
            owner, jj, stat = loo_stats(arm, ctx, geo, cfg, tools, st, np.arange(len(st.o)), path, kpt)
            if not len(owner):
                break
            mi = st.rec["mi"][owner, jj]
            cut = tools.cuts.chi2("rr", arm.dims_key("rr"), arm.cov_model, cut_surface(arm, st, owner, geo["surface"][mi]),
                                  jj == 0, scen=mask)
            ratio = stat / cut
            # PD19: never remove a track's only remaining SmartPixels cluster
            last = st.rec["acc"][owner].sum(1) <= 1
            protected[owner[last & (ratio > 1.0)]] = True
            ratio = np.where(last, 0.0, ratio)
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
            # the refit replaces the state and the accepted hits only; prevR, prevNT, Lv and gstat keep the
            # forward pass's values (nothing after this point reads them)
            for k in ("a", "C", "nUpd"):
                getattr(st, k)[hit] = getattr(R, k)
            for k in ("acc", "chi2"):
                st.rec[k][hit] = R.rec[k]
    if arm.inner_material:
        inner_material(arm, st, geo, tools, kpt)
    nacc = st.rec["acc"].sum(1)
    fill = {"mi": -1, "ci": -1, "surface": -1, "gstat": np.nan, "kchoice": -1}
    out = {"a": ctx["a"].copy(), "C": ctx["C"].copy(), "nacc": np.zeros(ntrk, np.int64),
           "refit": np.zeros(ntrk, bool), "branch": np.full(ntrk, -1, np.int8),
           "rec": {k: np.full((ntrk,) + v.shape[1:], fill.get(k, 0), v.dtype) for k, v in st.rec.items()},
           "removed": np.zeros((ntrk, seq.shape[1]), bool), "rr_protected": np.zeros(ntrk, bool), "seq": seq}
    ok = nacc >= cfg.get("minHits", 1)
    o = st.o
    out["a"][o[ok]], out["C"][o[ok]] = st.a[ok], st.C[ok]
    out["nacc"][o] = nacc
    out["refit"][o[ok]] = True
    out["branch"][o] = st.rec["kchoice"][:, 0] if arm.K == "tree2" else st.branch
    for k, v in st.rec.items():
        out["rec"][k][o] = v
    out["removed"][o] = removed
    out["rr_protected"][o] = protected
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
        cs = np.where(st.dg[:, None] & (S <= 4), S + BARREL_ON_DISC_GOING, S) if arm.disc_model else S
        cut[m] = tools.cuts.chi2("gate", arm.dims_key("gate"), arm.cov_model, cs[m], first[m])
        score = g + cut.sum(1)
    o = np.lexsort((st.branch, score, st.o))
    first = np.r_[True, st.o[o][1:] != st.o[o][:-1]]
    return o[first]


def inner_material(arm, st, geo, tools, kpt):
    """The material between the innermost update and the beam line, charged to C only (PLAN §1 item 4):
    every calibration plane at r <= R of the innermost updated layer (its own module included: the
    kink there separates the inner segment from the outer measurements, as the forward pass charges a
    layer's module when predicting the next one), the IL1 module (absent from the calibration's list)
    and the beam pipe. A helix state is unchanged by material. Disc-going tracks of the disc_model arms (scan
    plan B-pre): the path-ordered, extent-aware planes of disc_material (C1b's, at the disc scale) crossed between
    the POCA and the innermost update (its own plane included), plus the beam pipe; never the barrel planes by
    radius."""
    acc = st.rec["acc"]
    t = np.flatnonzero(acc.any(1))
    if not len(t):
        return
    last_j = np.array([np.flatnonzero(acc[x])[-1] for x in t])
    Lv = np.full(len(st.o), 5, np.int64)
    Lv[t] = st.rec["surface"][t, last_j]
    dg = st.dg[t] if (arm.disc_model and arm.inner_material_rule == "path") else np.zeros(len(t), bool)
    tb, td = t[~dg], t[dg]
    geo0 = dict(geo)
    geo0["R"] = dict(geo["R"])
    geo0["R"][0] = 0.0
    cal = dict(tools.cal)
    cal["planes"] = list(tools.cal["planes"]) + [IL1_MODULE, BEAM_PIPE]
    ORP.add_material(st.a, st.C, tb, 0, Lv, geo0, cal, kpt)
    if len(td):
        import disc_material as DM
        s_in = np.empty(len(td))
        for L in np.unique(Lv[td]):
            k = Lv[td] == L
            s_in[k] = OP.surface_s(st.a[td[k]], geo, int(L))
        planes = list(tools.disc.planes) + [BEAM_PIPE_PLANE]
        DM.charge(st.a, st.C, td, np.zeros(len(td)), s_in, planes, tools.disc.disc_scale, tools.cal["material_scale"],
                  kpt)
