"""THE CENSUS ANGLE MODEL (WP-D0b; census format 33): the SmartPixels angles as every census stage
sees them, from the shared angle-definition module (ngtagger.spdefs) and the locked P0.3 table, the
per-consumer cut levels (decision 2, sheet angle.cut_level) and the records they are calibrated on.

THE MODEL, per barrel cluster and per angle a in {alpha, beta} (spdefs representation: alpha = the
direction's azimuth relative to the radial, beta = the dip angle lambda = atan(cot theta)):
  1. the knob (sheet angle.k_knob, decision 10): a_k = truth + k_a (a - truth) on class T
     (tpEmuClass == 1), class C as emitted (spdefs.improve); sigma_k = sigma_pub in `fixed` mode,
     max(k_a, 0.05) sigma_pub in `scaled` mode (spdefs.sigma_k);
  2. the table (sheet angle.error_table, registry p03.table.v1a1e): the cluster's emitted-key row
     (surface, signed emitted cot theta); the corrected angle a' = a_k - k_a c_a sigma_pub (the centre
     scales with k in both modes, spdefs.centre_k) and its uncertainty sigma' = w_a sigma_k. Then
     (a' - prediction) / sigma' is spdefs.table_pull for any prediction;
  3. the census representation, unchanged in form: kappa_alpha = sin(alpha') / (c r) with
     sigma = sigma'_alpha |cos alpha'| / (c r); cot' = tan(beta') with sigma = sigma'_beta (1 + cot'^2);
     z0 = z - r cot' with sigma = r sigma_cot; then the optional bench quantisation (not re-validated).

CUTTING CONSUMERS (S1, S2a, S2z / S2j, S3, S4w, S6) cut at the per-bin 99 % quantile of their own
statistic, fitted on correct class-T candidates of TPs born inside 2.5 cm, each TP weighted once
(fit_cuts). WEIGHTING CONSUMERS (S4a, S5) use the table width and add their prediction error once,
with the key a switch (emitted: the cluster's prepared value; prediction: the true-key row at the
predicted cot theta).

Every function here that turns components into a statistic is used BOTH by the census (in-run) and
by the offline cut fitting and gate (records), so the two cannot drift.

THE DISCS (census format 36, wpdisc/PLAN.md sec. 1 B): the SmartPixels disc clusters (ID1-ID4) carry no barrel
quantity (prepare: a wildcard in every barrel field; on a disc one cluster's beta gives z0 to ~54 cm and its alpha
curvature to ~9 /GeV, so neither is routed there). Their stages are LOCAL COMPATIBILITY TESTS: each predicts the
cluster's module-frame (cotAlpha, cotBeta) on its own module (the D121 frames, spdefs.disc_frames) from the
hypothesis' helix, reads the disc table (registry p03.table.v1a1e-disclocal, its true_local rows, facings pooled) at
the PREDICTED key, and takes the pulls in cot units: m' = m_k - k c sigma_pub, sigma' = w sigma_k, with the knob on
the module-frame cots of class T. S1 (the cluster's cotAlpha against the pT band C r kappa_floor / |cot| about the
straight track's), S2a / S2z (each disc cluster's cotAlpha / cotBeta against the pair's prediction, the larger of
the pair's), S4w (a follow candidate's pulls against the seed's), S4a (those pulls, the prediction error added)
and S6 (the final state's pulls) act on them; S3 has no disc target in the menu and S5 does not feed disc angles
to the fit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np

from ngtagger.spdefs import angles as SD
import tracklet_topology_cost as M
import kf_emulation as KF

TABLE_DEFAULT = "/Volumes/WDMac/smartpixels_scratch/p03_angles/angle_table_v1a1e_canon_v3.json"
TABLE_REGISTRY = "p03.table.v1a1e"
CUT_SCHEMA = "census-angle-cuts/1"
LEVEL = SD.CUT_LEVEL                     # 0.99 (decision 2)
LEVELS = SD.LEVELS
KEYS = ("emitted", "prediction")
ANGLES = ("alpha", "beta")
# the angle subsets a per-angle stage can use (AngleUse.angles)
SUBSETS = {"ab": ("alpha", "beta"), "a": ("alpha",), "b": ("beta",)}
# seed-measured pT bins of the S3 / S4w cut: [1, 2), [2, 5), [5, inf) GeV from the seed's |kappa|
# (a seed below 1 GeV, possible through the d0 slack, goes to the first bin)
PT_EDGES = (2.0, 5.0)
N_PTBIN = len(PT_EDGES) + 1
# genealogy (sheet truth.genealogy): the calibration population is codes 0-2, born inside 2.5 cm
GEN_PROMPT_HS, GEN_PROMPT_PU, GEN_HF, GEN_BEYOND, GEN_NONE = 0, 1, 2, 3, -1
GEN_NAMES = {0: "prompt hard scatter", 1: "prompt pile-up", 2: "b/c daughter",
             3: "born beyond IL1", -1: "no L1TTP row"}
BORN_IN_CM = 2.5
MIN_TP = 200                              # merge minimum, in distinct TPs (stats review M2)
# S5 states recorded for S6 (four angle sets x two keys; "off" has no key)
S5_STATES = ("ab", "a", "b", "off")
S5_REFITS = tuple((s, k) for s in ("ab", "a", "b") for k in KEYS) + (("off", "emitted"),)
DISC_NOTE = ("disc angles (census format 36): module-frame compatibility tests at the predicted key of the "
             "disc table p03.table.v1a1e-disclocal (facings pooled); S1, S2a, S2z, S4w cut, S4a and S6 weigh, S3 "
             "has no disc target, S5 feeds no disc angle; the emulated response is a placeholder, so findings "
             "about the angles' value are traced to it only by the plan's pre-declared test")
DISC_TABLE_DEFAULT = "/Volumes/WDMac/smartpixels_scratch/p03_angles/angle_table_v1a1e_disclocal_canon_v3.json"
DISC_TABLE_REGISTRY = "p03.table.v1a1e-disclocal"
DISC_KEY = "true_local"                  # the disc table at the predicted key (the true key stands in for it)
# THE DISC KNOWN-BADS of the import gate (WP-Disc plan sec. 4, D-angle), evaluated offline on the records
# (Model.with_disc_known_bad): the module facing flipped; the barrel's frame on a disc module; the barrel-form
# statistics (alpha through curvature for S2a, beta through z0 for S2z, from the global disc angles); the global
# disc alpha in place of the module-frame cotAlpha
DISC_KNOWN_BADS = ("facing_flipped", "barrel_frame", "barrel_form", "global_alpha")
# S6 WITH THE DISCS (census format 36): a track's S6 group is its build, with S6_DISC appended when the track
# carries a disc angle measurement. Pooled, the disc tracks (lighter-tailed module-frame pulls) set the small-n
# thresholds, and the barrel tracks there lose 2-3 % to a 99 % cut (CAL36). A disc track also carries up to twice
# the barrel's measurements (s6_n_max).
S6_DISC = "/disc"


def s6_group(build, disc):
    """Per track: its S6 group label (`build`, a string; `disc`, whether it carries a disc angle measurement)."""
    disc = np.asarray(disc, bool)
    return np.where(disc, build + S6_DISC, build)


def s6_n_max(n_max, group):
    """The largest n of an S6 group: the barrel's n_max, twice that with disc measurements."""
    return 2 * n_max if str(group).endswith(S6_DISC) else n_max
CLASS_C_NOTE = ("the table describes class T (tpEmuClass == 1); class-C clusters (CNN angles, "
                "~30 %) get the same row and are not calibrated; their efficiencies are reported, "
                "not gated; scaled-mode rejection is an upper bound")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ==========================================================================
# the model
# ==========================================================================
class Model:
    """The table, the knob (k per angle, sigma mode) and the weighting keys of one census run."""

    def __init__(self, table_path=TABLE_DEFAULT, k_alpha=1.0, k_beta=1.0, mode="fixed",
                 registry=TABLE_REGISTRY, disc_table_path=DISC_TABLE_DEFAULT):
        self.table_path = os.path.realpath(table_path)
        self.table = SD.AngleTable.load(self.table_path)
        self.table_sha = sha256(self.table_path)
        self.registry = registry
        self.disc_table_path = os.path.realpath(disc_table_path)
        self.disc_table = SD.AngleTable.load(self.disc_table_path)
        self.disc_table_sha = sha256(self.disc_table_path)
        self.disc_kb = None
        self.k = {"alpha": SD.check_k(k_alpha), "beta": SD.check_k(k_beta)}
        if mode not in SD.K_MODES:
            raise ValueError(f"k mode {mode!r} not in {SD.K_MODES}")
        self.mode = mode

    @classmethod
    def from_identity(cls, am):
        """The model a census manifest's angle_model names, refusing a table that changed since (either one)."""
        m = cls(am["table_path"], am["k_alpha"], am["k_beta"], am["mode"],
                disc_table_path=am.get("disc_table_path", DISC_TABLE_DEFAULT))
        if m.table_sha != am["table_sha256"]:
            raise SystemExit("the angle table changed since the census ran")
        if "disc_table_sha256" in am and m.disc_table_sha != am["disc_table_sha256"]:
            raise SystemExit("the disc angle table changed since the census ran")
        return m

    def identity(self):
        d = self.table.doc
        return {"table_path": self.table_path, "table_sha256": self.table_sha,
                "table_registry": self.registry, "schema": d.get("schema"),
                "definition_sheet": d.get("definition_sheet"), "era": d.get("era"),
                "fit_sample": d.get("fit_sample"), "k_alpha": self.k["alpha"],
                "k_beta": self.k["beta"], "mode": self.mode,
                "disc_table_path": self.disc_table_path, "disc_table_sha256": self.disc_table_sha,
                "disc_table_registry": DISC_TABLE_REGISTRY}

    def with_disc_known_bad(self, name):
        """This model with a disc known-bad (DISC_KNOWN_BADS) in every disc statistic: the import gate's offline
        known-bads."""
        if name not in DISC_KNOWN_BADS:
            raise ValueError(f"unknown disc known-bad {name!r}")
        m = copy.copy(self)
        m.disc_kb = name
        return m

    def _predict(self, Q, g, kap, cot, d0=None):
        """disc_predict, or its known-bad form: the facing flipped, or the barrel's frame (local normal along r,
        local x along phi: cotAlpha = tan(dir - phi), cotBeta = cot / cos(dir - phi))."""
        if self.disc_kb == "facing_flipped":
            return disc_predict({**Q, "lsx": -Q["lsx"]}, g, kap, cot, d0)
        if self.disc_kb == "barrel_frame":
            r = Q["lr"][g]
            s = M.C_BEND * np.asarray(kap, np.float64) * r - (0.0 if d0 is None else np.asarray(d0) / r)
            d = -np.arcsin(np.clip(s, -1.0, 1.0))
            return np.tan(d), np.asarray(cot, np.float64) / np.cos(d)
        return disc_predict(Q, g, kap, cot, d0)

    def at_k(self, k_alpha, k_beta, mode):
        """This model (its tables) at another knob and sigma mode: the offline recomputation of a disc
        statistic (angle_cuts), whose table centre scales with k at the stage itself."""
        m = copy.copy(self)
        m.k = {"alpha": SD.check_k(k_alpha), "beta": SD.check_k(k_beta)}
        m.mode = mode
        return m

    def calib_k(self):
        """(k_alpha, k_beta) at which a cutting consumer's quantile is measured: k in scaled mode,
        1 in fixed mode (spdefs.calibration_k)."""
        return tuple(SD.calibration_k(self.k[a], self.mode) for a in ANGLES)

    def rows(self, key, layer, cot):
        """Table row per barrel cluster (-1 where none)."""
        return self.table.rows_of(key, np.asarray(layer), np.asarray(cot, np.float64),
                                  np.zeros(len(layer), np.int8))

    def cw(self, key, angle, rows):
        """(centre, width) per row, with (0, 1) where there is no row (such a cluster has no usable
        direction: a wildcard everywhere)."""
        c = self.table.get(key, angle, "centre", rows)
        w = self.table.get(key, angle, "width", rows)
        bad = ~(np.isfinite(c) & np.isfinite(w))
        return np.where(bad, 0.0, c), np.where(bad, 1.0, w)

    # ---- the statistics, as the census stages call them (M, KF) -----------
    def s1(self, Q, idx, floor):
        return s1_stat(Q["half_c"][idx], np.maximum(Q["s_half_c"][idx], 1e-12), Q["r"][idx], floor)

    def s2a(self, kap, Q, gA, gB, useA, useB):
        return s2a_stat(kap, Q["kap_a"][gA], Q["s_kap"][gA], Q["kap_a"][gB], Q["s_kap"][gB],
                        useA, useB)

    def s2z(self, Q, gA, gB):
        return s2z_stat(Q["z0"][gA], Q["s_z0"][gA], Q["z0"][gB], Q["s_z0"][gB])

    @staticmethod
    def ptbin(kap):
        return ptbin(kap)

    def prediction_values(self, Q, g, cot_pred):
        """(kap_a, s_kap, cot, s_cot) of IT clusters g read in the PREDICTION key: the table's
        true-key row at the predicted cot theta (its centre and width), at the run's knob."""
        C = {k: Q[k][g] for k in COMP_KEYS}
        rows = self.rows("true", C["layer"], cot_pred)
        ca, wa = self.cw("true", "alpha", rows)
        cb, wb = self.cw("true", "beta", rows)
        h, sh, lam, sl = corrected_of(C, self.k["alpha"], self.k["beta"], self.mode,
                                      (ca, wa, cb, wb))
        kap, skap, cot, scot, _, _ = represent(C["r"], C["z"], h, sh, lam, sl)
        return kap, skap, cot, scot

    def arb_pulls(self, Q, g, kap_pred, cot_pred, skap_pred, sct_pred, key):
        """S4a's (alpha, beta) pulls of candidates g: the table key's values, the prediction
        error added once."""
        if key == "emitted":
            ka, sk, ct, sc = Q["kap_a"][g], Q["s_kap"][g], Q["cot_b"][g], Q["s_cot"][g]
        elif key == "prediction":
            ka, sk, ct, sc = self.prediction_values(Q, g, cot_pred)
        else:
            raise ValueError(f"table key {key!r}")
        return cand_pulls(ka, sk, ct, sc, kap_pred, cot_pred, skap_pred, sct_pred)

    def kf_measurements(self, Q, G, use, cot_pred, emitted):
        """S5's (KA, SKA, CT, SCT) per (track, layer) in the prediction key: each IT hit with a
        usable direction re-read in the true-key row at its track's predicted cot theta; elsewhere
        the emitted values."""
        KA, SKA, CT, SCT = (np.array(x, np.float64, copy=True) for x in emitted)
        t, j = np.nonzero(use)
        if len(t):
            ka, sk, ct, sc = self.prediction_values(Q, G[t, j], np.asarray(cot_pred)[t])
            KA[t, j], SKA[t, j], CT[t, j], SCT[t, j] = ka, sk, ct, sc
        return KA, SKA, CT, SCT

    # ---- the discs (module frame; census format 36) ----------------------
    def disc_rows(self, Q, g, pa, pb):
        """The disc table's row per disc cluster g at the predicted module-frame key of (pa, pb)."""
        key = SD.local_incidence_key(pa, pb)
        sf = np.asarray(Q["lcode"][g]) + M.DISC_CODE_OFFSET
        return self.disc_table.rows_of(DISC_KEY, sf, key, np.zeros(len(sf), np.int8))

    def disc_pulls(self, Q, g, pa, pb, rows=None, s_pa=0.0, s_pb=0.0):
        """(alpha pull, beta pull) in cot units of disc clusters g against predicted module-frame cots (pa, pb):
        the cluster's knobbed cots less the table's centre (k c sigma_pub) at the predicted key, over its width
        times sigma_k; s_pa / s_pb a prediction error added once (S4a)."""
        rows = self.disc_rows(Q, g, pa, pb) if rows is None else rows
        out = []
        for a, m, sp, pred in (("alpha", "lca", s_pa, pa), ("beta", "lcb", s_pb, pb)):
            c = self.disc_table.get(DISC_KEY, a, "centre", rows)
            w = self.disc_table.get(DISC_KEY, a, "width", rows)
            bad = ~(np.isfinite(c) & np.isfinite(w))
            c, w = np.where(bad, 0.0, c), np.where(bad, 1.0, w)
            ka = self.k[a]
            mk = Q[f"{m}_k"][g]
            if self.disc_kb == "global_alpha" and a == "alpha":   # the known-bad: the global disc alpha's tangent
                mk = np.tan(SD.wrap_2pi(Q["lphi"][g] - Q["lg_dir"][g]))
            m2 = mk - SD.centre_k(c * Q[f"{m}_pub"][g], ka)
            sk = w * Q[f"{m}_sk"][g]
            out.append((m2 - pred) / np.maximum(np.hypot(sk, sp), 1e-12))
        return out[0], out[1]

    def s1d(self, Q, g, floor):
        """S1 on disc clusters g: (statistic, the disc table row it is binned by). The straight prompt track
        through the cluster (kappa 0, z0 0) predicts (pa0, pb0); the statistic is |alpha pull| less the pT band
        C r kappa_floor / |cot| in the same units (negative inside the band)."""
        r, z = Q["lr"][g], Q["lz"][g]
        cot0 = z / np.maximum(r, 1e-6)
        pa0, pb0 = self._predict(Q, g, np.zeros(len(g)), cot0)
        rows = self.disc_rows(Q, g, pa0, pb0)
        pa, _ = self.disc_pulls(Q, g, pa0, pb0, rows)
        band = M.C_BEND * r / (np.asarray(floor, np.float64) * np.maximum(np.abs(cot0), 1e-6))
        c = self.disc_table.get(DISC_KEY, "alpha", "width", rows)
        w = np.where(np.isfinite(c), c, 1.0)
        return np.abs(pa) - band / np.maximum(w * Q["lca_sk"][g], 1e-12), rows

    def disc_window(self, Q, g, kap, cot, d0=None, skap=None, sct=None):
        """(alpha pull, beta pull) of disc candidates g against a helix's module-frame prediction (S4w; with
        skap / sct, the seed's kappa and cot errors propagated into the prediction and added once: S4a)."""
        pa, pb = self._predict(Q, g, kap, cot, d0)
        if skap is None:
            return self.disc_pulls(Q, g, pa, pb)
        r = Q["lr"][g]
        c = np.maximum(np.abs(cot), 1e-9)
        d = Q["lphi"][g] - np.arcsin(np.clip(M.C_BEND * np.asarray(kap) * r, -1.0, 1.0)) - Q["lphim"][g]
        dk = M.C_BEND * r * np.asarray(skap) / c          # d(cot alpha, cot beta) / d kappa, times sigma(kappa)
        s_pa = np.hypot(dk * np.abs(np.cos(d)), np.abs(pa) * sct / c)
        s_pb = np.hypot(dk * np.abs(np.sin(d)), np.abs(pb) * sct / c)
        return self.disc_pulls(Q, g, pa, pb, s_pa=s_pa, s_pb=s_pb)

    def s2d(self, Q, gA, gB, kap, cot, useA, useB):
        """S2a / S2z on a pair with disc clusters: each used disc cluster's alpha / beta pull against the pair's
        prediction, the larger of the two (0 where neither is used); and the inner cluster's disc table row."""
        n = len(gA)
        st_a, st_z = np.zeros(n), np.zeros(n)
        row_a = np.full(n, -1, np.int64)
        for g, use, first in ((gA, useA, True), (gB, useB, False)):
            i = np.flatnonzero(use)
            if not len(i):
                continue
            pa, pb = self._predict(Q, g[i], kap[i], cot[i])
            rows = self.disc_rows(Q, g[i], pa, pb)
            qa, qb = self.disc_pulls(Q, g[i], pa, pb, rows)
            st_a[i] = np.maximum(st_a[i], np.abs(qa))
            st_z[i] = np.maximum(st_z[i], np.abs(qb))
            if first:
                row_a[i] = rows
        if self.disc_kb == "barrel_form":
            st_a, st_z = _disc_barrel_form(Q, gA, gB, kap, useA, useB)
        return st_a, st_z, row_a


def corrected(angle, a_e, a_t, is_T, sig_pub, c, w, k, mode):
    """(a', sigma') of one angle: the knob on class T, then the table's centre and width."""
    a_k = SD.improve(angle, a_e, a_t, is_T, k)[0] if k < 1.0 else np.asarray(a_e, np.float64)
    s_k = SD.sigma_k(sig_pub, k, mode)
    a2 = SD.WRAP[angle](a_k - SD.centre_k(np.asarray(c) * sig_pub, k))
    return a2, np.asarray(w, np.float64) * s_k


def represent(r, z, half, s_half, lam, s_lam):
    """The census representation of corrected angles: (kap_a, s_kap, cot, s_cot, z0, s_z0)."""
    r = np.asarray(r, np.float64)
    kap_a = np.sin(half) / (M.C_BEND * r)
    s_kap = s_half * np.abs(np.cos(half)) / (M.C_BEND * r)
    cot = np.tan(lam)
    s_cot = s_lam * (1.0 + cot * cot)
    z0 = np.asarray(z, np.float64) - r * cot
    return kap_a, s_kap, cot, s_cot, z0, r * s_cot


def components(D, model):
    """Per IT barrel cluster: the inputs of every angle statistic (emitted and true angles,
    published sigmas, class T, emitted-key row, centre and width per angle)."""
    cot_e = np.asarray(D["globalClusterCotTheta"], np.float64)
    is_T = SD.class_T(D["tpEmuClass"]) & np.isfinite(np.asarray(D["tpGlobalClusterPhi"], np.float64)) \
        & np.isfinite(np.asarray(D["tpGlobalClusterCotTheta"], np.float64))
    C = {"r": np.asarray(D["globalR"], np.float64), "z": np.asarray(D["globalZ"], np.float64),
         "layer": np.asarray(D["layer"]).astype(np.int8), "cot_e": cot_e, "is_T": is_T,
         "half_e": SD.emitted("alpha", D), "lam_e": SD.emitted("beta", D),
         "half_t": np.where(is_T, SD.truth("alpha", D), np.nan),
         "lam_t": np.where(is_T, SD.truth("beta", D), np.nan),
         "sa_pub": SD.sigma_of("alpha", D), "sb_pub": SD.sigma_of("beta", D)}
    C["row"] = model.rows("emitted", C["layer"], cot_e)
    C["ca"], C["wa"] = model.cw("emitted", "alpha", C["row"])
    C["cb"], C["wb"] = model.cw("emitted", "beta", C["row"])
    return C


def corrected_of(C, k_alpha, k_beta, mode, key_cw=None):
    """(half', s_half', lam', s_lam') of component table C at (k_alpha, k_beta, mode); key_cw =
    (ca, wa, cb, wb) overrides the emitted-key centres and widths (the prediction key)."""
    ca, wa, cb, wb = key_cw if key_cw is not None else (C["ca"], C["wa"], C["cb"], C["wb"])
    h, sh = corrected("alpha", C["half_e"], C["half_t"], C["is_T"], C["sa_pub"], ca, wa, k_alpha, mode)
    lam, sl = corrected("beta", C["lam_e"], C["lam_t"], C["is_T"], C["sb_pub"], cb, wb, k_beta, mode)
    return h, sh, lam, sl


def prepare(D, model, bench=None):
    """The census's per-IT-cluster angle quantities (M.it_prepare's fields, from the model), plus the
    components the records and the prediction key need. The per-layer scales are not used.
    IT DISC CLUSTERS (codes 5-8, census format 36) carry the OT stubs' values in the barrel stages' fields (a
    wildcard everywhere: on a disc the global alpha and beta are never routed through curvature or z0, plan sec. 1
    B) and no components; their angles enter as module-frame quantities (the disc stages)."""
    disc = np.asarray(D["layer"]) > 4
    if not disc.any():
        return _prepare_barrel(D, model, bench)
    b = np.flatnonzero(~disc)
    Qb = _prepare_barrel({k: np.asarray(v)[b] for k, v in D.items()}, model, bench)
    fill = {**M.OT_STUB_ANGLES, **COMP_FILL, **CORR_FILL}
    Q = {"n_moved": Qb["n_moved"]}
    for k, v in Qb.items():
        if k == "n_moved":
            continue
        v = np.asarray(v)
        out = np.full(len(disc), fill[k], dtype=v.dtype)
        out[b] = v
        Q[k] = out
    d = np.flatnonzero(disc)
    Cd = disc_components({k: np.asarray(v)[d] for k, v in D.items()})
    Cd.update(dict(zip(DISC_K_FILL, disc_knob(Cd, model.k["alpha"], model.k["beta"], model.mode))))
    for k, v in Cd.items():
        out = np.full(len(disc), {**DISC_FILL, **DISC_K_FILL}[k], dtype=np.asarray(v).dtype)
        out[d] = v
        Q[k] = out
    return Q


def _prepare_barrel(D, model, bench=None):
    """prepare on IT barrel clusters."""
    C = components(D, model)
    h, sh, lam, sl = corrected_of(C, model.k["alpha"], model.k["beta"], model.mode)
    kap_a, s_kap, cot, s_cot, z0, s_z0 = represent(C["r"], C["z"], h, sh, lam, sl)
    ovf_a = np.zeros(len(z0), bool)
    ovf_z = np.zeros(len(z0), bool)
    if bench is not None:
        ab, bb = bench
        q, ovf_a, _ = M.quantize(np.sin(h), ab, M.ALPHA_RANGE)
        kap_a = q / (M.C_BEND * C["r"])
        s_kap = np.hypot(s_kap, M.quant_sigma(ab, M.ALPHA_RANGE) / (M.C_BEND * C["r"]))
        z0, ovf_z, _ = M.quantize(z0, bb, M.Z0_RANGE)
        s_z0 = np.hypot(s_z0, M.quant_sigma(bb, M.Z0_RANGE))
        cot = (C["z"] - z0) / C["r"]
        s_cot = s_z0 / C["r"]
    moved = {}
    for a, ang in (("alpha", "half"), ("beta", "lam")):
        if model.k[a] < 1.0:
            m = SD.improve(a, C[f"{ang}_e"], C[f"{ang}_t"], C["is_T"], model.k[a])[1]
        else:
            m = np.zeros(len(z0), bool)
        moved[a] = {f"IL{L}": int(m[C["layer"] == L].sum()) for L in (1, 2, 3, 4)}
    Q = {"kap_a": kap_a, "s_kap": s_kap, "z0": z0, "s_z0": s_z0, "cot_b": cot, "s_cot": s_cot,
         "ovf_a": ovf_a, "ovf_z": ovf_z, "dir_ok": M.SPN.direction_usable(D),
         "half_c": h, "s_half_c": sh, "lam_c": lam, "s_lam_c": sl,
         "n_moved": moved}
    for k in COMP_KEYS:
        Q[k] = C[k]
    return Q


def disc_predict(Q, g, kap, cot, d0=None):
    """The module-frame (cotAlpha, cotBeta) a helix (kappa, cot theta, d0; TTTrack conventions) predicts at disc
    clusters g: its direction at the cluster has azimuth phi - half, sin(half) = c kappa r - d0 / r (the census's
    alpha), and dip cot theta; rotated into the module's D121 frame (spdefs.disc_local_from_global's relation:
    cotAlpha = sin(dir - phi_m) / cot, cotBeta = -sx cos(dir - phi_m) / cot)."""
    r = Q["lr"][g]
    s = M.C_BEND * np.asarray(kap, np.float64) * r - (0.0 if d0 is None else np.asarray(d0) / r)
    d = Q["lphi"][g] - np.arcsin(np.clip(s, -1.0, 1.0)) - Q["lphim"][g]
    c = np.where(np.abs(cot) > 1e-9, cot, 1e-9)
    return np.sin(d) / c, -Q["lsx"][g] * np.cos(d) / c


def _disc_barrel_form(Q, gA, gB, kap, useA, useB):
    """The barrel-form statistics on a disc pair (the known-bad): S2a, the larger |kappa_pair - kappa_alpha| /
    sigma of its disc clusters from their global alpha (alpha through curvature); S2z, the two clusters' global-beta
    z0 against each other (beta through z0), both disc clusters needed (else 0)."""
    def comp(g):
        r = Q["lr"][g]
        half = SD.wrap_2pi(Q["lphi"][g] - Q["lg_dir"][g])
        ka = np.sin(half) / (M.C_BEND * r)
        ska = Q["lg_sdir"][g] * np.abs(np.cos(half)) / (M.C_BEND * r)
        cot = Q["lg_cot"][g]
        return ka, ska, Q["lz"][g] - r * cot, r * Q["lg_scot"][g]
    kA, sA, zA, szA = comp(gA)
    kB, sB, zB, szB = comp(gB)
    st_a = s2a_stat(kap, kA, sA, kB, sB, useA, useB)
    st_z = np.where(useA & useB, s2z_stat(zA, szA, zB, szB), 0.0)
    return st_a, st_z


def disc_components(D):
    """Per IT disc cluster (codes 5-8): the inputs of every disc statistic: its code, position, module frame
    (sx, phi_m from the D121 table), emitted and true module-frame cots, published cot sigmas, class T, and
    whether its module-frame direction is usable (SPN.direction_usable)."""
    sx, phm = SD.disc_frame_lookup(D["detId"])
    is_T = SD.class_T(D["tpEmuClass"]) & np.isfinite(np.asarray(D["tpLocalCotAlpha"], np.float64)) \
        & np.isfinite(np.asarray(D["tpLocalCotBeta"], np.float64))
    return {"lcode": np.asarray(D["layer"]).astype(np.int8), "lr": np.asarray(D["globalR"], np.float64),
            "lz": np.asarray(D["globalZ"], np.float64), "lphi": np.asarray(D["globalPhi"], np.float64),
            "lsx": sx.astype(np.float64), "lphim": phm,
            "lca_e": np.asarray(D["localCotAlpha"], np.float64), "lcb_e": np.asarray(D["localCotBeta"], np.float64),
            "lca_t": np.where(is_T, np.asarray(D["tpLocalCotAlpha"], np.float64), np.nan),
            "lcb_t": np.where(is_T, np.asarray(D["tpLocalCotBeta"], np.float64), np.nan),
            "lca_pub": np.asarray(D["sigAlpha"], np.float64), "lcb_pub": np.asarray(D["sigBeta"], np.float64),
            "lis_T": is_T, "ldir_ok": M.SPN.direction_usable(D) & (sx != 0),
            # the published GLOBAL disc direction, for the import gate's barrel-form known-bads only
            "lg_dir": np.asarray(D["globalClusterPhi"], np.float64),
            "lg_cot": np.asarray(D["globalClusterCotTheta"], np.float64),
            "lg_sdir": np.asarray(D["sigGlobalClusterPhi"], np.float64),
            "lg_scot": np.asarray(D["sigGlobalClusterCotTheta"], np.float64)}


def disc_knob(C, k_alpha, k_beta, mode):
    """(lca_k, lcb_k, lca_sk, lcb_sk) of disc components C at (k, mode): the knob on the module-frame cots of
    class T (truth + k (emitted - truth), no wrap: cot units) and sigma_k."""
    out = []
    for m, k in (("lca", k_alpha), ("lcb", k_beta)):
        e, t = C[f"{m}_e"], C[f"{m}_t"]
        v = np.where(C["lis_T"] & np.isfinite(t), t + SD.check_k(k) * (e - t), e) if k < 1.0 else np.array(e)
        out.append(v)
    out += [SD.sigma_k(C["lca_pub"], k_alpha, mode), SD.sigma_k(C["lcb_pub"], k_beta, mode)]
    return tuple(out)


# the disc components carried in Q (and their value elsewhere), and the knobbed values
DISC_FILL = {"lcode": 0, "lr": np.nan, "lz": np.nan, "lphi": np.nan, "lsx": 0.0, "lphim": np.nan,
             "lca_e": np.nan, "lcb_e": np.nan, "lca_t": np.nan, "lcb_t": np.nan, "lca_pub": np.nan,
             "lcb_pub": np.nan, "lis_T": False, "ldir_ok": False, "lg_dir": np.nan, "lg_cot": np.nan,
             "lg_sdir": np.nan, "lg_scot": np.nan}
DISC_KEYS = tuple(DISC_FILL)
DISC_K_FILL = {"lca_k": np.nan, "lcb_k": np.nan, "lca_sk": np.nan, "lcb_sk": np.nan}


# per-cluster component arrays carried in Q (and their value on an OT stub)
COMP_FILL = {"r": np.nan, "z": np.nan, "layer": 0, "cot_e": np.nan, "is_T": False,
             "half_e": np.nan, "lam_e": np.nan, "half_t": np.nan, "lam_t": np.nan,
             "sa_pub": np.nan, "sb_pub": np.nan, "row": -1, "ca": 0.0, "wa": 1.0,
             "cb": 0.0, "wb": 1.0}
COMP_KEYS = tuple(COMP_FILL)
CORR_FILL = {"half_c": 0.0, "s_half_c": 1e9, "lam_c": 0.0, "s_lam_c": 1e9}


def with_ot_stubs(QI, n):
    """The model's Q extended by n trailing OT stubs: M.OT_STUB_ANGLES for the stage fields (huge
    sigmas: a wildcard everywhere), no components (barrel or disc)."""
    Q = M.q_with_ot_stubs(QI, n)
    for k, v in {**COMP_FILL, **CORR_FILL, **DISC_FILL, **DISC_K_FILL}.items():
        if k not in QI:                       # a chunk without disc clusters (discs off: format 35's Q)
            continue
        Q[k] = np.r_[QI[k], np.full(n, v, dtype=np.asarray(QI[k]).dtype)]
    Q["n_moved"] = QI["n_moved"]
    return Q


# ==========================================================================
# statistics (in-run and offline)
# ==========================================================================
def s1_stat(half_c, s_half_c, r, floor):
    """S1: the single-cluster alpha test against the pT floor, in table units (spdefs.s1_statistic
    on the corrected angle; negative inside the band)."""
    return SD.s1_statistic(half_c, s_half_c, 0.0, r, floor)


def s2a_stat(kap_pair, kapA, skapA, kapB, skapB, useA, useB):
    """S2a: the larger of the two clusters' |kappa_pair - kappa_alpha| / sigma; a cluster not used
    (no usable direction, or an OT stub) contributes 0."""
    a = np.where(useA, np.abs(kap_pair - kapA) / np.maximum(skapA, 1e-12), 0.0)
    b = np.where(useB, np.abs(kap_pair - kapB) / np.maximum(skapB, 1e-12), 0.0)
    return np.maximum(a, b)


def s2z_stat(z0A, sA, z0B, sB):
    """S2z: |z0_A - z0_B| / hypot(sigma_A, sigma_B)."""
    return np.abs(z0A - z0B) / np.maximum(np.hypot(sA, sB), 1e-12)


def cand_pulls(kap_a, s_kap, cot, s_cot, kap_pred, cot_pred, s_kap_pred=0.0, s_cot_pred=0.0):
    """(alpha pull, beta pull) of a candidate against the seed's prediction; the prediction errors
    enter only for a weighting consumer (S4a)."""
    pa = (kap_a - kap_pred) / np.maximum(np.hypot(s_kap, s_kap_pred), 1e-12)
    pb = (cot - cot_pred) / np.maximum(np.hypot(s_cot, s_cot_pred), 1e-12)
    return pa, pb


def ptbin(kap):
    """Seed-measured pT bin of the S3 / S4w cut (from the seed's |kappa|, 1/GeV)."""
    pt = 1.0 / np.maximum(np.abs(np.asarray(kap, np.float64)), 1e-9)
    return np.searchsorted(np.asarray(PT_EDGES), pt, side="right").astype(np.int8)


def s6_stat(fit, sub):
    """S6's (chi2, n_meas) of fitted tracks over the angle subset `sub` ("ab", "a", "b")."""
    c2 = np.zeros(len(fit["chi2_a"]))
    n = np.zeros(len(c2), np.int32)
    if "a" in sub:
        c2 = c2 + fit["chi2_a"]
        n = n + fit["n_a"]
    if "b" in sub:
        c2 = c2 + fit["chi2_b"]
        n = n + fit["n_b"]
    return c2, n


def chi2_n_tail(chi2, n):
    """Upper tail of chi2_n (the fallback reference shape; table pulls have heavier tails)."""
    from scipy import stats
    return stats.chi2.sf(np.asarray(chi2, np.float64), np.maximum(np.asarray(n), 1))


# ==========================================================================
# truth for the records
# ==========================================================================
def tp_info(U, event, tpidx):
    """(tp_key, pT, genealogy code) of each (event, tpIdx): L1TTP truth; code -1 and NaN where the
    TP has no L1TTP row."""
    L = U["l1ttp"]
    tpidx = np.asarray(tpidx)
    p, has = M.tp_rows(L, event, tpidx)
    key = M.tp_key(np.asarray(event, np.int64), np.maximum(tpidx, 0).astype(np.int64))
    pt = np.where(has, L["pt"][p], np.nan)
    rprod = np.hypot(L["vx"][p], L["vy"][p])
    hf = (L["genFromB"][p] > 0) | (L["genFromC"][p] > 0)
    hs = L["evt"][p] == 0
    gen = np.where(rprod > BORN_IN_CM, GEN_BEYOND,
                   np.where(hf, GEN_HF, np.where(hs, GEN_PROMPT_HS, GEN_PROMPT_PU)))
    gen = np.where(has, gen, GEN_NONE).astype(np.int8)
    return key, pt, gen


def credited_to(U, g, tp):
    """Whether each hit g is credited to TP tp (U["tps"])."""
    return (U["tps"][np.asarray(g)] == np.asarray(tp)[:, None]).any(axis=1)


# ==========================================================================
# records
# ==========================================================================
class Recorder:
    """Per-consumer records of one chunk: correct candidates (the hit belongs to a TP every seed
    hit is credited to, the TP has an L1TTP row) and their statistic BEFORE the consumer's own cut,
    whether they passed it, and the components needed to re-evaluate the statistic offline at
    another (k, mode). Class C and every genealogy are recorded; the calibration population (class
    T, born inside 2.5 cm, pT above the stage's floor) is selected offline."""

    def __init__(self, U, Q, model):
        self.U, self.Q, self.model = U, Q, model
        self.cols = {}
        self.n_dec = 0

    def _add(self, consumer, **cols):
        n = None
        for k, v in cols.items():
            v = np.asarray(v)
            n = len(v) if n is None else n
            if len(v) != n:
                raise ValueError(f"{consumer}.{k}: {len(v)} rows, expected {n}")
            self.cols.setdefault(consumer, {}).setdefault(k, []).append(v)

    def _cl(self, g, prefix=""):
        """Component columns of clusters g (and whether each has a usable direction)."""
        Q = self.Q
        return {**{f"{prefix}{k}": Q[k][g] for k in COMP_KEYS}, f"{prefix}dir_ok": Q["dir_ok"][g]}

    def _cld(self, g, prefix=""):
        """Disc component columns of clusters g (census format 36)."""
        return {f"{prefix}{k}": self.Q[k][g] for k in DISC_KEYS}

    def _truthful(self, g):
        """(indices, tp_key, tp pT, genealogy) of clusters g with a credited TP that has an L1TTP row."""
        U = self.U
        i = np.flatnonzero(M.SPN.tp_matched(U["tpIdx"][g]))
        key, pt, gen = tp_info(U, U["event"][g[i]], U["tpIdx"][g[i]])
        keep = gen != GEN_NONE
        return i[keep], key[keep], pt[keep], gen[keep]

    # ---- the disc consumers (census format 36) -------------------------
    def s1d(self, floor, cuts=None):
        """S1 on every IT disc cluster with a usable module-frame direction and a credited TP, once per chunk
        and floor; cuts: the run's cut set (None: the calibration run)."""
        U, Q = self.U, self.Q
        g = np.flatnonzero(np.isin(U["layer"], M.IT_DISC_CODES) & Q["ldir_ok"])
        if not len(g):
            return
        i, key, pt, gen = self._truthful(g)
        g = g[i]
        st, rows = self.model.s1d(Q, g, floor)
        ok = np.ones(len(g), bool) if cuts is None else st <= cuts.s1d_thr(floor, rows)
        self._add("s1d", event=U["event"][g], tp_key=key, tp_pt=pt, gen=gen, floor=np.full(len(g), floor),
                  row=rows.astype(np.int64), stat=st, passed=ok, **self._cld(g))

    def pairs_disc(self, gA, gB, kap, cot, floor, la, lb, stat_a, stat_z, row_a, pass_a, pass_z):
        """Correct disc pairs (both hits credited to one TP, at least one an IT disc cluster)."""
        U = self.U
        dA = np.isin(U["layer"][gA], M.IT_DISC_CODES)
        tp = np.where(dA, U["tpIdx"][gA], U["tpIdx"][gB])
        g_other = np.where(dA, gB, gA)
        m = M.SPN.tp_matched(tp)
        idx = np.flatnonzero(m)
        idx = idx[credited_to(U, g_other[idx], tp[idx])] if len(idx) else idx
        if not len(idx):
            return
        key, pt, gen = tp_info(U, U["event"][gA[idx]], tp[idx])
        keep = gen != GEN_NONE
        idx, key, pt, gen = idx[keep], key[keep], pt[keep], gen[keep]
        n = len(idx)
        self._add("paird", event=U["event"][gA[idx]], tp_key=key, tp_pt=pt, gen=gen,
                  floor=np.full(n, floor), la=np.full(n, la, np.int8), lb=np.full(n, lb, np.int8),
                  kap_pair=kap[idx], cot_pair=cot[idx], stat_a=stat_a[idx], stat_z=stat_z[idx],
                  A_row=row_a[idx].astype(np.int64), passed_a=pass_a[idx], passed_z=pass_z[idx],
                  **self._cld(gA[idx], "A_"), **self._cld(gB[idx], "B_"))

    def window_disc(self, role, scode, lc, floor, ja, gC, seed_hits, kap_pred, cot_pred, d0_pred, pa, pb, passed,
                    ptb, survived):
        """Correct candidates of one disc projection slice (S4w on an IT disc target)."""
        U = self.U
        tp = U["tpIdx"][gC]
        m = survived & M.SPN.tp_matched(tp)
        for g in seed_hits:
            if not m.any():
                return
            m[m] = credited_to(U, g[ja[m]], tp[m])
        idx = np.flatnonzero(m)
        if not len(idx):
            return
        key, pt, gen = tp_info(U, U["event"][gC[idx]], tp[idx])
        keep = gen != GEN_NONE
        idx, key, pt, gen = idx[keep], key[keep], pt[keep], gen[keep]
        n = len(idx)
        self._add("wind", event=U["event"][gC[idx]], tp_key=key, tp_pt=pt, gen=gen,
                  floor=np.full(n, floor), role=np.full(n, role, np.int8), seed=np.full(n, scode, np.int64),
                  lc=np.full(n, lc, np.int8), ptbin=ptb[idx], kap_pred=kap_pred[idx], cot_pred=cot_pred[idx],
                  d0_pred=d0_pred[idx], pa=pa[idx], pb=pb[idx], passed=passed[idx], **self._cld(gC[idx]))

    # ---- S1 ----------------------------------------------------------
    def s1(self, floor, thr=None):
        """S1 over every IT cluster with a usable direction and a credited TP, once per chunk and
        floor (the test is per cluster: every seed applies the same one)."""
        U, Q = self.U, self.Q
        it = np.flatnonzero((U["layer"] <= 4) & Q["dir_ok"] & M.SPN.tp_matched(U["tpIdx"]))
        if not len(it):
            return
        key, pt, gen = tp_info(U, U["event"][it], U["tpIdx"][it])
        keep = gen != GEN_NONE
        it, key, pt, gen = it[keep], key[keep], pt[keep], gen[keep]
        st = self.model.s1(Q, it, floor)
        ok = np.ones(len(it), bool) if thr is None else st <= thr[it]
        self._add("s1", event=U["event"][it], tp_key=key, tp_pt=pt, gen=gen,
                  floor=np.full(len(it), floor), stat=st, passed=ok, **self._cl(it))

    # ---- S2a / S2z ---------------------------------------------------
    def correct_pairs(self, idxA, idxB):
        """Every (A, B) of the two layers' cluster sets whose hits are credited to one TP (U["tps"]),
        each pair once: the truth enumeration of the S2a / S2z records."""
        U = self.U

        def keyed(g):
            g = np.asarray(g)
            T = U["tps"][g]
            r, c = np.nonzero(M.SPN.tp_matched(T))
            return M.tp_key(U["event"][g[r]].astype(np.int64), T[r, c].astype(np.int64)), g[r]
        kA, gA = keyed(idxA)
        kB, gB = keyed(idxB)
        if not len(kA) or not len(kB):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        o = np.argsort(kB, kind="stable")
        kB, gB = kB[o], gB[o]
        lo = np.searchsorted(kB, kA, "left")
        cnt = np.searchsorted(kB, kA, "right") - lo
        ia = np.repeat(np.arange(len(kA)), cnt)
        ib = lo[ia] + (np.arange(len(ia)) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        pr = np.unique(np.stack([gA[ia], gB[ib]], axis=1), axis=0) if len(ia) else np.zeros((0, 2), np.int64)
        return pr[:, 0], pr[:, 1]

    def pairs(self, gA, gB, kap, floor, la, lb, stat_a, stat_z, pass_a, pass_z):
        """Correct pairs (both hits credited to one TP, at least one an IT cluster)."""
        U = self.U
        if la > 4 and lb > 4:
            return
        itA = U["layer"][gA] <= 4
        tpA = np.where(itA, U["tpIdx"][gA], U["tpIdx"][gB])     # an IT cluster's TP
        g_other = np.where(itA, gB, gA)
        m = M.SPN.tp_matched(tpA)
        if not m.any():
            return
        idx = np.flatnonzero(m)
        idx = idx[credited_to(U, g_other[idx], tpA[idx])]
        if not len(idx):
            return
        key, pt, gen = tp_info(U, U["event"][gA[idx]], tpA[idx])
        keep = gen != GEN_NONE
        idx, key, pt, gen = idx[keep], key[keep], pt[keep], gen[keep]
        n = len(idx)
        self._add("pair", event=U["event"][gA[idx]], tp_key=key, tp_pt=pt, gen=gen,
                  floor=np.full(n, floor), la=np.full(n, la, np.int8), lb=np.full(n, lb, np.int8),
                  kap_pair=kap[idx], stat_a=stat_a[idx], stat_z=stat_z[idx],
                  passed_a=pass_a[idx], passed_z=pass_z[idx],
                  **self._cl(gA[idx], "A_"), **self._cl(gB[idx], "B_"))

    # ---- S3 / S4w (and S4a decisions) --------------------------------
    def window(self, role, scode, lc, floor, ja, gC, seed_hits, kap_pred, cot_pred, pa, pb,
               passed, ptb, skap, sct, survived):
        """Correct candidates of one projection slice (target cluster credited to a TP every seed
        hit is credited to). `survived` marks the candidates the position windows kept."""
        U = self.U
        if lc > 4:
            return
        tp = U["tpIdx"][gC]
        m = survived & M.SPN.tp_matched(tp)
        for g in seed_hits:
            if not m.any():
                return
            m[m] = credited_to(U, g[ja[m]], tp[m])
        idx = np.flatnonzero(m)
        if not len(idx):
            return
        key, pt, gen = tp_info(U, U["event"][gC[idx]], tp[idx])
        keep = gen != GEN_NONE
        idx, key, pt, gen = idx[keep], key[keep], pt[keep], gen[keep]
        n = len(idx)
        self._add("win", event=U["event"][gC[idx]], tp_key=key, tp_pt=pt, gen=gen,
                  floor=np.full(n, floor), role=np.full(n, role, np.int8),
                  seed=np.full(n, scode, np.int64), lc=np.full(n, lc, np.int8),
                  ptbin=ptb[idx], kap_pred=kap_pred[idx], cot_pred=cot_pred[idx],
                  skap=np.broadcast_to(skap, len(pa))[idx], sct=sct[idx],
                  pa=pa[idx], pb=pb[idx], passed=passed[idx], **self._cl(gC[idx]))

    def decisions(self, lc, ja, gC, seed_hits, kap_pred, cot_pred, skap, sct, pos):
        """S4a: every candidate of each decision (one seed object) that holds a correct candidate,
        with what both keys need."""
        correct_mask = correct_window_mask(self.U, ja, gC, seed_hits, lc)
        if lc > 4 or not correct_mask.any():
            return
        dec = np.unique(ja[correct_mask])
        sel = np.flatnonzero(np.isin(ja, dec))
        ids = self.n_dec + np.searchsorted(dec, ja[sel])
        self.n_dec += len(dec)
        self._add("arb", dec=ids, event=self.U["event"][gC[sel]], correct=correct_mask[sel],
                  pos=pos[sel],
                  kap_pred=kap_pred[sel], cot_pred=cot_pred[sel],
                  skap=np.broadcast_to(skap, len(ja))[sel], sct=sct[sel], **self._cl(gC[sel]))

    # ---- S6 ----------------------------------------------------------
    def tracks(self, **cols):
        self._add("trk", **cols)

    def arrays(self):
        return {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in self.cols.items()}


def record_s6(rec, U, Q, fit, keep, passed, G, trip, floor, build, entry, c2, n6, fopt):
    """S6 records: correct tracks among those reaching S6 (`keep`): every hit credited to the
    track's unique owner, which has an L1TTP row. Per track: the run's statistic and pass, the chi2
    parts (alpha, beta) under every S5 state and key (S5_REFITS: refits of the same hits; S5 does
    not change hit selection), the fitted (d0, z0, kappa) under S5 with both angles in both keys
    and the owner's truth (the S5 key choice), and whether every angle-bearing hit is class T."""
    idx = np.flatnonzero(keep)
    if not len(idx):
        return
    on = G[idx] >= 0
    own, _, owned, right, _ = KF.credited_owner(M.hit_tps(U, G[idx]), on)
    corr = owned & (right == on).all(axis=1) & (own >= 0)
    idx, own = idx[corr], own[corr]
    if not len(idx):
        return
    ev = U["event"][trip[0][idx]]
    key, pt, gen = tp_info(U, ev, own)
    k = gen != GEN_NONE
    idx, own, ev, key, pt, gen = idx[k], own[k], ev[k], key[k], pt[k], gen[k]
    if not len(idx):
        return
    T = fit["hits"]
    use = T["VALID"][idx] & T["ANG"][idx]
    gi = np.clip(T["GIDX"][idx], 0, None)
    all_T = (~use | Q["is_T"][gi]).all(axis=1)
    if "lis_T" in Q:                              # the disc hits' angles too (format 36)
        dang = T["DANG"][idx]
        all_T &= (~dang | Q["lis_T"][gi]).all(axis=1)
    trip_c = tuple(x[idx] for x in trip)
    n = len(idx)
    disc = (T["VALID"][idx] & T["DANG"][idx]).any(axis=1) if "DANG" in T else np.zeros(n, bool)
    cols = {"event": ev, "tp_key": key, "tp_pt": pt, "gen": gen, "all_T": all_T,
            "floor": np.full(n, float(floor)), "build": np.array([build] * n),
            "disc": disc, "s6g": s6_group(build, disc),
            "entry": np.full(n, entry, np.int32), "stat": c2[idx], "n6": n6[idx],
            "passed": passed[idx], "n_a": fit["n_a"][idx], "n_b": fit["n_b"][idx]}
    L = U["l1ttp"]
    p, has = M.tp_rows(L, ev, own)
    for c in ("d0", "z0", "pt"):
        cols[f"tp_{c}"] = np.where(has, L[c][p], np.nan)
    d0p = fopt.get("d0_prior_cm", KF.D0_PRIOR_CM)
    for s5, kk in S5_REFITS:
        f = KF.fit_tracks(U, Q, trip_c, gidx=G[idx], use_angles=s5 != "off",
                          angles=SUBSETS[s5] if s5 != "off" else (), key=kk,
                          model=fopt["model"], d0_prior_cm=d0p)
        cols[f"c2a_{s5}_{kk}"] = f["chi2_a"]
        cols[f"c2b_{s5}_{kk}"] = f["chi2_b"]
        cols[f"ok_{s5}_{kk}"] = f["ok"]
        if s5 == "ab":
            for c in ("d0", "z0", "kappa"):
                cols[f"fit_{c}_{kk}"] = f[c]
    rec.tracks(**cols)



def correct_window_mask(U, ja, gC, seed_hits, lc):
    """The candidates of a projection whose target is credited to a TP every seed hit is credited
    to (truth only: the S4a decision records)."""
    if lc > 4:
        return np.zeros(len(gC), bool)
    tp = U["tpIdx"][gC]
    m = M.SPN.tp_matched(tp)
    for g in seed_hits:
        if not m.any():
            break
        m[m] = credited_to(U, g[ja[m]], tp[m])
    return m


def load_records(d):
    """{consumer: {column: array}} of a census directory's records, over every shard."""
    import glob
    out = {}
    for sh in sorted(glob.glob(os.path.join(d, "chunk_*.npz"))):
        z = np.load(sh, allow_pickle=False)
        for f in z.files:
            if f.startswith("rec_"):
                c, k = f[4:].split("__", 1)
                out.setdefault(c, {}).setdefault(k, []).append(z[f])
    return {c: {k: np.concatenate(v) for k, v in d_.items()} for c, d_ in out.items()}


# ==========================================================================
# the cut file
# ==========================================================================
def _k(*parts):
    return "|".join(str(p) for p in parts)


def _bins_thr(doc, consumer):
    """(bins, thresholds) of a cut file's consumer section ((), () when absent)."""
    sec = doc.get(consumer)
    return (sec["bins"], sec["thr"]) if sec else ((), ())


class CutSet:
    """A census angle cut file: per consumer, per bin, the threshold of its own statistic at the
    cut level, with the identity it was made under. Lookups refuse a configuration the file was
    not made for."""

    def __init__(self, doc, path=None):
        if doc.get("schema") != CUT_SCHEMA:
            raise ValueError(f"not a census angle cut file ({doc.get('schema')!r})")
        self.doc = doc
        self.path = path
        self.sha = sha256(path) if path else None
        self.ident = doc["identity"]
        self.s1 = {_k(*b): t for b, t in zip(doc["s1"]["bins"], doc["s1"]["thr"])}
        self.s2a = {_k(*b): t for b, t in zip(doc["s2a"]["bins"], doc["s2a"]["thr"])}
        self.s2z = {_k(*b): t for b, t in zip(doc["s2z"]["bins"], doc["s2z"]["thr"])}
        self.win = {}
        for sub, rows in doc["win"]["thr"].items():
            for b, t in zip(doc["win"]["bins"], rows):
                self.win[_k(sub, *b)] = t
        self.s6 = {_k(*b): t for b, t in zip(doc["s6"]["bins"], doc["s6"]["thr"])}
        # the disc consumers (census format 36; a format-35 file has none)
        self.s1d = {_k(*b): t for b, t in zip(*_bins_thr(doc, "s1d"))}
        self.s2ad = {_k(*b): t for b, t in zip(*_bins_thr(doc, "s2ad"))}
        self.s2zd = {_k(*b): t for b, t in zip(*_bins_thr(doc, "s2zd"))}
        self.wind = {}
        for sub, rows in doc.get("wind", {}).get("thr", {}).items():
            for b, t in zip(doc["wind"]["bins"], rows):
                self.wind[_k(sub, *b)] = t

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f), path)

    def check(self, model, run):
        """Refuse a run the file was not made for. `run`: the run's identity fields (census format,
        bench, d0_prior_cm, floors, s5 state and key, S6 angle subset, S3 / S4w subsets, builds)."""
        I = self.ident
        bad = []
        if I["table_sha256"] != model.table_sha:
            bad.append("table sha256")
        ck = model.calib_k()
        if (I["k_alpha"], I["k_beta"]) != tuple(float(x) for x in ck):
            bad.append(f"calibration k {I['k_alpha'], I['k_beta']} != {ck}")
        if model.mode == "scaled" and I["mode"] != "scaled" and min(ck) < 1.0:
            bad.append("scaled-mode run needs a scaled-mode cut file")
        for k in ("bench", "d0_prior_cm", "level"):
            if I.get(k) != run.get(k):
                bad.append(f"{k} {I.get(k)!r} != {run.get(k)!r}")
        # the thresholds are quantiles of the calibration census's own candidates: a census format
        # moves those numbers (e.g. format 35's OT inputs), so a file calibrated on another is refused
        fmt = I.get("calibration", {}).get("format")
        if fmt != run.get("format"):
            bad.append(f"calibration census format {fmt!r} != this census's {run.get('format')!r}")
        miss = [f for f in run["floors"] if f not in I["floors"]]
        if miss:
            bad.append(f"floors {miss} not calibrated")
        if run.get("s6"):
            s5, key, sub = run["s6"]
            if not any(k.startswith(_k(b, s5, key, sub) + "|") for b in run["builds"] for k in self.s6):
                bad.append(f"S6 not calibrated for S5 state {s5!r} key {key!r} angle set {sub!r}")
        for sub in run.get("win_subsets", ()):
            if not any(k.startswith(sub + "|") for k in self.win):
                bad.append(f"S3/S4w not calibrated for angle set {sub!r}")
        if run.get("s6") and run.get("it_discs"):
            s5, key, sub = run["s6"]
            if not any(k.startswith(_k(b + S6_DISC, s5, key, sub) + "|") for b in run["builds"] for k in self.s6):
                bad.append("S6 not calibrated for tracks with disc angle measurements")
        for c in run.get("disc_stages", ()):
            tab = getattr(self, c)
            if not tab or (c == "wind" and not any(k.startswith(sub + "|") for sub in run.get("win_subsets", ())
                                                   for k in tab)):
                bad.append(f"no disc thresholds ({c}) for a run with the IT discs on")
        if bad:
            raise SystemExit(f"angle cut file {self.path} refused for this run: " + "; ".join(bad))

    def _get(self, table, key, what):
        if key not in table:
            raise SystemExit(f"angle cut file {self.path}: no {what} threshold for bin {key}")
        return table[key]

    def s1_thr(self, floor, rows):
        rows = np.asarray(rows)
        out = np.full(len(rows), np.inf)
        for r in np.unique(rows[rows >= 0]):
            out[rows == r] = self._get(self.s1, _k(floor, int(r)), "S1")
        return out

    def s2a_thr(self, floor, la, lb):
        return self._get(self.s2a, _k(floor, la, lb), "S2a")

    def s2z_thr(self, floor, la, lb, rowsA):
        rows = np.asarray(rowsA)
        out = np.full(len(rows), np.inf)
        for r in np.unique(rows[rows >= 0]):
            out[rows == r] = self._get(self.s2z, _k(floor, la, lb, int(r)), "S2z")
        return out

    def win_thr(self, sub, role, scode, lc, floor, ptb):
        """(q_alpha, q_beta) per candidate (inf for an angle not in the subset)."""
        ptb = np.asarray(ptb)
        qa = np.full(len(ptb), np.inf)
        qb = np.full(len(ptb), np.inf)
        for b in np.unique(ptb):
            t = self._get(self.win, _k(sub, role, scode, lc, b, floor), "S3/S4w")
            qa[ptb == b], qb[ptb == b] = t
        return qa, qb

    def s1d_thr(self, floor, rows):
        """S1 on the discs: the threshold per disc table row (inf where none)."""
        rows = np.asarray(rows)
        out = np.full(len(rows), np.inf)
        for r in np.unique(rows[rows >= 0]):
            out[rows == r] = self._get(self.s1d, _k(floor, int(r)), "S1 disc")
        return out

    def s2ad_thr(self, floor, la, lb):
        return self._get(self.s2ad, _k(floor, la, lb), "S2a disc")

    def s2zd_thr(self, floor, la, lb, rowsA):
        rows = np.asarray(rowsA)
        out = np.full(len(rows), np.inf)
        for r in np.unique(rows[rows >= 0]):
            out[rows == r] = self._get(self.s2zd, _k(floor, la, lb, int(r)), "S2z disc")
        return out

    def wind_thr(self, sub, role, scode, lc, floor, ptb):
        """S4w on a disc target: (q_alpha, q_beta) per candidate (inf for an angle not in the subset)."""
        ptb = np.asarray(ptb)
        qa = np.full(len(ptb), np.inf)
        qb = np.full(len(ptb), np.inf)
        for b in np.unique(ptb):
            t = self._get(self.wind, _k(sub, role, scode, lc, b, floor), "S4w disc")
            qa[ptb == b], qb[ptb == b] = t
        return qa, qb

    def s6_thr(self, build, s5, key, sub, n, disc=None):
        """Per track: S6's threshold at its n in its S6 group (the build; `disc`, per track, whether it carries
        a disc angle measurement: s6_group)."""
        n = np.asarray(n)
        g = s6_group(build, np.zeros(len(n), bool) if disc is None else disc)
        out = np.full(len(n), np.inf)
        for gv, v in {(a, int(b)) for a, b in zip(g[n > 0], n[n > 0])}:
            out[(g == gv) & (n == v)] = self._get(self.s6, _k(gv, s5, key, sub, v), "S6")
        return out
