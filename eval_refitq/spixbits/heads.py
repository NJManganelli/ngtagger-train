"""The deployed per-track quality heads on replay output (PLAN.md section 7, item 2).

Deployed heads = the reference heads of heads_cmssw/SPEC.md (a local copy in cache/heads): per active-layer mask and
window (prompt: |refit d0| <= 0.02 cm, else displaced) six conifer BDTs on 38 inputs: clean (P = 1 / (1 + exp(-m))) and
log sigma_hat of d0, z0 [um], tanL [1e-5], 1/R [1e-6 / cm], phi [urad]. Missing inputs are -999, never NaN; a split
goes left when x < threshold (strict).

The converter builds the 38 inputs of an arm from the replay's output on the extended collection:
  inputs 1-13    from the OT track (the nano's L1TExtTrack, unchanged by any arm)
  inputs 14-31   from the arm's delivered state and record; the three the replay does not write are computed as the
                 producer does: spixChi2ITAtRefit (accepted hits against the final helix, same module only, with the
                 applied position errors), spixShiftChi2 (da^T (C_OT - C_refit)^-1 da, -999 when not invertible or
                 negative) and spixLogDetRatio (ln det C_OT - ln det C_refit), C_OT after the weak d0 prior;
                 spixChi2ITAtOT is -999 (as in refit v3)
  inputs 32-38   aggregates over the accepted hits (trackmva.hit_aggregates). An angle that is not sent, or that sits
                 on an underflow / overflow code, counts as "the hit has no such angle" (missing), never as a pull
                 of 999
The per-hit chi2 increments are rounded to 12 mantissa bits, as the nano stores them (the models were trained on that).
No truth column enters: the nano's branches are screened with trackmva.refuse_truth_inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np

from ngtagger.train import trackmva as TM

HEADS = ("clean", "d0", "z0", "tanl", "rinv", "phi")
WINDOWS = ("prompt", "displaced")
MISSING = -999.0
FAMILY = "canonical"            # the replay reads the extended tables under the prompt tables' names (common.COLLECTION)


def round_mantissa(x, bits=12):
    """A float32 with its mantissa rounded to `bits` bits (NanoAOD's reduceMantissaToNbitsRounding)."""
    x = np.ascontiguousarray(x, np.float32)
    i = x.view(np.int32).astype(np.int64)
    shift = 23 - bits
    mask = (1 << shift) - 1
    half = 1 << (shift - 1)
    out = np.where((i & mask) != 0, ((i + half) >> shift) << shift, i)      # exact values are kept
    out = np.where(np.isfinite(x) & (x != 0), out, i).astype(np.int32)
    return out.view(np.float32).astype(np.float64)


class Heads:
    """The deployed heads of one mask: margins(X, d0_refit) -> (n, 6) in HEADS order."""

    def __init__(self, directory, mask):
        self.mask = mask
        self.models = {}
        for w in WINDOWS:
            for h in HEADS:
                m = json.load(open(Path(directory) / "SmartPixels" / mask / w / f"{h}.json"))
                assert m["splitting_convention"] == "<" and m["n_features"] == 38
                trees = [t[0] for t in m["trees"]]
                n = max(len(t["feature"]) for t in trees)
                pad = lambda key, fill, dt: np.array([list(t[key]) + [fill] * (n - len(t[key])) for t in trees], dt)
                self.models[(w, h)] = {"init": float(m["init_predict"][0]), "feature": pad("feature", -2, np.int64),
                                       "threshold": pad("threshold", 0.0, np.float32), "value": pad("value", 0.0, np.float64),
                                       "left": pad("children_left", -1, np.int64), "right": pad("children_right", -1, np.int64),
                                       "depth": int(m["max_depth"])}

    KEYS = ("init", "feature", "threshold", "value", "left", "right", "depth")

    def save_npz(self, path):
        """The twelve models (two windows x six heads) as one compact file: keys <window>/<head>/<array>."""
        np.savez_compressed(path, **{f"{w}/{h}/{k}": np.asarray(self.models[(w, h)][k]) for w in WINDOWS for h in HEADS for k in self.KEYS})

    @classmethod
    def from_npz(cls, path, mask=""):
        """The reader on a file written by save_npz (the per-quantization-scenario heads)."""
        self = cls.__new__(cls)
        self.mask, Z = mask, np.load(path)
        self.models = {(w, h): {k: (Z[f"{w}/{h}/{k}"] if k not in ("init", "depth") else Z[f"{w}/{h}/{k}"].item()) for k in cls.KEYS}
                       for w in WINDOWS for h in HEADS}
        return self

    @staticmethod
    def _margin(M, X):
        out = np.full(len(X), M["init"])
        rows = np.arange(len(X))
        for k in range(len(M["feature"])):
            node = np.zeros(len(X), np.int64)
            for _ in range(M["depth"] + 1):
                f = M["feature"][k][node]
                leaf = f < 0
                go_left = X[rows, np.maximum(f, 0)] < M["threshold"][k][node]
                node = np.where(leaf, node, np.where(go_left, M["left"][k][node], M["right"][k][node]))
            out += M["value"][k][node]
        return out

    def margins(self, X, d0_refit):
        X = np.asarray(X, np.float32)
        prompt = np.abs(d0_refit) <= TM.D0_SPLIT
        out = np.full((len(X), len(HEADS)), np.nan)
        for w, sel in (("prompt", prompt), ("displaced", ~prompt)):
            idx = np.flatnonzero(sel)
            if len(idx):
                for i, h in enumerate(HEADS):
                    out[idx, i] = self._margin(self.models[(w, h)], X[idx])
        return out


def matrix(T):
    """The (n, 38) float32 input matrix in model order, with the -999 rule; and the names."""
    F = TM.features(T, "refit")
    X = np.column_stack([F[n] for n in F]).astype(np.float64)
    X[~np.isfinite(X)] = MISSING
    X[X <= -998] = MISSING
    return X.astype(np.float32), list(F)


def _extended_only():
    """The heads exist for the extended collection only; FAMILY names the prompt tables, which are the extended ones
    only while common.COLLECTION says so."""
    import common as CM
    assert CM.COLLECTION == "extended", "the deployed heads are defined on the extended track collection"


def nano_branches(keys):
    return TM.nano_branches(keys, FAMILY, "", with_truth=False)


def stored_columns(A):
    """The per-track columns of the nano's own (CMSSW) refit result: trackmva.tracks_from_batch."""
    _extended_only()
    return TM.tracks_from_batch(A, FAMILY, "", with_truth=False)[0]


def refuse_truth(path):
    import uproot
    import common as CM
    _extended_only()
    with uproot.open(str(path)) as fh:
        t = fh["Events"]
        titles = {b: t[CM.file_branch(b)].title for b in TM.online_branches(t.keys(), FAMILY, "")}
    TM.refuse_truth_inputs(titles, FAMILY, "")


def replay_columns(S, ctx_a, out, T_stored, kpt, cfg, alpha_sent=True):
    """The arm's r_* and h_* columns (the keys trackmva.features reads) from the replay output `out` of the context
    `ctx_a` (the arm's emulated clusters); the b_* columns are taken from T_stored. alpha_sent: the arm sends the
    SmartPixels beta (the code's "alpha", the only angle the heads read)."""
    import engine as E
    from common import OP
    geo, K, rec = S.geo, ctx_a["K"], out["rec"]
    n = ctx_a["ntrk"]
    refit, acc = out["refit"], rec["acc"] & out["refit"][:, None]
    a, C = out["a"], out["C"]
    T = {k: v for k, v in T_stored.items() if k.startswith("b_")}
    T["r_pt"] = np.where(refit, kpt / np.maximum(np.abs(a[:, 0]), 1e-12), T["b_pt"])
    T["r_tanL"], T["r_z0"], T["r_d0"] = (np.where(refit, a[:, i], T["b_" + c]) for i, c in ((2, "tanL"), (3, "z0"), (4, "d0")))
    surf = rec["surface"]
    bit = np.where(surf > 10, surf - 11 + 4, surf - 1)
    chi2 = np.where(acc[:, :, None], round_mantissa(rec["chi2"].ravel()).reshape(rec["chi2"].shape), 0.0)
    cnt = lambda x: np.where(refit, x, 0).astype(np.float64)
    T["r_spixRefitPerformed"] = refit.astype(np.float64)
    T["r_spixAnyWindowTruncated"] = np.zeros(n)
    T["r_spixNCrossings"] = cnt(rec["visited"].sum(1))
    T["r_spixNAcceptedHits"] = cnt(acc.sum(1))
    T["r_spixLayerHitMask"] = cnt(np.where(acc, 1 << np.maximum(bit, 0), 0).sum(1))
    T["r_spixMaxWindowMult"] = cnt(np.where(rec["visited"], rec["ncand"], 0).max(1))
    for i, c in enumerate(("X", "Y", "Alpha")):
        T[f"r_spixChi2Inc{c}Tot"] = np.where(refit, chi2[:, :, i].astype(np.float32).sum(1, dtype=np.float32), 0.0).astype(np.float64)
    T["r_spixChi2ITAtOT"] = np.full(n, MISSING)
    # the accepted hits against the final helix (same module only)
    t_, j_ = np.nonzero(acc)
    ci, mi, S_ = rec["ci"][t_, j_], rec["mi"][t_, j_], surf[t_, j_]
    c2 = np.zeros(len(t_))
    for s in np.unique(S_):
        m = np.flatnonzero(S_ == s)
        same = OP.cross_module(a[t_[m]], int(s), geo) == mi[m]
        pj = OP.project_to_modules(a[t_[m]], OP.surface_s(a[t_[m]], geo, int(s)), geo, mi[m])
        sg = rec["sigpos"][t_[m], j_[m]]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = ((K["localX"][ci[m]] - pj["u"]) / sg[:, 0]) ** 2 + ((K["localY"][ci[m]] - pj["v"]) / sg[:, 1]) ** 2
        c2[m] = np.where(same & pj["ok"] & (sg > 0).all(1), np.nan_to_num(v), 0.0)
    T["r_spixChi2ITAtRefit"] = np.where(refit, np.bincount(t_, weights=c2, minlength=n), MISSING)
    # shift chi2 and log-determinant ratio against the OT state after the weak d0 prior
    a_ot = ctx_a["a"]
    C_ot, _ = E.d0_floor(ctx_a["C"], cfg)
    shift = np.full(n, MISSING)
    ldr = np.full(n, MISSING)
    r = np.flatnonzero(refit)
    if len(r):
        da = a[r] - a_ot[r]
        dC = C_ot[r] - C[r]
        with np.errstate(all="ignore"):
            okinv = np.abs(np.linalg.det(dC)) > 0
            inv = np.zeros_like(dC)
            inv[okinv] = np.linalg.inv(dC[okinv])
            q = np.einsum("ni,nij,nj->n", da, inv, da)
            shift[r] = np.where(okinv & np.isfinite(q) & (q >= 0), q, MISSING)
            d1, d2 = np.linalg.det(C_ot[r]), np.linalg.det(C[r])
            lr = np.log(np.where(d1 > 0, d1, 1.0)) - np.log(np.where(d2 > 0, d2, 1.0))
            ldr[r] = np.where((d1 > 0) & (d2 > 0) & np.isfinite(lr), lr, MISSING)
    T["r_spixShiftChi2"], T["r_spixLogDetRatio"] = shift, ldr
    # per-hit aggregates
    has = (K["hasAlpha"][ci] > 0) & alpha_sent
    if "sb_flagAlpha" in K:
        has &= K["sb_flagAlpha"][ci] == 0
    cx, cy, ca = (chi2[t_, j_, i] for i in range(3))
    applied = rec["chi2"][t_, j_] > 0
    pull = lambda i: np.where(applied[:, i], np.sqrt(rec["chi2"][t_, j_, i]), MISSING)
    hits = {"hitAccepted": np.ones(len(t_)), "hasAlpha": has.astype(np.float64), "windowMult": rec["ncand"][t_, j_].astype(np.float64),
            "pullX": pull(0), "pullY": pull(1), "pullAlpha": pull(2), "chi2IncX": cx, "chi2IncY": cy,
            "chi2IncAlpha": np.where(has, ca, 0.0)}
    T.update(TM.hit_aggregates(hits, t_, n))
    T["b_nmiss_interior"] = T_stored["b_nmiss_interior"]
    return T
