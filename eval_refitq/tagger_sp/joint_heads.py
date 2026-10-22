"""Jointly trained track-quality BDT heads (tagger_v5/joint_heads/PLAN.md).

Six per-track heads -- clean (classifier) and d0, z0, tanL, 1/R, phi (regressions) -- trained JOINTLY in one
multi-output XGBoost model per window (one tree per output per round) with a custom objective: the sum of the
heads' own losses plus lambda x n x C, C = the sum over head pairs of the squared distance correlation of the
OUTPUTS (variant A) or of the ERRORS (variant B), on a fresh subsample of matched tracks every round. lambda = 0
is the uncoupled reference (identical, to float rounding, to six independently trained heads).

  python eval_refitq/tagger_sp/joint_heads.py extra          per file: the extra truth / track columns
  python eval_refitq/tagger_sp/joint_heads.py fit            per mask and window: the lambda scan (both variants)
  python eval_refitq/tagger_sp/joint_heads.py choose         the lambda of each variant (validation only)
  python eval_refitq/tagger_sp/joint_heads.py mi             KSG mutual information on the test tracks
  python eval_refitq/tagger_sp/joint_heads.py quality        head quality (test tracks)
  python eval_refitq/tagger_sp/joint_heads.py conifer        the firmware check on one chosen model
  python eval_refitq/tagger_sp/joint_heads.py jets           jet caches per head version (the tagger inputs)

Targets (residual r = reco - truth, the truth = the OT track's TP, "otlink", as the existing heads): d0 and z0
[um], tanL [1e-5], 1/R [1e-6 / cm] with the truth q k / pT(TP) (k = each table's own bending constant, measured
on clean matched training tracks), phi [urad, wrapped] against the TP's phi0 (UNIT). Reco = the refit track (SmartPixels
masks) or the OT track (IIII). HEAD_MODE (plan section 1): "width" heads predict log sigma_hat of their residual
(Gaussian likelihood, Fisher scoring), "error" heads the residual itself (squared error at the residual's own
scale), "both" one of each per parameter. Errors (variant B): clean p - y; width log sigma_hat - log|r - mu|
(floored); error/mean F - r. Features and windows: the existing heads' (trackmva.feature_matrix / windows; every
d0-derived quantity and the z0 value withheld) WITHOUT the truth-only h_nmerged (TRUTH_ONLY); missing values -999.

Events: the heads are fitted on the tagger's training events minus its validation events (common.split_table,
common.val_table, by event identity); lambda is chosen on the validation events; the mutual information and
the head quality are measured on the test events.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PARAMS = ["d0", "z0", "tanl", "rinv", "phi"]
# residual units (multipliers of the nano's): d0, z0 um (as the existing heads); tanL 1e-5; 1/R 1e-6 / cm; phi urad.
# Fine enough that log sigma_hat > 0 always (as the existing heads' 1 um floor), so the gathered no-track
# sentinel -1 (trackmva_feature.SENTINEL) stays outside every width head's range; units shift the losses by
# constants only (information, MI and the coupling are unchanged).
UNIT = {"d0": 1e4, "z0": 1e4, "tanl": 1e5, "rinv": 1e6, "phi": 1e6}
HEAD_MODE = "width"                        # width | error | both (plan section 1: the physicist's choice)
GRID = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
VARIANTS = ("A", "B")
XGB = {"tree_method": "hist", "multi_strategy": "one_output_per_tree", "max_depth": 3, "eta": 0.1, "nthread": 4,
       "seed": 0}
ROUNDS = 100
M_SUB = 1024                               # coupling subsample per round
KEEP_INFO = 0.90                           # the lambda rule: every head keeps >= 90 % of its lambda = 0 information
WINSOR = 99.5                              # |r| percentile cap (training tracks)
CLEAN_H_FLOOR = 0.01
LOG_FLOOR_FRAC = 1e-3                      # log|r| floored at 1e-3 of the core (68.3 %) width in the errors
IDENTITY_INFO_TOL = 1e-3                   # joint(lambda = 0) vs independent heads: validation information, nats
MISSING = -999.0                           # a missing refit column = the nano's own sentinel, not NaN: firmware has no
                                           # NaN, and XGBoost's learned default direction for NaN has no conifer form
CONIFER_TOL = 1e-5                         # conifer C++ (float) vs the booster, absolute, output margin


def jdir(*p):
    d = os.path.join(C.OUT, "joint_heads", *p)
    os.makedirs(d if not os.path.splitext(d)[1] else os.path.dirname(d), exist_ok=True)
    return d


def head_list(mode=HEAD_MODE):
    """[(name, kind, param)]: kind logistic | width | mean."""
    if mode == "width":
        return [("clean", "logistic", None)] + [(p, "width", p) for p in PARAMS]
    if mode == "error":
        return [("clean", "logistic", None)] + [(p, "mean", p) for p in PARAMS]
    if mode == "both":
        return [("clean", "logistic", None)] + [h for p in PARAMS for h in ((p + "_mean", "mean", p),
                                                                           (p + "_width", "width", p))]
    raise ValueError(mode)


# ---------------------------------------------------------------- data
EXTRA_BASE = ["pt", "rInv", "phi", "tp_tanL", "tp_phi0", "tp_charge"]
EXTRA_REFIT = ["rInv", "phi"]


def coll_of(mask):
    return "base" if mask == C.BASELINE_MASK else "refit"


def extra_path(sample, mask, n):
    return os.path.join(jdir("cache"), C.tag(sample, mask, n) + "_extra.npz")


def build_extra(sample, mask, n, fn):
    """The columns the existing track cache lacks (truth tanL / phi0 / charge; reco 1/R and phi), in the
    cache's track order (entry order, then the table's order), checked against the cache's OT pT."""
    import awkward as ak
    import uproot

    out = extra_path(sample, mask, n)
    if os.path.exists(out):
        return out
    t = uproot.open(fn)["Events"]
    cols = {f"b_{c}": f"L1TExtTrack_{c}" for c in EXTRA_BASE}
    if coll_of(mask) == "refit":
        cols.update({f"r_{c}": f"{C.TRACK_TABLE}_{c}" for c in EXTRA_REFIT})
    A = {k: ak.to_numpy(ak.flatten(t[b].array())) for k, b in cols.items()}
    tc = np.load(os.path.join(C.cache_dir("tracks"), C.tag(sample, mask, n) + ".npz"))
    if not np.array_equal(A["b_pt"], tc["T_b_pt"].astype(A["b_pt"].dtype)):
        raise RuntimeError(f"{fn}: the extra columns are not aligned with the track cache")
    np.savez_compressed(out, **A)
    return out


def load_mask_tracks(mask):
    """Concatenated track cache + extra columns of the mask's built files, with per-track split flags
    (test / validation, by event identity, as the tagger) and the event key."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build as B

    man = json.load(open(C.manifest_path(mask)))
    parts = []
    for si, s in enumerate(man["samples"]):
        fs = [(n, p) for s2, n, p in man["files"] if s2 == s]
        loaded = {n: B.load_tracks(os.path.join(C.cache_dir("tracks"), C.tag(s, mask, n) + ".npz")) for n, _ in fs}
        evids = {n: loaded[n][2] for n, _ in fs}
        test = C.split_table(s, evids)
        val = C.val_table(s, evids, test)
        for n, p in fs:
            T, ntrk, evid = loaded[n]
            E = dict(np.load(build_extra(s, mask, n, p)))
            ent = np.repeat(np.arange(len(ntrk)), ntrk)
            T.update(E)
            T["_test"], T["_val"] = test[n][ent], val[n][ent]
            T["_key"] = C.event_key(si, n, evid[ent])
            parts.append(T)
    keys = set.intersection(*(set(p) for p in parts))
    return {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}


def targets(T, coll, k_bend=None):
    """(y_clean, R (n, 5) residuals in the plan's units, NaN when unmatched, k_bend used)."""
    from ngtagger.train import trackmva as tm

    matched, clean, rd0, rz0 = tm.truth(T, coll, C.REFIT_TRUTH)
    p = "r_" if coll == "refit" else "b_"
    q, tpt = T["b_tp_charge"].astype(float), np.where(T["b_tp_pt"] > 0, T["b_tp_pt"], np.nan)
    if k_bend is None:
        tr = matched & clean & ~T["_test"] & ~T["_val"] & (q != 0)
        k_bend = float(np.median(T[p + "rInv"][tr] * tpt[tr] / q[tr]))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    R = np.column_stack([
        rd0, rz0,
        np.where(matched, T[p + "tanL"] - T["b_tp_tanL"], np.nan) * UNIT["tanl"],
        np.where(matched, T[p + "rInv"] - q * k_bend / tpt, np.nan) * UNIT["rinv"],
        np.where(matched, wrap(T[p + "phi"] - T["b_tp_phi0"]), np.nan) * UNIT["phi"]])
    return clean.astype(float), R, k_bend


# ---------------------------------------------------------------- the objective
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Problem:
    """One window's training problem: targets, winsorised residuals, the heads and their constants."""

    def __init__(self, y, R, train, mode=HEAD_MODE):
        self.heads = head_list(mode)
        self.y = y
        self.cap = np.nanpercentile(np.abs(R[train]), WINSOR, axis=0)
        self.R = np.clip(R, -self.cap, self.cap)
        self.matched = np.isfinite(R[:, 0])
        self.core = np.nanpercentile(np.abs(R[train]), 68.3, axis=0)
        self.scale = np.nanstd(self.R[train], axis=0)            # the error heads' fixed loss scale
        self.pidx = {p: i for i, p in enumerate(PARAMS)}
        # mode "both": a width head measures r - mu, mu = its parameter's mean head
        self.mu_head = {pp: j for j, (_, kd, pp) in enumerate(self.heads) if kd == "mean"} if mode == "both" else {}
        ytr, Rtr = y[train], self.R[train]
        base = []
        for name, kind, prm in self.heads:
            if kind == "logistic":
                base.append(np.log(ytr.mean() / (1 - ytr.mean())))
            elif kind == "width":
                base.append(np.log(np.nanstd(Rtr[:, self.pidx[prm]])))
            else:
                base.append(float(np.nanmean(Rtr[:, self.pidx[prm]])))
        self.base = np.array(base)

    def resid(self, F, idx, prm):
        """r - mu for a width head (mu = its parameter's mean head in mode "both", else 0)."""
        r = self.R[idx, self.pidx[prm]]
        j = self.mu_head.get(prm)
        return r if j is None else r - F[:, j]

    def task_grad(self, F, idx):
        n, K = F.shape
        g, h = np.zeros_like(F), np.zeros_like(F)
        mt = self.matched[idx]
        for k, (name, kind, prm) in enumerate(self.heads):
            if kind == "logistic":
                p = sigmoid(F[:, k])
                g[:, k], h[:, k] = p - self.y[idx], np.maximum(p * (1 - p), CLEAN_H_FLOOR)
            elif kind == "width":
                d = np.where(mt, self.resid(F, idx, prm), 0.0)
                e2 = np.exp(-2 * F[:, k])
                g[:, k] = np.where(mt, 1 - d * d * e2, 0.0)
                h[:, k] = np.where(mt, 2.0, 1e-6)                       # Fisher scoring
            else:
                r = np.where(mt, self.R[idx, self.pidx[prm]], 0.0)
                wk = [j for j, (_, kd, pp) in enumerate(self.heads) if kd == "width" and pp == prm]
                inv = np.exp(-2 * F[:, wk[0]]) if wk else np.full(n, 1 / self.scale[self.pidx[prm]] ** 2)
                g[:, k] = np.where(mt, (F[:, k] - r) * inv, 0.0)
                h[:, k] = np.where(mt, inv, 1e-6)
        return g, h

    def errors(self, F, idx):
        """Variant B errors and dE/dF (per head)."""
        E, dE = np.empty_like(F), np.ones_like(F)
        for k, (name, kind, prm) in enumerate(self.heads):
            if kind == "logistic":
                p = sigmoid(F[:, k])
                E[:, k], dE[:, k] = p - self.y[idx], p * (1 - p)
            elif kind == "width":
                d = np.abs(self.resid(F, idx, prm))
                E[:, k] = F[:, k] - np.maximum(np.log(d + 1e-30), np.log(LOG_FLOOR_FRAC * self.core[self.pidx[prm]]))
            else:
                E[:, k] = F[:, k] - self.R[idx, self.pidx[prm]]
        return E, dE

    def track_losses(self, F, idx):
        """(n, K) per-track loss of every head on tracks idx (NaN for the regressions of unmatched tracks)."""
        out = np.full(F.shape, np.nan)
        mt = self.matched[idx]
        for k, (name, kind, prm) in enumerate(self.heads):
            if kind == "logistic":
                p = np.clip(sigmoid(F[:, k]), 1e-12, 1 - 1e-12)
                yy = self.y[idx]
                out[:, k] = -(yy * np.log(p) + (1 - yy) * np.log(1 - p))
            elif kind == "width":
                d = self.resid(F, idx, prm)
                out[mt, k] = (F[:, k] + 0.5 * d * d * np.exp(-2 * F[:, k]))[mt]
            else:
                r = self.R[idx, self.pidx[prm]]
                wk = [j for j, (_, kd, pp) in enumerate(self.heads) if kd == "width" and pp == prm]
                inv = np.exp(-2 * F[:, wk[0]]) if wk else 1 / self.scale[self.pidx[prm]] ** 2
                out[mt, k] = (0.5 * (F[:, k] - r) ** 2 * inv)[mt]
        return out

    def losses(self, F, idx):
        """Per-head mean loss on tracks idx (matched tracks for the regressions)."""
        return np.nanmean(self.track_losses(F, idx), axis=0)

    def info(self, F, idx):
        """Information each head carries about its target on tracks idx: the loss of the constant (base)
        prediction minus the head's loss, nats per track."""
        return self.losses(np.tile(self.base, (len(idx), 1)), idx) - self.losses(F, idx)


def dcor2_sum(U):
    """Sum over column pairs of the squared distance correlation of U (m, K), and its gradient."""
    m, K = U.shape
    S, A, V = [], [], []
    for k in range(K):
        d = U[:, k][:, None] - U[:, k][None, :]
        a = np.abs(d)
        Ak = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
        S.append(np.sign(d))
        A.append(Ak)
        V.append((Ak * Ak).mean())
    tot, G = 0.0, np.zeros_like(U)
    for j in range(K):
        for k in range(j + 1, K):
            if V[j] <= 0 or V[k] <= 0:
                continue
            c = (A[j] * A[k]).mean()
            den = np.sqrt(V[j] * V[k])
            tot += c / den
            for p, q in ((j, k), (k, j)):
                dC = 2 / m ** 2 * (S[p] * A[q]).sum(1)
                dV = 4 / m ** 2 * (S[p] * A[p]).sum(1)
                G[:, p] += (dC - 0.5 * c * dV / V[p]) / den
    return tot, G


def objective(P: Problem, idx, variant, lam, seed=0):
    rs = np.random.default_rng(seed)
    pool = np.flatnonzero(P.matched[idx])
    K = len(P.heads)

    def obj(pred, _d):
        F = pred.reshape(len(idx), K)
        g, h = P.task_grad(F, idx)
        if lam > 0:
            sub = rs.choice(pool, size=min(M_SUB, len(pool)), replace=False)
            if variant == "A":
                U, dU = F[sub], np.ones((len(sub), K))
            else:
                U, dU = P.errors(F[sub], idx[sub])
            _, G = dcor2_sum(U)
            scale = lam * len(pool)
            g[sub] += scale * G * dU
            h[sub] += scale * 2.0 / len(sub)                   # damping of the coupled tracks
        return g, h
    return obj


def with_base(b, base):
    """The booster with its base score set to the heads' base margins, so that its own prediction is the trained
    function (base margin + trees). Training passes the base as base_margin, which replaces XGBoost's base score;
    left at its default (0.5) the saved model would predict 0.5 too high on every head."""
    import xgboost as xgb

    m = json.loads(b.save_raw("json"))
    lp = m["learner"]["learner_model_param"]
    base = np.atleast_1d(np.asarray(base, float))
    lp["base_score"] = "[" + ",".join(repr(float(x)) for x in base) + "]" if len(base) > 1 else repr(float(base[0]))
    out = xgb.Booster()
    out.load_model(bytearray(json.dumps(m).encode()))
    out.set_param({"max_depth": XGB["max_depth"]})     # a reloaded model forgets it (conifer reads it as metadata)
    return out


def fit(X, P: Problem, idx, variant, lam, rounds=ROUNDS):
    import xgboost as xgb

    K = len(P.heads)
    dm = xgb.DMatrix(X[idx], label=np.zeros((len(idx), K)), base_margin=np.tile(P.base, (len(idx), 1)))
    b = xgb.train(dict(XGB, num_target=K), dm, num_boost_round=rounds, obj=objective(P, idx, variant, lam))
    return with_base(b, P.base)


def fit_independent(X, P: Problem, idx, rounds=ROUNDS):
    """The six heads as separate single-output models (the identity reference of lambda = 0)."""
    import xgboost as xgb

    K = len(P.heads)
    full = objective(P, idx, "A", 0.0)
    out = []
    for k in range(K):
        def objk(pred, d, k=k):
            Fk = np.tile(P.base, (len(idx), 1))
            Fk[:, k] = pred
            g, h = full(Fk.ravel(), d)
            return g[:, k], h[:, k]
        dm = xgb.DMatrix(X[idx], label=np.zeros(len(idx)), base_margin=np.full(len(idx), P.base[k]))
        bk = xgb.train({k2: v for k2, v in XGB.items() if k2 != "multi_strategy"}, dm, num_boost_round=rounds, obj=objk)
        out.append(with_base(bk, P.base[k]))
    return out


def predict(b, X):
    """The heads' outputs (margins: logit of P(clean), log sigma_hat) of a booster from fit / fit_independent."""
    import xgboost as xgb

    return b.predict(xgb.DMatrix(X), output_margin=True)


def split_booster(b, k):
    """Output k of a one_output_per_tree multi-target booster as a single-output booster (for conifer)."""
    import xgboost as xgb

    m = json.loads(b.save_raw("json"))
    g = m["learner"]["gradient_booster"]["model"]
    keep = [i for i, t in enumerate(g["tree_info"]) if t == k]
    g["trees"] = [dict(g["trees"][i], id=j) for j, i in enumerate(keep)]
    g["tree_info"] = [0] * len(keep)
    g["iteration_indptr"] = list(range(len(keep) + 1))
    g["gbtree_model_param"]["num_trees"] = str(len(keep))
    lp = m["learner"]["learner_model_param"]
    lp["num_target"] = "1"
    bs = lp["base_score"]
    lp["base_score"] = str(json.loads(bs)[k]) if bs.startswith("[") else bs
    out = xgb.Booster()
    out.load_model(bytearray(json.dumps(m).encode()))
    out.set_param({"max_depth": XGB["max_depth"]})
    return out


# ---------------------------------------------------------------- mutual information (KSG)
def ksg_pair(u, v, k=3, seed=0):
    from sklearn.feature_selection import mutual_info_regression

    return float(mutual_info_regression(u[:, None], v, n_neighbors=k, random_state=seed)[0])


def ksg_binary(u, y, k=3, seed=0):
    from sklearn.feature_selection import mutual_info_classif

    return float(mutual_info_classif(u[:, None], y.astype(int), n_neighbors=k, random_state=seed)[0])


def mi_summary(F, E, y, R, P: Problem):
    """{"outputs": {pair: MI}, "errors": {pair: MI}, "own": {head: MI(output; target)}} on these tracks."""
    names = [h[0] for h in P.heads]
    K = len(names)
    out = {"outputs": {}, "errors": {}, "own": {}}
    for j in range(K):
        for k in range(j + 1, K):
            out["outputs"][f"{names[j]}|{names[k]}"] = ksg_pair(F[:, j], F[:, k])
            out["errors"][f"{names[j]}|{names[k]}"] = ksg_pair(E[:, j], E[:, k])
    for k, (name, kind, prm) in enumerate(P.heads):
        out["own"][name] = ksg_binary(F[:, k], y) if kind == "logistic" else ksg_pair(F[:, k], R[:, P.pidx[prm]])
    out["sum_outputs"] = sum(out["outputs"].values())
    out["sum_errors"] = sum(out["errors"].values())
    return out


# ---------------------------------------------------------------- stages
def settings():
    """[(variant, lambda)]: the uncoupled reference once, then every nonzero lambda of each variant."""
    return [("ref", 0.0)] + [(v, lam) for v in VARIANTS for lam in GRID if lam > 0]


def sname(v, lam):
    return f"{v}_{lam:g}"


def mask_problem(mask):
    """(T, X, y, R, k_bend, {window: (train, val, test) index arrays})."""
    from ngtagger.train import trackmva as tm

    coll = coll_of(mask)
    T = load_mask_tracks(mask)
    y, R, k = targets(T, coll)
    X = features(T, coll)
    W = {}
    for wname, w in tm.windows(T, coll).items():
        W[wname] = (np.flatnonzero(w & ~T["_test"] & ~T["_val"]), np.flatnonzero(w & T["_val"]),
                    np.flatnonzero(w & T["_test"]))
    return T, X, y, R, k, W


# TRUTH-ONLY inputs of the existing heads' feature list, dropped here: h_nmerged counts accepted IT hits whose
# cluster has a second simulated contributor above 10 % of its charge (the producer's clusterMerged, "TRUTH-ONLY";
# SmartPixelsRecHitProducer qSecond / qTot > clusterMergeFrac). It cannot exist online.
TRUTH_ONLY = ("h_nmerged",)


def feature_names(T, coll):
    from ngtagger.train import trackmva as tm

    return [n for n in tm.features(T, coll) if n not in TRUTH_ONLY]


def features(T, coll):
    """The existing heads' feature matrix without the truth-only inputs, with missing values at MISSING."""
    from ngtagger.train import trackmva as tm

    X, names = tm.feature_matrix(T, coll)
    keep = [i for i, n in enumerate(names) if n not in TRUTH_ONLY]
    return np.where(np.isnan(X[:, keep]), MISSING, X[:, keep])


def stage_extra(masks):
    for mask in masks:
        man = json.load(open(C.manifest_path(mask)))
        for s, n, p in man["files"]:
            build_extra(s, mask, n, p)
        print(f"extra {mask}: {len(man['files'])} files", flush=True)


def stage_fit(masks):
    """The lambda scan per mask and window: models, validation information, validation and test outputs; the
    identity check of lambda = 0 against six independent heads."""
    for mask in masks:
        T, X, y, R, k, W = mask_problem(mask)
        for wname, (tr, va, te) in W.items():
            P = Problem(y, R, tr)
            d = jdir("fits", mask, wname)
            meta = {"mask": mask, "window": wname, "coll": coll_of(mask), "head_mode": HEAD_MODE,
                    "heads": P.heads, "k_bend": k, "base": P.base.tolist(), "cap": P.cap.tolist(),
                    "core": P.core.tolist(), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                    "xgb": XGB, "rounds": ROUNDS, "m_sub": M_SUB}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            for v, lam in settings():
                out = os.path.join(d, sname(v, lam) + ".npz")
                if os.path.exists(out):
                    continue
                t0 = time.time()
                b = fit(X, P, tr, "A" if v == "ref" else v, lam)
                b.save_model(os.path.join(d, sname(v, lam) + ".json"))
                Fv, Ft = predict(b, X[va]), predict(b, X[te])
                np.savez_compressed(out, F_val=Fv.astype(np.float32), F_test=Ft.astype(np.float32),
                                    info_val=P.info(Fv, va), loss_val=P.losses(Fv, va), fit_s=time.time() - t0,
                                    tl_val=P.track_losses(Fv, va).astype(np.float32),
                                    tl_const=P.track_losses(np.tile(P.base, (len(va), 1)), va).astype(np.float32))
                print(f"fit {mask} {wname} {sname(v, lam)}: {time.time() - t0:.0f}s info "
                      f"{np.round(P.info(Fv, va), 4)}", flush=True)
            ip = os.path.join(d, "identity.json")
            if not os.path.exists(ip):
                Fj = np.load(os.path.join(d, sname("ref", 0.0) + ".npz"))["F_val"]
                ind = fit_independent(X, P, tr)
                Fi = np.column_stack([predict(bk, X[va]) for bk in ind])
                dI = np.abs(P.info(Fj.astype(float), va) - P.info(Fi, va))
                frac = (np.abs(Fj - Fi) > 1e-4).mean(0)
                rec = {"max_info_diff": float(dI.max()), "frac_tracks_diff_gt_1e-4": frac.tolist(),
                       "pass": bool(dI.max() <= IDENTITY_INFO_TOL)}         # the fraction is reported only
                json.dump(rec, open(ip, "w"), indent=1)
                print(f"identity {mask} {wname}: {rec}", flush=True)


def stage_choose(masks):
    """lambda per variant: the largest lambda of the grid at which every head of every mask passes, on the
    validation tracks of the mask's windows pooled: the paired drop of information D = mean(loss_lambda -
    loss_0) <= max(0.1 G(0), 0.01 nats/track) + 2 SE(D), G(0) = mean(loss_constant - loss_0)."""
    rec = {"rule": "paired drop D <= max(0.1 G(0), 0.01 nats/track) + 2 SE(D), every head and mask, the mask's "
                   "windows pooled (validation tracks)", "per_mask": {}, "choice": {}, "identity": {}}
    ok = {v: {lam: True for lam in GRID if lam > 0} for v in VARIANTS}
    for mask in masks:
        wins = sorted(os.listdir(jdir("fits", mask)))
        ld = lambda s_: {w: np.load(os.path.join(jdir("fits", mask, w), s_ + ".npz")) for w in wins}  # noqa: E731
        ref = ld(sname("ref", 0.0))
        L0 = np.concatenate([ref[w]["tl_val"] for w in wins]).astype(float)
        Lc = np.concatenate([ref[w]["tl_const"] for w in wins]).astype(float)
        G0 = np.nanmean(Lc - L0, axis=0)
        for w in wins:
            rec["identity"][f"{mask}/{w}"] = json.load(open(os.path.join(jdir("fits", mask, w), "identity.json")))
        for v in VARIANTS:
            for lam in [x for x in GRID if x > 0]:
                Ll = ld(sname(v, lam))
                d = np.concatenate([Ll[w]["tl_val"] for w in wins]).astype(float) - L0
                D = np.nanmean(d, axis=0)
                se = np.nanstd(d, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(d), axis=0))
                tol = np.maximum(0.1 * G0, 0.01) + 2 * se
                passed = bool(np.all(D <= tol))
                rec["per_mask"][f"{mask}/{sname(v, lam)}"] = {"G0": G0.tolist(), "drop": D.tolist(), "se": se.tolist(),
                                                             "tol": tol.tolist(), "pass": passed}
                ok[v][lam] &= passed
    for v in VARIANTS:
        good = [lam for lam, p in ok[v].items() if p]
        rec["choice"][v] = {"lambda": max(good) if good else min(ok[v]), "qualified": bool(good),
                            "flag": None if good else "no nonzero lambda qualifies: the smallest is used"}
    rec["identity_all_pass"] = all(x["pass"] for x in rec["identity"].values())
    json.dump(rec, open(jdir("choice.json"), "w"), indent=1)
    print(json.dumps(rec["choice"]), "identity all pass:", rec["identity_all_pass"], flush=True)
    return rec


def chosen():
    c = json.load(open(jdir("choice.json")))["choice"]
    return {"ref": ("ref", 0.0), "A": ("A", c["A"]["lambda"]), "B": ("B", c["B"]["lambda"])}


def stage_mi(masks, n_scan=10000, n_full=20000, n_half=10):
    """KSG MI on test tracks (matched): every setting at n_scan; the reference and the chosen settings at
    n_full with errors from n_half event-level half-samples (SD / sqrt 2)."""
    out_p = jdir("mi.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    sel = {sname(*x) for x in chosen().values()}
    for mask in masks:
        T, X, y, R, k, W = mask_problem(mask)
        for wname, (tr, va, te) in W.items():
            P = Problem(y, R, tr)
            d = jdir("fits", mask, wname)
            mt = np.flatnonzero(P.matched[te])
            rng = np.random.default_rng([7, len(mask), len(wname)])
            for v, lam in settings():
                key = f"{mask}/{wname}/{sname(v, lam)}"
                if key in res:
                    continue
                F = np.load(os.path.join(d, sname(v, lam) + ".npz"))["F_test"].astype(float)[mt]
                idx = te[mt]
                E, _ = P.errors(F, idx)
                full = sname(v, lam) in sel
                n = min(n_full if full else n_scan, len(idx))
                pick = np.sort(rng.choice(len(idx), size=n, replace=False))
                r = {"n": int(n), "est": mi_summary(F[pick], E[pick], y[idx[pick]], R[idx[pick]], P)}
                if full:
                    keys = T["_key"][idx[pick]]
                    ev = np.unique(keys)
                    halves = []
                    for h in range(n_half):
                        e1 = set(rng.choice(ev, size=len(ev) // 2, replace=False).tolist())
                        m1 = np.array([kk in e1 for kk in keys])
                        for m_ in (m1, ~m1):
                            q = pick[m_]
                            halves.append(mi_summary(F[q], E[q], y[idx[q]], R[idx[q]], P))
                    r["half_sd_over_sqrt2"] = {
                        g: {kk: float(np.std([hh[g][kk] for hh in halves], ddof=1) / np.sqrt(2)) for kk in halves[0][g]}
                        for g in ("outputs", "errors", "own")}
                    r["half_sd_over_sqrt2"]["sum_outputs"] = float(np.std([hh["sum_outputs"] for hh in halves], ddof=1)
                                                                   / np.sqrt(2))
                    r["half_sd_over_sqrt2"]["sum_errors"] = float(np.std([hh["sum_errors"] for hh in halves], ddof=1)
                                                                  / np.sqrt(2))
                res[key] = r
                json.dump(res, open(out_p, "w"), indent=1)
                print(f"mi {key}: n {n} sum outputs {r['est']['sum_outputs']:.3f} errors {r['est']['sum_errors']:.3f}",
                      flush=True)


def stage_quality(masks):
    """Head quality on the test tracks for the reference and the chosen settings: width heads' coverage of
    |r - mu| < sigma_hat and the 68 % width of the pull; the clean head's AUC; error heads' 68 % |F - r|."""
    from sklearn.metrics import roc_auc_score

    res = {}
    for mask in masks:
        T, X, y, R, k, W = mask_problem(mask)
        for wname, (tr, va, te) in W.items():
            P = Problem(y, R, tr)
            d = jdir("fits", mask, wname)
            for ver, (v, lam) in chosen().items():
                F = np.load(os.path.join(d, sname(v, lam) + ".npz"))["F_test"].astype(float)
                mt = P.matched[te]
                q = {}
                for j, (name, kind, prm) in enumerate(P.heads):
                    if kind == "logistic":
                        q[name] = {"auc": float(roc_auc_score(y[te], F[:, j]))}
                    elif kind == "width":
                        dlt = np.abs(P.resid(F, te, prm))[mt]
                        pull = dlt / np.exp(F[mt, j])
                        q[name] = {"coverage_1sigma": float(np.mean(pull < 1)),
                                   "pull_68": float(np.percentile(pull, 68.3))}
                    else:
                        q[name] = {"resid_68": float(np.percentile(np.abs(F[mt, j] - R[te, P.pidx[prm]][mt]), 68.3))}
                res[f"{mask}/{wname}/{ver}"] = q
    json.dump(res, open(jdir("quality.json"), "w"), indent=1)


def stage_conifer(mask="AAAA", window="prompt", n=10000):
    """Firmware check: the chosen B model of one window split into single-output boosters, converted with
    conifer (C++ backend, float) and compared with the booster on n tracks."""
    import conifer
    import conifer.backends.cpp.writer as W
    import xgboost as xgb

    W._json_include = lambda: f"-I{os.environ.get('JSON_ROOT', '/opt/homebrew/include')}"
    v, lam = chosen()["B"]
    d = jdir("fits", mask, window)
    b = xgb.Booster()
    b.load_model(os.path.join(d, sname(v, lam) + ".json"))
    T, X, y, R, k, Wn = mask_problem(mask)
    te = Wn[window][2][:n]
    full = b.predict(xgb.DMatrix(X[te]), output_margin=True)
    rec = {"model": f"{mask}/{window}/{sname(v, lam)}", "n_tracks": int(len(te)), "heads": {}}
    for j in range(full.shape[1]):
        bj = split_booster(b, j)
        cfg = conifer.backends.cpp.auto_config()
        cfg["OutputDir"] = jdir("conifer", f"head{j}")
        cfg["Precision"] = "float"
        cm = conifer.converters.convert_from_xgboost(bj, cfg)
        cm.compile()
        pc = cm.decision_function(X[te].astype(np.float32)).ravel()
        ps = bj.predict(xgb.DMatrix(X[te]), output_margin=True)
        rec["heads"][j] = {"n_trees": len(cm.trees), "max_abs_split_vs_full": float(np.abs(ps - full[:, j]).max()),
                           "max_abs_conifer_vs_booster": float(np.abs(pc - ps).max())}
    rec["pass"] = all(h["max_abs_conifer_vs_booster"] <= CONIFER_TOL and h["max_abs_split_vs_full"] == 0
                      for h in rec["heads"].values())
    json.dump(rec, open(jdir("conifer.json"), "w"), indent=1)
    print(json.dumps(rec), flush=True)


def joint_mva(mask, version):
    """The chosen heads of one head version (ref, A, B) for one mask, as a trackmva.TrackMVA, so that
    nano.load_jets' "trackjoint" group gathers them like the existing heads: clean as P(clean), the
    regressions as their raw outputs (log sigma_hat for width heads)."""
    import xgboost as xgb

    from ngtagger.train import trackmva as tm

    class JointMVA(tm.TrackMVA):
        def predict(self, T, coll="refit", quantiles=False):
            X = features(T, coll)
            out = np.full((len(X), len(self.outputs)), np.nan)
            for wname, w in tm.windows(T, coll).items():
                idx = np.flatnonzero(w)
                if len(idx):
                    out[idx] = predict(self.models[coll][wname], X[idx])
            for k, (_, kind, _) in enumerate(self.heads):
                if kind == "logistic":
                    out[:, k] = sigmoid(out[:, k])
            return {h: out[:, k] for k, h in enumerate(self.outputs)}

    v, lam = chosen()[version]
    coll = coll_of(mask)
    models, heads = {}, None
    for wname in tm.window_names(coll):
        d = jdir("fits", mask, wname)
        b = xgb.Booster()
        b.load_model(os.path.join(d, sname(v, lam) + ".json"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        models[wname] = b
        heads = [tuple(h) for h in meta["heads"]]
    m = JointMVA({coll: models}, {"family": C.FAMILY, "config": "", "tables": tm.table_names(C.FAMILY, ""),
                                  "setting": sname(v, lam)})
    m.heads, m.outputs = heads, [h[0] for h in heads]
    return m


def stage_jets(masks):
    """Per head version and mask: jet caches with the "track" inputs + the six heads (trackjoint), in
    common.jets_dir("joint_<version>"), the main caches' test split, keyed by the setting."""
    import build as B

    for version in C.JOINT_VERSIONS:
        for mask in masks:
            mva = joint_mva(mask, version)
            man = json.load(open(C.manifest_path(mask)))
            for s, n, p in man["files"]:
                main = np.load(os.path.join(C.cache_dir("jets"), C.tag(s, mask, n) + ".npz"))
                B.build_jets(s, mask, n, p, None, f"{version}:{mva.meta['setting']}", main["test_entries"],
                             groups=C.VARIANTS[f"joint_{version}"], jets_dir=C.jets_dir(f"joint_{version}"),
                             models={"trackjoint_model": mva})
            print(f"jets joint_{version} {mask}: {len(man['files'])} files ({mva.meta['setting']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extra", "fit", "choose", "mi", "quality", "conifer", "jets"])
    ap.add_argument("--masks", nargs="+", default=C.MASKS)
    a = ap.parse_args()
    {"extra": stage_extra, "fit": stage_fit, "choose": stage_choose, "mi": stage_mi, "quality": stage_quality,
     "jets": stage_jets}.get(a.stage, lambda m: stage_conifer())(a.masks)


if __name__ == "__main__":
    main()
