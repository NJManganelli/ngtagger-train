"""FAST class-C study: shared pieces (paths, test-event loader, per-cluster hash uniforms, response, metrics, leak probe).

The response and metrics replicate v2_emu_angle.py (the approved v4 angle-space prototype) with an EXPLICIT rng so every variant
sees the same dump-row picks: out = pseudo + bias + 1.5 (r - bias) from the 32 nearest dump rows in (alpha, beta) [deg]
(discs: alpha key = atan(cotA - 0.312)), saturated at +-(90 - 0.01).
"""
import os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from v2_common import V2, HERE, D_ALL, deg, core_fit

FAST = "/Volumes/WDMac/smartpixels_scratch/emu_fast"
PY = os.path.expanduser("~/smartpixels/ngtagger-train/.pixi/envs/default/bin/python")
RES = os.path.join(HERE, "results"); FIGS = os.path.join(HERE, "figs")
K = 1.5; C_LOR = -0.312; KNN_D = 32; EPS = 0.01
SALTS = [0xA2A2A2A2A2A2A201, 0xA2A2A2A2A2A2A202, 0xA2A2A2A2A2A2A203, 0xA2A2A2A2A2A2A204]
sat = lambda a: np.clip(a, -(90 - EPS), 90 - EPS)
tocot = lambda a: np.tan(np.radians(a))
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_u(run, event, detId, idx, salt):
    """uniform in [0, 1) from the stable cluster key -- the emu_common / CMSSW emuHashIndex construction, as a float"""
    with np.errstate(over="ignore"):
        u64 = lambda v: np.asarray(v).astype(np.uint64)
        h = _mix(((u64(detId) << np.uint64(16)) | u64(idx)) + np.uint64(0x9E3779B97F4A7C15))
        h = _mix(h ^ u64(event)); h = _mix(h ^ u64(run)); h = _mix(h ^ np.uint64(salt))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def uniforms(m, n=4):
    return np.stack([hash_u(m.run.values, m.event.values, m.detId.values, m.idxInDet.values, SALTS[i]) for i in range(n)], 1)


def load_test(reg):
    """all clusters of the 50 test events (cluster_cnn_v2 arrays) + run / idxInDet / CPE (for the hash and the probe)"""
    m = pd.read_parquet(os.path.join(V2, "arrays", f"{reg}_test_meta.parquet"))
    x = pd.read_parquet(os.path.join(D_ALL, "meta.parquet"), columns=["row", "run", "idxInDet", "tx_cpe", "ty_cpe", "pitchX", "pitchY"],
                        filters=[("split", "==", 2)]).set_index("row").loc[m.row.values]
    for c in x.columns:
        m[c] = x[c].values
    return m


def classes(m):
    valid = ((m.linkClass == 2) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900)).values
    clsT = valid & (m.hxOk == 1).values & (m.hxDist < 1.0).values
    return valid, clsT


_D = None


def dump():
    global _D
    if _D is None:
        from v2_emulate import load_dump
        D = load_dump(); _D = (D, cKDTree(np.c_[D.tA.values, D.tB.values]))
    return _D


def respond(aA, aB, disc, seed):
    """approved v4 response (angle space). aA, aB pseudo-truth [deg]; returns emitted angles + saturation flags"""
    D, tree = dump(); rng = np.random.default_rng(seed)
    qA = np.where(disc, deg(tocot(aA) + C_LOR), aA)
    _, idx = tree.query(np.c_[qA, aB], k=KNN_D, workers=4)
    pick = idx[np.arange(len(aA)), rng.integers(0, KNN_D, len(aA))]
    out, fl = [], []
    for a, col in ((aA, "rA"), (aB, "rB")):
        r = D[col].values; med = np.median(r[idx], 1); raw = a + med + K * (r[pick] - med)
        fl.append(np.abs(raw) >= 90 - EPS); out.append(sat(raw))
    return out[0], out[1], fl[0], fl[1]


def closure(o, t):
    r = o - t; r = r[np.isfinite(r)]
    return dict(core=core_fit(r)[0], full=float(np.sqrt(np.mean(r ** 2))), gt10=float(np.mean(np.abs(r) > 10)), n=int(len(r)))


def auc(X, y, half, seed=5, grp=None):
    """HGB unlinked-vs-linked classifier trained on one event half, AUC on the other; with grp: {"all": .., group: ..}
    (per-group AUC = the region classifier evaluated on that group's test rows)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    tr = np.where(half)[0]; tr = rng.choice(tr, min(len(tr), 1_000_000), replace=False)
    clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1).fit(X[tr], y[tr])
    p = clf.predict_proba(X[~half])[:, 1]
    if grp is None:
        return float(roc_auc_score(y[~half], p))
    g = grp[~half]; out = {"all": float(roc_auc_score(y[~half], p))}
    for k in np.unique(g):
        out[k] = float(roc_auc_score(y[~half][g == k], p[g == k]))
    return out


def groups(m, reg):
    L = "layer" if reg == "barrel" else "disk"
    surf = m[L].values.astype(int)
    return surf, np.where(reg == "disc", "ID1-4", np.where(surf == 1, "IL1", "IL2-4"))
