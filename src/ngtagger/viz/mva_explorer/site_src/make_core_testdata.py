"""Generate synthetic test fixtures + python reference results validating
explorer_core.js bit-for-bit (grid slicing, binned stats, efficiency, AUC,
histograms).  Consumed by site_src/test_core_jxa.js via the pytest runner
(tests/test_mva_explorer.py::test_js_core_matches_python) or standalone:

    pixi run python src/ngtagger/viz/mva_explorer/site_src/make_core_testdata.py
"""

from __future__ import annotations

import json
import os

import numpy as np

SCALE = 3000


# ---------------------------------------------------------------- grid ref

def _bin_assign(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mirror of explorer_core.binAssign: last edge inclusive, -1 outside."""
    idx = np.digitize(x, edges) - 1
    idx = np.where(x == edges[-1], len(edges) - 2, idx)
    idx = np.where((x >= edges[0]) & (x <= edges[-1]), idx, -1)
    return idx.astype(np.int64)


def _agg(vals, agg):
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mid = float(np.mean(vals)) if agg == "mean" else float(np.median(vals))
    return mid, float(np.percentile(vals, 16)), float(np.percentile(vals, 84))


def grid_reference(values: np.ndarray, centers: list, fixed: dict, x_axis: int,
                   y_axis, cuts: dict, agg: str, invert: bool) -> dict:
    v = 1.0 / values if invert else values
    ndim = v.ndim
    included = []
    for i in range(ndim):
        if i in fixed:
            included.append([fixed[i]])
        elif centers[i] is None:
            included.append(list(range(v.shape[i])))
        else:
            lo, hi = cuts.get(i, (-np.inf, np.inf))
            c = np.asarray(centers[i])
            included.append(list(np.where((c >= lo) & (c <= hi))[0]))
    iy = -1 if (y_axis is None or y_axis == x_axis) else y_axis

    sub = v
    for i, inc in enumerate(included):
        sub = np.take(sub, inc, axis=i)

    def collect(bx, by=None):
        sl = [slice(None)] * ndim
        sl[x_axis] = bx
        if iy >= 0:
            sl[iy] = by
        out = sub[tuple(sl)].ravel()
        return out[np.isfinite(out)]  # mirror explorer_core's gather filter

    def coord(axis, b):
        return float(centers[axis][b]) if centers[axis] is not None else float(b)

    x = [coord(x_axis, b) for b in included[x_axis]]
    if iy < 0:
        mid, lo_, hi_, n = [], [], [], []
        for p in range(len(included[x_axis])):
            vals = collect(p)
            m, l, h = _agg(vals, agg)
            mid.append(m); lo_.append(l); hi_.append(h); n.append(int(vals.size))
        return {"dim": 1, "x": x, "mid": mid, "lo": lo_, "hi": hi_, "n": n}
    y = [coord(iy, b) for b in included[iy]]
    z = []
    for py in range(len(included[iy])):
        z.append([_agg(collect(px, py), agg)[0]
                  for px in range(len(included[x_axis]))])
    return {"dim": 2, "x": x, "y": y, "z": z}


def make_grid_cases(rng) -> list[dict]:
    # (cat=3, real 5 x 4 x 3) int16 log-quantized cube + a signed f32 cube
    shape = (3, 5, 4, 3)
    raw = 10.0 ** rng.normal(0.0, 0.4, size=shape)
    q = np.round(np.log10(raw) * SCALE).astype(np.int16)
    deq = 10.0 ** (q.astype(np.float64) / SCALE)
    centers = [None,
               [1.0, 2.0, 4.0, 8.0, 16.0],
               [0.2, 0.6, 1.0, 1.4],
               [0.05, 0.15, 0.3]]
    f32 = rng.normal(0.0, 1.0, size=shape).astype(np.float32)

    cases = []

    def add(dataset, values, spec):
        ref = grid_reference(values, centers, spec.get("fixedBins", {}),
                             spec["xAxis"], spec.get("yAxis"),
                             spec.get("cuts", {}), spec["agg"],
                             spec.get("invert", False))
        cases.append({"dataset": dataset, "spec": spec, "expected": ref})

    add("i16", deq, {"fixedBins": {0: 1}, "xAxis": 1, "agg": "median"})
    add("i16", deq, {"fixedBins": {0: 2}, "xAxis": 2, "agg": "mean",
                     "cuts": {1: [1.5, 10.0], 3: [0.0, 0.2]}})
    add("i16", deq, {"fixedBins": {0: 0}, "xAxis": 1, "yAxis": 2,
                     "agg": "median", "cuts": {3: [0.1, 0.4]}})
    add("i16", deq, {"fixedBins": {0: 1}, "xAxis": 3, "agg": "mean",
                     "invert": True})
    # ratio num/den across the categorical axis: value = v[2]/v[0]
    ratio = deq[2] / deq[0]
    ref = grid_reference(ratio, centers[1:], {}, 0, None,
                         {1: [0.4, 1.2]}, "median", False)
    cases.append({"dataset": "i16_ratio", "numFixed": 2, "denFixed": 0,
                  "spec": {"xAxis": 0, "agg": "median",
                           "cuts": {1: [0.4, 1.2]}},
                  "expected": ref})
    add("f32", f32.astype(np.float64), {"fixedBins": {0: 1}, "xAxis": 2,
                                        "agg": "mean", "cuts": {3: [0.1, 0.4]}})

    return cases, q, f32, shape, centers


# --------------------------------------------------------------- table ref

def make_table_cases(rng) -> tuple[list[dict], np.ndarray, list[str]]:
    from sklearn.metrics import roc_auc_score

    n = 400
    label = (rng.random(n) < 0.35).astype(np.float32)
    score = np.clip(rng.normal(0.35 + 0.3 * label, 0.2), 0, 1).astype(np.float32)
    # force some exact score ties to exercise average-rank handling
    score[: n // 4] = np.round(score[: n // 4] * 20) / 20
    pt = (2.0 * 10 ** rng.random(n)).astype(np.float32)     # 2..20
    eta = (rng.random(n) * 2.4).astype(np.float32)
    cls = rng.integers(0, 3, size=n).astype(np.float32)      # fake 3-class label
    columns = ["score", "label", "pt", "abs_eta", "cls"]
    rows = np.stack([score, label, pt, eta, cls], axis=1).astype("<f4")

    edges_pt = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    edges_score = np.linspace(0, 1, 11)

    cuts = {"abs_eta": [0.3, 1.8]}
    sel = (eta >= 0.3) & (eta <= 1.8)
    xs = pt[sel].astype(np.float64)
    ss = score[sel].astype(np.float64)
    ls = label[sel].astype(np.float64)

    bins = _bin_assign(xs, edges_pt)
    nb = len(edges_pt) - 1
    centers = (0.5 * (edges_pt[:-1] + edges_pt[1:])).tolist()

    # stats (median)
    stat = {"centers": centers, "mid": [], "lo": [], "hi": [], "n": []}
    for b in range(nb):
        vals = ss[bins == b]
        m, l, h = _agg(vals, "median")
        stat["mid"].append(m); stat["lo"].append(l); stat["hi"].append(h)
        stat["n"].append(int(vals.size))

    # efficiency at cut
    cut = 0.5
    eff = {"centers": centers, "eff": [], "mistag": [], "nPos": [], "nNeg": []}
    for b in range(nb):
        inb = bins == b
        p = inb & (ls == 1)
        m = inb & (ls == 0)
        eff["nPos"].append(int(p.sum())); eff["nNeg"].append(int(m.sum()))
        eff["eff"].append(float((ss[p] > cut).mean()) if p.sum() else float("nan"))
        eff["mistag"].append(float((ss[m] > cut).mean()) if m.sum() else float("nan"))

    # per-bin AUC (sklearn reference)
    auc = {"centers": centers, "auc": [], "nPos": [], "nNeg": []}
    for b in range(nb):
        inb = bins == b
        npos, nneg = int(ls[inb].sum()), int((ls[inb] == 0).sum())
        auc["nPos"].append(npos); auc["nNeg"].append(nneg)
        auc["auc"].append(float(roc_auc_score(ls[inb], ss[inb]))
                          if npos and nneg else float("nan"))

    # global AUC (tie block check)
    auc_all = float(roc_auc_score(ls, ss))

    # score histogram (normalized), positives only
    pos_scores = ss[ls == 1]
    hcounts, _ = np.histogram(pos_scores, bins=edges_score)
    total = int(hcounts.sum())
    width = np.diff(edges_score)
    hist = {"centers": (0.5 * (edges_score[:-1] + edges_score[1:])).tolist(),
            "y": (hcounts / (total * width)).tolist() if total else [0.0] * 10,
            "counts": hcounts.tolist(), "total": total}

    # combineCurves over three shifted curves
    curves = [{"mid": [0.1, 0.2, np.nan]}, {"mid": [0.3, 0.4, 0.5]},
              {"mid": [0.2, np.nan, 0.7]}]
    comb = {"mid": [0.2, 0.3, 0.6], "lo": [0.1, 0.2, 0.5],
            "hi": [0.3, 0.4, 0.7], "n": [3, 2, 2]}

    cases = [{
        "cuts": cuts, "xColumn": "pt", "scoreColumn": "score",
        "labelColumn": "label", "positiveValue": 1, "cut": cut,
        "edgesX": edges_pt.tolist(), "edgesScore": edges_score.tolist(),
        "agg": "median",
        "expected": {"stats": stat, "efficiency": eff, "auc": auc,
                     "aucAll": auc_all, "histogram": hist},
    }]
    # one-vs-rest on the 3-class column: positive = cls==2, mean agg, no cuts
    pos2 = (cls == 2).astype(np.float64)
    auc2 = float(roc_auc_score(pos2, score.astype(np.float64)))
    cases.append({
        "cuts": {}, "xColumn": "abs_eta", "scoreColumn": "score",
        "labelColumn": "cls", "positiveValue": 2, "cut": 0.4,
        "edgesX": np.linspace(0, 2.4, 5).tolist(),
        "edgesScore": edges_score.tolist(), "agg": "mean",
        "expected": {"aucAll": auc2},
    })
    return cases, rows, columns, comb, curves


def generate(out_path: str) -> dict:
    rng = np.random.default_rng(20260721)
    grid_cases, q, f32, shape, centers = make_grid_cases(rng)
    table_cases, rows, columns, comb, comb_in = make_table_cases(rng)

    def _jsonable(o):
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, float) and o != o:
            return None  # NaN -> null; the JS harness maps null back to NaN
        if isinstance(o, (np.floating, np.integer)):
            return _jsonable(o.item())
        return o

    out = {
        "scale": SCALE,
        "grid": {
            "shape": list(shape),
            "centers": [c if c is None else list(c) for c in centers],
            "i16": q.ravel().tolist(),
            "f32": f32.ravel().astype(float).tolist(),
            "cases": _jsonable(grid_cases),
        },
        "table": {
            "columns": columns,
            "rows": rows.ravel().astype(float).tolist(),
            "nRows": int(len(rows)),
            "cases": _jsonable(table_cases),
        },
        "combine": {"input": _jsonable(comb_in), "expected": _jsonable(comb)},
    }
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate(os.path.join(here, "core_testdata.json"))
    print("wrote core_testdata.json")
