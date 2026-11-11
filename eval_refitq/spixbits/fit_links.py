"""Fit the per-link angle ranges and the angle-error tables (PLAN.md section 4b), in-sample on all ten files.

The ranges and tables depend on the clusters only (the cluster table is the same in every scenario), so the AAAA
files 1-10 are read. Target clusters: class T (angle truth), dominant particle charged with pT >= 2 GeV and charge
fraction >= 0.5, prompt or displaced.

Pools. A link with fewer than 8,000 target clusters is pooled with other links of its layer or disc, nearest first:
its +-z mirror (same ring distance from the centre, same facing), then the other facing, then neighbouring rings.
The member's sign per angle is the one whose range edges agree best with the pool's (all four sign pairs are tried;
edge agreement in units of the bootstrap sd of the edges); a +-z mirror is accepted only if the edges then agree
within 2 sd. Ranges: the central 99.5 % of the pool's network-resolution angles. Global range: all links together.

Angle-error table, per pool and angle, for the network-resolution angles ("net") and for today's emulated angles
("today"), in the link's own local frame (no sign flips): intervals of the TRUE cot with at least 400 clusters each (equal counts, at most 12); per interval the
median of (reported - true) (the "mean") and the half-width of its central 68.27 % interval (the "width").
Quantization variance: the measured variance of (decoded - input) on the pool's in-range target clusters per bit
count (2-7), for the pool's range ("link") and for the global range ("global").

  python fit_links.py            -> cache/link_tables.json, cache/link_fit_clusters.npz, checks/angle_widths.json
"""
import json
import sys
import time

import numpy as np

import emulate as EM
import study as ST
from study import AR, CM

MIN_LINK, MIN_INTERVAL, MAX_INTERVALS = 8000, 400, 12
COVER = 0.995
QLO, QHI = (1 - COVER) / 2, 1 - (1 - COVER) / 2
BITS = (2, 3, 4, 5, 6, 7)
NBOOT = 60
RNG = np.random.default_rng(20261111)


def collect(files=ST.ALL_FILES, scen="AAAA", nev=None):
    geo = ST.geometry()
    net = EM.Network(ST.CACHE / "network_residuals.npz")
    links = EM.Links(geo, CM.GEOMETRY)
    keep = ("link", "net_A", "net_B", "true_A", "true_B", "today_A", "today_B", "today_hasA", "today_hasB", "eta",
            "exact_cell")
    parts = []
    stats = {"clusters": 0, "class_T": 0, "target": 0}
    t0 = time.time()
    for f in files:
        path = ST.input_path(scen, f)
        br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]
                                + AR.SPN.cluster_branches(AR.OP._keys(str(path)), list(EM.EXTRA_CLUSTER_COLS),
                                                          disc_scope="all")))
        for start, A in CM.iterate(path, br, chunk=10, nev=nev):
            ctx = AR.prepare(A, geo)
            B = EM.base(ctx, geo, A, net, links)
            m = B["target"] & (B["link"] >= 0)
            stats["clusters"] += len(m)
            stats["class_T"] += int(B["class_T"].sum())
            stats["target"] += int(m.sum())
            parts.append({k: B[k][m] for k in keep})
        print(f"[{time.time() - t0:6.0f}s] file {f}: {stats}", flush=True)
    return CM.concat(parts), stats


def edges_sd(v):
    """(lo, hi) of the central 99.5 % and their bootstrap sd."""
    e = np.quantile(v, [QLO, QHI])
    if len(v) < 40:
        return e, np.array([np.inf, np.inf])
    idx = RNG.integers(0, len(v), (NBOOT, min(len(v), 20000)))
    b = np.quantile(v[idx], [QLO, QHI], axis=1)
    return e, b.std(1) + 1e-12


def agreement(P, Q, sA, sB):
    """Largest |edge difference| / sd over both angles and both edges, for Q's values signed (sA, sB)."""
    z = 0.0
    for a, s in (("A", sA), ("B", sB)):
        eP, sdP = edges_sd(P[a])
        eQ, sdQ = edges_sd(s * Q[a])
        with np.errstate(invalid="ignore"):
            z = max(z, float(np.nanmax(np.abs(eP - eQ) / np.hypot(sdP, sdQ))))
    return z


def attributes(codes):
    surf, ring, zs, fac = codes // 10000, (codes // 100) % 100, (codes // 10) % 10, codes % 10
    rabs = ring.astype(float).copy()
    for S in np.unique(surf[surf <= 4]):                   # barrel: distance of the module from the layer's centre
        m = surf == S
        rabs[m] = np.floor(np.abs(ring[m] - (ring[m].max() + ring[m].min()) / 2.0))
    return surf, rabs, zs, fac


def pool_links(D):
    codes = np.unique(D["link"])
    surf, rabs, zs, fac = attributes(codes)
    vals = {int(c): {"A": D["net_A"][D["link"] == c], "B": D["net_B"][D["link"] == c]} for c in codes}
    pools = [{"links": {int(c): (1.0, 1.0)}, "A": vals[int(c)]["A"], "B": vals[int(c)]["B"], "stuck": False}
             for c in codes]
    attr = {int(c): (surf[i], rabs[i], zs[i], fac[i]) for i, c in enumerate(codes)}
    log = []

    def distance(P, Q):
        best = None
        for a in P["links"]:
            for b in Q["links"]:
                A_, B_ = attr[a], attr[b]
                if A_[0] != B_[0]:
                    continue
                dr = abs(A_[1] - B_[1])
                kind = 0 if (dr == 0 and A_[3] == B_[3]) else (1 if dr == 0 else 2)     # mirror, other facing, ring
                d = (dr, A_[3] != B_[3], A_[2] == B_[2], kind)
                best = d if best is None or d < best else best
        return best

    while True:
        short = [p for p in pools if len(p["A"]) < MIN_LINK and not p["stuck"]]
        if not short:
            break
        P = min(short, key=lambda p: len(p["A"]))
        cands = sorted(((distance(P, Q), i) for i, Q in enumerate(pools) if Q is not P and distance(P, Q) is not None),
                       key=lambda x: x[0])
        merged = False
        for d, i in cands:
            Q = pools[i]
            zbest, sbest = min((agreement(Q, P, sA, sB), (sA, sB)) for sA in (1.0, -1.0) for sB in (1.0, -1.0))
            mirror = d[3] == 0
            if mirror and zbest > 2.0:
                log.append({"rejected_mirror": [EM.Links.name(c) for c in P["links"]],
                            "with": [EM.Links.name(c) for c in Q["links"]], "edge_disagreement_sd": zbest})
                continue
            sA, sB = sbest
            for c, (a_, b_) in P["links"].items():
                Q["links"][c] = (a_ * sA, b_ * sB)
            Q["A"], Q["B"] = np.concatenate([Q["A"], sA * P["A"]]), np.concatenate([Q["B"], sB * P["B"]])
            log.append({"merged": [EM.Links.name(c) for c in P["links"]], "into_n": int(len(Q["A"])),
                        "kind": ("mirror", "other facing", "neighbouring ring")[d[3]], "signs": [sA, sB],
                        "edge_disagreement_sd": zbest})
            pools.remove(P)
            merged = True
            break
        if not merged:
            P["stuck"] = True
    return pools, log


def table(x, r):
    """Intervals of x (equal counts, >= 400 each, at most 12) with the median and the central-68.27 % half-width of r."""
    ok = np.isfinite(x) & np.isfinite(r)
    x, r = x[ok], r[ok]
    if len(x) < 20:
        return {"edges": [-1e9, 1e9], "mean": [0.0], "width": [1.0], "n": [int(len(x))]}
    k = int(np.clip(len(x) // MIN_INTERVAL, 1, MAX_INTERVALS))
    e = np.quantile(x, np.linspace(0, 1, k + 1))
    e = np.unique(e)
    if len(e) < 2:
        e = np.array([e[0], e[0] + 1e-9])
    b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)
    mean, width, n = [], [], []
    for i in range(len(e) - 1):
        ri = r[b == i]
        q = np.quantile(ri, [0.158655, 0.5, 0.841345]) if len(ri) else [0, 0, 2]
        mean.append(float(q[1]))
        width.append(float(max((q[2] - q[0]) / 2, 1e-6)))
        n.append(int(len(ri)))
    return {"edges": [float(v) for v in e], "mean": mean, "width": width, "n": n}


def qvars(v, lo, hi):
    out = {}
    for b in BITS:
        q, f = EM.quantize_angle(v, lo, hi, b)
        m = f == 0
        out[str(b)] = float(np.mean((q[m] - v[m]) ** 2)) if m.any() else float(((hi - lo) / (2 ** b - 2)) ** 2 / 12)
    return out


def main():
    nev = int(sys.argv[1]) if len(sys.argv) > 1 else None
    D, stats = collect(nev=nev)
    tmp = ST.CACHE / "link_fit_clusters.tmp.npz"
    np.savez_compressed(tmp, **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in D.items()})
    tmp.rename(ST.CACHE / "link_fit_clusters.npz")
    pools, log = pool_links(D)
    glob = {"A": [float(v) for v in np.quantile(D["net_A"], [QLO, QHI])],
            "B": [float(v) for v in np.quantile(D["net_B"], [QLO, QHI])]}
    doc = {"what": __doc__.split("\n\n")[0], "frames": "A = SmartPixels alpha = CMS local cot beta (the code's beta); "
           "B = SmartPixels beta = CMS local cot alpha (the code's alpha); pool-frame value = sign x link-frame value",
           "sample": "canon_v5 AAAA files 1-10" + (f", first {nev} events each" if nev else ""), "stats": stats,
           "cover": COVER, "min_link": MIN_LINK, "min_interval": MIN_INTERVAL,
           "width_definition": "half of the central 68.27 % interval; mean = median", "global_range": glob,
           "links": {}, "pools": [], "pooling_log": log}
    members = [{int(c): (1.0, 1.0) for c in np.unique(D["link"])}] + [p["links"] for p in pools]
    for pid, mem in enumerate(members):
        sel = np.isin(D["link"], list(mem))
        sA = np.ones(sel.sum())
        sB = np.ones(sel.sum())
        lk = D["link"][sel]
        for c, (a_, b_) in mem.items():
            sA[lk == c], sB[lk == c] = a_, b_
            if pid > 0:
                doc["links"][str(c)] = {"name": EM.Links.name(c), "pool": pid, "sA": a_, "sB": b_,
                                        "n_target": int((lk == c).sum())}
        P = {"n_target": int(sel.sum()), "members": [EM.Links.name(c) for c in mem] if pid else ["(all links: fallback)"],
             "range": {}, "table": {"net": {}, "today": {}}, "qvar": {"link": {}, "global": {}}}
        for a, s in (("A", sA), ("B", sB)):
            v = s * D["net_" + a][sel]
            tr = s * D["true_" + a][sel]
            lo, hi = (float(q) for q in np.quantile(v, [QLO, QHI]))
            P["range"][a] = [lo, hi]
            # the tables in the link's own local frame, no sign flips (the network's bias has an even part that a
            # flip would turn; code review LOW): the pool's signs apply to the ranges and the quantization only
            trl, vl, td = D["true_" + a][sel], D["net_" + a][sel], D["today_" + a][sel]
            P["table"]["net"][a] = table(trl, vl - trl)
            ok = D["today_has" + a][sel] & (np.abs(td) <= EM.MEAS_ANGLE_MAX)
            P["table"]["today"][a] = table(trl[ok], (td - trl)[ok])
            P["qvar"]["link"][a] = qvars(v, lo, hi)
            P["qvar"]["global"][a] = qvars(s * v, glob[a][0], glob[a][1])
            P.setdefault("beyond_each_edge", {})[a] = [int((v < lo).sum()), int((v >= hi).sum())]
        doc["pools"].append(P)
    tmpj = ST.CACHE / "link_tables.tmp.json"
    json.dump(doc, open(tmpj, "w"), indent=1)
    tmpj.rename(ST.CACHE / "link_tables.json")
    n = [p["n_target"] for p in doc["pools"][1:]]
    print(json.dumps({"links": len(doc["links"]), "pools": len(n), "pools_below_8000": int(sum(x < MIN_LINK for x in n)),
                      "smallest_pool": min(n), "global_range": glob, "stats": stats}, indent=1))
    # emulated network-angle widths per location against ../spix-output-bits (68 % half-width in degrees, shortest interval)
    ref = json.load(open("/Users/nmangane/smartpixels/doc-archive/spix-output-bits/quant_bits.json"))["locations"]

    def w68(r):
        r = np.sort(r[np.isfinite(r)])
        k = int(np.floor(0.68 * len(r)))
        return float(np.min(r[k:] - r[:len(r) - k]) / 2) if len(r) >= 20 else float("nan")

    def deg(c):
        t = np.degrees(np.arctan(1.0 / c))
        return np.where(t < 0, t + 180.0, t)
    surf = D["link"] // 10000
    res = {}
    for name, S, eta0 in (("IL1, η ≈ 0", 1, 0.0), ("IL1, η ≈ 1.0", 1, 1.0), ("IL1, η ≈ 1.4", 1, 1.4),
                          ("ID1, η ≈ 2.0", 11, 2.0), ("ID1, η ≈ 2.4", 11, 2.4)):
        m = (surf == S) & (np.abs(np.abs(D["eta"]) - eta0) < 0.1)
        e = {"n": int(m.sum())}
        for a, key in (("A", "cotA"), ("B", "cotB")):
            mine = w68(deg(D["net_" + a][m]) - deg(D["true_" + a][m]))
            theirs = ref[name]["vars"][key]["sigma0"]
            e[a] = {"here_deg": mine, "spix_output_bits_deg": theirs, "ratio": mine / theirs}
        res[name] = e
    worst = max(abs(e[a]["ratio"] - 1) for e in res.values() for a in "AB")
    res["verdict"] = "MATERIAL (> 10 %)" if worst > 0.10 else ("minor (5-10 %)" if worst > 0.05 else "pass (< 5 %)")
    res["worst_deviation"] = worst
    json.dump(res, open(ST.LOCAL / "checks" / "angle_widths.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
