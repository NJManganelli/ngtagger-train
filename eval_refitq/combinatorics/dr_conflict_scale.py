"""How big is the duplicate-removal conflict graph, and how fast can DR be?

WHY THIS EXISTS. The interactive page needs duplicate removal that a real system
could perform -- i.e. keyed on SHARED HITS, never on truth. Grouping tracks by
TrackingParticle is an oracle no L1 system has, and it distorts in both
directions at once: real tracks get deduplicated better than any implementable
DR could manage, while fakes get no deduplication at all because they have no TP
to group by.

Doing it honestly in the browser needs either the hit lists (and a sequential
greedy) or a PRECOMPUTED CONFLICT GRAPH over which the browser runs the greedy
restricted to enabled seeds. Which is viable depends on the edge count, and that
is a measurement, not a guess.

It also tests the implementation itself. A slow greedy indicts the algorithm
before it indicts the approach, so the naive per-track loop currently in
seed_arity is timed head to head against a vectorised construction that builds
the conflict graph with sorts and group-bys instead of per-track dictionary
work.

Run on at least 10% of the sample: a single low-statistics point is not a basis
for claiming anything about interactivity.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import tp_findability as TF          # noqa: E402
import seed_arity as SA              # noqa: E402
import kf_emulation as KF            # noqa: E402


def collect(spec, nev, chunk, ptmin, calib_ev):
    """Per-track hit tables and scores, exactly as process_chunk pass 1 builds
    them, with cluster indices offset so they stay unique across chunks."""
    cal = TF.calibrate(spec, calib_ev, ptmin)
    sigz = {int(k): float(v) for k, v in cal["sigz_ot"].items()}
    seeds = TF.seed_universe_over_builds(
        {int(k): float(v) for k, v in cal["r_median"].items()}, TF.BUILD_MASKS)
    targets = list(SA.IL) + list(SA.OT_BARREL)
    G_all, sc_all, off, nev_seen = [], [], 0, 0
    for (U, Q), ev in TF.unified_chunks(spec, nev, chunk, sigz):
        allidx = np.arange(len(U["layer"]))
        for sd in seeds:
            try:
                o = SA.run_seed(U, Q, sd, ptmin, targets)
            except M.TooWide:
                continue
            if o is None or not o.get("tracks_to_fit"):
                continue
            g = KF.hits_from_seed(U, o)
            gc = o.get("_gC")
            trip = (o["_gA"], o["_gB"], gc if gc is not None else o["_gB"])
            fit = KF.fit_tracks(U, Q, trip, gidx=g, use_angles=False)
            keep, chi2s = KF.good_state(fit, ptmin)
            if not keep.any():
                continue
            w = 0.30 if sd.arity == 2 else 0.03
            gk = g[keep]
            G_all.append(np.where(gk >= 0, gk + off, -1))
            sc_all.append(KF.rank_score(fit, chi2s, w_angle=w)[keep])
        off += len(U["layer"])
        nev_seen += len(ev)
        print(f"  {nev_seen} events, {sum(len(s) for s in sc_all):,} tracks",
              flush=True)
    return np.concatenate(G_all), np.concatenate(sc_all), nev_seen


def conflict_edges(G, min_shared=SA.MIN_SHARED_LAYERS, block=40_000_000):
    """Pairs of tracks sharing >= min_shared identical hits, fully vectorised.

    Sort the (track, cluster) incidences by cluster, emit the within-cluster
    track pairs, then group-by-count. No per-track dictionary work: the cost is
    three sorts, which is where numpy is strong and Python is not.
    """
    rows, _ = np.nonzero(G >= 0)
    cl = G[G >= 0].astype(np.int64)
    o = np.argsort(cl, kind="stable")
    cl, tr = cl[o], rows[o].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, cl[1:] != cl[:-1]])
    ends = np.r_[starts[1:], len(cl)]
    k = (ends - starts).astype(np.int64)
    npair = int((k * (k - 1) // 2).sum())
    stats = {"incidences": int(len(cl)), "clusters": int(len(starts)),
             "max_tracks_per_cluster": int(k.max()) if len(k) else 0,
             "co_occurrence_pairs": npair}
    if npair == 0:
        return np.zeros((0, 2), np.int64), stats
    # emit pairs group by group, in blocks, so the pair array never dominates
    keys = []
    buf, nbuf = [], 0
    big = np.flatnonzero(k >= 2)
    for gi in big:
        s, e = starts[gi], ends[gi]
        t = np.sort(tr[s:e])
        a, b = np.triu_indices(len(t), 1)
        buf.append(t[a] * np.int64(1 << 32) + t[b])
        nbuf += len(a)
        if nbuf >= block:
            keys.append(np.concatenate(buf)); buf, nbuf = [], 0
    if buf:
        keys.append(np.concatenate(buf))
    kk = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    kk.sort()
    if not len(kk):
        return np.zeros((0, 2), np.int64), stats
    first = np.r_[True, kk[1:] != kk[:-1]]
    uniq = kk[first]
    cnt = np.diff(np.r_[np.flatnonzero(first), len(kk)])
    sel = cnt >= min_shared
    e = np.stack([uniq[sel] >> np.int64(32), uniq[sel] & np.int64(0xFFFFFFFF)],
                 axis=1)
    stats["distinct_pairs"] = int(len(uniq))
    stats["edges"] = int(len(e))
    return e, stats


def greedy_from_edges(edges, score, ntr):
    """Exact greedy DR over a precomputed conflict graph, CSR adjacency.

    This is what the browser would run per menu, so its cost is the number that
    decides whether option (a) is viable.
    """
    a = np.r_[edges[:, 0], edges[:, 1]]
    b = np.r_[edges[:, 1], edges[:, 0]]
    o = np.argsort(a, kind="stable")
    a, b = a[o], b[o]
    indptr = np.searchsorted(a, np.arange(ntr + 1))
    keep = np.zeros(ntr, bool)
    dead = np.zeros(ntr, bool)
    for t in np.argsort(-score, kind="stable"):
        if dead[t]:
            continue
        keep[t] = True
        dead[b[indptr[t]:indptr[t + 1]]] = True
    return keep, indptr, b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--calib-events", type=int, default=50)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    G, sc, nev = collect(a.input, a.nev, a.chunk, a.ptmin, a.calib_events)
    ntr = len(sc)
    print(f"\ncollected {ntr:,} tracks over {nev} events "
          f"({ntr / nev:,.0f}/ev) in {time.perf_counter() - t0:.0f}s")

    t1 = time.perf_counter()
    edges, st = conflict_edges(G)
    t_build = time.perf_counter() - t1
    deg = np.bincount(np.r_[edges[:, 0], edges[:, 1]], minlength=ntr)
    print(f"\nCONFLICT GRAPH")
    for k, v in st.items():
        print(f"  {k:<26}{v:>15,}")
    print(f"  {'build time':<26}{t_build:>14.1f}s")
    print(f"  {'tracks with >=1 conflict':<26}{int((deg > 0).sum()):>15,}"
          f"  ({100 * (deg > 0).mean():.1f}%)")
    print(f"  {'mean / max degree':<26}{deg.mean():>10.1f} / {deg.max():,}")
    print(f"  {'edge list bytes (2x int32)':<26}{8 * len(edges):>15,}"
          f"  -> {8 * len(edges) / ntr * 12.6e6 / 1e6:,.0f} MB at 12.6M tracks")

    t2 = time.perf_counter()
    keep_v, _, _ = greedy_from_edges(edges, sc, ntr)
    t_greedy = time.perf_counter() - t2
    t3 = time.perf_counter()
    keep_n = SA.duplicate_removal(G, sc)
    t_naive = time.perf_counter() - t3
    agree = int((keep_v == keep_n).sum())
    print(f"\nDUPLICATE REMOVAL, {ntr:,} tracks")
    print(f"  naive per-track loop (seed_arity)  {t_naive:8.2f}s  "
          f"{100 * keep_n.mean():5.1f}% kept")
    print(f"  greedy over precomputed edges      {t_greedy:8.2f}s  "
          f"{100 * keep_v.mean():5.1f}% kept")
    print(f"  agreement                          {100 * agree / ntr:8.2f}%")
    print(f"  speedup (greedy only)              {t_naive / max(t_greedy, 1e-9):8.1f}x")
    print(f"\n  extrapolated to 12.6M tracks: naive "
          f"{t_naive / ntr * 12.6e6:,.0f}s, edge-greedy "
          f"{t_greedy / ntr * 12.6e6:,.0f}s, graph build "
          f"{t_build / ntr * 12.6e6:,.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "n_tracks": ntr, **st,
                   "t_build": t_build, "t_greedy": t_greedy, "t_naive": t_naive,
                   "frac_with_conflict": float((deg > 0).mean()),
                   "mean_degree": float(deg.mean()),
                   "max_degree": int(deg.max()),
                   "agreement": agree / ntr},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
