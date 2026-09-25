"""How parameter resolutions scale with a track's HIT COMPOSITION.

The exported table carries one contamination number, n_wrong, counted against
the TP of the SEED's inner cluster. That is not a usable quality axis:

  - a 6-hit track all from one TP and a 10-hit track with 7+3 get compared on
    n_wrong alone, which says 0 against 3 and nothing about the 7 good hits;
  - a 4-hit track with 3+1 has the same n_wrong as a 10-hit track with 9+1,
    though one has one r-phi degree of freedom left and the other has seven;
  - if the seed cluster is the minority owner, n_wrong counts the MAJORITY's
    hits as wrong and the residual is taken against the wrong particle.

So this measures resolution directly against the composition, splitting correct
and incorrect hits by system, and referencing every residual to the track's
majority owner rather than its seed. The output is the empirical answer to
"which scalar, if any, is the quality axis", including a weighted fit of
log sigma against the four counts.

Independent of the census cache except for its calibration and seed list, since
the composition columns do not exist in the shards.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF            # noqa: E402
import tracklet_topology_cost as M     # noqa: E402
import kf_emulation as KF              # noqa: E402
import seed_arity as SA                # noqa: E402

# The base block is exactly the exported track table, whose truth is now
# referenced to the majority owner -- so d_d0, n_wrong, n_it_right and the rest
# come straight from KF.track_rows and are not recomputed here. EXTRA adds only
# what the export does not carry: the DR flag, the unassociated-hit count, the
# owner's share of the hits, and the pulls against the fit's claimed error.
#
# NAMING, because two quantities both reference d0:
#   d_d0       residual against the majority owner (base) -- the resolution
#   sig_kf_d0  sqrt of the fit covariance (base)          -- the fit's CLAIM
#   pull_d0    d_d0 / sig_kf_d0                           -- does the claim hold
EXTRA = ["dr", "n_noise", "own_frac", "pull_d0", "pull_z0", "pull_kappa"]
COLS = list(KF.TRACK_COLS) + EXTRA


majority_owner = KF.majority_owner   # canonical implementation


def rows_for_tracks(U, Q, TP, gk, trip, fit, seed_idx, arity, sysclass, rank,
                    dr):
    T = KF.gather_hits(U, Q, gk, KF.LAYER_ORDER)
    base = KF.track_rows(U, Q, T, fit, trip, TP, seed_idx, arity, sysclass, rank)
    col = {n: i for i, n in enumerate(KF.TRACK_COLS)}
    G, valid = T["GIDX"], T["VALID"]
    tp = np.where(valid, U["tpIdx"][np.clip(G, 0, None)], -1)
    nhit = np.maximum(valid.sum(axis=1), 1)
    sk = {k: np.maximum(base[:, col[f"sig_kf_{k}"]], 1e-30) if f"sig_kf_{k}" in col
          else np.sqrt(np.maximum(fit[f"var_{k}"], 1e-30))
          for k in ("d0", "z0", "kappa")}
    out = {
        "dr": dr.astype(np.float64),
        "n_noise": (valid & (tp < 0)).sum(axis=1).astype(np.float64),
        "own_frac": base[:, col["n_own"]] / nhit,
        "pull_d0": base[:, col["d_d0"]] / sk["d0"],
        "pull_z0": base[:, col["d_z0"]] / sk["z0"],
        "pull_kappa": base[:, col["d_kappa"]] / sk["kappa"],
    }
    ex = np.stack([out[c] for c in EXTRA], axis=1).astype(np.float32)
    return np.concatenate([base, ex], axis=1)


def collect(spec, nev, chunk, sigz, seeds, ptmin, targets, min_layers,
            verbose=True):
    parts, t0 = [], time.time()
    for ci, ((U, Q), ev) in enumerate(TF.unified_chunks(spec, nev, chunk, sigz)):
        TP = KF.tp_truth_table(U)
        held = []
        for s_i, sd in enumerate(seeds):
            try:
                o = SA.run_seed(U, Q, sd, ptmin, targets, min_layers=min_layers)
            except M.TooWide:
                continue
            if o is None or not o.get("tracks_to_fit"):
                continue
            gidx = KF.hits_from_seed(U, o)
            gc = o.get("_gC")
            trip = (o["_gA"], o["_gB"], gc if gc is not None else o["_gB"])
            fit = KF.fit_tracks(U, Q, trip, gidx=gidx, use_angles=False)
            keep, chi2s = KF.good_state(fit, ptmin)
            if not keep.any():
                continue
            ok = SA.filter_tracks(o, keep)
            gck = ok.get("_gC")
            held.append({
                "s_i": s_i, "sd": sd, "gk": gidx[keep],
                "trip": (ok["_gA"], ok["_gB"],
                         gck if gck is not None else ok["_gB"]),
                "score": KF.rank_score(fit, chi2s,
                                       w_angle=0.30 if sd.arity == 2 else 0.03)[keep],
            })
            del o, fit
        if not held:
            continue
        # DR exactly as the census runs it, but the survivors are a FLAG rather
        # than a filter: the losers are what populate the contaminated cells,
        # and dropping them would measure resolution on a sample already
        # cleaned by the very correlation under study.
        surv = SA.duplicate_removal(
            np.concatenate([h["gk"] for h in held]),
            np.concatenate([h["score"] for h in held]))
        off = 0
        for h in held:
            n = len(h["score"])
            fit = KF.fit_tracks(U, Q, h["trip"], gidx=h["gk"], use_angles=False)
            parts.append(rows_for_tracks(
                U, Q, TP, h["gk"], h["trip"], fit, h["s_i"], h["sd"].arity,
                KF.sysclass_of(h["sd"].layers), h["score"],
                surv[off:off + n]))
            off += n
        if verbose:
            print(f"  chunk {ci:>3}  {len(ev):>3} ev  "
                  f"{sum(len(p) for p in parts):>9,} tracks  "
                  f"{time.time() - t0:7.1f}s", flush=True)
        del U, Q, TP, held
    return np.concatenate(parts) if parts else np.zeros((0, len(COLS)), np.float32)


# ==========================================================================
# reporting
# ==========================================================================
def sig(x, nmin=100):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < nmin:
        return float("nan"), len(x)
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0])), len(x)


def cell_table(R, c, keys, val, scale, nmin):
    """sigma per unique combination of `keys`, plus the cell population."""
    K = np.stack([R[:, c[k]] for k in keys], axis=1)
    uq, inv = np.unique(K, axis=0, return_inverse=True)
    out = []
    for i, u in enumerate(uq):
        m = inv == i
        s, n = sig(R[m, c[val]] * scale, nmin)
        out.append((tuple(float(x) for x in u), s, int(m.sum()), n))
    return out


def wfit(X, y, w):
    """Weighted least squares; returns (coefficients, weighted R^2)."""
    W = np.sqrt(w)[:, None]
    A = np.concatenate([np.ones((len(y), 1)), X], axis=1)
    b, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)
    pred = A @ b
    ybar = np.average(y, weights=w)
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    return b, 1.0 - ss_res / max(ss_tot, 1e-30)


def report(R, c, nmin, dr_only):
    J = {}
    if dr_only:
        R = R[R[:, c["dr"]] > 0]
    n = len(R)
    own_ok = np.isfinite(R[:, c["d_d0"]])
    print(f"\n{n:,} accepted tracks"
          f"{' (post-DR survivors only)' if dr_only else ' (pre-DR, all accepted)'}")
    print(f"  {100 * own_ok.mean():.1f}% have a majority owner with an IT-based "
          f"truth helix, so a residual is defined")
    seed_dis = (R[:, c["own_is_seed"]] == 0)
    print(f"  {100 * seed_dis.mean():.2f}% of tracks have a majority owner that "
          f"is NOT the seed's TP")
    share = R[seed_dis, c["n_own"]] >= 3
    print(f"     of those, {100 * share.mean():.1f}% are owned by >= 3 hits "
          f"of that other TP")
    J["n_tracks"] = n
    J["frac_residual_defined"] = float(own_ok.mean())
    J["frac_owner_not_seed"] = float(seed_dis.mean())

    R = R[own_ok]
    print(f"\n{len(R):,} tracks enter the resolution tables")

    # ---- what the seed-TP convention costs -------------------------------
    print("\nsigma(d0) [um] under the two contamination conventions")
    print("  n_wrong  own-referenced      seed-referenced")
    for w in (0, 1, 2, 3):
        mo = (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == w) if w < 3 else \
             (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] >= 3)
        ms = (R[:, c["n_wrong"]] == w) if w < 3 else (R[:, c["n_wrong"]] >= 3)
        so, no = sig(R[mo, c["d_d0"]] * 1e4, nmin)
        ss, ns = sig(R[ms, c["d_d0"]] * 1e4, nmin)
        lab = f">={w}" if w == 3 else f"  {w}"
        print(f"    {lab}     {so:8.0f} ({no:>8,})   {ss:8.0f} ({ns:>8,})")

    # ---- does the CLAIMED error predict the observed spread? -------------
    # The covariance update never touches the measured residuals, so the fit's
    # claimed sigma is a function of the hit pattern and the assumed per-hit
    # errors alone. It therefore cannot know that a hit belongs to another
    # particle, and the pull width is the size of that blind spot.
    print("\npull width = own-referenced residual / fit's claimed sigma "
          "(1.00 = claim holds)")
    print("   wrong hits      d0        z0      1/pT    median claimed "
          "sigma(d0) [um]")
    J["pull"] = {}
    nw = R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]]
    for w in (0, 1, 2, 3):
        m = (nw == w) if w < 3 else (nw >= 3)
        pd, n = sig(R[m, c["pull_d0"]], nmin)
        pz, _ = sig(R[m, c["pull_z0"]], nmin)
        pk, _ = sig(R[m, c["pull_kappa"]], nmin)
        sk = (np.nanmedian(R[m, c["sig_kf_d0"]]) * 1e4 if m.any() else np.nan)
        lab = f">={w}" if w == 3 else f"  {w}"
        print(f"     {lab}     {pd:8.2f}  {pz:8.2f}  {pk:8.2f}      {sk:10.0f}"
              f"   ({n:,})")
        J["pull"][lab.strip()] = {"d0": pd, "z0": pz, "invpt": pk,
                                  "claimed_sigma_d0_um": float(sk), "n": n}
    print("  same, for CLEAN tracks only, by correct-hit count "
          "(tests the error model itself, not contamination)")
    for nr in range(3, 11):
        m = (R[:, c["n_own"]] == nr) & (nw == 0)
        pd, n = sig(R[m, c["pull_d0"]], nmin)
        pz, _ = sig(R[m, c["pull_z0"]], nmin)
        sd, _ = sig(R[m, c["d_d0"]] * 1e4, nmin)
        sk = (np.nanmedian(R[m, c["sig_kf_d0"]]) * 1e4 if m.any() else np.nan)
        print(f"     {nr:>2} hits  pull(d0) {pd:6.2f}  pull(z0) {pz:6.2f}   "
              f"observed {sd:7.0f} um   claimed {sk:7.0f} um   ({n:,})")

    # ---- correct hits against wrong hits ---------------------------------
    for val, scale, unit in (("d_d0", 1e4, "sigma(d0) [um]"),
                             ("d_kappa", 1.0, "sigma(1/pT) [1/GeV]"),
                             ("d_z0", 1e4, "sigma(z0) [um]")):
        print(f"\n{unit}: rows = correct hits (majority owner), "
              f"cols = wrong hits")
        wmax = 4
        print("        " + "".join(f"{w:>10}" for w in range(wmax + 1)))
        for nr in range(3, 11):
            cells = []
            for w in range(wmax + 1):
                m = ((R[:, c["n_own"]] == nr)
                     & (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == w))
                s, k = sig(R[m, c[val]] * scale, nmin)
                cells.append("         -" if not np.isfinite(s)
                             else (f"{s:10.0f}" if scale > 1 else f"{s:10.5f}"))
            print(f"  {nr:>5} " + "".join(cells))

    # ---- system split at fixed total ------------------------------------
    print("\nsigma(d0) [um] by system: rows = correct IT hits, "
          "cols = correct OT hits (wrong hits summed over)")
    print("        " + "".join(f"{o:>9}" for o in range(7)))
    for i in range(5):
        cells = []
        for o in range(7):
            m = (R[:, c["n_it_right"]] == i) & (R[:, c["n_ot_right"]] == o)
            s, k = sig(R[m, c["d_d0"]] * 1e4, nmin)
            cells.append("        -" if not np.isfinite(s) else f"{s:9.0f}")
        print(f"  {i:>5} " + "".join(cells))

    print("\nsame, but for tracks carrying exactly ONE wrong hit, "
          "split by where it sits")
    print("   correct IT / correct OT     wrong in IT      wrong in OT")
    for i in (2, 3, 4):
        for o in (0, 3, 6):
            mi = ((R[:, c["n_it_right"]] == i) & (R[:, c["n_ot_right"]] == o)
                  & (R[:, c["n_it_wrong"]] == 1) & (R[:, c["n_ot_wrong"]] == 0))
            mo = ((R[:, c["n_it_right"]] == i) & (R[:, c["n_ot_right"]] == o)
                  & (R[:, c["n_it_wrong"]] == 0) & (R[:, c["n_ot_wrong"]] == 1))
            si, ni = sig(R[mi, c["d_d0"]] * 1e4, nmin)
            so, no = sig(R[mo, c["d_d0"]] * 1e4, nmin)
            f = lambda s, k: "        -" if not np.isfinite(s) else f"{s:7.0f} ({k:>6,})"
            print(f"        {i} / {o}          {f(si, ni)}   {f(so, no)}")

    # ---- the three cases from the question ------------------------------
    print("\nthe three configurations put forward, measured")
    cases = [("6 hits, all one TP", lambda: (R[:, c["n_own"]] == 6)
              & (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == 0)),
             ("10 hits, 7 + 3", lambda: (R[:, c["n_own"]] == 7)
              & (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == 3)),
             ("4 hits, 3 + 1", lambda: (R[:, c["n_own"]] == 3)
              & (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == 1)),
             ("10 hits, all one TP", lambda: (R[:, c["n_own"]] == 10)
              & (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == 0)),
             ("4 hits, all one TP", lambda: (R[:, c["n_own"]] == 4)
              & (R[:, c["n_it_wrong"]] + R[:, c["n_ot_wrong"]] == 0))]
    print(f"  {'configuration':<22}{'N':>10}{'sigma(d0)':>11}"
          f"{'sigma(1/pT)':>13}{'sigma(z0)':>11}")
    J["cases"] = {}
    for name, f in cases:
        m = f()
        sd, nd = sig(R[m, c["d_d0"]] * 1e4, nmin)
        sk, _ = sig(R[m, c["d_kappa"]], nmin)
        sz, _ = sig(R[m, c["d_z0"]] * 1e4, nmin)
        print(f"  {name:<22}{nd:>10,}{sd:>11.0f}{sk:>13.5f}{sz:>11.0f}")
        J["cases"][name] = {"n": nd, "sigma_d0_um": sd, "sigma_invpt": sk,
                            "sigma_z0_um": sz}

    # ---- which scalar IS the quality axis --------------------------------
    keys = ["n_it_right", "n_it_wrong", "n_ot_right", "n_ot_wrong"]
    for val, scale, unit in (("d_d0", 1e4, "sigma(d0)"),
                             ("d_kappa", 1.0, "sigma(1/pT)")):
        cells = [x for x in cell_table(R, c, keys, val, scale, max(nmin, 200))
                 if np.isfinite(x[1]) and x[1] > 0]
        if len(cells) < 8:
            continue
        K = np.array([x[0] for x in cells], float)
        y = np.log(np.array([x[1] for x in cells], float))
        w = np.array([x[3] for x in cells], float)
        it_r, it_w, ot_r, ot_w = K.T
        nown, nwrong = it_r + ot_r, it_w + ot_w
        nhit = nown + nwrong
        models = {
            "n_wrong (current axis)": nwrong[:, None],
            "n_own": nown[:, None],
            "own_frac": (nown / np.maximum(nhit, 1))[:, None],
            "n_own, n_wrong": np.stack([nown, nwrong], axis=1),
            "n_own-3 (rphi dof), n_wrong": np.stack([nown - 3, nwrong], axis=1),
            "four counts by system": K,
        }
        print(f"\nwhich composition scalar explains log {unit} across "
              f"{len(cells)} cells ({int(w.sum()):,} tracks, weighted R^2)")
        J.setdefault("models", {})[val] = {}
        for name, X in models.items():
            b, r2 = wfit(X, y, w)
            print(f"  {name:<30} R^2 = {r2:6.3f}   "
                  + "  ".join(f"{v:+.3f}" for v in b[1:]))
            J["models"][val][name] = {"r2": round(r2, 4),
                                      "coef": [round(float(v), 4) for v in b]}
        b, r2 = wfit(K, y, w)
        print(f"  per-hit effect on {unit}, exp(coefficient): "
              + ", ".join(f"{k} x{np.exp(v):.3f}" for k, v in zip(keys, b[1:])))
        J["models"][val]["per_hit_factor"] = {
            k: round(float(np.exp(v)), 4) for k, v in zip(keys, b[1:])}

    print("\nlargest cells, worst resolution first "
          "(it_right, it_wrong, ot_right, ot_wrong)")
    cells = [x for x in cell_table(R, c, keys, "d_d0", 1e4, max(nmin, 200))
             if np.isfinite(x[1])]
    cells.sort(key=lambda x: -x[1])
    print(f"  {'cell':<26}{'N':>10}{'sigma(d0) um':>14}")
    for u, s, ntot, k in cells[:12]:
        print(f"  {str(tuple(int(v) for v in u)):<26}{k:>10,}{s:>14.0f}")
    for u, s, ntot, k in cells[-8:]:
        print(f"  {str(tuple(int(v) for v in u)):<26}{k:>10,}{s:>14.0f}")
    J["cells"] = [{"cell": list(map(int, u)), "sigma_d0_um": s, "n": k}
                  for u, s, ntot, k in cells]
    return J


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--nmin", type=int, default=100,
                    help="minimum tracks in a cell before a sigma is quoted")
    ap.add_argument("--dr-only", action="store_true",
                    help="restrict to duplicate-removal survivors")
    ap.add_argument("--npz", default=None, help="write the per-track table")
    ap.add_argument("-o", "--out", default=None, help="write the report as JSON")
    a = ap.parse_args()

    ds = sorted(glob.glob(f"{a.cache_dir}/tpcensus_*"))
    if not ds:
        raise SystemExit(f"no census under {a.cache_dir} to take calibration from")
    meta = json.load(open(Path(ds[-1]) / "manifest.json"))
    sigz = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
    seeds = [SA.Seed(tuple(x)) for x in meta["seeds"]]
    targets = list(SA.IL) + list(SA.OT_BARREL)
    print(f"calibration and {len(seeds)} seeds from {Path(ds[-1]).name}, "
          f"{a.nev} events, ptmin {a.ptmin}", flush=True)
    R = collect(a.input, a.nev, a.chunk, sigz, seeds, a.ptmin, targets,
                meta["config"].get("min_layers", SA.MIN_LAYERS))
    if not len(R):
        raise SystemExit("no tracks")
    c = {n: i for i, n in enumerate(COLS)}
    if a.npz:
        np.savez_compressed(a.npz, rows=R, cols=np.array(COLS))
        print(f"wrote {a.npz}")
    J = report(R, c, a.nmin, a.dr_only)
    if a.out:
        json.dump({"n_events": a.nev, "ptmin": a.ptmin, "dr_only": a.dr_only,
                   **J}, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
