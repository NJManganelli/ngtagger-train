"""Three L1 architectures, compared on the same events.

  PARALLEL       IT-only seeds find IT mini-tracks with NO OT information, OT-only
                 seeds find OT tracks, duplicate removal runs inside each system
                 separately, then the two outputs are MATCHED track-to-track and
                 refit on the union of their hits. This is the hardware-limited
                 design: during the 4-5 us the OT finder occupies, a SmartPixels
                 finder runs standalone alongside it.

  COMBINED       every seed follows all ten layers; mixed IT+OT seeds exist.

  DIGIREFIT-LIKE an OT track projected into the IT, attaching clusters, refit --
                 the IT is guided by the OT and finds nothing itself. Obtained as
                 COMBINED restricted to OT-only seeds, so it costs nothing extra.
                 The existing SmartPixels digiRefit is a simplified,
                 hardware-ignorant precursor of the parallel design.

The question the PARALLEL design lives or dies on is whether an IT mini-track can
be matched to the right OT track. They share zero hits, so only helix agreement
is available, and the IT track's lever arm is at most 13 cm.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import tp_findability as TF          # noqa: E402
import seed_arity as SA              # noqa: E402
import kf_emulation as KF            # noqa: E402
import arch_parallel as AP           # noqa: E402


def rs(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def scatter_rows(U, IT, OT, ia, ib, TP):
    """Rows for measuring the INTER-SYSTEM scattering angle directly.

    The IT mini-track and the OT track are fitted INDEPENDENTLY on the same
    particle -- neither saw the other's hits -- so any disagreement between them
    is either the two fits' own resolution or a real deflection in between.
    Nothing else can produce it.

    From the kink decomposition, a scatter of angle delta at radius r_s makes the
    DOWNSTREAM description read phi0 + delta (and d0 - delta*r_s) while the
    upstream one reads the truth. So phi0_IT - phi0_OT measures -delta directly,
    and cot_IT - cot_OT measures the longitudinal kink the same way.

    Both tracks are required TRUTH-CLEAN, every hit from the matched TP: a single
    wrong hit would masquerade as a large deflection and bias the width.
    """
    ta, tb = IT["tpIdx"][ia], OT["tpIdx"][ib]
    ok = (ta >= 0) & (ta == tb)
    for S, idx in ((IT, ia), (OT, ib)):
        G = S["gidx"][idx]
        on = G >= 0
        tp = np.where(on, U["tpIdx"][np.clip(G, 0, None)], -2)
        ok &= (on & (tp == ta[:, None])).sum(1) == on.sum(1)
    if not ok.any():
        return None
    a_, b_ = ia[ok], ib[ok]
    kk = M.tp_key(IT["event"][a_], IT["tpIdx"][a_])
    p = np.clip(np.searchsorted(TP["key"], kk), 0, max(len(TP["key"]) - 1, 0))
    good = TP["key"][p] == kk
    a_, b_, p = a_[good], b_[good], p[good]
    return {"inv_pt": TP["kappa"][p],
            "abs_eta": np.abs(np.arcsinh(np.clip(TP["cot"][p], -30, 30))),
            "dphi": M.wrap(IT["phi0"][a_] - OT["phi0"][b_]),
            "dcot": IT["cot"][a_] - OT["cot"][b_],
            "vphi": IT["var_phi0"][a_] + OT["var_phi0"][b_],
            "vcot": IT["var_cot"][a_] + OT["var_cot"][b_]}


ETA_BANDS = ((0.0, 0.5), (0.5, 0.9), (0.9, 1.4))


def report_scattering(S, eta_bands=ETA_BANDS):
    """Inter-system disagreement vs 1/pT, per |eta| band.

    BINNING IN ETA IS THE POINT, not a refinement. Material grows with |eta|
    while the measurement variance's pT dependence does not, so the eta profile
    both breaks the degeneracy that made the phi0 channel unusable in a single
    barrel-wide fit AND is the quantity a tkLayout x/X0-vs-eta curve can be
    compared against directly. A barrel average can only be checked against a
    barrel average.
    """
    for lo, hi in eta_bands:
        m = (S["abs_eta"] >= lo) & (S["abs_eta"] < hi)
        if m.sum() < 120:
            continue
        print(f"\n--- |eta| {lo}-{hi}   {int(m.sum()):,} pairs")
        _report_one({k: v[m] for k, v in S.items()})
    print("\n--- all |eta| < 1.4")
    return _report_one(S)


def _report_one(S):
    """Width of the inter-system disagreement vs 1/pT, with the fits' own
    resolution subtracted in quadrature. The residual slope IS the scattering
    constant, in the same rad*GeV units as the OT's KalmanMultiScattTerm."""
    print(f"{'pT band':<12}{'n':>7}{'sig(dphi0)':>12}{'expected':>10}{'excess':>10}"
          f"{'sig(dcot)':>12}{'expected':>10}{'excess':>10}")
    bands = [(2, 3), (3, 5), (5, 10), (10, 1e9)]
    kk_, wps, eps, wcs, ecs = [], [], [], [], []
    for lo, hi in bands:
        pt = 1.0 / np.maximum(S["inv_pt"], 1e-6)
        m = (pt >= lo) & (pt < hi)
        if m.sum() < 30:
            continue
        wp, wc = rs(S["dphi"][m]), rs(S["dcot"][m])
        ep = float(np.sqrt(np.median(S["vphi"][m])))
        ec = float(np.sqrt(np.median(S["vcot"][m])))
        xp = np.sqrt(max(wp * wp - ep * ep, 0.0))
        xc = np.sqrt(max(wc * wc - ec * ec, 0.0))
        lab = f"{lo}-{'MAX' if hi > 1e8 else hi}"
        print(f"{lab:<12}{int(m.sum()):>7,}{wp:>12.5f}{ep:>10.5f}{xp:>10.5f}"
              f"{wc:>12.5f}{ec:>10.5f}{xc:>10.5f}")
        kk_.append(float(np.median(S["inv_pt"][m])))
        wps.append(wp); eps.append(ep); wcs.append(wc); ecs.append(ec)
    # TWO COMPONENTS, and they must be separated or the constant is wrong.
    # Subtracting the covariance in quadrature assumes the covariance is right.
    # It is not: the pull does not go to 1 at high pT, where scattering has
    # vanished, so the fits are reporting sigmas that are too small by a
    # pT-INDEPENDENT factor. Fitting
    #       width^2 = c^2 * expected^2 + (b / pT)^2
    # separates a covariance scale c from the scattering constant b, where
    # subtracting in quadrature would have folded all of c into b.
    out = {}
    for nm, W, E in (("phi0", wps, eps), ("cot", wcs, ecs)):
        if len(kk_) < 2:
            continue
        A = np.stack([np.array(E) ** 2, np.array(kk_) ** 2], axis=1)
        y = np.array(W) ** 2
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        c = float(np.sqrt(max(coef[0], 0.0)))
        b = float(np.sqrt(max(coef[1], 0.0)))
        out[nm] = {"scatter_radGeV": b, "cov_scale": c}
        print(f"  d{nm}: scattering {b:.5f} rad*GeV, covariance scale {c:.2f}"
              f"   (OT KalmanMultiScattTerm = 0.00075)")
        naive = float(np.sum(np.array(kk_) * np.array(
            [np.sqrt(max(w * w - e * e, 0.0)) for w, e in zip(W, E)]))
            / max(np.sum(np.array(kk_) ** 2), 1e-30))
        print(f"        quadrature-subtraction would have said {naive:.5f}, "
              f"inflated {naive / max(b, 1e-9):.1f}x by the covariance error")
    return out


# TRACK_COLS plus two columns the census table cannot carry: which architecture
# produced the row, and which stage of it. Written to a SEPARATE file rather than
# extending the census schema, so the combined-architecture census can run
# undisturbed -- its shards are keyed on a format version that must not move.
ARCH_EXTRA = ("arch_mode", "track_stage")
ARCH_COMBINED, ARCH_PARALLEL = 0, 1
STAGE_IT, STAGE_OT, STAGE_MATCHED = 0, 1, 2


def arch_rows(U, Q, TP, gidx, trip, fit, rank, stage, seed_idx):
    """One block of parallel-architecture track rows, TRACK_COLS + the two extras."""
    T = KF.gather_hits(U, Q, gidx, KF.LAYER_ORDER)
    base = KF.track_rows(U, Q, T, fit, trip, TP, seed_idx, 0,
                         KF.SYS_IT if stage == STAGE_IT else
                         (KF.SYS_OT if stage == STAGE_OT else KF.SYS_MIX), rank)
    n = len(base)
    extra = np.stack([np.full(n, float(ARCH_PARALLEL)),
                      np.full(n, float(stage))], axis=1).astype(np.float32)
    return np.concatenate([base, extra], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--mask", default="AAAA")
    ap.add_argument("--it-min-layers", type=int, default=AP.IT_MIN_LAYERS)
    ap.add_argument("--ms-scan", default="",
                    help="comma-separated process-noise scales (rad*GeV per "
                         "step) to scan for the matched refit, e.g. "
                         "0,0.0003,0.001,0.003,0.01")
    ap.add_argument("--it-ot-scale", type=float, default=3.0,
                    help="extra factor on the IT->OT crossing step, where the "
                         "support and services sit")
    ap.add_argument("--scattering", action="store_true",
                    help="measure the inter-system scattering angle from "
                         "independently fitted IT and OT tracks")
    ap.add_argument("--export-tracks", default=None,
                    help="npz path for per-track rows of the PARALLEL "
                         "architecture: IT mini-tracks, OT tracks and matched "
                         "refits, in the census TRACK_COLS schema plus "
                         "arch_mode and track_stage")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    cal = TF.calibrate(a.input, min(a.nev, 50), a.ptmin)
    sigz = {int(k): float(v) for k, v in cal["sigz_ot"].items()}
    il = [TF.IL_OF[i] for i, ch in enumerate(a.mask) if ch == "A"]
    seeds = SA.enumerate_seeds(il)
    print(f"{a.mask}: {len(seeds)} seeds "
          f"({sum(1 for s in seeds if all(L <= 4 for L in s.layers))} IT-only, "
          f"{sum(1 for s in seeds if all(L > 10 for L in s.layers))} OT-only, "
          f"{sum(1 for s in seeds if any(L <= 4 for L in s.layers) and any(L > 10 for L in s.layers))} mixed)")

    CUTS = (5.0, 10.0, 25.0, 50.0, 200.0, 1e9)
    PARSETS = {"phi0+z0": ("phi0", "z0"),
               "phi0+z0+cot": ("phi0", "z0", "cot"),
               "all four": ("phi0", "z0", "cot", "kappa")}
    acc = {"n_it": 0, "n_ot": 0, "n_it_real": 0, "n_ot_real": 0, "nev": 0}
    mtot = {}
    res = {}
    scat = {}
    trows = []
    for (U, Q), ev in TF.unified_chunks(a.input, a.nev, a.chunk, sigz):
        TP = KF.tp_truth_table(U)
        IT = AP.pool_and_dr(U, AP.find_system(U, Q, seeds, a.ptmin, "IT",
                                              a.it_min_layers))
        OT = AP.pool_and_dr(U, AP.find_system(U, Q, seeds, a.ptmin, "OT",
                                              SA.MIN_LAYERS))
        acc["nev"] += len(ev)
        if IT is None or OT is None:
            continue
        acc["n_it"] += len(IT["gA"]); acc["n_ot"] += len(OT["gA"])
        acc["n_it_real"] += int((IT["tpIdx"] >= 0).sum())
        acc["n_ot_real"] += int((OT["tpIdx"] >= 0).sum())
        for pname, pars in PARSETS.items():
            for cut in CUTS:
                ia, ib, c2 = AP.match_systems(IT, OT, pars, cut)
                k = (pname, cut)
                d = mtot.setdefault(k, {"n": 0, "right": 0, "both_real": 0})
                d["n"] += len(ia)
                if len(ia):
                    ta, tb = IT["tpIdx"][ia], OT["tpIdx"][ib]
                    d["right"] += int(((ta >= 0) & (ta == tb)).sum())
                    d["both_real"] += int(((ta >= 0) & (tb >= 0)).sum())
        # resolutions for the three architectures, at the default cut
        ia, ib, _ = AP.match_systems(IT, OT, ("phi0", "z0", "cot"), 25.0)
        if a.export_tracks:
            # the two ingredients, then the matched-and-refit product
            for S, stage in ((IT, STAGE_IT), (OT, STAGE_OT)):
                gA, gB = S["gA"], S["gB"]
                f = {k: S[k] for k in ("kappa", "phi0", "d0", "cot", "z0",
                                       "var_kappa", "var_phi0", "var_d0",
                                       "var_cot", "var_z0", "nhit")}
                # the per-track chi2 is not carried through pool_and_dr; refit
                # to recover it rather than plumbing a parallel array
                fr = KF.fit_tracks(U, Q, (gA, gB, gA), gidx=S["gidx"],
                                   use_angles=False)
                trows.append(arch_rows(U, Q, TP, S["gidx"], (gA, gB, gA), fr,
                                       S["score"], stage, -1))
            if len(ia):
                fitm, Gm = AP.refit_matched(U, Q, IT, OT, ia, ib)
                gA = IT["gA"][ia]
                trows.append(arch_rows(U, Q, TP, Gm, (gA, IT["gB"][ia], gA),
                                       fitm, IT["score"][ia], STAGE_MATCHED, -1))
        if a.scattering and len(ia):
            sr = scatter_rows(U, IT, OT, ia, ib, TP)
            if sr:
                for k, v in sr.items():
                    scat.setdefault(k, []).append(v)
        if len(ia):
            ta, tb = IT["tpIdx"][ia], OT["tpIdx"][ib]
            good = (ta >= 0) & (ta == tb)
            scales = ([float(x) for x in a.ms_scan.split(",") if x.strip()]
                      if a.ms_scan else [0.0])
            for msc, rev in [(m, r) for m in scales for r in (False, True)]:
                fit, G = AP.refit_matched(U, Q, IT, OT, ia, ib,
                                          ms_scale=msc,
                                          it_ot_scale=a.it_ot_scale,
                                          reverse=rev)
                if not good.any():
                    continue
                gA = IT["gA"][ia][good]
                tr = (gA, IT["gB"][ia][good], gA)
                f2 = {k: (v[good] if isinstance(v, np.ndarray) and v.ndim == 1
                          else v) for k, v in fit.items() if k != "hits"}
                dk, dd, dc, dz, rm = KF.truth_residuals(U, tr, f2, TP)
                d = "out->in" if rev else "in->out"
                lab = (f"PARALLEL refit, Q=0 {d}" if msc == 0.0 else
                       f"  + Q {msc:g} {d}")
                r = res.setdefault(lab, [[], [], [], []])
                for i, v in enumerate((dk, dd, dc, dz)):
                    r[i].append(v)
        # the two ingredients on their own
        for lab, S in (("  IT mini-track alone", IT), ("  OT track alone", OT)):
            # the WHOLE seed must be truth-consistent, not just its inner
            # cluster: requiring only gA let contaminated tracks in and read the
            # OT-alone resolution as 66 cm in z0, which is contamination and not
            # a measurement of anything.
            m = (S["tpIdx"] >= 0) & (S["tpIdx"] == U["tpIdx"][S["gB"]])
            if m.sum() < 20:
                continue
            kk = M.tp_key(S["event"][m], S["tpIdx"][m])
            p = np.clip(np.searchsorted(TP["key"], kk), 0, len(TP["key"]) - 1)
            hit = TP["key"][p] == kk
            r = res.setdefault(lab, [[], [], [], []])
            r[0].append(np.abs(S["kappa"][m][hit]) - TP["kappa"][p[hit]])
            r[1].append(S["d0"][m][hit] - TP["d0"][p[hit]])
            r[2].append(S["cot"][m][hit] - TP["cot"][p[hit]])
            r[3].append(S["z0"][m][hit] - TP["z0"][p[hit]])
    nev = max(acc["nev"], 1)
    print(f"\n{nev} events")
    print(f"  IT mini-tracks {acc['n_it'] / nev:>8,.0f}/ev  "
          f"({100 * acc['n_it_real'] / max(acc['n_it'], 1):.1f}% seed-real)")
    print(f"  OT tracks      {acc['n_ot'] / nev:>8,.0f}/ev  "
          f"({100 * acc['n_ot_real'] / max(acc['n_ot'], 1):.1f}% seed-real)")
    print(f"\nIT<->OT MATCH: which parameters, and how tight\n"
          f"{'parameters':<14}{'chi2 cut':>10}{'matches/ev':>12}{'correct':>9}"
          f"{'of both-real':>14}")
    for (pname, cut), d in mtot.items():
        if not d["n"]:
            continue
        print(f"{pname:<14}{cut:>10.0f}{d['n'] / nev:>12,.0f}"
              f"{100 * d['right'] / d['n']:>8.0f}%"
              f"{100 * d['right'] / max(d['both_real'], 1):>13.0f}%")
    print(f"\n{'stage':<26}{'n':>8}{'s(kap)':>9}{'s(d0)um':>9}{'s(cot)':>9}{'s(z0)um':>9}")
    for lab, v in res.items():
        dk, dd, dc, dz = [np.concatenate(x) for x in v]
        print(f"{lab:<26}{len(dk):>8,}{rs(dk):>9.4f}{1e4 * rs(dd):>9.0f}"
              f"{rs(dc):>9.4f}{1e4 * rs(dz):>9.0f}")
    sconst = {}
    if a.scattering and scat:
        print(f"\nINTER-SYSTEM SCATTERING, from independently fitted IT and OT "
              f"tracks on the same particle")
        sconst = report_scattering({k: np.concatenate(v) for k, v in scat.items()})
    if a.export_tracks and trows:
        A = np.concatenate(trows)
        Path(a.export_tracks).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.export_tracks, tracks=A,
                            cols=np.array(list(KF.TRACK_COLS) + list(ARCH_EXTRA)),
                            n_events=nev)
        print(f"\nwrote {a.export_tracks}: {len(A):,} parallel-architecture "
              f"track rows ({len(A) / nev:,.0f}/ev)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "mask": a.mask, "counts": acc,
                   "scattering_const": sconst,
                   "match": {f"{k[0]}|{k[1]}": v for k, v in mtot.items()},
                   "resolutions": {k: [len(np.concatenate(v[0])),
                                       rs(np.concatenate(v[0])),
                                       1e4 * rs(np.concatenate(v[1])),
                                       rs(np.concatenate(v[2])),
                                       1e4 * rs(np.concatenate(v[3]))]
                                   for k, v in res.items()}},
                  open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
