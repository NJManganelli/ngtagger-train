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
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_events": nev, "mask": a.mask, "counts": acc,
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
