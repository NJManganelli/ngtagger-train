"""Every IT+OT seed pair, every projection target, at two working points.

WHY TWO WORKING POINTS. Windows can be set two ways, and the difference is not
a detail -- it is the efficiency being traded for combinatorics, and it must be
visible rather than buried in a constant.

  OT-LIKE. The Phase-2 OT tracklet does NOT compute its match windows from a
  resolution model: rphimatchcut_ / zmatchcut_ in Settings.h are tabulated
  constants tuned per (seed, projection layer). MEASURED on correct projections
  of the OT L1L2 seed, those constants contain 0.93 / 0.87 / 0.87 / 0.71 of the
  rphi residual at L3-L6 and 0.99 / 0.99 / 0.98 / 0.98 of the z residual. So the
  OT deliberately runs TIGHT in the bending plane -- giving up ~12% per
  projection -- and loose longitudinally. This working point reproduces that
  posture: the rphi window is the Q_RPHI_OT quantile of the measured residual,
  the z window the Q_Z_OT quantile.

  ANALYTIC. Three times the ROBUST half-width (the 68.27% quantile) of the same
  residual -- the "3 sigma of the core" a resolution model would give. NOT the
  99.7% quantile: these residuals have a catastrophic tail, because the correct
  triple is taken as the TP's first cluster on each layer and a delta ray or a
  split cluster can substitute a wrong one. Sizing at the raw 99.7% quantile gave
  a 4.1 cm rphi window for IL1+IL2 -> IL3 against the OT-like 0.013 cm, a 315x
  ratio set entirely by mis-association. The containment each window actually
  ACHIEVES is reported, so the tail stays visible instead of being absorbed.

Both are QUANTILES OF A MEASURED RESIDUAL, not N*sigma. The residuals have tails,
and an earlier revision sized everything at 3*sigma of a Gaussian model whose
sigma(cot) came from cluster positions alone -- optimistic by 2x -- while its
sigma(kappa) assumed 1 mrad per cluster -- pessimistic by 2.2x. Quantiles of the
thing itself carry no such model.

The enumeration is exhaustive: every ordered pair of the ten barrel layers
(IT L1-L4 as 1-4, OT L1-L6 as 11-16) as a seed, and every remaining layer as a
projection target. Seed quality and window containment need only TRUTH-MATCHED
triples, so all 45 pairs x 8 targets are affordable without running combinatorics.

Writes a settings file carrying both working points, so the cost model can be
driven from measured constants instead of an analytic model.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402

IT_LAYERS = (1, 2, 3, 4)
OT_LAYERS = (11, 12, 13, 14, 15, 16)
ALL_LAYERS = IT_LAYERS + OT_LAYERS
NAME = {1: "IT_L1", 2: "IT_L2", 3: "IT_L3", 4: "IT_L4", 11: "OT_L1", 12: "OT_L2",
        13: "OT_L3", 14: "OT_L4", 15: "OT_L5", 16: "OT_L6"}
# measured from the OT L1L2 seed against Settings.h; see the module docstring
Q_RPHI_OT, Q_Z_OT = 88.0, 99.0
Q_CORE, N_SIGMA_ANALYTIC = 68.27, 3.0


def build(src, nev):
    """One unified hit table."""
    I, n, _ = M.load_flat(src, M.IT_TABLE,
                          ["layer", "globalR", "globalZ", "globalPhi", "sigY",
                           "tpIdx", "tpPt"], nev)
    O, on, _ = M.load_ot(src, nev)
    bar = (O["isBarrel"] > 0) & (O["eta"] <= M.ETA_MATCHED)
    U = {"layer": np.r_[I["layer"], O["layer"][bar] + 10],
         "r": np.r_[I["globalR"], O["r"][bar]],
         "z": np.r_[I["globalZ"], O["z"][bar]],
         "phi": np.r_[I["globalPhi"], O["phi"][bar]],
         "tpIdx": np.r_[I["tpIdx"], O["tpIdx"][bar]],
         "tpPt": np.r_[I["tpPt"], O["tpPt"][bar]],
         "event": np.r_[I["event"], O["event"][bar]]}
    return U, n, on


def first_per_tp(U, L, ptmin):
    m = (U["layer"] == L) & (U["tpIdx"] >= 0) & (U["tpPt"] >= ptmin)
    idx = np.flatnonzero(m)
    k = M.tp_key(U["event"][idx], U["tpIdx"][idx])
    o = np.argsort(k, kind="stable")
    k, idx = k[o], idx[o]
    f = np.r_[True, k[1:] != k[:-1]]
    return k[f], idx[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--min-triples", type=int, default=500,
                    help="skip a (pair, target) with fewer truth-matched triples")
    ap.add_argument("-o", "--out", default="eval_refitq/combinatorics/results/"
                                           "seed_working_points.json")
    ap.add_argument("--settings", default="eval_refitq/combinatorics/results/"
                                          "spix_seed_settings.py")
    a = ap.parse_args()

    U, nev, onev = build(a.input, a.nev)
    FP = {L: first_per_tp(U, L, a.ptmin) for L in ALL_LAYERS}
    occ = {L: float((U["layer"] == L).sum()) / nev for L in ALL_LAYERS}
    zspan = {L: float(np.ptp(U["z"][U["layer"] == L])) for L in ALL_LAYERS}
    rmed = {L: float(np.median(U["r"][U["layer"] == L])) for L in ALL_LAYERS}
    print(f"{nev} events, pT > {a.ptmin}.  layers: "
          + ", ".join(f"{NAME[L]}@{rmed[L]:.1f}cm/{occ[L]:.0f}per-ev" for L in ALL_LAYERS))

    rows = []
    for i, la in enumerate(ALL_LAYERS):
        for lb in ALL_LAYERS[i + 1:]:
            ka, ia = FP[la]; kb, ib = FP[lb]
            com_ab = np.intersect1d(ka, kb)
            if len(com_ab) < a.min_triples:
                continue
            for lc in ALL_LAYERS:
                if lc in (la, lb):
                    continue
                kc, ic = FP[lc]
                com = np.intersect1d(com_ab, kc)
                if len(com) < a.min_triples:
                    continue
                ga = ia[np.searchsorted(ka, com)]
                gb = ib[np.searchsorted(kb, com)]
                gc = ic[np.searchsorted(kc, com)]
                dr = U["r"][gb] - U["r"][ga]
                ok = np.abs(dr) > 0.5
                if ok.sum() < a.min_triples:
                    continue
                kap = M.wrap(U["phi"][ga] - U["phi"][gb]) / (M.C_BEND * np.where(ok, dr, 1e9))
                cot = (U["z"][gb] - U["z"][ga]) / np.where(ok, dr, 1e9)
                z0 = U["z"][ga] - U["r"][ga] * cot
                phi0 = M.wrap(U["phi"][ga] + M.C_BEND * U["r"][ga] * kap)
                rc = U["r"][gc]
                drphi = np.abs(M.wrap(U["phi"][gc] - M.wrap(phi0 - M.C_BEND * rc * kap))) * rc
                dz = np.abs(U["z"][gc] - (z0 + rc * cot))
                g = ok & (np.abs(kap) <= 1.0 / a.ptmin)
                if g.sum() < a.min_triples:
                    continue
                wr_ot = float(np.percentile(drphi[g], Q_RPHI_OT))
                wz_ot = float(np.percentile(dz[g], Q_Z_OT))
                wr_an = N_SIGMA_ANALYTIC * float(np.percentile(drphi[g], Q_CORE))
                wz_an = N_SIGMA_ANALYTIC * float(np.percentile(dz[g], Q_CORE))
                # candidates in each window: occupancy x window / full extent
                def cand(wr, wz):
                    fphi = min(2 * wr / rmed[lc] / (2 * np.pi), 1.0)
                    fz = min(2 * wz / zspan[lc], 1.0)
                    return occ[lc] * min(fphi, fz), occ[lc] * fphi * fz
                c_ot, j_ot = cand(wr_ot, wz_ot)
                c_an, j_an = cand(wr_an, wz_an)
                rows.append({
                    "seed": f"{NAME[la]}+{NAME[lb]}", "target": NAME[lc],
                    "la": la, "lb": lb, "lc": lc, "n_triples": int(g.sum()),
                    "acceptance_tp_per_event": len(com_ab) / nev,
                    "dr_cm": float(np.median(dr)),
                    "sig_kappa": float(np.percentile(np.abs(np.abs(kap[g])
                                       - 1.0 / U["tpPt"][ga][g]), 68.27)),
                    "ot_like": {"rphi_cm": wr_ot, "z_cm": wz_ot,
                                "best_search": c_ot, "joint": j_ot,
                                "contain_rphi": float((drphi[g] < wr_ot).mean()),
                                "contain_z": float((dz[g] < wz_ot).mean())},
                    "analytic": {"rphi_cm": wr_an, "z_cm": wz_an,
                                 "best_search": c_an, "joint": j_an,
                                 # what the window ACHIEVES, not what it assumes
                                 "contain_rphi": float((drphi[g] < wr_an).mean()),
                                 "contain_z": float((dz[g] < wz_an).mean())}})
    res = {"n_events": nev, "pt_min": a.ptmin,
           "quantiles": {"rphi_ot_like": Q_RPHI_OT, "z_ot_like": Q_Z_OT,
                         "analytic_core": Q_CORE,
                         "analytic_nsigma": N_SIGMA_ANALYTIC},
           "layers": {NAME[L]: {"r_cm": rmed[L], "obj_per_event": occ[L],
                                "z_span_cm": zspan[L]} for L in ALL_LAYERS},
           "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n{len(rows)} (seed pair, target) combinations measured -> {a.out}")
    write_settings(res, a.settings)
    report(res)


def write_settings(res, path):
    """A settings file carrying BOTH working points, keyed (seed, target)."""
    with open(path, "w") as fh:
        fh.write('"""Seed match windows, MEASURED, at two working points.\n\n'
                 "Generated by seed_working_points.py -- do not hand-edit.\n\n"
                 "OT_LIKE   reproduces the Phase-2 OT tracklet's posture: the rphi\n"
                 f"          window contains {res['quantiles']['rphi_ot_like']:.0f}%"
                 " of correct projections and the z window\n"
                 f"          {res['quantiles']['z_ot_like']:.0f}%, matching what"
                 " Settings.h achieves. Tight in the bending\n"
                 "          plane, loose longitudinally.\n"
                 "ANALYTIC  "
                 f"{res['quantiles']['analytic_nsigma']:.0f}x the robust half-width"
                 " (core 3 sigma), sized to KEEP tracks.\n          The\n"
                 "          difference between the two IS the efficiency being"
                 " traded for\n          combinatorics.\n\n"
                 "Both are quantiles of a measured residual, not N*sigma: the\n"
                 "residuals have tails, and a Gaussian model of them was measured\n"
                 "optimistic by 2.0x in z and pessimistic by 2.2x in phi.\n"
                 f'\n{res["n_events"]} PU200 ttbar events, pT > {res["pt_min"]} GeV.\n"""\n\n')
        for wp in ("ot_like", "analytic"):
            fh.write(f"# (seed, target) -> (rphi window [cm], z window [cm])\n")
            fh.write(f"{wp.upper()} = {{\n")
            for r in res["rows"]:
                fh.write(f'    ("{r["seed"]}", "{r["target"]}"): '
                         f'({r[wp]["rphi_cm"]:.4f}, {r[wp]["z_cm"]:.3f}),\n')
            fh.write("}\n\n")
    print(f"wrote {path}")


def report(res):
    rows = res["rows"]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)
    print(f"\nSEEDS RANKED by total search cost over every target, OT-like WP")
    print(f"{'seed':<16}{'dr':>7}{'acc TP/ev':>11}{'sig(kap)':>10}"
          f"{'targets':>9}{'OT-like':>10}{'analytic':>10}{'ratio':>7}")
    agg = []
    for s, rs_ in by_seed.items():
        o = sum(x["ot_like"]["best_search"] for x in rs_)
        an = sum(x["analytic"]["best_search"] for x in rs_)
        agg.append((o, s, rs_[0], len(rs_), an))
    for o, s, r0, nt, an in sorted(agg)[:14]:
        print(f"{s:<16}{r0['dr_cm']:>7.1f}{r0['acceptance_tp_per_event']:>11.1f}"
              f"{r0['sig_kappa']:>10.4f}{nt:>9d}{o:>10.1f}{an:>10.1f}{an/max(o,1e-9):>7.2f}")


if __name__ == "__main__":
    main()
