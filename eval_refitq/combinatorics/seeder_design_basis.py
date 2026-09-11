"""The measurements the seeder's design constants rest on, so they can be redone.

Every number here was originally measured ad hoc while deciding a constant in
tracklet_topology_cost.py, which means it could not be re-checked when the inputs
changed. Each study below states which constant it justifies. Run one with
--study NAME or all of them by default.

  geometry           per-layer occupancy, radial structure, z density
                     -> justifies N_RBINS = 48 and the z search window
  sigma-z0           where sigma(z0) comes from and why it differs by layer
                     -> justifies the PER-LAYER wildcard split
  selectivity        phi vs z0 vs joint search-key selectivity
                     -> justifies pairing on (z0, phi) jointly
  pt-threshold       kappa_slack against kappa_max, and the effective pT floor
                     -> justifies beamline-constrained pair seeding
  firmware-rom       can the three-point solve use layer-constant coefficients?
                     -> justifies quantising radius to N_RBINS in firmware
  sigma-provenance   is the published angle sigma a model output or a lookup,
                     and is it keyed on reco or true angles?
                     -> guards against presumptions that cannot go in hardware
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import triplet_d0_solve as T3        # noqa: E402

IT = M.IT_TABLE
LAYERS = (1, 2, 3, 4)


def _flat(path, cols, nev):
    """One or many input files; see tracklet_topology_cost.expand_inputs."""
    return M.load_flat(path, IT, cols, nev)


# ---------------------------------------------------------------- geometry ---
def study_geometry(a):
    """Occupancy and geometry per layer. Sets N_RBINS and the z window.

    A layer is NOT a thin shell: the radial spread is ~1.1 cm of staggered and
    tilted ladders. A z search at one reference radius must be padded by
    (spread/2)*|cot|, which swamps the ~232 um resolution window -- so the layer
    has to be binned in radius until the padding drops below it.
    """
    D, nev, ncl = _flat(a.input, ["layer", "globalR", "globalZ"], a.nev)
    print(f"=== geometry   {a.input.split('/')[-1]}  {nev} events\n")
    print(f"{'layer':<7}{'clusters/ev':>12}{'r med':>8}{'r ptp':>8}"
          f"{'z span':>8}{'cl per cm z':>12}{'r bin @48':>10}{'pad@|cot|1.16':>15}")
    rows = {}
    for L in LAYERS:
        m = D["layer"] == L
        r, z = D["globalR"][m], D["globalZ"][m]
        span = float(np.ptp(z)); ptp = float(np.ptp(r))
        dens = m.sum() / nev / span
        binw = ptp / M.N_RBINS
        pad = 0.5 * binw * 1.16
        rows[L] = dict(clusters_per_event=m.sum() / nev, r_median=float(np.median(r)),
                       r_ptp=ptp, z_span=span, clusters_per_cm_z=dens,
                       r_bin_cm=binw, pad_um=pad * 1e4)
        print(f"L{L:<6}{m.sum()/nev:>12.0f}{np.median(r):>8.2f}{ptp:>8.3f}"
              f"{span:>8.1f}{dens:>12.1f}{binw*1e4:>9.0f}u{pad*1e4:>14.0f}u")
    print(f"\nclusters/event total = {len(D['layer'])/nev:.0f}"
          f" +- {ncl.std():.0f} (event-to-event RMS)")
    print(f"3-sigma z window from projection_residuals ~232 um: the pad above must"
          f" sit under it, which is why N_RBINS = {M.N_RBINS}")
    return rows


# ---------------------------------------------------------------- sigma-z0 ---
def study_sigma_z0(a):
    """Where sigma(z0) comes from. Justifies the per-layer wildcard split.

    s_z0 = globalR * sigGlobalClusterCotTheta * SIG_PULL_INFLATE, so it is a
    propagated per-cluster uncertainty, not a measured spread. sigma(cot theta) is
    flat across layers, so the whole per-layer difference is LEVER ARM: z0
    extrapolates to r = 0 and the outer layers sit further away. A single global
    percentile on s_z0 is therefore a disguised cut on radius.
    """
    D, nev, _ = _flat(a.input, list(M.IT_COLS), a.nev)
    Q = M.it_prepare(D, None)
    s, sct, r, lay = Q["s_z0"], D["sigGlobalClusterCotTheta"], D["globalR"], D["layer"]
    gsplit = float(np.percentile(s, M.Z0_SPLIT_Q))
    print(f"=== sigma-z0   s_z0 = globalR * sigGlobalClusterCotTheta"
          f" * {M.SIG_PULL_INFLATE}\n")
    print(f"{'layer':<7}{'r med':>8}{'sig(cot) med':>14}{'s_z0 med':>11}"
          f"{'s_z0 q99':>10}{'share of global-q99 wildcards':>31}")
    nw = int((s > gsplit).sum())
    rows = {}
    for L in LAYERS:
        m = lay == L
        share = 100 * ((s > gsplit) & m).sum() / max(nw, 1)
        rows[L] = dict(r_median=float(np.median(r[m])),
                       sig_cot_median=float(np.median(sct[m])),
                       s_z0_median=float(np.median(s[m])),
                       s_z0_q99=float(np.percentile(s[m], 99)),
                       global_wildcard_share_pct=share)
        print(f"L{L:<6}{np.median(r[m]):>8.2f}{np.median(sct[m]):>14.5f}"
              f"{np.median(s[m]):>11.4f}{np.percentile(s[m],99):>10.3f}{share:>30.1f}%")
    print(f"\nsigma(cot theta) is FLAT across layers, so s_z0 tracks r: L4/L1 median"
          f" ratio = {np.median(s[lay==4])/np.median(s[lay==1]):.1f}x"
          f" against r ratio {np.median(r[lay==4])/np.median(r[lay==1]):.1f}x.")
    print("A GLOBAL percentile would put most wildcards in the outer layers for a"
          " purely geometric reason, which is why the split is per layer.")
    return rows


# ------------------------------------------------------------- selectivity ---
def study_selectivity(a):
    """phi vs z0 vs joint search-key selectivity. Justifies joint pairing.

    The pair stage used to search phi and then discard ~89% at the z0 gate.
    Searching both at once forms only the cluster pairs that satisfy both.
    """
    D, nev, _ = _flat(a.input, list(M.IT_COLS), a.nev)
    Q = M.it_prepare(D, None)
    print(f"=== selectivity   {nev} events\n")
    out = {}
    for (la, lb) in ((1, 2), (2, 3), (3, 4)):
        ma, mb = D["layer"] == la, D["layer"] == lb
        nA, nB = ma.sum() / nev, mb.sum() / nev
        rA, rB = float(np.median(D["globalR"][ma])), float(np.median(D["globalR"][mb]))
        dr = abs(rB - rA)
        kmax = 1.0 / a.ptmin
        ms = (M.THETA_MS_MRAD * 1e-3 / a.ptmin) * np.sqrt(2.0)
        half = M.C_BEND * dr * kmax + ms + 5e-4        # d0 = 0: beamline-constrained
        sel_phi = 2 * half / (2 * np.pi)
        sw = 2 * M.NSIG * float(np.median(Q["s_z0"][mb])) * np.sqrt(2.0)
        zr = float(np.percentile(Q["z0"], 99) - np.percentile(Q["z0"], 1))
        sel_z0 = sw / zr
        print(f"L{la}L{lb}: {nA:.0f} x {nB:.0f} = {nA*nB/1e6:.1f}M possible cluster pairs/event")
        for nm, sel in (("phi only", sel_phi), ("z0 only", sel_z0),
                        ("z0 AND phi", sel_phi * sel_z0)):
            print(f"    {nm:<12} selectivity {sel:>9.4%} -> {nA*nB*sel:>12,.0f} cluster pairs/event")
        out[f"L{la}L{lb}"] = dict(sel_phi=sel_phi, sel_z0=sel_z0,
                                  sel_joint=sel_phi * sel_z0,
                                  pairs_phi=nA * nB * sel_phi,
                                  pairs_joint=nA * nB * sel_phi * sel_z0)
    return out


# ------------------------------------------------------------ pt-threshold ---
def study_pt_threshold(a):
    """kappa_slack against kappa_max. Justifies beamline-constrained pairs.

    A pair solves curvature only by ASSUMING d0 = 0, so a real d0 biases it by
    d0*(1/rA - 1/rB)/(c*dr). Opening the curvature gate to keep such tracks also
    admits genuinely softer tracks, so the pT threshold silently drops.
    """
    D, nev, _ = _flat(a.input, ["layer", "globalR"], a.nev)
    med = {L: float(np.median(D["globalR"][D["layer"] == L])) for L in LAYERS}
    print(f"=== pt-threshold   layer median radii "
          f"{', '.join(f'L{L}={med[L]:.3f}' for L in LAYERS)} cm\n")
    rows = []
    print(f"{'pair':<7}{'d0 [um]':>9}{'kap_bias/cm':>13}{'kap_slack':>11}"
          f"{'kap_max':>9}{'kap cut':>9}{'pT label':>10}{'pT EFFECTIVE':>14}")
    for (la, lb) in ((1, 2), (2, 3), (3, 4)):
        rA, rB = med[la], med[lb]
        bias = abs(1 / rA - 1 / rB) / (M.C_BEND * max(abs(rB - rA), 0.1))
        for pt in (2.0, 1.5, 1.0, 0.5):
            for d0um in (0.0, 500.0):
                slack = bias * d0um * 1e-4
                kmax = 1.0 / pt
                cut = kmax + slack
                rows.append(dict(pair=f"L{la}L{lb}", pt_label=pt, d0_um=d0um,
                                 kappa_bias_per_cm=bias, kappa_slack=slack,
                                 pt_effective=1.0 / cut))
                print(f"L{la}L{lb}  {d0um:>9.0f}{bias:>13.2f}{slack:>11.3f}"
                      f"{kmax:>9.3f}{cut:>9.3f}{pt:>10.2f}{1.0/cut:>14.2f}")
    print("\nAt d0 = 500 um the slack EXCEEDS kappa_max at 2 GeV, so the threshold is"
          " destroyed outright. d0 = 0 is what makes it real.")
    return rows


# ------------------------------------------------------------ firmware-rom ---
def study_firmware_rom(a):
    """Can the three-point solve use layer-constant coefficients? Measured.

    The solve's coefficients depend only on the three radii, so precomputing them
    is what makes it cheap. The question is the granularity: one set per layer
    triple, or one per radial bin combination.
    """
    D, nev, _ = T3.load(a.input, a.nev), None, None
    D = D[0] if isinstance(D, tuple) else D
    sel = D["tpPt"] >= a.ptmin
    d0t_all = (-D["tpVx"] * np.sin(D["tpPhi"]) + D["tpVy"] * np.cos(D["tpPhi"]))
    print(f"=== firmware-rom   pT > {a.ptmin} GeV\n")
    print(f"{'config':<9}{'radii used':<24}{'sigma(d0) core':>15}{'sigma(kappa)':>14}")
    out = {}
    for (la, lb, lc) in ((1, 2, 3), (2, 3, 4)):
        gA, gB, gC = T3.correct_triples(D, la, lb, lc, sel)
        if not len(gA):
            continue
        R = [D["globalR"][g] for g in (gA, gB, gC)]
        P = [D["globalPhi"][g] for g in (gA, gB, gC)]
        d0t, kt = d0t_all[gA], 1.0 / D["tpPt"][gA]
        variants = {"per-cluster actual": R,
                    "layer medians": [np.full_like(R[0], np.median(x)) for x in R]}
        q = []
        for x in R:
            e = np.linspace(x.min(), x.max() + 1e-9, M.N_RBINS + 1)
            b = np.clip(np.searchsorted(e, x, "right") - 1, 0, M.N_RBINS - 1)
            q.append(0.5 * (e[b] + e[b + 1]))
        variants[f"medians, {M.N_RBINS} r-bins"] = q
        for nm, rr in variants.items():
            _, A, B, ok = T3.solve_triplet(rr[0], P[0], rr[1], P[1], rr[2], P[2])
            core = ok & (np.abs(d0t) < 50e-4) & (np.abs(A) < 1.0) & (np.abs(B) < 0.1)
            sd = T3.robust_sigma(A[core] - d0t[core]) * 1e4
            sk = T3.robust_sigma(np.abs(-B[core] / M.C_BEND) - kt[core])
            out[f"L{la}L{lb}L{lc}|{nm}"] = dict(sigma_d0_um=sd, sigma_kappa=sk)
            print(f"L{la}L{lb}L{lc}  {nm:<24}{sd:>13.0f}um{sk:>14.4f}")
    print("\nLayer-median coefficients are NOT sufficient; radial binning recovers the"
          " exact per-cluster result, and it is the binning the projection search"
          " already builds.")
    return out


# -------------------------------------------------------- sigma-provenance ---
def study_sigma_provenance(a):
    """Is the published angle sigma a model output or a lookup, and keyed how?

    A per-cluster sigma head would be unaffordable in hardware. A parametrised
    lookup on (layer, cotAlpha, cotBeta, bLocalY) is not. This confirms which one
    the samples actually carry, and whether it is keyed on reco or true angles --
    hardware only ever has reco.
    """
    if not a.payload or not Path(a.payload).exists():
        print("=== sigma-provenance  SKIPPED (no --payload given)")
        return {}
    cols = ["layer", "localCotAlpha", "localCotBeta", "tpLocalCotAlpha",
            "tpLocalCotBeta", "sigAlpha", "sigBeta"]
    D, nev, _ = _flat(a.input, cols, a.nev)
    pay = json.load(open(a.payload))
    print(f"=== sigma-provenance   payload {Path(a.payload).name}\n")
    out = {}
    sel = (D["sigBeta"] > 0) & (np.abs(D["tpLocalCotBeta"]) < 900)
    for nm in ("alpha", "beta"):
        c = next(x for x in pay["corrections"] if x["name"] == f"spix_angle_{nm}_sigma")
        tab, naxes = {}, None
        for it in c["data"]["content"]:
            mb = it["value"]
            edges = [np.asarray(e) for e in mb["edges"]]
            naxes = mb["inputs"]
            shape = tuple(len(e) - 1 for e in edges)
            tab[it["key"]] = (np.asarray(mb["content"]).reshape(shape), edges)
        nvals = sum(v.size for v, _ in tab.values())
        print(f"  spix_angle_{nm}_sigma: axes {naxes}, "
              f"{len(tab)} layers x {nvals//len(tab)} = {nvals} values TOTAL"
              f"  (a lookup, not a model head)")
        for ax, e in zip(naxes, tab[next(iter(tab))][1]):
            print(f"      {ax:<9}{len(e)-1:>3} bins  [{e[0]:>7.3g} .. {e[-1]:>7.3g}]")

        def look(ca, cb):
            r = np.full(sel.sum(), np.nan)
            for L, (v, e) in tab.items():
                m = D["layer"][sel] == L
                if not m.any():
                    continue
                ia = np.clip(np.searchsorted(e[0], ca[m], "right") - 1, 0, v.shape[0] - 1)
                ib = np.clip(np.searchsorted(e[1], cb[m], "right") - 1, 0, v.shape[1] - 1)
                r[m] = v[ia, ib, 0]
            return r
        st = D[f"sig{nm.capitalize()}"][sel]
        reco = look(D["localCotAlpha"][sel], D["localCotBeta"][sel])
        true = look(D["tpLocalCotAlpha"][sel], D["tpLocalCotBeta"][sel])
        fr, ft = (100 * np.isclose(st, reco, rtol=1e-3).mean(),
                  100 * np.isclose(st, true, rtol=1e-3).mean())
        out[nm] = dict(n_values=nvals, axes=naxes, match_reco_pct=fr, match_true_pct=ft)
        print(f"      stored sigma matches lookup at RECO angles {fr:.1f}%,"
              f" at TRUE angles {ft:.1f}%")
    print("\nHardware only has the reco angle, so matching at RECO is the affordable"
          " case. Any bLocalY axis with a single bin carries no y dependence.")
    return out


STUDIES = {"geometry": study_geometry, "sigma-z0": study_sigma_z0,
           "selectivity": study_selectivity, "pt-threshold": study_pt_threshold,
           "firmware-rom": study_firmware_rom,
           "sigma-provenance": study_sigma_provenance}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--payload", default=None,
                    help="PixelAV angle-response json, for sigma-provenance")
    ap.add_argument("--study", action="append", choices=sorted(STUDIES),
                    help="repeatable; default all")
    ap.add_argument("-o", "--out", default=None, help="write results as json")
    a = ap.parse_args()
    res = {}
    for nm in (a.study or sorted(STUDIES)):
        res[nm] = STUDIES[nm](a)
        print()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, default=float)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
