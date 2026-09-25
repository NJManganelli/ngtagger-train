"""Step 2: angles of TP-linked vs unlinked clusters.

- localCotAlpha/Beta distributions, hasAlpha/hasBeta rates, sigAlpha/sigBeta
- coverage vs the NN training range (Conv1D_Full-2bit eval dump) and payload binning
- "valid-looking" rates under origin-compatibility windows, with the 1.0/1.5/2.0x
  angle-resolution inflation (post-hoc: linked angle' = true + k*(reco - true), sigma' = k*sigma;
  unlinked keep their inclusive draw, sigma' = k*sigma)
- discreteness tell of the noise-v6 inverse-CDF draw

Writes results/angles.json, results/angles_*.md, figs/f2_*.png.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as c

RES = c.HERE / "results"
RES.mkdir(exist_ok=True)

Z_LUM = 15.0     # cm: +-3 sigma of an HL-LHC luminous region (sigma_z ~ 5 cm)
PT_MIN = 2.0     # GeV: the "pT > 2 GeV" alpha band
NSIG = 3.0
KS = (1.0, 1.5, 2.0)
SIZEY_EDGES = [1, 2, 3, 4, 5, 6, 8, 12]   # = SmartPixelsRecHitProducer kSizeYEdges


def sizey_bin(sy):
    return np.searchsorted(np.array(SIZEY_EDGES), sy, side="left")


def fit_geometry(d: pd.DataFrame) -> pd.DataFrame:
    """Per-module map local (cotA, cotB) -> global direction, fitted on the nano's own columns:
    cot(theta_dir) = sX * cotB / sqrt(1 + cotA^2),  phi_dir = phiN + sY * atan(cotA)."""
    ok = (d["hasAlpha"] == 1) & (d["hasBeta"] == 1) & (d["globalClusterCotTheta"] > -900) & \
         (d["localCotBeta"].abs() > 0.05) & (d["unlinked"] == 0)
    x = d.loc[ok, ["detId", "localCotAlpha", "localCotBeta", "globalClusterCotTheta", "globalClusterPhi"]]
    sx = np.sign(x["globalClusterCotTheta"] * x["localCotBeta"]).groupby(x["detId"]).mean()
    out = pd.DataFrame({"sX": np.sign(sx)})
    best = {}
    for sy in (1, -1):
        z = np.exp(1j * (x["globalClusterPhi"].to_numpy() - sy * np.arctan(x["localCotAlpha"].to_numpy())))
        m = pd.Series(z, index=x["detId"].to_numpy()).groupby(level=0).mean()
        best[sy] = m
    r_p, r_m = best[1].abs(), best[-1].abs()
    out["sY"] = np.where(r_p.reindex(out.index) > r_m.reindex(out.index), 1, -1)
    out["phiN"] = np.where(out["sY"] == 1, np.angle(best[1].reindex(out.index)),
                           np.angle(best[-1].reindex(out.index)))
    return out


def global_dir(d, geo, ca, cb):
    g = geo.reindex(d["detId"].to_numpy())
    cot_t = g["sX"].to_numpy() * cb / np.sqrt(1 + ca ** 2)
    phi = g["phiN"].to_numpy() + g["sY"].to_numpy() * np.arctan(ca)
    return cot_t, phi


def windows(d, cot_t, phi, sig_ct, sig_phi):
    r = d["globalR"].to_numpy()
    z0 = d["globalZ"].to_numpy() - r * cot_t
    sz0 = r * sig_ct
    dphi = np.angle(np.exp(1j * (phi - d["globalPhi"].to_numpy())))
    bend = np.arcsin(np.minimum(1.0, r * c.C_CURV / (2 * PT_MIN)))
    beta_strict = np.abs(z0) < Z_LUM
    beta_res = np.abs(z0) < Z_LUM + NSIG * sz0
    alpha_res = np.abs(dphi) < bend + NSIG * sig_phi
    return {"beta_strict": beta_strict, "beta": beta_res, "alpha": alpha_res,
            "both": beta_res & alpha_res}


def main():
    d = c.load()
    nev = d["evid"].nunique()
    res = {}
    u = d["unlinked"].to_numpy() == 1
    lk = ~u
    lay = d["layer"].to_numpy()

    # ---------- has* rates, with and without the noise payload
    rates = pd.DataFrame({
        "hasAlpha linked": d[lk].groupby("layer")["hasAlpha"].mean(),
        "hasBeta linked": d[lk].groupby("layer")["hasBeta"].mean(),
        "hasAlpha unlinked (noise v6)": d[u].groupby("layer")["hasAlpha"].mean(),
        "hasBeta unlinked (noise v6)": d[u].groupby("layer")["hasBeta"].mean(),
        "hasAlpha unlinked (no noiseSet: itot_tp nanos)": d[u].groupby("layer")["hasAlpha_T"].mean(),
    })
    (RES / "angles_has_rates.md").write_text(c.md(rates))
    res["has_rates"] = rates.reset_index().to_dict(orient="list")

    # ---------- sigma summaries
    sg = []
    for name, m in (("linked", lk), ("unlinked", u)):
        for L in (1, 2, 3, 4):
            s = d.loc[m & (lay == L) & (d["hasAlpha"] == 1), "sigAlpha"]
            b = d.loc[m & (lay == L) & (d["hasBeta"] == 1), "sigBeta"]
            sg.append({"class": name, "layer": L, "sigAlpha median": s.median(), "sigAlpha q90": s.quantile(.9),
                       "sigBeta median": b.median(), "sigBeta q90": b.quantile(.9),
                       # payload cotBeta bins beyond |1| are filled from only 851 source clusters
                       "has-beta frac with |reco cotBeta|>1": np.mean(np.abs(d.loc[b.index, "localCotBeta"]) > 1)})
    sg = pd.DataFrame(sg).set_index(["class", "layer"])
    (RES / "angles_sigmas.md").write_text(c.md(sg))
    res["sigmas"] = sg.reset_index().to_dict(orient="list")

    # ---------- coverage vs the training range
    true_a, true_b = d["tpLocalCotAlpha"].to_numpy(), d["tpLocalCotBeta"].to_numpy()
    reco_a, reco_b = d["localCotAlpha"].to_numpy(), d["localCotBeta"].to_numpy()
    cov = []

    def outside(v, rng):
        return (v < rng[0]) | (v > rng[1])

    for L in (1, 2, 3, 4, "all"):
        ml = lk if L == "all" else lk & (lay == L)
        mu = u if L == "all" else u & (lay == L)
        mlh = ml & (d["hasAlpha"].to_numpy() == 1) & (d["hasBeta"].to_numpy() == 1)
        cov.append({
            "layer": L,
            "linked TRUE alpha outside train": np.mean(outside(true_a[ml], c.TRAIN_ALPHA)),
            "linked TRUE |alpha|>0.6 (payload edge)": np.mean(np.abs(true_a[ml]) > 0.6),
            "linked TRUE beta outside train": np.mean(outside(true_b[ml], c.TRAIN_BETA)),
            "linked TRUE outside (either)": np.mean(outside(true_a[ml], c.TRAIN_ALPHA) | outside(true_b[ml], c.TRAIN_BETA)),
            "linked RECO beta outside train (has*)": np.mean(outside(reco_b[mlh], c.TRAIN_BETA)),
            "unlinked ASSIGNED alpha outside train": np.mean(outside(reco_a[mu], c.TRAIN_ALPHA)),
            "unlinked ASSIGNED beta outside train": np.mean(outside(reco_b[mu], c.TRAIN_BETA)),
            "unlinked ASSIGNED outside (either)": np.mean(outside(reco_a[mu], c.TRAIN_ALPHA) | outside(reco_b[mu], c.TRAIN_BETA)),
            "linked TRUE outside, per event": np.sum(outside(true_a[ml], c.TRAIN_ALPHA) | outside(true_b[ml], c.TRAIN_BETA)) / nev,
            "unlinked ASSIGNED outside, per event": np.sum(outside(reco_a[mu], c.TRAIN_ALPHA) | outside(reco_b[mu], c.TRAIN_BETA)) / nev,
        })
    cov = pd.DataFrame(cov).set_index("layer")
    (RES / "angles_coverage.md").write_text(c.md(cov.T))
    res["coverage"] = cov.reset_index().astype({"layer": str}).to_dict(orient="list")
    # coverage by truth pT class for linked
    ptc = pd.cut(d.loc[lk, "tpPt"], [0, 0.3, 1, 2, 1e9], labels=["<0.3", "0.3-1", "1-2", ">=2"])
    ob = pd.Series(outside(true_b[lk], c.TRAIN_BETA), index=d.index[lk])
    oa = pd.Series(outside(true_a[lk], c.TRAIN_ALPHA), index=d.index[lk])
    res["coverage_by_pt"] = {"beta_out": ob.groupby(ptc, observed=True).mean().to_dict(),
                             "alpha_out": oa.groupby(ptc, observed=True).mean().to_dict()}

    # ---------- valid-looking windows, with resolution inflation
    geo = fit_geometry(d)
    ct1, ph1 = global_dir(d, geo, reco_a, reco_b)
    okg = (d["globalClusterCotTheta"] > -900).to_numpy()
    closure_ct = np.nanquantile(np.abs(ct1[okg] - d["globalClusterCotTheta"].to_numpy()[okg]), [0.5, 0.999])
    closure_ph = np.nanquantile(np.abs(np.angle(np.exp(1j * (ph1[okg] - d["globalClusterPhi"].to_numpy()[okg])))), [0.5, 0.999])
    res["geometry_closure"] = {"cotTheta_absdiff_q50_q999": closure_ct.tolist(),
                               "phi_absdiff_q50_q999_rad": closure_ph.tolist()}
    has_both = (d["hasAlpha"].to_numpy() == 1) & (d["hasBeta"].to_numpy() == 1)
    prim = lk & (d["tpVr"].to_numpy() < 0.5) & (np.abs(d["tpVz"].to_numpy()) < Z_LUM)
    groups = {
        "linked pT>=2 primary": prim & (d["tpPt"].to_numpy() >= 2),
        "linked pT 1-2": lk & (d["tpPt"].to_numpy() >= 1) & (d["tpPt"].to_numpy() < 2),
        "linked pT<1": lk & (d["tpPt"].to_numpy() < 1),
        "linked secondary (vr>1cm)": lk & (d["tpVr"].to_numpy() > 1),
        "linked all": lk,
        "unlinked": u,
    }
    rows = []
    passmask = {}
    for k in KS:
        ca = np.where(lk, true_a + k * (reco_a - true_a), reco_a)
        cb = np.where(lk, true_b + k * (reco_b - true_b), reco_b)
        ct, ph = global_dir(d, geo, ca, cb)
        s_ct = k * d["sigGlobalClusterCotTheta"].to_numpy()
        s_ph = k * d["sigGlobalClusterPhi"].to_numpy()
        w = windows(d, ct, ph, s_ct, s_ph)
        for L in (1, 2, 3, 4, "all"):
            ml = np.ones(len(d), bool) if L == "all" else lay == L
            for gname, gm in groups.items():
                m = gm & ml & has_both
                n_all = np.sum(gm & ml)
                row = {"k": k, "layer": L, "group": gname, "N/ev": n_all / nev,
                       "has both angles": np.sum(m) / max(n_all, 1)}
                for wn, wv in w.items():
                    row[f"pass {wn}"] = np.sum(wv & m) / max(n_all, 1)
                row["pass both, per event"] = np.sum(w["both"] & m) / nev
                rows.append(row)
        passmask[k] = w["both"] & has_both
    vt = pd.DataFrame(rows)
    res["valid_looking"] = vt.astype({"layer": str}).to_dict(orient="list")
    (RES / "angles_valid_looking.md").write_text(
        "Windows: beta = |z0_implied| < 15 cm + 3*r*sig(cotTheta); beta_strict = |z0| < 15 cm;\n"
        "alpha = |phi_dir - phi_pos| < asin(r/2R(pT=2 GeV)) + 3*sig(phi); both = beta & alpha.\n"
        "Fractions are of ALL clusters in the group (a cluster without both angles fails).\n\n"
        + c.md(vt[vt.layer == "all"].set_index(["k", "group"]).drop(columns="layer"))
        + "\n\nPer layer, k = 1.0\n\n"
        + c.md(vt[(vt.layer != "all") & (vt.k == 1.0)].set_index(["layer", "group"]).drop(columns="k")))
    # purity of the passing population
    pur = []
    for k in KS:
        pm = passmask[k]
        for L in (1, 2, 3, 4, "all"):
            ml = np.ones(len(d), bool) if L == "all" else lay == L
            pur.append({"k": k, "layer": L, "passing clusters/ev": np.sum(pm & ml) / nev,
                        "unlinked share of passing": np.sum(pm & ml & u) / max(np.sum(pm & ml), 1),
                        "unlinked share of all": np.sum(ml & u) / np.sum(ml)})
    pur = pd.DataFrame(pur)
    (RES / "angles_passing_purity.md").write_text(c.md(pur.set_index(["k", "layer"])))
    res["passing_purity"] = pur.astype({"layer": str}).to_dict(orient="list")

    # ---------- discreteness of the noise draw (40-bin inverse CDF -> step function)
    syb = sizey_bin(d["sizeY"].to_numpy())
    key_a = pd.Series(list(zip(lay, syb, np.round(reco_a, 6))))
    tab_a = set(key_a[u])
    key_b = pd.Series(list(zip(lay, syb, np.round(reco_b, 6))))
    tab_b = set(key_b[u])
    in_a = key_a.isin(tab_a).to_numpy()
    in_b = key_b.isin(tab_b).to_numpy()
    res["discreteness"] = {
        "unique_unlinked_cotAlpha": int(d.loc[u, "localCotAlpha"].nunique()),
        "unique_unlinked_cotBeta": int(d.loc[u, "localCotBeta"].nunique()),
        "n_unlinked": int(u.sum()),
        "unique_linked_cotAlpha": int(d.loc[lk, "localCotAlpha"].nunique()),
        "linked_with_both_angles_on_noise_table_frac": float(np.mean(in_a[lk] & in_b[lk])),
        "unlinked_on_table_frac": float(np.mean(in_a[u] & in_b[u])),
    }

    # ---------- shape-angle links: sizeX vs |cotAlpha| and sizeY vs |cotBeta|
    lnk = []
    for name, m in (("linked", lk & has_both), ("unlinked", u)):
        xb = pd.cut(np.abs(reco_a[m]), [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 100])
        yb = pd.cut(np.abs(reco_b[m]), [0, 0.3, 0.6, 1, 1.5, 2.5, 4, 100])
        lnk.append(pd.Series(d["sizeX"].to_numpy()[m]).groupby(xb, observed=True).mean().rename(f"{name} <sizeX> vs |cotAlpha|"))
        lnk.append(pd.Series(d["sizeY"].to_numpy()[m]).groupby(yb, observed=True).mean().rename(f"{name} <sizeY> vs |cotBeta|"))
    (RES / "angles_shape_links.md").write_text(
        "\n".join(c.md(s.to_frame()) for s in lnk))
    # sign link between cotTheta and z (origin pointing)
    sgn = {}
    for name, m in (("linked", lk & has_both & (np.abs(d["globalZ"].to_numpy()) > 5)),
                    ("unlinked", u & (np.abs(d["globalZ"].to_numpy()) > 5))):
        sgn[name] = float(np.mean(np.sign(ct1[m]) == np.sign(d["globalZ"].to_numpy()[m])))
    res["sign_cotTheta_eq_sign_z_for_absz_gt5"] = sgn

    (RES / "angles.json").write_text(json.dumps(res, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)))

    # ---------- figures
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for j, L in enumerate((1, 2, 3, 4)):
        ml = lay == L
        ax = axs[0, j]
        bins = np.linspace(-1.5, 1.5, 121)
        ax.hist(true_a[lk & ml], bins, histtype="step", density=True, color="k", label="linked, TRUE")
        ax.hist(reco_a[lk & ml & has_both], bins, histtype="step", density=True, color="#1f77b4", label="linked, reco")
        ax.hist(reco_a[u & ml], bins, histtype="step", density=True, color="#d62728", lw=1.6, label="unlinked, synthetic (noise v6)")
        ax.axvspan(*c.TRAIN_ALPHA, color="g", alpha=0.08, label="NN eval-sample coverage")
        ax.axvline(-0.6, color="g", ls=":"); ax.axvline(0.6, color="g", ls=":")
        ax.set_yscale("log"); ax.set_xlabel("local cot(alpha) [unitless]"); ax.set_ylabel("density [1/unit cot]")
        ax.set_title(f"TBPX L{L}: alpha (r-phi bending)")
        ax = axs[1, j]
        bins = np.linspace(-8, 8, 161)
        ax.hist(true_b[lk & ml], bins, histtype="step", density=True, color="k", label="linked, TRUE")
        ax.hist(reco_b[lk & ml & has_both], bins, histtype="step", density=True, color="#1f77b4", label="linked, reco")
        ax.hist(reco_b[u & ml], bins, histtype="step", density=True, color="#d62728", lw=1.6, label="unlinked, synthetic (noise v6)")
        ax.axvspan(*c.TRAIN_BETA, color="g", alpha=0.08, label="NN eval-sample coverage")
        ax.set_yscale("log"); ax.set_xlabel("local cot(beta) [unitless]"); ax.set_ylabel("density [1/unit cot]")
        ax.set_title(f"TBPX L{L}: beta (along z)")
    axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
    fig.suptitle(f"Angle distributions, {nev} PU200 events; green band = Conv1D_Full NN eval-sample truth range, dotted = payload alpha bin edge +-0.6")
    fig.tight_layout(); fig.savefig(c.FIGS / "f2_angle_distributions.png", dpi=100)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.linspace(-60, 60, 121)
    for name, m, col in (("linked pT>=2 primary", groups["linked pT>=2 primary"] & has_both, "#1f77b4"),
                         ("linked pT<1", groups["linked pT<1"] & has_both, "#2ca02c"),
                         ("unlinked (synthetic angle)", u, "#d62728")):
        z0 = d["globalZ"].to_numpy()[m] - d["globalR"].to_numpy()[m] * ct1[m]
        axs[0].hist(z0, bins, histtype="step", density=True, color=col, label=name)
        r = d["globalR"].to_numpy()[m]
        dph = np.angle(np.exp(1j * (ph1[m] - d["globalPhi"].to_numpy()[m])))
        bend = np.arcsin(np.minimum(1, r * c.C_CURV / (2 * PT_MIN)))
        axs[1].hist(dph / bend, np.linspace(-10, 10, 161), histtype="step", density=True, color=col, label=name)
    axs[0].axvspan(-Z_LUM, Z_LUM, color="grey", alpha=0.15, label="|z0| < 15 cm")
    axs[0].set_xlabel("implied origin z0 = z - r cot(theta_dir) [cm]"); axs[0].set_ylabel("density [1/cm]")
    axs[1].axvspan(-1, 1, color="grey", alpha=0.15, label="pT > 2 GeV band (before +3 sigma)")
    axs[1].set_xlabel("(phi_dir - phi_pos) / asin(r / 2R[pT = 2 GeV]) [unitless]"); axs[1].set_ylabel("density [1/unit]")
    for ax in axs:
        ax.legend(fontsize=8); ax.set_yscale("log")
    fig.suptitle("Origin-compatibility of the reported angle (k = 1.0)")
    fig.tight_layout(); fig.savefig(c.FIGS / "f2_origin_windows.png", dpi=110)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, m, col in (("linked", lk, "#1f77b4"), ("unlinked", u, "#d62728")):
        axs[0].hist(d.loc[m & (d["hasAlpha"].to_numpy() == 1), "sigAlpha"], np.linspace(0, 0.08, 81), histtype="step", density=True, color=col, label=name)
        axs[1].hist(d.loc[m & (d["hasBeta"].to_numpy() == 1), "sigBeta"], np.linspace(0, 0.15, 76), histtype="step", density=True, color=col, label=name)
    axs[0].set_xlabel("sigAlpha [unitless, cot]"); axs[1].set_xlabel("sigBeta [unitless, cot]")
    for ax in axs:
        ax.set_ylabel("density [1/unit cot]"); ax.legend(); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(c.FIGS / "f2_sigmas.png", dpi=110)

    print(c.md(rates)); print(c.md(cov.T)); print(c.md(vt[vt.layer == "all"].set_index(["k", "group"]).drop(columns="layer")))
    print(c.md(pur.set_index(["k", "layer"])))
    print(json.dumps({k: res[k] for k in ("discreteness", "geometry_closure", "sign_cotTheta_eq_sign_z_for_absz_gt5", "coverage_by_pt")}, indent=1, default=str))
    print("\n".join(c.md(s.to_frame()) for s in lnk))


if __name__ == "__main__":
    main()
