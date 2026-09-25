"""Internal truth validation of the ClusterAdcNtuple (no nano join).

1. Helix-propagated TP truth (producer convention = tpLocalCot*, and module-plane hx*) vs
   the PSimHit (signal crossing only) direction/position, by TP pT.
2. Label-consistency: fraction of TP-linked clusters whose TP helix never comes within
   1 mm / 1 cm of the hit (delta rays attributed to the parent TP, loopers), by pT/layer,
   and what the PSimHit says about them.
3. Sanity: cluster extent vs true angle (Lorentz dip position, sizeY slope = T/pitchY).
4. Window truncation per |cot| bin; B-field and Lorentz-angle ranges.
"""
import numpy as np, pandas as pd, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
m = pd.read_parquet(os.path.join(HERE, "data/meta.parquet"))
out = {}
smad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan
deg = lambda c: np.degrees(np.arctan(c))
L = (m.linkClass == 2) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900)
d = m.hxDist.where(m.hxOk == 1, 1e9)
print("clusters", len(m), "events", m.event.nunique(), "linkClass fractions", np.bincount(m.linkClass, minlength=3) / len(m))
out["linkClass_frac"] = (np.bincount(m.linkClass, minlength=3) / len(m)).tolist()
out["neutral_dominant_frac_linked"] = float(((m.linkClass == 2) & (m.tpCharge == 0)).sum() / (m.linkClass == 2).sum())

# 1+2: signal clusters with the dominant SimTrack's PSimHit
S = L & (m.shN > 0) & (m.domSimHasTp == 1) & (m.shCotAlpha > -900)
rows = []
for lo, hi in [(0.1, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 1e9)]:
    for dn, dc in [("all", d < 1e10), ("hxDist<1cm", d < 1), ("hxDist>=1cm", d >= 1)]:
        s = S & (m.tpPt >= lo) & (m.tpPt < hi) & dc
        if s.sum() < 20:
            continue
        r = dict(pt=f"{lo}-{hi}", sel=dn, n=int(s.sum()))
        for nm in ["CotAlpha", "CotBeta"]:
            dd = deg(m["tpLocal" + nm][s]) - deg(m["sh" + nm][s])
            r[nm + "_prod_vs_sim_smad_deg"] = smad(dd)
            r[nm + "_prod_vs_sim_q90abs_deg"] = float(np.percentile(np.abs(dd), 90))
        r["pdgMismatch"] = float((np.abs(m.tpPdgId[s]) != np.abs(m.shPdg[s])).mean())
        ok = s & (d < 1)
        if ok.sum() > 20:
            r["x_hx_vs_sim_smad_um"] = smad((m.hxLocalX[ok] - m.shLocalX[ok]) * 1e4)
            r["y_hx_vs_sim_smad_um"] = smad((m.hxLocalY[ok] - m.shLocalY[ok]) * 1e4)
        r["x_cpe_vs_sim_smad_um"] = smad((m.localX[s] - m.shLocalX[s]) * 1e4)
        r["y_cpe_vs_sim_smad_um"] = smad((m.localY[s] - m.shLocalY[s]) * 1e4)
        rows.append(r)
t1 = pd.DataFrame(rows)
print(t1.round(3).to_string())
out["truth_vs_psimhit"] = rows

rows = []
for lo, hi in [(0.1, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 1e9)]:
    s = L & (m.tpPt >= lo) & (m.tpPt < hi)
    r = dict(pt=f"{lo}-{hi}", n=int(s.sum()), helixFail=float((m.hxOk[s] == 0).mean()),
             far1mm=float((d[s] >= 0.1).mean()), far1cm=float((d[s] >= 1).mean()))
    for l in range(1, 5):
        r[f"far1cm_L{l}"] = float((d[s & (m.layer == l)] >= 1).mean())
    rows.append(r)
t2 = pd.DataFrame(rows); print(t2.round(4).to_string()); out["helix_consistency"] = rows

# 3: sanity -- extent vs angle (truth-consistent, pT >= 0.25)
G = L & (m.tpPt >= 0.25) & (d < 1)
cb = np.arange(-1.0, 1.01, 0.05)
for lay in [1, 2]:
    s = G & (m.layer == lay) & (m.sizeY <= 2)
    h = pd.Series(m.sizeX[s].values).groupby(pd.cut(m.tpLocalCotAlpha[s].values, cb)).mean()
    mn = h.idxmin()
    print(f"L{lay}: mean sizeX vs cotAlpha minimum at {mn} (mean {h.min():.2f})")
    out[f"sizeX_min_cotAlpha_bin_L{lay}"] = str(mn)
s = G & (np.abs(m.tpLocalCotBeta) < 6)
A = np.vstack([np.abs(m.tpLocalCotBeta[s]), np.ones(s.sum())]).T
slope = np.linalg.lstsq(A, m.sizeY[s].values, rcond=None)[0]
print("sizeY = %.3f*|cotBeta| + %.3f  (expect T/pitchY = %.3f)" % (slope[0], slope[1], 0.015 / 0.01))
out["sizeY_slope"] = slope.tolist()

# 4: truncation per |cot| bin (24x16 window), training-like selection
bins = [0, 0.5, 1.07, 2, 4, 1e9]
tr = {}
for nm, ax in [("tpLocalCotAlpha", "truncX"), ("tpLocalCotBeta", "truncY")]:
    c = np.abs(m[nm])
    tr[nm] = [float(m[ax][G & (c >= bins[i]) & (c < bins[i + 1])].mean()) for i in range(5)]
tr["anyPixelLost_all_clusters"] = float((m.nPxLost > 0).mean())
tr["anyPixelLost_G"] = float((m.nPxLost[G] > 0).mean())
print("truncation", json.dumps(tr)); out["truncation"] = tr
rng = {k: [float(m[k].min()), float(np.percentile(m[k], 1)), float(np.percentile(m[k], 99)), float(m[k].max())]
       for k in ["bLocalX", "bLocalY", "bLocalZ", "bMag", "bModLocalY", "tanLAx", "tanLAy", "laPerTesla", "laSimPerTesla"]}
rng["laSim_by_layer"] = {int(l): sorted(set(np.round(m.laSimPerTesla[m.layer == l].unique(), 5).tolist())) for l in range(1, 5)}
rng["is3D_by_layer"] = {int(l): float(m.is3D[m.layer == l].mean()) for l in range(1, 5)}
rng["thickness"] = sorted(m.thickness.unique().tolist()); rng["pitchX"] = sorted(m.pitchX.unique().tolist()); rng["pitchY"] = sorted(m.pitchY.unique().tolist())
print(json.dumps(rng, indent=0)); out["ranges"] = rng
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results/truth_validation.json"), "w"), indent=1)
