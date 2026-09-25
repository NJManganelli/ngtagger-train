"""Static CNN angle-resolution table for SmartPixelsRecHitProducer (angleSource = "cnn").

  sigma_table.py OUT.json [--barrel base] [--disc disc]

sigma = (q84 - q16) / 2 of (predicted - true) cot, the payload's robust width
(extract_pixelav_angle_payload.py), measured on the truth-consistent linked TEST clusters and
binned ONLY in what the sensor reports: detKey (TBPX layer 1-4, or 100 + disk for TFPX/TEPX
disks 1-12) and the |reported cot| of that angle. It is a resolution model, not a per-cluster
network output. Bins with < MIN_N clusters take the nearest populated bin of the same detKey.
Writes the correctionlib set (spix_cnn_alpha_sigma, spix_cnn_beta_sigma) and
results/cnn_sigma_table.csv.
"""
import argparse, os, numpy as np, pandas as pd
import correctionlib.schemav2 as cs

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--barrel", default="base")
p.add_argument("--disc", default="disc")
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.07, 1.5, 2, 3, 4, 6, 1e6]
MIN_N = 200

P = pd.read_parquet("/Volumes/WDMac/smartpixels_scratch/cluster_cnn_preds/predictions_test_all.parquet",
                    columns=["row", "subdet", "layer", "disk"] + [f"{m}_cot{x}" for m in {a.barrel, a.disc} for x in "AB"])
T = pd.read_parquet(os.path.join(HERE, "data_all/meta.parquet"), filters=[("split", "==", 2)],
                    columns=["row", "linkClass", "tpPt", "tpCharge", "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist"])
m = P.merge(T, on="row", validate="one_to_one")
m = m[(m.linkClass == 2) & (m.tpPt >= 0.25) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900) & (m.hxOk == 1) & (m.hxDist < 1.0)]
m["detKey"] = np.where(m.subdet == 2, 100 + m.disk, m.layer).astype(int)
KEYS = [1, 2, 3, 4] + [100 + d for d in range(1, 13)]

rows, corrs = [], []
for ang, nm in [("A", "alpha"), ("B", "beta")]:
    content = []
    for key in KEYS:
        mod = a.disc if key > 100 else a.barrel
        d = m[m.detKey == key]
        pc = d[f"{mod}_cot{ang}"].values; tc = d[f"tpLocalCot{'Alpha' if ang == 'A' else 'Beta'}"].values
        b = np.digitize(np.abs(pc), EDGES[1:-1])
        sig, cnt = np.full(len(EDGES) - 1, np.nan), np.zeros(len(EDGES) - 1, int)
        for i in range(len(EDGES) - 1):
            r = (pc - tc)[b == i]
            cnt[i] = len(r)
            if len(r) >= MIN_N:
                q16, q84 = np.quantile(r, [0.16, 0.84]); sig[i] = (q84 - q16) / 2
        pop = np.where(np.isfinite(sig))[0]
        if len(pop) == 0:
            raise SystemExit(f"no populated bin for detKey {key}")
        filled = np.array([sig[pop[np.argmin(np.abs(pop - i))]] for i in range(len(sig))])
        for i in range(len(sig)):
            rows.append(dict(angle=nm, detKey=key, model=mod, lo=EDGES[i], hi=EDGES[i + 1], n=int(cnt[i]),
                             sigma_measured=sig[i], sigma=filled[i]))
        content.append(cs.CategoryItem(key=key, value=cs.Binning(nodetype="binning", input="absCot", edges=EDGES,
                                                                  content=[round(float(v), 6) for v in filled], flow="clamp")))
    corrs.append(cs.Correction(
        name=f"spix_cnn_{nm}_sigma", version=1,
        description=(f"Cluster-CNN cot{nm.capitalize()} resolution: (q84-q16)/2 of (pred - true) cot on truth-consistent "
                     f"linked test clusters (barrel model '{a.barrel}', disc model '{a.disc}'), by detKey (TBPX layer, or "
                     f"100 + disk) and |reported cot|. Static resolution model; inputs are sensor-reported only."),
        inputs=[cs.Variable(name="detKey", type="int", description="TBPX layer 1-4, or 100 + pxfDisk (1-12)"),
                cs.Variable(name="absCot", type="real", description=f"|reported cot{nm.capitalize()}|")],
        output=cs.Variable(name="sigma", type="real", description=f"cot{nm.capitalize()} resolution"),
        data=cs.Category(nodetype="category", input="detKey", content=content)))
cset = cs.CorrectionSet(schema_version=2, description="SmartPixels cluster-CNN angle resolution (sigma) tables", corrections=corrs)
open(a.out, "w").write(cset.model_dump_json(exclude_unset=True, indent=1))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results/cnn_sigma_table.csv"), index=False)
print("wrote", a.out)
