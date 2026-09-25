"""Apply the base (sensor-local) CNN + the k=1.5 truth-free inflation to ALL clusters.

Output (WDMac): cluster_cnn_preds/cnn_angles_all_clusters.parquet, one row per cluster:
  event, clIdx (row within event in the ntuple), detId, layer, linkClass, split
  (0 train / 1 val / 2 test: train/val rows are IN-SAMPLE for the TP-linked pT>=0.25 subset),
  cnnCotAlpha/Beta, cnnSignProbAlpha/Beta, cnnLocalX/Y [cm, module-local],
  sigmaAlpha/BetaDeg (cell width), inflCotAlpha/Beta (k = 1.5).
Needs results/inflation_table_{fine,coarse,layer}.csv from evaluate.py (k = 1.5, calibrated widths).
"""
import os, json, numpy as np, pandas as pd, torch
from models_def import C_LAYER, BASE_COLS, scalars, make
import inflation as INF_

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_preds/cnn_angles_all_clusters.parquet"
torch.set_num_threads(4)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # NB: MPS Conv2d is WRONG at batch 65536 (verified vs CPU); keep batches <= 8192
cols = sorted(set(BASE_COLS + ["event", "clIdx", "detId", "linkClass", "split", "originLocalX", "originLocalY", "pitchX", "pitchY"]))
m = pd.read_parquet(os.path.join(HERE, "data/meta.parquet"), columns=cols)
imgs = np.load(os.path.join(HERE, "data/images.npy"), mmap_mode="r")
nd = json.load(open(os.path.join(HERE, "models/base/norm.json")))
S = ((scalars(m, "base") - pd.Series(nd["mu"])) / pd.Series(nd["sd"])).values.astype(np.float32)
net = make("base", S.shape[1]); net.load_state_dict(torch.load(os.path.join(HERE, "models/base/model.pt"))); net.to(dev).eval()
outs = []
with torch.no_grad():
    for i in range(0, len(m), 8192):
        X = torch.from_numpy(np.ascontiguousarray(imgs[i:i + 8192])).to(dev)
        outs.append(net(X, torch.from_numpy(S[i:i + 8192]).to(dev)).cpu().numpy())
o = np.concatenate(outs)
c = C_LAYER[m.layer.values]
res = pd.DataFrame({"event": m.event.values, "clIdx": m.clIdx.values, "detId": m.detId.values, "layer": m.layer.values,
                    "linkClass": m.linkClass.values, "split": m.split.values})
res["cnnCotAlpha"] = (np.where(o[:, 2] > 0, 1, -1) * np.tan(o[:, 0]) - c).astype(np.float32)
res["cnnCotBeta"] = (np.where(o[:, 3] > 0, 1, -1) * np.tan(o[:, 1])).astype(np.float32)
res["cnnSignProbAlpha"] = (1 / (1 + np.exp(-o[:, 2]))).astype(np.float32)
res["cnnSignProbBeta"] = (1 / (1 + np.exp(-o[:, 3]))).astype(np.float32)
res["cnnLocalX"] = (m.originLocalX + (o[:, 4] + m.sizeX / 2) * m.pitchX).astype(np.float32)
res["cnnLocalY"] = (m.originLocalY + (o[:, 5] + m.sizeY / 2) * m.pitchY).astype(np.float32)
T = {l: pd.read_csv(os.path.join(HERE, f"results/inflation_table_{l}.csv")) for l in ["fine", "coarse", "layer"]}
for ang, nm in [("A", "Alpha"), ("B", "Beta")]:
    pc = res[f"cnnCot{nm}"].values
    cell, cz, _ = INF_.cells(m.layer.values, pc, m.sizeX.values, m.sizeY.values)
    keys = {"fine": cell, "coarse": cz, "layer": m.layer.values}
    mp = lambda l, v: dict(zip(T[l].key[T[l].angle == ang], T[l][v][T[l].angle == ang]))
    sig = INF_.lookup([(keys[l], mp(l, "sigma_deg")) for l in ["fine", "coarse", "layer"]])
    add = INF_.lookup([(keys[l], mp(l, "add_deg")) for l in ["fine", "coarse", "layer"]])
    res[f"sigma{nm}Deg"] = sig.astype(np.float32)
    th = INF_.inflate(np.degrees(np.arctan(pc)), add, m.event.values, m.clIdx.values, ang)
    res[f"inflCot{nm}"] = np.tan(np.radians(th)).astype(np.float32)
res.to_parquet(OUT)
print("wrote", OUT, len(res), "nan sigma frac", float(np.isnan(res.sigmaAlphaDeg).mean()), float(np.isnan(res.sigmaBetaDeg).mean()))
