"""Closure: CMSSW angleSource="cnn" vs the python models on the same clusters.

  closure_cmssw.py NTUPLE.root [--barrel base] [--disc disc] [--out results/closure_cmssw.json]

NTUPLE is a ClusterAdcNtuple made with angleSource=cnn (its pCot*/pSig* hold the CMSSW
output). The python side rebuilds the input window from the SAME ntuple rows exactly as
prep.py does and runs the torch models on CPU in float32; the comparison is on the raw
(uninflated) cot.
"""
import argparse, os, json, numpy as np, pandas as pd, uproot, awkward as ak, torch
from models_def import C_LAYER, scalars, make

p = argparse.ArgumentParser()
p.add_argument("ntuple")
p.add_argument("--barrel", default="base")
p.add_argument("--disc", default="disc")
p.add_argument("--out", default="results/closure_cmssw.json")
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
WX, WY = 24, 16
t = uproot.open(a.ntuple)["clusterAdcNtuple/clusters"]
cols = ["event", "clIdx", "subdet", "layer", "disk", "sizeX", "sizeY", "size", "charge", "pCotAlpha", "pCotBeta", "pSigAlpha",
        "pSigBeta", "pHasAlpha", "pHasBeta", "linkClass"]
arr = t.arrays(cols + ["pxRow", "pxCol", "pxAdc"], library="ak")
m = pd.DataFrame({k: ak.to_numpy(arr[k]) for k in cols})
n = len(m)
sx, sy = m.sizeX.values, m.sizeY.values
offx, offy = (WX - sx) // 2, (WY - sy) // 2
ci = np.repeat(np.arange(n), ak.to_numpy(ak.num(arr["pxRow"])))
r = ak.to_numpy(ak.flatten(arr["pxRow"])).astype(np.int32) + offx[ci]
c = ak.to_numpy(ak.flatten(arr["pxCol"])).astype(np.int32) + offy[ci]
adc = ak.to_numpy(ak.flatten(arr["pxAdc"])).astype(np.int32)
ok = (r >= 0) & (r < WX) & (c >= 0) & (c < WY)
img = np.zeros((n, WX, WY), np.uint8)
img[ci[ok], r[ok], c[ok]] = np.clip(adc[ok] + 1, 1, 255)
m["nPxLost"] = np.bincount(ci[~ok], minlength=n)
m["truncX"] = (sx > WX).astype(np.int8); m["truncY"] = (sy > WY).astype(np.int8)
m["adcSum"] = np.bincount(ci, weights=adc + 1, minlength=n)

out = {}
for region, sel, tag in [("barrel", m.subdet.values == 1, a.barrel), ("disc", m.subdet.values == 2, a.disc)]:
    if sel.sum() == 0:
        continue
    nd = json.load(open(os.path.join(HERE, "models", tag, "norm.json")))
    net = make(nd["variant"], len(nd["cols"])); net.load_state_dict(torch.load(os.path.join(HERE, "models", tag, "model.pt"), map_location="cpu")); net.eval()
    mm = m[sel]
    S = ((scalars(mm, nd["variant"]) - pd.Series(nd["mu"])) / pd.Series(nd["sd"])).values.astype(np.float32)
    with torch.no_grad():
        o = np.concatenate([net(torch.from_numpy(img[sel][i:i + 8192]), torch.from_numpy(S[i:i + 8192])).numpy()
                            for i in range(0, sel.sum(), 8192)])
    cA = (np.where(o[:, 2] > 0, 1, -1) * np.tan(o[:, 0]) - C_LAYER[mm.layer.values]).astype(np.float32)
    cB = (np.where(o[:, 3] > 0, 1, -1) * np.tan(o[:, 1])).astype(np.float32)
    res = {"model": tag, "n": int(sel.sum())}
    for nm, py, cm in [("cotAlpha", cA, mm.pCotAlpha.values), ("cotBeta", cB, mm.pCotBeta.values)]:
        # "reported": |cot| <= 12, the producer's measAngleMaxAbs; beyond it tan() amplifies a
        # 1-ulp difference in the magnitude head near pi/2 and the angle is flagged off anyway
        for sub, s_ in [("all", np.ones(len(py), bool)), ("reported", np.abs(py) <= 12)]:
            d = np.abs(py - cm)[s_]; rel = d / np.maximum(np.abs(py[s_]), 1e-3)
            res[f"{nm}_{sub}"] = dict(n=int(s_.sum()), max_abs=float(d.max()), p999_abs=float(np.quantile(d, 0.999)),
                                      median_abs=float(np.median(d)), max_rel=float(rel.max()),
                                      frac_sign_differs=float(np.mean(np.sign(py[s_]) != np.sign(cm[s_]))),
                                      frac_abs_gt_1e5=float(np.mean(d > 1e-5)))
    for nm in ["pHasAlpha", "pHasBeta"]:
        res[f"{nm}_linked"] = float(mm[nm][mm.linkClass == 2].mean())
        res[f"{nm}_unlinked"] = float(mm[nm][mm.linkClass != 2].mean())
    out[region] = res
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, a.out), "w"), indent=1)
