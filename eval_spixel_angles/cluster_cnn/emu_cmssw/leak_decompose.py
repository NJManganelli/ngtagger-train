"""Leak-test decomposition on one nano file: AUC(unlinked vs linked) gain of each angle column group over the no-angle probe,
for the emu nano (cnn_nano_v3) and, as the reference, the cnn_nano_v2 nano (the rejected factorised CNN alone) of the same events.
usage: leak_decompose.py V3.root V2.root  -> emu_cmssw/leak_decompose.json
Inputs must link TPs by tpIdx (unlinked = not SPN.tp_matched); the key-linked cnn_nano v2/v3 files are no longer readable."""
import sys, os, json, numpy as np, pandas as pd, uproot, awkward as ak
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from ngtagger.data import spix_nano as SPN
P = "L1TSmartPixelsCluster_"
BASE = ["sizeX", "sizeY", "size", "charge", "surface", "globalR", "globalZ", "localX", "localY", "sigX", "sigY"]
SETS = {"+cot": ["localCotAlpha", "localCotBeta"], "+cot+sig": ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta"],
        "+has": ["hasAlpha", "hasBeta"], "+all angle columns": ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta"]}
out = {}
for tag, f in (("emu (v3)", sys.argv[1]), ("cnn (v2, reference)", sys.argv[2])):
    cols = BASE + ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "tpIdx"]
    arr = uproot.open(f)["Events"].arrays([P + c for c in cols] + ["event"], library="ak")
    D = pd.DataFrame({c: ak.to_numpy(ak.flatten(arr[P + c])) for c in cols})
    D["event"] = np.repeat(ak.to_numpy(arr["event"]), ak.to_numpy(ak.num(arr[P + cols[0]])))
    y = (~SPN.tp_matched(D.tpIdx.values)).astype(int); half = D.event.values % 2 == 0
    grp = np.where(D.surface > 10, "ID1-4", np.where(D.surface == 1, "IL1", "IL2-4"))
    rng = np.random.default_rng(3); res = {}
    for g in ("IL2-4", "IL1", "ID1-4"):
        gm = grp == g; r = {}
        for name, extra in [("base", [])] + list(SETS.items()):
            X = D[BASE + extra].values.astype(np.float64); X[~np.isfinite(X)] = -999; X = np.clip(X, -1e6, 1e6)
            tri = np.where(half & gm)[0]; tri = rng.choice(tri, min(len(tri), 1_000_000), replace=False); tei = np.where(~half & gm)[0]
            clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1).fit(X[tri], y[tri])
            r[name] = round(float(roc_auc_score(y[tei], clf.predict_proba(X[tei])[:, 1])), 4)
        res[g] = {k: (v if k == "base" else round(v - r["base"], 4)) for k, v in r.items()}
        print(tag, g, res[g], flush=True)
    out[tag] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leak_decompose.json"), "w"), indent=1)
