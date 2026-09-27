"""Closure: CMSSW angleSource="emu" vs the python reference, on the same clusters.

  emu_closure_cmssw.py NTUPLE.root LIBRARY.root [--train-ntuples BARREL.root,DISC.root] [--out emu_cmssw/closure.json]

NTUPLE is a ClusterAdcNtuple of a job with angleSource=emu, emuDebug=True (branches emu*: the producer's CNN outputs,
class, full-scan hxDist, donor, pseudo-truth, dump row, biases, output angles; pCot*, pSat*). The python side
 1. rebuilds the CNN window and scalars from the SAME ntuple rows as prep.py / v2_common.derived do and runs the torch
    pos-hubsig models (CPU, float32)                                             -> CNN closure (float precision)
 2. classifies from the ntuple truth (charged dominant TP, pT >= 0.25, helix crossing within 1 cm)  -> class identity
 3. runs emu_common.emulate with the same library and hash keys, (a) on the CMSSW CNN values (the selection logic in
    isolation) and (b) on its own torch CNN values (end to end)                  -> donor / dump row / output identity
--train-ntuples: the ORIGINAL training ntuples (made before the helix code moved into SmartPixelsTpHelix.h); their
hxDist is compared with the producer's on the clusters both contain (event, detId, idxInDet)."""
import argparse, os, sys, json, numpy as np, pandas as pd, uproot, awkward as ak, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emu_common import load_net, normalise, run_net, decode, scalars_v2, load_library, emulate, PT_MIN, HXDIST_MAX, WX, WY
from v2_common import V2

p = argparse.ArgumentParser()
p.add_argument("ntuple"); p.add_argument("library")
p.add_argument("--train-ntuples", default="")
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "emu_cmssw", "closure.json"))
a = p.parse_args()
torch.set_num_threads(4)
t = uproot.open(a.ntuple)["clusterAdcNtuple/clusters"]
sc = [k for k in t.keys() if not k.startswith("px")]
arr = t.arrays(sc + ["pxRow", "pxCol", "pxAdc"], library="ak")
m = pd.DataFrame({k: ak.to_numpy(arr[k]) for k in sc})
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
m["nPxLost"] = np.bincount(ci[~ok], minlength=n); m["truncX"] = (sx > WX).astype(np.int8); m["truncY"] = (sy > WY).astype(np.int8)
m["adcSum"] = np.bincount(ci, weights=adc + 1, minlength=n)
m["surface"] = np.where(m.subdet == 1, m.layer, 10 + m.disk).astype(np.int64)
res = {"n": int(n), "surfaces": {int(k): int(v) for k, v in m.surface.value_counts().sort_index().items()}}

# 1. CNN
pyA, pyB, pySA, pySB = (np.full(n, np.nan) for _ in range(4))
for reg, sel in (("barrel", m.subdet.values == 1), ("disc", m.subdet.values == 2)):
    net, nd = load_net(os.path.join(V2, "models", f"{reg}_pos_hubsig"))
    mm = m[sel].reset_index(drop=True)
    o = run_net(net, img[sel], normalise(scalars_v2(mm, reg, nd["cols"]), nd))
    pyA[sel], pyB[sel], pySA[sel], pySB[sel] = decode(o)
cnn = {}
for nm, py, cm in (("alpha", pyA, m.emuCnnA.values), ("beta", pyB, m.emuCnnB.values),
                   ("sigAlpha", pySA, m.emuCnnSigA.values), ("sigBeta", pySB, m.emuCnnSigB.values)):
    d = np.abs(py - cm)
    for reg, sel in (("barrel", m.subdet.values == 1), ("disc", m.subdet.values == 2)):
        cnn[f"{nm} {reg}"] = dict(max_abs_deg=float(d[sel].max()), p999_abs_deg=float(np.quantile(d[sel], 0.999)),
                                  median_abs_deg=float(np.median(d[sel])), frac_gt_1e4=float(np.mean(d[sel] > 1e-4)),
                                  frac_gt_1e2=float(np.mean(d[sel] > 1e-2)))
res["cnn_cmssw_vs_torch"] = cnn

# 2. class
pyT = ((m.tpIdx >= 0) & (m.tpCharge != 0) & (m.tpPt >= PT_MIN) & (m.hxOk == 1) & (m.hxDist < HXDIST_MAX)).values
cmT = m.emuClass.values == 1
res["class"] = dict(fracT=float(cmT.mean()), n_differ=int((pyT != cmT).sum()),
                    fracT_by_surface={int(s): float(cmT[m.surface == s].mean()) for s in np.unique(m.surface)},
                    pClass_matches_debug=bool(np.all(m.pEmuClass.values == m.emuClass.values)))
hx = (m.emuHxDist >= 0) & (m.hxOk == 1)
res["hxDist_producer_vs_same_job_ntuple"] = dict(n=int(hx.sum()), max_abs=float(np.abs(m.emuHxDist - m.hxDist)[hx].max()))
if a.train_ntuples:
    old = []
    for f in a.train_ntuples.split(","):
        tt = uproot.open(f)["clusterAdcNtuple/clusters"]
        old.append(tt.arrays(["event", "detId", "idxInDet", "hxOk", "hxDist", "hxNRoots", "tpIdx"], library="pd",
                             cut=None, entry_stop=None))
    O = pd.concat(old, ignore_index=True)
    O = O[O.event.isin(m.event.unique())]
    J = m.merge(O, on=["event", "detId", "idxInDet"], suffixes=("", "_train"))
    both = (J.hxOk == 1) & (J.hxOk_train == 1)
    d = np.abs(J.emuHxDist - J.hxDist_train)[both & (J.emuHxDist >= 0)]
    res["hxDist_producer_vs_training_ntuple"] = dict(
        n_joined=int(len(J)), n_ntuple=int(n), tpIdx_differ=int((J.tpIdx != J.tpIdx_train).sum()),
        hxOk_differ=int((J.hxOk != J.hxOk_train).sum()), max_abs_cm=float(d.max()) if len(d) else None,
        frac_abs_gt_1e5=float(np.mean(d > 1e-5)) if len(d) else None,
        class_threshold_flips=int((((J.hxDist_train < HXDIST_MAX) & (J.hxOk_train == 1)) != ((J.emuHxDist < HXDIST_MAX) & (J.emuHxDist >= 0)))[
            (J.tpIdx >= 0) & (J.tpCharge != 0) & (J.tpPt >= PT_MIN)].sum()))

# 3. emulation
lib = load_library(a.library)
obs_base = np.c_[m.sizeX, m.sizeY, np.log(m.charge.clip(lower=1)).values, m.globalR, np.abs(m.globalZ.values), m.localX, m.localY]
keys = (m.run.values, m.event.values, m.detId.values, m.idxInDet.values)
out = {}
for tag, (A_, B_, SA, SB) in (("cmssw_cnn", (m.emuCnnA.values, m.emuCnnB.values, m.emuCnnSigA.values, m.emuCnnSigB.values)),
                              ("torch_cnn", (pyA, pyB, pySA, pySB))):
    e = emulate(lib, m.surface.values, cmT, m.tpLocalCotAlpha.values, m.tpLocalCotBeta.values, A_, B_,
                np.c_[A_, B_, SA, SB, obs_base], *keys)
    C = ~cmT
    dd = lambda x, y, s=slice(None): float(np.nanmax(np.abs(np.asarray(x)[s] - np.asarray(y)[s]))) if np.any(s) else 0.0
    o = dict(donor_differ=int((e["donor"][C] != m.emuDonor.values[C]).sum()), n_classC=int(C.sum()),
             dumpRow_differ=int((e["dumpRow"] != m.emuDumpRow.values).sum()),
             pseudo_max_abs_deg=max(dd(e["pseudoA"], m.emuPseudoA), dd(e["pseudoB"], m.emuPseudoB)),
             bias_max_abs_deg=max(dd(e["biasA"], m.emuBiasA), dd(e["biasB"], m.emuBiasB)),
             out_max_abs_deg=max(dd(e["outA"], m.emuOutA), dd(e["outB"], m.emuOutB)),
             out_frac_abs_gt_1e6=float(np.mean((np.abs(e["outA"] - m.emuOutA) > 1e-6) | (np.abs(e["outB"] - m.emuOutB) > 1e-6))),
             sat_differ=int(((e["satA"] != (m.pSatAlpha.values == 1)) | (e["satB"] != (m.pSatBeta.values == 1))).sum()),
             cot_max_rel=float(np.max(np.abs(e["cotA"].astype(np.float64) - m.pCotAlpha.values) / np.maximum(1e-3, np.abs(m.pCotAlpha.values)))),
             classT_out_max_abs_deg=dd(e["outA"], m.emuOutA, cmT))
    out[tag] = o
    print(tag, json.dumps(o), flush=True)
res["emulation"] = out
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, "w"), indent=1)
