"""Closure: CMSSW angleSource="emu" vs the python reference (emu_common), on the same clusters.

  emu_closure_cmssw.py NTUPLE.root [--out emu_cmssw/closure.json]

NTUPLE is a ClusterAdcNtuple of a job with angleSource=emu, emuDebug=True (branches emu*: the producer's CNN outputs,
class, full-scan hxDist, A1 draws, pseudo-truth, dump row, biases, output angles; pCot*, pSat*). The python side
 1. rebuilds the CNN window and scalars from the SAME ntuple rows as prep.py / v2_common.derived do and runs the torch
    tiny CNN (CPU, float32)                                                    -> CNN closure (float precision)
 2. classifies from the ntuple truth (linked, charged, helix crossing within 1 cm; no pT cut)  -> class identity
 3. draws the A1 residuals with the same hash uniforms, (a) from the CMSSW CNN values (the draw in isolation) and
    (b) from its own torch CNN values                                          -> draw identity / bin-flip rate
 4. runs emu_common.emulate with the same library and keys, (a) on the CMSSW CNN values and draws (the selection
    logic in isolation) and (b) end to end on its own values                   -> dump row / output identity
"""
import argparse, os, sys, json, numpy as np, pandas as pd, uproot, awkward as ak
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emu_common import (load_cnn, cnn_scalars, run_cnn, decode, load_a1, a1_inputs, a1_draw, a1_uniforms, class_t,
                        load_library, emulate, WX, WY, CNN_DIR)
from v2_common import derived

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("ntuple")
p.add_argument("--library", default="/Volumes/WDMac/smartpixels_scratch/emu_fix/spix_emu_library_v2.root")
p.add_argument("--out", default=os.path.join(HERE, "emu_cmssw", "closure.json"))
a = p.parse_args()
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
m = pd.concat([m, derived(m)], axis=1)
m["surface"] = np.where(m.subdet == 1, m.layer, 10 + m.disk).astype(np.int64)
lod = np.where(m.subdet == 1, m.layer, m.disk)
keys = (m.run.values, m.event.values, m.detId.values, m.idxInDet.values)
res = {"n": int(n), "surfaces": {int(k): int(v) for k, v in m.surface.value_counts().sort_index().items()}}
q = lambda d: dict(max=float(np.max(d)), p999=float(np.quantile(d, 0.999)), median=float(np.median(d)),
                   frac_gt_1e4=float(np.mean(d > 1e-4)), frac_gt_1e2=float(np.mean(d > 1e-2)))

# 1. CNN
py = [np.full(n, np.nan) for _ in range(4)]
a1m = {}
for reg, sel in (("barrel", m.subdet.values == 1), ("disc", m.subdet.values == 2)):
    net, nd = load_cnn(CNN_DIR.format(reg=reg))
    o = decode(run_cnn(net, nd, img[sel], cnn_scalars(m[sel].reset_index(drop=True), reg, nd)))
    for k in range(4):
        py[k][sel] = o[k]
    a1m[reg] = load_a1(os.path.join(HERE, "emu_cmssw", f"a1_{reg}.pt"))
cm = [m.emuCnnA.values, m.emuCnnB.values, m.emuCnnSigA.values, m.emuCnnSigB.values]
res["cnn_cmssw_vs_torch_deg"] = {f"{nm} {reg}": q(np.abs(py[k] - cm[k])[m.subdet.values == sd])
                                 for k, nm in enumerate(("alpha", "beta", "sigAlpha", "sigBeta")) for reg, sd in (("barrel", 1), ("disc", 2))}

# 2. class
pyT = class_t(m.tpIdx.values >= 0, m.tpCharge.values, m.hxOk.values, m.hxDist.values)
cmT = m.emuClass.values == 1
res["class"] = dict(fracT=float(cmT.mean()), n_differ=int((pyT != cmT).sum()),
                    fracT_by_surface={int(s): float(cmT[m.surface == s].mean()) for s in np.unique(m.surface)},
                    pClass_matches_debug=bool(np.all(m.pEmuClass.values == m.emuClass.values)))
hx = (m.emuHxDist >= 0) & (m.hxOk == 1)
res["hxDist_producer_vs_same_job_ntuple"] = dict(n=int(hx.sum()), max_abs=float(np.abs(m.emuHxDist - m.hxDist)[hx].max()))

# 3. A1 draws
U = a1_uniforms(*keys)
draws = {}
for tag, (A_, B_, SA, SB) in (("cmssw_cnn", cm), ("torch_cnn", py)):
    d = np.full((n, 2), np.nan)
    for reg, sd_ in (("barrel", 1), ("disc", 2)):
        s = m.subdet.values == sd_
        Z = a1_inputs(A_[s], B_[s], SA[s], SB[s], m.sizeX.values[s], m.sizeY.values[s], m.charge.values[s], m.globalR.values[s],
                      np.abs(m.globalZ.values[s]), m.localX.values[s], m.localY.values[s], lod[s])
        d[s] = a1_draw(a1m[reg], Z, U[s])
    draws[tag] = d
    dd = np.abs(d - np.c_[m.emuDrawA.values, m.emuDrawB.values])
    res[f"a1_draw_{tag}_vs_cmssw_deg"] = {"alpha": q(dd[:, 0]), "beta": q(dd[:, 1]),
                                          "frac_either_differs_gt_1e-3": float(np.mean((dd > 1e-3).any(1)))}

# 4. emulation
lib = load_library(a.library)
for tag, (A_, B_), d in (("cmssw_cnn_and_draws", (cm[0], cm[1]), np.c_[m.emuDrawA.values, m.emuDrawB.values]),
                         ("end_to_end_python", (py[0], py[1]), draws["torch_cnn"])):
    e = emulate(lib, m.surface.values, cmT, m.tpLocalCotAlpha.values, m.tpLocalCotBeta.values, A_, B_, d[:, 0], d[:, 1], *keys)
    dmax = lambda x, y, s=slice(None): float(np.nanmax(np.abs(np.asarray(x)[s] - np.asarray(y)[s]))) if np.any(s) else 0.0
    oA, oB = np.abs(e["outA"] - m.emuOutA.values), np.abs(e["outB"] - m.emuOutB.values)
    o = dict(dumpRow_differ=int((e["dumpRow"] != m.emuDumpRow.values).sum()),
             pseudo_max_abs_deg=max(dmax(e["pseudoA"], m.emuPseudoA), dmax(e["pseudoB"], m.emuPseudoB)),
             bias_max_abs_deg=max(dmax(e["biasA"], m.emuBiasA), dmax(e["biasB"], m.emuBiasB)),
             out_max_abs_deg=float(max(oA.max(), oB.max())),
             out_frac_abs_gt_1e6=float(np.mean((oA > 1e-6) | (oB > 1e-6))), out_frac_abs_gt_1e2=float(np.mean((oA > 1e-2) | (oB > 1e-2))),
             out_classC_frac_abs_gt_1e2=float(np.mean(((oA > 1e-2) | (oB > 1e-2))[~cmT])),
             sat_differ=int(((e["satA"] != (m.pSatAlpha.values == 1)) | (e["satB"] != (m.pSatBeta.values == 1))).sum()),
             classT_out_max_abs_deg=float(max(oA[cmT].max(), oB[cmT].max())),
             cot_max_rel_classT=float(np.max(np.abs(e["cotA"][cmT].astype(np.float64) - m.pCotAlpha.values[cmT])
                                              / np.maximum(1e-3, np.abs(m.pCotAlpha.values[cmT])))))
    res[f"emulation_{tag}"] = o
    print(tag, json.dumps(o), flush=True)
res["has_all_true"] = bool(np.all(m.pHasAlpha.values == 1) and np.all(m.pHasBeta.values == 1))
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, "w"), indent=1)
