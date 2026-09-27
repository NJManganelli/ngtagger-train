"""Input-fragility check of the production tiny CNNs (the check that caught the pos-hubsig model using 1e-16 geometry
residues; its result is check_frozen_poshubsig_v1.json). On 40k test clusters per region:
  kept_inputs     every scalar input with its training sd; none may sit at rounding level (sd < 1e-3 flags it)
  orientation     the kept orientation inputs after the 1e-6 rounding: distinct values and the smallest gap between them
  freeze_residue  output change when every orientation input is replaced by its value rounded to 1e-4 (i.e. any structure
                  finer than 1e-4 frozen out): |d angle| quantiles [deg]
  ulp_noise       output change when every raw scalar input moves by +-1 float32 ULP (the size of an FMA / op-order
                  difference across arch/release) BEFORE the orientation rounding: |d angle| quantiles [deg]
  unrounded       output change when the orientation inputs are fed WITHOUT the 1e-6 rounding
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emu_common import load_cnn, run_cnn, decode, CNN_DIR
from fast_models import raw_scalars, ORIENT

V2 = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_v2"
q = lambda d: [float(np.quantile(np.abs(d), x)) for x in (0.5, 0.9, 0.99, 0.999, 1.0)]


def scal(R, nd, rnd=6):
    S = R.copy()
    for c in ORIENT:
        if c in S and rnd is not None:
            S[c] = S[c].round(rnd)
    return S[nd["cols"]].values.astype(np.float32)


res = {}
for reg in ("barrel", "disc"):
    net, nd = load_cnn(CNN_DIR.format(reg=reg))
    m = pd.read_parquet(f"{V2}/arrays/{reg}_test_meta.parquet")
    img = np.load(f"{V2}/arrays/{reg}_test_img.npy", mmap_mode="r")
    sel = np.sort(np.random.default_rng(1).choice(len(m), 40000, replace=False))
    mm = m.iloc[sel].reset_index(drop=True); im = np.asarray(img[sel])
    R = raw_scalars(mm, reg)
    base = decode(run_cnn(net, nd, im, scal(R, nd)))
    d = lambda o: {"absdA_q50_90_99_999_max": q(o[0] - base[0]), "absdB_q50_90_99_999_max": q(o[1] - base[1])}
    r = {"n": len(mm), "kept_inputs": {c: nd["sd"][c] for c in nd["cols"]},
         "inputs_with_sd_below_1e-3": [c for c in nd["cols"] if nd["sd"][c] < 1e-3], "dropped_at_training": nd["dropped"]}
    ori = {}
    for c in ORIENT:
        if c in nd["cols"]:
            u = np.unique(R[c].round(6).values)
            ori[c] = {"distinct": int(len(u)), "min_gap": float(np.min(np.diff(u))) if len(u) > 1 else None,
                      "values_if_few": u.tolist() if len(u) <= 8 else None}
    r["orientation"] = ori
    r["freeze_residue"] = d(decode(run_cnn(net, nd, im, scal(R, nd, 4))))
    rng = np.random.default_rng(2); Rn = R.copy()
    for c in R.columns:
        v = R[c].values.astype(np.float32)
        Rn[c] = np.where(rng.random(len(v)) < 0.5, np.nextafter(v, np.float32(np.inf)), np.nextafter(v, np.float32(-np.inf))).astype(np.float64)
    for c in ("l1", "l2", "l3", "l4", "d1", "d2", "d3", "d4", "is3D", "truncX", "truncY", "sizeX", "sizeY", "size", "nPxLost", "sideZ"):
        if c in Rn:
            Rn[c] = R[c]                                   # integer-valued inputs cannot move by an ULP
    r["ulp_noise"] = d(decode(run_cnn(net, nd, im, scal(Rn, nd))))
    r["unrounded"] = d(decode(run_cnn(net, nd, im, scal(R, nd, None))))
    res[reg] = r
    print(reg, json.dumps({k: v for k, v in r.items() if k != "kept_inputs"}, indent=1), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_frozen.json"), "w"), indent=1)
