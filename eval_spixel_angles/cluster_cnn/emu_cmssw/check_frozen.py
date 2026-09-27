"""Sensitivity (a REJECTED option, kept as the evidence)  of pos-hubsig to freezing the degenerate scalar inputs (training sd < 1e-6) at their mean."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emu_common import load_net, run_net, decode, scalars_v2, norm_arrays
V2 = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_v2"
res = {}
for reg in ("barrel", "disc"):
    net, nd = load_net(f"{V2}/models/{reg}_pos_hubsig")
    m = pd.read_parquet(f"{V2}/arrays/{reg}_test_meta.parquet")
    img = np.load(f"{V2}/arrays/{reg}_test_img.npy", mmap_mode="r")
    sel = np.sort(np.random.default_rng(1).choice(len(m), 40000, replace=False))
    mm = m.iloc[sel].reset_index(drop=True); im = np.asarray(img[sel])
    S = scalars_v2(mm, reg, nd["cols"])
    mu, sd = norm_arrays(nd); frozen = sd < 1e-6
    Z0 = ((S - mu) / sd).astype(np.float32); Z1 = Z0.copy(); Z1[:, frozen] = 0.0
    o0 = run_net(net, im, Z0); o1 = run_net(net, im, Z1)
    a0, b0, sa0, sb0 = decode(o0); a1, b1, sa1, sb1 = decode(o1)
    P = pd.read_parquet(f"{V2}/preds/{reg}_pos_hubsig.parquet").set_index("row").loc[mm.row.values]
    pa = np.degrees(np.arctan(P.cotA.values))
    dA, dB = a1 - a0, b1 - b0
    q = lambda d: [float(np.quantile(np.abs(d), x)) for x in (0.5, 0.9, 0.99, 0.999, 1.0)]
    res[reg] = dict(frozen=[c for c, f in zip(nd["cols"], frozen) if f], n=len(mm),
                    absdA_q50_90_99_999_max=q(dA), absdB_q50_90_99_999_max=q(dB),
                    absdSigA_q=q(sa1 - sa0), unfrozen_vs_stored_pred_absdA_max=float(np.max(np.abs(a0 - pa))))
    print(json.dumps(res[reg], indent=1), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_frozen.json"), "w"), indent=1)
