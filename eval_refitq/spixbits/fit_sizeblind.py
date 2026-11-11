"""The size-blind position-error table: per |predicted cot| interval of v3's R tables, the robust width (1.4826 x MAD,
refit_calibration_fit.rs: the definition the table's own cells use) of the truth residual of the SAME fit population
with the cluster sizes pooled. (Averaging the cells' variances instead lets the 1-2 % of clusters in the 80-100 um
size >= 5 cells dominate the r-phi width: 11-17 um in place of 2-7 um. Code review, 2026-11-11.)

The fit population is rebuilt from the calibration sample that reproduces v3's tables exactly (counts and widths of
every cell are asserted): combinatorics/results/refit_calibration/calib_perfect_ttbar_pu200_tp.npz (main tree).

  python fit_sizeblind.py     -> sizeblind_R_v3.json (beside this file; read by study.tools), checks/sizeblind_widths.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import study as ST
import refit_calibration_fit as RCF

SAMPLE = Path.home() / "smartpixels/ngtagger-train/eval_refitq/combinatorics/results/refit_calibration/calib_perfect_ttbar_pu200_tp.npz"


def main():
    cal = json.load(open(ST.PAYLOAD / "refit_calibration_v3.json"))["R_tables"]
    z = np.load(SAMPLE)
    D = {k: z[k] for k in z.files}
    R = RCF.build_R_tables(D)
    for nm in "uv":
        assert R[nm]["n"] == cal[nm]["n"] and np.allclose(R[nm]["width_cm"], cal[nm]["width_cm"], rtol=0, atol=0), \
            "the sample does not reproduce v3's R tables"
    tr = D["truth"]
    base = (D["layer"] == 1) & (D["pt"] >= 5.0) & (np.abs(D["d0"]) < 0.005) & RCF.primary_mask(D)
    res = D["meas"] - tr
    doc = {"what": "size-blind position errors: 1.4826 x MAD of the truth residual per |predicted cot| interval, cluster "
                   "sizes pooled, on the fit population of v3's R tables (" + cal["built_from"] + ")",
           "sample": str(SAMPLE), "sample_sha256": hashlib.sha256(open(SAMPLE, "rb").read()).hexdigest(),
           "reproduces_v3_cells": True}
    chk = {}
    for k, nm, cotk, label in ((0, "u", 2, "localX (r-phi, SmartPixels y)"), (1, "v", 3, "localY (SmartPixels x)")):
        e = cal[nm]["cot_edges"]
        bc = RCF._bin(np.abs(tr[:, cotk]), e)
        w = [RCF.rs(res[base & (bc == j), k])[0] for j in range(len(e) - 1)]
        n = [int((base & (bc == j)).sum()) for j in range(len(e) - 1)]
        doc[nm] = {"cot_edges": e, "width_cm": w, "n": n}
        W, N = np.asarray(cal[nm]["width_cm"]), np.asarray(cal[nm]["n"], float)
        chk[label] = {"cot_edges": e, "size_blind_um": [round(x * 1e4, 2) for x in w], "n": n,
                      "old_variance_average_um (wrong)": [round(x * 1e4, 2) for x in np.sqrt((N * W ** 2).sum(0) / N.sum(0))],
                      "dominant_size_cell_um": [round(W[np.argmax(N[:, j]), j] * 1e4, 2) for j in range(W.shape[1])],
                      "dominant_size_cell_share": [round(float(N[:, j].max() / N[:, j].sum()), 2) for j in range(W.shape[1])]}
    json.dump(doc, open(Path(__file__).resolve().parent / "sizeblind_R_v3.json", "w"), indent=1)
    json.dump(chk, open(ST.LOCAL / "checks" / "sizeblind_widths.json", "w"), indent=1)
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
