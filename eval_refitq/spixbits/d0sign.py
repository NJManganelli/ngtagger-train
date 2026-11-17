"""d0 sign check (PLAN.md section 8): refit d0 against true d0 on b/c daughters with |d0| > 200 um; correlation must be
>= 0.9 (material below). Reads a unit output of file 4 (run_scan.py unit ... --out DIR).

  python d0sign.py DIR [scenario]
"""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
scen = sys.argv[2] if len(sys.argv) > 2 else "AAAA"
T = np.load(d / f"{scen}_f04_truth.npz")
Z = np.load(d / f"{scen}_f04_arms.npz")
m = (T["origin"] == 1) & (np.abs(T["tp_d0"]) > 0.02)
res = {"n_tracks": int(m.sum()), "arms": {}, "material": []}
for arm in sorted({k.split("__")[0] for k in Z.files}):
    ok = m & Z[f"{arm}__refit"]
    x, y = T["tp_d0"][ok], Z[f"{arm}__a"][ok, 4]
    c = float(np.corrcoef(x, y)[0, 1])
    res["arms"][arm] = {"n_refit": int(ok.sum()), "correlation": c, "same_sign_share": float((np.sign(x) == np.sign(y)).mean())}
    if c < 0.9:
        res["material"].append(f"{arm}: correlation {c:.3f} < 0.9")
out = Path.home() / "smartpixels/scratch_local/spixbits/checks" / f"d0_sign_{scen}_f04.json"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
