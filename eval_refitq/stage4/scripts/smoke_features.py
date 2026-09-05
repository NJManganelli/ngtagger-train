"""Smoke: build each feature group on 1 file and print shapes + score stats."""
import sys

import numpy as np

from ngtagger.train.trainer import prepare_dataset

NANO = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano"
view = sys.argv[1] if len(sys.argv) > 1 else "1111"
paths = {
    "1111": (f"{NANO}/nano_fat_1111_coopt_file1.root", "AAAA"),
    "1100": (f"{NANO}/nano_fat_1100_coopt_file1.root", "AAII"),
    "0000": (f"{NANO}/nano_fat_0000_baseline_file1.root", None),
}
f, cfg = paths[view]
MODELS = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage3/models"
bdt_json = f"{MODELS}/refitq_{cfg}_conifer.json" if cfg else None

cases = [["baseline"], ["baseline", "vertexdxy_pv"]]
if cfg:
    cases += [["baseline", "refitbdt"], ["baseline", "refitbdt", "vertexdxy_pv"]]

for groups in cases:
    ds = prepare_dataset([f], feature_groups=groups, seed=0,
                         refit_config=cfg, refit_bdt_json=bdt_json)
    X = np.concatenate([ds["X_train"], ds["X_test"]], axis=0)
    names = ds["feature_names"]
    print(f"\ngroups={groups}  X={X.shape}  nfeat={len(names)}")
    print("  names:", names)
    if "refit_bdt_score" in names:
        i = names.index("refit_bdt_score")
        col = X[:, :, i]
        nz = col[col != 0]
        print(f"  refit_bdt_score: nonzero frac {np.mean(col!=0):.3f} "
              f"range [{nz.min():.3f},{nz.max():.3f}] mean {nz.mean():.3f}" if len(nz) else "  all-zero")
    if "vtxpv_dxy" in names:
        i = names.index("vtxpv_dxy")
        col = X[:, :, i]
        nz = col[col != 0]
        print(f"  vtxpv_dxy: nonzero frac {np.mean(col!=0):.3f} "
              f"range [{nz.min():.3f},{nz.max():.3f}]" if len(nz) else "  all-zero")
