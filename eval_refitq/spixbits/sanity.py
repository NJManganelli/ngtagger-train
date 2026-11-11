"""Sanity pass over finished scan units that reads NO physics result: files loadable, the same number of tracks in the
truth file and in every arm of a file x scenario (fixed population), no NaN in the delivered parameters and
covariances, refit flags present; --rerun SCEN FILE: the priority unit recomputed directly and compared bit for bit.

  python sanity.py [--group priority] [--rerun IAAA 5]      -> checks/sanity_<group>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import run_scan as RS
import study as ST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="priority")
    ap.add_argument("--rerun", nargs=2, default=None)
    a = ap.parse_args()
    out = ST.LOCAL
    reg = RS.load_registry(out)
    units = [u for u in RS.units_of(reg, ST.SCAN_FILES, ST.SCENARIOS) if u["group"] == a.group]
    res = {"group": a.group, "units": len(units), "finished": 0, "loadable": 0, "arms_checked": 0, "problems": []}
    for u in units:
        f = out / "units" / u["name"]
        if not f.exists():
            continue
        res["finished"] += 1
        try:
            Z = np.load(f)
            T = np.load(out / "units" / "truth" / f"{u['scen']}_f{u['file']:02d}_truth.npz")
            n = len(T["run"])
            for arm in u["arms"]:
                for k in ("a", "cov", "refit", "forced_a", "hit_accepted", "n_accepted"):
                    x = Z[f"{arm}__{k}"]
                    if len(x) != n:
                        res["problems"].append(f"{u['name']} {arm} {k}: {len(x)} tracks, truth {n}")
                    if x.dtype.kind == "f" and not np.isfinite(x).all():
                        res["problems"].append(f"{u['name']} {arm} {k}: non-finite values")
                if Z[f"{arm}__refit"].dtype != bool:
                    res["problems"].append(f"{u['name']} {arm}: refit flag not boolean")
                res["arms_checked"] += 1
            if len(np.unique(np.stack([T["run"], T["lumi"], T["event"], T["track"]], 1), axis=0)) != n:
                res["problems"].append(f"{u['name']}: pairing key not unique")
            res.setdefault("tracks", {})[f"{u['scen']}_f{u['file']:02d}"] = n
            res["loadable"] += 1
        except Exception as e:                                   # noqa: BLE001
            res["problems"].append(f"{u['name']}: {e!r}")
    if a.rerun:
        scen, fnum = a.rerun[0], int(a.rerun[1])
        u = next(x for x in units if x["scen"] == scen and x["file"] == fnum)
        S = RS.Study()
        truth, arms = RS.run_unit(S, reg, fnum, scen, u["arms"])
        Z = np.load(out / "units" / u["name"])
        T = np.load(out / "units" / "truth" / f"{scen}_f{fnum:02d}_truth.npz")
        bad = [f"{n}__{k}" for n, d in arms.items() for k, v in d.items()
               if not np.array_equal(v, Z[f"{n}__{k}"], equal_nan=True)]
        bad += [f"truth {k}" for k, v in truth.items() if not np.array_equal(v, T[k], equal_nan=True)]
        res["rerun"] = {"unit": u["name"], "arrays_compared": sum(len(d) for d in arms.values()) + len(truth),
                        "different": bad}
        if bad:
            res["problems"].append(f"rerun of {u['name']}: {len(bad)} arrays differ")
    res["pass"] = not res["problems"] and res["finished"] == res["units"]
    json.dump(res, open(out / "checks" / f"sanity_{a.group}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
