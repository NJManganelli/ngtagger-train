"""Compare census runs that differ in one knob: window allowance, or alpha/beta bits.

Reads every census under a scan cache directory, groups them by the knob that
varied, and reports what it bought and what it cost:

  cost        cluster pairs, projection matches and fits per event
  reach       efficiency over findable TPs, BANDED BY TRUE |d0|, which is where
              a projection-window change is supposed to show up
  resolution  sigma(d0) and sigma(1/pT) from the per-seed quality samples

Efficiency here is the SEED-TRUTHFUL kind (the seed's own clusters belong to the
TP), OR'd over every seed: it answers "can the seeding reach this particle at
all", which is the question a window or bit-width change bears on.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF   # noqa: E402
import seed_arity as SA       # noqa: E402

D0_BANDS = [(0, 0.001), (0.001, 0.005), (0.005, 0.02), (0.02, 0.1), (0.1, 1e9)]
D0_NAMES = ["<10um", "10-50", "50-200", "200um-1mm", ">1mm"]


def sig(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def summarise(d):
    C = TF.load(d, verbose=False)
    cfg = json.load(open(Path(d) / "manifest.json"))["config"]
    n = C["n_events"]
    nit = np.unpackbits(C["hit_it"][:, None], axis=1).sum(1)
    not_ = np.unpackbits(C["hit_ot"][:, None], axis=1).sum(1)
    findable = (nit + not_ >= 4) & (C["pt"] >= cfg["ptmin"])
    reached = C["found"].any(axis=1)
    d0 = np.abs(C["d0"])
    eff, den = [], []
    for lo, hi in D0_BANDS:
        m = findable & (d0 >= lo) & (d0 < hi) & np.isfinite(d0)
        den.append(int(m.sum()))
        eff.append(float(reached[m].mean()) if m.sum() > 20 else float("nan"))
    ct = {k: sum(c.get(k, 0) for c in C["counters"].values()) / max(n, 1)
          for k in ("pairs", "cand", "fit", "owned")}
    Q = np.concatenate([v for v in C["qual"].values() if len(v)]) \
        if any(len(v) for v in C["qual"].values()) else np.zeros((0, 5))
    return {"dir": Path(d).name, "n_events": n,
            "d0_window_cm": cfg.get("d0_window_cm", 0.0),
            "bench": cfg.get("bench"), "cost": ct, "eff": eff, "den": den,
            "sigma_d0_um": sig(Q[:, 1] * 1e4) if len(Q) else float("nan"),
            "sigma_invpt": sig(Q[:, 0]) if len(Q) else float("nan"),
            "sigma_z0_um": sig(Q[:, 3] * 1e4) if len(Q) else float("nan"),
            "n_qual": int(len(Q))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default="cache-scan")
    ap.add_argument("--mode", choices=["window", "bits"], default="window")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    ds = sorted(glob.glob(f"{a.cache_dir}/tpcensus_*"))
    if not ds:
        raise SystemExit(f"no censuses under {a.cache_dir}")
    rows = []
    for d in ds:
        # a run still in flight has a manifest but no shards yet, and load()
        # raises SystemExit for that -- which `except Exception` does not catch
        if not list(Path(d).glob("chunk_*.npz")):
            continue
        try:
            rows.append(summarise(d))
        except (Exception, SystemExit) as e:
            print(f"  skipped {Path(d).name}: {e}")
    if a.mode == "window":
        rows = [r for r in rows if not r["bench"]]
        rows.sort(key=lambda r: r["d0_window_cm"])
        key = lambda r: f"{r['d0_window_cm'] * 1e4:.0f} um"
        head = "window allowance"
    else:
        rows = [r for r in rows if r["bench"]]
        rows.sort(key=lambda r: (r["bench"][0], r["bench"][1]))
        key = lambda r: f"a{r['bench'][0]} b{r['bench'][1]}"
        head = "alpha/beta bits"
    if not rows:
        raise SystemExit(f"no runs matching mode={a.mode} yet")
    print(f"{len(rows)} run(s), {rows[0]['n_events']} events each\n")
    print(f"{head:>16} {'pairs/ev':>10} {'proj/ev':>12} {'fit/ev':>9} "
          f"{'own/ev':>8} {'s(d0)um':>9} {'s(1/pT)':>9} " +
          " ".join(f"{n:>10}" for n in D0_NAMES))
    print(f"{'':>16} {'':>10} {'':>12} {'':>9} {'':>8} {'':>9} {'':>9} " +
          " ".join(f"{('N=' + format(rows[0]['den'][i], ',')):>10}"
                   for i in range(len(D0_NAMES))))
    for r in rows:
        print(f"{key(r):>16} {r['cost']['pairs']:10,.0f} {r['cost']['cand']:12,.0f}"
              f" {r['cost']['fit']:9,.0f} {r['cost']['owned']:8,.0f}"
              f" {r['sigma_d0_um']:9.1f} {r['sigma_invpt']:9.5f} " +
              " ".join(f"{100 * e:9.1f}%" if np.isfinite(e) else f"{'-':>10}"
                       for e in r["eff"]))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
