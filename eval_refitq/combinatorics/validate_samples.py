"""Are these nano files what they claim: complete, distinct, and consistent?

Three checks that were each done ad hoc and each caught something real:

  COLUMN COMPLETENESS. A study that silently loses a column reads as a physics
  result. The IT cost model needs every entry of IT_COLS, and a missing sigY once
  aborted a run only after the loader had already been patched twice.

  EVENT DISTINCTNESS. Two samples that look like 40 and 60 events can be 60 total
  rather than 100, if one is a subset of the other -- which is exactly what the
  40-event truth sample turned out to be. Comparing (run, lumi, event) across
  files is the only way to know how much independent statistics exist.

  OCCUPANCY CONSISTENCY. Files of the same dataset should agree on clusters and
  stubs per event to within the event-to-event RMS. A file that does not belong
  shows up here, and so does a production that silently changed configuration.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="comma-separated paths and/or globs")
    ap.add_argument("--no-overlap", action="store_true",
                    help="skip the pairwise (run,lumi,event) comparison")
    a = ap.parse_args()
    paths = M.expand_inputs(a.input)

    print(f"{'file':<34}{'events':>7}{'clusters/ev':>13}{'stubs/ev':>10}"
          f"{'OT truth':>9}{'missing IT cols':>18}")
    ids, occ, bad = {}, {}, []
    for p in paths:
        t = uproot.open(f"{p}:Events")
        keys = set(t.keys())
        miss = [c for c in M.IT_COLS if f"{M.IT_TABLE}_{c}" not in keys]
        want = [f"{M.IT_TABLE}_layer"]
        if f"{M.OT_TABLE}_layer" in keys:
            want.append(f"{M.OT_TABLE}_layer")
        A = t.arrays(want + ["run", "luminosityBlock", "event"])
        ncl = ak.to_numpy(ak.num(A[f"{M.IT_TABLE}_layer"]))
        nst = (ak.to_numpy(ak.num(A[f"{M.OT_TABLE}_layer"]))
               if f"{M.OT_TABLE}_layer" in keys else np.zeros(len(ncl)))
        ids[p] = set(zip(ak.to_numpy(A["run"]).tolist(),
                         ak.to_numpy(A["luminosityBlock"]).tolist(),
                         ak.to_numpy(A["event"]).tolist()))
        occ[p] = (ncl.mean(), ncl.std(), nst.mean())
        if miss:
            bad.append((p, miss))
        print(f"{Path(p).name[:33]:<34}{t.num_entries:>7}"
              f"{ncl.mean():>9.0f}+-{ncl.std():<3.0f}{nst.mean():>10.0f}"
              f"{'Y' if f'{M.OT_TABLE}_tpIdx' in keys else 'N':>9}"
              f"{(','.join(miss) if miss else 'none'):>18}")

    tot = sum(len(v) for v in ids.values())
    uni = len(set().union(*ids.values())) if ids else 0
    print(f"\nevents summed over files : {tot}")
    print(f"DISTINCT (run,lumi,event): {uni}"
          + ("" if uni == tot else f"   <-- {tot-uni} DUPLICATED across files"))
    if not a.no_overlap and len(paths) > 1 and uni != tot:
        print("\npairwise overlaps (only non-zero shown):")
        for i, p in enumerate(paths):
            for q in paths[i+1:]:
                n = len(ids[p] & ids[q])
                if n:
                    sub = (" p subset of q" if ids[p] <= ids[q] else
                           " q subset of p" if ids[q] <= ids[p] else "")
                    print(f"  {Path(p).name[:28]:<30}{Path(q).name[:28]:<30}{n:>6}{sub}")

    if len(paths) > 1:
        mu = np.array([occ[p][0] for p in paths])
        rms = np.mean([occ[p][1] for p in paths])
        spread = mu.max() - mu.min()
        print(f"\nclusters/event across files: {mu.min():.0f} .. {mu.max():.0f}"
              f"  (spread {spread:.0f}, mean event-to-event RMS {rms:.0f})")
        # ONLY MEANINGFUL WITHIN ONE DATASET. Mixing PU200 with noPU, or ttbar
        # with H->tautau, is supposed to disagree, so a large spread there is the
        # samples differing as intended rather than a production fault.
        if spread <= 2 * rms:
            print("  consistent with one dataset")
        else:
            print("  spread exceeds 2x the event-to-event RMS: EITHER these are"
                  " different processes/PU (expected) or one file does not belong."
                  " Re-run per dataset to tell the two apart.")
    if bad:
        print(f"\n{len(bad)} file(s) MISSING IT columns -- studies will fail on them")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
