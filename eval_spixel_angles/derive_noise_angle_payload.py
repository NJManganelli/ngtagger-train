#!/usr/bin/env python
"""Derive smarthit_noise_cotAlpha/cotBeta from a Clusters-tier nano.

WHY THIS HAD TO BE REDONE. The previous payload was built by
fitSmartHitPayloads.py from `digi_parCotAlpha`, the parent momentum at its
PRODUCTION VERTEX rotated into the module frame. That is not the incidence angle:
the particle bends on the way in, by 0.041/0.088/0.157/0.196 in cot on TBPX L1-L4
(median) against a sensor resolution of 0.0225. So the old inverse CDF described
a quantity dominated by B-field bending.

WHY IT MATTERS BEYOND CORRECTNESS. Without it, SmartPixelsRecHitProducer gives
clusters with no simlink NO angle at all, and that turned out to be a perfect
truth proxy: hasAlpha was 99.5% for TrackingParticle-linked clusters and 0.0% for
unlinked ones. Every angle-weighted selection rule and every angle-using MVA was
therefore buying "is this cluster real" for free rather than using angle
information. The chi2 weight scan ran to the top of every grid and the
per-cluster MVA hit AUC 0.9996 for that reason. This payload closes the leak.

WHAT THE DISTRIBUTION IS. For a cluster with no simlink the sensor still sees a
charge pattern and its network still emits an angle. The right stand-in is the
INCLUSIVE distribution of angles the sensor reports on that layer -- a noise
cluster should look like an arbitrary cluster, not like a flagged special case.
So the CDF is taken over the RECO (post-response) angles of all clusters that
report one, per layer. Using the TRUE angles instead would omit the sensor
response and make noise clusters subtly sharper than real ones, which is exactly
the kind of tell this is meant to remove.

Schema matches the payload it replaces: category(layer) -> binning(quantile).

    pixi run python eval_spixel_angles/derive_noise_angle_payload.py \
        -i <clusters-tier nano.root> -o smarthit_noise_v5.json
"""
from __future__ import annotations

import argparse
import datetime
import glob as _glob
import json
import os

import awkward as ak
import numpy as np
import uproot

CLUSTER_TABLE = "L1TSmartPixelsCluster"
NQ = 40  # quantile bins, matching the payload this replaces


def build(paths, layers=(1, 2, 3, 4), use_truth=False):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    src = "truthCot" if use_truth else "recoCot"
    cols = ["layer", f"{src}Alpha", f"{src}Beta", "hasAlpha", "hasBeta", "truthTpIdx"]
    with uproot.open(f"{files[0]}:Events") as t:
        keys = set(t.keys())
    miss = [c for c in cols if f"{CLUSTER_TABLE}_{c}" not in keys]
    if miss:
        raise SystemExit(
            f"input lacks {CLUSTER_TABLE} columns {miss}. Needs a Clusters-tier nano produced "
            "AFTER the cluster table began reading SmartPixelsRecHit.")
    A = uproot.concatenate([f"{f}:Events" for f in files],
                           filter_name=[f"{CLUSTER_TABLE}_{c}" for c in cols])
    K = {c: ak.to_numpy(ak.flatten(A[f"{CLUSTER_TABLE}_{c}"])) for c in cols}
    n_ev = sum(uproot.open(f"{f}:Events").num_entries for f in files)

    edges = list(np.linspace(0.0, 1.0, NQ + 1))
    qmid = (np.array(edges[:-1]) + np.array(edges[1:])) / 2.0
    out = {}
    print(f"files={len(files)} events={n_ev} clusters={len(K['layer'])} source={src}*")
    for ang, hasf in (("Alpha", "hasAlpha"), ("Beta", "hasBeta")):
        per_layer = {}
        for L in layers:
            m = (K["layer"] == L) & (K[hasf] > 0) & (K[f"{src}{ang}"] > -900)
            v = K[f"{src}{ang}"][m]
            if len(v) < 200:
                raise SystemExit(f"layer {L} has only {len(v)} clusters with a reported "
                                 f"cot{ang}; too few for a {NQ}-bin inverse CDF")
            per_layer[L] = (np.quantile(v, qmid).tolist(), int(len(v)),
                            float(np.median(v)), float(np.quantile(np.abs(v), 0.95)))
            print(f"   cot{ang} L{L}: n={len(v):>7}  median {np.median(v):+.4f}  "
                  f"|q95| {np.quantile(np.abs(v),0.95):.4f}")
        out[ang] = per_layer
    return out, edges, files, n_ev, src


def correction(name, desc, per_layer, edges):
    return {
        "name": name,
        "version": 5,
        "inputs": [{"name": "layer", "type": "int", "description": "TBPX layer 1..4"},
                   {"name": "quantile", "type": "real",
                    "description": "uniform draw in [0,1); the caller supplies it, so the "
                                   "draw is the CALLER's to make deterministic"}],
        "output": {"name": "cot", "type": "real"},
        "description": desc,
        "data": {
            "nodetype": "category", "input": "layer",
            "content": [{"key": L,
                         "value": {"nodetype": "binning", "input": "quantile",
                                   "edges": edges, "content": vals,
                                   "flow": "clamp"}}
                        for L, (vals, _, _, _) in sorted(per_layer.items())]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--use-truth", action="store_true",
                    help="build from TRUE incidence angles instead of the sensor-reported "
                         "ones. NOT recommended: it omits the sensor response and would make "
                         "noise clusters sharper than real ones, which is itself a tell.")
    args = ap.parse_args()

    per, edges, files, n_ev, src = build(args.inputs, use_truth=args.use_truth)
    stamp = (f"derive_noise_angle_payload.py generated={datetime.datetime.utcnow().isoformat()}Z; "
             f"source={src}* over {n_ev} events, {len(files)} file(s); "
             f"basis={os.path.basename(files[0])}")
    common = ("Inverse CDF of the INCLUSIVE per-layer distribution of sensor-reported {a}, used "
              "to give an angle to clusters with no simlink. Replaces the pre-2026-09 payload, "
              "which was built from the parent momentum at its PRODUCTION VERTEX and was "
              "therefore dominated by B-field bending rather than by the sensor. ")
    cset = {
        "schema_version": 2,
        "description": "SmartPixels noise-angle stand-in. " + stamp,
        "corrections": [
            correction("smarthit_noise_cotAlpha", common.format(a="cotAlpha") + stamp,
                       per["Alpha"], edges),
            correction("smarthit_noise_cotBeta", common.format(a="cotBeta") + stamp,
                       per["Beta"], edges),
        ],
    }
    with open(args.output, "w") as fh:
        json.dump(cset, fh, indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
