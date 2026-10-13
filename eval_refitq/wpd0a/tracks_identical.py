"""WP-D0a: are two censuses' TRACKS identical, so that they differ only in truth?

    python eval_refitq/wpd0a/tracks_identical.py OLD NEW -o out.json

OLD and NEW are census caches (a tpcensus_* directory, or a directory holding
exactly one) with exported tracks, built on the same input and configuration, e.g.
run B (format 27) and run C (format 28: a stub credited to every TP behind its
clusters). The stub credit is truth only, so every non-truth quantity must be
bit-identical, shard by shard: the per-entry hit lists (which hits each track
holds), every non-truth track column (fitted parameters, chi2s, angle pulls, rank
score, the fit's claimed sigma), the angle rows' measured columns, the non-truth
counters (pairs, candidates, fits, DR survivors, stage rejections), the per-region
occupancy and the seed multiplicity. The truth columns are reported (how many
values changed), not judged.

With --fit-shift, for censuses whose tracks differ by design (run A vs B: the KF variance),
tracks are matched by hit list instead and the r-z chi2 shift reported per seed class.

Reads the shards as stored (numpy only), so censuses of different formats compare
without either code version; the column names are this tree's KF.TRACK_COLS and
KF.ANGLE_COLS, whose widths are checked against the shards.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "combinatorics"))
import kf_emulation as KF   # noqa: E402

TRUTH_TRACK_COLS = ("tp_key", "n_wrong", "n_own", "n_it_right", "n_it_wrong", "n_ot_right",
                    "n_ot_wrong", "n_ot_combinatoric", "n_ot_unknown", "own_is_seed",
                    "own_tpidx", "is_clean", "tp_pt", "tp_eta", "tp_d0", "tp_z0", "tp_phi0",
                    "d_kappa", "d_phi0", "d_d0", "d_cot", "d_z0")
TRUTH_ANGLE_COLS = ("hit_wrong", "n_wrong")
TRUTH_COUNTERS = ("found_standalone", "trip_true")


def census_dir(p):
    if os.path.exists(os.path.join(p, "manifest.json")):
        return p
    ds = sorted(glob.glob(os.path.join(p, "tpcensus_*")))
    if len(ds) != 1:
        raise SystemExit(f"{p}: expected one tpcensus_* directory, found {len(ds)}")
    return ds[0]


def _eq(a, b):
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def compare(old, new, allow_prefix=False):
    do, dn = census_dir(old), census_dir(new)
    mo, mn = (json.load(open(os.path.join(d, "manifest.json"))) for d in (do, dn))
    out = {"old": do, "new": dn, "format_old": mo["format"], "format_new": mn["format"],
           "same_config": mo["config"] == mn["config"], "same_inputs":
           [r["path"] for r in mo["inputs"]] == [r["path"] for r in mn["inputs"]],
           "same_seeds": (mo["seeds"], mo["seed_soft"], mo["seed_follow"], mo["builds"], mo["dr_groups"])
           == (mn["seeds"], mn["seed_soft"], mn["seed_follow"], mn["builds"], mn["dr_groups"]),
           "same_calibration": mo["calibration"] == mn["calibration"]
           and mo.get("ot_bend_cal") == mn.get("ot_bend_cal")}
    so = sorted(glob.glob(os.path.join(do, "chunk_*.npz")))
    sn = sorted(glob.glob(os.path.join(dn, "chunk_*.npz")))
    out["n_shards"] = [len(so), len(sn)]
    tc = {c: j for j, c in enumerate(KF.TRACK_COLS)}
    ac = {c: j for j, c in enumerate(KF.ANGLE_COLS)}
    nt_trk = [j for c, j in tc.items() if c not in TRUTH_TRACK_COLS]
    nt_ang = [j for c, j in ac.items() if c not in TRUTH_ANGLE_COLS]
    bad, n_tracks, n_hits = [], 0, 0
    truth_changed = {c: 0 for c in TRUTH_TRACK_COLS}
    cnt_changed = {c: 0 for c in TRUTH_COUNTERS}
    for a, b in zip(so, sn):
        za, zb = np.load(a), np.load(b)
        tag = os.path.basename(a)
        for k in ("n_events", "n_clusters", "nonant", "seed_eta"):
            if not _eq(za[k], zb[k]):
                bad.append(f"{tag}: {k}")
        fa = {f for f in za.files if f[:4] in ("trk_", "hit_", "ang_")}
        fb = {f for f in zb.files if f[:4] in ("trk_", "hit_", "ang_")}
        if fa != fb:
            bad.append(f"{tag}: entries differ ({sorted(fa ^ fb)[:5]})")
        for f in sorted(fa & fb):
            x, y = za[f], zb[f]
            if f.startswith("hit_"):
                n_hits += int((x >= 0).sum())
                if not _eq(x, y):
                    bad.append(f"{tag}: {f}")
            elif f.startswith("trk_"):
                if x.shape[1] != len(KF.TRACK_COLS) or y.shape[1] != len(KF.TRACK_COLS):
                    raise SystemExit(f"{tag}: {f} width {x.shape[1]}/{y.shape[1]} != {len(KF.TRACK_COLS)}")
                n_tracks += len(x)
                if x.shape != y.shape or not _eq(x[:, nt_trk], y[:, nt_trk]):
                    bad.append(f"{tag}: {f} non-truth columns")
                elif len(x):
                    for c in TRUTH_TRACK_COLS:
                        truth_changed[c] += int(np.sum(~((x[:, tc[c]] == y[:, tc[c]])
                                                         | (np.isnan(x[:, tc[c]]) & np.isnan(y[:, tc[c]])))))
            else:
                if x.shape != y.shape or not _eq(x[:, nt_ang], y[:, nt_ang]):
                    bad.append(f"{tag}: {f} non-truth angle columns")
        ca, cb = json.loads(str(za["counters"])), json.loads(str(zb["counters"]))
        if set(ca) != set(cb):
            bad.append(f"{tag}: counter entries differ")
        for e in set(ca) & set(cb):
            for k in set(ca[e]) | set(cb[e]):
                if k in TRUTH_COUNTERS:
                    cnt_changed[k] += int(ca[e].get(k) != cb[e].get(k))
                elif ca[e].get(k) != cb[e].get(k):
                    bad.append(f"{tag}: counter {e}/{k} {ca[e].get(k)} vs {cb[e].get(k)}")
    if len(so) != len(sn) and not (allow_prefix and len(sn) < len(so)):
        bad.append(f"shard count {len(so)} vs {len(sn)}")
    for k in ("same_config", "same_inputs", "same_seeds", "same_calibration"):
        if not out[k]:
            bad.append(k)
    out.update(n_tracks=n_tracks, n_track_hits=n_hits, identical=not bad, differences=bad[:50],
               n_differences=len(bad), truth_track_values_changed=truth_changed,
               truth_counters_changed_entries=cnt_changed)
    return out


def fit_shift(old, new):
    """For censuses whose tracks DIFFER by design (e.g. run A vs run B, the KF variance): tracks
    matched by (shard, entry, hit list); per seed class x arity, how many matched tracks changed
    their r-z chi2 and by how much, and how many tracks exist on one side only."""
    do, dn = census_dir(old), census_dir(new)
    tc = {c: j for j, c in enumerate(KF.TRACK_COLS)}
    agg = {}
    for a, b in zip(sorted(glob.glob(os.path.join(do, "chunk_*.npz"))),
                    sorted(glob.glob(os.path.join(dn, "chunk_*.npz")))):
        za, zb = np.load(a), np.load(b)
        for f in sorted(f for f in za.files if f.startswith("hit_") and f in zb.files):
            e = f[4:]
            ha, hb, ta, tb = za[f], zb[f], za[f"trk_{e}"], zb[f"trk_{e}"]
            ia = {r.tobytes(): i for i, r in enumerate(ha)}
            pairs = [(ia[r.tobytes()], j) for j, r in enumerate(hb) if r.tobytes() in ia]
            cat = f"{['it', 'ot', 'mixed'][int(ta[0, tc['sysclass']]) if len(ta) else int(tb[0, tc['sysclass']])]}" \
                  f"{int(ta[0, tc['arity']]) if len(ta) else int(tb[0, tc['arity']])}"
            g = agg.setdefault(cat, {"matched": 0, "changed": 0, "rel": [], "only_old": 0, "only_new": 0})
            g["matched"] += len(pairs)
            g["only_old"] += len(ha) - len(pairs)
            g["only_new"] += len(hb) - len(pairs)
            if pairs:
                i, j = np.array(pairs).T
                x, y = ta[i, tc["chi2_rz_per_layer"]], tb[j, tc["chi2_rz_per_layer"]]
                ch = x != y
                g["changed"] += int(ch.sum())
                g["rel"].append((y[ch] - x[ch]) / np.maximum(np.abs(x[ch]), 1e-12))
    out = {}
    for cat, g in sorted(agg.items()):
        r = np.concatenate(g.pop("rel")) if g["rel"] else np.zeros(0)
        out[cat] = {**g, "frac_changed": g["changed"] / max(g["matched"], 1),
                    "median_rel_chi2_rz_change_of_changed": float(np.median(r)) if len(r) else None}
    tot = {k: sum(v[k] for v in out.values()) for k in ("matched", "changed", "only_old", "only_new")}
    return {"old": do, "new": dn, "by_seed_class_arity": out, "total": tot}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--allow-prefix", action="store_true",
                    help="NEW may hold fewer shards than OLD (a short smoke census): compare those")
    ap.add_argument("--fit-shift", action="store_true",
                    help="tracks differ by design: match them by hit list and report the r-z chi2 shift")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.fit_shift:
        r = fit_shift(a.old, a.new)
        json.dump(r, open(a.out, "w"), indent=1)
        print(json.dumps(r["total"]), {k: (round(v["frac_changed"], 3), v["median_rel_chi2_rz_change_of_changed"])
                                       for k, v in r["by_seed_class_arity"].items()})
        return 0
    r = compare(a.old, a.new, a.allow_prefix)
    json.dump(r, open(a.out, "w"), indent=1)
    print(f"{r['n_tracks']:,} tracks ({r['n_track_hits']:,} hits) over {r['n_shards']} shards, formats "
          f"{r['format_old']} -> {r['format_new']}: "
          + ("IDENTICAL outside truth" if r["identical"] else f"{r['n_differences']} DIFFERENCES"))
    for d in r["differences"][:20]:
        print("  ", d)
    print("truth values changed:", {k: v for k, v in r["truth_track_values_changed"].items() if v})
    return 0 if r["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
