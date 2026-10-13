"""WP-D0a: are two censuses' TRACKS identical, so that they differ only in truth?

    python eval_refitq/wpd0a/tracks_identical.py OLD NEW -o out.json
    python eval_refitq/wpd0a/tracks_identical.py OLD NEW --fit-shift [--input F.root] -o out.json

OLD and NEW are census caches (a tpcensus_* directory, or a directory holding exactly one)
with exported tracks, built on the same input and configuration, e.g. run B (format 27) and
run C (format 28: an OT stub credited to its principal TP and its clusters' owners). The
stub credit is truth only, so every non-truth quantity must be bit-identical, shard by
shard: the per-entry hit lists (which hits each track holds), every non-truth track column
both censuses carry (fitted parameters, chi2s, angle pulls, rank score, the fit's claimed
sigma), the angle rows' measured columns, the non-truth counters (pairs, candidates, fits,
DR survivors, stage rejections), the per-region occupancy and the seed multiplicity. The
truth both carry is reported, not judged: the track and angle truth columns (how many values
changed), the census TP table (cen_*), the seed bitmaps (found, found_dr, found_dr_pos), the
quality samples (qual_*) and the truth counters (how many shards or entries changed), with
`truth_identical` true when none did; columns and arrays only one side has are listed.

Track column names: the manifest's "track_cols" (census format >= 29); for an older
census, the prefix of this tree's KF.TRACK_COLS of the shard's width (columns are only ever
appended).

With --fit-shift, for censuses whose tracks differ by design (run A vs B: the KF variance),
tracks are matched by hit list instead and the r-z chi2 shift reported per seed class; with
--input (the censuses' input file) each matched track's OT stubs are looked up in the
census's own unified table (TF.unified_chunks of this tree, same order of hits) to test
run B's prereg: the fit moves exactly on the tracks holding a flat PS stub whose format-26
z variance was scaled (approxB(z, r) > 1 on a PS module that is not tilted).

Reads the shards as stored (numpy only), so censuses of different formats compare without
either code version.
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
                    "d_kappa", "d_phi0", "d_d0", "d_cot", "d_z0",
                    "n_ot_right_outer_only", "n_tp_majority")
TRUTH_ANGLE_COLS = ("hit_wrong", "n_wrong")
TRUTH_COUNTERS = ("found_standalone", "trip_true")


def census_dir(p):
    if os.path.exists(os.path.join(p, "manifest.json")):
        return p
    ds = sorted(glob.glob(os.path.join(p, "tpcensus_*")))
    if len(ds) != 1:
        raise SystemExit(f"{p}: expected one tpcensus_* directory, found {len(ds)}")
    return ds[0]


def track_cols(man, width):
    """The column names of a census's track rows: the manifest's own list (format >= 29),
    else the prefix of KF.TRACK_COLS of the rows' width (columns are only ever appended)."""
    cols = list(man.get("track_cols") or KF.TRACK_COLS[:width])
    if len(cols) != width:
        raise SystemExit(f"track rows of width {width} against {len(cols)} known column names")
    return cols


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
    ac = {c: j for j, c in enumerate(KF.ANGLE_COLS)}
    nt_ang = [j for c, j in ac.items() if c not in TRUTH_ANGLE_COLS]
    bad, n_tracks, n_hits = [], 0, 0
    truth_changed, cnt_changed, cols_info = {}, {c: 0 for c in TRUTH_COUNTERS}, None
    arrays_changed, arrays_one_side, ang_truth_changed = {}, set(), 0
    truth_array = lambda f: f.startswith(("cen_", "qual_")) or f in ("found", "found_dr", "found_dr_pos")
    for a, b in zip(so, sn):
        za, zb = np.load(a), np.load(b)
        tag = os.path.basename(a)
        for k in ("n_events", "n_clusters", "nonant", "seed_eta"):
            if not _eq(za[k], zb[k]):
                bad.append(f"{tag}: {k}")
        for f in sorted({f for f in za.files if truth_array(f)} | {f for f in zb.files if truth_array(f)}):
            if f not in za.files or f not in zb.files:
                arrays_one_side.add(f)
            elif not _eq(za[f], zb[f]):
                arrays_changed[f] = arrays_changed.get(f, 0) + 1
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
                ca, cb = track_cols(mo, x.shape[1]), track_cols(mn, y.shape[1])
                common = [c for c in ca if c in cb]
                if cols_info is None:
                    cols_info = {"old_only": [c for c in ca if c not in cb],
                                 "new_only": [c for c in cb if c not in ca]}
                ia, ib = [ca.index(c) for c in common], [cb.index(c) for c in common]
                nt = [k for k, c in enumerate(common) if c not in TRUTH_TRACK_COLS]
                n_tracks += len(x)
                if len(x) != len(y) or not _eq(x[:, ia][:, nt], y[:, ib][:, nt]):
                    bad.append(f"{tag}: {f} non-truth columns")
                elif len(x):
                    for k, c in enumerate(common):
                        if c in TRUTH_TRACK_COLS:
                            u, v = x[:, ia[k]], y[:, ib[k]]
                            truth_changed[c] = truth_changed.get(c, 0) + int(
                                np.sum(~((u == v) | (np.isnan(u) & np.isnan(v)))))
            else:
                if x.shape != y.shape or not _eq(x[:, nt_ang], y[:, nt_ang]):
                    bad.append(f"{tag}: {f} non-truth angle columns")
                elif not _eq(x, y):
                    ang_truth_changed += 1
        ca_, cb_ = json.loads(str(za["counters"])), json.loads(str(zb["counters"]))
        if set(ca_) != set(cb_):
            bad.append(f"{tag}: counter entries differ")
        for e in set(ca_) & set(cb_):
            for k in set(ca_[e]) | set(cb_[e]):
                if k in TRUTH_COUNTERS:
                    cnt_changed[k] += int(ca_[e].get(k) != cb_[e].get(k))
                elif ca_[e].get(k) != cb_[e].get(k):
                    bad.append(f"{tag}: counter {e}/{k} {ca_[e].get(k)} vs {cb_[e].get(k)}")
    if len(so) != len(sn) and not (allow_prefix and len(sn) < len(so)):
        bad.append(f"shard count {len(so)} vs {len(sn)}")
    for k in ("same_config", "same_inputs", "same_seeds", "same_calibration"):
        if not out[k]:
            bad.append(k)
    out.update(n_tracks=n_tracks, n_track_hits=n_hits, identical=not bad, differences=bad[:50],
               n_differences=len(bad), columns=cols_info, truth_track_values_changed=truth_changed,
               truth_counters_changed_entries=cnt_changed,
               truth_arrays_changed_shards=arrays_changed, arrays_on_one_side=sorted(arrays_one_side),
               angle_entries_with_truth_changed=ang_truth_changed,
               truth_identical=not bad and not any(truth_changed.values()) and not any(cnt_changed.values())
               and not arrays_changed and not ang_truth_changed)
    return out


def fit_shift(old, new, input_path=None):
    """For censuses whose tracks DIFFER by design (e.g. run A vs run B, the KF variance):
    tracks matched by (shard, entry, hit list); per seed class x arity, how many matched
    tracks changed their r-z chi2 and by how much, and how many tracks exist on one side
    only. With input_path, each matched track is also classified by whether it holds a flat
    PS stub with approxB > 1 (the stubs whose z variance format 27 changed)."""
    do, dn = census_dir(old), census_dir(new)
    mo = json.load(open(os.path.join(do, "manifest.json")))
    shards = list(zip(sorted(glob.glob(os.path.join(do, "chunk_*.npz"))),
                      sorted(glob.glob(os.path.join(dn, "chunk_*.npz")))))
    chunks = None
    if input_path:
        import tp_findability as TF
        sigz = {int(k): float(v) for k, v in mo["calibration"]["sigz_ot"].items()}
        chunks = TF.unified_chunks(input_path, int(mo["n_events"]), int(mo["config"]["chunk"]), sigz)
    agg, flat = {}, {"changed_with_flat_stub": 0, "changed_without_flat_stub": 0,
                     "unchanged_with_flat_stub": 0, "unchanged_without_flat_stub": 0}
    for a, b in shards:
        za, zb = np.load(a), np.load(b)
        ca = track_cols(mo, za[[f for f in za.files if f.startswith("trk_")][0]].shape[1])
        tc = {c: j for j, c in enumerate(ca)}
        flat_hit = None
        if chunks is not None:
            (U, _), _ = next(chunks)
            if len(U["layer"]) != int(zb["n_clusters"]):
                raise SystemExit(f"{os.path.basename(b)}: the unified table has {len(U['layer'])} hits, "
                                 f"the shard {int(zb['n_clusters'])}")
            flat_hit = (U["ot_ps"] & ~U["ot_tilted"]
                        & (KF.approx_b(U["globalZ"], U["globalR"]) > 1.0))
        for f in sorted(f for f in za.files if f.startswith("hit_") and f in zb.files):
            e = f[4:]
            ha, hb, ta, tb = za[f], zb[f], za[f"trk_{e}"], zb[f"trk_{e}"]
            ia = {r.tobytes(): i for i, r in enumerate(ha)}
            pairs = [(ia[r.tobytes()], j) for j, r in enumerate(hb) if r.tobytes() in ia]
            t0 = ta if len(ta) else tb
            cat = f"{['it', 'ot', 'mixed'][int(t0[0, tc['sysclass']])]}{int(t0[0, tc['arity']])}"
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
                if flat_hit is not None:
                    G = hb[j]
                    has = (flat_hit[np.clip(G, 0, None)] & (G >= 0)).any(axis=1)
                    flat["changed_with_flat_stub"] += int((ch & has).sum())
                    flat["changed_without_flat_stub"] += int((ch & ~has).sum())
                    flat["unchanged_with_flat_stub"] += int((~ch & has).sum())
                    flat["unchanged_without_flat_stub"] += int((~ch & ~has).sum())
    out = {}
    for cat, g in sorted(agg.items()):
        r = np.concatenate(g.pop("rel")) if g["rel"] else np.zeros(0)
        out[cat] = {**g, "frac_changed": g["changed"] / max(g["matched"], 1),
                    "median_rel_chi2_rz_change_of_changed": float(np.median(r)) if len(r) else None}
    tot = {k: sum(v[k] for v in out.values()) for k in ("matched", "changed", "only_old", "only_new")}
    return {"old": do, "new": dn, "by_seed_class_arity": out, "total": tot,
            "flat_ps_stub_check": flat if input_path else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--allow-prefix", action="store_true",
                    help="NEW may hold fewer shards than OLD (a short smoke census): compare those")
    ap.add_argument("--fit-shift", action="store_true",
                    help="tracks differ by design: match them by hit list and report the r-z chi2 shift")
    ap.add_argument("--input", default=None,
                    help="with --fit-shift: the input file, to test which tracks hold a flat PS stub")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.fit_shift:
        r = fit_shift(a.old, a.new, a.input)
        json.dump(r, open(a.out, "w"), indent=1)
        print(json.dumps(r["total"]), {k: (round(v["frac_changed"], 3), v["median_rel_chi2_rz_change_of_changed"])
                                       for k, v in r["by_seed_class_arity"].items()})
        if r["flat_ps_stub_check"]:
            print("flat PS stub check:", r["flat_ps_stub_check"])
        return 0
    r = compare(a.old, a.new, a.allow_prefix)
    json.dump(r, open(a.out, "w"), indent=1)
    print(f"{r['n_tracks']:,} tracks ({r['n_track_hits']:,} hits) over {r['n_shards']} shards, formats "
          f"{r['format_old']} -> {r['format_new']}: "
          + ("IDENTICAL outside truth" if r["identical"] else f"{r['n_differences']} DIFFERENCES"))
    for d in r["differences"][:20]:
        print("  ", d)
    print("columns on one side only:", r["columns"])
    print("truth values changed:", {k: v for k, v in r["truth_track_values_changed"].items() if v})
    print("truth arrays changed (shards):", r["truth_arrays_changed_shards"], "on one side only:",
          r["arrays_on_one_side"], "angle entries with truth changed:", r["angle_entries_with_truth_changed"],
          "-> truth identical:", r["truth_identical"])
    return 0 if r["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
