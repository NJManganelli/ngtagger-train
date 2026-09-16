"""Export the census as typed binary arrays for the interactive efficiency page.

TWO TABLES, and the split is normalisation rather than a choice about metrics:

  TP     one row per TrackingParticle, with the seed bitmap. It exists because
         the efficiency DENOMINATOR must include TPs that no seed found, and
         those have no track row by definition.
  TRACK  one row per emitted track, PRE-duplicate-removal, carrying rank_score.

Everything the page shows is a mask and a bin over those. Efficiency for a seed
menu is `distinct tp_key present`, which is a set union and correct by
construction -- pre-binned per-seed histograms could not be summed without
double-counting a TP that several seeds find. Fake rate is additive over tracks
and needs no such care. Resolution needs the per-menu DR, which is why the track
table is PRE-DR and sorted:

  SORTED BY tp_key WITH FAKES IN A CONTIGUOUS tp_key < 0 BLOCK. The per-menu
  duplicate removal in the browser -- group by TP, take the best rank_score among
  ENABLED seeds -- is then a single linear scan over the sorted region instead of
  a hash over millions of rows, and fake queries skip that region entirely. At
  ~12.6M rows that is the difference between interactive and not.

float32 throughout. The hardware bit encoding (TTTrack_TrackWord) is deliberately
not used: it is still being evolved upstream, and baking a moving target into an
exported payload would age badly. See the note in kf_emulation for what an
encoding option would need.

NO SUBSETTING AND NO APPROXIMATION. This is a diagnostic tool, so the whole
sample goes out and duplicate removal is the exact hit-sharing greedy, not a
cheaper stand-in. At full scale that is ~2.4 GB in the tab and of order ten
seconds per DR recomputation, which is the right trade for an analysis page:
approximations introduced before anything has been verified are how a tool
starts lying quietly.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF   # noqa: E402
import kf_emulation as KF     # noqa: E402
import seed_arity as SA       # noqa: E402
import dr_conflict_scale as DRS  # noqa: E402

TP_COLS = ("key", "event", "pt", "eta", "phi", "d0", "z0", "vr",
           "hit_it", "hit_ot", "n_layers")


def _u32(b):
    """uint64 seed bitmap -> uint32 word pairs, little-endian, lo word first."""
    return np.ascontiguousarray(b.astype(np.uint64)).view(np.uint32)


def build_tp_table(C):
    """Per-TP rows plus the two seed bitmaps, restricted to a usable denominator.

    The >= 4 LAYER CUT is not cosmetic: MinLayers = 4 means a TP with fewer
    cannot be emitted by any seed, so including it would put an unreachable
    population in the denominator and make every efficiency look worse by a
    fixed, meaningless factor.
    """
    nit = np.unpackbits(C["hit_it"][:, None], axis=1).sum(1)
    not_ = np.unpackbits(C["hit_ot"][:, None], axis=1).sum(1)
    nlay = nit + not_
    keep = nlay >= 4
    cols = {"key": C["key"], "event": C["event"], "pt": C["pt"], "eta": C["eta"],
            "phi": C["phi"], "d0": C["d0"], "z0": C["z0"], "vr": C["vr"],
            "hit_it": C["hit_it"].astype(np.float32),
            "hit_ot": C["hit_ot"].astype(np.float32),
            "n_layers": nlay.astype(np.float32)}
    A = np.stack([np.asarray(cols[c], np.float32)[keep] for c in TP_COLS], axis=1)
    return A, C["found"][keep], C["found_dr"][keep]


def build_track_table(C):
    """Concatenate every seed's track rows and sort for the browser's DR scan."""
    if not C["tracks"]:
        raise SystemExit("census has no track rows; rerun with --export-tracks")
    T = np.concatenate([C["tracks"][i] for i in sorted(C["tracks"])])
    col = {c: j for j, c in enumerate(C["track_cols"])}
    k = T[:, col["tp_key"]]
    # fakes (tp_key < 0) first as one contiguous block, then real rows sorted by
    # TP so the per-menu DR is a linear scan
    order = np.lexsort((k, k >= 0))
    T = T[order]
    n_fake = int((T[:, col["tp_key"]] < 0).sum())
    # the permutation is returned so the hit lists can be put in the SAME order;
    # the conflict graph indexes rows of the sorted table, not the raw one
    return T, col, n_fake, order


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--arch-tracks", default=None,
                    help="npz of parallel-architecture rows from "
                         "arch_comparison.py --export-tracks")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--nev", type=int, default=None)
    a = ap.parse_args()
    ds = sorted(glob.glob(f"{a.cache_dir}/tpcensus_*"))
    if not ds:
        raise SystemExit(f"no census under {a.cache_dir}")
    C = TF.load(ds[-1], nev=a.nev, with_tracks=True)
    os.makedirs(a.outdir, exist_ok=True)
    a_outdir = a.outdir

    TPA, found, found_dr = build_tp_table(C)
    T, col, n_fake, order_orig = build_track_table(C)
    conflict = None
    if C.get("hits"):
        # the conflict graph, built once here rather than per interaction in the
        # browser. 178 s at full scale against ~10 s per menu change, so it
        # belongs on this side of the wire.
        H = np.concatenate([C["hits"][i] for i in sorted(C["hits"])])[order_orig]
        t0 = time.perf_counter()
        edges, st = DRS.conflict_edges(H)
        a = np.r_[edges[:, 0], edges[:, 1]]
        b = np.r_[edges[:, 1], edges[:, 0]]
        o = np.argsort(a, kind="stable")
        a, b = a[o], b[o]
        indptr = np.searchsorted(a, np.arange(len(T) + 1)).astype(np.int32)
        # descending rank_score, MENU-INDEPENDENT so it ships once
        rank_order = np.argsort(-T[:, col["rank_score"]],
                                kind="stable").astype(np.int32)
        indptr.tofile(os.path.join(a_outdir, "conf_indptr.bin"))
        b.astype(np.int32).tofile(os.path.join(a_outdir, "conf_nbr.bin"))
        rank_order.tofile(os.path.join(a_outdir, "conf_order.bin"))
        conflict = {"indptr": "conf_indptr.bin", "neighbours": "conf_nbr.bin",
                    "order": "conf_order.bin", "edges": int(len(edges)),
                    "min_shared": SA.MIN_SHARED_LAYERS,
                    "build_seconds": round(time.perf_counter() - t0, 1), **st}
    meta = {
        "n_events": C["n_events"],
        "source": ds[-1],
        "seeds": [{"idx": i, "tag": t,
                   "layers": list(C["seeds"][i].layers),
                   "arity": C["seeds"][i].arity,
                   "sysclass": KF.sysclass_of(C["seeds"][i].layers)}
                  for i, t in enumerate(C["seed_tags"])],
        # which seeds each activeSP build can form, so the page can gate the
        # seed selector on the build rather than offering seeds a build cannot
        # physically produce
        "builds": {m: sorted({s.tag for s in SA.enumerate_seeds(
            [TF.IL_OF[i] for i, ch in enumerate(m) if ch == "A"])})
            for m in TF.BUILD_MASKS},
        "tp": {"file": "tp.bin", "cols": list(TP_COLS), "rows": int(len(TPA)),
               "dtype": "float32"},
        "found": {"file": "found.bin", "rows": int(len(TPA)),
                  "words": int(found.shape[1]) * 2, "dtype": "uint32"},
        "found_dr": {"file": "found_dr.bin", "rows": int(len(TPA)),
                     "words": int(found_dr.shape[1]) * 2, "dtype": "uint32"},
        "track": {"file": "track.bin", "cols": list(C["track_cols"]),
                  "rows": int(len(T)), "dtype": "float32",
                  "n_fake": n_fake, "sorted_by": "tp_key, fakes first"},
        **({"conflict": conflict} if conflict else {}),
        "notes": [
            "Efficiency denominator is TPs with >= 4 layers hit: MinLayers = 4 "
            "means fewer cannot be emitted by any seed.",
            "Track rows are PRE duplicate removal and carry rank_score, so the "
            "browser can run DR over whatever seed subset is selected. Numbers "
            "will not match a fixed full-menu DR.",
            "DR is not a fake filter: it removes duplicate REAL tracks, so the "
            "fake FRACTION of a selection rises as more seeds are enabled.",
            "Fake tracks have no TrackingParticle and are binned in FITTED "
            "coordinates; TP maps are binned in truth. The two axes are not "
            "the same quantity.",
            "chi2_rphi_per_layer is per LAYER, not per dof, so it is not "
            "directly comparable to the hardware chi2/dof bin tables.",
            "Seed quality columns describe each seed's STANDALONE population; "
            "post-DR ownership is a separate bitmap (found_dr).",
            "Duplicate removal is the EXACT hit-sharing greedy (>= 3 shared "
            "hits), run in the browser over a precomputed conflict graph. No "
            "truth is used anywhere in it.",
        ],
    }
    if a.arch_tracks and os.path.exists(a.arch_tracks):
        z = np.load(a.arch_tracks, allow_pickle=False)
        A = z["tracks"].astype(np.float32)
        A.tofile(os.path.join(a.outdir, "arch_track.bin"))
        meta["arch_track"] = {"file": "arch_track.bin",
                              "cols": [str(x) for x in z["cols"]],
                              "rows": int(len(A)), "dtype": "float32",
                              "n_events": int(z["n_events"])}
        meta["notes"].append(
            "arch_track rows are the PARALLEL architecture (standalone IT and "
            "OT finding, then matched refit) and carry arch_mode/track_stage. "
            "They are a different pipeline from the main track table, not a "
            "subset of it.")
    # BITMAPS GO OUT AS uint32, NOT uint64. JavaScript has no fast 64-bit
    # integer type -- BigUint64Array forces BigInt arithmetic, which is an order
    # of magnitude slower than Uint32 ops and allocates per operation. The seed
    # mask is ANDed against every TP on every control change, so this is the
    # single hottest operation on the page. 30 seeds fit in one 32-bit word.
    TPA.astype(np.float32).tofile(os.path.join(a.outdir, "tp.bin"))
    _u32(found).tofile(os.path.join(a.outdir, "found.bin"))
    _u32(found_dr).tofile(os.path.join(a.outdir, "found_dr.bin"))
    T.astype(np.float32).tofile(os.path.join(a.outdir, "track.bin"))
    json.dump(meta, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
    # assemble the page next to its data. plotly is VENDORED, never a CDN, so
    # the directory works on a machine with no outbound network.
    import shutil
    src = Path(__file__).parent / "site_src"
    for f in ("explorer.html", "explorer_core.js"):
        shutil.copy(src / f, Path(a.outdir) / f)
    plotly = next((p for p in (
        Path(__file__).parents[2] / "eval_mva_explorer/site/plotly.min.js",
        Path(__file__).parents[2] / "eval_spixel/site/plotly.min.js")
        if p.exists()), None)
    if plotly:
        shutil.copy(plotly, Path(a.outdir) / "plotly.min.js")
    else:
        print("WARNING: no vendored plotly.min.js found; copy one into the "
              "output directory or the page will not render")
    tot = sum(os.path.getsize(os.path.join(a.outdir, f))
              for f in os.listdir(a.outdir))
    print(f"{len(TPA):,} TP rows, {len(T):,} track rows ({n_fake:,} fake), "
          f"{C['n_events']} events")
    print(f"wrote {a.outdir}  ({tot / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
