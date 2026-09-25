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
and needs no such care, but its DEFINITION does: see the selector on the page,
which defaults to >= 2 wrong hits rather than the much rarer tp_key < 0. Resolution needs the per-menu DR, which is why the track
table is PRE-DR and sorted:

  SORTED BY tp_key, WITH NOISE-SEEDED ROWS (tp_key < 0) IN A CONTIGUOUS BLOCK.
  The per-menu duplicate removal in the browser is then a linear scan over the
  sorted region instead of a hash over millions of rows. At ~12.6M rows that is
  the difference between interactive and not. Note those rows are NOT "the
  fakes" -- see n_no_owner below.

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
import argparse, re, glob, json, os, sys, time
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
    # the int64 keys of the SURVIVING rows, for the exact track -> TP row link.
    # The float32 "key" column cannot do this: (event << 20) | tpIdx needs 30
    # bits and float32 holds 24, so at event 999 neighbouring keys 128 apart
    # collapse onto the same value.
    return A, C["found"][keep], C["found_dr"][keep], np.asarray(C["key"])[keep]


# COLUMNS THE BROWSER ACTUALLY READS. The MVA feature set is needed to TRAIN
# the scores here and never again: shipping all 54 columns put track.bin at
# 2.7 GB, and a browser caps a single ArrayBuffer near 2 GB, so the page failed
# to allocate and (before the loader was guarded) showed nothing at all. The
# discriminants are trained on the full set and only these travel.
SITE_TRACK_COLS = (
    "seed_idx", "tp_key", "tp_row",            # identity and the TP link
    "pt", "phi", "eta", "d0", "z0", "n_layers",  # axes and cuts
    "nhit", "n_own", "n_it_right", "n_it_wrong",
    "n_ot_right", "n_ot_wrong", "n_ot_combinatoric", "n_ot_unknown",
    "sig_kf_d0", "n_wrong",                    # composition axes
    "d_d0", "d_z0", "d_kappa", "d_phi0", "d_cot",   # all five residuals
)


def trim_for_browser(T, cols, mva_cols):
    """Keep the browser set plus whatever scores were trained, in order."""
    keep = [c for c in SITE_TRACK_COLS if c in cols] + list(mva_cols)
    idx = [cols.index(c) for c in keep]
    return np.ascontiguousarray(T[:, idx]), keep


def build_track_table(C, tp_keys=None):
    """Concatenate every seed's track rows, add pt and the MVA score, and sort.

    pt IS ADDED HERE, not derived in the browser, because the axis menu offers
    TP-table names and the track table stores inv_pt. Leaving them out of step
    meant cols.indexOf('pt') returned -1 and trk[i*nc - 1] read the LAST COLUMN
    OF THE PREVIOUS ROW -- valid memory, plausible values, and every track-based
    plot silently binning d_z0 of track i-1 as though it were pT. Same for phi
    (stored phi0) and n_layers (stored nhit), which are aliased below.
    """
    if not C["tracks"]:
        raise SystemExit("census has no track rows; rerun with --export-tracks")
    T = np.concatenate([C["tracks"][i] for i in sorted(C["tracks"])])
    cols = list(C["track_cols"])
    col = {c: j for j, c in enumerate(cols)}
    add = {"pt": 1.0 / np.maximum(np.abs(T[:, col["inv_pt"]]), 1e-6),
           "phi": T[:, col["phi0"]],
           "n_layers": T[:, col["nhit"]]}
    T = np.concatenate([T] + [v.astype(np.float32)[:, None]
                              for v in add.values()], axis=1)
    cols += list(add)
    col = {c: j for j, c in enumerate(cols)}
    C["track_cols"] = cols
    # ---- exact link from a track to its owner's row in the TP table -------
    # This is what the DELIVERED efficiency needs: mark a TP as found when a
    # surviving track is majority-owned by it. Done on int64 keys rebuilt from
    # the event and the owner's tpIdx, both small enough to survive float32,
    # because the packed tp_key column does not.
    if tp_keys is not None and "own_tpidx" in col:
        import export_baseline as EB       # the one definition of the link
        row = EB.link_tp_rows(T[:, col["event"]], T[:, col["own_tpidx"]], tp_keys)
        okrow = row >= 0
        T = np.concatenate([T, row[:, None].astype(np.float32)], axis=1)
        C["track_cols"] = list(C["track_cols"]) + ["tp_row"]
        col = {c: j for j, c in enumerate(C["track_cols"])}
        print(f"  tp_row: {int(okrow.sum()):,} of {len(T):,} tracks linked to a "
              f"TP row in the denominator")
    elif tp_keys is not None:
        print("  no own_tpidx column in this census; delivered efficiency will "
              "be unavailable on the page")
    k = T[:, col["tp_key"]]
    # fakes (tp_key < 0) first as one contiguous block, then real rows sorted by
    # TP so the per-menu DR is a linear scan
    order = np.lexsort((k, k >= 0))
    T = T[order]
    # tp_key < 0 now means NO HIT ON THE TRACK BELONGS TO ANY TrackingParticle,
    # since truth is referenced to the majority owner rather than to the seed
    # cluster. That is a rarer and stricter condition than the old
    # "seed cluster is noise", and it is still NOT the fake rate: the page
    # offers three definitions and defaults to >= 2 wrong hits.
    n_no_owner = int((T[:, col["tp_key"]] < 0).sum())
    # scores are built AFTER the sort and travel in their own uint8 table, so
    # the float32 track table stays inside the browser's allocation limit
    S, score_names, mva_info = build_scores(T, col, C)
    C["_mva_info"] = mva_info
    C["_scores"] = S
    C["_score_names"] = score_names
    # the permutation is returned so the hit lists can be put in the SAME order;
    # the conflict graph indexes rows of the sorted table, not the raw one
    return T, col, n_no_owner, order


# THE SHIPPED DISCRIMINANT SET.
#
# Three legacy scores, kept so the page can show what changed, plus the variant
# matrix: six targets (cleanliness and the accuracy of each of the five fitted
# parameters) x two variants split on the FITTED |d0| at 200 um, so the variant
# a track is scored under is decidable online exactly as Phase-2 routes tracks
# through prompt versus extended tracking.
#
# EVERY VARIANT SCORE IS TRAINED WITHOUT THE d0-DERIVED FEATURES. Measured: they
# cost 0.002 AUC on cleanliness but they triple the rejection of well-measured
# strongly displaced tracks (36.7% -> 11.3% retention above 1 mm) and they
# invert the prompt->displaced transfer (AUC 0.707 -> 0.453), because on a
# prompt-dominated sample "small fitted |d0|" nearly restates "small residual".
#
# Scores travel as uint8 over [0, 1]. A threshold slider steps 0.01, so 8 bits
# is finer than the control that reads them, and float32 would have put the
# page back over the browser's ArrayBuffer limit.
D0_LAUNDER_FEATS = ("d0", "d0_over_sigma", "sig_kf_d0")
LEGACY_DEFS = {
    # Kept for COMPARISON, not because it wins: these are the six features the
    # real TrackQuality MVA is given, so this score says what the deployed
    # system's discriminant achieves on our tracks.
    "mva_tq": {"feats": ("cot", "z0", "nhit", "n_miss_interior",
                         "chi2_rphi_per_layer", "chi2_rz_per_layer"),
               "pop": "all"},
    "mva_clean": {"feats": None, "pop": "all"},
}
# mva_refit DROPPED. Measured in-window on 1.2M tracks it scored 0.994 / 0.965
# for clean against mva_clean's 0.996 / 0.984 (prompt / displaced windows) --
# dominated in both, and restricted to the refit population on top.

# ---- which resolution taggers ship, and under what name -------------------
# EIGHT SCORES, NOT FIFTEEN. Measured in-window (each variant judged only where
# its |fitted d0| window applies, which is the only fair test) on the deployed
# table, three resolution axes are distinguishable and the rest are copies:
#
#   d0     ORTHOGONAL. Spearman 0.18-0.33 against every other score, and the
#          only score that wins its own target: 0.730 prompt / 0.815 displaced
#          against ~0.69 / ~0.43 for everything else. The prompt/displaced
#          split earns its keep here and nowhere else -- mva_d0_prompt falls to
#          0.715 in the displaced window where mva_d0_displaced holds 0.815.
#   rz     z0 and cot(theta) are ONE axis, being the two r-z parameters the
#          same two measurements determine. The z0-trained score reaches 0.961
#          on the cot target against cot's own 0.969, so shipping both bought
#          0.008. Named rz rather than z0 because it ranks both.
#   rphi   1/pT and phi0, likewise: the invpt-trained score reaches 0.945 /
#          0.819 on the phi0 target against phi0's own 0.954 / 0.843.
#
# NOT named for d0 even though d0 is also an r-phi parameter: it needs its own
# score precisely because it does NOT move with the other two.
#
# The clean variants are dropped because contamination does not depend on
# displacement the way a resolution does -- unsplit mva_clean matched or beat
# both variants in both windows (0.996/0.984 against 0.997/0.978 and
# 0.996/0.984).
#
# RE-CHECK THESE AGAINST THE AUC LINES A FRESH EXPORT PRINTS. The numbers above
# were measured on scores trained before the third-order helix terms, which move
# every residual target by 12-45%; the correlation structure is set by the fit's
# parameter correlations and should not reshuffle, but that is an argument, not
# a measurement.
VARIANT_KEEP = {"d0": "d0", "z0": "rz", "invpt": "rphi"}
D0_SPLIT_CM = 0.02
SCORE_SCALE = 255.0


def _train_one(X, y, ok, is_te, rng, seed=1, max_train=1_500_000):
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    from sklearn.metrics import roc_auc_score
    tr = np.flatnonzero(ok & ~is_te)
    te = np.flatnonzero(ok & is_te)
    if len(tr) > max_train:
        tr = tr[rng.choice(len(tr), max_train, replace=False)]
    if len(tr) < 20000 or len(te) < 5000 or len(np.unique(y[tr])) < 2:
        return None, None, None
    g = GB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[tr], y[tr])
    auc = float(roc_auc_score(y[te], g.predict_proba(X[te])[:, 1]))
    return g, auc, te


def _score_all(g, X, n):
    out = np.zeros(n, np.float32)
    for i in range(0, n, 1_000_000):
        j = min(i + 1_000_000, n)
        out[i:j] = g.predict_proba(X[i:j])[:, 1].astype(np.float32)
    return out


def build_angle_table(C, order, seed_track_counts):
    """Per-cluster angle residuals, with track_row remapped to the sorted table.

    THE JOIN IS THE WHOLE DIFFICULTY. The census stores each seed's angle rows
    with track_row indexed inside THAT SEED's block, while the exported track
    table is every seed concatenated and then sorted by TP. So each block's
    indices are offset by the running total, and then pushed through the inverse
    of the sort permutation. Getting this wrong would attach every residual to
    the wrong track -- silently, since all the values stay in range.
    """
    if not C.get("angles"):
        return None, []
    cols = list(C["angle_cols"])
    jr = cols.index("track_row")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    parts, off = [], 0
    for i in sorted(C["angles"]):
        A = C["angles"][i].copy()
        if len(A):
            g = A[:, jr].astype(np.int64) + off
            ok = (g >= 0) & (g < len(inv))
            A = A[ok]
            A[:, jr] = inv[g[ok]].astype(np.float32)
            parts.append(A)
        off += seed_track_counts.get(i, 0)
    if not parts:
        return None, []
    A = np.concatenate(parts)
    return A.astype(np.float32), cols


def build_scores(T, col, C):
    """Train every discriminant and return (uint8 score table, names, info)."""
    try:
        from sklearn.metrics import roc_auc_score
        import quality_taggers as QT          # one definition of the targets
    except ImportError as e:
        print(f"  no scores: {e}")
        return None, [], {}
    n = len(T)
    rng = np.random.default_rng(1)
    ev = T[:, col["event"]]
    is_te = np.isin(ev, np.unique(ev)[::3])   # split by EVENT, never by track
    d0f = np.abs(T[:, col["d0"]])
    WIN = {"prompt": d0f <= D0_SPLIT_CM, "displaced": d0f > D0_SPLIT_CM}
    d0t = np.abs(T[:, col["tp_d0"]])
    disp_true = np.isfinite(d0t) & (d0t > D0_SPLIT_CM)
    names, cols_out, info = [], [], {}

    def fmat(feats):
        return np.nan_to_num(T[:, [col[f] for f in feats]], posinf=0, neginf=0)

    # ---- the legacy three, on the full feature set ----------------------
    y_leg = (T[:, col["n_wrong"]] <= 1)
    refit = T[:, col["sysclass"]] != KF.SYS_MIX
    for name, spec in LEGACY_DEFS.items():
        feats = [c for c in (spec["feats"] or KF.MVA_FEATURES) if c in col]
        X = fmat(feats)
        ok = np.isfinite(X).all(axis=1)
        if spec["pop"] == "refit":
            ok = ok & refit
        g, auc, te = _train_one(X, y_leg, ok, is_te, rng)
        if g is None:
            continue
        sc = _score_all(g, X, n)
        names.append(name); cols_out.append(sc)
        info[name] = {"auc": round(auc, 4), "features": feats,
                      "population": spec["pop"], "variant": None,
                      "target": "n_wrong <= 1 (owner-referenced)",
                      "keep_rate_at_0.9_clean": _fairness(sc, y_leg, disp_true,
                                                          d0t)}
        print(f"  {name:<22} AUC {auc:.4f}  ({len(feats)} features)")

    # ---- the variant matrix, d0-derived features withheld ---------------
    feats = [c for c in KF.MVA_FEATURES
             if c in col and c not in D0_LAUNDER_FEATS]
    X = fmat(feats)
    finite = np.isfinite(X).all(axis=1)
    for tname in QT.TARGETS:
        if tname not in VARIANT_KEEP:
            continue
        for wname, win in WIN.items():
            y, ok, thr = QT.make_target(T, col, tname, win)
            if y is None:
                continue
            g, auc, te = _train_one(X, y, ok & finite, is_te, rng)
            if g is None:
                print(f"  {tname}/{wname}: too few rows, skipped")
                continue
            sc = _score_all(g, X, n)
            nm = f"mva_{VARIANT_KEEP[tname]}_{wname}"
            names.append(nm); cols_out.append(sc)
            info[nm] = {"auc": round(auc, 4), "features": feats,
                        "population": f"|fitted d0| "
                                      f"{'<=' if wname == 'prompt' else '>'} "
                                      f"{D0_SPLIT_CM * 1e4:.0f} um",
                        "variant": wname,
                        "ranks": ({"rz": "d_z0 and d_cot",
                                   "rphi": "d_kappa and d_phi0"}
                                  .get(VARIANT_KEEP[tname],
                                       QT.TARGETS[tname][0])),
                        "target": (f"|{QT.TARGETS[tname][0]}| below the "
                                   f"clean-track 68.3 percentile"
                                   if tname != "clean"
                                   else "n_wrong <= 1 (owner-referenced)"),
                        "keep_rate_at_0.9_clean": _fairness(sc, y, disp_true,
                                                            d0t)}
            print(f"  {nm:<22} AUC {auc:.4f}")
    if not names:
        return None, [], {}
    S = np.stack([np.clip(c, 0, 1) for c in cols_out], axis=1)
    return (S * SCORE_SCALE).round().astype(np.uint8), names, info


def _fairness(sc, y, disp_true, d0t):
    """Retention of GOOD tracks at a 0.9 cut, prompt against displaced."""
    out = {}
    for lab, m in (("prompt", ~disp_true), ("displaced", disp_true)):
        mm = m & (y == 1) & np.isfinite(sc)
        out[lab] = round(float((sc[mm] >= 0.9).mean()), 4) if mm.sum() > 200 \
            else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=str(Path(__file__).parent / "cache"))
    ap.add_argument("--arch-tracks", default=None,
                    help="npz of parallel-architecture rows from "
                         "arch_comparison.py --export-tracks")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--baseline-input", default=None,
                    help="ntuples for the Phase-2 baseline (the L1TTrack "
                         "collection CMSSW produced). Omit to leave that "
                         "scenario out of the site.")
    ap.add_argument("--baseline-census", default=None,
                    help="census cache for the re-emulated baseline: OT seeds, "
                         "OT targets, no inner tracker. Omit to leave it out.")
    ap.add_argument("--baseline-nev", type=int, default=None)
    ap.add_argument("--baseline-ptmin", type=float, default=2.0)
    a = ap.parse_args()
    ds = sorted(glob.glob(f"{a.cache_dir}/tpcensus_*"))
    if not ds:
        raise SystemExit(f"no census under {a.cache_dir}")
    C = TF.load(ds[-1], nev=a.nev, with_tracks=True)
    os.makedirs(a.outdir, exist_ok=True)

    has_soft = any(getattr(sd, "soft", False) for sd in C["seeds"])

    def _build_menu(il, with_soft):
        base = SA.enumerate_seeds(il)
        return base + (SA.soft_seeds(base) if with_soft else [])

    TPA, found, found_dr, tp_keys = build_tp_table(C)
    seed_counts = {i: len(v) for i, v in C["tracks"].items()}
    T, col, n_no_owner, order_orig = build_track_table(C, tp_keys)
    ANG, ang_cols = build_angle_table(C, order_orig, seed_counts)
    if ANG is not None:
        print(f"  angle table: {len(ANG):,} cluster residuals "
              f"({ANG.nbytes / 1e6:.0f} MB, 1 track in {TF.ANGLE_SAMPLE})")
    conflict = None
    if C.get("hits"):
        # the conflict graph, built once here rather than per interaction in the
        # browser. 178 s at full scale against ~10 s per menu change, so it
        # belongs on this side of the wire.
        H = np.concatenate([C["hits"][i] for i in sorted(C["hits"])])[order_orig]
        t0 = time.perf_counter()
        edges, st = DRS.conflict_edges(H)
        # ea/eb, not a/b: `a` is the argparse namespace in this scope, and
        # shadowing it here is what broke the first full-scale export run. The
        # earlier a_outdir workaround treated the symptom.
        ea = np.r_[edges[:, 0], edges[:, 1]]
        eb = np.r_[edges[:, 1], edges[:, 0]]
        o = np.argsort(ea, kind="stable")
        ea, eb = ea[o], eb[o]
        indptr = np.searchsorted(ea, np.arange(len(T) + 1)).astype(np.int32)
        # descending rank_score, MENU-INDEPENDENT so it ships once
        rank_order = np.argsort(-T[:, col["rank_score"]],
                                kind="stable").astype(np.int32)
        indptr.tofile(os.path.join(a.outdir, "conf_indptr.bin"))
        eb.astype(np.int32).tofile(os.path.join(a.outdir, "conf_nbr.bin"))
        rank_order.tofile(os.path.join(a.outdir, "conf_order.bin"))
        conflict = {"indptr": "conf_indptr.bin", "neighbours": "conf_nbr.bin",
                    "order": "conf_order.bin", "edges": int(len(edges)),
                    "min_shared": SA.MIN_SHARED_LAYERS,
                    "build_seconds": round(time.perf_counter() - t0, 1), **st}
    # trim BEFORE the manifest is built, so the column list it advertises is
    # the one actually written
    n_before = len(C["track_cols"])
    T, C["track_cols"] = trim_for_browser(T, list(C["track_cols"]), [])
    print(f"  track table trimmed {n_before} -> {len(C['track_cols'])} browser "
          f"columns, {T.nbytes / 1e9:.2f} GB")
    meta = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_events": C["n_events"],
        "source": ds[-1],
        # so the page can tell whether a pT band is below the seeding threshold
        # rather than asserting it in baked-in prose
        "ptmin": float(C.get("ptmin", 2.0)),
        "mva": C.get("_mva_info", {}),
        # THE LOSS FUNNEL, per seed, already counted during the census:
        #   seed_objects -> pairs -> cand (projections tried) ->
        #   before_minlayers -> before_chi2 -> fit -> before_dr -> owned
        # This is what says WHERE a seed's tracks disappear, which no per-track
        # table can show: the candidates killed before a fit never become rows.
        "counters": {str(i): {k: float(v) for k, v in d.items()}
                     for i, d in C["counters"].items() if d},
        # A TRUE FUNNEL, in track-like units. Two counters are deliberately
        # NOT in it: `before_minlayers` is the same quantity as `seed_objects`
        # (both int(len(gA)) in run_seed), and `cand` counts projection MATCHES
        # summed over target layers -- a hit-level cost, not a track stage, so
        # putting it in the chain would imply a 45M -> 155k collapse that means
        # nothing. It is reported separately as the projection cost.
        "funnel_stages": ["pairs", "seed_objects", "before_chi2", "fit",
                          "owned"],
        "funnel_labels": {"pairs": "cluster pairs formed",
                          "seed_objects": "seeds after their own layers",
                          "before_chi2": "tracks passing the layer rule",
                          "fit": "tracks passing chi2",
                          "owned": "surviving duplicate removal"},
        "cost_counters": ["cand", "targets_followed"],
        # soft marks a dedicated sub-2-GeV recovery entry, which the page
        # groups separately: it is an IT-only seed by layers but it runs at its
        # own floor under its own rule, so lumping it with the standard IT
        # seeds would make the pT extension impossible to switch off.
        "seeds": [{"idx": i, "tag": t,
                   "layers": list(C["seeds"][i].layers),
                   "arity": C["seeds"][i].arity,
                   "soft": bool(getattr(C["seeds"][i], "soft", False)),
                   "sysclass": KF.sysclass_of(C["seeds"][i].layers)}
                  for i, t in enumerate(C["seed_tags"])],
        # which seeds each activeSP build can form, so the page can gate the
        # seed selector on the build rather than offering seeds a build cannot
        # physically produce
        # THE BUILD MAP MUST LIST THE RECOVERY SEEDS TOO. The page gates the
        # seed menu on this set, so a tag missing here is a seed the user
        # cannot select no matter what the census contains -- which is exactly
        # how the sub-2-GeV entries went invisible: enumerate_seeds returns the
        # standard menu, and the recovery duplicates are appended afterwards by
        # tp_findability. Mirror that here, and only when this census actually
        # has them, so a 2 GeV-only site does not advertise seeds it lacks.
        "builds": {m: sorted({s.tag for s in _build_menu(
            [TF.IL_OF[i] for i, ch in enumerate(m) if ch == "A"], has_soft)})
            for m in TF.BUILD_MASKS},
        "tp": {"file": "tp.bin", "cols": list(TP_COLS), "rows": int(len(TPA)),
               "dtype": "float32"},
        "found": {"file": "found.bin", "rows": int(len(TPA)),
                  "words": int(found.shape[1]) * 2, "dtype": "uint32"},
        "found_dr": {"file": "found_dr.bin", "rows": int(len(TPA)),
                     "words": int(found_dr.shape[1]) * 2, "dtype": "uint32"},
        **({"angle": {"file": "angle.bin", "cols": ang_cols,
                      "rows": int(len(ANG)), "dtype": "float32",
                      "sample": TF.ANGLE_SAMPLE}}
           if ANG is not None else {}),
        **({"score": {"file": "score.bin",
                      "cols": list(C["_score_names"]),
                      "rows": int(len(T)), "dtype": "uint8",
                      "scale": SCORE_SCALE}}
           if C.get("_scores") is not None else {}),
        "track": {"file": "track.bin", "cols": list(C["track_cols"]),
                  "rows": int(len(T)), "dtype": "float32",
                  "n_no_owner": n_no_owner,
                  "sorted_by": "tp_key, noise-seeded rows first"},
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
            "Track truth is referenced to the MAJORITY OWNER of the hits, not "
            "to the seed cluster. Measured over 200 events, 45.8% of accepted "
            "tracks have an owner that is not the seed's TP, and the seed "
            "convention reported sigma(d0) = 1440 um where the owner value is "
            "372 um at 2 wrong hits.",
            "n_no_owner means no hit on the track belongs to any "
            "TrackingParticle. It is not the fake rate.",
            "sig_kf_d0 is the fit's CLAIMED error, not a resolution: measured "
            "1.6-3.4x optimistic on clean tracks, worsening with hit count. "
            "sigma(d0) panels are truth residuals and never use it.",
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
    if ANG is not None:
        ANG.tofile(os.path.join(a.outdir, "angle.bin"))
    if C.get("_scores") is not None:
        C["_scores"].tofile(os.path.join(a.outdir, "score.bin"))
        print(f"  {len(C['_score_names'])} scores as uint8: "
              f"{C['_scores'].nbytes / 1e6:.0f} MB")
    # THE BASELINES ARE PART OF THE SITE, not something bolted on afterwards.
    # They used to be written by a separate script run after this one, which
    # meant this manifest overwrote theirs and silently dropped both scenarios
    # while their .bin files sat in the directory unreferenced. Building them
    # here makes one command produce a complete site; export_baseline.py still
    # runs standalone, through the same build_block, for refreshing one of them
    # without repeating the whole export.
    import export_baseline as EB
    for src, arg in (("cmssw", a.baseline_input), ("census", a.baseline_census)):
        if not arg:
            print(f"  no --baseline-{'input' if src == 'cmssw' else 'census'}: "
                  f"the {src} baseline scenario is not in this site")
            continue
        A, key, fn, block, SCB = EB.build_block(
            src, inputs=arg if src == "cmssw" else None,
            cache_dir=arg if src == "census" else None,
            nev=a.baseline_nev, ptmin=a.baseline_ptmin,
            tp_keys=tp_keys, tp_pt=TPA[:, TP_COLS.index("pt")].astype(np.float64))
        A.tofile(os.path.join(a.outdir, fn))
        if SCB is not None:
            SCB.tofile(os.path.join(a.outdir, block["score"]["file"]))
        meta[key] = block
        print(f"  {key}: {len(A):,} tracks, {A.nbytes / 1e6:.1f} MB")
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
    # CACHE BUSTING. explorer.html and explorer_core.js are rewritten on every
    # export; a browser that keeps an old copy would run yesterday's code
    # against today's columns. Stamping the query string makes that impossible.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    hp = Path(a.outdir) / "explorer.html"
    if hp.exists():
        t = hp.read_text()
        t = re.sub(r'src="(explorer_core\.js|plotly\.min\.js)(\?v=[^"]*)?"',
                   lambda m: f'src="{m.group(1)}?v={stamp}"', t)
        hp.write_text(t)
    tot = sum(os.path.getsize(os.path.join(a.outdir, f))
              for f in os.listdir(a.outdir))
    print(f"{len(TPA):,} TP rows, {len(T):,} track rows "
          f"({n_no_owner:,} with no owner), "
          f"{C['n_events']} events")
    print(f"wrote {a.outdir}  ({tot / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
