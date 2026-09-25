"""Write the Phase-2 baseline tracks into the site, from either source.

Two datasets share one column layout so the page can plot them on the same axes:

  --source cmssw    the L1TTrack collection CMSSW produced. What the system
                    delivers today, and the thing SmartPixels has to beat.
  --source census   our own emulation run in the same configuration -- OT
                    seeds, OT targets, no inner tracker anywhere. Comparing it
                    against the CMSSW one says whether the emulator reproduces
                    the system when given the system's inputs, which is what
                    makes any IT-vs-no-IT conclusion trustworthy.

For the CMSSW source, per-stub truth is joined from L1TOTStub on the full stub
identity (ot_oracle.join_stub_truth: event, detId, bend, x/y/z within tolerance):
L1TTrack carries per-track truth but not per-stub truth, and the contamination
axes the page bins on are per stub. An earlier (event, detId, z) key was
ambiguous for ~half of PU200 OT stubs and silently attached a neighbour's truth.
"""
from __future__ import annotations
import sys, json, glob, argparse
from pathlib import Path
import numpy as np
from ngtagger.truth_helix import tp_phi0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M      # noqa: E402
import kf_emulation as KF               # noqa: E402
import tp_findability as TF             # noqa: E402
from ot_oracle import load, join_stub_truth   # noqa: E402

# THE TP LINK. Both baselines carry, per track, its majority owner's row in the
# site's own TP census (tp_row) and how many of its hits that owner holds
# (n_own), which is exactly what the page's DELIVERED efficiency needs: a TP is
# found when a surviving track is owned by it. Keyed on (event << 20) | tpIdx,
# the same int64 key the census and export_site's own track table use, so all
# three collections share one denominator. The seed-truthful efficiency stays
# undefined for a baseline: its tracks never entered our seed bitmaps.
LINK_COLS = ("n_own", "tp_row")
# Tracks whose owner owns every hit (n_wrong == 0) must name the same particle
# on both sides of the link, so their recorded matched-TP pT must equal the
# census TP's pT at tp_row. Anything short of this means the event numbering of
# the two inputs disagrees, and every efficiency would be silently wrong.
LINK_MIN_AGREEMENT = 0.99
LINK_PT_RTOL = 1e-3


def link_tp_rows(event, tpidx, tp_keys):
    """Row of (event, tpIdx) in the site's TP table, or -1. The one definition
    of the track -> TP link, shared by export_site's track table."""
    keys = np.asarray(tp_keys, np.int64)
    assert np.all(np.diff(keys) > 0), \
        "TP table keys are not strictly ascending; tp_row would be wrong"
    ev = np.asarray(event).astype(np.int64)
    ti = np.asarray(tpidx).astype(np.int64)
    kk = (ev << 20) | np.maximum(ti, 0)
    pos = np.clip(np.searchsorted(keys, kk), 0, max(len(keys) - 1, 0))
    ok = (len(keys) > 0) & (keys[pos] == kk) & (ti >= 0)
    return np.where(ok, pos, -1)


def check_tp_link(label, tp_pt_track, row, tp_pt_table, clean):
    m = clean & (row >= 0) & np.isfinite(tp_pt_track) & (tp_pt_track > 0)
    if not m.any():
        raise SystemExit(f"{label}: no clean track links to the TP census; the "
                         f"inputs do not share its events")
    agree = np.abs(tp_pt_track[m] - tp_pt_table[row[m]]) \
        <= LINK_PT_RTOL * tp_pt_track[m]
    frac = float(agree.mean())
    print(f"  {label}: {int((row >= 0).sum()):,} of {len(row):,} tracks linked "
          f"to a TP row; clean-track pT agreement {100 * frac:.2f}% "
          f"({int(m.sum()):,} clean linked tracks)")
    if frac < LINK_MIN_AGREEMENT:
        raise SystemExit(
            f"{label}: only {100 * frac:.2f}% of clean linked tracks name a TP "
            f"with their own matched pT -- the baseline inputs and the TP census "
            f"are not the same events in the same order. Refusing to write a "
            f"tp_row that would mislabel the delivered efficiency.")
    return frac


COLS = ("pt", "phi", "eta", "d0", "z0", "n_layers", "nhit",
        "n_ot_right", "n_ot_wrong", "n_ot_combinatoric", "n_ot_unknown",
        "n_wrong", "tp_pt", "tp_eta", "tp_d0", "tp_z0",
        "d_d0", "d_z0", "d_kappa", "d_phi0", "d_cot")
EXTRA_CMSSW = ("genuine", "combinatoric", "unknown", "trkMVA1")


def from_cmssw(spec, nev, ptmin, eta_max, tp_keys=None, tp_pt=None):
    T, S, O = load(spec, nev)
    tpidx, flag, hit = join_stub_truth(S, O)
    bar = S["isBarrel"] > 0
    frac = float(hit[bar].mean()) if bar.any() else 0.0
    print(f"  stub truth join: {100 * frac:.2f}% of barrel track stubs matched")
    if frac < 0.999:
        raise SystemExit(
            f"the stub-identity join matched only {100 * frac:.2f}% of barrel stubs; "
            f"the per-stub counters would be built on a subset and would not be "
            f"comparable to the emulation's")

    off = np.concatenate([[0], np.cumsum(T["_n"])])
    gid = off[S["event"]] + S["trackIdx"]
    ok = (S["trackIdx"] >= 0) & bar
    ntrk = len(T["pt"])
    nst = np.zeros(ntrk, np.int32)
    np.add.at(nst, gid[ok], 1)

    # majority owner over the track's genuine stubs -- the same rule
    # KF.track_rows applies, so "wrong" means the same thing on both sides
    gen = ok & (flag == 1) & (tpidx >= 0)
    owner = np.full(ntrk, -1, np.int64)
    order = np.lexsort((tpidx[gen], gid[gen]))
    g_s, t_s = gid[gen][order], tpidx[gen][order]
    if len(g_s):
        new = np.r_[True, (g_s[1:] != g_s[:-1]) | (t_s[1:] != t_s[:-1])]
        st = np.flatnonzero(new)
        cnt = np.diff(np.r_[st, len(g_s)])
        best = np.zeros(ntrk, np.int32)
        for g_, t_, c_ in zip(g_s[st], t_s[st], cnt):
            if c_ > best[g_]:
                best[g_], owner[g_] = c_, t_
    right = gen & (tpidx == owner[gid])

    def per_track(mask):
        v = np.zeros(ntrk, np.float32); np.add.at(v, gid[mask], 1); return v
    n_r, n_w = per_track(right), per_track(gen & ~right)
    n_c, n_u = per_track(ok & (flag == 2)), per_track(ok & (flag == 3))

    sel = (T["pt"] >= ptmin) & (np.abs(T["eta"]) <= eta_max) & (nst >= 4)
    # tp_d0 carries +-999 cm sentinels for unmatched tracks
    fin = sel & np.isfinite(T["tp_d0"]) & (np.abs(T["tp_d0"]) < 10.0) \
        & (T["tp_pt"] > 0)
    print(f"  {int(sel.sum()):,} barrel tracks pT>={ptmin} |eta|<={eta_max}, "
          f"{int(fin.sum()):,} with a physical matched particle")

    def wrap(x):
        return np.arctan2(np.sin(x), np.cos(x))
    d = {
        "pt": T["pt"], "phi": T["phi"], "eta": T["eta"],
        # L1TTrack_d0 and tp_d0 share CMSSW's convention, and the emulation
        # computes in that convention throughout, so the axes agree
        "d0": T["d0"], "z0": T["z0"],
        "n_layers": nst.astype(np.float32), "nhit": nst.astype(np.float32),
        "n_ot_right": n_r, "n_ot_wrong": n_w,
        "n_ot_combinatoric": n_c, "n_ot_unknown": n_u,
        "n_wrong": n_w + n_c + n_u,
        "tp_pt": T["tp_pt"], "tp_eta": T["tp_eta"],
        "tp_d0": np.where(fin, T["tp_d0"], np.nan),
        "tp_z0": np.where(fin, T["tp_z0"], np.nan),
        "d_d0": np.where(fin, T["d0"] - T["tp_d0"], np.nan),
        "d_z0": np.where(fin, T["z0"] - T["tp_z0"], np.nan),
        "d_kappa": np.where(fin, 1.0 / np.maximum(T["pt"], 1e-6)
                            - 1.0 / np.maximum(T["tp_pt"], 1e-6), np.nan),
        "d_phi0": np.where(fin, wrap(T["phi"] - tp_phi0(T)), np.nan),
        "d_cot": np.where(fin, np.sinh(T["eta"]) - np.sinh(T["tp_eta"]), np.nan),
        "genuine": T["genuine"], "combinatoric": T["combinatoric"],
        "unknown": T["unknown"], "trkMVA1": T["trkMVA1"],
    }
    keep = np.flatnonzero(sel)
    names = list(COLS) + list(EXTRA_CMSSW)
    info = {"join_barrel_frac": frac}
    if tp_keys is not None:
        # owner = majority owner over the track's genuine stubs, as above; its
        # stubs are the track's owned hits (no IT hits on a baseline)
        d["n_own"] = n_r
        d["tp_row"] = link_tp_rows(T["event"], owner, tp_keys).astype(np.float32)
        info["tp_link_agreement"] = check_tp_link(
            "CMSSW baseline", np.asarray(T["tp_pt"], np.float64)[keep],
            d["tp_row"][keep].astype(np.int64), tp_pt,
            (d["n_wrong"] == 0)[keep] & fin[keep])
        names += list(LINK_COLS)
    A = np.stack([np.asarray(d[c], np.float64)[keep] for c in names],
                 axis=1).astype(np.float32)
    # THE COLLECTION'S OWN DISCRIMINANT, in both precisions. L1TTrack carries
    # only the PROMPT track-quality MVA: trkMVA2/3 and hwMVAOther are zero on
    # every track (the displaced model runs on L1TExtTrack, a different
    # collection), so slot B stays empty rather than holding an invented score.
    hw = np.asarray(T["hwMVAQuality"], np.float64)[keep]
    scores = {"cmssw_tq_hw3": hw / 7.0,
              "cmssw_tq_float": np.asarray(T["trkMVA1"], np.float64)[keep]}
    score_info = {
        "cmssw_tq_hw3": {
            "what": "L1TTrack_hwMVAQuality / 7: the 3-bit track-quality word the "
                    "hardware DELIVERS. 8 levels (k/7, k = 0..7), so a threshold "
                    "only changes the selection where it crosses a level.",
            "levels": 8},
        "cmssw_tq_float": {
            "what": "L1TTrack_trkMVA1: the same prompt track-quality MVA at "
                    "emulator float precision, before it is packed into 3 bits. "
                    "A comparison, not what the system delivers."}}
    return A, names, info, (scores, score_info, ["cmssw_tq_hw3", ""])


def _train_tq_variants(T, c, keep):
    """mva_tq_prompt / mva_tq_displaced on the re-emulated tracks themselves.

    The six features the real TrackQuality MVA is given (export_site's mva_tq),
    target n_wrong <= 1, one model per |fitted d0| window split at 200 um --
    the prompt/displaced split every other variant score uses, decidable online.
    Each model scores EVERY track; it is judged (AUC) only in its own window, on
    held-out events (every third event, never split by track).
    """
    import export_site as ES
    feats = ES.LEGACY_DEFS["mva_tq"]["feats"]
    X = np.nan_to_num(T[np.ix_(keep, [c[f] for f in feats])].astype(np.float64),
                      posinf=0, neginf=0)
    y = T[keep, c["n_wrong"]] <= 1
    ev = T[keep, c["event"]]
    is_te = np.isin(ev, np.unique(ev)[::3])
    d0f = np.abs(T[keep, c["d0"]])
    d0t = np.abs(T[keep, c["tp_d0"]])
    disp_true = np.isfinite(d0t) & (d0t > ES.D0_SPLIT_CM)
    rng = np.random.default_rng(1)
    scores, info = {}, {}
    for wname, win in (("prompt", d0f <= ES.D0_SPLIT_CM),
                       ("displaced", d0f > ES.D0_SPLIT_CM)):
        nm = f"mva_tq_{wname}"
        g, auc, _ = ES._train_one(X, y, win & np.isfinite(X).all(1), is_te, rng)
        if g is None:
            print(f"  {nm}: too few tracks in the {wname} window, not trained")
            continue
        sc = ES._score_all(g, X, len(X))
        scores[nm] = sc
        info[nm] = {"auc": round(auc, 4), "features": list(feats),
                    "population": f"|fitted d0| {'<=' if wname == 'prompt' else '>'} "
                                  f"{ES.D0_SPLIT_CM * 1e4:.0f} um (re-emulated baseline)",
                    "variant": wname, "target": "n_wrong <= 1 (owner-referenced)",
                    "keep_rate_at_0.9_clean": ES._fairness(sc, y, disp_true, d0t)}
        print(f"  {nm:<22} AUC {auc:.4f} (held-out events, {wname} window)")
    return scores, info


def from_census(cache_dir, nev, ptmin, eta_max, tp_keys=None, tp_pt=None):
    ds = sorted(glob.glob(f"{cache_dir}/tpcensus_*"))
    if not ds:
        raise SystemExit(f"no census under {cache_dir}")
    C = TF.load(ds[-1], nev=nev, with_tracks=True)
    if not C.get("tracks"):
        raise SystemExit("that census carries no track rows; re-run it with "
                         "--export-tracks")
    T = np.concatenate([v for _, v in sorted(C["tracks"].items())], axis=0)
    c = {n: i for i, n in enumerate(KF.TRACK_COLS)}
    tags = C["seed_tags"]
    if any(not t.startswith("OL") for t in tags):
        raise SystemExit(
            f"this census contains non-OT seeds ({', '.join(tags)}); the "
            f"re-emulated baseline has to be OT seeds with OT targets only, or "
            f"it is not the same configuration as the CMSSW one")
    pt = 1.0 / np.maximum(np.abs(T[:, c["inv_pt"]]), 1e-6)
    eta = T[:, c["eta"]]
    nhit = T[:, c["nhit"]]
    sel = (pt >= ptmin) & (np.abs(eta) <= eta_max) & (nhit >= 4)
    print(f"  {int(sel.sum()):,} of {len(T):,} emulated tracks pass "
          f"pT>={ptmin} |eta|<={eta_max} and >= 4 hits")
    d = {"pt": pt, "phi": T[:, c["phi0"]], "eta": eta,
         "d0": T[:, c["d0"]], "z0": T[:, c["z0"]],
         "n_layers": nhit, "nhit": nhit,
         "n_ot_right": T[:, c["n_ot_right"]], "n_ot_wrong": T[:, c["n_ot_wrong"]],
         "n_ot_combinatoric": T[:, c["n_ot_combinatoric"]],
         "n_ot_unknown": T[:, c["n_ot_unknown"]],
         "n_wrong": T[:, c["n_wrong"]],
         "tp_pt": T[:, c["tp_pt"]], "tp_eta": T[:, c["tp_eta"]],
         "tp_d0": T[:, c["tp_d0"]], "tp_z0": T[:, c["tp_z0"]],
         "d_d0": T[:, c["d_d0"]], "d_z0": T[:, c["d_z0"]],
         "d_kappa": T[:, c["d_kappa"]], "d_phi0": T[:, c["d_phi0"]],
         "d_cot": T[:, c["d_cot"]]}
    keep = np.flatnonzero(sel)
    names = list(COLS)
    info = {"seeds": tags, "n_events": C["n_events"]}
    if tp_keys is not None:
        d["n_own"] = T[:, c["n_own"]]
        d["tp_row"] = link_tp_rows(T[:, c["event"]], T[:, c["own_tpidx"]],
                                   tp_keys).astype(np.float32)
        info["tp_link_agreement"] = check_tp_link(
            "re-emulated baseline", T[keep, c["tp_pt"]].astype(np.float64),
            d["tp_row"][keep].astype(np.int64), tp_pt,
            (T[keep, c["n_wrong"]] == 0))
        names += list(LINK_COLS)
    A = np.stack([np.asarray(d[x], np.float64)[keep] for x in names],
                 axis=1).astype(np.float32)
    scores, score_info = _train_tq_variants(T, c, keep)
    return A, names, info, (scores, score_info,
                            ["mva_tq_prompt" if "mva_tq_prompt" in scores else "",
                             "mva_tq_displaced" if "mva_tq_displaced" in scores else ""])


def build_block(source, *, inputs=None, cache_dir=None, nev=None,
                ptmin=2.0, eta_max=None, tp_keys=None, tp_pt=None):
    """(array, manifest key, filename, manifest block, score table) for one
    baseline. tp_keys / tp_pt are the site TP table's int64 keys and pT; with
    them the block carries the delivered-efficiency link (LINK_COLS).

    The single definition of what a baseline is. export_site calls this so a
    site is produced by ONE command; running this file directly calls it too,
    for refreshing one baseline without redoing a 20-minute site export.
    """
    eta_max = M.ETA_MATCHED if eta_max is None else eta_max
    if source == "cmssw":
        if not inputs:
            raise SystemExit("the CMSSW baseline needs input ntuples")
        A, names, info, SC = from_cmssw(inputs, nev, ptmin, eta_max, tp_keys, tp_pt)
        key, fn = "baseline", "baseline.bin"
        label = "Phase-2 baseline"
        what = ("The L1TTrack collection CMSSW produced, with per-stub truth "
                "joined from L1TOTStub on the full stub identity (event, detId, "
                "bend, x/y/z within 1e-4 cm).")
        extra = ""
    elif source == "census":
        if not cache_dir:
            raise SystemExit("the re-emulated baseline needs a census cache")
        A, names, info, SC = from_census(cache_dir, nev, ptmin, eta_max, tp_keys, tp_pt)
        key, fn = "baseline_emu", "baseline_emu.bin"
        label = "Phase-2 baseline (re-emulated)"
        what = ("Our emulation in the same configuration as the CMSSW "
                "baseline: OT seeds, OT targets, no inner tracker. Its job is "
                "to show whether the emulator reproduces the system given the "
                "system's inputs.")
        extra = (" PRE duplicate removal, where the CMSSW baseline is POST: "
                 "census track rows are kept per seed so the browser can run DR "
                 "over any seed subset, so the same particle appears once per "
                 "seed that found it. Resolutions are barely affected, since "
                 "duplicates are near-copies, but track COUNTS are not "
                 "comparable and neither is anything derived from them.")
    else:
        raise SystemExit(f"unknown baseline source {source!r}")
    block = {"file": fn, "cols": names, "rows": int(len(A)),
             "dtype": "float32", "label": label, "what": what,
             "ptmin": ptmin, "eta_max": eta_max,
             "caveat": ("Residuals are referenced to each track's own matched "
                        "particle. Efficiency: only the DELIVERED definition "
                        "applies, through tp_row (majority owner -> the site's TP "
                        "census), so it shares the seed-menu scenarios' "
                        "denominator; the seed-truthful one is undefined, since "
                        "these tracks never entered our seed bitmaps." + extra
                        if "tp_row" in names else
                        "Residuals are referenced to each track's own matched "
                        "particle, and this export has no tp_row, so no "
                        "efficiency; compare resolutions and contamination." + extra),
             **info}
    scores, score_info, default = SC
    SCB = None
    if scores:
        sn = list(scores)
        SCB = (np.stack([np.clip(np.asarray(scores[k], np.float64), 0, 1) for k in sn],
                        axis=1) * 255.0).round().astype(np.uint8)
        block["score"] = {"file": fn.replace(".bin", "_score.bin"), "cols": sn,
                          "rows": int(len(SCB)), "dtype": "uint8", "scale": 255.0,
                          "default": default, "mva": score_info}
    return A, key, fn, block, SCB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("cmssw", "census"), required=True)
    ap.add_argument("-i", "--input", default=None, help="ntuples, for cmssw")
    ap.add_argument("--cache-dir", default=None, help="census dir, for census")
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--eta-max", type=float, default=M.ETA_MATCHED)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--tp-census", default=None,
                    help="census whose TP table the site was built from, for the "
                         "tp_row link; defaults to the site manifest's 'source'")
    a = ap.parse_args()

    out = Path(a.outdir)
    mpath = out / "manifest.json"
    m = json.loads(mpath.read_text())
    tpc = a.tp_census or m.get("source")
    tp_keys = tp_pt = None
    if tpc:
        import export_site as ES
        root = Path(__file__).resolve().parents[2]
        tpd = Path(tpc) if Path(tpc).is_absolute() else root / tpc
        TPA, _, _, tp_keys = ES.build_tp_table(TF.load(str(tpd), nev=m.get("n_events")))
        if len(tp_keys) != m["tp"]["rows"]:
            raise SystemExit(f"{tpd} rebuilds {len(tp_keys):,} TP rows but the site "
                             f"holds {m['tp']['rows']:,}: not the census it was built from")
        tp_pt = TPA[:, ES.TP_COLS.index("pt")].astype(np.float64)
    else:
        print("  no TP census known: writing the baseline without tp_row")
    A, key, fn, block, SCB = build_block(
        a.source, inputs=a.input, cache_dir=a.cache_dir, nev=a.nev,
        ptmin=a.ptmin, eta_max=a.eta_max, tp_keys=tp_keys, tp_pt=tp_pt)
    A.tofile(out / fn)
    if SCB is not None:
        SCB.tofile(out / block["score"]["file"])
    m[key] = block
    mpath.write_text(json.dumps(m, indent=1))
    print(f"wrote {out/fn}  ({A.nbytes/1e6:.1f} MB, {len(A):,} tracks, "
          f"{len(block['cols'])} columns) as '{key}'")


if __name__ == "__main__":
    main()
