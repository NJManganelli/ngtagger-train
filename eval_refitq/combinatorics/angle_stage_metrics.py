"""Per-build figures of merit of one or more censuses, for comparing angle
configurations (M.ANGLE_STAGES, the resolution knob) side by side.

Each census is read with its tracks; per build (the census's "builds"):

  efficiency   over the site's denominator: TPs with >= 4 layers hit (any of
               IL1-IL4, OL1-OL6) and the stated true pT / |eta| range, split
               prompt / displaced at |true d0| = 200 um (the site's D0_SPLIT_CM),
               the displaced split again at 1 mm: "hf" 0.2-1 mm (heavy flavour),
               "ll" > 1 mm (long-lived).
    seed       seed-truthful after the census's duplicate removal: some entry of
               the build owns a track whose SEED clusters all belong to the TP
               (found_dr), as on the site with every seed of the build enabled
    deliv      delivered: a track surviving duplicate removal over the menu is
               majority-owned by the TP (more than half its hits, >= 3), the
               site's default
  menus        "std" = the build's standard entries, DR over them alone (the
               site with the recovery seeds deselected); "full" = standard +
               the 1 GeV recovery entries, DR over all (as the census ran it)
  cost         per event: cluster pairs examined (after the S2j pre-gate) and
               kept, seeds formed, fitted tracks, DR survivors; std and recovery
               entries separately
  fake         DR survivors with >= 2 wrong hits (the site's default definition)
  sigma        robust sigma (half the 15.9-84.1% interval) of d0 and z0
               residuals [um] of delivered tracks whose owner is in the
               denominator at pT >= 2, |eta| < 1.1: all of them, and those with
               >= 2 IT hits ("it2", where the angle stages can act)
  mva          (--mva) held-out AUC of a clean-track GBDT (n_wrong <= 1; split
               by event, every third event held out) on all tracks of the census,
               with every MVA feature and without the angle ones (S8 on / off)
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_findability as TF   # noqa: E402
import kf_emulation as KF     # noqa: E402
import seed_arity as SA       # noqa: E402
M = TF.M

D0_SPLIT_CM = 0.02
# the displaced split further: heavy flavour (0.2-1 mm, M.TRIPLET_D0_MAX_CM) and
# long-lived (> 1 mm)
HF_D0_MAX_CM = M.TRIPLET_D0_MAX_CM
ETA_RANGES = {"c": (0.0, 1.1), "f": (1.1, 1.8)}


def _rs(x):
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def tp_rows(C):
    nl = (np.unpackbits(C["hit_it"][:, None], axis=1).sum(1)
          + np.unpackbits(C["hit_ot"][:, None], axis=1).sum(1))
    return {"key": np.asarray(C["key"], np.int64), "pt": C["pt"],
            "aeta": np.abs(C["eta"]), "ad0": np.abs(C["d0"]), "nlay": nl}


def menu(C, entries):
    """Track rows and hit lists of every track of `entries`, in entry order
    (the census's DR order)."""
    es = [i for i in sorted(entries) if i in C["tracks"] and len(C["tracks"][i])]
    if not es:
        return None, None
    return (np.concatenate([C["tracks"][i] for i in es]),
            np.concatenate([C["hits"][i] for i in es]))


def delivered_keys(T, col):
    own, nh = T[:, col["n_own"]], T[:, col["nhit"]]
    ok = (T[:, col["own_tpidx"]] >= 0) & (own >= 3) & (own > 0.5 * nh)
    return M.tp_key(T[ok, col["event"]].astype(np.int64),
                    T[ok, col["own_tpidx"]].astype(np.int64)), ok


def eff_block(P, found, prefix, out, pt_lo, pt_hi):
    """Efficiency of `found` (bool over P) per |eta| range and prompt/displaced."""
    base = (P["nlay"] >= 4) & (P["pt"] >= pt_lo) & (P["pt"] < pt_hi)
    for er, (lo, hi) in ETA_RANGES.items():
        m = base & (P["aeta"] >= lo) & (P["aeta"] < hi)
        for sp, ms in (("all", m), ("prompt", m & (P["ad0"] <= D0_SPLIT_CM)),
                       ("disp", m & (P["ad0"] > D0_SPLIT_CM)),
                       ("hf", m & (P["ad0"] > D0_SPLIT_CM) & (P["ad0"] <= HF_D0_MAX_CM)),
                       ("ll", m & (P["ad0"] > HF_D0_MAX_CM))):
            n = int(ms.sum())
            out[f"{prefix}_{er}_{sp}"] = float(found[ms].mean()) if n else None
            out[f"den_{prefix.split('_')[-1]}_{er}_{sp}"] = n


def build_metrics(C, build, P, col, nev):
    ent = C["builds"][build]
    std = [i for i in ent if not C["seeds"][i].soft]
    soft = [i for i in ent if C["seeds"][i].soft]
    out = {"n_std": len(std), "n_soft": len(soft)}
    # ---- seed-truthful after the census DR (the build's own DR group) ----
    seed = np.zeros(len(P["key"]), bool)
    for i in ent:
        seed |= TF.seed_mask(C, i, after_dr=True)
    eff_block(P, seed, "seed_hi", out, 2.0, np.inf)
    eff_block(P, seed, "seed_lo", out, 1.0, 2.0)
    # ---- delivered, DR recomputed per menu -------------------------------
    for mname, idx in (("std", std), ("full", ent)):
        T, H = menu(C, idx)
        if T is None:
            continue
        surv = SA.duplicate_removal(H, T[:, col["rank_score"]].astype(np.float64))
        S = T[surv]
        k, okm = delivered_keys(S, col)
        dl = np.isin(P["key"], k)
        eff_block(P, dl, f"deliv_{mname}_hi", out, 2.0, np.inf)
        eff_block(P, dl, f"deliv_{mname}_lo", out, 1.0, 2.0)
        out[f"fit_ev_{mname}"] = len(T) / nev
        out[f"dr_ev_{mname}"] = len(S) / nev
        out[f"fake_{mname}"] = float((S[:, col["n_wrong"]] >= 2).mean()) if len(S) else None
        # resolution of delivered tracks, owner in the pT >= 2, |eta| < 1.1 denominator
        own_key = M.tp_key(S[:, col["event"]].astype(np.int64),
                           np.maximum(S[:, col["own_tpidx"]], 0).astype(np.int64))
        pos = np.clip(np.searchsorted(P["key"], own_key), 0, len(P["key"]) - 1)
        inden = okm & (P["key"][pos] == own_key) & (P["nlay"][pos] >= 4) \
            & (P["pt"][pos] >= 2.0) & (P["aeta"][pos] < 1.1)
        itb = S[:, col["n_it"]] >= 2
        for nm, sel in (("all", inden), ("prompt", inden & (P["ad0"][pos] <= D0_SPLIT_CM)),
                        ("it2", inden & itb),
                        ("it2_prompt", inden & itb & (P["ad0"][pos] <= D0_SPLIT_CM))):
            out[f"sig_d0_um_{mname}_{nm}"] = 1e4 * _rs(S[sel, col["d_d0"]])
            out[f"sig_z0_um_{mname}_{nm}"] = 1e4 * _rs(S[sel, col["d_z0"]])
        out[f"n_res_{mname}"] = int(inden.sum())
    # ---- cost per event, std and recovery entries -------------------------
    for mname, idx in (("std", std), ("soft", soft)):
        for k in ("pairs_phi", "pairs", "seed_objects", "fit", "owned",
                  "s3_rejected", "s4w_rejected", "s6_rejected"):
            out[f"{k}_ev_{mname}"] = sum(C["counters"].get(i, {}).get(k, 0.0)
                                         for i in idx) / nev
    return out


def mva_auc(C, col, feats):
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    from sklearn.metrics import roc_auc_score
    T = np.concatenate([C["tracks"][i] for i in sorted(C["tracks"])])
    X = np.nan_to_num(T[:, [col[f] for f in feats]], posinf=0, neginf=0)
    y = T[:, col["n_wrong"]] <= 1
    ev = T[:, col["event"]]
    te = np.isin(ev, np.unique(ev)[::3])
    g = GB(max_iter=200, learning_rate=0.1, random_state=1).fit(X[~te], y[~te])
    return float(roc_auc_score(y[te], g.predict_proba(X[te])[:, 1]))


def census_seconds_per_event(log):
    """Seconds per event of the census's chunk loop, from its log."""
    try:
        t = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(
            r"chunk\s+\d+\s+(\d+) ev\s+[\d,]+ TPs\s+([\d.]+)s", open(log).read())]
    except OSError:
        return None
    return t[-1][1] / sum(n for n, _ in t) if t else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("census", nargs="+", help="LABEL=PATH (a census or the cache "
                                              "dir holding exactly one)")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--logs", default=None,
                    help="directory of LABEL.log census logs, for s/event")
    ap.add_argument("--mva", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    res = {}
    for spec in a.census:
        lab, path = spec.split("=", 1)
        C = TF.load(TF.census_dir(path), nev=a.nev, with_tracks=True, verbose=False)
        col = {c: j for j, c in enumerate(C["track_cols"])}
        P = tp_rows(C)
        nev = C["n_events"]
        r = {"angles": C["angle_use"].key(), "angle_res": C["angle_res"],
             "label": C["angle_use"].label(), "n_events": nev, "builds": {}}
        if C["angle_res_caveat"]:
            # a k < 1 census: its numbers travel with the caveat
            r["angle_res_caveat"] = C["angle_res_caveat"]
        if a.logs:
            r["s_per_event"] = census_seconds_per_event(Path(a.logs) / f"{lab}.log")
        for b in C["builds"]:
            r["builds"][b] = build_metrics(C, b, P, col, nev)
        if a.mva:
            fall = [f for f in KF.MVA_FEATURES if f in col]
            fno = [f for f in KF.mva_features(M.AngleUse(set(M.ANGLE_STAGES) - {"mva"}))
                   if f in col]
            r["mva_auc_all_feats"] = mva_auc(C, col, fall)
            r["mva_auc_no_angle_feats"] = mva_auc(C, col, fno)
        res[lab] = r
        print(f"{lab}: {r['label']} k={C['angle_res']}"
              + (f"  CAVEAT: {C['angle_res_caveat']['text']}" if C["angle_res_caveat"] else ""),
              flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
