#!/usr/bin/env python
"""Does visiting IT layers outside-in beat inside-out? PAIRED A/B.

Why the question is not cosmetic. The particle traverses L1->L4 outward, but the
seed is an OT-ONLY fit, so the Kalman state enters the IT from OUTSIDE. The
projection to layer L therefore carries the multiple scattering accumulated
between L and the innermost OT layer -- material the OT fit never saw and which
`helixCovMat` does not describe -- and that budget grows INWARD. The first
update is also the most influential, because it is where the covariance
collapses. So:

  insideOut (L1 first)  -> first update on the MOST-extrapolated projection,
                           but the tightest r-phi window (least ambiguous match)
  outsideIn (L4 first)  -> first update on the LEAST-extrapolated projection,
                           but the widest window (most ambiguous match)

Both arguments are real, which is why this is measured rather than argued.

MEASURED 2026-09-03, PU200 D121, 100 events, trackCov: outsideIn wins on every
metric. Wrong-hit fraction per layer 0.225/0.136/0.077/0.052 vs
0.482/0.205/0.131/0.094; clean-refit share 75.3% vs 53.9%; and paired d0 error
34.1 vs 77.1 um. It also accepts MORE hits (46189 vs 44630), so it is not
trading efficiency for purity.

CAVEAT ON THE FLOOR TEST BELOW -- it does not work, and the reason is worth
keeping. The idea was: a single 5-parameter helix cannot represent a scattered
trajectory, that approximation should be least strained when the first update
sits closest to the OT, so a pT-independent pull floor shrinking under outsideIn
would identify the floor as the approximation rather than missing process noise.
But the measurement shows the floor simply tracks VISIT ORDER (L1: 12.8 um when
visited last vs 127.4 first; L4: 72.7 last vs 12.4 first), exactly as the pulls
do. That is because `pred` in `res^2 - pred^2` comes from the covariance, and the
covariance COLLAPSES at the first update and never grows again -- so every layer
after the first has an optimistic `pred` and the "floor" is dominated by that,
not by any helix-approximation term. The floor numbers are still printed because
they are a clean diagnostic of the missing-Q problem, but they CANNOT separate it
from the single-helix limitation. That separation needs Q implemented first.

The comparison is PAIRED: both arms run the same events with the same seed
tracks and differ only in the knob, so every track appears in both arms at the
same index. Pairing is verified, not assumed (seed parameters must match
bit-for-bit); if it fails the script refuses to report paired numbers.

    pixi run python eval_refitq/ordering/layer_order_ab.py \
        -a <outsideIn nano.root> -b <insideOut nano.root>
"""
from __future__ import annotations

import argparse
import glob as _glob
import json

import awkward as ak
import numpy as np
import uproot
from ngtagger.truth_helix import tp_phi0

SENTINEL = -900.0
REF = "L1TTrack"
PT_BINS = [(2, 3), (3, 4), (4, 6), (6, 10), (10, 1e9)]
PARAMS = [("d0", "d0", "tp_d0", 1e4, "um"),
          ("z0", "z0", "tp_z0", 1e4, "um"),
          ("phi", "phi", "tp_phi0", 1e3, "mrad"),   # derived in load(): tp_phi is at production
          ("tanL", "tanL", "tp_tanL", 1e3, "1e-3"),
          ("pt", "pt", "tp_pt", 1.0, "GeV")]


def robust_sigma(v):
    if len(v) < 30:
        return float("nan")
    q1, q3 = np.quantile(v, [0.25, 0.75])
    return float((q3 - q1) / 1.349)


def _discover(files):
    with uproot.open(f"{files[0]}:Events") as t:
        keys = list(t.keys())
    import re
    cfgs = sorted({m.group(1) for m in
                   (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k) for k in keys)
                   if m})
    if not cfgs:
        raise SystemExit("no L1TSmartPixelsRefitHitDigiRefit* table in the input")
    cfg = cfgs[-1]
    kset = set(keys)
    # Pre-v2.6 nano called the OT-projection residual resX/resY, which was
    # ambiguous (a residual is generally truth-vs-measured). v2.6 renamed it
    # projResX/projResY. There is no back-compat path here on purpose: the
    # ordering question is about v2.6 behaviour and reading a v2.5 file would
    # silently compare against a build that had no layerOrder knob.
    need = [f"L1TSmartPixelsRefitHitDigiRefit{cfg}_{c}" for c in ("projResX", "projResY")]
    if any(n not in kset for n in need):
        legacy = f"L1TSmartPixelsRefitHitDigiRefit{cfg}_resX" in kset
        raise SystemExit(
            f"{files[0]}: missing {[n.split('_')[-1] for n in need if n not in kset]}"
            + (" - this looks like PRE-v2.6 nano (it still carries resX/resY). "
               "Re-produce with a v2.6 build; there is no back-compat path."
               if legacy else " - not a v2.6 refit nano."))
    if f"{REF}_genuine" not in kset:
        raise SystemExit(
            f"{files[0]}: no {REF}_genuine / tp_* truth columns, so the paired "
            "parameter-resolution arm cannot run. Produce with a config that keeps "
            "the TrackingParticle truth extension (e.g. the stepCLAMP base), not one "
            "whose pruneAbsentSimpleTables list drops it.")
    return cfg, keys


def load(paths, label):
    files = [f for p in paths for f in (sorted(_glob.glob(p)) or [p])]
    cfg, keys = _discover(files)
    HIT = f"L1TSmartPixelsRefitHitDigiRefit{cfg}"
    VAR = f"L1TSmartPixelsTrackDigiRefit{cfg}"

    ref_cols = ["pt", "genuine", "tpFromHardInteraction", "tp_phi", "tp_charge", "tp_vx", "tp_vy"] + \
               [c for _, c, _, _, _ in PARAMS] + [t for _, _, t, _, _ in PARAMS if t != "tp_phi0"]
    ref_cols = sorted(set(ref_cols))
    var_cols = [c for _, c, _, _, _ in PARAMS] + ["spixRefitPerformed", "spixNAcceptedHits"]
    hit_cols = ["trackIdx", "layer", "hitAccepted", "selHitClass", "windowMult",
                "projResX", "projResY", "pullX", "pullY"]

    ref = uproot.concatenate([f"{f}:Events" for f in files],
                             filter_name=[f"{REF}_{c}" for c in ref_cols])
    var = uproot.concatenate([f"{f}:Events" for f in files],
                             filter_name=[f"{VAR}_{c}" for c in var_cols])
    hits = uproot.concatenate([f"{f}:Events" for f in files],
                              filter_name=[f"{HIT}_{c}" for c in hit_cols])
    ids = uproot.concatenate([f"{f}:Events" for f in files],
                             filter_name=["run", "luminosityBlock", "event"])

    # Multi-stream cmsRun writes events in completion order, so two runs of the
    # same job emit the same events in DIFFERENT positions (measured: 52 of 100).
    # Sort both arms by (run, lumi, event) or the pairing is a permutation and
    # every paired statistic below is meaningless.
    order = np.lexsort((ak.to_numpy(ids["event"]),
                        ak.to_numpy(ids["luminosityBlock"]),
                        ak.to_numpy(ids["run"])))
    ref, var, hits = ref[order], var[order], hits[order]

    counts = ak.to_numpy(ak.num(ref[f"{REF}_pt"]))
    off = np.concatenate([[0], np.cumsum(counts)])
    R = {c: ak.to_numpy(ak.flatten(ref[f"{REF}_{c}"])) for c in ref_cols}
    R["tp_phi0"] = tp_phi0(R)
    V = {c: ak.to_numpy(ak.flatten(var[f"{VAR}_{c}"])) for c in var_cols}
    H = {c: ak.to_numpy(ak.flatten(hits[f"{HIT}_{c}"])) for c in hit_cols}

    ev = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits[f"{HIT}_trackIdx"])))
    H["g"] = off[ev] + H["trackIdx"].astype(np.int64)
    n_tracks = int(off[-1])

    acc = H["hitAccepted"] > 0
    n_wrong = np.zeros(n_tracks)
    np.add.at(n_wrong, H["g"][acc & ((H["selHitClass"] == 1) | (H["selHitClass"] == 2))], 1.0)
    n_acc = np.zeros(n_tracks)
    np.add.at(n_acc, H["g"][acc], 1.0)

    print(f"[{label}] files={len(files)} config={cfg} events={len(counts)} "
          f"tracks={n_tracks} crossings={len(H['layer'])} accepted={int(acc.sum())}")
    return {"label": label, "files": files, "cfg": cfg, "n_tracks": n_tracks,
            "R": R, "V": V, "H": H, "n_wrong": n_wrong, "n_acc": n_acc,
            "pt_track": R["pt"]}


def check_paired(A, B):
    """Both arms must be the same tracks in the same order, or nothing below is paired."""
    if A["n_tracks"] != B["n_tracks"]:
        return False, f"track counts differ ({A['n_tracks']} vs {B['n_tracks']})"
    for c in ("pt", "d0", "z0"):
        if not np.array_equal(A["R"][c], B["R"][c]):
            return False, f"seed {c} differs between arms - not the same seed tracks"
    if A["cfg"] != B["cfg"]:
        return False, f"different activeSP configs ({A['cfg']} vs {B['cfg']})"
    return True, "seed tracks identical (pt, d0, z0 bit-for-bit)"


def boot_median_diff(x, y, n=2000, seed=0):
    """Paired bootstrap CI on median(|x|) - median(|y|)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    d = np.median(x[idx], axis=1) - np.median(y[idx], axis=1)
    return float(np.median(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def per_layer(arm):
    H, out = arm["H"], {}
    acc = H["hitAccepted"] > 0
    for L in (1, 2, 3, 4):
        m = acc & (H["layer"] == L)
        if m.sum() < 30:
            continue
        rec = {"n_accepted": int(m.sum()),
               "windowMult_mean": float(H["windowMult"][m].astype(float).mean()),
               "wrong_frac": float((H["selHitClass"][m] == 1).mean())}
        for dim, pull in (("x", "pullX"), ("y", "pullY")):
            ok = m & (H["selHitClass"] == 0) & (H[pull] > SENTINEL) & (np.abs(H[pull]) > 1e-9)
            rec[f"pull_{dim}_correct"] = robust_sigma(H[pull][ok])
        out[L] = rec
    return out


def pull_floor(arm, dim="x"):
    """pT-independent floor and the 1/pT-like term, per layer (see ms_term_shape.py).

    A nonzero floor is the part scattering CANNOT explain; it is the candidate
    signature of the single-helix approximation.
    """
    H = arm["H"]
    res, pull = ("projResX", "pullX") if dim == "x" else ("projResY", "pullY")
    pt = arm["pt_track"][H["g"]]
    acc = H["hitAccepted"] > 0
    out = {}
    for L in (1, 2, 3, 4):
        base = (acc & (H["selHitClass"] == 0) & (H["layer"] == L)
                & (H[pull] > SENTINEL) & (np.abs(H[pull]) > 1e-9))
        rows = []
        for lo, hi in PT_BINS:
            m = base & (pt >= lo) & (pt < hi)
            if m.sum() < 30:
                continue
            r = H[res][m] * 1e4
            s_pred = np.abs(r) / np.abs(H[pull][m])
            rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "res": robust_sigma(r), "pred": float(np.median(s_pred)),
                         "pull": robust_sigma(H[pull][m])})
        if not rows:
            continue
        hi_bins = [x for x in rows if x["lo"] >= 6]
        c2 = np.median([x["res"] ** 2 - x["pred"] ** 2 for x in hi_bins]) if hi_bins else float("nan")
        floor = float(np.sqrt(max(c2, 0.0))) if np.isfinite(c2) else float("nan")
        lo_rows = [x for x in rows if x["lo"] < 4]
        k = float("nan")
        if lo_rows and np.isfinite(c2):
            x0 = lo_rows[0]
            rem = x0["res"] ** 2 - x0["pred"] ** 2 - c2
            k = float(np.sqrt(max(rem, 0.0)) * 0.5 * (x0["lo"] + x0["hi"]))
        out[L] = {"floor_um": floor, "k_um_GeV": k, "bins": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--arm-a", nargs="+", required=True, help="outsideIn nano")
    ap.add_argument("-b", "--arm-b", nargs="+", required=True, help="insideOut nano")
    ap.add_argument("--name-a", default="outsideIn")
    ap.add_argument("--name-b", default="insideOut")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    A = load(args.arm_a, args.name_a)
    B = load(args.arm_b, args.name_b)
    ok, why = check_paired(A, B)
    print(f"\npairing: {'OK' if ok else 'FAILED'} - {why}")

    out = {"arm_a": {"label": A["label"], "files": A["files"]},
           "arm_b": {"label": B["label"], "files": B["files"]},
           "paired": ok, "pairing_note": why}

    # ---- (1) hit selection and pulls per layer -----------------------------
    la, lb = per_layer(A), per_layer(B)
    out["per_layer"] = {"a": la, "b": lb}
    print("\n(1) per-layer hit selection and correct-hit pulls")
    print(f"  {'L':>2}  {'metric':<22} {args.name_a:>12} {args.name_b:>12}   delta")
    for L in sorted(set(la) & set(lb)):
        for key, fmt in (("n_accepted", "{:.0f}"), ("windowMult_mean", "{:.2f}"),
                         ("wrong_frac", "{:.3f}"), ("pull_x_correct", "{:.3f}"),
                         ("pull_y_correct", "{:.3f}")):
            va, vb = la[L][key], lb[L][key]
            d = va - vb
            print(f"  {L:>2}  {key:<22} {fmt.format(va):>12} {fmt.format(vb):>12}   {d:+.4g}")
        print()

    # ---- (2) the pT-independent floor: approximation or missing Q? ---------
    fa, fb = pull_floor(A, "x"), pull_floor(B, "x")
    out["pull_floor_x"] = {"a": fa, "b": fb}
    print("(2) local-x correct-hit pull decomposition (floor is what scattering CANNOT explain)")
    print(f"  {'L':>2}  {'floor [um]':>22} {'k [um*GeV]':>26}")
    print(f"  {'':>2}  {args.name_a:>10} {args.name_b:>10}      {args.name_a:>10} {args.name_b:>10}")
    for L in sorted(set(fa) & set(fb)):
        print(f"  {L:>2}  {fa[L]['floor_um']:>10.1f} {fb[L]['floor_um']:>10.1f}      "
              f"{fa[L]['k_um_GeV']:>10.0f} {fb[L]['k_um_GeV']:>10.0f}")
    print("  NOTE: this does NOT separate missing-Q from the single-helix limitation.")
    print("  The floor tracks VISIT ORDER because pred comes from a covariance that")
    print("  collapses at the first update and never grows. See the module docstring.")

    # ---- (3) clean-refit share (what a trust gate has to work with) --------
    R, V = A["R"], A["V"]
    m = (V["spixRefitPerformed"] > 0) & (B["V"]["spixRefitPerformed"] > 0) \
        & (R["genuine"] > 0) & (R["tp_pt"] > 0)
    print(f"\n(3) refit tracks usable in BOTH arms: {int(m.sum())} of {A['n_tracks']}")
    print(f"  {'n_wrong':>8} {args.name_a:>12} {args.name_b:>12}")
    nwa, nwb = A["n_wrong"][m], B["n_wrong"][m]
    rec = {}
    for nm, sa, sb in (("0", nwa == 0, nwb == 0), ("1", nwa == 1, nwb == 1),
                       ("2", nwa == 2, nwb == 2), ("3+", nwa >= 3, nwb >= 3)):
        pa, pb = 100 * sa.mean(), 100 * sb.mean()
        print(f"  {nm:>8} {pa:>11.1f}% {pb:>11.1f}%")
        rec[nm] = {"a_frac": float(sa.mean()), "b_frac": float(sb.mean())}
    print(f"  {'<n_acc>':>8} {A['n_acc'][m].mean():>12.2f} {B['n_acc'][m].mean():>12.2f}")
    out["n_wrong"] = rec
    out["n_accepted_mean"] = {"a": float(A["n_acc"][m].mean()), "b": float(B["n_acc"][m].mean())}

    # ---- (4) paired parameter resolution ----------------------------------
    if ok:
        print("\n(4) PAIRED refit-parameter error, same tracks both arms")
        print("    (negative delta = outsideIn is better; CI excluding 0 = significant)")
        out["params"] = {}
        for pname, col, tcol, scale, unit in PARAMS:
            truth = R[tcol][m].astype(np.float64)
            ea = np.abs(A["V"][col][m].astype(np.float64) - truth) * scale
            eb = np.abs(B["V"][col][m].astype(np.float64) - truth) * scale
            es = np.abs(R[col][m].astype(np.float64) - truth) * scale
            d, lo, hi = boot_median_diff(ea, eb)
            win = float((ea < eb).mean())
            print(f"  {pname:<5} [{unit:>5}]  seed {np.median(es):>9.4g}   "
                  f"{args.name_a} {np.median(ea):>9.4g}   {args.name_b} {np.median(eb):>9.4g}   "
                  f"delta {d:+.4g} [{lo:+.4g}, {hi:+.4g}]   a-wins {win*100:.1f}%")
            out["params"][pname] = {
                "unit": unit, "seed_err": float(np.median(es)),
                "a_err": float(np.median(ea)), "b_err": float(np.median(eb)),
                "delta_median": d, "ci95": [lo, hi], "a_win_frac": win}

        # and restricted to tracks CLEAN IN BOTH arms, where the refit is
        # actually meant to be used (the oracle-gated population).
        both_clean = (nwa == 0) & (nwb == 0)
        print(f"\n    restricted to tracks CLEAN IN BOTH arms (n={int(both_clean.sum())}) -- "
              "the population a trust gate would keep")
        out["params_both_clean"] = {}
        for pname, col, tcol, scale, unit in PARAMS:
            if both_clean.sum() < 30:
                break
            truth = R[tcol][m][both_clean].astype(np.float64)
            ea = np.abs(A["V"][col][m][both_clean].astype(np.float64) - truth) * scale
            eb = np.abs(B["V"][col][m][both_clean].astype(np.float64) - truth) * scale
            d, lo, hi = boot_median_diff(ea, eb)
            print(f"  {pname:<5} [{unit:>5}]  {args.name_a} {np.median(ea):>9.4g}   "
                  f"{args.name_b} {np.median(eb):>9.4g}   delta {d:+.4g} [{lo:+.4g}, {hi:+.4g}]")
            out["params_both_clean"][pname] = {
                "unit": unit, "a_err": float(np.median(ea)), "b_err": float(np.median(eb)),
                "delta_median": d, "ci95": [lo, hi]}

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
