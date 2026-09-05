#!/usr/bin/env python
"""How many bits does a position-only refit chi2 DELTA need as a BDT input?

Context. The v2.6 sidecar splits the refit chi2 increments per measurement
dimension. The refit-quality BDT is being redesigned to take a
hardware-plausible input set: layer mask, occupancy bits, the ORIGINAL track
chi2 variables, and quantized per-dimension chi2 DELTAS (the increments the
refit added on top of the seed's OT-only chi2). The compact word's provisional
layout spends 4 bits per chi2 total with q(c) = clamp(round(2*log2(1+c)),0,15);
whether 4 is the right number was never measured. This measures it.

Why v2.5 nano can answer this. In the producer's scalar KF update
(L1SmartPixelsTrackProducer.cc) pull[k] = r/sqrt(S) and chi2inc[k] = r*r/S from
the SAME r, S, behind the SAME chi2UpdateGate, so

    chi2Inc<D>Tot  ==  sum over accepted crossings of pull<D>^2      (exact)

for every dimension D in {X, Y, Alpha, Beta}. v2.5 nano stores the four pulls
per crossing at FULL float precision (no mantissa truncation), so the v2.6
split totals are exactly recoverable from v2.5 files. The script validates this
identity against the v2.5 joint columns before using it.

Sample. The only POST-guard v2.5 sidecar productions are clamp_on_f{1,2}
(200 events, PU200 TT, config AAAA). Post-guard is the right sample: the
pre-guard chi2 tail (increments to 3.3e9) is a numerical pathology the deployed
configuration removes at source, and quantizer range must not be sized for it.
The pre-guard files can be pointed at with --files for contrast.

Statistics caveat, stated up front. ~97% of refit tracks are label-positive, so
the sample holds ~1e3 negatives. Independent AUCs are therefore only good to
~0.005, which is coarser than the effects of interest. Every comparison here is
consequently PAIRED (identical tracks, identical folds, features differing only
in quantization) and CI'd by paired bootstrap, which cancels most of the shared
noise. Scores are 5-fold OUT-OF-FOLD so all ~1e3 negatives enter the AUC while
every prediction stays out-of-sample. Absolute AUCs are reported for orientation
only; conclusions are drawn from paired deltas and from the label-free
occupancy/fidelity diagnostics.

Usage:
    pixi run python eval_refitq/quantstudy/chi2_bitwidth_study.py
    pixi run python eval_refitq/quantstudy/chi2_bitwidth_study.py --quick
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import awkward as ak
import numpy as np
import uproot

NANO_DIR = pathlib.Path("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano")
DEFAULT_FILES = [str(NANO_DIR / f"clamp_on_f{i}.root") for i in (1, 2)]

REF = "L1TTrack"
VAR = "L1TSmartPixelsTrackDigiRefitAAAA"
HIT = "L1TSmartPixelsRefitHitDigiRefitAAAA"

SENTINEL = -900.0
DIMS = ("X", "Y", "Alpha", "Beta")
_PULL_OF = {"X": "pullX", "Y": "pullY", "Alpha": "pullAlpha", "Beta": "pullBeta"}

REF_COLS = ["hwChi2RPhi", "hwChi2RZ", "hwBendChi2", "hwTanl", "hwZ0",
            "nStubs", "trkMVA1", "genuine"]
VAR_COLS = ["spixRefitPerformed", "spixLayerHitMask", "spixMaxWindowMult",
            "spixNCrossings", "spixNAcceptedHits", "spixAnyWindowTruncated",
            "spixChi2IncRPhiTot", "spixChi2IncRZTot"]
HIT_COLS = ["trackIdx", "pullX", "pullY", "pullAlpha", "pullBeta",
            "selHitClass", "hitAccepted"]


# ---------------------------------------------------------------- quantizers
def q_log(c: np.ndarray, nbits: int, per_octave: float) -> np.ndarray:
    """The compact-word family: clamp(round(k*log2(1+c)), 0, 2^n-1).

    per_octave = k = codes per factor-of-2 in (1+c). The deployed 4-bit
    compact word uses k=2.
    """
    hi = (1 << nbits) - 1
    code = np.rint(per_octave * np.log2(1.0 + np.maximum(c, 0.0)))
    return np.clip(code, 0, hi).astype(np.int32)


def q_quantile(c: np.ndarray, nbits: int, ref: np.ndarray | None = None) -> np.ndarray:
    """Equal-frequency binning: the best any n-bit scalar code can do.

    Included as an UPPER BOUND so that a shortfall at n bits can be attributed
    either to the bit budget (quantile bound also loses) or to the log
    quantizer's placement of its codes (only the log family loses).
    """
    src = c if ref is None else ref
    nb = 1 << nbits
    edges = np.quantile(src[src > 0], np.linspace(0, 1, nb, endpoint=False)[1:])
    edges = np.unique(edges)
    return np.searchsorted(edges, c, side="right").astype(np.int32)


# ------------------------------------------------------------------- loading
def _load_table(files, prefix, cols):
    br = [f"{prefix}_{c}" for c in cols]
    arrs = uproot.concatenate([f"{f}:Events" for f in files], filter_name=br)
    return {c: arrs[f"{prefix}_{c}"] for c in cols}


def load(files):
    """Per-track flat arrays + the exactly-reconstructed per-dimension totals."""
    ref = _load_table(files, REF, REF_COLS)
    var = _load_table(files, VAR, VAR_COLS)
    hits = _load_table(files, HIT, HIT_COLS)

    n_ref = ak.num(ref["genuine"])
    n_var = ak.num(var["spixRefitPerformed"])
    if not ak.all(n_ref == n_var):
        sys.exit("reference and variant track tables are not 1:1 per event")

    # global track offsets so per-event hit trackIdx can index a flat array
    counts = ak.to_numpy(n_ref)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_tracks = int(offsets[-1])

    flat = {f"ref_{c}": ak.to_numpy(ak.flatten(v)) for c, v in ref.items()}
    flat.update({f"var_{c}": ak.to_numpy(ak.flatten(v)) for c, v in var.items()})

    # per-hit -> per-track: sum pull^2 over applied updates, per dimension
    hit_track = ak.to_numpy(ak.flatten(hits["trackIdx"])).astype(np.int64)
    ev_of_hit = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits["trackIdx"])))
    gidx = offsets[ev_of_hit] + hit_track

    sums = {}
    for d in DIMS:
        p = ak.to_numpy(ak.flatten(hits[_PULL_OF[d]])).astype(np.float64)
        ok = p > SENTINEL
        acc = np.zeros(n_tracks, np.float64)
        np.add.at(acc, gidx[ok], p[ok] ** 2)
        sums[d] = acc

    # TRUTH-ONLY per-track wrong-hit counts, for the 'clean' label. selHitClass:
    # 0 sameTP, 1 otherTP, 2 noise, -1 none.
    cls = ak.to_numpy(ak.flatten(hits["selHitClass"]))
    acc_ok = ak.to_numpy(ak.flatten(hits["hitAccepted"])) > 0
    for name, sel in (("n_wrong_hits", acc_ok & (cls == 1)),
                      ("n_noise_hits", acc_ok & (cls == 2)),
                      ("n_acc_hits", acc_ok)):
        c = np.zeros(n_tracks, np.float64)
        np.add.at(c, gidx[sel], 1.0)
        flat[name] = c
    return flat, sums, n_tracks


def validate_identity(flat, sums) -> dict:
    """chi2IncXTot + chi2IncAlphaTot must equal the v2.5 joint spixChi2IncRPhiTot."""
    out = {}
    for joint, (a, b) in (("RPhi", ("X", "Alpha")), ("RZ", ("Y", "Beta"))):
        stored = flat[f"var_spixChi2Inc{joint}Tot"].astype(np.float64)
        recon = sums[a] + sums[b]
        m = (flat["var_spixRefitPerformed"] > 0) & (stored > SENTINEL) & (stored > 0)
        rel = np.abs(recon[m] - stored[m]) / np.maximum(stored[m], 1e-12)
        out[joint] = {
            "n": int(m.sum()),
            "median_rel_dev": float(np.median(rel)),
            "p99_rel_dev": float(np.quantile(rel, 0.99)),
            "max_rel_dev": float(rel.max()),
        }
    return out


# ------------------------------------------------------------------- metrics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks (vectorized).

    Quantized features produce MANY ties, so tie-averaging is required, not
    cosmetic - integer codes would otherwise be scored by argsort order.
    """
    from scipy.stats import rankdata
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def fit_predict(Xtr, ytr, Xte, seed=0, quick=False):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=60 if quick else 200,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def oof_scores(X, y, n_folds=5, seed=0, quick=False):
    """Out-of-fold predictions so EVERY track contributes to the AUC.

    With ~1e3 negatives in the whole sample, a single 35% test split leaves
    ~350 - too few. K-fold OOF scoring uses all of them while keeping every
    prediction out-of-sample, and keeps the comparison paired across feature
    sets (identical folds, identical tracks).
    """
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % n_folds
    s = np.empty(len(y), float)
    for f in range(n_folds):
        te = fold == f
        s[te] = fit_predict(X[~te], y[~te], X[te], seed=seed, quick=quick)
    return s


def paired_bootstrap(y, s_a, s_b, n_boot=400, seed=0):
    """CI on AUC(a) - AUC(b) resampling TRACKS (paired: same resample both)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or (yy == 0).sum() == 0:
            d[i] = np.nan
            continue
        d[i] = auc(yy, s_a[idx]) - auc(yy, s_b[idx])
    d = d[~np.isnan(d)]
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# --------------------------------------------------------------------- study
def occ_bits(max_window_mult: np.ndarray) -> np.ndarray:
    """The compact word's 3-bit occupancy code."""
    return np.clip(np.floor(np.log2(1.0 + np.maximum(max_window_mult, 0))), 0, 7)


def build_base(flat) -> tuple[np.ndarray, list[str]]:
    """Everything the redesigned BDT gets EXCEPT the refit chi2 deltas.

    Includes the ORIGINAL track chi2 variables (the seed's own binned chi2),
    which the delta must be interpreted against: a delta of 5 means something
    different on a seed with chi2 bin 1 than on one with bin 12.
    """
    cols = {
        "orig_hwChi2RPhi": flat["ref_hwChi2RPhi"],
        "orig_hwChi2RZ": flat["ref_hwChi2RZ"],
        "orig_hwBendChi2": flat["ref_hwBendChi2"],
        "orig_nStubs": flat["ref_nStubs"],
        "orig_hwTanl": flat["ref_hwTanl"],
        "orig_hwZ0": flat["ref_hwZ0"],
        "spixLayerHitMask": flat["var_spixLayerHitMask"],
        "spixNAcceptedHits": flat["var_spixNAcceptedHits"],
        "spixAnyWindowTruncated": flat["var_spixAnyWindowTruncated"],
        "occ3": occ_bits(flat["var_spixMaxWindowMult"]),
    }
    names = list(cols)
    return np.column_stack([np.asarray(cols[n], float) for n in names]), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--outdir", default="eval_refitq/quantstudy")
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--per-octave", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--label", choices=["genuine", "clean"], default="genuine",
                    help="'genuine' = the deployed objective (track truth-match of "
                         "the OT seed; only ~1e3 negatives). 'clean' = every "
                         "accepted IT hit came from the track's own TP - the "
                         "failure mode the refit itself controls, and ~17x more "
                         "negatives, so bit-width effects actually resolve.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading {len(args.files)} file(s)")
    flat, sums, n_tracks = load(args.files)

    ident = validate_identity(flat, sums)
    print("\n== identity check: sum(pull^2) per dimension vs stored joint totals ==")
    for k, v in ident.items():
        print(f"  {k}: n={v['n']}  median rel dev={v['median_rel_dev']:.3e}  "
              f"p99={v['p99_rel_dev']:.3e}  max={v['max_rel_dev']:.3e}")

    # refit tracks only (passthrough tracks are never scored)
    m = flat["var_spixRefitPerformed"] > 0
    if args.label == "genuine":
        y = (flat["ref_genuine"][m] > 0).astype(np.int32)
    else:
        y = ((flat["n_wrong_hits"][m] + flat["n_noise_hits"][m] == 0)
             & (flat["n_acc_hits"][m] > 0)).astype(np.int32)
    d = {k: sums[k][m] for k in DIMS}
    fl = {k: v[m] for k, v in flat.items()}

    print(f"\n== sample ==\n  refit tracks={m.sum()}  positives={y.sum()}  "
          f"negatives={(y == 0).sum()}")

    print("\n== per-dimension delta distributions (post-guard) ==")
    dist = {}
    for k in DIMS:
        v = d[k]
        qs = np.quantile(v, [0.5, 0.9, 0.99, 0.999, 1.0])
        dist[k] = {"frac_zero": float((v == 0).mean()), "median": float(qs[0]),
                   "p90": float(qs[1]), "p99": float(qs[2]),
                   "p999": float(qs[3]), "max": float(qs[4]),
                   "mean": float(v.mean())}
        print(f"  {k:>5}: zero={dist[k]['frac_zero']*100:5.1f}%  med={qs[0]:9.3g}  "
              f"p99={qs[2]:9.3g}  max={qs[4]:9.3g}")

    base, base_names = build_base(fl)

    def run(extra: dict, seed=0):
        if extra:
            X = np.column_stack([base] + [np.asarray(v, float) for v in extra.values()])
        else:
            X = base
        s = oof_scores(X, y, n_folds=args.n_folds, seed=args.seed + seed,
                       quick=args.quick)
        return s, auc(y, s)

    results = {"files": args.files, "label": args.label,
               "identity": ident, "distributions": dist,
               "n_refit_tracks": int(m.sum()), "n_pos": int(y.sum()),
               "n_neg": int((y == 0).sum()), "base_features": base_names,
               "runs": {}, "paired": {}}

    # ---- reference points -------------------------------------------------
    s_base, a_base = run({})
    s_float_pos, a_float_pos = run({"dX": d["X"], "dY": d["Y"]})
    s_float_all, a_float_all = run({"dX": d["X"], "dY": d["Y"],
                                    "dA": d["Alpha"], "dB": d["Beta"]})
    results["runs"]["base_no_delta"] = a_base
    results["runs"]["float_position"] = a_float_pos
    results["runs"]["float_position_and_angle"] = a_float_all
    print("\n== reference AUCs ==")
    print(f"  base (orig chi2 + counters, NO refit delta) : {a_base:.5f}")
    print(f"  + float position deltas                     : {a_float_pos:.5f}")
    print(f"  + float position AND angle deltas           : {a_float_all:.5f}")

    mu, lo, hi = paired_bootstrap(y, s_float_pos, s_base, args.n_boot, args.seed)
    results["paired"]["float_position_vs_base"] = {"mean": mu, "lo": lo, "hi": hi}
    print(f"  paired dAUC(float position - base) = {mu:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    mu, lo, hi = paired_bootstrap(y, s_float_all, s_float_pos, args.n_boot, args.seed)
    results["paired"]["angle_on_top_of_position"] = {"mean": mu, "lo": lo, "hi": hi}
    print(f"  paired dAUC(angle on top of position) = {mu:+.5f}  [{lo:+.5f}, {hi:+.5f}]")

    # ---- quantizer sweep: position deltas ---------------------------------
    print("\n== position-delta quantization sweep (vs FLOAT position deltas) ==")
    print("  bits  k/oct  codes_used  sat%   zero%   AUC      paired dAUC vs float [95% CI]")
    sweep = []
    for nb in args.bits:
        for k in args.per_octave:
            qx, qy = q_log(d["X"], nb, k), q_log(d["Y"], nb, k)
            hi_code = (1 << nb) - 1
            used = len(np.unique(np.concatenate([qx, qy])))
            sat = float(((qx == hi_code) | (qy == hi_code)).mean())
            zero = float(((qx == 0) & (qy == 0)).mean())
            s_q, a_q = run({"dX": qx, "dY": qy})
            mu, lo, hi = paired_bootstrap(y, s_q, s_float_pos, args.n_boot, args.seed)
            row = {"bits": nb, "per_octave": k, "codes_used": used,
                   "saturation_frac": sat, "both_zero_frac": zero, "auc": a_q,
                   "dauc_vs_float": mu, "ci_lo": lo, "ci_hi": hi}
            sweep.append(row)
            print(f"  {nb:>4}  {k:>5.1f}  {used:>10}  {sat*100:5.2f} {zero*100:6.2f}  "
                  f"{a_q:.5f}  {mu:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    results["position_sweep"] = sweep

    # ---- quantile upper bound at each bit width ---------------------------
    print("\n== quantile (equal-frequency) upper bound at each bit width ==")
    bound = []
    for nb in args.bits:
        qx = q_quantile(d["X"], nb)
        qy = q_quantile(d["Y"], nb)
        s_q, a_q = run({"dX": qx, "dY": qy})
        mu, lo, hi = paired_bootstrap(y, s_q, s_float_pos, args.n_boot, args.seed)
        bound.append({"bits": nb, "auc": a_q, "dauc_vs_float": mu,
                      "ci_lo": lo, "ci_hi": hi})
        print(f"  {nb} bits: AUC={a_q:.5f}  dAUC vs float={mu:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    results["position_quantile_bound"] = bound

    # ---- angle deltas: how many bits, given position is already there -----
    print("\n== angle-delta bits (on top of 4-bit k=2 position deltas) ==")
    qx4, qy4 = q_log(d["X"], 4, 2.0), q_log(d["Y"], 4, 2.0)
    s_pos4, a_pos4 = run({"dX": qx4, "dY": qy4})
    results["runs"]["q4k2_position_only"] = a_pos4
    ang = []
    for nb in [1, 2, 3, 4]:
        qa, qb = q_log(d["Alpha"], nb, 2.0), q_log(d["Beta"], nb, 2.0)
        s_a, a_a = run({"dX": qx4, "dY": qy4, "dA": qa, "dB": qb})
        mu, lo, hi = paired_bootstrap(y, s_a, s_pos4, args.n_boot, args.seed)
        ang.append({"bits": nb, "auc": a_a, "dauc_vs_position_only": mu,
                    "ci_lo": lo, "ci_hi": hi})
        print(f"  angle {nb} bit(s): AUC={a_a:.5f}  dAUC vs position-only="
              f"{mu:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    results["angle_sweep"] = ang

    out = outdir / f"chi2_bitwidth_results_{args.label}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
