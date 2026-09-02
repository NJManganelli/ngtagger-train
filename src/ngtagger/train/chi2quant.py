"""Calibrate the per-field quantizer constants for the refit chi2 delta inputs.

The refit-quality BDT consumes the four per-dimension chi2 deltas
(chi2IncX/Y/Alpha/BetaTot) as small integer codes produced by

    q(c) = clamp(round(k * log2(1 + c)), 0, 2**bits - 1)

`bits` sets how many codes exist; `k` (codes per octave) sets how those codes
are spread over the value range, since one code step is a factor 2**(1/k) in
(1+c). At fixed `bits` the covered span is (2**bits - 1)/k octaves, so a `k`
sized for a narrower range than the data actually spans dumps the tail into the
top code, where no tree can ever separate it again.

Each field gets its OWN (bits, k): the four ranges differ substantially, and
there is no structural cost to four constants. If two happen to agree, fine.

These are CONTRACT values: the producer's C++ and this package must apply the
identical encoding, and changing one requires retraining plus a lockstep
producer update. Hence this module: run it over a glob of nano files and it
emits the constants, the evidence, and pasteable C++/Python.

Method. The scan metric is the SINGLE-FEATURE AUC of the code compared with
that of the unquantized value: it isolates one field (no other field can mask
its loss), needs no model fit, and so is cheap enough to run routinely. The
chosen configuration is then validated once with a real BDT (paired, 5-fold
out-of-fold, bootstrap CI) - but only to SHORTLIST. The proxy is optimistic:
measured on the reference sample it understated the 4-bit angle loss by ~0.007
AUC, because a tree ensemble extracts more from a fine-grained value in
combination with the other features than a lone ranking variable shows. So the
choice itself is made on real per-field fits over the shortlist, and the full
set is validated jointly at the end. --no-bdt falls back to proxy-only.

Label. Default 'clean' = every accepted IT hit came from the track's own TP,
i.e. the failure mode the refit itself controls. It is the label these features
respond to and it carries ~17x more negatives than 'genuine', so quantization
effects actually resolve. 'genuine' (the deployed fake-rejection objective) is
available but cannot resolve them - see docs/refit-chi2-bitwidth-study.md.
"""
from __future__ import annotations

import glob as _glob
import json
import re
from dataclasses import dataclass, field as _dcfield

import awkward as ak
import numpy as np
import uproot

FIELDS = ("X", "Y", "Alpha", "Beta")
SENTINEL = -900.0
_PULL_OF = {"X": "pullX", "Y": "pullY", "Alpha": "pullAlpha", "Beta": "pullBeta"}
_TOT_OF = {f: f"spxChi2Inc{f}Tot" for f in FIELDS}

DEFAULT_BITS_GRID = (3, 4, 5, 6)
DEFAULT_K_GRID = (0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
# Coarser grid for the (much more expensive, and decisive) per-field BDT stage.
DEFAULT_BDT_BITS = (4, 5, 6)
DEFAULT_BDT_K_GRID = (0.75, 1.0, 1.5, 2.0, 3.0)


# --------------------------------------------------------------- the encoder
def quantize(c, bits: int, k: float) -> np.ndarray:
    """The canonical code. MUST match the producer's C++ bit-for-bit.

    Values <= 0 (including the -999 sentinel) map to code 0, matching the
    producer's `if (!(c > 0.)) return 0;`.
    """
    hi = (1 << bits) - 1
    c = np.asarray(c, dtype=np.float64)
    out = np.zeros(c.shape, dtype=np.int32)
    pos = c > 0.0
    out[pos] = np.clip(np.rint(k * np.log2(1.0 + c[pos])), 0, hi).astype(np.int32)
    return out


def quantile_edges(c: np.ndarray, bits: int) -> np.ndarray:
    """Equal-frequency thresholds: the information-optimal MONOTONE code.

    A BDT only ever compares a feature to thresholds, so any monotone encoding
    is admissible - the log form has no special status beyond being one
    multiply and a clamp. For a fixed number of levels, equal-frequency
    placement carries the most entropy, and in firmware it is a fixed
    comparator tree over 2^bits - 1 constants. If a log code needs more bits
    than a quantile code for the same accuracy, those thresholds are the
    cheaper way to buy it.

    Edges are derived from the calibration sample and become CONTRACT
    constants, so they must be emitted and frozen alongside the model.
    """
    nz = c[c > 0]
    if len(nz) < (1 << bits):
        return np.unique(nz)
    qs = np.linspace(0, 1, (1 << bits), endpoint=False)[1:]
    return np.unique(np.quantile(nz, qs))


def quantize_edges(c, edges: np.ndarray) -> np.ndarray:
    """Apply a threshold LUT. Values <= 0 land in code 0, as for the log form."""
    c = np.asarray(c, dtype=np.float64)
    out = np.searchsorted(edges, c, side="right").astype(np.int32)
    out[c <= 0.0] = 0
    return out


def code_lower_edge(code: int, k: float) -> float:
    """Smallest c mapping to `code` (round() puts edges at half-integers)."""
    return 0.0 if code <= 0 else 2.0 ** ((code - 0.5) / k) - 1.0


def saturation_threshold(bits: int, k: float) -> float:
    return code_lower_edge((1 << bits) - 1, k)


# --------------------------------------------------------------- discovery
@dataclass
class Tables:
    reference: str
    variant: str
    hits: str
    config: str
    has_split: bool
    has_pulls: bool


def discover_tables(a_file: str, config: str | None = None,
                    extended: bool = False) -> Tables:
    """Find the reference / scenario / per-hit table names in a nano file."""
    with uproot.open(f"{a_file}:Events") as t:
        keys = set(t.keys())
    ext = "Ext" if extended else ""
    pat = re.compile(rf"^L1TSmartPixels{ext}TrackDigiRefit([A-Za-z0-9]+)_spx")
    configs = sorted({m.group(1) for m in (pat.match(k) for k in keys) if m})
    if not configs:
        raise RuntimeError(
            f"{a_file}: no L1TSmartPixels{ext}TrackDigiRefit*_spx* tables found - "
            f"is this a digiRefit nano with the sidecar tables enabled?")
    if config is None:
        config = configs[-1]
    elif config not in configs:
        raise RuntimeError(f"{a_file}: config {config!r} not present; have {configs}")
    var = f"L1TSmartPixels{ext}TrackDigiRefit{config}"
    hit = f"L1TSmartPixels{ext}RefitHitDigiRefit{config}"
    ref = f"L1T{ext}Track" if ext else "L1TTrack"
    if f"{ref}_genuine" not in keys:
        raise RuntimeError(f"{a_file}: reference table {ref}_* not found (need truth columns)")
    return Tables(
        reference=ref, variant=var, hits=hit, config=config,
        has_split=all(f"{var}_{_TOT_OF[f]}" in keys for f in FIELDS),
        has_pulls=all(f"{hit}_{_PULL_OF[f]}" in keys for f in FIELDS),
    )


def expand_inputs(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        hits = sorted(_glob.glob(p))
        out.extend(hits if hits else [p])
    if not out:
        raise RuntimeError(f"no input files matched {patterns}")
    return out


# ------------------------------------------------------------------ loading
@dataclass
class Sample:
    deltas: dict
    labels: dict
    n_tracks: int
    tables: Tables
    source: str
    files: list = _dcfield(default_factory=list)


def load_sample(files: list[str], tables: Tables, allow_pull_derived: bool = False,
                max_events: int | None = None) -> Sample:
    """Per-track chi2 deltas plus the two candidate labels."""
    def cat(prefix, cols):
        arrs = uproot.concatenate([f"{f}:Events" for f in files],
                                  filter_name=[f"{prefix}_{c}" for c in cols])
        if max_events is not None:
            arrs = arrs[:max_events]
        return {c: arrs[f"{prefix}_{c}"] for c in cols}

    ref = cat(tables.reference, ["genuine"])
    var = cat(tables.variant, ["spxRefitPerformed", "spxNAcceptedHits"]
              + ([_TOT_OF[f] for f in FIELDS] if tables.has_split else []))
    counts = ak.to_numpy(ak.num(ref["genuine"]))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_tracks = int(offsets[-1])

    hit_cols = ["trackIdx", "selHitClass", "hitAccepted"]
    if not tables.has_split:
        if not (allow_pull_derived and tables.has_pulls):
            raise RuntimeError(
                f"{files[0]}: table {tables.variant!r} has no 4-way split chi2 columns "
                f"(spxChi2IncX/Y/Alpha/BetaTot). This is pre-v2.6 nano. Reproduce it with "
                f"v2.6+ software, or pass --allow-pull-derived to reconstruct the totals "
                f"from the per-hit pulls (exact: chi2Inc<D>Tot == sum(pull<D>^2)).")
        hit_cols += [_PULL_OF[f] for f in FIELDS]
    hits = cat(tables.hits, hit_cols)

    ti = ak.to_numpy(ak.flatten(hits["trackIdx"])).astype(np.int64)
    per_ev = ak.to_numpy(ak.num(hits["trackIdx"]))
    gidx = offsets[np.repeat(np.arange(len(counts)), per_ev)] + ti

    if tables.has_split:
        deltas = {f: ak.to_numpy(ak.flatten(var[_TOT_OF[f]])).astype(np.float64)
                  for f in FIELDS}
        source = "split columns (v2.6+)"
    else:
        deltas = {}
        for f in FIELDS:
            p = ak.to_numpy(ak.flatten(hits[_PULL_OF[f]])).astype(np.float64)
            ok = p > SENTINEL
            acc = np.zeros(n_tracks, np.float64)
            np.add.at(acc, gidx[ok], p[ok] ** 2)
            deltas[f] = acc
        source = "reconstructed from per-hit pulls (pre-v2.6 nano)"

    cls = ak.to_numpy(ak.flatten(hits["selHitClass"]))
    acc_ok = ak.to_numpy(ak.flatten(hits["hitAccepted"])) > 0
    n_bad = np.zeros(n_tracks); n_acc = np.zeros(n_tracks)
    np.add.at(n_bad, gidx[acc_ok & ((cls == 1) | (cls == 2))], 1.0)
    np.add.at(n_acc, gidx[acc_ok], 1.0)

    keep = ak.to_numpy(ak.flatten(var["spxRefitPerformed"])) > 0
    labels = {
        "genuine": (ak.to_numpy(ak.flatten(ref["genuine"]))[keep] > 0).astype(np.int32),
        "clean": ((n_bad[keep] == 0) & (n_acc[keep] > 0)).astype(np.int32),
    }
    return Sample(deltas={f: v[keep] for f, v in deltas.items()}, labels=labels,
                  n_tracks=int(keep.sum()), tables=tables, source=source,
                  files=list(files))


# ------------------------------------------------------------------ metrics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Tie-averaged Mann-Whitney AUC (quantized codes are heavily tied)."""
    from scipy.stats import rankdata
    npos = int(y.sum()); nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def entropy_bits(code: np.ndarray) -> float:
    _, cnt = np.unique(code, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def paired_bootstrap(y, s_a, s_b, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() in (0, n):
            continue
        d.append(auc(yy, s_a[idx]) - auc(yy, s_b[idx]))
    d = np.asarray(d)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# --------------------------------------------------------------- the scan
def scan_field(delta: np.ndarray, y: np.ndarray, bits_grid, k_grid) -> list[dict]:
    """Single-feature AUC loss vs the unquantized value, per (bits, k).

    A larger chi2 delta indicates a worse refit, so the score is -delta.
    """
    a_float = auc(y, -delta)
    rows = []
    for bits in bits_grid:
        for k in k_grid:
            c = quantize(delta, bits, k)
            hi = (1 << bits) - 1
            rows.append({
                "bits": bits, "k": k,
                "auc_code": auc(y, -c.astype(float)),
                "auc_float": a_float,
                "dauc": auc(y, -c.astype(float)) - a_float,
                "entropy_bits": entropy_bits(c),
                "efficiency": entropy_bits(c) / bits,
                "saturation_frac": float((c == hi).mean()),
                "code0_frac": float((c == 0).mean()),
                "codes_used": int(len(np.unique(c))),
                "spearman_vs_float": spearman(c, delta),
                "saturates_at": saturation_threshold(bits, k),
            })
    return rows


def _parity(row: dict, tol: float, sat_max: float, key: str = "dauc") -> bool:
    """Only DEGRADATION counts against a code.

    A coarser code can score marginally above the unquantized value (coarsening
    can break uninformative ordering); that is not a loss to be penalized, so
    the test is one-sided: dAUC >= -tol. A populated top code, by contrast, is a
    hard constraint - it is an information sink no training can undo, and it
    only gets worse if the input distribution shifts.
    """
    return row[key] >= -tol and row["saturation_frac"] <= sat_max


def choose_field(rows: list[dict], tol: float, sat_max: float,
                 key: str = "dauc") -> dict:
    """Fewest bits that reach parity, then the cheapest STATISTICALLY equal code.

    Order of preference:
      1. fewest bits among candidates ACCEPTED by the tolerance
         (dAUC >= -tol and saturation <= sat_max);
      2. among those, prefer codes whose dAUC confidence interval includes 0,
         i.e. statistically indistinguishable from unquantized;
      3. then least saturation, then highest code entropy.

    Note what step 3 means when step 2's subset is empty: every candidate at the
    minimal width is measurably (if acceptably) below float, and the choice then
    goes to robustness rather than to the best point estimate. That is
    deliberate - a sub-tolerance AUC difference is not worth a populated top
    code, which degrades further if the input distribution shifts. If you want
    true statistical parity instead, read `min_config_at_parity` in the report:
    it names the narrowest field width whose CI includes 0, at the cost of bits.
    Rows without a CI (the proxy path) skip step 2.
    """
    ok = [r for r in rows if _parity(r, tol, sat_max, key)]
    if ok:
        min_bits = min(r["bits"] for r in ok)
        at_min = [r for r in ok if r["bits"] == min_bits]
        stat_equal = [r for r in at_min
                      if r.get("ci_lo", -1.0) <= 0.0 <= r.get("ci_hi", 1.0)]
        pool = stat_equal or at_min
        best = min(pool, key=lambda r: (r["saturation_frac"], -r["efficiency"]))
        return {**best, "reached_parity": True}
    best = min(rows, key=lambda r: (-r[key], r["saturation_frac"]))
    return {**best, "reached_parity": False}


def bdt_candidates(delta: np.ndarray, bits_grid, k_grid, sat_max: float) -> list[dict]:
    """Configurations worth a real fit, filtered ONLY on label-free criteria.

    Do NOT rank candidates by the single-feature proxy. The proxy is not merely
    optimistic in magnitude, its ORDERING disagrees with real fits: on the
    reference sample it prefers k=0.5 for X, which a fit shows losing -0.0022,
    over k=1.0-1.25, which a fit shows at parity. Shortlisting by proxy rank
    therefore discards the good candidates. Saturation, by contrast, is a
    property of the encoding and the data alone, so it is a safe filter.
    """
    out = []
    for bits in bits_grid:
        for k in k_grid:
            c = quantize(delta, bits, k)
            hi = (1 << bits) - 1
            sat = float((c == hi).mean())
            if sat > sat_max:
                continue
            out.append({"bits": bits, "k": k, "saturation_frac": sat,
                        "entropy_bits": entropy_bits(c),
                        "efficiency": entropy_bits(c) / bits,
                        "code0_frac": float((c == 0).mean()),
                        "codes_used": int(len(np.unique(c))),
                        "spearman_vs_float": spearman(c, delta),
                        "saturates_at": saturation_threshold(bits, k)})
        # one equal-frequency reference per width: the ceiling for a monotone
        # code of this many levels, so a log-code shortfall is attributable
        edges = quantile_edges(delta, bits)
        c = quantize_edges(delta, edges)
        out.append({"bits": bits, "k": None, "edges": edges.tolist(),
                    "saturation_frac": float((c == (1 << bits) - 1).mean()),
                    "entropy_bits": entropy_bits(c),
                    "efficiency": entropy_bits(c) / bits,
                    "code0_frac": float((c == 0).mean()),
                    "codes_used": int(len(np.unique(c))),
                    "spearman_vs_float": spearman(c, delta),
                    "saturates_at": float(edges[-1]) if len(edges) else float("nan")})
    return out


def emit_constants(chosen: dict) -> str:
    """Pasteable C++ and Python for the producer / trainer contract."""
    cxx = ["  // Refit chi2 delta encoding, per field. Log form:",
           "  //   q(c) = clamp(round(k*log2(1+c)), 0, 2^bits-1)",
           "  // LUT form: code = count of thresholds < c (c <= 0 -> 0).",
           "  // Calibrated by ngtagger-train `calibrate-chi2-quant`. CONTRACT: keep in",
           "  // lockstep with the python mirror; changing any value forces a retrain."]
    for f in FIELDS:
        c = chosen[f]
        cxx.append(f"  inline constexpr int kChi2{f}Bits = {c['bits']};")
        if c.get("k") is None:
            edges = ", ".join(f"{e:.6g}f" for e in c["edges"])
            cxx.append(f"  inline constexpr std::array<float, {len(c['edges'])}> "
                       f"kChi2{f}Edges = {{{edges}}};")
        else:
            cxx.append(f"  inline constexpr double kChi2{f}PerOctave = {c['k']};")
    py = ["CHI2_QUANT = {"]
    for f in FIELDS:
        c = chosen[f]
        if c.get("k") is None:
            py.append(f"    {f!r}: {{'bits': {c['bits']}, 'k': None, "
                      f"'edges': {[round(float(e), 6) for e in c['edges']]}}},")
        else:
            py.append(f"    {f!r}: {{'bits': {c['bits']}, 'k': {c['k']}}},")
    py.append("}")
    return "\n".join(cxx) + "\n\n" + "\n".join(py)


# ----------------------------------------------------------------- BDT check
def _base_features(files, tables, n_expect, max_events=None):
    """Non-chi2 context for the validation fit (counters + seed chi2 bins)."""
    def cat(prefix, cols):
        arrs = uproot.concatenate([f"{f}:Events" for f in files],
                                  filter_name=[f"{prefix}_{c}" for c in cols])
        if max_events is not None:
            arrs = arrs[:max_events]
        return {c: ak.to_numpy(ak.flatten(arrs[f"{prefix}_{c}"])) for c in cols}

    ref = cat(tables.reference, ["hwChi2RPhi", "hwChi2RZ", "hwBendChi2", "nStubs"])
    var = cat(tables.variant, ["spxRefitPerformed", "spxLayerHitMask",
                               "spxNAcceptedHits", "spxMaxWindowMult",
                               "spxAnyWindowTruncated"])
    keep = var["spxRefitPerformed"] > 0
    occ = np.clip(np.floor(np.log2(1.0 + np.maximum(var["spxMaxWindowMult"], 0))), 0, 7)
    cols = [ref["hwChi2RPhi"], ref["hwChi2RZ"], ref["hwBendChi2"], ref["nStubs"],
            var["spxLayerHitMask"], var["spxNAcceptedHits"],
            var["spxAnyWindowTruncated"], occ]
    X = np.column_stack([np.asarray(c, float)[keep] for c in cols])
    assert len(X) == n_expect, f"base/delta row mismatch: {len(X)} vs {n_expect}"
    return X


def encode(delta: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply whichever code `cfg` describes (log k, or a threshold LUT)."""
    if cfg.get("k") is None:
        return quantize_edges(delta, np.asarray(cfg["edges"]))
    return quantize(delta, cfg["bits"], cfg["k"])


def oof_scores(X, y, n_folds=5, seed=0, n_trees=200):
    """5-fold out-of-fold predictions, so every negative enters the AUC."""
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % n_folds
    s = np.empty(len(y))
    for f in range(n_folds):
        te = fold == f
        clf = xgb.XGBClassifier(
            n_estimators=n_trees, max_depth=4, learning_rate=0.15,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=4)
        clf.fit(X[~te], y[~te])
        s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def bdt_isolated_scan(base, delta, y, candidates, n_folds=5, seed=0,
                      n_boot=200, n_trees=60):
    """Re-measure shortlisted codes for ONE field with a real fit.

    base + this field only, so no other chi2 field can mask the loss (in situ,
    with the other three at full precision, every field looks free - the four
    are mutually redundant enough to cover for each other).
    """
    s_float = oof_scores(np.column_stack([base, delta]), y, n_folds, seed, n_trees)
    a_float = auc(y, s_float)
    out = []
    for cand in candidates:
        c = (quantize_edges(delta, np.asarray(cand["edges"])) if cand["k"] is None
             else quantize(delta, cand["bits"], cand["k"])).astype(float)
        s_c = oof_scores(np.column_stack([base, c]), y, n_folds, seed, n_trees)
        mu, lo, hi = paired_bootstrap(y, s_c, s_float, n_boot, seed)
        out.append({**cand, "auc_bdt": auc(y, s_c), "auc_bdt_float": a_float,
                    "dauc_bdt": mu, "ci_lo": lo, "ci_hi": hi})
    return out


def validate_with_bdt(sample: Sample, chosen: dict, y: np.ndarray, base,
                      n_folds=5, seed=0, n_boot=400, quick=False):
    """Final paired comparison: all four chosen codes vs all four unquantized."""
    n_trees = 60 if quick else 200
    X_float = np.column_stack([base] + [sample.deltas[f] for f in FIELDS])
    X_code = np.column_stack([base] + [encode(sample.deltas[f], chosen[f]).astype(float)
                                       for f in FIELDS])
    s_f = oof_scores(X_float, y, n_folds, seed, n_trees)
    s_c = oof_scores(X_code, y, n_folds, seed, n_trees)
    mu, lo, hi = paired_bootstrap(y, s_c, s_f, n_boot, seed)
    return {"auc_float": auc(y, s_f), "auc_coded": auc(y, s_c),
            "dauc": mu, "ci_lo": lo, "ci_hi": hi}


# --------------------------------------------------------------------- main
def run(args) -> dict:
    files = expand_inputs(args.inputs)
    tables = discover_tables(files[0], config=args.config, extended=args.extended)
    print(f"inputs: {len(files)} file(s); config {tables.config}")
    print(f"tables: ref={tables.reference} var={tables.variant} hits={tables.hits}")

    sample = load_sample(files, tables, allow_pull_derived=args.allow_pull_derived,
                         max_events=args.max_events)
    y = sample.labels[args.label]
    print(f"source: {sample.source}")
    print(f"refit tracks: {sample.n_tracks}  label {args.label!r}: "
          f"pos={int(y.sum())} neg={int((y == 0).sum())}")
    if int((y == 0).sum()) < 200:
        print("  WARNING: <200 negatives; quantization effects will not resolve. "
              "Add files, or use --label clean.")

    bits_grid = tuple(args.bits) if args.bits else DEFAULT_BITS_GRID
    k_grid = tuple(args.k_grid) if args.k_grid else DEFAULT_K_GRID
    bdt_bits = tuple(args.bdt_bits) if args.bdt_bits else DEFAULT_BDT_BITS
    bdt_k = tuple(args.bdt_k_grid) if args.bdt_k_grid else DEFAULT_BDT_K_GRID

    out = {"files": files, "config": tables.config, "label": args.label,
           "delta_source": sample.source, "n_tracks": sample.n_tracks,
           "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "tolerance_dauc": args.tol, "saturation_max": args.sat_max,
           "bits_grid": list(bits_grid), "k_grid": list(k_grid),
           "bdt_bits_grid": list(bdt_bits), "bdt_k_grid": list(bdt_k), "fields": {}}

    base = None if args.no_bdt else _base_features(
        files, tables, len(y), args.max_events)

    chosen = {}
    for f in FIELDS:
        v = sample.deltas[f]
        nz = v[v > 0]
        octaves = (float(np.log2(np.quantile(nz, 0.999) / max(np.quantile(nz, 0.01), 1e-9)))
                   if len(nz) > 100 else float("nan"))
        rows = scan_field(v, y, bits_grid, k_grid)
        print(f"\n{f}: span {octaves:.1f} octaves, median {np.median(v):.3g}, "
              f"p99 {np.quantile(v, 0.99):.3g}, max {v.max():.3g}")

        if args.no_bdt:
            pick = choose_field(rows, args.tol, args.sat_max, key="dauc")
            bdt_rows = None
            print("  (scan-only: single-feature proxy, OPTIMISTIC for the angle fields)")
        else:
            cand = bdt_candidates(v, bdt_bits, bdt_k, args.sat_max)
            if not cand:
                raise RuntimeError(
                    f"{f}: no (bits, k) in the BDT grid keeps saturation <= "
                    f"{args.sat_max}; widen --bdt-bits or raise --sat-max")
            print(f"  measuring {len(cand)} candidate(s) with fits "
                  f"(saturation <= {args.sat_max:.0%})")
            bdt_rows = bdt_isolated_scan(base, v, y, cand, seed=args.seed,
                                         n_boot=max(args.n_boot // 2, 100),
                                         n_trees=60 if args.quick else 120)
            for r in bdt_rows:
                kname = "quantile" if r["k"] is None else f"k={r['k']}"
                print(f"    bits={r['bits']} {kname:<10} sat={r['saturation_frac']*100:5.2f}%"
                      f"  dAUC={r['dauc_bdt']:+.5f} [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")
            pick = choose_field(bdt_rows, args.tol, args.sat_max, key="dauc_bdt")

        # narrowest measured config that is statistically indistinguishable from
        # unquantized, so the bits/parity trade is visible rather than implied
        at_parity = None
        if bdt_rows:
            eq = [r for r in bdt_rows
                  if r["ci_lo"] <= 0 <= r["ci_hi"] and r["saturation_frac"] <= args.sat_max]
            if eq:
                at_parity = min(eq, key=lambda r: (r["bits"], r["saturation_frac"]))

        chosen[f] = pick
        out["fields"][f] = {"octaves_p1_p999": octaves, "median": float(np.median(v)),
                            "p99": float(np.quantile(v, 0.99)), "max": float(v.max()),
                            "frac_zero": float((v == 0).mean()),
                            "chosen": pick, "proxy_scan": rows, "bdt_scan": bdt_rows,
                            "min_config_at_parity": (
                                {"bits": at_parity["bits"], "k": at_parity["k"],
                                 "dauc_bdt": at_parity["dauc_bdt"],
                                 "edges": at_parity.get("edges")}
                                if at_parity else None)}
        flag = "" if pick["reached_parity"] else "   <-- NO PARITY IN GRID"
        dkey = "dauc" if args.no_bdt else "dauc_bdt"
        kname = "quantile LUT" if pick["k"] is None else f"k={pick['k']}"
        print(f"  chosen: bits={pick['bits']} {kname}  "
              f"dAUC={pick[dkey]:+.5f} sat={pick['saturation_frac']*100:.2f}% "
              f"H={pick['entropy_bits']:.2f}/{pick['bits']} "
              f"rho={pick['spearman_vs_float']:.4f} "
              f"top code from c>={pick['saturates_at']:.3g}{flag}")
        if not pick["reached_parity"]:
            print(f"  {f}: nothing in the grid reaches dAUC >= -{args.tol} with "
                  f"saturation <= {args.sat_max}; widen --bits (this field wants more "
                  f"resolution than the grid offers).")
        if bdt_rows:
            if at_parity is None:
                print(f"  no measured config for {f} is statistically equal to "
                      f"unquantized within this grid")
            elif (at_parity["bits"], at_parity["k"]) != (pick["bits"], pick["k"]):
                pname = ("quantile LUT" if at_parity["k"] is None
                         else f"k={at_parity['k']}")
                print(f"  statistical parity would need bits={at_parity['bits']} "
                      f"{pname} (dAUC {at_parity['dauc_bdt']:+.5f}); the choice "
                      f"above is {pick['bits']} bits within the {args.tol} tolerance")

    out["chosen"] = {f: {"bits": chosen[f]["bits"], "k": chosen[f]["k"],
                         "edges": chosen[f].get("edges")} for f in FIELDS}

    if not args.no_bdt:
        print("\nvalidating all four chosen codes jointly (paired, 5-fold OOF)...")
        out["bdt_validation"] = validate_with_bdt(
            sample, chosen, y, base, seed=args.seed, n_boot=args.n_boot,
            quick=args.quick)
        v = out["bdt_validation"]
        print(f"  AUC float={v['auc_float']:.5f}  coded={v['auc_coded']:.5f}  "
              f"dAUC={v['dauc']:+.5f} [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}]")

    consts = emit_constants(chosen)
    out["constants_snippet"] = consts
    print("\n" + consts)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.output}")
    return out
