#!/usr/bin/env python3
"""Extract a spec-compliant PixelAV/SmartPixels angle-response CorrectionSet from
ngtagger-train regression-eval parquet dumps.

Spec: L1Trigger/Phase3SmartPixels/doc/PixelAVAngleResponseSpec.md
Validator: L1Trigger/Phase3SmartPixels/python/validatePixelAVAngleSet.py

WHAT THIS CONSUMES
------------------
Per-cluster regression-eval parquet(s) produced EXTERNALLY by the smart-pixels-ml
regression evaluation (github.com/smart-pix/smart-pixels-ml, plus smart-pixels-ml-2bit-Synth
for the *-2bit variants; dumps published under CERNBox
regression_outputs/<dataset>/<tier>/...-vars.parquet, where <tier> is one of
full-precision / model-quantized / input-digitized).

They are NOT produced by ngtagger-train: this script contains no model, imports no ML
framework and evaluates no network. It is a pure reducer over already-computed per-cluster
predictions -- the NN's entire contribution arrives in the residual columns below. Record the
upstream source with --nn-source / --nn-dataset / --nn-tier so the payload provenance names
the real producer (the residual widths differ materially between quantization tiers, and the
input filename does not survive into the payload).

Each row is one PixelAV-simulated cluster; columns:
  cotA/cotB          : NN-PREDICTED cot angles  (SmartPixels 'alpha'/'beta' labels)
  cotAtrue/cotBtrue  : PixelAV TRUE cot angles
  residuals_cotA/_B  : (true - pred)   [note the sign; see RESIDUAL_IS_TRUE_MINUS_PRED]
  sigmacotA/_B       : NN's own predicted per-cluster uncertainty (Conv variants only)
The 'Mlp_Slim' variant regresses only beta (cotB); the Conv variants regress both.
ALL of these are stored NORMALISED (the training labels); physical = PIXELAV_LABEL_SCALE x stored
(see below). The payload is built in physical units.

ALPHA/BETA CONVENTION  (see ALPHA_BETA_MAPPING below -- the ONE place the mapping lives)
--------------------------------------------------------------------------------------
The SmartPixels/PixelAV 'alpha'/'beta' labels are SWAPPED relative to the CMSSW-local-frame
spec convention. Empirically (see study): PixelAV cotAlpha = n_x/n_z is the WIDE, symmetric,
coarse-pitch (200 um) non-bending / global-z direction; PixelAV cotBeta = n_y/n_z is the
narrow, fine-pitch (50 um) r-phi BENDING-plane direction. The spec's 'alpha' is the bending
plane. Therefore:  spec_alpha <- pixelav_BETA ,  spec_beta <- pixelav_ALPHA.

COVERAGE / LIMITATIONS honestly encoded as degenerate clamp axes:
  - No 'layer' column      -> one fit duplicated across layer categories 1..4.
  - No 'bLocalY' column    -> single clamp bin centred on the nominal -3.81 T.
  - Both angle axes are binned over the measured sample range (see the edge tables); the
    residual clamped fraction is measured at runtime and reported + recorded in provenance,
    so silent loss of resolution outside the binning cannot go unnoticed again.
All recorded in provenance per spec section 6.

MULTI-FILE / MULTI-VARIANT / INCREMENTAL
----------------------------------------
Input is a list/glob of parquet files, streamed row-group by row-group and accumulated into
a fixed binning as running (count, sum, sum-of-squares, and reservoir-free robust quantile
proxy via per-bin sample buffers capped in size). Adding more files later refines the SAME
binning without a redesign. --model-variant is mandatory: it enters provenance AND the output
filename (spec: variants are separate payload files, never extra axes).

Usage:
  extract_pixelav_angle_payload.py --model-variant Mlp_Slim-2bit --version v6 \\
      --out-dir eval_spixel_angles \\
      'PATH/2t-Mlp_Slim-2bit_optimized-vars.parquet' [more files/globs...]
"""
import argparse
import datetime
import glob
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import correctionlib.schemav2 as cs


# =====================================================================================
#  THE ALPHA/BETA MAPPING  -- the single, clearly-marked place the convention lives.
#  A future model variant using a different labelling only edits THIS function.
# =====================================================================================
# AUTHORITATIVE COORDINATE TRANSFORMATION (PixelAV author, 2026-07-22):
#   "pixelav was developed using our test beam coordinates. The cmssw coordinates were
#    defined later by others. The transformation is
#        x_cms = -y_pix,  y_cms = -x_pix,  z_cms = -z_pix"
# (a proper rotation, det = +1). Momenta transform like coordinates, so
#   cotAlpha_cms = p_x_cms/p_z_cms = (-p_y_pix)/(-p_z_pix) = +cotBeta_pix
#   cotBeta_cms  = p_y_cms/p_z_cms = (-p_x_pix)/(-p_z_pix) = +cotAlpha_pix
# i.e. for the COTANGENT angles the transformation reduces to a PURE SWAP with NO sign
# flip (the z negation cancels the transverse negation in both ratios). The same is NOT
# true of non-ratio quantities: local positions/residuals swap AND negate
# (dx_cms = -dy_pix), and local B components swap AND negate (B_y_cms = -B_x_pix) --
# relevant only if position residuals or a real bLocalY axis are ever sourced from
# PixelAV-side dumps.
# This analytic derivation CONFIRMS the earlier empirical verdict (study 2026-07-18:
# pitch layout + angle-range evidence): spec_alpha := pixelav_beta ;
# spec_beta := pixelav_alpha, values carried over unchanged.
SWAP_ALPHA_BETA = True

# The formula string embedded (redundantly) in the payload descriptions.
CMS_PIX_TRANSFORM = (
    "COORDINATE CONVENTION (PixelAV author, 2026-07-22): x_cms=-y_pix, y_cms=-x_pix, "
    "z_cms=-z_pix (proper rotation, det=+1) => cotAlpha_cms=+cotBeta_pix, "
    "cotBeta_cms=+cotAlpha_pix (PURE SWAP, no sign flip: z negation cancels in the "
    "ratios). Positions/residuals and local-B components instead swap AND negate "
    "(dx_cms=-dy_pix, B_y_cms=-B_x_pix)."
)

# The consumer contract (embedded in the CorrectionSet description and every compound;
# the producer implements exactly the PRIMARY form -- one evaluate per angle).
CONSUMER_CONTRACT = (
    "CONSUMER CONTRACT (PRIMARY, fused): cot(X)_meas = cot(X)_true"
    " + spix_angle_X_shift(layer,cotAlpha,cotBeta,bLocalY, prngAcc=1.0), X in {alpha,beta}"
    " -- ONE CompoundCorrection evaluate per angle; the trailing prngAcc input MUST be "
    "passed as 1.0 (it is the compound's internal accumulator seed). "
    "EQUIVALENT two-piece form (kept for visualization + cross-validation): "
    "cot(X)_meas = cot(X)_true + spix_angle_X_bias(layer,cotAlpha,cotBeta,bLocalY)"
    " + spix_angle_X_smear(layer,cotAlpha,cotBeta,bLocalY), where spix_angle_X_smear is the "
    "stack [spix_angle_X_sigma, spix_angle_prng(_beta)] with output_op '*' => sigma*N(0,1). "
    "The fused and two-piece forms agree bit-for-bit by construction (same rounded bias "
    "decimal, same prng node). Both reproduce a throw ~ N(bias, sigma) per bin. "
    "Validity gate: accept the synthesized angle pair iff "
    "spix_angle_valid_flat(inputs) < spix_angle_valid_prob(inputs)."
)

# Determinism + correlation semantics of the HashPRNG factorization (empirically probed
# with correctionlib 2.9.0; see eval_spixel_angles/hashprng_ab/probe_findings.json).
PRNG_SEMANTICS = (
    "HashPRNG determinism: the throw is a pure hash of the input tuple -- identical "
    "inputs give identical throws (a reproducibility FEATURE: replay/debug gets "
    "bit-identical synthesis; float-distinct angles give independent throws). All four "
    "inputs including the int 'layer' are entropy (int entropy verified working in the "
    "correctionlib C++ evaluator). Alpha and beta throws are INDEPENDENT, matching the "
    "legacy producer's two independent engine draws: spix_angle_prng (alpha) hashes "
    "(layer,cotAlpha,cotBeta,bLocalY); spix_angle_prng_beta hashes the DISTINCT "
    "permutation (cotBeta,bLocalY,layer,cotAlpha); spix_angle_valid_flat hashes the "
    "REVERSED order (bLocalY,cotBeta,cotAlpha,layer). Distinct entropy permutations "
    "give decorrelated hash streams (verified empirically)."
)

# Entropy permutations -- pairwise distinct, the decorrelation mechanism.
PRNG_HASH_ORDERS = {
    "spix_angle_prng": ["layer", "cotAlpha", "cotBeta", "bLocalY"],
    "spix_angle_prng_beta": ["cotBeta", "bLocalY", "layer", "cotAlpha"],
    "spix_angle_valid_flat": ["bLocalY", "cotBeta", "cotAlpha", "layer"],
}

# The eval dumps store the NORMALISED training labels. Physical value = scale x stored, for the
# prediction, the truth, the residual and the NN sigma alike (rescale() in smart-pixels-ml
# summary-plots/read-all-models-2s.ipynb). Payloads up to v4 were built from the stored values,
# so their CMS alpha axis/sigma were x1.88 and their CMS beta axis/sigma x6.53 too small.
PIXELAV_LABEL_SCALE = {"cotA": 6.528905408493022, "cotB": 1.8827422609846889}


def _source_label(spec_angle):
    """the dump label ('cotA' | 'cotB') that feeds spec 'alpha' | 'beta'"""
    return {"alpha": "cotB", "beta": "cotA"}[spec_angle] if SWAP_ALPHA_BETA else \
        {"alpha": "cotA", "beta": "cotB"}[spec_angle]


def ALPHA_BETA_MAPPING(cols_true, cols_pred, cols_resid):
    """Return (spec_alpha_dict, spec_beta_dict), each mapping 'true'/'pred'/'resid' to the
    source column name in the parquet (None if unavailable in this file), 'sigma_nn' to the NN
    sigma column and 'scale' to the physical-unit factor of that source label.

    cols_* are the sets of available column names for true / pred / residual."""
    def pick(kind_cols, name):
        return name if name in kind_cols else None

    def spec(src):
        return dict(true=pick(cols_true, f"{src}true"), pred=pick(cols_pred, src),
                    resid=pick(cols_resid, f"residuals_{src}"), sigma_nn=f"sigma{src}",
                    scale=PIXELAV_LABEL_SCALE[src])
    return spec(_source_label("alpha")), spec(_source_label("beta"))


# Residual column sign: verified empirically residuals == (true - pred) for all variants.
# Spec wants residual = (predicted - true), so bias(spec) = -median(residual_column).
RESIDUAL_IS_TRUE_MINUS_PRED = True


# =====================================================================================
#  Binning (spec input order: layer, cotAlpha, cotBeta, bLocalY).  layer + bLocalY are
#  DEGENERATE here (no such columns in the eval dumps).  cotAlpha (spec, = bending) is the
#  primary resolved axis; cotBeta (spec) gets a coarse axis where data supports it.
# =====================================================================================
LAYERS = [1, 2, 3, 4]
# v5 edges: sized to the ACTUAL sample range of the Conv1D_Full-2bit dump (108222 rows), chosen on
# the NORMALISED labels and kept as the same bins in physical units (so each cell holds the same
# clusters as in the v4 payload): in physical units spec-alpha (= pixelav cotB) spans
# [-2.03, +1.40] and spec-beta (= pixelav cotA) spans [-6.97, +6.98].
# spec-alpha (bending): normalised 0.1-wide core, widening to 0.2/0.3 in the sparse tails.
_NORM_ALPHA_EDGES = [-1.1, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0.0,
                     0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
# spec-beta (non-bending): coarse (|beta| 0-63 deg is one bin), symmetric.
_NORM_BETA_EDGES = [-1.1, -0.6, -0.3, 0.0, 0.3, 0.6, 1.1]
_phys = lambda angle, norm: [round(PIXELAV_LABEL_SCALE[_source_label(angle)] * e, 6) for e in norm]
SPEC_ALPHA_EDGES = _phys("alpha", _NORM_ALPHA_EDGES)
SPEC_BETA_EDGES = _phys("beta", _NORM_BETA_EDGES)
# v6: the same alpha edges without normalised 0.6 (cot 1.13): the dump ends at cot 1.40, so the top
# alpha bin holds only 2.5% of the rows, too few for fine beta cells. The beta edges are derived from
# the dump (derive_beta_edges), with the v5 outer edges kept so the clamp region is unchanged.
SPEC_ALPHA_EDGES_V6 = _phys("alpha", [e for e in _NORM_ALPHA_EDGES if e != 0.6])
BETA_ANGLE_STEP_DEG = 5
BETA_ANGLE_TOP_DEG = 80
# bLocalY: single degenerate clamp bin around the nominal field
BLOCALY_EDGES = [-4.0, 4.0]
MIN_BIN_COUNT = 25          # below this, fall back to the pooled (all-bins) estimate
SIGMA_FLOOR = 1e-4          # keep sigma strictly > 0 for the validator


class BinAccumulator:
    """Per-cell running store of residual samples, capped, so multi-file streaming refines
    the SAME binning.  Robust sigma = (q84-q16)/2, bias = median, over the accumulated buffer.
    A per-cell cap keeps memory bounded regardless of total row count (reservoir would be more
    rigorous; a hard cap is adequate here and deterministic)."""
    def __init__(self, shape, cap=200_000):
        self.shape = shape
        self.cap = cap
        self.buffers = {}   # flat_index -> list of residual values (spec-signed: pred-true)

    def add(self, flat_idx, values):
        for fi, val in zip(flat_idx, values):
            buf = self.buffers.setdefault(int(fi), [])
            if len(buf) < self.cap:
                buf.append(float(val))

    def pooled(self):
        allv = np.array([v for buf in self.buffers.values() for v in buf], dtype=np.float64)
        return allv

    def stats(self, flat_idx):
        buf = self.buffers.get(int(flat_idx), [])
        n = len(buf)
        if n == 0:
            return 0, np.nan, np.nan
        a = np.asarray(buf, dtype=np.float64)
        q16, q50, q84 = np.quantile(a, [0.16, 0.5, 0.84])
        return n, float(q50), float((q84 - q16) / 2.0)


def _edges_index(x, edges):
    """Clamp-index into [0, len(edges)-2]."""
    idx = np.digitize(x, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def derive_beta_edges(files, spec_alpha, spec_beta, alpha_edges, min_rows):
    """Beta (spec) edges symmetric in the ANGLE atan(cotBeta): grow each |beta| bin outwards from 0
    in BETA_ANGLE_STEP_DEG steps until every (alpha, beta) cell on both signs holds >= min_rows dump
    rows; the last bin runs to BETA_ANGLE_TOP_DEG, then one outer bin to the v5 outer edge."""
    cols = [spec_alpha["true"], spec_beta["true"]]
    t = {c: np.concatenate([np.asarray(pq.read_table(f, columns=[c]).column(c), np.float64) for f in files]) for c in cols}
    ia = _edges_index(spec_alpha["scale"] * t[cols[0]], alpha_edges)
    deg = np.degrees(np.arctan(spec_beta["scale"] * t[cols[1]]))
    na = len(alpha_edges) - 1

    def fills(lo, hi):
        return all(np.bincount(ia[(s * deg >= lo) & (s * deg < hi)], minlength=na).min() >= min_rows for s in (1, -1))

    top = SPEC_BETA_EDGES[-1]
    if not fills(BETA_ANGLE_TOP_DEG, 90.0):
        raise SystemExit(f"[fatal] the outer beta bin |beta| >= {BETA_ANGLE_TOP_DEG} deg has a cell below {min_rows} rows")
    ang = [0]
    for e in range(BETA_ANGLE_STEP_DEG, BETA_ANGLE_TOP_DEG + 1, BETA_ANGLE_STEP_DEG):
        if fills(ang[-1], e):
            ang.append(e)
    ang[-1] = BETA_ANGLE_TOP_DEG  # an unfilled remainder merges into the last bin
    if not fills(ang[-2], ang[-1]):
        raise SystemExit(f"[fatal] no beta binning gives >= {min_rows} rows per cell")
    pos = [round(float(np.tan(np.radians(a))), 6) for a in ang[1:]] + [top]
    return [-p for p in reversed(pos)] + [0.0] + pos


def _verify_residual_sign(d, spec, label):
    """Check RESIDUAL_IS_TRUE_MINUS_PRED against the data instead of trusting the comment.

    The dumps carry true, pred AND residual, so the convention -- which sign-flips every bias
    in the payload if wrong -- is verifiable at runtime from columns that were already being
    read. Measured on 2t-Mlp_Full-2bit (108222 rows): the declared convention reproduces the
    residual column bit-exactly (median|diff| = 0.0e+00) for cotA, cotB, x and y, while the
    opposite sign is off by ~3.3e-02. Returns the measured median deviation, or None when the
    variant dumps no prediction for this angle.
    """
    if not spec or not spec.get("pred") or spec["pred"] not in d:
        return None
    true, pred, resid = d[spec["true"]], d[spec["pred"]], d[spec["resid"]]
    tmp = float(np.median(np.abs((true - pred) - resid)))   # residual == true - pred ?
    pmt = float(np.median(np.abs((pred - true) - resid)))   # residual == pred - true ?
    stated, other = (tmp, pmt) if RESIDUAL_IS_TRUE_MINUS_PRED else (pmt, tmp)
    name = "true-pred" if RESIDUAL_IS_TRUE_MINUS_PRED else "pred-true"
    if not (stated <= 1e-6 and stated < other):
        raise SystemExit(
            f"[fatal] {label}: residual column '{spec['resid']}' does not follow the declared "
            f"convention RESIDUAL_IS_TRUE_MINUS_PRED={RESIDUAL_IS_TRUE_MINUS_PRED} ({name}): "
            f"median|({name}) - resid| = {stated:.3e}, versus {other:.3e} for the opposite "
            f"sign. Every bias in the payload would come out sign-flipped -- fix the flag (or "
            f"the input dump) before trusting this output.")
    return stated


def accumulate_files(files, spec_alpha, spec_beta, edges):
    """Stream all files row-group-wise; accumulate spec-signed residuals for alpha and (if
    available) beta into (n_alpha_bins, n_beta_bins) cells indexed by (spec_alpha, spec_beta).
    layer + bLocalY are degenerate (single bin). edges = (alpha edges, beta edges)."""
    alpha_edges, beta_edges = edges
    na = len(alpha_edges) - 1
    nb = len(beta_edges) - 1
    acc_alpha = BinAccumulator((na, nb))
    acc_beta = BinAccumulator((na, nb)) if spec_beta["true"] is not None else None
    # NN self-reported sigma accumulation (diagnostic, not part of payload)
    nn_sigma_alpha, nn_sigma_beta = [], []
    total_rows = 0
    # residual-sign verification (once, on the first batch that has predictions) and
    # clamped-row counting: rows falling outside the binned range get no resolution, so the
    # fraction is measured rather than assumed.
    sign_checked = False
    sign_dev = dict(alpha=None, beta=None)
    clamped_alpha = clamped_beta = 0
    beta_axis_rows = 0
    a_lo, a_hi = alpha_edges[0], alpha_edges[-1]
    b_lo, b_hi = beta_edges[0], beta_edges[-1]

    # which columns to read
    read_cols = set()
    for d in (spec_alpha, spec_beta):
        for k in ("true", "pred", "resid"):
            if d and d[k]:
                read_cols.add(d[k])

    for path in files:
        pf = pq.ParquetFile(path)
        have = set(f.name for f in pf.schema_arrow)
        cols = [c for c in read_cols if c in have]
        sig_a = spec_alpha["sigma_nn"] if spec_alpha["sigma_nn"] in have else None
        sig_b = spec_beta["sigma_nn"] if (spec_beta and spec_beta["sigma_nn"] in have) else None
        read = cols + [c for c in (sig_a, sig_b) if c]
        scale = {spec[k]: spec["scale"] for spec in (spec_alpha, spec_beta) if spec
                 for k in ("true", "pred", "resid", "sigma_nn") if spec.get(k)}
        for batch in pf.iter_batches(batch_size=200_000, columns=read):
            d = {c: scale[c] * np.asarray(batch.column(c)).astype(np.float64) for c in read}
            n = len(next(iter(d.values())))
            total_rows += n
            if not sign_checked:
                dev_a = _verify_residual_sign(d, spec_alpha, "spec-alpha")
                dev_b = _verify_residual_sign(d, spec_beta, "spec-beta")
                if dev_a is not None or dev_b is not None:
                    sign_dev = dict(alpha=dev_a, beta=dev_b)
                    sign_checked = True
                    print(f"[ok] residual sign convention verified against the data "
                          f"(true-pred): median deviation alpha={dev_a} beta={dev_b}")
            # spec-signed residuals
            a_true = d[spec_alpha["true"]]
            a_resid = d[spec_alpha["resid"]]
            a_pred_minus_true = -a_resid if RESIDUAL_IS_TRUE_MINUS_PRED else a_resid
            clamped_alpha += int(np.count_nonzero((a_true < a_lo) | (a_true > a_hi)))
            # bin index: alpha axis from spec-alpha-true; beta axis from spec-beta-true if present
            ia = _edges_index(a_true, alpha_edges)
            if spec_beta and spec_beta["true"] and spec_beta["true"] in d:
                b_true = d[spec_beta["true"]]
                clamped_beta += int(np.count_nonzero((b_true < b_lo) | (b_true > b_hi)))
                beta_axis_rows += n
                ib = _edges_index(b_true, beta_edges)
            else:
                # no spec-beta axis data -> put everything in the central beta bin
                ib = np.full(n, (nb - 1) // 2, dtype=int)
            flat = ia * nb + ib
            acc_alpha.add(flat, a_pred_minus_true)
            if acc_beta is not None:
                b_resid = d[spec_beta["resid"]]
                b_pmt = -b_resid if RESIDUAL_IS_TRUE_MINUS_PRED else b_resid
                acc_beta.add(flat, b_pmt)
            if sig_a:
                nn_sigma_alpha.append(d[sig_a])
            if sig_b:
                nn_sigma_beta.append(d[sig_b])

    diag = dict(
        total_rows=total_rows,
        nn_sigma_alpha_med=float(np.median(np.concatenate(nn_sigma_alpha))) if nn_sigma_alpha else None,
        nn_sigma_beta_med=float(np.median(np.concatenate(nn_sigma_beta))) if nn_sigma_beta else None,
        residual_sign_verified=sign_checked,
        residual_sign_dev=sign_dev,
        clamped_frac_alpha=(clamped_alpha / total_rows) if total_rows else None,
        clamped_frac_beta=(clamped_beta / beta_axis_rows) if beta_axis_rows else None,
    )
    return acc_alpha, acc_beta, diag


def _nbins(edges):
    return len(edges[0]) - 1, len(edges[1]) - 1


def _content_values(acc, kind, na, nb):
    """Per-cell rounded values (row-major over (cotAlpha, cotBeta)); kind in {'sigma','bias'}.
    The SINGLE source of the payload numerics: the plain bias/sigma tables AND the fused
    shift compound's Formula constants come from here, so fused == bias + smear exactly."""
    pooled = acc.pooled() if acc is not None else np.array([])
    if pooled.size:
        pq16, pq50, pq84 = np.quantile(pooled, [0.16, 0.5, 0.84])
        pool_sigma = max(SIGMA_FLOOR, float((pq84 - pq16) / 2.0))
        pool_bias = float(pq50)
    else:
        pool_sigma, pool_bias = 0.05, 0.0   # inert fallback (unavailable angle)
    content = []
    for ia in range(na):
        for ib in range(nb):
            flat = ia * nb + ib
            if acc is None:
                sig, bias, n = pool_sigma, 0.0, 0
            else:
                n, med, robsig = acc.stats(flat)
                if n < MIN_BIN_COUNT or not np.isfinite(robsig):
                    sig, bias = pool_sigma, pool_bias
                else:
                    sig, bias = max(SIGMA_FLOOR, robsig), med
            content.append(round(sig if kind == "sigma" else bias, 6))
    return content


def _multibinning(acc, kind, edges):
    """Build a schemav2 MultiBinning over (cotAlpha, cotBeta, bLocalY) from an accumulator.
    kind in {'sigma','bias'}.  bLocalY has one bin (degenerate)."""
    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=["cotAlpha", "cotBeta", "bLocalY"],
        edges=[*edges, BLOCALY_EDGES],
        content=_content_values(acc, kind, *_nbins(edges)),
        flow="clamp",
    )


def _final_multibinning(acc, edges):
    """The fused-shift terminal node: bias table whose cells are Formula nodes
    '<bias> + x' over the prngAcc accumulator (which holds sigma*z when reached),
    so the cell outputs bias + sigma*z. Bias constants are the SAME rounded decimals
    as the plain bias table (repr of round(...,6)) -> fused == bias + smear exactly."""
    content = [
        cs.Formula(nodetype="formula",
                   expression=f"{repr(bias)} + x",
                   parser="TFormula",
                   variables=["prngAcc"])
        for bias in _content_values(acc, "bias", *_nbins(edges))
    ]
    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=["cotAlpha", "cotBeta", "bLocalY"],
        edges=[*edges, BLOCALY_EDGES],
        content=content,
        flow="clamp",
    )


def _prob_multibinning(edges):
    """valid_prob: no validity/quality flag in these eval dumps -> 1.0 everywhere."""
    na, nb = _nbins(edges)
    ncell = na * nb  # * 1 bLocalY bin
    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=["cotAlpha", "cotBeta", "bLocalY"],
        edges=[*edges, BLOCALY_EDGES],
        content=[1.0] * ncell,
        flow="clamp",
    )


def _category(builder):
    return cs.Category(
        nodetype="category", input="layer",
        content=[cs.CategoryItem(key=l, value=builder()) for l in LAYERS],
    )


def _correction(name, desc, builder):
    return cs.Correction(
        name=name, description=desc, version=0,
        inputs=[
            cs.Variable(name="layer", type="int", description="TBPX layer 1-4 (degenerate: one fit duplicated)"),
            cs.Variable(name="cotAlpha", type="real",
                        description="spec-alpha = CMSSW-local bending plane p_x/p_z = PixelAV cotBeta. " + CMS_PIX_TRANSFORM),
            cs.Variable(name="cotBeta", type="real",
                        description="spec-beta = CMSSW-local non-bending p_y/p_z = PixelAV cotAlpha. " + CMS_PIX_TRANSFORM),
            cs.Variable(name="bLocalY", type="real",
                        description="CMSSW local-y B [T]; degenerate single clamp bin at -3.81. WARNING if a real "
                                    "B axis is ever sourced from PixelAV dumps: B_y_cms = -B_x_pix (swap AND negate)."),
        ],
        output=cs.Variable(name=name, type="real"),
        data=_category(builder),
    )


def _prng_inputs():
    """The shared 4-input signature, identical (names/types/order) to the plain corrections."""
    return [
        cs.Variable(name="layer", type="int", description="TBPX layer 1-4 (entropy)"),
        cs.Variable(name="cotAlpha", type="real", description="spec-alpha true angle (entropy). " + CMS_PIX_TRANSFORM),
        cs.Variable(name="cotBeta", type="real", description="spec-beta true angle (entropy). " + CMS_PIX_TRANSFORM),
        cs.Variable(name="bLocalY", type="real", description="CMSSW local-y B [T] (entropy)"),
    ]


def _shift_inputs():
    """The fused-compound signature: the 4 plain inputs + the prngAcc accumulator seed."""
    return _prng_inputs() + [
        cs.Variable(name="prngAcc", type="real",
                    description="internal accumulator seed -- the consumer MUST pass 1.0; "
                                "the compound updates it to sigma, then sigma*N(0,1)"),
    ]


def _prng_name(x):
    """The stdnormal throw node for angle x (distinct entropy permutations => independent)."""
    return "spix_angle_prng" if x == "alpha" else "spix_angle_prng_beta"


def _prng_correction(name, dist):
    """A deterministic HashPRNG variate node hashed from the input tuple (order per
    PRNG_HASH_ORDERS -- distinct permutations decorrelate the streams)."""
    order = PRNG_HASH_ORDERS[name]
    what = ("standard-normal deviate (stdnormal)" if dist == "stdnormal"
            else "U(0,1) gate variate (stdflat)")
    role = {
        "spix_angle_prng": "the cotAlpha smear throw",
        "spix_angle_prng_beta": "the cotBeta smear throw (independent of alpha)",
        "spix_angle_valid_flat": ("the validity gate: accept iff "
                                 "spix_angle_valid_flat(inputs) < spix_angle_valid_prob(inputs)"),
    }[name]
    return cs.Correction(
        name=name,
        description=(f"Deterministic {what} for {role}; HashPRNG entropy order "
                     f"({', '.join(order)}). " + PRNG_SEMANTICS + " " + CONSUMER_CONTRACT),
        version=0,
        inputs=_prng_inputs(),
        output=cs.Variable(name=name, type="real"),
        data=cs.HashPRNG(nodetype="hashprng", inputs=order, distribution=dist),
    )


def _final_correction(x, acc, edges, prov):
    """spix_angle_{x}_final: terminal node of the fused shift compound. Same
    (layer category; cotAlpha,cotBeta,bLocalY multibinning) structure as the bias
    table, but each cell is the Formula '<bias> + prngAcc' -> bias + sigma*N(0,1)
    when reached through the compound."""
    return cs.Correction(
        name=f"spix_angle_{x}_final",
        description=(f"Fused-shift terminal for cot{x.capitalize()}: per-bin Formula "
                     f"'<bias> + prngAcc' with the SAME rounded bias constants as "
                     f"spix_angle_{x}_bias. Only meaningful inside spix_angle_{x}_shift "
                     f"(where prngAcc has been updated to sigma*N(0,1)); standalone "
                     f"evaluation returns bias + whatever prngAcc is passed. "
                     + CONSUMER_CONTRACT + " " + prov),
        version=0,
        inputs=_shift_inputs(),
        output=cs.Variable(name=f"spix_angle_{x}_final", type="real"),
        data=cs.Category(
            nodetype="category", input="layer",
            content=[cs.CategoryItem(key=l, value=_final_multibinning(acc, edges))
                     for l in LAYERS]),
    )


def _smear_compound(x, prov):
    """spix_angle_{x}_smear = spix_angle_{x}_sigma * spix_angle_prng[_beta] (output_op '*').
    The two-piece stochastic term -- kept for visualization + cross-validation."""
    return cs.CompoundCorrection(
        name=f"spix_angle_{x}_smear",
        description=(f"Stochastic term for cot{x.capitalize()} synthesis: CompoundCorrection "
                     f"stack [spix_angle_{x}_sigma, {_prng_name(x)}], output_op '*' => "
                     f"sigma(inputs) * N(0,1). " + CONSUMER_CONTRACT + " " + PRNG_SEMANTICS
                     + " " + prov),
        inputs=_prng_inputs(),
        output=cs.Variable(name=f"spix_angle_{x}_smear", type="real",
                           description=f"additive stochastic term for cot{x.capitalize()}_meas"),
        inputs_update=[],
        input_op="*",
        output_op="*",
        stack=[f"spix_angle_{x}_sigma", _prng_name(x)],
    )


def _shift_compound(x, prov):
    """spix_angle_{x}_shift: the FUSED bias + sigma*N(0,1) in ONE compound evaluate.
    Mechanism: prngAcc starts at 1.0 (consumer-passed); inputs_update=['prngAcc'] with
    input_op '*' folds each stack output into it (1 -> sigma -> sigma*z); the terminal
    spix_angle_{x}_final returns bias + prngAcc; output_op 'last' emits that."""
    return cs.CompoundCorrection(
        name=f"spix_angle_{x}_shift",
        description=(f"FUSED synthesis shift for cot{x.capitalize()}: stack "
                     f"[spix_angle_{x}_sigma, {_prng_name(x)}, spix_angle_{x}_final], "
                     f"inputs_update=['prngAcc'], input_op '*', output_op 'last' => "
                     f"bias(inputs) + sigma(inputs)*N(0,1) in one evaluate "
                     f"(pass prngAcc=1.0). " + CONSUMER_CONTRACT + " " + PRNG_SEMANTICS
                     + " " + prov),
        inputs=_shift_inputs(),
        output=cs.Variable(name=f"spix_angle_{x}_shift", type="real",
                           description=f"additive total shift for cot{x.capitalize()}_meas"),
        inputs_update=["prngAcc"],
        input_op="*",
        output_op="last",
        stack=[f"spix_angle_{x}_sigma", _prng_name(x), f"spix_angle_{x}_final"],
    )


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_payload(variant, files, acc_alpha, acc_beta, diag, upstream, edges, binning):
    beta_measurable = acc_beta is not None
    cf_a = diag["clamped_frac_alpha"]
    cf_b = diag["clamped_frac_beta"]
    prov = (
        f"SmartPixels PixelAV angle-response payload | model-variant={variant} | "
        # Attribution split: the NN evaluation happens upstream, this script only reduces its
        # per-cluster dumps into binned bias/sigma. Conflating the two misled a reader once.
        f"NN EVALUATION (upstream; produced the input dumps): source={upstream['source']} "
        f"dataset={upstream['dataset']} tier={upstream['tier']} | "
        f"REDUCTION (this payload): ngtagger-train "
        f"eval_spixel_angles/extract_pixelav_angle_payload.py fitrev={git_rev()} -- no model "
        f"is evaluated here | "
        f"files={[os.path.basename(f) for f in files]} | rows={diag['total_rows']} | "
        f"date={datetime.date.today().isoformat()} | "
        f"MAPPING: SmartPixels alpha/beta SWAPPED vs CMSSW spec frame "
        f"(spec_alpha<-pixelav_beta[cotB], spec_beta<-pixelav_alpha[cotA]); "
        f"residual column = (true-pred), spec bias = -median(residual); convention VERIFIED "
        f"against this dump at runtime (verified={diag['residual_sign_verified']}, median "
        f"deviation alpha={diag['residual_sign_dev']['alpha']} "
        f"beta={diag['residual_sign_dev']['beta']}). | "
        f"{CMS_PIX_TRANSFORM} | "
        f"UNITS: PHYSICAL -- every dump column (true, pred, residual, NN sigma) multiplied by the "
        f"smart-pixels-ml rescale() factor of its label {PIXELAV_LABEL_SCALE}; payloads before v5 "
        f"used the stored normalised values (CMS alpha x1.88, CMS beta x6.53 too small). | "
        f"COVERAGE LIMITS: no layer column (one fit duplicated across layers 1-4); "
        f"no bLocalY column (single clamp bin at -3.81 T); "
        f"beta(spec) measurable={beta_measurable}; "
        f"axis edges cover the measured sample range -- clamped row fraction "
        f"alpha={'%.4f' % cf_a if cf_a is not None else 'n/a'} "
        f"beta={'%.4f' % cf_b if cf_b is not None else 'n/a'}; "
        f"BINNING {binning}: cotAlpha edges {list(edges[0])}, cotBeta edges {list(edges[1])}; "
        f"valid_prob=1.0 (no per-cluster validity flag in eval dumps). "
        f"NN self-reported sigma median: alpha={diag['nn_sigma_alpha_med']} beta={diag['nn_sigma_beta_med']}."
    )
    corrs = [
        _correction("spix_angle_alpha_sigma", "sigma(cotAlpha_NN - cotAlpha_true), robust (q84-q16)/2. " + prov,
                    lambda: _multibinning(acc_alpha, "sigma", edges)),
        _correction("spix_angle_alpha_bias", "median(cotAlpha_NN - cotAlpha_true). " + prov,
                    lambda: _multibinning(acc_alpha, "bias", edges)),
        _correction("spix_angle_beta_sigma",
                    ("sigma(cotBeta_NN - cotBeta_true), robust. " if beta_measurable
                     else "beta NOT measurable in this variant (no cotA prediction); inert positive sigma. ") + prov,
                    lambda: _multibinning(acc_beta, "sigma", edges)),
        _correction("spix_angle_beta_bias",
                    ("median(cotBeta_NN - cotBeta_true). " if beta_measurable
                     else "beta NOT measurable in this variant; zero bias. ") + prov,
                    lambda: _multibinning(acc_beta, "bias", edges)),
        _correction("spix_angle_valid_prob", "P(NN emits usable angle). No validity flag -> 1.0 everywhere. " + prov,
                    lambda: _prob_multibinning(edges)),
        _prng_correction("spix_angle_prng", "stdnormal"),
        _prng_correction("spix_angle_prng_beta", "stdnormal"),
        _prng_correction("spix_angle_valid_flat", "stdflat"),
        _final_correction("alpha", acc_alpha, edges, prov),
        _final_correction("beta", acc_beta, edges, prov),
    ]
    compounds = [_smear_compound("alpha", prov), _smear_compound("beta", prov),
                 _shift_compound("alpha", prov), _shift_compound("beta", prov)]
    cset = cs.CorrectionSet(
        schema_version=2,
        description=prov + " | " + CONSUMER_CONTRACT + " " + PRNG_SEMANTICS,
        corrections=corrs,
        compound_corrections=compounds,
    )
    return cset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="parquet files and/or globs")
    ap.add_argument("--model-variant", required=True, help="e.g. Mlp_Slim-2bit (into provenance + filename)")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--gzip", action="store_true", help="write .json.gz")
    ap.add_argument("--version", default="v6",
                    help="payload version, appended to the file name (never overwrite a deployed payload)")
    ap.add_argument("--binning", choices=("v5", "v6"), default="v6",
                    help="v5: the fixed coarse edges (|beta| 0-63 deg is one bin); v6: beta edges symmetric in "
                         "angle, derived from the dump with >= --min-cell-rows rows per (alpha, beta) cell")
    ap.add_argument("--min-cell-rows", type=int, default=100, help="v6 binning: minimum dump rows per cell")
    # Upstream attribution: the NN evaluation that produced the input dumps is NOT done here.
    ap.add_argument("--nn-source", default="smart-pix/smart-pixels-ml (rev unrecorded)",
                    help="repo/commit that evaluated the NN and wrote the input parquet(s)")
    ap.add_argument("--nn-dataset", default="unrecorded",
                    help="PixelAV dataset the NN was evaluated on, e.g. dataset_3src_16x16_50x12P5")
    ap.add_argument("--nn-tier", default="unrecorded",
                    help="quantization tier of the dump: full-precision | model-quantized | "
                         "input-digitized (residual widths differ materially between them)")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        m = sorted(glob.glob(pat))
        files.extend(m if m else ([pat] if os.path.exists(pat) else []))
    if not files:
        ap.error("no input files matched")

    # discover available columns from the first file to drive the mapping
    have = set(f.name for f in pq.ParquetFile(files[0]).schema_arrow)
    spec_alpha, spec_beta = ALPHA_BETA_MAPPING(have, have, have)
    if spec_alpha["true"] is None or spec_alpha["resid"] is None:
        ap.error(f"spec-alpha source columns missing in {files[0]} (need cotBtrue/residuals_cotB under swap)")
    if spec_beta["true"] is None:  # variant regresses only the bending angle (Mlp_Slim)
        print("[info] beta (spec) not present in this variant -> beta corrections inert.")

    print(f"[info] {len(files)} file(s); variant={args.model_variant}")
    print(f"[info] mapping: spec_alpha<-{spec_alpha['true']}/{spec_alpha['pred']} x{spec_alpha['scale']}; "
          f"spec_beta<-{spec_beta['true']} x{spec_beta['scale']}")
    if args.binning == "v6" and spec_beta["true"] is not None:
        edges = (SPEC_ALPHA_EDGES_V6,
                 derive_beta_edges(files, spec_alpha, spec_beta, SPEC_ALPHA_EDGES_V6, args.min_cell_rows))
        binning = f"v6 (beta symmetric in angle, >= {args.min_cell_rows} dump rows per cell)"
    else:
        edges = (SPEC_ALPHA_EDGES, SPEC_BETA_EDGES)
        binning = "v5"
    print(f"[info] binning {binning}: {len(edges[0]) - 1} cotAlpha x {len(edges[1]) - 1} cotBeta bins; "
          f"cotBeta edges {edges[1]}")
    acc_alpha, acc_beta, diag = accumulate_files(files, spec_alpha, spec_beta, edges)
    print(f"[info] accumulated {diag['total_rows']} rows")

    cf_a, cf_b = diag["clamped_frac_alpha"], diag["clamped_frac_beta"]
    print("[info] clamped rows (outside the binned range): "
          + (f"alpha={cf_a:.4%}" if cf_a is not None else "alpha=n/a")
          + (f" beta={cf_b:.4%}" if cf_b is not None else ""))
    if cf_a is not None and cf_a > 0.02:
        print(f"[warn] {cf_a:.2%} of rows fall outside the cotAlpha edges "
              f"[{edges[0][0]}, {edges[0][-1]}] and are clamped into the edge "
              f"bins -- they get no angular resolution. Widen the edges to the sample range.")
    if cf_b is not None and cf_b > 0.02:
        print(f"[warn] {cf_b:.2%} of rows fall outside the cotBeta edges "
              f"[{edges[1][0]}, {edges[1][-1]}] and are clamped.")

    upstream = dict(source=args.nn_source, dataset=args.nn_dataset, tier=args.nn_tier)
    cset = build_payload(args.model_variant, files, acc_alpha, acc_beta, diag, upstream, edges, binning)

    os.makedirs(args.out_dir, exist_ok=True)
    ext = ".json.gz" if args.gzip else ".json"
    out = os.path.join(args.out_dir, f"spix_angle_response_{args.model_variant}_{args.version}{ext}")
    raw = cset.json(exclude_unset=True)  # schemav2 model -> json string
    opener = gzip.open if args.gzip else open
    with opener(out, "wt") as f:
        f.write(raw)
    print(f"[ok] wrote {out}")
    return out


if __name__ == "__main__":
    main()
