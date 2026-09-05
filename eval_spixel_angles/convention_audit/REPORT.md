# Convention audit: PixelAV -> CMSSW cot-angle mapping (adversarial)

Claim: `x_cms=-y_pix, y_cms=-x_pix, z_cms=-z_pix` (det=+1) => cotAlpha_cms=+cotBeta_pix,
cotBeta_cms=+cotAlpha_pix - PURE SWAP, no sign flip - as implemented by
`SWAP_ALPHA_BETA=True` in `eval_spixel_angles/extract_pixelav_angle_payload.py` and
deployed in `spx_angle_response_Conv1D_Full-2bit_v4fixed.json`.
Alternatives tested: NO-SWAP; SWAP+SIGN-FLIP; residual-sign error.
All numbers re-derived from data (evidence.json); no code comments trusted.

## Verdict: **CONFIRMED**

The pure-swap mapping is correct in axis assignment AND in sign. The decisive new
data test is the Lorentz-compensation dip (leg 5a): the CMSSW digitized clusters are
shortest at trk_cotAlpha = -0.15 (fit -0.185, argmin bin -0.137), exactly where the
PixelAV-side bending response is worst (NN sigmacotB peak at source cotBtrue = -0.15).
Same signed location => no mirror; a SWAP+FLIP mapping would demand the dip at +0.15,
which the data excludes. NO-SWAP is excluded by geometry: the PixelAV grid's
cotAlpha spans +-8.6 (flat) - feeding that to the CMSSW alpha axis (physical range
+-0.56, domain +-0.6) is impossible, while it covers the CMSSW beta range (+-6.0)
with the spec-mandated margin.

## Per-leg results

| Leg | Strength | Result |
|---|---|---|
| 1 analytic | decisive (flip-vs-no-flip GIVEN the author transform) | det(R)=+1 exactly; numeric check: cotAlpha_cms==+cotBeta_pix, cotBeta_cms==+cotAlpha_pix to machine precision (err 0.0). A sign flip requires an improper map (e.g. z not negated, det=-1). Residual columns re-derived: residuals_cot* == (true - pred) EXACTLY (max err 0.0), so the extractor's negation to spec (pred-true) is required and correct. |
| 2 source geometry | decisive (swap vs no-swap) | part.93: cotAlpha == n_x/n_z, cotBeta == n_y/n_z exactly (physical, unscaled). cotAlpha_pix is the WIDE flat grid (q01..q99 = -8.58..+8.62); cotBeta_pix narrower (-3.62..+3.68). x-midplane spans 200 um (long axis), y-midplane 50 um (fine axis); n_z strictly negative. Eval dumps: cotBtrue is Lorentz-shifted (med -0.176, 62% negative) with NN sigmacotB peaked at -0.15 (bending signature); cotAtrue symmetric with sigmacotA peaked at ~0 (non-bending). |
| 3 CMSSW ranges | decisive (swap vs no-swap) | Re-derived from payload_v4fixed_numEvent300.root (140754 crossings), independent of the spec doc: L1 trk_cotAlpha q01..q99 = -0.287..+0.475 (max extent -0.31..+0.56) - matches doc table. trk_cotBeta = +-5.3 (extent +-6.0). b_localy = -3.811 T for 100% of crossings. Under SWAP the wide pixelav axis covers the +-6 beta axis; under NO-SWAP the +-8.6 grid would collide with the +-0.6 alpha axis and the +-3.7 grid would undercover beta beyond 3.7. |
| 4 end-to-end pulls | supporting + one flagged limitation | Producer (L1SmartPixelsTrackProducer.cc:1707-1721) evaluates the payload at CMSSW-local parent angles plv.x/plv.z, plv.y/plv.z and synthesizes measurements with the payload's own bias+sigma; sidecar pulls are Kalman innovation pulls including wrong-hit contamination and prediction error (pullAlpha w=3.48, pullBeta w=2.48), so width>1 is NOT a mapping diagnostic. Discriminating observable: deployed sigBeta-vs-parCotBeta profile reproduces the implemented payload's beta_sigma curve (flat 0.018 / peak 0.027 near 0 / 0.014 above +1) - the deployed keying matches the swap payload, not a transposed one. FLAG: pullBeta width grows 2.6 -> 4.9 for |cotBeta| 0 -> 4-6: consequence of the documented source undercoverage (44.4% of real crossings have |cotBeta|>1.07, the eval-sample limit, where sigma is a clamped edge value). Coverage limitation, not a convention error - no-swap would be equally undercovered (eval cotB also spans only +-1.08) and additionally mis-keyed. |
| 5a Lorentz dip sign | **decisive (flip-vs-no-flip, from data)** | CMSSW cluster x-extent vs trk_cotAlpha has its minimum at -0.185 (fit) / -0.137 (argmin), matching the pixelav bending proxy at -0.15 in sign and magnitude; control y-extent dip at -0.011 ~ 0 (no Lorentz along beta). Kills the mirrored alternative (+0.15) and simultaneously validates that the PixelAV sample's B orientation corresponds to the deployed modules' bLocalY = -3.811 T. |
| 5b counterfactual A/B | supporting | Built no-swap payload (scratch/counterfactual_noswap_Conv1D_Full-2bit.json, extractor file untouched) and evaluated both over all 140754 real crossings: alpha_sigma differs by >20% for 64.4% of tracks (median |log ratio| 0.29; impl med 0.0195 vs cf 0.0258); beta_sigma >20% for 68.9%. Implemented alpha_sigma has its maximum on the negative-alpha side, coinciding with the measured short-cluster dip (-0.15); the counterfactual peaks at 0..+0.1 - inconsistent with the measured response. Bias curves differ by up to 0.02-0.03 (same order as the bias itself). 0% of real tracks fall outside the payload domains (+-0.6 / +-6.0). |
| 6 description/bit identity | decisive (deployment integrity) | Deployed v4fixed contains "x_cms=-y_pix" in the CorrectionSet description, all 5 correction descriptions, and the cotAlpha and cotBeta variable descriptions of every correction. Numeric content with descriptions stripped is IDENTICAL to eval_spixel_angles/spx_angle_response_Conv1D_Full-2bit.json. |

## Prediction vs measured

| Observable | PURE SWAP | NO-SWAP | SWAP+FLIP | Measured |
|---|---|---|---|---|
| det of claimed R | +1 | - | needs det=-1 map | +1 exact |
| pixelav wide axis -> CMSSW wide axis | cotAlpha_pix(+-8.6)->beta(+-6) | cotBeta_pix(+-3.7)->beta | same as swap | pixelav cotAlpha +-8.6 flat; CMSSW beta +-6.0; CMSSW alpha +-0.56 |
| long pitch axis pairing | x_pix(200um) <-> y_cms(long) | x_pix <-> x_cms | same as swap | x-midplane 200um, y-midplane 50um; CMSSW y-extent reaches 800um, x-extent ~100um |
| CMSSW x-extent dip location | -0.15 | (n/a; keys beta) | **+0.15** | **-0.185 (fit) / -0.137 (bin)** |
| y-extent dip (control) | 0 | - | 0 | -0.011 |
| residual column sign | (true-pred), negate | - | - | exactly (true-pred), max err 0.0 |
| deployed = eval numerics | identical | - | - | identical |

## Adversarial notes (reported plainly, not harmonized)

1. The raw sample-asymmetry comparison initially looked like a CONTRADICTION: source
   cotBtrue is negative-shifted (med -0.176) while CMSSW trk_cotAlpha is positive-shifted
   (med +0.036). This is NOT evidence for a flip: the source shift is a deliberate
   sampling choice around the drift-compensation region, while the CMSSW track-incidence
   skew is geometric. The physically meaningful comparator - the cluster-size /
   resolution-response dip - agrees in sign at -0.15 on both sides (leg 5a). The spec's
   statement that "the Lorentz bias SIGN is not measurable without a two-sign bLocalY
   scan" is now superseded for this sample: leg 5a measures it, and it is consistent.
2. Real limitation (independent of convention): 44% of deployed crossings sit beyond the
   source beta coverage (|cotBeta|>1.07) and receive clamped edge sigmas; innovation-pull
   widths grow to ~5 there (leg 4). Extending the PixelAV eval sample in the non-bending
   angle is the fix; the convention needs no change.
3. Parent-momentum true angles in the analyzer contain pathological tails (|cot| up to
   2e5, grazing p_z) - the producer's measured-angle clamp handles these; they do not
   affect the payload derivation.

## Plots (eval_spixel_angles/convention_audit/plots/)

- leg2_source_angle_hists.png - source true-angle distributions: cotAtrue symmetric, cotBtrue Lorentz-shifted negative.
- leg2c_response_vs_source_angles.png - NN bias/sigma vs source angles: sigmacotB peaks at -0.15 (bending), sigmacotA at ~0 (non-bending).
- leg3_range_overlay.png - CMSSW local angles vs source angles under SWAP / NO-SWAP / SWAP+FLIP; only SWAP covers both axes sensibly.
- leg4_pulls_and_sigma_keying.png - deployed refit pulls, pullBeta growth vs |cotBeta| (coverage flag), and sigma keying vs parent angles.
- leg5a_lorentz_dip_sign.png - decisive sign test: cluster x-extent dip at cotAlpha=-0.15 (matches pixelav proxy; mirrored +0.15 excluded); y-extent control V at 0.
- leg5b_counterfactual_ab.png - implemented vs no-swap payload response curves and assigned sigmas over real tracks (64-69% differ by >20%).

Evidence numbers: evidence.json. Scripts: scripts/ (00-01 from prior attempt, 10-60 this run).
Counterfactual payload: scratch/counterfactual_noswap_Conv1D_Full-2bit.json (audit artifact only - do not deploy).

**VERDICT: CONFIRMED** - pure swap, no sign flip, residual negation correct, deployed
payload bit-identical to the extractor output with the convention documented in every
required description field.
