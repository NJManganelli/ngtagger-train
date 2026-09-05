# Convention Audit Plan (adversarial)

Claim under audit: x_cms=-y_pix, y_cms=-x_pix, z_cms=-z_pix ⇒ cotAlpha_cms=+cotBeta_pix,
cotBeta_cms=+cotAlpha_pix (PURE SWAP, no sign flip), as implemented by SWAP_ALPHA_BETA=True
in eval_spixel_angles/extract_pixelav_angle_payload.py.

Alternatives to falsify against:
- NO-SWAP: cotAlpha_cms=cotAlpha_pix, cotBeta_cms=cotBeta_pix
- SWAP+FLIP: cotAlpha_cms=-cotBeta_pix and/or cotBeta_cms=-cotAlpha_pix
- Residual-sign error: residuals not negated (or double-negated)

Legs (append results to evidence.json after each):
1. Analytic: det of claimed rotation; re-derive cot transform; derive observables
   distinguishing alternatives (range widths for swap; distribution asymmetry for sign).
2. Source geometry (2t-*.parquet scalar cols + part.93 scalar cols; stats already in
   column_stats.json from prior attempt):
   a. which of cotA/cotB (pix) is wide/narrow; compare vs CMSSW alpha(narrow)/beta(wide)
   b. pitch/midplane spans in part.93 (x-midplane vs y-midplane) -> which pix axis is long
   c. NN sigma/bias vs source angle: bending-plane (Lorentz) structure identifies alpha axis
3. CMSSW-side true-angle ranges: spec doc section 4 quantile tables; optionally
   payload_v4fixed_numEvent300.root analyzer ntuple. Swapped source ranges must match.
4. End-to-end pulls: nano sidecar (nano_fat_1111_coopt_file1.root or nano_pG*) pull
   branches vs |eta|/cotBeta; misassigned sigma axes => pulls blow up at high |eta|.
5. Counterfactual A/B: build WRONG (no-swap) payload in scratch, evaluate both over real
   CMSSW true-angle distribution, compare assigned sigma vs NN per-cluster sigma.
6. Description/bit-identity: deployed v4fixed JSON contains "x_cms=-y_pix" in all required
   descriptions; numeric content equals eval_spixel_angles/spx_angle_response_Conv1D_Full-2bit.json.

Verdict: CONFIRMED / REFUTED (with fix) / INCONCLUSIVE, per-leg strength, prediction-vs-measured table.
