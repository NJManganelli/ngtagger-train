# Definition sheet v2.8 (2026-10-14)

One line per definition; the full text, parameters and code pointers are in `sheet_v2.8.yaml` (`python -m ngtagger.definitions render-full`). Generated: edit the YAML, never this file. Amendments get a new version; every registered result names the version it used.

Sources: plan.md round 7 (2026-10-06): App. B.1, B.4, C.2, D.1, D.3, D.9, F, J.3; report.md decisions of 2026-10-08/09 (they override the plan where they differ); WP-D0a (2026-10-09): decision 12 implemented in the census, format 28; WP-D0a review 01 (2026-10-13; WDMac wpd0a/review-01.md) and the CMSSW truth spec (wpd0a/cmssw_truth_spec.md): v2.1 wording; WP-D0a review 02 (2026-10-13; WDMac wpd0a/review-02.md): v2.2, the position qualifier; decision of 2026-10-13 (the census denominator) and the out-of-time truth investigation (WDMac wpd0a/oot_truth_investigation.md): v2.3; decision of 2026-10-14 (one hit per module in the census denominator): v2.4; decisions of 2026-10-14 (P0.3 amendment 1d: quantile rank, median centre, width = q95 / 1.96): v2.5; decision 12 'as CMSSW' (ties: CMSSW truth spec sec. 4, WDMac wpd0a/cmssw_truth_spec.md), applied 2026-10-14: v2.6; physicist's approval of 2026-10-14 of statistics review 02 round 4 (P0.3 amendment 1e: trimmed-mean centre and width): v2.7; P0.3 amendment 2 (module-frame disc rows; reviews stats-review-a2-01, geom-review-a2-01; the physicist's decision to pool the facings, 2026-10-14) and census format 33 (297852e): v2.8.

## Scenarios, samples, surfaces

- **Scenario masks** (`scenario.masks`): A/I per inner layer IL1..IL4 (and disc IDk); IIII = outer tracker only (passthrough).
- **Track parameters** (`scenario.track_parameters`): 5-parameter tracks in every arm; no 4-parameter IIII.
- **Production eras** (`scenario.eras`): canon_v2 and canon_v3; never mix their jet labels; validation on canon_v3 TTbar files 1-10.
- **Samples and the in-sample files** (`scenario.samples`): TTbar files 1-2 are cluster-CNN training data: train-only for tagging, in-sample for angles.
- **Surfaces and their codes** (`scenario.surfaces`): IL1-IL4, ID1-ID4, OL1-OL6; the nano and the census number surfaces differently (11-14 discs vs 11-16 OT).

## Truth

- **Cluster truth** (`truth.cluster_tp`): Dominant-charge TP (its own hits or its unstored daughters'); -1/-2 none; <= -3 a TP not in L1TTP.
- **Class T and class C** (`truth.class_T`): Class T = TP helix within 1 cm (truth + 1.5x PixelAV residual); class C = the rest; truth columns never inputs.
- **Shared cluster** (`truth.shared_cluster`) *[provisional: the threshold is fixed by the refit-v2 audit (P0.2) as amendment v1.1; until then every result using 'shared' states its threshold]*: Dominant TP below a charge fraction; threshold set by the refit-v2 audit (provisional).
- **OT stub truth (census)** (`truth.stub`): A stub counts for its principal TP and each cluster's dominant TP, one hit per TP; every credited hit counts the same; join by row index only.
- **OT track truth** (`truth.ot_track`): CMSSW TTTrackAssociator categories.
- **SmartPixels track truth** (`truth.sp_track`): SP track truth = its OT track's TP; hits correct / wrong-T / wrong-C / shared.
- **Census owner** (`truth.census_owner`): Majority owner in int64, each hit voting once per TP it is credited to; unmatched never wins; a tie gives no owner.
- **Seed-truthful (census)** (`truth.seed_truthful`): A seed finds a TP when every seed hit is credited to it.
- **Delivery (census) -- two definitions, always both** (`truth.delivery`): Delivered = majority owner (> half, >= 3 hits); clean = no wrong IT hit; always both (+ no wrong hit at all).
- **Census fake** (`truth.census_fake`): DR survivor with >= 2 wrong hits; quoted at fixed clean efficiency after an angle-free quality cut.
- **TP genealogy** (`truth.genealogy`) *[adopted (the IL2 radius is a v1 choice; amend if a package needs another)]*: b/c daughter, hard scatter vs pile-up, born beyond IL1 (r > 2.5 cm) / IL2 (r > 5.65 cm); |d0| bins are separate.
- **Jet labels** (`truth.tagger_labels`): 8 classes by gen matching (dR < 0.4); kept = labelled and gen pT > 5 GeV; udsg = light + gluon.
- **Jet population and split** (`truth.tagger_population`) *[adopted (the tie-breaking rule of the across-arm match is set by the stock-tagger package P0.5a)]*: pT > 15 GeV, |eta| < 2.4; event-identity split, files 1-2 train-only; the across-arm jet key needs a tie-break.

## Frames and conventions

- **d0 sign** (`frame.d0`): d0 = x0 sin phi0 - y0 cos phi0 = -dxy at the POCA to the beam line.
- **phi0 and the helix** (`frame.phi0`): phi0 at the POCA (tp_phi is at production); pT = KPT / |rInv|, KPT = 0.29979246 x 3.8112 / 100.
- **Module orientation and the local-to-global sign** (`frame.module_orientation`): The global sense is outward; cotBeta's local-to-global sign follows each module's orientation.
- **Angle columns** (`frame.angles`): Emitted local cotAlpha/cotBeta; usable if has-flag and |cot| <= 12; has-flags are uint8 (cast to bool).
- **Incidence key of angle tables** (`frame.incidence_key`): Signed cot theta (barrel), signed 1/cot theta per module facing (discs); each consumer in its own key.
- **PF/PUPPI candidate dxy** (`frame.puppi_dxy`): Read hwDxy through the code table, never the float as cm; code 3 = d0 in [-39, +78) um.

## Angles

- **Improved-angle knob k and its two uncertainty modes** (`angle.k_knob`): angle' = truth + k(angle - truth) on class T, k <= 1; fixed and scaled (floor 5%) uncertainties, both reported.
- **Angle-uncertainty table** (`angle.error_table`): Per surface x signed incidence bin: 25%-trimmed-mean centre, 95%-trimmed-mean width, 95-99.7% quantiles (rank ceil(L(n+1))); fit 1-2, test 3-10.
- **Angle-uncertainty table, module-frame disc rows** (`angle.error_table_disc_local`): Discs ID1-ID4 x signed local-incidence bin (facings pooled): the angle.error_table recipe on module-frame cot pulls with the local sigma.
- **Census angle cuts** (`angle.cut_level`): Each census angle cut keeps 99% of correct particles in its own statistic; S6 per number of angle measurements.
- **Disc angles** (`angle.disc_default`): Disc angles on by default, labelled unvalidated until the disc calibration.

## Census conventions

- **OT-truncated region** (`census.ot_truncated`): |eta| 0.94-1.4 labelled OT truncated; headlines also at |eta| < 0.9.
- **Findable TP (the census efficiency denominator)** (`census.findable`): Charged TP with credited hits on >= 4 distinct modules, IT plus OT in any mix; one hit per module, each stub one hit.
- **Builds and duplicate removal** (`census.builds`): Each build follows only into its own surfaces; duplicate removal within a build.
- **Census stages** (`census.stages`): S1-S8; each switch changes only its own stage; the no-angle arm is position-complete.

## Figures of merit

- **Common absolute thresholds** (`fom.thresholds`): |dd0| 0.1/0.2/0.5/1/5 mm, |dz0| 0.5/1/5 mm, |dpT/pT| 0.2/0.5, same for every arm.
- **Harm** (`fom.harm`): SP worse than OT by > 0.5 mm in d0 or > 1 mm in z0.
- **First-visited surface** (`fom.first_visited_surface`): First IT surface of an outside-in refit (IL4; IL3 in AAAI).
- **Attachment** (`fom.attachment`) *[adopted (the denominator is a v1 choice)]*: Correct hit at a surface among tracks whose TP has a cluster there.
- **Resolution reporting** (`fom.resolution`): Core, fraction outside, full RMS, thresholds, harm; |eta| x pT x origin bins.
- **Matched working points** (`fom.matched_wp`): Exactly matched background with tie splitting, in every arm and replica; realised fraction reported.
- **Vertexing** (`fom.vertexing`): Correct vertex = nearest gen vertex is the hard scatter; common |dz| windows.

## Estimator and statistics

- **The one width estimator (B.1 core)** (`estimator.core`): RMS about zero of a mode-seeded iterative 3-sigma clip; unit-free (no magnitude cuts).
- **Sentinels and validity** (`estimator.sentinels`): Remove sentinels with explicit validity masks (tpIdx >= 0, has-flags), never by magnitude.
- **Errors** (`stats.bootstrap`): Event-level paired bootstrap, >= 250 replicas, every fitted step re-fitted.
- **Many bins** (`stats.family_wise`): Per-bin threshold t(N) (Sidak, 1%) and chi2 p > 0.01 / number of tests.
- **No change and saturation** (`stats.equivalence`): No change only by an equivalence test with a pre-set margin and measured power >= 80%.
- **Gates** (`stats.gates`): Tolerance from a named known-good; shown to fail on a named known-bad first.

## Tagger

- **Training recipe** (`tagger.recipe`): Deployed architecture, linear outputs, fixed epoch count past saturation of both arms, no early stopping.
- **The deployed tagger on SmartPixels tracks** (`tagger.deployed_comparison`): The deployed tagger (4-parameter training) on SP tracks is not used; retrained taggers decide.

## Result statuses

- **Status vocabulary** (`registry.statuses`): valid / void / pathology-affected / superseded / refit-v2 / payload-era / in-sample / exploratory / not-used.
