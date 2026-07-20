"""Smoke tests for the refit-quality BDT study on synthetic, row-aligned
SmartPixels digiRefit tables (mirrors test_trkquality.py conventions).

Builds a small synthetic sample where genuine tracks have cleaner refit info
(smaller pulls, smaller kicks, more accepted hits) so every tier is separable
by construction; checks the tier feature counts, the 1:1 alignment handling of
the per-hit trackIdx aggregation, and that AUC improves with richer tiers.
"""
import numpy as np
import pytest


def _synth(n_events=60, tracks_per_event=30, seed=0):
    """Synthetic reference + one variant track table + per-hit link table,
    row-aligned, returned as three awkward arrays (ref, var, hits)."""
    import awkward as ak

    rng = np.random.default_rng(seed)
    ref_ev, var_ev, hit_ev = [], [], []
    for _ in range(n_events):
        ntr = tracks_per_event
        genuine = rng.random(ntr) < 0.75
        ref_rows, var_rows, hit_rows = [], [], []
        for it in range(ntr):
            g = bool(genuine[it])
            ref_rows.append({
                "hwTanl": int(rng.integers(0, 2**12)), "hwZ0": int(rng.integers(0, 2**10)),
                "hwBendChi2": int(rng.integers(0, 3) if g else rng.integers(3, 8)),
                "hwChi2RPhi": int(rng.integers(0, 6) if g else rng.integers(6, 16)),
                "hwChi2RZ": int(rng.integers(0, 6) if g else rng.integers(6, 16)),
                "hitPattern": int(rng.integers(1, 2**7)), "nStubs": int(rng.integers(4, 7)),
                "hwRinv": 0, "hwPhi": 0, "hwD0": 0,
                "rInv": 0.01, "phi": 0.1, "tanL": 0.5, "z0": 1.0, "d0": 0.0,
                "pt": 20.0, "eta": 0.5,
                "trkMVA1": float(rng.random() * (0.5 if not g else 1.0)),
                "genuine": g, "looselyGenuine": g, "combinatoric": (not g),
                "unknown": False, "tpPt": 10.0 if g else -1.0, "tpFromHardInteraction": g,
            })
            # cleaner refit for genuine tracks
            kick = (0.002 if g else 0.05)
            var_rows.append({
                "spxRefitPerformed": True, "spxSeedCovOK": True,
                "spxNCrossings": 2, "spxNAcceptedHits": 2,
                "spxLayerHitMask": 3, "spxMaxWindowMult": int(rng.integers(2, 6)),
                "spxAnyWindowTruncated": (not g), "spxNKFUpdates": 2,
                "spxChi2IncRPhiTot": float(rng.random() * (5 if g else 500)),
                "spxChi2IncRZTot": float(rng.random() * (5 if g else 500)),
                "rInv": 0.01 + rng.normal(0, kick), "phi": 0.1 + rng.normal(0, kick),
                "tanL": 0.5 + rng.normal(0, kick), "z0": 1.0 + rng.normal(0, kick),
                "d0": rng.normal(0, kick),
            })
            # two hits per track
            for lyr in (1, 2):
                pull = rng.normal(0, 0.5 if g else 3.0)
                hasang = rng.random() < (0.9 if g else 0.4)
                hit_rows.append({
                    "trackIdx": it, "layer": lyr, "windowMult": int(rng.integers(1, 6)),
                    "windowTruncated": (not g), "hasAlpha": hasang, "hasBeta": hasang,
                    "resX": rng.normal(0, 0.01), "resY": rng.normal(0, 0.05),
                    "pullX": pull, "pullY": pull,
                    "pullAlpha": pull if hasang else -999.0,
                    "pullBeta": pull if hasang else -999.0,
                    "sigAlpha": 0.02 if hasang else -999.0,
                    "sigBeta": 0.02 if hasang else -999.0,
                    "chi2IncRPhi": float(rng.random() * (2 if g else 50)),
                    "chi2IncRZ": float(rng.random() * (2 if g else 50)),
                })
        ref_ev.append(ref_rows)
        var_ev.append(var_rows)
        hit_ev.append(hit_rows)
    return ak.Array(ref_ev), ak.Array(var_ev), ak.Array(hit_ev)


def test_activesp_and_tier_tables():
    from ngtagger.train.refitquality import CONFIG_ACTIVESP, SMARTPIXELS_CONFIGS, TIERS

    assert SMARTPIXELS_CONFIGS == ("AIII", "AAII", "AAAI", "AAAA")
    assert CONFIG_ACTIVESP == {"AIII": "1000", "AAII": "1100", "AAAI": "1110", "AAAA": "1111"}
    assert TIERS == ("A", "B", "C", "D")


def test_dataset_shapes_and_alignment():
    pytest.importorskip("awkward")
    from ngtagger.train.refitquality import build_refitq_dataset

    ref, var, hits = _synth()
    ntr = 60 * 30
    xa, ya, na, ia = build_refitq_dataset(ref, var, hits, "A", "AAAA")
    assert xa.shape == (ntr, 7)
    assert len(na) == 7
    assert ia["n_tracks"] == ntr

    # tiers add features monotonically; B/C/D share the same labels + row count
    dims = {}
    for tier in ("B", "C", "D"):
        x, y, n, i = build_refitq_dataset(ref, var, hits, tier, "AAAA")
        assert x.shape[0] == ntr
        assert (y == ya).all()  # identical labels across tiers (1:1 alignment)
        dims[tier] = x.shape[1]
    assert dims["B"] < dims["C"] < dims["D"]
    # no NaN/inf leaked from sentinels or chi2 tails
    xd, *_ = build_refitq_dataset(ref, var, hits, "D", "AAAA")
    assert np.isfinite(xd).all()


def test_hit_aggregation_respects_trackidx_offset():
    """The per-hit trackIdx is per-event; the global offset must land each
    hit's pulls on the correct global track row (no cross-event bleed)."""
    from ngtagger.train.refitquality import build_refitq_dataset

    ref, var, hits = _synth(n_events=5, tracks_per_event=10, seed=3)
    x, y, names, info = build_refitq_dataset(ref, var, hits, "B", "AAAA")
    nh = x[:, names.index("hit_nhits")]
    assert (nh == 2).all()  # two hits per track by construction, correctly scattered


def test_train_one_smoke(tmp_path):
    pytest.importorskip("xgboost")
    from ngtagger.train.refitquality import train_one

    ref, var, hits = _synth()
    aucs = {}
    for tier in ("A", "B", "C", "D"):
        _, auc, meta = train_one(ref, var, hits, tier, "AAAA", str(tmp_path),
                                 log_mlflow=False, seed=0)
        aucs[tier] = auc
        tag = "A" if tier == "A" else f"{tier}-AAAA"
        assert (tmp_path / f"refitq_{tag}_xgb.json").exists()
        assert (tmp_path / f"refitq_{tag}_meta.json").exists()
    # separable by construction; richer tiers should not be worse than baseline
    assert aucs["A"] > 0.75
    assert aucs["D"] >= aucs["A"] - 0.05


def test_spec_dataset_v0_v1_shapes_and_decode():
    """The SPEC-ORDER dataset builder yields 17 features for v0 and 24 for v1,
    and the v1 tail (indices 17-23) must be the classic-7 TRKQ_FEATURES decoded
    EXACTLY as trkquality does (bit-for-bit the producer's in-flight v1 vector)."""
    pytest.importorskip("awkward")
    import awkward as ak
    from ngtagger.train.refitquality import (
        build_spec_dataset, REFIT_SPEC_FEATURES, REFIT_SPEC_FEATURES_V1)
    from ngtagger.train.trkquality import (
        TRKQ_FEATURES, K_TANL_SIZE, K_Z0_SIZE, twos_complement, nlaymiss_interior)

    ref, var, hits = _synth(n_events=6, tracks_per_event=10, seed=7)

    x0, y0, n0, aux0 = build_spec_dataset(ref, var, hits, "AAAA", refit_only=False,
                                          spec_version=0)
    assert x0.shape[1] == 17 and n0 == list(REFIT_SPEC_FEATURES)

    x1, y1, n1, aux1 = build_spec_dataset(ref, var, hits, "AAAA", refit_only=False,
                                          spec_version=1)
    assert x1.shape[1] == 24 and n1 == list(REFIT_SPEC_FEATURES_V1)
    assert n1[17:] == list(TRKQ_FEATURES)
    # v1's first 17 columns are byte-identical to the v0 vector (append-only).
    assert np.array_equal(x1[:, :17], x0)
    assert (y1 == y0).all()

    # v1 tail decode matches trkquality's on the flattened reference hw columns.
    ref_flat = {b: ak.to_numpy(ak.flatten(ref[b]))
                for b in ("hwTanl", "hwZ0", "hwBendChi2", "hwChi2RPhi", "hwChi2RZ",
                          "hitPattern", "nStubs")}
    expected = np.stack([
        twos_complement(ref_flat["hwTanl"], K_TANL_SIZE).astype(np.float32),
        twos_complement(ref_flat["hwZ0"], K_Z0_SIZE).astype(np.float32),
        ref_flat["hwBendChi2"].astype(np.float32),
        ref_flat["nStubs"].astype(np.float32),
        nlaymiss_interior(ref_flat["hitPattern"]).astype(np.float32),
        ref_flat["hwChi2RPhi"].astype(np.float32),
        ref_flat["hwChi2RZ"].astype(np.float32),
    ], axis=1)
    assert np.array_equal(x1[:, 17:], expected)


def test_spec_dataset_rejects_bad_version():
    pytest.importorskip("awkward")
    from ngtagger.train.refitquality import build_spec_dataset
    ref, var, hits = _synth(n_events=3, tracks_per_event=6, seed=2)
    with pytest.raises(ValueError):
        build_spec_dataset(ref, var, hits, "AAAA", refit_only=False, spec_version=2)


def test_require_truth_fails_loudly():
    """Truth-required mode must throw, not silently degrade, when no positives."""
    import awkward as ak

    from ngtagger.train.refitquality import build_refitq_dataset

    ref, var, hits = _synth(n_events=4, tracks_per_event=8, seed=1)
    # force all-fake labels
    flds = {f: ref[f] for f in ref.fields}
    flds["genuine"] = ak.zeros_like(ref["genuine"])
    ref0 = ak.zip(flds, depth_limit=2)
    with pytest.raises(RuntimeError):
        build_refitq_dataset(ref0, var, hits, "A", "AAAA", label="genuine")
