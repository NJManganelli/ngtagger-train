"""E2E NNVtx tests: fastHisto reference, stock-vs-new comparison contract,
and a joint-training smoke test on separable synthetic vertexing data.

The synthetic nano file mimics L1PFTrkNanowithGen: L1TTrack tracks with
truth columns, GenVtx_z, and a stock L1Vertex whose z0 is produced by the
same fastHisto reference (standing in for l1tVertexFinderEmulator)."""

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

N_EVENTS = 200
N_PV_TRK, N_PU_TRK = 12, 30


def _make_events(seed=0):
    rng = np.random.default_rng(seed)
    pv_z = rng.uniform(-10, 10, N_EVENTS)
    evs = []
    for i in range(N_EVENTS):
        pv_z0 = rng.normal(pv_z[i], 0.05, N_PV_TRK)
        pu_z0 = rng.uniform(-15, 15, N_PU_TRK)
        z0 = np.concatenate([pv_z0, pu_z0])
        pt = np.concatenate([rng.uniform(3, 50, N_PV_TRK), rng.uniform(2, 8, N_PU_TRK)])
        mva = np.concatenate([rng.uniform(0.7, 1.0, N_PV_TRK), rng.uniform(0.0, 0.6, N_PU_TRK)])
        hard = np.concatenate([np.ones(N_PV_TRK, bool), np.zeros(N_PU_TRK, bool)])
        eta = rng.uniform(-2.4, 2.4, N_PV_TRK + N_PU_TRK)
        evs.append({"z0": z0, "pt": pt, "trkMVA1": mva, "eta": eta, "hard": hard})
    return pv_z, evs


def _write_nano(tmp_path, name="vtx.root", seed=0):
    from ngtagger.train.nnvtx import fast_histo_z0

    pv_z, evs = _make_events(seed)
    n_trk = N_PV_TRK + N_PU_TRK
    z0 = np.array([e["z0"] for e in evs])
    pt = np.array([e["pt"] for e in evs])
    stock_z0 = fast_histo_z0(z0, pt, np.ones_like(z0))

    tree = {
        "run": np.ones(N_EVENTS, dtype=np.uint32),
        "luminosityBlock": np.ones(N_EVENTS, dtype=np.uint32),
        "event": np.arange(N_EVENTS, dtype=np.uint64),
        "GenVtx_z": pv_z.astype(np.float64),
        "L1TTrack": ak.zip({
            "pt": [e["pt"] for e in evs],
            "eta": [e["eta"] for e in evs],
            "z0": [e["z0"] for e in evs],
            "trkMVA1": [e["trkMVA1"] for e in evs],
            "tpFromHardInteraction": [e["hard"] for e in evs],
            "genuine": [e["hard"] for e in evs],
            "hitPattern": [np.full(n_trk, 0b1111, dtype=np.int64)] * N_EVENTS,
            "chi2XYRed": [np.ones(n_trk)] * N_EVENTS,
            "chi2ZRed": [np.ones(n_trk)] * N_EVENTS,
            "chi2BendRed": [np.ones(n_trk)] * N_EVENTS,
            "nStubs": [np.full(n_trk, 4, dtype=np.int64)] * N_EVENTS,
        }),
        "L1Vertex": ak.zip({
            "z0": [[z] for z in stock_z0],
            "sumPt": [[float(p.sum())] for p in pt],
        }),
    }
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name), pv_z, stock_z0


def test_fast_histo_reference():
    pv_z, evs = _make_events(1)
    z0 = np.array([e["z0"] for e in evs])
    pt = np.array([e["pt"] for e in evs])
    from ngtagger.train.nnvtx import fast_histo_z0

    found = fast_histo_z0(z0, pt, np.ones_like(z0))
    res = found - pv_z
    assert np.median(np.abs(res)) < 0.2  # PV recovered by the baseline


def test_stock_vs_recomputed_comparison(tmp_path):
    from ngtagger.train.nnvtx import compare_vertex_scores, fast_histo_z0, load_vtx_data, to_padded

    path, pv_z, stock_written = _write_nano(tmp_path)
    feats, gen_z, stock_z0 = load_vtx_data([path])
    assert stock_z0 is not None
    np.testing.assert_allclose(stock_z0, stock_written, rtol=1e-6)

    z0, mask = to_padded(feats, ["z0"])
    pt, _ = to_padded(feats, ["pt"])
    recomputed = fast_histo_z0(z0[..., 0], pt[..., 0], mask)

    metrics = compare_vertex_scores(gen_z, stock_z0, recomputed)
    # recomputation of the same algorithm on the same tracks reproduces stock
    assert metrics["stock_vs_new_std"] < 1e-6
    assert metrics["stock_efficiency"] > 0.8
    assert "new_res_std" in metrics


def test_e2e_training_smoke(tmp_path):
    pytest.importorskip("keras")
    from ngtagger.train.nnvtx import (compare_vertex_scores, load_vtx_data,
                                      to_padded, train_nnvtx)

    path, pv_z, _ = _write_nano(tmp_path, "vtx_train.root", seed=2)
    feats, gen_z, stock_z0 = load_vtx_data([path], extra_features=["abs_eta"])
    max_tracks = 64
    X, mask = to_padded(feats, ["pt", "trkMVA1", "abs_eta"], max_tracks)
    z0_trk, _ = to_padded(feats, ["z0"], max_tracks)
    y_assoc, _ = to_padded(feats, ["tpFromHardInteraction"], max_tracks)
    dataset = (X, mask, z0_trk[..., 0], y_assoc[..., 0], gen_z, stock_z0)

    model, history = train_nnvtx([path], str(tmp_path / "out"), max_tracks=max_tracks,
                                 epochs=25, batch_size=32, dataset=dataset)
    z0_pred, assoc_pred = model.predict(
        {"trk_z0": z0_trk[..., 0], "trk_features": X, "trk_mask": mask}, verbose=0)

    metrics = compare_vertex_scores(gen_z, stock_z0, z0_pred[:, 0],
                                    y_assoc=y_assoc[..., 0], assoc_pred=assoc_pred, mask=mask)
    # separable-by-construction synthetic: association learnable in a few epochs
    assert metrics["assoc_auc"] > 0.8, metrics
    assert (tmp_path / "out" / "e2e_nnvtx.keras").exists()
