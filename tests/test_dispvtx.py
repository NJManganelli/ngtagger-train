"""Displaced-vertex tagger tests: synthetic L1TrkNano with a L1DispVertex
table carrying a 'stock' score (monotone in the discriminating features,
standing in for the deployed conifer GBDT), retraining, and stock-vs-new
comparison. Also exercises the schema crossrefs for the track pair."""

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

N_EVENTS = 150
N_VTX = 8


def _make(tmp_path, name="dv.root", seed=0):
    rng = np.random.default_rng(seed)
    n_trk = 2 * N_VTX

    vtx, trk = [], []
    for _ in range(N_EVENTS):
        real = rng.random(N_VTX) < 0.4
        # real vertices: consistent pair (small del_Z, high cos_T), fakes: broad
        d_T = np.where(real, rng.uniform(1, 10, N_VTX), rng.uniform(0, 3, N_VTX))
        R_T = np.where(real, rng.uniform(1, 15, N_VTX), rng.uniform(0.02, 20, N_VTX))
        cos_T = np.where(real, rng.uniform(0.95, 1.0, N_VTX), rng.uniform(-1, 1, N_VTX))
        del_Z = np.where(real, abs(rng.normal(0, 0.05, N_VTX)), abs(rng.normal(0, 1.0, N_VTX)))
        mva = np.where(real, rng.uniform(0.6, 1.0, N_VTX), rng.uniform(0, 0.7, N_VTX))
        # 'stock' score: monotone function of the discriminating quantities
        stock = 1 / (1 + np.exp(-(3 * cos_T - 4 * del_Z + 2 * mva - 1.5)))
        pt = rng.uniform(3, 60, n_trk)
        trk.append({
            "pt": pt, "eta": rng.uniform(-2.4, 2.4, n_trk), "phi": rng.uniform(-np.pi, np.pi, n_trk),
            "d0": rng.normal(0, 1, n_trk), "z0": rng.uniform(-15, 15, n_trk),
            "chi2ZRed": rng.uniform(0, 3, n_trk), "chi2BendRed": rng.uniform(0, 4, n_trk),
            "trkMVA1": np.repeat(mva, 2),
        })
        vtx.append({
            "score": stock, "isReal": real, "d_T": d_T, "R_T": R_T, "cos_T": cos_T,
            "del_Z": del_Z, "x": rng.normal(0, 2, N_VTX), "y": rng.normal(0, 2, N_VTX),
            "z": rng.uniform(-15, 15, N_VTX), "openingAngle": rng.uniform(0, 1, N_VTX),
            "parentPt": rng.uniform(5, 100, N_VTX),
            "firstIndexTrk": np.arange(0, n_trk, 2), "secondIndexTrk": np.arange(1, n_trk, 2),
            "inTraj": np.zeros(N_VTX, dtype=np.int64),
        })

    tree = {
        "run": np.ones(N_EVENTS, dtype=np.uint32),
        "luminosityBlock": np.ones(N_EVENTS, dtype=np.uint32),
        "event": np.arange(N_EVENTS, dtype=np.uint64),
        "L1TExtTrack": ak.zip({k: [t[k] for t in trk] for k in trk[0]}),
        "L1DispVertex": ak.zip({k: [v[k] for v in vtx] for k in vtx[0]}),
    }
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name)


def test_dataset_and_stock_score(tmp_path):
    from ngtagger.train.dispvtx import FEATURE_NAMES, build_dispvtx_dataset, load_dispvtx_data

    path = _make(tmp_path)
    vertices, tracks = load_dispvtx_data([path])
    X, y, stock = build_dispvtx_dataset(vertices, tracks)
    assert X.shape == (N_EVENTS * N_VTX, len(FEATURE_NAMES)) == (N_EVENTS * N_VTX, 20)
    # first track has even index -> its pt is column 0; second track column 1
    assert not np.allclose(X[:, 0], X[:, 1])
    # trkMVA1 identical for pair by construction: columns 14, 15
    np.testing.assert_allclose(X[:, 14], X[:, 15], rtol=1e-6)
    assert 0 < y.mean() < 1


def test_retrain_and_compare(tmp_path):
    pytest.importorskip("xgboost")
    from ngtagger.train.dispvtx import train_dispvtx

    path = _make(tmp_path, "dv_train.root", seed=1)
    model, metrics = train_dispvtx([path], str(tmp_path / "out"))
    # stock score is monotone in the truth-separating features
    assert metrics["stock_auc"] > 0.8
    # retrained model must at least match the stock separation on this toy
    assert metrics["new_auc"] >= metrics["stock_auc"] - 0.02, metrics
    assert -1 <= metrics["rank_correlation"] <= 1
    assert (tmp_path / "out" / "dispvtx_xgb.json").exists()


def test_conifer_export(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("conifer")
    from ngtagger.train.dispvtx import export_conifer, train_dispvtx

    path = _make(tmp_path, "dv_c.root", seed=2)
    train_dispvtx([path], str(tmp_path / "outc"))
    export_conifer(str(tmp_path / "outc"))
    assert (tmp_path / "outc" / "dispvtx_conifer.json").exists()


def test_schema_track_pair_crossrefs(tmp_path):
    pytest.importorskip("coffea")
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1NanoSchema

    path = _make(tmp_path, "dv_schema.root", seed=3)
    events = NanoEventsFactory.from_root({path: "Events"}, schemaclass=L1NanoSchema).events()
    dv = events.L1DispVertex
    assert ak.all(dv.first_track.pt[0] == events.L1TExtTrack.pt[0][::2])
    assert ak.all(dv.second_track.pt[0] == events.L1TExtTrack.pt[0][1::2])
