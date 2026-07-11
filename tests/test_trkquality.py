import numpy as np
import pytest

from ngtagger.train.trkquality import nlaymiss_interior


def test_nlaymiss_interior():
    #                 0b1111111 -> no gaps      0b1000001 -> 5 interior misses
    hp = np.array([0b1111111, 0b1000001, 0b0000101, 0b0000001, 0])
    out = nlaymiss_interior(hp)
    assert out.tolist() == [0, 5, 1, 0, 0]


def test_trkquality_training(tmp_path):
    pytest.importorskip("xgboost")
    import awkward as ak

    from ngtagger.train.trkquality import build_trkq_dataset, train_trkquality

    # synthetic track table: genuine tracks have lower chi2 bins
    rng = np.random.default_rng(0)
    n = 4000
    genuine = rng.random(n) < 0.6
    rec = ak.Array([{
        "hwTanl": int(t), "hwZ0": int(z), "hwBendChi2": int(b),
        "hwChi2RPhi": int(cp), "hwChi2RZ": int(cz), "hitPattern": int(h),
        "nStubs": int(s), "genuine": bool(g), "looselyGenuine": bool(g),
        "unknown": False, "tpPt": 10.0 if g else -1.0, "tpFromHardInteraction": bool(g),
        "pt": 20.0, "eta": 0.5,
    } for t, z, b, cp, cz, h, s, g in zip(
        rng.integers(0, 2**10, n), rng.integers(0, 2**8, n),
        np.where(genuine, rng.integers(0, 3, n), rng.integers(3, 8, n)),
        np.where(genuine, rng.integers(0, 6, n), rng.integers(6, 16, n)),
        np.where(genuine, rng.integers(0, 6, n), rng.integers(6, 16, n)),
        rng.integers(1, 2**7, n), rng.integers(4, 7, n), genuine)])

    X, y, _ = build_trkq_dataset(ak.Array([rec]), label="genuine")
    assert X.shape == (n, 7)
    assert y.sum() == genuine.sum()

    # monkeypatch loader to skip file IO
    import ngtagger.train.trkquality as tq

    orig = tq.load_tracks
    tq.load_tracks = lambda *a, **k: ak.Array([rec])
    try:
        model, auc = train_trkquality(["dummy.root"], str(tmp_path), max_events=None)
    finally:
        tq.load_tracks = orig
    assert auc > 0.9  # separable by construction
    assert (tmp_path / "trkquality_xgb.json").exists()


def test_conifer_export(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("conifer")
    import awkward as ak

    import ngtagger.train.trkquality as tq

    rng = np.random.default_rng(1)
    n = 500
    genuine = rng.random(n) < 0.5
    rec = ak.Array([{
        "hwTanl": int(t), "hwZ0": 1, "hwBendChi2": int(b), "hwChi2RPhi": 1,
        "hwChi2RZ": 1, "hitPattern": 3, "nStubs": 4, "genuine": bool(g),
        "looselyGenuine": bool(g), "unknown": False, "tpPt": 1.0,
        "tpFromHardInteraction": bool(g), "pt": 20.0, "eta": 0.5,
    } for t, b, g in zip(rng.integers(0, 512, n),
                         np.where(genuine, 0, 5), genuine)])

    orig = tq.load_tracks
    tq.load_tracks = lambda *a, **k: ak.Array([rec])
    try:
        tq.train_trkquality(["dummy.root"], str(tmp_path))
    finally:
        tq.load_tracks = orig
    tq.export_conifer(str(tmp_path))
    assert (tmp_path / "trkquality_conifer.json").exists()
