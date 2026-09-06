"""The per-track combination count means exactly what it looks like.

Pins the definition against worked examples: candidate counts per layer multiply,
and a layer with NO candidate contributes a factor of 1 rather than 0, because the
refit skips that layer and carries on. A zero would erase the track from the
distribution entirely and bias every summary toward busy tracks.

    2 x 1 x (0 -> 1) x 2 = 4
    3 x 2 x  1        x 1 = 6

This is the number the activeSP design table sums over tracks, so if this drifts
the whole cost side of that table drifts with it silently.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_OMNI = (pathlib.Path(__file__).resolve().parents[1]
         / "eval_refitq" / "combinatorics" / "spix_combinatorics_omnibus.py")


@pytest.fixture(scope="module")
def om():
    spec = importlib.util.spec_from_file_location("omni", _OMNI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(per_track_layer_counts, far=50.0):
    """Synthetic crossings + clusters realising the requested candidates per layer.

    Every crossing sits at projLocal (0,0) with unit projected sigma, so a cluster
    at radius < k counts and one parked at `far` does not. Layers asked for zero
    candidates still get a CROSSING (the layer is instrumented and was searched);
    they simply get no cluster inside the cone, which is the case that must yield a
    factor of one.
    """
    xs = {c: [] for c in ("event", "trackIdx", "layer", "detId",
                          "projLocalX", "projLocalY", "projSigX", "projSigY")}
    ks = {c: [] for c in ("event", "detId", "localX", "localY", "tpIdx")}
    det = 100
    for trk, counts in enumerate(per_track_layer_counts):
        for lay, n in zip((1, 2, 3, 4), counts):
            det += 1
            xs["event"].append(0); xs["trackIdx"].append(trk)
            xs["layer"].append(lay); xs["detId"].append(det)
            xs["projLocalX"].append(0.0); xs["projLocalY"].append(0.0)
            xs["projSigX"].append(1.0); xs["projSigY"].append(1.0)
            # n clusters inside the cone, plus one deliberately outside
            for i in range(n):
                ks["event"].append(0); ks["detId"].append(det)
                ks["localX"].append(0.01 * i); ks["localY"].append(0.01 * i)
                ks["tpIdx"].append(-1)
            ks["event"].append(0); ks["detId"].append(det)
            ks["localX"].append(far); ks["localY"].append(far); ks["tpIdx"].append(-1)
    X = {k: np.asarray(v) for k, v in xs.items()}
    X["trk_tpIdx"] = np.full(len(X["layer"]), -1)
    K = {k: np.asarray(v) for k, v in ks.items()}
    K["_evt_base"] = np.array([0, len(K["detId"])])
    return X, K


@pytest.mark.parametrize("counts,expected", [
    ((2, 1, 0, 2), 4),      # the empty layer must contribute 1, not 0
    ((3, 2, 1, 1), 6),
    ((1, 1, 1, 1), 1),
    ((0, 0, 0, 0), 1),      # nothing found anywhere is still one (empty) trial
    ((2, 2, 2, 2), 16),
])
def test_per_track_combinations(om, counts, expected):
    X, K = _build([counts])
    combos, ntrk, _, _ = om._combinations_per_track(X, K, [2.0])
    assert ntrk == 1
    assert combos[2.0][0] == pytest.approx(float(expected))


def test_total_is_the_sum_over_tracks(om):
    """The design table's TOTAL is a plain sum of the per-track counts."""
    X, K = _build([(2, 1, 0, 2), (3, 2, 1, 1)])
    combos, ntrk, _, _ = om._combinations_per_track(X, K, [2.0])
    v = combos[2.0]
    assert ntrk == 2
    assert sorted(v.tolist()) == [4.0, 6.0]
    assert v.sum() == pytest.approx(10.0)


def test_widening_the_cone_cannot_reduce_candidates(om):
    """Monotonicity: a wider cone is a superset, so work never falls."""
    X, K = _build([(2, 1, 0, 2)], far=3.0)
    combos, _, _, _ = om._combinations_per_track(X, K, [1.0, 2.0, 4.0])
    tot = [combos[k].sum() for k in (1.0, 2.0, 4.0)]
    assert tot[0] <= tot[1] <= tot[2]
    assert tot[2] > tot[0]          # the far cluster enters at k=4
