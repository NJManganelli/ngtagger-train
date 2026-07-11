"""On-demand hw -> float decoding on the track tables: values must match the
TTTrack_TrackWord get* conventions (two's complement + 0.5 offset times LSB;
bin lookups for the chi2/MVA fields). The raw hw fields stay untouched."""

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")
pytest.importorskip("coffea")


def _write(tmp_path):
    tree = {
        "run": np.array([1], dtype=np.uint32),
        "luminosityBlock": np.array([1], dtype=np.uint32),
        "event": np.array([1], dtype=np.uint64),
        "L1TTrack": ak.zip({
            # hwZ0: 0 -> +0.5 LSB; 2^11 -> most negative
            "hwZ0": ak.Array([[0, 1 << 11, (1 << 12) - 1]]),
            "hwTanl": ak.Array([[0, 1 << 15, 4096]]),
            "hwRinv": ak.Array([[0, 1 << 14, 1]]),
            "hwPhi": ak.Array([[0, 1 << 11, 5]]),
            "hwD0": ak.Array([[0, 1 << 12, 256]]),
            "hwBendChi2": ak.Array([[0, 5, 7]]),
            "hwChi2RPhi": ak.Array([[0, 10, 15]]),
            "hwChi2RZ": ak.Array([[1, 11, 15]]),
            "hwMVAQuality": ak.Array([[0, 3, 7]]),
            "pt": ak.Array([[10.0, 5.0, 2.0]]),
        }),
    }
    f = uproot.recreate(tmp_path / "tw.root")
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / "tw.root")


def test_trackword_decoding(tmp_path):
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1NanoSchema

    events = NanoEventsFactory.from_root({_write(tmp_path): "Events"},
                                         schemaclass=L1NanoSchema).events()
    trk = events.L1TTrack

    # raw hw fields untouched
    assert ak.to_list(trk.hwZ0[0]) == [0, 1 << 11, (1 << 12) - 1]

    # signed undigitization: (twos_complement + 0.5) * lsb
    step_z0 = 2.0 * 20.46912512 / (1 << 12)
    z0 = ak.to_list(trk.z0FromHw[0])
    assert z0[0] == pytest.approx(0.5 * step_z0)                       # bits 0
    assert z0[1] == pytest.approx((-(1 << 11) + 0.5) * step_z0)        # most negative
    assert z0[2] == pytest.approx(-0.5 * step_z0)                      # bits all-ones = -1

    step_tanl = 1.0 / (1 << 12)
    tanl = ak.to_list(trk.tanlFromHw[0])
    assert tanl[2] == pytest.approx((4096 + 0.5) * step_tanl)          # ~1.0

    # bin lookups
    assert ak.to_list(trk.bendChi2FromHw[0]) == [0.0, 3.5, 20.0]
    assert ak.to_list(trk.chi2RPhiFromHw[0]) == [0.0, 10.0, 200.0]
    assert ak.to_list(trk.chi2RZFromHw[0]) == [0.5, 6.0, 50.0]
    assert ak.to_list(trk.mvaQualityFromHw[0]) == [0.0, 0.375, 0.875]

    # double precision outputs
    assert "float64" in str(ak.type(trk.z0FromHw))
