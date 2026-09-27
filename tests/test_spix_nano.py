"""The SmartPixels nano compat reader on synthetic batches of each generation:
unified (one cluster table with geometry columns), split (barrel table plus
L1TSmartPixelsDiscCluster) and barrel-only."""

import awkward as ak
import numpy as np
import pytest

from ngtagger.data import spix_nano as S

CL, DC = S.CLUSTER_TABLE, S.OLD_DISC_TABLE


def _split():
    # event 0: barrel detIds 10,10,11 ; discs: disk 1 (x2, one module), disk 5, disk 9
    # event 1: barrel 12 ; discs: disk 4
    return ak.Array({
        f"{CL}_layer": [[1, 1, 2], [4]],
        f"{CL}_detId": [[10, 10, 11], [12]],
        f"{CL}_tpIdx": [[3, -1, 3], [7]],
        f"{DC}_disk": [[1, 1, 5, 9], [4]],
        f"{DC}_side": [[1, 1, 2, 2], [1]],
        f"{DC}_ring": [[2, 2, 1, 3], [4]],
        f"{DC}_detId": [[20, 20, 21, 22], [23]],
        f"{DC}_tpIdx": [[5, 5, -1, 8], [9]],
    })


def _unified(order_ok=True):
    sub = [[1, 1, 1, 2, 2], [1, 2]] if order_ok else [[1, 2, 1, 2, 2], [1, 2]]
    return ak.Array({
        f"{CL}_subdet": sub,
        f"{CL}_layer": [[1, 1, 2, 0, 0], [4, 0]],
        f"{CL}_disk": [[0, 0, 0, 1, 1], [0, 4]],
        f"{CL}_surface": [[1, 1, 2, 11, 11], [4, 14]],
        f"{CL}_detId": [[10, 10, 11, 20, 20], [12, 23]],
        f"{CL}_idxInDet": [[0, 1, 0, 0, 1], [0, 0]],
        f"{CL}_tpKey": [[3, -1, 3, 5, 5], [7, 9]],
    })


def test_layouts():
    assert S.cluster_layout(_split().fields) == "split"
    assert S.cluster_layout(_unified().fields) == "unified"
    assert S.cluster_layout([f"{CL}_layer", f"{CL}_tpIdx"]) == "barrel"


def test_aliases_both_ways():
    assert S.branch_name([f"{CL}_tpIdx"], CL, "tpKey") == f"{CL}_tpIdx"
    assert S.branch_name([f"{CL}_tpKey"], CL, "tpIdx") == f"{CL}_tpKey"
    assert S.branch_name(["L1TTP_idx"], "L1TTP", "key") == "L1TTP_idx"
    assert S.branch_name(["L1TTP_key", "L1TTP_idx"], "L1TTP", "idx") == "L1TTP_idx"
    assert S.branch_name(["X_spixMatchedTpIdx"], "X", "spixMatchedTpKey") == "X_spixMatchedTpIdx"
    # "key"/"idx" alias only on L1TTP
    assert S.branch_name(["X_idx"], "X", "key", required=False) is None
    with pytest.raises(KeyError):
        S.branch_name([f"{CL}_layer"], CL, "tpKey")


def test_split_becomes_unified_sp_scope():
    cols = ["detId", "tpKey"] + list(S.GEOMETRY)
    A = _split()
    assert set(S.cluster_branches(A.fields, cols)) <= set(A.fields)
    J = S.cluster_columns(A, cols)
    # disk 5 and 9 dropped by the default "sp" scope, barrel first
    assert J["detId"].tolist() == [[10, 10, 11, 20, 20], [12, 23]]
    assert J["tpKey"].tolist() == [[3, -1, 3, 5, 5], [7, 9]]
    assert J["subdet"].tolist() == [[1, 1, 1, 2, 2], [1, 2]]
    assert J["layer"].tolist() == [[1, 1, 2, 0, 0], [4, 0]]
    assert J["disk"].tolist() == [[0, 0, 0, 1, 1], [0, 4]]
    assert J["side"].tolist() == [[0, 0, 0, 1, 1], [0, 1]]
    assert J["ring"].tolist() == [[0, 0, 0, 2, 2], [0, 4]]
    assert J["surface"].tolist() == [[1, 1, 2, 11, 11], [4, 14]]
    assert J["idxInDet"].tolist() == [[0, 1, 0, 0, 1], [0, 0]]
    # the split synthesis equals what the unified producer writes
    U = S.cluster_columns(_unified(), ["detId", "tpKey", "subdet", "layer", "disk",
                                       "surface", "idxInDet"])
    for c in U:
        assert U[c].tolist() == J[c].tolist(), c


def test_scopes_and_barrel_prefix():
    A = _split()
    J = S.cluster_columns(A, ["surface"], disc_scope="all")
    assert J["surface"].tolist() == [[1, 1, 2, 11, 11, 15, 19], [4, 14]]
    J = S.cluster_columns(A, ["surface"], disc_scope="tfpx")
    assert J["surface"].tolist() == [[1, 1, 2, 11, 11, 15], [4, 14]]
    for B in (A, _unified()):
        J = S.cluster_columns(B, ["detId", "tpKey"], surfaces=S.BARREL)
        assert J["detId"].tolist() == [[10, 10, 11], [12]]
        J = S.cluster_columns(B, ["detId"], disc_scope="none")
        assert J["detId"].tolist() == [[10, 10, 11], [12]]
    # barrel-only request does not read the disc table
    assert not any(b.startswith(DC) for b in S.cluster_branches(A.fields, ["detId"],
                                                                 surfaces=S.BARREL))


def test_barrel_only_file():
    A = ak.Array({f"{CL}_layer": [[3, 1]], f"{CL}_tpIdx": [[1, 2]]})
    J = S.cluster_columns(A, ["tpKey", "surface", "subdet", "layer"])
    assert J["surface"].tolist() == [[3, 1]] and J["subdet"].tolist() == [[1, 1]]


def test_unified_barrel_must_be_prefix():
    with pytest.raises(AssertionError):
        S.cluster_columns(_unified(order_ok=False), ["detId"])


def test_flat_clusters():
    D, n = S.flat_clusters(_split(), ["detId"], surfaces=S.BARREL, event_offset=10)
    assert n.tolist() == [3, 1]
    assert D["event"].tolist() == [10, 10, 10, 11]
    assert isinstance(D["detId"], np.ndarray)


def test_angle_cap():
    # emu-mode flags (all 1): the cap drops |cot| > MEAS_ANGLE_MAX_ABS; a stored 0 stays 0
    cap = S.MEAS_ANGLE_MAX_ABS
    A = ak.Array({
        f"{CL}_subdet": [[1, 1, 1, 1]], f"{CL}_surface": [[1, 1, 2, 2]],
        f"{CL}_localCotAlpha": [[0.5, -cap - 0.1, cap, 3.0]],
        f"{CL}_localCotBeta": [[cap + 1, 2.0, -cap, -999.0]],
        f"{CL}_hasAlpha": np.array([[1, 1, 1, 0]], np.uint8),
        f"{CL}_hasBeta": np.array([[1, 1, 1, 0]], np.uint8),
    })
    br = S.cluster_branches(A.fields, ["hasAlpha", "hasBeta"])
    assert f"{CL}_localCotAlpha" in br and f"{CL}_localCotBeta" in br
    J = S.cluster_columns(A, ["hasAlpha", "hasBeta"])
    assert set(J) == {"hasAlpha", "hasBeta"}
    assert J["hasAlpha"].tolist() == [[1, 0, 1, 0]]
    assert J["hasBeta"].tolist() == [[0, 1, 1, 0]]
    assert str(ak.type(J["hasAlpha"])).endswith("uint8")
    R = S.cluster_columns(A, ["hasAlpha", "hasBeta"], cap_angles=False)
    assert R["hasAlpha"].tolist() == [[1, 1, 1, 0]]
    assert S.cluster_branches(A.fields, ["hasAlpha"], cap_angles=False) == [f"{CL}_hasAlpha", f"{CL}_subdet", f"{CL}_surface"]
