"""The SmartPixels nano compat reader on synthetic batches of each generation:
unified (one cluster table with geometry columns), split (barrel table plus
L1TSmartPixelsDiscCluster) and barrel-only."""

import awkward as ak
import numpy as np
import pytest

from ngtagger.data import spix_nano as S

CL, DC = S.CLUSTER_TABLE, S.OLD_DISC_TABLE


# cluster tpIdx (rows into L1TTP; <= -3 pruned) of the unified/split files below
CLUSTER_TP_IDX = [[0, -1, 0, -8, -8], [-10, 0]]


def _split():
    # event 0: barrel detIds 10,10,11 ; discs: disk 1 (x2, one module), disk 5, disk 9
    # event 1: barrel 12 ; discs: disk 4
    return ak.Array({
        f"{CL}_layer": [[1, 1, 2], [4]],
        f"{CL}_detId": [[10, 10, 11], [12]],
        f"{CL}_tpIdx": [[0, -1, 0], [-10]],
        f"{DC}_disk": [[1, 1, 5, 9], [4]],
        f"{DC}_side": [[1, 1, 2, 2], [1]],
        f"{DC}_ring": [[2, 2, 1, 3], [4]],
        f"{DC}_detId": [[20, 20, 21, 22], [23]],
        f"{DC}_tpIdx": [[-8, -8, -1, 1], [0]],
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
        f"{CL}_tpIdx": CLUSTER_TP_IDX,
    })


def test_layouts():
    assert S.cluster_layout(_split().fields) == "split"
    assert S.cluster_layout(_unified().fields) == "unified"
    assert S.cluster_layout([f"{CL}_layer", f"{CL}_tpIdx"]) == "barrel"


def test_branch_name():
    keys = [f"{CL}_tpIdx", "nL1TTP"]
    assert S.branch_name(keys, CL, "tpIdx") == f"{CL}_tpIdx"
    assert S.branch_name(keys, CL, "tpKey", required=False) is None
    with pytest.raises(KeyError):
        S.branch_name(keys, CL, "layer")


def test_tp_links_read_as_columns():
    A = ak.Array({"nL1TTP": [2, 1], f"{CL}_tpIdx": CLUSTER_TP_IDX, "Y_tpIdx": [[1, -2], [-10]]})
    assert S.column(A, CL, "tpIdx").tolist() == CLUSTER_TP_IDX
    assert S.table_columns(A, "Y", ["tpIdx"])["tpIdx"].tolist() == [[1, -2], [-10]]
    assert S.branch_names(A.fields, CL, ["tpIdx"]) == [f"{CL}_tpIdx"]
    assert S.has_column(A.fields, CL, "tpIdx")
    assert not S.has_column(A.fields, "Z", "tpIdx")


def test_split_becomes_unified_sp_scope():
    cols = ["detId", "tpIdx"] + list(S.GEOMETRY)
    A = _split()
    assert set(S.cluster_branches(A.fields, cols)) <= set(A.fields)
    J = S.cluster_columns(A, cols)
    # disk 5 and 9 dropped by the default "sp" scope, barrel first
    assert J["detId"].tolist() == [[10, 10, 11, 20, 20], [12, 23]]
    assert J["tpIdx"].tolist() == CLUSTER_TP_IDX
    assert J["subdet"].tolist() == [[1, 1, 1, 2, 2], [1, 2]]
    assert J["layer"].tolist() == [[1, 1, 2, 0, 0], [4, 0]]
    assert J["disk"].tolist() == [[0, 0, 0, 1, 1], [0, 4]]
    assert J["side"].tolist() == [[0, 0, 0, 1, 1], [0, 1]]
    assert J["ring"].tolist() == [[0, 0, 0, 2, 2], [0, 4]]
    assert J["surface"].tolist() == [[1, 1, 2, 11, 11], [4, 14]]
    assert J["idxInDet"].tolist() == [[0, 1, 0, 0, 1], [0, 0]]
    # the split synthesis equals what the unified producer writes
    U = S.cluster_columns(_unified(), ["detId", "tpIdx", "subdet", "layer", "disk",
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
        J = S.cluster_columns(B, ["detId", "tpIdx"], surfaces=S.BARREL)
        assert J["detId"].tolist() == [[10, 10, 11], [12]]
        assert J["tpIdx"].tolist() == [[0, -1, 0], [-10]]
        J = S.cluster_columns(B, ["detId"], disc_scope="none")
        assert J["detId"].tolist() == [[10, 10, 11], [12]]
    # barrel-only request does not read the disc table
    assert not any(b.startswith(DC) for b in S.cluster_branches(A.fields, ["detId"],
                                                                 surfaces=S.BARREL))


def test_barrel_only_file():
    A = ak.Array({f"{CL}_layer": [[3, 1]], f"{CL}_tpIdx": [[1, -1]]})
    J = S.cluster_columns(A, ["tpIdx", "surface", "subdet", "layer"])
    assert J["surface"].tolist() == [[3, 1]] and J["subdet"].tolist() == [[1, 1]]
    assert J["tpIdx"].tolist() == [[1, -1]]


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


def test_direction_usable():
    cap = S.MEAS_ANGLE_MAX_ABS
    c = {"localCotAlpha": np.array([0.5, cap + 1, 0.5, -999.0]),
         "localCotBeta": np.array([cap, 0.5, 0.5, 0.5]),
         "hasAlpha": np.array([1, 1, 0, 1]), "hasBeta": np.array([1, 1, 1, 1])}
    assert S.direction_usable(c).tolist() == [True, False, False, False]


def test_angle_cap_census():
    cap = S.MEAS_ANGLE_MAX_ABS
    c = {"localCotAlpha": np.array([0.5, cap + 1, -999.0, 2.0, -cap - 3]),
         "localCotBeta": np.array([1.0, 0.5, 0.5, cap + 2, 0.1]),
         "hasAlpha": np.array([1, 1, 1, 1, 0]), "hasBeta": np.array([1, 1, 1, 1, 1])}
    surf = np.array([1, 1, 11, 11, 11])
    key = np.array([3, -1, -1, 4, 5])
    g = S.angle_cap_groups(surf, key, emu_class=np.array([1, 0, 0, 1, 1]))
    assert list(g)[:3] == ["all", "IL1", "ID1"] and "ID1/unlinked" in g and "tpEmuClass=0" in g
    # a pruned TP (tpIdx <= -3) is linked; -2 (truth unavailable) is not
    rows = S.angle_cap_groups(surf, np.array([0, -1, -2, -7, 1]))
    assert rows["linked"].tolist() == [True, False, False, True, True]
    assert g["linked"].tolist() == [True, False, False, True, True]
    C = S.angle_cap_census(c, g)
    a = C["angles"]["alpha"]["all"]
    assert (a["n_total"], a["n_valid"], a["n_pass"], a["n_fail"]) == (5, 4, 2, 2)
    assert a["frac_fail"] == 0.5
    # the loader's cap drops a set flag over |cot| > cap or over the -999 fill; a clear flag is not counted
    assert (a["n_flag_set"], a["n_dropped_by_cap"]) == (4, 2)
    assert C["angles"]["beta"]["ID1/linked"]["n_fail"] == 1
    d = C["angles"]["direction"]["all"]
    assert (d["n_valid"], d["n_pass"], d["n_fail"]) == (4, 1, 3)
    assert d["n_dropped_by_cap"] == int((~S.direction_usable(c)).sum()) - 1   # the clear flag
    # summing two halves equals the whole
    h1 = S.angle_cap_census({k: v[:2] for k, v in c.items()}, {"all": None})
    h2 = S.angle_cap_census({k: v[2:] for k, v in c.items()}, {"all": None})
    s = S.angle_cap_census_add(S.angle_cap_census_add(None, h1), h2)
    assert s["angles"]["alpha"]["all"] == a
    assert S.angle_cap_census({k: v[:0] for k, v in c.items()}, {"all": None}
                              )["angles"]["beta"]["all"]["frac_fail"] is None
