"""SmartPixels nano IO.

The cluster table is ONE ``L1TSmartPixelsCluster``: every barrel cluster
(TBPX layers 1-4) THEN the disc clusters of the configured scope (default
ID1-ID4, both sides), with geometry columns ``subdet`` (1 PXB, 2 PXF),
``layer`` (0 on disc), ``disk``/``side``/``ring`` (0 on barrel), ``surface``
(1-4 = IL1-IL4, 10 + disk on disc: ID1-ID4 = 11-14, TFPX5-8 = 15-18, TEPX =
19-22) and ``idxInDet`` (position within the module; (event, detId, idxInDet)
is a stable cluster key). Barrel rows are a per-event PREFIX, so
``selClusterIdx`` (a barrel row index written by the refit) indexes the table,
and filtering to ``surfaces=BARREL`` preserves it. Older layouts (a barrel
table plus ``L1TSmartPixelsDiscCluster``, or a barrel table without the
geometry columns) are not supported: ``require_unified_clusters`` refuses them.

TrackingParticle links are ROWS into L1TTP, like every other nano link:
``tpIdx`` on L1TTrack, L1TExtTrack, L1TOTStub, L1TSmartPixelsCluster and the
SmartPixels track tables (plus ``tpIdxInner`` / ``tpIdxOuter`` on L1TOTStub),
each with its table's own definition (see the DPGAnalysis/Phase3SmartPixelsNanoAOD
README, "Truth links"), with
  >= 0  row in L1TTP
  -1    not truth-matched
  -2    TrackingParticle product unavailable
  <= -3 matched TP not written to L1TTP (outside the L1TTP selection, which the
        L1TTP table doc states; 1 GeV nanos: charged |p| > 1 GeV); -3 - tpIdx
        is its index in the UNPRUNED TrackingParticle collection (not written)
Two objects share a TP iff their tpIdx are equal. They are read as ordinary
columns. Nanos that linked TPs by collection key (cnn_nano v1-v4: tpKey /
spixMatchedTpKey + L1TTP_key, or keys under the *Idx names + L1TTP_idx) are not
supported.

Measured-angle cap. ``hasAlpha``/``hasBeta`` say whether a cluster's
``localCotAlpha``/``localCotBeta`` may be used. The payload-mode rec-hit
producer (truth + parametrised smear) cleared them beyond |cot| > 12; the emu
mode sets them to 1 on EVERY cluster, so the flag cannot leak truth, and the
cap moved into the algorithms that use the angles (the CMSSW refit applies it
itself). ``cluster_columns`` therefore returns has* = has* AND |cot| <=
``MEAS_ANGLE_MAX_ABS`` by default, which is a no-op on payload-mode files;
``cap_angles=False`` returns the stored flags (for audits of the table itself).
The global direction columns (``DIRECTION_COLUMNS``) are built from both
angles regardless of the flags and are NOT masked by the loader; a consumer
gates them with ``direction_usable``. ``angle_cap_census`` counts what the cap
passes and fails (the omnibus reports it on every run as a sentinel).

Track -> stub link tables. L1TTrackStub / L1TExtTrackStub and the SmartPixels
L1TSmartPixels[Ext]TrackStub are link-only: trackIdx and otStubIdx, the stub's
row in L1TOTStub. ``branch_names`` / ``table_columns`` / ``column`` /
``has_column`` return the stub columns (``STUB_COLUMNS``) of the tables in
``TRACK_STUB_TABLES`` through that join (``track_stub_branches`` /
``track_stub_columns``).

"""

from __future__ import annotations

import awkward as ak
import numpy as np

CLUSTER_TABLE = "L1TSmartPixelsCluster"
OT_TABLE = "L1TOTStub"
TP_TABLE = "L1TTP"

DISC_SURFACE_OFFSET = 10
BARREL = (1, 2, 3, 4)                       # surfaces IL1-IL4
SP_DISCS = (11, 12, 13, 14)                 # surfaces ID1-ID4
# disc_scope -> largest pxfDisk kept (TFPX = disks 1-8, TEPX = 9-12)
DISC_SCOPES = {"none": 0, "sp": 4, "tfpx": 8, "all": 12}

# Largest |cot| of a measured cluster angle that is used: the default of the
# CMSSW refit's seedlessRefitMeasAngleMaxAbs (~85 deg from normal incidence).
MEAS_ANGLE_MAX_ABS = 12.0
# has-flag -> the angle it qualifies
ANGLE_FLAGS = {"hasAlpha": "localCotAlpha", "hasBeta": "localCotBeta"}
# the producers fill an absent value with -999; anything at or below this is that fill
ABSENT_BELOW = -900.0
# global direction of the cluster, built from BOTH angles (see direction_usable)
DIRECTION_COLUMNS = ("globalClusterPhi", "globalClusterCotTheta",
                     "sigGlobalClusterPhi", "sigGlobalClusterCotTheta")

# track -> stub link tables, and the L1TOTStub columns read through them
TRACK_STUB_TABLES = ("L1TTrackStub", "L1TExtTrackStub",
                     "L1TSmartPixelsTrackStub", "L1TSmartPixelsExtTrackStub")
STUB_COLUMNS = ("layer", "isBarrel", "detId", "x", "y", "z", "r", "phi", "bend")


def _fields(A):
    return list(A.fields) if hasattr(A, "fields") else list(A.keys())


def branch_name(keys, table, col, required=True):
    """The branch `table`_`col` if `keys` holds it. KeyError when absent, or
    None if not `required`."""
    name = f"{table}_{col}"
    if name in set(keys):
        return name
    if required:
        raise KeyError(f"no branch {name}")
    return None


def _branches(keys, table, col):
    if _stub_joined(table, col):
        return track_stub_branches(keys, table, [col])
    return [branch_name(keys, table, col)]


def _column(A, f, table, col):
    if _stub_joined(table, col):
        return track_stub_columns(A, table, [col])[col]
    return A[branch_name(f, table, col)]


def has_column(keys, table, col):
    if _stub_joined(table, col):
        keys = set(keys)
        return f"{table}_otStubIdx" in keys and f"{OT_TABLE}_{col}" in keys
    return branch_name(keys, table, col, required=False) is not None


def branch_names(keys, table, cols):
    """The branches to hand uproot for `cols` of `table`."""
    return list(dict.fromkeys(b for c in cols for b in _branches(keys, table, c)))


def column(A, table, col):
    """`table`_`col` from an uproot batch `A`."""
    return _column(A, _fields(A), table, col)


def table_columns(A, table, cols):
    """{col: jagged array} for `cols` of `table`, keyed by the REQUESTED names."""
    f = _fields(A)
    return {c: _column(A, f, table, c) for c in cols}


# ---------------------------------------------------------------------------
# track -> stub link tables (see the module docstring)
# ---------------------------------------------------------------------------
def _stub_joined(table, col):
    """`table`_`col` is an L1TOTStub column read through a track -> stub link table."""
    return table in TRACK_STUB_TABLES and col in STUB_COLUMNS


def track_stub_branches(keys, table, cols):
    """The branches ``track_stub_columns`` needs for `cols` of the track -> stub
    link table `table`: its own columns, and otStubIdx plus the L1TOTStub
    column for the stub columns."""
    keys = set(keys)
    out = []
    for c in cols:
        if _stub_joined(table, c):
            out += [branch_name(keys, table, "otStubIdx"), branch_name(keys, OT_TABLE, c)]
        else:
            out.append(branch_name(keys, table, c))
    return list(dict.fromkeys(out))


def track_stub_columns(A, table, cols):
    """{col: jagged array, one entry per link row} for `cols` of the track ->
    stub link table `table`, from an uproot batch read with
    ``track_stub_branches``; stub columns are L1TOTStub[otStubIdx]."""
    f = set(_fields(A))
    out = {}
    for c in cols:
        if _stub_joined(table, c):
            out[c] = A[branch_name(f, OT_TABLE, c)][A[branch_name(f, table, "otStubIdx")]]
        else:
            out[c] = A[branch_name(f, table, c)]
    return out


# ---------------------------------------------------------------------------
# the measured-angle cap (see the module docstring)
# ---------------------------------------------------------------------------
def angle_usable(has, cot, max_abs=MEAS_ANGLE_MAX_ABS):
    """Boolean: the angle `cot` may be used (flag `has` set and |cot| <= max_abs).
    Works on numpy, awkward and pandas columns alike."""
    return (has > 0) & (abs(cot) <= max_abs)


def cap_angle_flags(cols, max_abs=MEAS_ANGLE_MAX_ABS):
    """In the {column: array} dict `cols`, replace each has-flag whose angle is
    also present by ``angle_usable`` (keeping the flag's dtype). A flag whose
    angle is absent is left as stored. Returns `cols` (modified in place)."""
    for flag, cot in ANGLE_FLAGS.items():
        if flag in cols and cot in cols:
            cols[flag] = cols[flag] * angle_usable(cols[flag], cols[cot], max_abs)
    return cols


def direction_usable(cols, max_abs=MEAS_ANGLE_MAX_ABS):
    """Boolean: the cluster's global direction (DIRECTION_COLUMNS) may be used.

    The table producer builds it from BOTH localCotAlpha and localCotBeta
    without looking at hasAlpha/hasBeta (it only needs them finite), so a
    capped angle -- or a -999 fill, which on payload-era files becomes a
    constant globalClusterCotTheta of ~-1 rather than a -999 -- passes straight
    into it. Usable iff both angles pass ``angle_usable``. `cols` maps
    hasAlpha, hasBeta, localCotAlpha, localCotBeta to arrays."""
    return (angle_usable(cols["hasAlpha"], cols["localCotAlpha"], max_abs)
            & angle_usable(cols["hasBeta"], cols["localCotBeta"], max_abs))


def surface_name(code):
    """IL1-IL4 for barrel surface codes 1-4, ID<n> for disc codes 10+n."""
    code = int(code)
    return f"ID{code - DISC_SURFACE_OFFSET}" if code > DISC_SURFACE_OFFSET else f"IL{code}"


def _cap_counts(cot, has, max_abs):
    cot = np.asarray(cot)
    valid = cot > ABSENT_BELOW
    ok = valid & (np.abs(cot) <= max_abs)
    n_valid, n_pass = int(valid.sum()), int(ok.sum())
    c = {"n_total": int(len(cot)), "n_valid": n_valid, "n_pass": n_pass,
         "n_fail": n_valid - n_pass,
         "frac_fail": (n_valid - n_pass) / n_valid if n_valid else None}
    if has is not None:
        has = np.asarray(has)
        c["n_flag_set"] = int((has > 0).sum())
        c["n_dropped_by_cap"] = int(((has > 0) & ~angle_usable(has, cot, max_abs)).sum())
    return c


def angle_cap_groups(surface=None, tp_link=None, emu_class=None):
    """{label: boolean mask or None (= every cluster)} for ``angle_cap_census``:
    'all'; each surface (IL1.., ID1..); 'linked'/'unlinked' (truth-matched:
    tpIdx >= 0 or <= -3); each tpEmuClass value;
    and surface x link."""
    g = {"all": None}
    surfs = [] if surface is None else [int(s) for s in np.unique(surface)]
    for s in surfs:
        g[surface_name(s)] = surface == s
    linked = None if tp_link is None else (tp_link >= 0) | (tp_link <= -3)
    if linked is not None:
        g["linked"], g["unlinked"] = linked, ~linked
    if emu_class is not None:
        for v in np.unique(emu_class):
            g[f"tpEmuClass={int(v)}"] = emu_class == v
    if linked is not None:
        for s in surfs:
            g[f"{surface_name(s)}/linked"] = (surface == s) & linked
            g[f"{surface_name(s)}/unlinked"] = (surface == s) & ~linked
    return g


def angle_cap_census(cols, groups, max_abs=MEAS_ANGLE_MAX_ABS):
    """How many measured angles the cap passes and fails, per group.

    `cols` maps localCotAlpha, localCotBeta and optionally the STORED (not
    capped, i.e. read with cap_angles=False) hasAlpha/hasBeta to flat arrays;
    `groups` is from ``angle_cap_groups``. Returns {"max_abs", "angles":
    {angle: {group: counts}}} for angle in alpha, beta and 'direction' (the
    pair, as globalClusterPhi/CotTheta use it: valid when both are, passing
    when both pass). counts: n_total; n_valid (not the -999 fill); n_pass
    (valid, |cot| <= max_abs); n_fail (valid, |cot| > max_abs); frac_fail =
    n_fail / n_valid (None when n_valid = 0); and, with the flags,
    n_flag_set and n_dropped_by_cap (flag set but not ``angle_usable``: what
    the loader's cap removes, a -999 fill under a set flag included)."""
    ca, cb = np.asarray(cols["localCotAlpha"]), np.asarray(cols["localCotBeta"])
    ha, hb = cols.get("hasAlpha"), cols.get("hasBeta")
    both = (ca > ABSENT_BELOW) & (cb > ABSENT_BELOW)
    series = {"alpha": (ca, ha), "beta": (cb, hb),
              "direction": (np.where(both, np.maximum(np.abs(ca), np.abs(cb)), ABSENT_BELOW - 99.0),
                            None if ha is None or hb is None
                            else np.minimum(np.asarray(ha), np.asarray(hb)))}
    out = {"max_abs": float(max_abs), "angles": {}}
    for name, (cot, has) in series.items():
        out["angles"][name] = {
            lab: _cap_counts(cot if m is None else cot[m],
                             None if has is None else (has if m is None else has[m]), max_abs)
            for lab, m in groups.items()}
    return out


def angle_cap_census_add(acc, census):
    """Sum `census` into `acc` (both from ``angle_cap_census``; `acc` may be
    None), recomputing frac_fail. Returns the sum."""
    if acc is None:
        acc = {"max_abs": census["max_abs"], "angles": {}}
    for name, cells in census["angles"].items():
        A = acc["angles"].setdefault(name, {})
        for lab, c in cells.items():
            s = A.setdefault(lab, {k: 0 for k in c if k != "frac_fail"})
            for k, v in c.items():
                if k != "frac_fail":
                    s[k] = s.get(k, 0) + v
            s["frac_fail"] = s["n_fail"] / s["n_valid"] if s["n_valid"] else None
    return acc


def _cap_partners(keys, cols):
    """The angle columns the cap needs that `cols` lacks and `keys` holds."""
    return [cot for flag, cot in ANGLE_FLAGS.items()
            if flag in cols and cot not in cols and has_column(keys, CLUSTER_TABLE, cot)]


# ---------------------------------------------------------------------------
# the cluster table
# ---------------------------------------------------------------------------
def require_unified_clusters(keys):
    """KeyError naming the problem unless `keys` hold the unified cluster
    table (see the module docstring)."""
    keys = set(keys)
    if f"{CLUSTER_TABLE}_surface" in keys:
        return
    if f"{CLUSTER_TABLE}_layer" in keys:
        raise KeyError(f"{CLUSTER_TABLE} has no surface column: a pre-unification nano "
                       "(barrel table, discs absent or in L1TSmartPixelsDiscCluster), "
                       "which is not supported")
    raise KeyError(f"no {CLUSTER_TABLE} table")


def _max_disk(disc_scope):
    try:
        return DISC_SCOPES[disc_scope]
    except KeyError:
        raise ValueError(f"disc_scope {disc_scope!r} not in {sorted(DISC_SCOPES)}") from None


def cluster_branches(keys, cols, disc_scope="sp", surfaces=None, cap_angles=True):
    """The raw branches `cluster_columns` needs for `cols` on a file with
    branch names `keys`. Pass them to uproot alongside any other table's.
    With `cap_angles` a requested has-flag also reads its angle (for the cap).
    `disc_scope` and `surfaces` do not change the branches (the filter reads
    surface either way); they are accepted to mirror ``cluster_columns``."""
    require_unified_clusters(keys)
    cols = list(cols)
    if cap_angles:
        cols += _cap_partners(keys, cols)
    return branch_names(keys, CLUSTER_TABLE, cols + ["subdet", "surface"])


def _surface_mask(surf, disc_scope, surfaces):
    keep = surf <= DISC_SURFACE_OFFSET + _max_disk(disc_scope)
    if surfaces is not None:
        allowed = np.zeros(256, bool)
        allowed[np.asarray(list(surfaces), dtype=np.int64)] = True
        flat = ak.to_numpy(ak.flatten(surf)).astype(np.int64)
        keep = keep & ak.unflatten(allowed[flat], ak.num(surf))
    return keep


def _check_barrel_first(subdet):
    """Barrel rows must be a per-event prefix, or selClusterIdx is not a row."""
    s = ak.to_numpy(ak.flatten(subdet)).astype(np.int64)
    n = ak.to_numpy(ak.num(subdet))
    ev = np.repeat(np.arange(len(n)), n)
    # within an event subdet must never step down (2 -> 1)
    bad = (np.diff(s) < 0) & (ev[1:] == ev[:-1])
    if bad.any():
        raise AssertionError(f"{CLUSTER_TABLE}: a barrel row follows a disc row "
                             f"({int(bad.sum())} times): barrel is not a prefix")


def cluster_columns(A, cols, disc_scope="sp", surfaces=None, cap_angles=True):
    """{col: jagged per-event array} of the cluster table.

    `A` is an uproot batch read with (at least) ``cluster_branches(...)``.

    `disc_scope` keeps discs up to pxfDisk 4 ("sp", default: ID1-ID4 only,
    as the producer writes by default), 8 ("tfpx"), 12 ("all") or none
    ("none"); it can only narrow what the producer wrote. `surfaces` (e.g.
    BARREL) further keeps only those surface codes.

    Row numbers: barrel rows keep their table row number whatever the filter
    (they are a per-event prefix), so ``surfaces=BARREL`` is safe to index
    with selClusterIdx. Disc rows keep theirs only when no disc is dropped.

    `cap_angles` (default) returns hasAlpha/hasBeta AND |cot| <=
    MEAS_ANGLE_MAX_ABS, reading the angle when `A` holds it (it does when `A`
    was read with ``cluster_branches``); False returns the stored flags.
    """
    cols = list(cols)
    if not cap_angles:
        return _cluster_columns(A, cols, disc_scope, surfaces)
    out = cap_angle_flags(_cluster_columns(A, cols + _cap_partners(_fields(A), cols),
                                           disc_scope, surfaces))
    return {c: out[c] for c in cols}


def _cluster_columns(A, cols, disc_scope, surfaces):
    f = _fields(A)
    require_unified_clusters(f)
    _check_barrel_first(A[f"{CLUSTER_TABLE}_subdet"])
    out = {c: _column(A, f, CLUSTER_TABLE, c) for c in cols}
    mask = _surface_mask(A[f"{CLUSTER_TABLE}_surface"], disc_scope, surfaces)
    if not ak.all(mask):
        out = {c: v[mask] for c, v in out.items()}
    return out


def read_clusters(path, cols, disc_scope="sp", surfaces=None,
                  entry_start=None, entry_stop=None, tree="Events", cap_angles=True):
    """cluster_columns on one file (for small reads and checks)."""
    import uproot

    t = uproot.open(f"{path}:{tree}")
    br = cluster_branches(t.keys(), cols, disc_scope, surfaces, cap_angles)
    A = t.arrays(br, entry_start=entry_start, entry_stop=entry_stop)
    return cluster_columns(A, cols, disc_scope, surfaces, cap_angles)


def flat_clusters(A, cols, disc_scope="sp", surfaces=None, event_offset=0, cap_angles=True):
    """cluster_columns flattened to numpy, plus 'event' (event_offset + entry)
    and the per-event counts: (dict, counts)."""
    if not cols:
        raise ValueError("flat_clusters needs at least one column")
    J = cluster_columns(A, cols, disc_scope, surfaces, cap_angles)
    n = ak.to_numpy(ak.num(J[cols[0]]))
    D = {c: ak.to_numpy(ak.flatten(v)) for c, v in J.items()}
    D["event"] = np.repeat(np.arange(event_offset, event_offset + len(n)), n)
    return D, n
