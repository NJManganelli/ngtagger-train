"""SmartPixels nano IO across the disc-unification schema change.

One reader API for three generations of the SmartPixels nano:

  unified   (current) ONE ``L1TSmartPixelsCluster`` table: every barrel cluster
            (TBPX layers 1-4, in the order of the old barrel table) THEN the
            disc clusters of the configured scope (default ID1-ID4, both
            sides), with geometry columns ``subdet`` (1 PXB, 2 PXF), ``layer``
            (0 on disc), ``disk``/``side``/``ring`` (0 on barrel), ``surface``
            (1-4 = IL1-IL4, 10 + disk on disc: ID1-ID4 = 11-14, TFPX5-8 =
            15-18, TEPX = 19-22) and ``idxInDet`` (position within the module;
            (event, detId, idxInDet) is a stable cluster key).
  split     (e.g. the itot_cnn_f* nanos) a barrel ``L1TSmartPixelsCluster``
            with ``layer`` and a separate ``L1TSmartPixelsDiscCluster`` with
            ``disk``/``side``/``ring`` for EVERY TFPX and TEPX disk.
  barrel    (e.g. itot_tp_f*) the barrel table alone.

``cluster_columns`` returns the unified layout from any of them: on split files
it concatenates per event barrel THEN disc and synthesizes the geometry
columns, and by default keeps only ID1-ID4 (``disc_scope="sp"``) so old files
look like new ones. Barrel rows are always a per-event PREFIX with their
original row numbers, so ``selClusterIdx`` (a barrel row index written by the
refit) stays valid on every layout, and filtering to ``surfaces=BARREL``
preserves it.

Key columns were renamed from *Idx to *Key where they hold a TrackingParticle
collection KEY rather than a row index (``tpIdx`` -> ``tpKey`` on the cluster
and OT stub tables, ``L1TTP_idx`` -> ``L1TTP_key``, ``spixMatchedTpIdx`` ->
``spixMatchedTpKey`` on the refit track tables). ``branch_name`` resolves
either spelling in both directions, new name preferred, so a reader written
against one generation reads the other. True row indices keep *Idx
(selClusterIdx, trackIdx, genPartIdx, otStubIdx).

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

Mixed generations in one ``uproot.iterate`` call are not supported (uproot
itself needs the same branches in every file).
"""

from __future__ import annotations

import awkward as ak
import numpy as np

CLUSTER_TABLE = "L1TSmartPixelsCluster"
OLD_DISC_TABLE = "L1TSmartPixelsDiscCluster"
OT_TABLE = "L1TOTStub"
TP_TABLE = "L1TTP"

SUBDET_PXB, SUBDET_PXF = 1, 2
DISC_SURFACE_OFFSET = 10
BARREL = (1, 2, 3, 4)                       # surfaces IL1-IL4
SP_DISCS = (11, 12, 13, 14)                 # surfaces ID1-ID4
# disc_scope -> largest pxfDisk kept (TFPX = disks 1-8, TEPX = 9-12)
DISC_SCOPES = {"none": 0, "sp": 4, "tfpx": 8, "all": 12}
GEOMETRY = ("subdet", "layer", "disk", "side", "ring", "surface", "idxInDet")

# Largest |cot| of a measured cluster angle that is used: the default of the
# CMSSW refit's digiRefitMeasAngleMaxAbs (~85 deg from normal incidence).
MEAS_ANGLE_MAX_ABS = 12.0
# has-flag -> the angle it qualifies
ANGLE_FLAGS = {"hasAlpha": "localCotAlpha", "hasBeta": "localCotBeta"}
# the producers fill an absent value with -999; anything at or below this is that fill
ABSENT_BELOW = -900.0
# global direction of the cluster, built from BOTH angles (see direction_usable)
DIRECTION_COLUMNS = ("globalClusterPhi", "globalClusterCotTheta",
                     "sigGlobalClusterPhi", "sigGlobalClusterCotTheta")

# new spelling -> old spelling; a None table applies to every table
_RENAMES = {
    None: {"tpKey": "tpIdx", "spixMatchedTpKey": "spixMatchedTpIdx"},
    TP_TABLE: {"key": "idx"},
}


def _fields(A):
    return list(A.fields) if hasattr(A, "fields") else list(A.keys())


def aliases(table, col):
    """Every spelling of `table`_`col`, the requested one first."""
    out = [col]
    for scope in (table, None):
        ren = _RENAMES.get(scope, {})
        for new, old in ren.items():
            if col == new and old not in out:
                out.append(old)
            elif col == old and new not in out:
                out.append(new)
    return out


def branch_name(keys, table, col, required=True):
    """The branch in `keys` that holds `table`_`col`, under the requested
    spelling or its renamed counterpart (tpKey <-> tpIdx, L1TTP key <-> idx,
    spixMatchedTpKey <-> spixMatchedTpIdx). The requested spelling wins when
    both exist. KeyError when neither does, or None if not `required`."""
    keys = set(keys)
    for c in aliases(table, col):
        if f"{table}_{c}" in keys:
            return f"{table}_{c}"
    if required:
        raise KeyError(f"no branch {table}_{col} (tried {aliases(table, col)})")
    return None


def has_column(keys, table, col):
    return branch_name(keys, table, col, required=False) is not None


def branch_names(keys, table, cols):
    """branch_name for each of `cols` (the list to hand uproot)."""
    return [branch_name(keys, table, c) for c in cols]


def column(A, table, col):
    """`table`_`col` from an uproot batch `A`, by either spelling."""
    return A[branch_name(_fields(A), table, col)]


def table_columns(A, table, cols):
    """{col: jagged array} for `cols` of `table`, keyed by the REQUESTED names."""
    f = _fields(A)
    return {c: A[branch_name(f, table, c)] for c in cols}


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


def angle_cap_groups(surface=None, tp_key=None, emu_class=None):
    """{label: boolean mask or None (= every cluster)} for ``angle_cap_census``:
    'all'; each surface (IL1.., ID1..); 'linked'/'unlinked' (tpKey >= 0); each
    tpEmuClass value; and surface x link."""
    g = {"all": None}
    surfs = [] if surface is None else [int(s) for s in np.unique(surface)]
    for s in surfs:
        g[surface_name(s)] = surface == s
    if tp_key is not None:
        g["linked"], g["unlinked"] = tp_key >= 0, tp_key < 0
    if emu_class is not None:
        for v in np.unique(emu_class):
            g[f"tpEmuClass={int(v)}"] = emu_class == v
    if tp_key is not None:
        for s in surfs:
            g[f"{surface_name(s)}/linked"] = (surface == s) & (tp_key >= 0)
            g[f"{surface_name(s)}/unlinked"] = (surface == s) & (tp_key < 0)
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
def cluster_layout(keys):
    """'unified', 'split' or 'barrel' (see the module docstring)."""
    keys = set(keys)
    if f"{CLUSTER_TABLE}_surface" in keys:
        return "unified"
    if any(k.startswith(OLD_DISC_TABLE + "_") for k in keys):
        return "split"
    if f"{CLUSTER_TABLE}_layer" in keys:
        return "barrel"
    raise KeyError(f"no {CLUSTER_TABLE} table")


def _max_disk(disc_scope):
    try:
        return DISC_SCOPES[disc_scope]
    except KeyError:
        raise ValueError(f"disc_scope {disc_scope!r} not in {sorted(DISC_SCOPES)}") from None


def _wants_disc(disc_scope, surfaces):
    if _max_disk(disc_scope) == 0:
        return False
    return surfaces is None or any(int(s) > DISC_SURFACE_OFFSET for s in surfaces)


def cluster_branches(keys, cols, disc_scope="sp", surfaces=None, cap_angles=True):
    """The raw branches `cluster_columns` needs for `cols` on a file with
    branch names `keys`. Pass them to uproot alongside any other table's.
    With `cap_angles` a requested has-flag also reads its angle (for the cap)."""
    lay = cluster_layout(keys)
    cols = list(cols)
    if cap_angles:
        cols += _cap_partners(keys, cols)
    if lay == "unified":
        need = cols + ["subdet", "surface"]
        return list(dict.fromkeys(branch_name(keys, CLUSTER_TABLE, c) for c in need))
    plain = [c for c in cols if c not in GEOMETRY]
    need_det = "idxInDet" in cols
    out = [f"{CLUSTER_TABLE}_layer"]
    out += [branch_name(keys, CLUSTER_TABLE, c) for c in plain]
    if need_det:
        out.append(f"{CLUSTER_TABLE}_detId")
    if lay == "split" and _wants_disc(disc_scope, surfaces):
        out.append(f"{OLD_DISC_TABLE}_disk")
        out += [f"{OLD_DISC_TABLE}_{c}" for c in ("side", "ring") if c in cols]
        out += [branch_name(keys, OLD_DISC_TABLE, c) for c in plain]
        if need_det:
            out.append(f"{OLD_DISC_TABLE}_detId")
    return list(dict.fromkeys(out))


def _idx_in_det(event_counts, det):
    """Position of each row within its run of equal detId, per event.

    The old producers wrote every cluster of a module contiguously, in
    DetSet order, so this reproduces the unified table's idxInDet."""
    counts = ak.to_numpy(event_counts)
    flat = ak.to_numpy(ak.flatten(det)).astype(np.int64)
    ev = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    key = (ev << 32) | flat
    start = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, bool)
    grp = np.cumsum(start) - 1
    pos = np.arange(len(key)) - np.flatnonzero(start)[grp] if len(key) else np.zeros(0, np.int64)
    return ak.unflatten(pos.astype(np.uint16), counts)


def _const_like(jag, value, dtype=np.uint8):
    return ak.values_astype(ak.zeros_like(jag) + value, dtype)


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
    """{col: jagged per-event array} of the cluster table in the unified layout.

    `A` is an uproot batch read with (at least) ``cluster_branches(...)``.
    `cols` may name any cluster column under either key spelling, plus the
    geometry columns (synthesized on split/barrel files).

    `disc_scope` keeps discs up to pxfDisk 4 ("sp", default: ID1-ID4 only,
    as the unified producer writes), 8 ("tfpx"), 12 ("all") or none ("none").
    On a unified file it can only narrow what the producer wrote.
    `surfaces` (e.g. BARREL) further keeps only those surface codes.

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
    lay = cluster_layout(f)
    cols = list(cols)
    if lay == "unified":
        subdet = A[f"{CLUSTER_TABLE}_subdet"]
        _check_barrel_first(subdet)
        out = {c: A[branch_name(f, CLUSTER_TABLE, c)] for c in cols}
        mask = _surface_mask(A[f"{CLUSTER_TABLE}_surface"], disc_scope, surfaces)
        if not ak.all(mask):
            out = {c: v[mask] for c, v in out.items()}
        return out

    blayer = A[f"{CLUSTER_TABLE}_layer"]
    B = {c: A[branch_name(f, CLUSTER_TABLE, c)] for c in cols if c not in GEOMETRY}
    B["subdet"] = _const_like(blayer, SUBDET_PXB)
    B["layer"] = ak.values_astype(blayer, np.uint8)
    B["disk"] = B["side"] = B["ring"] = _const_like(blayer, 0)
    B["surface"] = B["layer"]
    if "idxInDet" in cols:
        B["idxInDet"] = A[f"{CLUSTER_TABLE}_detId"]      # replaced below
    if lay == "split" and _wants_disc(disc_scope, surfaces):
        disk = A[f"{OLD_DISC_TABLE}_disk"]
        D = {c: A[branch_name(f, OLD_DISC_TABLE, c)] for c in cols if c not in GEOMETRY}
        D["subdet"] = _const_like(disk, SUBDET_PXF)
        D["layer"] = _const_like(disk, 0)
        D["disk"] = ak.values_astype(disk, np.uint8)
        for c in ("side", "ring"):
            D[c] = (ak.values_astype(A[f"{OLD_DISC_TABLE}_{c}"], np.uint8)
                    if c in cols else _const_like(disk, 0))
        D["surface"] = ak.values_astype(disk + DISC_SURFACE_OFFSET, np.uint8)
        if "idxInDet" in cols:
            D["idxInDet"] = A[f"{OLD_DISC_TABLE}_detId"]
        full = {c: ak.concatenate([B[c], D[c]], axis=1) for c in B}
    else:
        full = B
    if "idxInDet" in cols:
        full["idxInDet"] = _idx_in_det(ak.num(full["surface"]), full["idxInDet"])
    mask = _surface_mask(full["surface"], disc_scope, surfaces)
    if not ak.all(mask):
        full = {c: v[mask] for c, v in full.items()}
    return {c: full[c] for c in cols}


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
