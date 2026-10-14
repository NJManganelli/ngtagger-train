"""The angle-error table and the k knob in the replay's module frame, through ngtagger.spdefs only
(PLAN §1 item 5; review L3; §10 PD3, PD6).

Barrel only. The frame check (PLAN §0) measured that on the barrel the module-frame pull
(localCot - tpLocalCot) / sigma_local equals the table's global pull with the spdefs orientation sign, so a
table row applies in the module frame in pull units:

    measurement   m' = m - k c_local sigma_pub        c_local = to_local(angle, c, orientation)
    uncertainty   sigma' = w sigma_k(sigma_pub, k, mode)

with sigma_pub the published LOCAL uncertainty (sigAlpha / sigBeta, cot units), never the angle-space
centre converted by its sign alone (that misses d cot / d angle). The row is the table's TRUE key
(the stand-in for a prediction) at the PREDICTED incidence: key_from_local of the predicted local
angles and the module's orientation from the table's own map.

Discs: no table (PLAN §7: the module-frame disc rows are C1b). A disc cluster keeps its published sigma
and no centre, and the k knob refuses it at k != 1 (PD3): it keeps its emitted angle.

The knob (decision 10) is a measurement transform: spdefs.improve on the cluster's GLOBAL angles
(alpha = globalPhi - direction phi, beta = lambda) of class-T clusters, then the improved direction is
rotated into the module frame (cotAlpha = d.ex / d.ez, cotBeta = d.ey / d.ez; the ratios do not depend on
the direction's sense). k = 1 returns the stored values untouched.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ngtagger.spdefs import angles as SA

TABLE = os.environ.get("WPC_ANGLE_TABLE",
                       "/Volumes/WDMac/smartpixels_scratch/p03_angles/angle_table_v1a1e_canon_v3.json")
KEY = "true"
ANGLES = ("alpha", "beta")
GLOBAL_COLS = ("globalPhi", "globalClusterPhi", "globalClusterCotTheta", "tpGlobalClusterPhi",
               "tpGlobalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta", "tpEmuClass",
               "hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta")        # the last four: spdefs.usable


class AngleImport:
    def __init__(self, path=TABLE):
        self.path = path
        self.table = SA.AngleTable.load(path)
        self.provenance = {"path": path, "schema": self.table.doc.get("schema"),
                           "era": self.table.doc.get("era"), "fit_sample": self.table.doc.get("fit_sample"),
                           "definition_sheet": self.table.doc.get("definition_sheet")}

    def rows(self, surface, det_id, cot_a_pred, cot_b_pred):
        """(table rows of the true key at the predicted incidence, orientation sign) per cluster;
        rows -1 on the discs, for an unknown module orientation or an unbinned key."""
        surface = np.asarray(surface)
        orient = self.table.orientation(det_id)
        key = SA.key_from_local(surface, cot_a_pred, cot_b_pred, orient)
        rows = self.table.rows_of(KEY, surface, key, SA.facing(surface, orient))
        rows = np.where(SA.is_disc(surface) | (orient == 0), -1, rows)
        return rows, orient

    def centre_width(self, angle, rows, orient, surface):
        """Module-frame centre (pull units, with the frame sign) and width per cluster; NaN where
        the cluster has no row."""
        c = self.table.get(KEY, angle, "centre", rows)
        w = self.table.get(KEY, angle, "width", rows)
        return SA.to_local(angle, c, orient, surface), w


def folded(imp):
    """KNOWN-BAD (PLAN §10 PD10 (d)): the table with each signed pair of rows (fine bins b and N - 1 - b,
    same surface and facing) averaged, n-weighted: the signs pooled, as a |key| table would."""
    doc = json.loads(json.dumps(imp.table.doc))
    for key, kd in doc["keys"].items():
        rows = kd["rows"]
        by = {(r["surface"], r["facing"], f): i for i, r in enumerate(rows) for f in r["fine"]}
        done = set()
        for i, r in enumerate(rows):
            nf = SA.N_FINE[int(r["surface"])]
            j = by.get((r["surface"], r["facing"], nf - 1 - r["fine"][0]))
            if j is None or j == i or (j, i) in done:
                continue
            done.add((i, j))
            for ang in SA.ANGLES:
                a, b = rows[i].get(ang), rows[j].get(ang)
                if not a or not b:
                    continue
                n = a["n"] + b["n"]
                for q in ("centre", "width"):
                    v = (a[q] * a["n"] + b[q] * b["n"]) / n
                    a[q] = b[q] = v
    out = AngleImport.__new__(AngleImport)
    out.path = imp.path + "#folded"
    out.table = SA.AngleTable(doc)
    out.provenance = dict(imp.provenance, variant="folded (known-bad)")
    return out


def calibrate(imp, surface, det_id, cot_a_pred, cot_b_pred, m_a, m_b, sig_a, sig_b, k_alpha, k_beta, mode,
              variant="", sig_global=None):
    """Barrel table import for weighting consumers: (m'_a, m'_b, sigma'_a, sigma'_b, has_row).
    Clusters without a row (discs, unknown orientation) keep m and get sigma_k of the published
    sigma (discs: k is refused, so their sigma stays published at any k).
    variant (KNOWN-BADS of E1(a)): "global_sign" applies the centre with orientation +1 on every module
    (the global cot theta sign); "angle_units" applies k c sigma_global (radians, sig_global = (alpha,
    beta) global published sigmas) to the local cot without the d cot / d angle conversion."""
    rows, orient = imp.rows(surface, det_id, cot_a_pred, cot_b_pred)
    if variant == "global_sign":
        orient = np.where(orient != 0, 1, 0).astype(np.int8)
    out = []
    disc = SA.is_disc(surface)
    for i, (angle, m, sig, k) in enumerate((("alpha", m_a, sig_a, k_alpha), ("beta", m_b, sig_b, k_beta))):
        c, w = imp.centre_width(angle, rows, orient, surface)
        kk = np.where(disc, 1.0, k)
        sk = np.where(disc, sig, SA.sigma_k(sig, k, mode))
        ok = rows >= 0
        unit = sig_global[i] if variant == "angle_units" else sig
        out.append((np.where(ok, m - kk * np.nan_to_num(c) * unit, m),
                    np.where(ok, np.nan_to_num(w, nan=1.0) * sk, sk)))
    return out[0][0], out[1][0], out[0][1], out[1][1], rows >= 0


def knob(cols, ex, ey, ez, surface, k_alpha, k_beta, local_a, local_b):
    """The improved local angles (cotAlpha', cotBeta') of class-T clusters at (k_alpha, k_beta); class C,
    disc clusters (PD3) and k = 1 keep local_a / local_b exactly. cols: the GLOBAL_COLS of the clusters;
    ex, ey, ez: their modules' local axes (global vectors)."""
    k_alpha, k_beta = SA.check_k(k_alpha), SA.check_k(k_beta)
    a_out = np.array(local_a, np.float64, copy=True)
    b_out = np.array(local_b, np.float64, copy=True)
    if k_alpha == 1.0 and k_beta == 1.0:
        return a_out, b_out
    isT = SA.class_T(cols["tpEmuClass"]) & ~SA.is_disc(surface) & SA.usable(cols)
    al, _ = SA.improve("alpha", SA.emitted("alpha", cols), SA.truth("alpha", cols), isT, k_alpha)
    be, _ = SA.improve("beta", SA.emitted("beta", cols), SA.truth("beta", cols), isT, k_beta)
    phi = SA.angle_from("alpha", al, cols)
    cot = SA.angle_from("beta", be, cols)
    d = np.stack([np.cos(phi), np.sin(phi), cot], -1)
    lz = (d * ez).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ca = (d * ex).sum(-1) / lz
        cb = (d * ey).sum(-1) / lz
    a_out[isT] = ca[isT]
    b_out[isT] = cb[isT]
    return a_out, b_out
