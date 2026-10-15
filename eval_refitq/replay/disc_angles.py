"""C1b: the disc angle rows imported into the replay (c1b/PLAN.md rev 4 §1 item 1).

The rows of p03.table.v1a1e-disclocal are already in the module frame (local cot units, facings pooled), so
the import is exactly

    row = rows_of("true_local", surface, local_incidence_key(cotA_pred, cotB_pred), 0)
    m'  = m - c sigma_local,     sigma' = w sigma_local      (fixed mode)
                                 sigma' = w max(k, K_MIN) sigma_local      (scaled mode, decision 10)

per angle, sigma_local = sigAlpha / sigBeta. It deliberately does NOT use angle_import.AngleImport.rows()
(every disc row -1), SA.key_from_local (the key mirrored on one facing) or SA.to_local (the beta centre
flipped on one facing, the alpha centre zeroed): the C1b facing known-bads plant exactly those two.

The k knob on the discs is linear in cot, the table's unit: cot' = tpLocalCot + k (localCot - tpLocalCot) on
class T under the |cot| <= 12 cap; class C is untouched.
"""
from __future__ import annotations

import numpy as np

from ngtagger.spdefs import angles as SA

KEY = "true_local"
K_MIN = 0.05                 # the smallest k used (rev 4 §1 item 1)
CAP = 12.0                   # the measured-angle cap (|cot| <= 12)
DISC_TABLE = "/Volumes/WDMac/smartpixels_scratch/p03_angles/angle_table_v1a1e_disclocal_canon_v3.json"


class DiscImport:
    """The disc rows: lookup and the calibrated (m', sigma') per angle."""

    def __init__(self, path=DISC_TABLE):
        self.path = path
        self.table = SA.AngleTable.load(path)

    def rows(self, surface, cotA_pred, cotB_pred):
        """Row per cluster (-1 none), keyed by the module-frame incidence key of the predicted cots."""
        return self.table.rows_of(KEY, surface, SA.local_incidence_key(cotA_pred, cotB_pred),
                                  np.zeros(len(np.atleast_1d(surface)), np.int8))

    def rows_known_bad_key(self, surface, cotA_pred, cotB_pred, facing):
        """The facing known-bad: the key through SA.key_from_local (x the facing sign)."""
        key = SA.key_from_local(surface, cotA_pred, cotB_pred, facing)
        return self.table.rows_of(KEY, surface, key, np.zeros(len(np.atleast_1d(surface)), np.int8))

    def centre_width(self, angle, rows):
        """(c, w) per cluster in local sigma units (NaN where no row)."""
        return self.table.get(KEY, angle, "centre", rows), self.table.get(KEY, angle, "width", rows)

    def calibrate(self, angle, m, sig_local, rows, k=1.0, mode="fixed", variant="", facing=None):
        """(m', sigma') for one angle. variant "to_local" (the facing known-bad): the centre passed through
        the barrel's frame conversion, i.e. -s c for beta and 0 for alpha."""
        c, w = self.centre_width(angle, rows)
        if variant == "to_local":
            c = np.zeros_like(c) if angle == "alpha" else -np.asarray(facing, np.float64) * c
        scale = 1.0 if mode == "fixed" else max(k, K_MIN)
        if mode == "scaled":
            c = k * c
        return m - c * sig_local, w * scale * sig_local


def knob(local_cot, tp_local_cot, k, class_t):
    """The cot-linear disc knob: cot' = tpLocalCot + k (localCot - tpLocalCot) on class T, under the cap; other
    clusters (class C, |cot| beyond the cap) untouched."""
    local_cot = np.asarray(local_cot, np.float64)
    out = local_cot.copy()
    if k == 1.0:                                 # bit-identical (tp + (loc - tp) need not round back to loc)
        return out
    m = np.asarray(class_t, bool) & (np.abs(local_cot) <= CAP) & np.isfinite(tp_local_cot)
    out[m] = tp_local_cot[m] + k * (local_cot[m] - tp_local_cot[m])
    return out
