"""PF/PUPPI candidate transverse impact parameter (definition sheet v1, item `frame.puppi_dxy`).

The stock tagger receives hwDxy: an 8-bit, sqrt-like code of |d0| of the linked track. The
float branch `dxy` is hwDxy / 64 in sqrt(cm) units -- reading it as cm turns a prompt track
(code 3) into 469 um. The only valid reading is through the measured code table:
hwD0 = floor(d0 / (1/256 cm)) of the linked track, code = table(|hwD0|). Code 3 <-> |hwD0| <= 1
<-> d0 in [-78.125, 78.125) um; the next codes step by 78.125 um (7, 9, 11, 12, 13, ...).
"""
from __future__ import annotations

import json
import os

import numpy as np

D0_LSB_CM = 1.0 / 256          # track word: hwD0 = floor(d0 / LSB)
HWD0_BITS = 13
DXY_FLOAT_PER_CODE = 1.0 / 64  # the nano float `dxy` = hwDxy / 64 (sqrt(cm) units)
LUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "puppi_dxy_lut_v1.json")
_LUT = None


def hwd0(d0_cm):
    lim = 2 ** (HWD0_BITS - 1)
    return np.clip(np.floor(np.asarray(d0_cm, np.float64) / D0_LSB_CM), -lim, lim - 1).astype(np.int64)


def _lut():
    global _LUT
    if _LUT is None:
        with open(LUT_PATH) as f:
            d = json.load(f)
        _LUT = (np.asarray(d["abs_hwD0"], np.int64), np.asarray(d["hwDxy"], np.int64))
    return _LUT


def code_of_d0(d0_cm):
    """(hwDxy code, known) for track d0 values: known False where |hwD0| is not in the table."""
    ks, vs = _lut()
    a = np.abs(hwd0(d0_cm))
    i = np.clip(np.searchsorted(ks, a), 0, len(ks) - 1)
    return vs[i], ks[i] == a


def d0_interval_um(code):
    """[lo, hi) of |d0| in um covered by a code (from the table's |hwD0| set)."""
    ks, vs = _lut()
    sel = ks[vs == int(code)]
    if sel.size == 0:
        return (np.nan, np.nan)
    return (float(sel.min() * D0_LSB_CM * 1e4), float((sel.max() + 1) * D0_LSB_CM * 1e4))


def code_contains_d0(code, d0_cm):
    """True where the candidate's code is the table's code for its linked track's d0."""
    c, known = code_of_d0(d0_cm)
    return known & (c == np.asarray(code, np.int64))
