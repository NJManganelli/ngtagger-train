"""The OT stubs as the L1 tracklet finder reads them: the DTC frames of L1TOTStubDTC (canon_v4).

Each L1TOTStubDTC row is one stub copy in one processing region (nonant) and DTC link, in the finder's
loop order (region, channel, position in the stream); otStubIdx is the L1TOTStub row. The 64-bit frame
(StubDTC.cc; L1FPGATrackProducer.cc, "READ DTC STUB INFORMATION"): bit 0 valid, bits 2-1 layer code,
then upward bend, alpha, phi, z, r, negDisk. decode() returns the integer fields as trklet::Stub reads
them (Stub.cc: r signed in the barrel, z signed, phi and bend raw unsigned), and the tracklet units
that turn them into physics: phi relative to the sector's phimin in kphi(layer), z in kz(layer),
barrel r about the layer's rmean in krbarrel... checked against Setup::stubPosDTC on the canon_v4 test
production (tests/test_ot_chain_kat.py).
"""
from __future__ import annotations
import math
import numpy as np

import cmssw_kf as KF

# TrackerDTC Setup_cfi Hybrid widths per module type (barrelPS, barrel2S, diskPS, disk2S)
W_ND = (0, 0, 1, 1)
W_R = (7, 7, 11, 6)
W_Z = (12, 8, 7, 7)
W_PHI = (14, 17, 14, 14)
W_ALPHA = (0, 0, 0, 4)
W_BEND = (3, 4, 3, 4)
BARREL_PS, BARREL_2S, DISK_PS, DISK_2S = range(4)

# trklet::Settings (Settings.h)
IRMEAN = (851, 1269, 1784, 2347, 2936, 3697)
NZBITS_L = (12, 12, 12, 8, 8, 8)
NPHIBITS_L = (14, 14, 14, 17, 17, 17)
NRBITS_L = (7, 7, 7, 7, 7, 7)
RMAXDISK, ZLENGTH, DELTARZFRACT = 120.0, 120.0, 32.0


def module_type(is_barrel, is_ps):
    is_barrel, is_ps = np.asarray(is_barrel, bool), np.asarray(is_ps, bool)
    return np.where(is_barrel, np.where(is_ps, BARREL_PS, BARREL_2S), np.where(is_ps, DISK_PS, DISK_2S))


def _field(word, shift, width):
    mask = (np.uint64(1) << width.astype(np.uint64)) - np.uint64(1)
    return ((word >> shift.astype(np.uint64)) & mask).astype(np.int64)


def _signed(v, width):
    half = np.int64(1) << (width - 1)
    return np.where(v >= half, v - (np.int64(1) << width), v)


def decode(frame_hi, frame_lo, mtype):
    """(layer_code, bend, alpha, phi, z, r, neg_disk) integer arrays from the frame words."""
    w = (np.asarray(frame_hi, np.uint64) << np.uint64(32)) | np.asarray(frame_lo, np.uint64)
    mt = np.asarray(mtype, np.int64)
    wb, wa, wp, wz, wr = (np.take(np.array(x, np.int64), mt) for x in (W_BEND, W_ALPHA, W_PHI, W_Z, W_R))
    layer_code = ((w >> np.uint64(1)) & np.uint64(3)).astype(np.int64)
    s = np.full(len(mt), 3, np.int64)
    bend = _field(w, s, wb); s = s + wb
    alpha = _field(w, s, wa); s = s + wa
    phi = _field(w, s, wp); s = s + wp
    z = _signed(_field(w, s, wz), wz); s = s + wz
    r_raw = _field(w, s, wr); s = s + wr
    barrel = mt < DISK_PS
    r = np.where(barrel, _signed(r_raw, wr), r_raw)
    alpha = np.where(wa > 0, _signed(alpha, np.maximum(wa, 1)), 0)
    neg_disk = np.where(barrel, 0, _field(w, s, np.ones(len(mt), np.int64)))
    return layer_code, bend, alpha, phi, z, r, neg_disk


def kphi(layer, k=KF.K):
    """trklet::Settings::kphi(layerdisk) for barrel layer 1..6."""
    return k.dphisector_hg / (1 << NPHIBITS_L[layer - 1])


def kz(layer):
    return 2.0 * ZLENGTH / (1 << NZBITS_L[layer - 1])


def rmean(layer):
    return IRMEAN[layer - 1] * RMAXDISK / 4096


def kr_barrel():
    """Settings::krbarrel(): 2 drmax / 2^nrbits."""
    return 2.0 * (RMAXDISK / DELTARZFRACT) / (1 << NRBITS_L[0])


def barrel_physical(layer, iphi, iz, ir, region, k=KF.K):
    """(phi global, z, r) of barrel stub integers: phi = iphi kphi + the sector's phimin."""
    layer = np.asarray(layer)
    kp = np.array([kphi(int(L), k) for L in range(1, 7)])[layer - 1]
    kzz = np.array([kz(L) for L in range(1, 7)])[layer - 1]
    rm = np.array([rmean(L) for L in range(1, 7)])[layer - 1]
    phimin = np.array([KF.sector_phimin(s, k) for s in range(KF.N_SECTOR)])[np.asarray(region)]
    phi = np.asarray(iphi) * kp + phimin
    return (phi + np.pi) % (2 * np.pi) - np.pi, np.asarray(iz) * kzz, rm + np.asarray(ir) * kr_barrel()
