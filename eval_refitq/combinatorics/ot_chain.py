"""Emulation of the prompt hybrid OT track finder (barrel seeds, barrel targets) from the DTC frames.

THE CHAIN, as CMSSW_20_1_0_pre1 runs it for L1TTrack, stage by stage, and what is emulated:

  inputs     L1TOTStubDTC (ot_dtc): every stub copy per processing region (nonant = sector) in stream
             order, with its integer phi / z / r and bend code. A stub in no stream was dropped by the
             DTC. The phi correction (Sector.cc:60-78, VMPhiCorrL<n>) is applied exactly.
  seeding    per sector and barrel seed (L1L2, L2L3, L3L4, L5L6): the VMRouter cuts on the inner stub
             (VMRouterCM.cc, integer |z|), the tracklet-engine stub-pT LUTs on the pair (TP_<seed>
             stubpt inner / outer tables, indexed by the fine-phi difference and the bend code), and the
             tracklet-calculator gates: |rinv| below the 1.91 GeV cut, |z0| < 15 cm (L1L2) or 22.5 cm,
             and inSector (the phicrit window at rcrit = 55 cm). EMULATED IN FLOAT: the gates use the
             exact two-stub helix (exacttracklet, the KF prior) instead of the integer calcPars; the
             VM r/z bin windows of the tracklet engine are not emulated (the z0 gate is their tighter
             successor).
  matching   per tracklet and target layer: the exact projection (phiproj, zproj and derivatives at the
             layer's rmean), the match engine's bend LUT (METable_L<n>, indexed by the projection's
             5-bit rinv and the stub bend code) and the match processor's integer residual cut
             (MP_L<n>PHIA phicut / zcut by seed) with its IMPROVING CHAIN (MatchProcessor.cc:684-714):
             a candidate is kept iff |dphi| <= best and -best <= dz < best, and then becomes the best.
             EMULATED: the projection is the float one floored to the stub's integer units (CMSSW's
             IMATH projection can differ by an LSB); the match engine's fine-bin windows are not
             emulated; candidates arrive in stream order (the `dtc` order mode) or in a seeded random
             order (`random`), where CMSSW orders them by VM bin and then memory order.
  builder    a tracklet with matches in >= 2 layers becomes a candidate; its stub list is the two seed
             stubs, then the matches by layer (FitTrack.cc:67-89). Two track builders per sector, seeds
             {L1L2, L2L3, L5L6, L2D1} and {L3L4, D1D2, D3D4, L1D1}, each processing tracklets in TCID
             order (seed, then tracklet order) up to 108 (caps on).
  fit        cmssw_kf: the exact port of the hybrid KF.
  no-DR      every candidate is fitted on its own list (RemovalType = "").
  merge DR   PurgeDuplicate (merge): per sector, the candidates of builder 0 then builder 1 up to 108,
             ordered by the static seed rank {1, 5, 2, 7, 4, 3, 8, 6}; CompareBest on the best stub per
             layer by |phi_stub - phi_proj(rmean)| (0 for the seed layers); duplicates when >= 3 layers
             share their best stub; only the first 32 unmerged candidates compare; the preferred
             candidate takes the rejected one's new stubs; each survivor is fitted once on its merged
             list with its own tracklet.

CAPS: TB 108, DR-input 108 and the 32 comparators are emulated here (switch `caps`). The DTC's are in
the table. Not emulated: IR 156, VMR 108 / 128, VM bins of 15, projection pages of 63, TP 108 steps /
127 tracklets, PC 108, MP 108 steps; the no-truncation sibling production measures them.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import cmssw_kf as KF
import ot_dtc as D

LUT_DIR = Path(__file__).resolve().parent / "luts"
SEEDS = {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (5, 6)}            # barrel prompt seeds (Seed enum order)
SEED_NAME = {0: "L1L2", 1: "L2L3", 2: "L3L4", 3: "L5L6"}
TARGETS = {0: (3, 4, 5, 6), 1: (1, 4, 5, 6), 2: (1, 2, 5, 6), 3: (1, 2, 3, 4)}
TB_OF_SEED = {0: 0, 1: 0, 3: 0, 7: 0, 2: 1, 4: 1, 5: 1, 6: 1}   # Settings tbseed_ {{0,1,3,7},{2,4,5,6}}
SEED_RANK = (1, 5, 2, 7, 4, 3, 8, 6)                           # PurgeDuplicate.cc ranks
TB_MAX, DR_MAX, DR_COMPARED, MIN_SHARE = 108, 108, 32, 3
NRINV = 5                                                       # NRINVBITS
NBITS_PHIDER = 10                                               # nbitsphiprojderL123 / L456
NBEND = {L: (3 if L <= 3 else 4) for L in range(1, 7)}
RINVCUT_PT, Z0CUT = 1.91, 15.0                                  # Settings ptcut_, z0cut_
RCRIT, DPHICRIT_MC = 55.0, 0.005
# VMRouterCM inner-stub |z| cuts in integer units (VMRouterCM.cc; tracklet_topology_cost OT_VMR_Z)
VMR_ZMAX = 95.0
VMR_ZMIN_L2 = 50.0
ZCUT_L2L3_LUT = 52.0                                            # TrackletLUT zcutL2L3


def _tab(name):
    t = (LUT_DIR / name).read_text().strip().strip("{};")
    return np.array([int(x) for x in t.replace("\n", "").split(",") if x.strip()], np.int64)


class Luts:
    """The firmware tables the chain reads (dumped from CMSSW, luts/PROVENANCE.md)."""

    def __init__(self):
        self.phicorr = {L: _tab(f"VMPhiCorrL{L}.tab") for L in range(1, 7)}
        self.me = {L: _tab(f"METable_L{L}.tab") for L in range(1, 7)}
        self.vmrme = {L: _tab(f"VMRME_L{L}.tab") for L in range(1, 7)}
        self.phicut = {L: _tab(f"MP_L{L}PHIA_phicut.tab") for L in range(1, 7)}
        self.zcut = {L: _tab(f"MP_L{L}PHIA_zcut.tab") for L in range(1, 7)}
        self.te = {}
        for sd, nm in SEED_NAME.items():
            ti = np.loadtxt(LUT_DIR / f"TP_{nm}A_stubptinnercut.dat", dtype=np.int64)
            to = np.loadtxt(LUT_DIR / f"TP_{nm}A_stubptoutercut.dat", dtype=np.int64)
            la = SEEDS[sd][0]
            nbd = int(round(math.log2(len(ti) >> NBEND[la])))
            self.te[sd] = (ti, to, nbd)
        self.tp_inner = {sd: _tab(f"TP_{nm}.tab") for sd, nm in SEED_NAME.items()}


_LUTS = None


def luts():
    global _LUTS
    if _LUTS is None:
        _LUTS = Luts()
    return _LUTS


# ---- inputs ------------------------------------------------------------------
@dataclass
class Copies:
    """The barrel stub copies of one event, one row per (stub, region) frame."""
    row: np.ndarray          # L1TOTStub row
    region: np.ndarray
    order: np.ndarray        # position in the finder's loop (region, channel, stream index)
    layer: np.ndarray        # 1..6
    ps: np.ndarray
    tilted: np.ndarray
    iphi: np.ndarray
    iz: np.ndarray
    ir: np.ndarray
    bend: np.ndarray
    phicorr: np.ndarray
    r: np.ndarray            # rmean + ir krbarrel (the integer r, physical)
    x: np.ndarray            # float stub position (the KF's input)
    y: np.ndarray
    z: np.ndarray


def event_copies(stub, dtc):
    """stub: dict of L1TOTStub numpy columns for one event; dtc: dict of L1TOTStubDTC columns."""
    row = np.asarray(dtc["otStubIdx"], np.int64)
    bar = np.asarray(stub["isBarrel"], bool)[row]
    keep = bar
    row = row[keep]
    region = np.asarray(dtc["region"], np.int64)[keep]
    order = np.arange(len(keep))[keep]
    layer = np.asarray(stub["layer"], np.int64)[row]
    ps = np.asarray(stub["isPS"], bool)[row]
    mt = D.module_type(np.ones(len(row), bool), ps)
    _, bend, _, iphi, iz, ir, _ = D.decode(np.asarray(dtc["frameHi"])[keep], np.asarray(dtc["frameLo"])[keep], mt)
    L = luts()
    rbin = (ir + (1 << 6)) >> 4                                  # (r + 2^(nrbits-1)) >> (nrbits - 3)
    corr = np.array([L.phicorr[int(l)][int(b) * 8 + int(rb)] for l, b, rb in zip(layer, bend, rbin)], np.int64)
    nphib = np.array(D.NPHIBITS_L, np.int64)[layer - 1]
    phicorr = np.clip(iphi - corr, 0, (np.int64(1) << nphib) - 1)
    rm = np.array([D.rmean(l) for l in range(1, 7)])[layer - 1]
    return Copies(row, region, order, layer, ps, np.asarray(stub["isTilted"], bool)[row], iphi, iz, ir, bend,
                  phicorr, rm + ir * D.kr_barrel(),
                  np.asarray(stub["x"], np.float64)[row], np.asarray(stub["y"], np.float64)[row],
                  np.asarray(stub["z"], np.float64)[row])


# ---- tracklets -----------------------------------------------------------------
@dataclass
class Tracklet:
    sector: int
    seed: int
    inner: int               # index into Copies
    outer: int
    rinv: float
    phi0: float              # relative to the sector's phimin
    t: float
    z0: float
    tcid: tuple = ()
    matches: dict = field(default_factory=dict)   # layer -> list of Copies indices (improving-chain order)


def _inner_ok(cp, idx, seed):
    L = SEEDS[seed][0]
    absz = np.abs(cp.iz[idx])
    kzl = D.kz(L)
    ok = absz <= VMR_ZMAX / kzl
    if L == 2:
        ok = (absz >= VMR_ZMIN_L2 / kzl) & (np.abs(cp.z[idx]) >= ZCUT_L2L3_LUT)
    return ok


def _float_geom(cp, i):
    return math.hypot(cp.x[i], cp.y[i]), float(cp.z[i]), math.atan2(cp.y[i], cp.x[i])


def _vmr_lut_index(cp, idx, layer):
    """The (z, r) index of the VMRouter / TrackletProcessor tables: the top 7 bits of z and 4 of r."""
    nz, nr = D.NZBITS_L[layer - 1], D.NRBITS_L[layer - 1]
    iz = (cp.iz[idx] >> (nz - VMR_LUT_ZBITS)) & ((1 << VMR_LUT_ZBITS) - 1)
    ir = (cp.ir[idx] >> (nr - VMR_LUT_RBITS)) & ((1 << VMR_LUT_RBITS) - 1)
    return (iz << VMR_LUT_RBITS) + ir


def te_rz_window(cp, inn, out, seed):
    """The tracklet engine's r-z window (TrackletProcessor.cc:362-395, TrackletEngineUnit::step): the inner
    stub's TP_<seed> table entry gives the outer layer's first rz bin, whether the next one is used, the
    first fine bin and the allowed spread; the outer stub's bin and fine position are its VMRME entry."""
    la, lb = SEEDS[seed]
    lut = luts().tp_inner[seed][_vmr_lut_index(cp, inn, la)]
    first = lut & ((1 << NFINERZ) - 1)
    nxt = (lut >> NFINERZ) & 1
    start = (lut >> (NFINERZ + 1)) & ((1 << ME_BINS_BITS) - 1)
    diffmax = (lut >> (NFINERZ + 1 + ME_BINS_BITS)) & ((1 << NFINERZ) - 1)
    melut = luts().vmrme[lb][_vmr_lut_index(cp, out, lb)]
    obin, ofine = melut >> NFINERZ, melut & ((1 << NFINERZ) - 1)
    in_first = obin[None, :] == start[:, None]
    in_next = (nxt[:, None] == 1) & (obin[None, :] == start[:, None] + 1)
    rz = np.where(in_next, ofine[None, :] + (1 << NFINERZ), ofine[None, :])
    return (in_first | in_next) & (rz >= first[:, None]) & (rz - first[:, None] <= diffmax[:, None])


def seed_sector(cp, sector, seed, k=KF.K):
    """Tracklets of one barrel seed in one sector (TE LUT + TC gates; module docstring)."""
    la, lb = SEEDS[seed]
    inn = np.nonzero((cp.region == sector) & (cp.layer == la))[0]
    out = np.nonzero((cp.region == sector) & (cp.layer == lb))[0]
    inn = inn[_inner_ok(cp, inn, seed)]
    if len(inn) == 0 or len(out) == 0:
        return []
    ti, to, nbd = luts().te[seed]
    rzok = te_rz_window(cp, inn, out, seed)
    fa = cp.phicorr[inn] >> (D.NPHIBITS_L[la - 1] - 8)
    fb = cp.phicorr[out] >> (D.NPHIBITS_L[lb - 1] - 8)
    d = fb[None, :] - fa[:, None]
    inr = (d < (1 << (nbd - 1))) & (d >= -(1 << (nbd - 1)))
    dm = d & ((1 << nbd) - 1)
    ii = (dm << NBEND[la]) + cp.bend[inn][:, None]
    io = (dm << NBEND[lb]) + cp.bend[out][None, :]
    ok = inr & (ii < len(ti)) & (io < len(to))
    acc = np.zeros(d.shape, bool)
    acc[ok] = (ti[ii[ok]] > 0) & (to[io[ok]] > 0)
    acc &= rzok
    rinvcut = 0.01 * KF.C_TRK * k.b / RINVCUT_PT
    z0cut = Z0CUT if seed == 0 else 1.5 * Z0CUT
    phicrit_min = 0.5 * k.dphisector_hg - math.pi / KF.N_SECTOR - DPHICRIT_MC
    phicrit_max = 0.5 * k.dphisector_hg + math.pi / KF.N_SECTOR + DPHICRIT_MC
    res = []
    for a, b in zip(*np.nonzero(acc)):
        i, j = int(inn[a]), int(out[b])
        r1, z1, p1 = _float_geom(cp, i)
        r2, z2, p2 = _float_geom(cp, j)
        rinv, phi0, t, z0 = KF.exact_tracklet(r1, z1, p1, r2, z2, p2, sector, k)
        if not (abs(rinv) < rinvcut and abs(z0) < z0cut):
            continue
        phicrit = phi0 - math.asin(0.5 * RCRIT * rinv)
        if not (phi0 >= 0 and phicrit_min < phicrit < phicrit_max):
            continue
        res.append(Tracklet(sector, seed, i, j, rinv, phi0, t, z0,
                            tcid=(seed, int(cp.order[i]), int(cp.order[j]))))
    return res


# ---- projections and matching ------------------------------------------------
def projection(tr, layer):
    """Exact projection to the layer's rmean: (phiproj rel. phimin, zproj, phider, zder) or None."""
    rp = D.rmean(layer)
    s = 0.5 * rp * tr.rinv
    if abs(s) >= 1:
        return None
    root = math.sqrt(1 - s * s)
    phiproj = tr.phi0 - math.asin(s)
    zproj = tr.z0 + tr.t * 2 * math.asin(s) / tr.rinv
    return phiproj, zproj, -0.5 * tr.rinv / root, tr.t / root


NBITS_ALLSTUBS = (3, 2, 2, 2, 2, 2)      # Settings nbitsallstubs_ (barrel)
NBITS_VMME = (2, 3, 3, 3, 3, 3)          # Settings nbitsvmme_ (barrel)
NFINEPHI, NFINERZ, ME_BINS_BITS = 3, 3, 3
VMR_LUT_ZBITS, VMR_LUT_RBITS = 7, 4      # Settings vmrlutzbits_ / vmrlutrbits_ (barrel)


def me_candidates(cp, cand, layer, iphiproj, izproj, ps_seed):
    """The match engine's view (MatchProcessor.cc:436-545, MatchEngineUnit.cc:100-200): only stubs of the
    projection's allstub phi region, in the projection's VM bin or the neighbour it looks into near a
    bin edge, within 3 fine-phi units and the fine-z window (1 for a PS seed, 5 otherwise), in the order
    the engine reads them: VM bin (minus, then plus), z slot (first, then second), then memory order."""
    nb, nz = D.NPHIBITS_L[layer - 1], D.NZBITS_L[layer - 1]
    nall, nvm = NBITS_ALLSTUBS[layer - 1], NBITS_VMME[layer - 1]
    nvmtot = nall + nvm
    nvmbins = 1 << nvm
    reg_p = iphiproj >> (nb - nall)
    vm_p = (iphiproj >> (nb - nvmtot)) & (nvmbins - 1)
    extra = (iphiproj >> (nb - nvmtot - 2)) & 3
    fine_p = (iphiproj >> (nb - nvmtot - NFINEPHI)) & ((1 << NFINEPHI) - 1)
    vm_plus, vm_minus = vm_p, vm_p
    if extra == 3 and vm_p != nvmbins - 1:
        vm_plus = vm_p + 1
    if extra == 0 and vm_p != 0:
        vm_minus = vm_p - 1
    # z slots (Projection.cc:58-84): the projection's first bin, a second one when it straddles
    offset = 1 if ps_seed else 4
    ztemp = izproj >> (nz - ME_BINS_BITS - NFINERZ)
    zbin1 = (1 << (ME_BINS_BITS - 1)) + ((ztemp - offset) >> NFINERZ)
    zbin2 = (1 << (ME_BINS_BITS - 1)) + ((ztemp + offset) >> NFINERZ)
    zbin1 = 0 if zbin1 >= (1 << ME_BINS_BITS) or zbin1 < 0 else zbin1
    zbin2 = min(zbin2, (1 << ME_BINS_BITS) - 1)
    finez_p = ((1 << (ME_BINS_BITS + NFINERZ - 1)) + ztemp) - (zbin1 << NFINERZ)
    pc = cp.phicorr[cand]
    reg_s = pc >> (nb - nall)
    vm_s = (pc >> (nb - nvmtot)) & (nvmbins - 1)
    fine_s = (pc >> (nb - nvmtot - NFINEPHI)) & ((1 << NFINEPHI) - 1)
    # the stub's rz bin and fine z: VMStubMERouter.cc, the VMRME_L<n> table of z projected to rmean
    nr = D.NRBITS_L[layer - 1]
    indexz = (cp.iz[cand] >> (nz - VMR_LUT_ZBITS)) & ((1 << VMR_LUT_ZBITS) - 1)
    indexr = (cp.ir[cand] >> (nr - VMR_LUT_RBITS)) & ((1 << VMR_LUT_RBITS) - 1)
    melut = luts().vmrme[layer][(indexz << VMR_LUT_RBITS) + indexr]
    zbin_s, finez_s = melut >> NFINERZ, melut & ((1 << NFINERZ) - 1)
    dphi = (vm_s * (1 << NFINEPHI) + fine_s) - (vm_p * (1 << NFINEPHI) + fine_p)
    slot = np.where(zbin_s == zbin1, 0, np.where((zbin_s == zbin1 + 1) & (zbin2 != zbin1), 1, -1))
    drz = finez_s + slot * (1 << NFINERZ) - finez_p
    drz_cut = 1 if ps_seed else 5
    ok = ((reg_s == reg_p) & ((vm_s == vm_minus) | (vm_s == vm_plus)) & (np.abs(dphi) < 3) & (slot >= 0)
          & (np.abs(drz) <= drz_cut))
    key = np.lexsort((np.arange(len(cand)), slot, (vm_s != vm_minus).astype(int)))
    return [int(cand[i]) for i in key if ok[i]]


def match_tracklet(cp, tr, layer, cand, k=KF.K, rng=None):
    """The improving chain of one (tracklet, layer) over the candidate copies `cand` (array of indices,
    memory order), after the match engine's selection (me_candidates)."""
    pj = projection(tr, layer)
    if pj is None:
        return []
    phiproj, zproj, phider, zder = pj
    if not (abs(zproj) < D.ZLENGTH and 0 <= phiproj < k.dphisector_hg):
        return []
    L = luts()
    kp, kzl = D.kphi(layer, k), D.kz(layer)
    iphiproj = min(math.floor(phiproj / kp), (1 << D.NPHIBITS_L[layer - 1]) - 1)
    izproj = math.floor(zproj / kzl)
    cand = np.asarray(cand)
    if rng is not None:
        cand = rng.permutation(cand)
    cand = me_candidates(cp, cand, layer, iphiproj, izproj, ps_seed=tr.seed in (0, 1))   # Tracklet::PSseed: inner L1 or L2
    phishift = KF.NPHIBITS_L6 - D.NPHIBITS_L[layer - 1]
    dzshift = D.NZBITS_L[0] - D.NZBITS_L[layer - 1]
    cut_phi = int(L.phicut[layer][tr.seed])
    cut_z = int(L.zcut[layer][tr.seed])
    if cut_phi <= 0 or cut_z <= 0:
        return []
    kphider = k.dphisector_hg / (1 << D.NPHIBITS_L[0]) / (D.RMAXDISK / (1 << 12)) / 256
    iphider = int(np.clip(math.floor(phider / kphider), -(1 << (NBITS_PHIDER - 1)), (1 << (NBITS_PHIDER - 1)) - 1))
    projrinv = (1 << (NRINV - 1)) - 1 - (iphider >> (NBITS_PHIDER - NRINV))
    me = L.me[layer]
    best_phi, best_z = cut_phi, cut_z
    kept = []
    for c in cand:
        if not me[(projrinv << NBEND[layer]) + int(cp.bend[c])]:
            continue
        dr = cp.r[c] - D.rmean(layer)
        iphi_proj = math.floor((phiproj + dr * phider) / kp)
        iz_proj = math.floor((zproj + dr * zder) / kzl + 0.5)
        dphi = (int(cp.iphi[c]) - iphi_proj) << phishift
        dz = (int(cp.iz[c]) - iz_proj) << dzshift
        if abs(dphi) <= best_phi and -best_z <= dz < best_z:
            best_phi, best_z = abs(dphi), abs(dz)
            kept.append(int(c))
    return kept


# ---- candidates, fit, duplicate removal -------------------------------------------
@dataclass
class Candidate:
    tracklet: Tracklet
    stubs: list              # Copies indices: seed stubs, then matches by layer
    tb: int


@dataclass
class Track:
    sector: int
    seed: int
    rows: tuple              # L1TOTStub rows of the fitted stubs
    out: object              # cmssw_kf.FitOutput
    input_rows: tuple        # the candidate's input stub rows


def kf_stub(cp, c):
    return KF.make_stub(int(cp.row[c]), cp.x[c], cp.y[c], cp.z[c], int(cp.layer[c]), bool(cp.ps[c]),
                        bool(cp.tilted[c]))


def fit(cp, tr, stub_idx, k=KF.K):
    stubs = [kf_stub(cp, c) for c in stub_idx]
    seed = KF.make_seed(tr.rinv, tr.phi0, tr.t, tr.z0, tr.sector, len(stubs), k)
    state = KF.do_kf(seed, stubs)
    out = KF.fitted_output(state, seed, k)
    return out


def build_candidates(cp, sector, tracklets, caps=True):
    """FitTrack: tracklets with matches in >= 2 layers, per track builder in TCID order (cap 108)."""
    cands = []
    count = {0: 0, 1: 0}
    for tr in sorted(tracklets, key=lambda q: q.tcid):
        if not tr.matches:
            continue
        tb = TB_OF_SEED[tr.seed]
        if caps and count[tb] >= TB_MAX:
            continue
        count[tb] += 1
        if len([L for L, v in tr.matches.items() if v]) < 2:
            continue
        stubs = [tr.inner, tr.outer] + [c for L in sorted(tr.matches) for c in tr.matches[L]]
        cands.append(Candidate(tr, stubs, tb))
    return cands


def phi_res(cp, tr, c, k=KF.K):
    """PurgeDuplicate::getPhiRes: 0 for a seed-layer stub, else |deltaPhi(stub phi, phiproj + phimin)|."""
    L = int(cp.layer[c])
    if L in SEEDS[tr.seed]:
        return 0.0
    pj = projection(tr, L)
    phiproj = pj[0] + (tr.sector * k.dphisector - 0.5 * k.dphisector_hg)
    return abs(KF.delta_phi(math.atan2(cp.y[c], cp.x[c]), phiproj))


def merge_dr(cp, cands, caps=True, k=KF.K):
    """PurgeDuplicate (merge) for one sector; returns the surviving (tracklet, merged stub list)."""
    inp = [c for tb in (0, 1) for c in cands if c.tb == tb]
    if caps:
        inp = inp[:DR_MAX]
    n = len(inp)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (SEED_RANK[inp[i].tracklet.seed], i))
    srt = [inp[i] for i in order]
    ids = [[(int(cp.layer[c]), int(cp.row[c])) for c in s.stubs] for s in srt]
    lists = [list(s.stubs) for s in srt]
    best = []
    for s in srt:
        b = {}
        for pos, c in enumerate(s.stubs):
            L = int(cp.layer[c])
            if L not in b:
                b[L] = pos
            elif phi_res(cp, s.tracklet, c, k) < phi_res(cp, s.tracklet, s.stubs[b[L]], k):
                b[L] = pos
        best.append(b)
    dup = np.zeros((n, n), bool)
    merged = np.zeros(n, bool)
    cm = 0
    for i in range(n - 1):
        if merged[i]:
            continue
        cm += 1
        if caps and cm > DR_COMPARED:
            continue
        for j in range(i + 1, n):
            share = sum(1 for L, pi in best[i].items() if L in best[j] and ids[i][pi] == ids[j][best[j][L]])
            if share >= MIN_SHARE:
                dup[i, j] = dup[j, i] = True
                merged[j] = True
    rejected = np.zeros(n, bool)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if dup[i, j]:
                have = {(int(cp.layer[c]), int(cp.row[c])) for c in lists[i]}
                lists[i] = lists[i] + [c for c in lists[j] if (int(cp.layer[c]), int(cp.row[c])) not in have]
                rejected[j] = True
    return [(srt[i].tracklet, lists[i]) for i in range(n) if not rejected[i]]


def run_event(stub, dtc, caps=True, order="dtc", seed=0, k=KF.K, do_dr=True, collect=None):
    """The chain on one event. Returns (no-DR tracks, DR tracks), each a list of Track.
    collect: an optional dict that receives the event's Copies ("copies") and tracklets ("tracklets")."""
    cp = event_copies(stub, dtc)
    if collect is not None:
        collect["copies"] = cp
    rng = None if order == "dtc" else np.random.default_rng(seed)
    nodr, dr = [], []
    for sector in range(KF.N_SECTOR):
        trs = []
        for sd in SEEDS:
            trs += seed_sector(cp, sector, sd, k)
        for tr in trs:
            for L in TARGETS[tr.seed]:
                cand = np.nonzero((cp.region == sector) & (cp.layer == L))[0]
                kept = match_tracklet(cp, tr, L, cand, k, rng)
                if kept:
                    tr.matches[L] = kept
        if collect is not None:
            collect.setdefault("tracklets", []).extend(trs)
        cands = build_candidates(cp, sector, trs, caps)
        for c in cands:
            out = fit(cp, c.tracklet, c.stubs, k)
            if out.accepted and out.consistent_sector:
                nodr.append(Track(sector, c.tracklet.seed, tuple(s.idx for s in out.stubs), out,
                                  tuple(int(cp.row[x]) for x in c.stubs)))
        if do_dr:
            for tr, lst in merge_dr(cp, cands, caps, k):
                out = fit(cp, tr, lst, k)
                if out.accepted and out.consistent_sector:
                    dr.append(Track(sector, tr.seed, tuple(s.idx for s in out.stubs), out,
                                    tuple(int(cp.row[x]) for x in lst)))
    return nodr, dr
