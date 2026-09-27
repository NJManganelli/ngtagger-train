"""Shared pieces of the CMSSW port of the SmartPixels-output angle emulator (angleSource = "emu").

The CMSSW producer (L1Trigger/Phase3SmartPixels SmartPixelsRecHitProducer, angleSource "emu") and the python
reference (emu_reference.py) implement the SAME procedure; this module holds the python side of every piece that must
agree bit-for-bit: the pos-hubsig network + its scalar inputs, the cluster hash, the exact k-NN ordering, the response.

Procedure (approved design, v4 angle space): pseudo-truth angle [deg] = TP helix angle (class T: charged dominant TP,
pT >= 0.25, helix within 1 cm of the hit) or CNN + donor residual (class C, one of the 10 nearest class-T donors per
surface in 11 standardised observables); saturate at +-(90 - 0.01). Response: the 32 nearest dump rows in (atan alpha,
atan beta) [deg] of the pseudo-truth (discs: alpha key = atan(cotAlpha_pseudo - 0.312)), bias = median r, one row r
drawn by the cluster hash, the SAME row for alpha and beta; out = pseudo + bias + 1.5 (r - bias), saturated, cot = tan.
"""
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(4)

WX, WY = 24, 16
K_INFL = 1.5
C_LOR = -0.312
EPS_DEG = 0.01
SAT = 90.0 - EPS_DEG
KNN_DUMP, KNN_DONOR = 32, 10
PT_MIN, HXDIST_MAX = 0.25, 1.0
SALT_DONOR = 0xD0A0D0A0D0A0D0A1
SALT_DUMP = 0x5A7E11175A7E1117
DONOR_OBS = ["cnnA", "cnnB", "sigA", "sigB", "sizeX", "sizeY", "logq", "r", "absz", "localX", "localY"]


class V2Net(nn.Module):
    """pos-hubsig network, identical to v2_train.Net (inputs ADC/16 + occupancy + coordinates; no canon flip)."""
    def __init__(self, ns, nout=6):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.conv = nn.Sequential(blk(4, 32), blk(32, 32), blk(32, 64, 2), blk(64, 64), blk(64, 64, 2))
        gx, gy = torch.meshgrid(torch.linspace(-1, 1, WX), torch.linspace(-1, 1, WY), indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy])[None])
        self.register_buffer("qlut", torch.zeros(17))          # unused (no qenc); keeps the state_dict loadable
        nflat = 64 * ((WX + 3) // 4) * ((WY + 3) // 4)
        self.head = nn.Sequential(nn.Linear(nflat + ns, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, nout))

    def forward(self, x, s):
        x = x.float(); adc = x / 16.0; occ = (x > 0).float()
        z = torch.cat([adc[:, None], occ[:, None], self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        return self.head(torch.cat([self.conv(z).flatten(1), s], 1))


def load_net(mdir):
    import json, os
    nd = json.load(open(os.path.join(mdir, "norm.json")))
    net = V2Net(len(nd["cols"]))
    sd = torch.load(os.path.join(mdir, "model.pt"), map_location="cpu")
    sd.setdefault("qlut", torch.zeros(17))
    net.load_state_dict(sd); net.eval()
    return net, nd


def norm_arrays(nd):
    cols = nd["cols"]
    return np.array([nd["mu"][c] for c in cols], np.float32), np.array([nd["sd"][c] for c in cols], np.float32)


def normalise(S, nd):
    """float32, the arithmetic of the exported graph. NB some inputs are geometry rounding residues (training sd ~1e-16,
    e.g. barrel axXz/axZz/axYr): the network uses them (freezing them moves barrel beta by a median 1.4 deg, up to 30 deg,
    emu_cmssw/check_frozen.json), so the producer must reproduce the raw values bit-exactly."""
    mu, sd = norm_arrays(nd)
    return ((np.asarray(S, np.float32) - mu) / sd).astype(np.float32)


def decode(o):
    """raw network output -> CNN angles and sigmas [deg] (hubsig head, atan target)"""
    o = np.asarray(o, np.float64)
    aA = np.degrees(np.clip(o[:, 0], -1.5707, 1.5707)); aB = np.degrees(np.clip(o[:, 1], -1.5707, 1.5707))
    s = np.degrees(np.exp(np.clip(o[:, 2:4], -7, 3)))
    return aA, aB, s[:, 0], s[:, 1]


def run_net(net, img, Z, bs=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(img), bs):
            out.append(net(torch.as_tensor(np.ascontiguousarray(img[i:i + bs])), torch.as_tensor(Z[i:i + bs])).numpy())
    return np.concatenate(out) if out else np.zeros((0, 6), np.float32)


# ---------------- hash (MUST match SmartPixelsRecHitProducer emuHashIndex) ----------------
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_index(run, event, detId, idxInDet, salt, k):
    """index in [0, k) from the stable cluster key (run, event, detId, idxInDet) and a salt:
    h = mix(((detId << 16) | idxInDet) + golden); h = mix(h ^ event); h = mix(h ^ run); h = mix(h ^ salt);
    index = floor(((h >> 11) * 2^-53) * k)   [mix = the splitmix64 finaliser]"""
    with np.errstate(over="ignore"):
        u64 = lambda v: np.asarray(v).astype(np.uint64)
        h = _mix(((u64(detId) << np.uint64(16)) | u64(idxInDet)) + np.uint64(0x9E3779B97F4A7C15))
        h = _mix(h ^ u64(event)); h = _mix(h ^ u64(run)); h = _mix(h ^ np.uint64(salt))
    u = (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return np.minimum((u * k).astype(np.int64), k - 1)


# ---------------- exact k-NN, deterministic order (squared distance, then library index) ----------------
def knn_exact(lib, q, k, pre=4):
    """lib [N, D] float64, q [M, D] float64 -> idx [M, k] sorted by (d2, index). A cKDTree proposes k + pre candidates;
    d2 is recomputed with a sequential per-dimension sum (the C++ order) and the k best re-sorted; a query whose
    (k + pre)-th candidate is not strictly farther than the k-th is redone by brute force."""
    from scipy.spatial import cKDTree
    tree = cKDTree(lib)
    kk = min(k + pre, len(lib))
    _, cand = tree.query(q, k=kk, workers=4)
    d2 = np.zeros(cand.shape)
    for j in range(lib.shape[1]):
        d2 += (q[:, None, j] - lib[cand, j]) ** 2
    order = np.lexsort((cand, d2), axis=1)
    cand = np.take_along_axis(cand, order, 1); d2 = np.take_along_axis(d2, order, 1)
    bad = np.where(d2[:, -1] <= d2[:, k - 1])[0] if kk > k else np.array([], int)
    for i in bad:
        dd = np.zeros(len(lib))
        for j in range(lib.shape[1]):
            dd += (q[i, j] - lib[:, j]) ** 2
        o = np.lexsort((np.arange(len(lib)), dd))[:k]
        cand[i, :k] = o
    return cand[:, :k]


sat = lambda a: np.clip(a, -SAT, SAT)


def scalars_v2(m, region, cols):
    """raw pos-hubsig scalar inputs, exactly v2_train.feats (--inputs pos) on a meta frame carrying the ntuple columns
    (derived() columns are recomputed here if absent)."""
    import pandas as pd
    from models_def import scalars as base_scalars
    from v2_common import derived
    if "pu" not in m:
        m = pd.concat([m.reset_index(drop=True), derived(m.reset_index(drop=True))], axis=1)
    s = base_scalars(m, "base" if region == "barrel" else "disc")
    s["is3D"] = ((m.subdet == 1) & (m.layer == 1)).values.astype(np.float32)
    for c in ["axXz", "axYz", "axZz", "axXr", "axYr", "axZr", "sideZ", "pu", "pv", "pw", "globalR", "globalZ", "localX", "localY"]:
        s[c] = m[c].values
    s = s.astype(np.float32)
    assert list(s.columns) == list(cols), (list(s.columns), cols)
    return s.values


# ---------------- the library file (emu_build_library.py) ----------------
def load_library(path):
    import uproot
    f = uproot.open(path)
    dump = {k: f["dump"][k].array(library="np") for k in ("tA", "tB", "rA", "rB")}
    don = {k: f["donors"][k].array(library="np") for k in ["surface"] + DONOR_OBS + ["resA", "resB"]}
    nrm = {k: f["norm"][k].array(library="np") for k in f["norm"].keys()}
    lib = {"dump_pts": np.c_[dump["tA"].astype(np.float64), dump["tB"].astype(np.float64)],
           "rA": dump["rA"].astype(np.float64), "rB": dump["rB"].astype(np.float64), "donors": {}}
    for i, s in enumerate(nrm["surface"]):
        sel = don["surface"] == s
        mu = np.array([nrm[f"mu_{c}"][i] for c in DONOR_OBS]); sd = np.array([nrm[f"sd_{c}"][i] for c in DONOR_OBS])
        X = np.stack([don[c][sel].astype(np.float64) for c in DONOR_OBS], 1)
        lib["donors"][int(s)] = dict(mu=mu, sd=sd, Z=(X - mu) / sd, resA=don["resA"][sel].astype(np.float64),
                                     resB=don["resB"][sel].astype(np.float64))
    return lib


def emulate(lib, surface, classT, trueCotA, trueCotB, cnnA, cnnB, obs, run, event, detId, idxInDet):
    """The emu procedure on arrays (one entry per cluster). obs: [n, 11] donor observables (DONOR_OBS order; the CNN
    ones = cnnA, cnnB, sigA, sigB). Returns a dict of per-cluster arrays (pseudo, donor, dump row, biases, outputs,
    saturation flags, cot outputs)."""
    n = len(surface)
    pA = np.where(classT, np.degrees(np.arctan(np.asarray(trueCotA, np.float64))), np.nan)
    pB = np.where(classT, np.degrees(np.arctan(np.asarray(trueCotB, np.float64))), np.nan)
    donor = np.full(n, -1, np.int64); psat = np.zeros(n, bool)
    for s in np.unique(surface[~classT]):
        sel = np.where((surface == s) & ~classT)[0]
        L = lib["donors"][int(s)]
        Zq = (np.asarray(obs[sel], np.float64) - L["mu"]) / L["sd"]
        nb = knn_exact(L["Z"], Zq, KNN_DONOR)
        j = hash_index(run[sel], event[sel], detId[sel], idxInDet[sel], SALT_DONOR, KNN_DONOR)
        d = nb[np.arange(len(sel)), j]; donor[sel] = d
        rawA = cnnA[sel] + L["resA"][d]; rawB = cnnB[sel] + L["resB"][d]
        psat[sel] = (np.abs(rawA) >= SAT) | (np.abs(rawB) >= SAT)
        pA[sel] = sat(rawA); pB[sel] = sat(rawB)
    disc = surface > 10
    qA = np.where(disc, np.degrees(np.arctan(np.tan(np.radians(pA)) + C_LOR)), pA)
    nb = knn_exact(lib["dump_pts"], np.c_[qA, pB], KNN_DUMP)
    j = hash_index(run, event, detId, idxInDet, SALT_DUMP, KNN_DUMP)
    row = nb[np.arange(n), j]
    bA = np.median(lib["rA"][nb], 1); bB = np.median(lib["rB"][nb], 1)
    rawA = (pA + bA) + K_INFL * (lib["rA"][row] - bA); rawB = (pB + bB) + K_INFL * (lib["rB"][row] - bB)
    oA, oB = sat(rawA), sat(rawB)
    return dict(pseudoA=pA, pseudoB=pB, donor=donor, pseudoSat=psat, dumpRow=row, biasA=bA, biasB=bB,
                outA=oA, outB=oB, satA=np.abs(rawA) >= SAT, satB=np.abs(rawB) >= SAT,
                cotA=np.tan(np.radians(oA)).astype(np.float32), cotB=np.tan(np.radians(oB)).astype(np.float32))
