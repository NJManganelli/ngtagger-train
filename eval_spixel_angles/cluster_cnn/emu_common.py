"""Python reference of the CMSSW SmartPixels-output angle emulator (angleSource = "emu").

The CMSSW producer (L1Trigger/Phase3SmartPixels SmartPixelsRecHitProducer, angleSource "emu") and this module implement
the SAME procedure; everything that must agree between the two lives here: the class-T definition, the tiny CNN and its
scalar inputs, the A1 residual model and its inputs, the cluster hash, the exact dump k-NN ordering and the response.

Procedure (approved design: v4 angle space, fix round 2026-09-28):
  CLASS T: the dominant TP is linked and charged and its helix crosses the module plane within 1 cm of the hit (NO pT
           cut). pseudo = atan(helix-propagated true cot) [deg].
  CLASS C: everything else. pseudo = tiny-CNN angle + a residual r DRAWN from the A1 model p(rA | z) p(rB | z, rA)
           (autoregressive 156-bin histogram, fast_models.EDGES), z = the CNN angles and sigmas + 7 cluster observables +
           the surface one-hot, with 4 hash uniforms (SALTS_A1); saturated at +-(90 - 0.01).
  RESPONSE: the 32 nearest dump rows in (atan alpha, atan beta) [deg] of the pseudo-truth (discs: alpha key =
           atan(cotAlpha_pseudo - 0.312)), bias = median r, one row drawn by the cluster hash (SALT_DUMP), the SAME row for
           alpha and beta; out = pseudo + bias + 1.5 (r - bias), saturated at +-(90 - 0.01), never wrapped; cot = tan.
Models: trained/tiny_{barrel,disc} (the tiny CNNs, CNN_DIR) and emu_cmssw/a1_{barrel,disc}.pt (A1, emu_train_a1.py).
"""
import json, os
import numpy as np
import torch

torch.set_num_threads(4)

WX, WY = 24, 16
K_INFL = 1.5
C_LOR = -0.312
EPS_DEG = 0.01
SAT = 90.0 - EPS_DEG
KNN_DUMP = 32
HXDIST_MAX = 1.0
SALT_DUMP = 0x5A7E11175A7E1117
SALTS_A1 = (0xA2A2A2A2A2A2A201, 0xA2A2A2A2A2A2A202, 0xA2A2A2A2A2A2A203, 0xA2A2A2A2A2A2A204)
# A1 inputs, in graph order: the CNN outputs [deg], 7 cluster observables, then the surface one-hot (4 per region)
A1_OBS = ["cnnA", "cnnB", "sigA", "sigB", "sizeX", "sizeY", "logq", "r", "absz", "localX", "localY"]
N_A1_IN = len(A1_OBS) + 4
REGIONS = {"barrel": 0, "disc": 10}          # surface code = offset + layer | disk
# the tiny CNNs = the fast-study barrel_tiny / disc_tiny weights (fast_train.py), exported by emu_export_onnx.py
CNN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained", "tiny_{reg}")

sat = lambda a: np.clip(a, -SAT, SAT)


def class_t(linked, charge, hxOk, hxDist):
    """class T: linked to a TP, charged, helix crossing within HXDIST_MAX of the hit (no pT cut)"""
    return (np.asarray(linked) & (np.asarray(charge) != 0) & (np.asarray(hxOk) == 1)
            & (np.asarray(hxDist) < HXDIST_MAX))


# ---------------- tiny CNN ----------------
def load_cnn(mdir):
    from fast_models import FastNet
    nd = json.load(open(os.path.join(mdir, "norm.json")))
    net = FastNet(nd["arch"], len(nd["cols"]), nd["wx"], nd["wy"])
    net.load_state_dict(torch.load(os.path.join(mdir, "model.pt"), map_location="cpu")); net.eval()
    return net, nd


def cnn_scalars(m, region, nd):
    """RAW scalar inputs of the exported CNN graph, float32, in norm.json 'cols' order. The orientation columns are
    rounded to 1e-6 (float64 rint, as numpy.round) -- the producer does the same before the graph."""
    from fast_models import raw_scalars, ORIENT
    S = raw_scalars(m, region)
    for c in ORIENT:
        if c in S:
            S[c] = S[c].round(6)
    return S[nd["cols"]].values.astype(np.float32)


def run_cnn(net, nd, img, S, bs=8192):
    """the exported graph's arithmetic: (s - mu) / sd in float32, then the point head"""
    mu = torch.tensor([nd["mu"][c] for c in nd["cols"]], dtype=torch.float32)
    sd = torch.tensor([nd["sd"][c] for c in nd["cols"]], dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(img), bs):
            s = (torch.as_tensor(np.ascontiguousarray(S[i:i + bs])) - mu) / sd
            out.append(net(torch.as_tensor(np.ascontiguousarray(img[i:i + bs])), s).numpy())
    return np.concatenate(out) if out else np.zeros((0, 6), np.float32)


def decode(o):
    """raw network output -> CNN angles and sigmas [deg] (atan target, log sigma head)"""
    o = np.asarray(o, np.float64)
    aA = np.degrees(np.clip(o[:, 0], -1.5707, 1.5707)); aB = np.degrees(np.clip(o[:, 1], -1.5707, 1.5707))
    s = np.degrees(np.exp(np.clip(o[:, 2:4], -7, 3)))
    return aA, aB, s[:, 0], s[:, 1]


# ---------------- A1 residual model ----------------
def a1_inputs(cnnA, cnnB, sigA, sigB, sizeX, sizeY, charge, r, absz, localX, localY, layer_or_disk):
    """RAW A1 inputs [n, 15] (the graph standardises): logq = log(max(charge, 1)) in float32 as the producer"""
    logq = np.log(np.maximum(np.asarray(charge, np.float32), np.float32(1)))
    oh = np.stack([np.asarray(layer_or_disk) == k for k in (1, 2, 3, 4)], 1)
    return np.c_[cnnA, cnnB, sigA, sigB, sizeX, sizeY, logq, r, absz, localX, localY, oh].astype(np.float32)


def load_a1(path):
    """A1 checkpoint (emu_train_a1.py): {'state': ARH state_dict, 'mu': [15], 'sd': [15]}"""
    from fast_a1 import ARH
    ck = torch.load(path, map_location="cpu")
    net = ARH(N_A1_IN); net.load_state_dict(ck["state"]); net.eval()
    return net, np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)


def a1_draw(a1, Z, U, bs=65536):
    """the exported A1 graph in torch float32: standardise, trunk, draw rA (bin by u0, position by u1), then rB given
    the drawn rA (u2, u3). Returns [n, 2] residuals [deg] (float64 of the float32 values)."""
    from fast_models import EDGES, NB
    net, mu, sd = a1
    E = torch.tensor(EDGES, dtype=torch.float32)

    def draw(lg, ub, ui):
        c = torch.cumsum(torch.softmax(lg, 1), 1); k = torch.clamp((c < ub[:, None]).sum(1), max=NB - 1)
        return E[k] + ui * (E[k + 1] - E[k])
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), bs):
            z = (torch.as_tensor(np.ascontiguousarray(Z[i:i + bs], np.float32)) - torch.from_numpy(mu)) / torch.from_numpy(sd)
            u = torch.as_tensor(np.ascontiguousarray(U[i:i + bs], np.float32))
            h = net.t(z); rA = draw(net.logitsA(h), u[:, 0], u[:, 1]); rB = draw(net.logitsB(h, rA), u[:, 2], u[:, 3])
            out.append(torch.stack([rA, rB], 1).numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros((0, 2))


# ---------------- hash (MUST match SmartPixelsRecHitProducer emuHashU) ----------------
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_u(run, event, detId, idxInDet, salt):
    """uniform in [0, 1) from the stable cluster key (run, event, detId, idxInDet) and a salt:
    h = mix(((detId << 16) | idxInDet) + golden); h = mix(h ^ event); h = mix(h ^ run); h = mix(h ^ salt);
    u = (h >> 11) * 2^-53   [mix = the splitmix64 finaliser]"""
    with np.errstate(over="ignore"):
        u64 = lambda v: np.asarray(v).astype(np.uint64)
        h = _mix(((u64(detId) << np.uint64(16)) | u64(idxInDet)) + np.uint64(0x9E3779B97F4A7C15))
        h = _mix(h ^ u64(event)); h = _mix(h ^ u64(run)); h = _mix(h ^ np.uint64(salt))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def hash_index(run, event, detId, idxInDet, salt, k):
    return np.minimum((hash_u(run, event, detId, idxInDet, salt) * k).astype(np.int64), k - 1)


def a1_uniforms(run, event, detId, idxInDet):
    """the 4 A1 uniforms [n, 4]; the graph input is float32 (the producer casts the same double)"""
    return np.stack([hash_u(run, event, detId, idxInDet, s) for s in SALTS_A1], 1).astype(np.float32)


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
        cand[i, :k] = np.lexsort((np.arange(len(lib)), dd))[:k]
    return cand[:, :k]


# ---------------- the library file (emu_build_library.py): the rescaled SmartPixels dump ----------------
def load_library(path):
    import uproot
    d = {k: uproot.open(path)["dump"][k].array(library="np").astype(np.float64) for k in ("tA", "tB", "rA", "rB")}
    return {"dump_pts": np.c_[d["tA"], d["tB"]], "rA": d["rA"], "rB": d["rB"]}


def emulate(lib, surface, classT, trueCotA, trueCotB, cnnA, cnnB, drawA, drawB, run, event, detId, idxInDet):
    """The emu procedure on arrays (one entry per cluster). cnnA/B [deg] and the A1 draws drawA/B [deg] are used for
    class C only. Returns per-cluster pseudo-truth, dump row, biases, outputs [deg], saturation flags, cot outputs."""
    classT = np.asarray(classT, bool)
    rawPA = np.where(classT, np.degrees(np.arctan(np.asarray(trueCotA, np.float64))), np.asarray(cnnA, np.float64) + drawA)
    rawPB = np.where(classT, np.degrees(np.arctan(np.asarray(trueCotB, np.float64))), np.asarray(cnnB, np.float64) + drawB)
    psat = ~classT & ((np.abs(rawPA) >= SAT) | (np.abs(rawPB) >= SAT))
    pA, pB = sat(rawPA), sat(rawPB)
    qA = np.where(np.asarray(surface) > 10, np.degrees(np.arctan(np.tan(np.radians(pA)) + C_LOR)), pA)
    nb = knn_exact(lib["dump_pts"], np.c_[qA, pB], KNN_DUMP)
    row = nb[np.arange(len(pA)), hash_index(run, event, detId, idxInDet, SALT_DUMP, KNN_DUMP)]
    bA = np.median(lib["rA"][nb], 1); bB = np.median(lib["rB"][nb], 1)
    rawA = (pA + bA) + K_INFL * (lib["rA"][row] - bA); rawB = (pB + bB) + K_INFL * (lib["rB"][row] - bB)
    oA, oB = sat(rawA), sat(rawB)
    return dict(pseudoA=pA, pseudoB=pB, pseudoSat=psat, dumpRow=row, biasA=bA, biasB=bB, outA=oA, outB=oB,
                satA=np.abs(rawA) >= SAT, satB=np.abs(rawB) >= SAT,
                cotA=np.tan(np.radians(oA)).astype(np.float32), cotB=np.tan(np.radians(oB)).astype(np.float32))
