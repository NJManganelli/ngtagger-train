"""FAST class-C study: network definitions (slim backbones + point/sigma head + autoregressive-histogram residual head),
residual binning, MAC counter, scalar features without the geometry rounding residues.

Every network has ONE trunk and two heads:
  point head  (the pos-hubsig head): atan(cotA), atan(cotB) [rad] (Huber), log sigma (detached GNLL), aux tx, ty (6 outputs)
  ARH head    (A2 "single network"): the conditional distribution of the residual r = true angle - point [deg], as an
              autoregressive histogram p(rA | x) p(rB | x, rA) on fixed non-uniform bins (EDGES); trained with cross-entropy
              against the DETACHED point (so the point head is unchanged by it); sampled by inverse CDF with hash uniforms
              (bin from u1, position inside the bin from u2; then rB the same way given the drawn rA).
Backbones (input = 24 x 16 digi-ADC window, channels ADC/16, occupancy, x/y coordinate maps as v2_train):
  full    : the v2 pos-hubsig backbone (5 conv 32-32-64s2-64-64s2, trunk 256-128), the reference
  narrow  : same topology, widths 16-16-32s2-32-32s2, trunk 128-64
  dws     : first conv 4->32, then depthwise-separable 3x3 blocks 32-64s2-64-64s2, trunk 256-128
  tiny    : 3 conv 16-32s2-32s2, trunk 64-64
  proj1d  : projections of ADC/16 and occupancy on x (24) and y (16) (SmartPixels Conv1D style): Conv1d 2->16->16s2 per axis,
            trunk 128-64 (no 2D convolution at all)
--wx/--wy crop the centred window (clusters are centred in the 24 x 16 window by prep.py / the producer).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from models_def import scalars as base_scalars

# residual bin edges [deg], symmetric: 0.1 deg to 2, 0.25 to 6, 1 to 20, 2.5 to 60, 10 to 180
_pos = np.unique(np.r_[np.arange(0, 2, 0.1), np.arange(2, 6, 0.25), np.arange(6, 20, 1.0), np.arange(20, 60, 2.5),
                       np.arange(60, 180.1, 10.0)].round(6))
EDGES = np.r_[-_pos[::-1], _pos[1:]].astype(np.float64)
NB = len(EDGES) - 1
ORIENT = ["axXz", "axYz", "axZz", "axXr", "axYr", "axZr"]
POS = ["pu", "pv", "pw", "globalR", "globalZ", "localX", "localY"]


def raw_scalars(m, region):
    s = base_scalars(m, "base" if region == "barrel" else "disc")
    s["is3D"] = ((m.subdet == 1) & (m.layer == 1)).values.astype(np.float32)
    for c in ORIENT + ["sideZ"] + POS:
        s[c] = m[c].values
    return s.astype(np.float64)


def fit_norm(S):
    """drop constant columns and geometry ROUNDING RESIDUES (sd < 1e-6: e.g. barrel axXz, axZz, axYr ~1e-16), which the v2
    model used (fragile across geometry/release/arch); the kept orientation columns are rounded to 1e-6 at use."""
    mu, sd = S.mean(), S.std()
    keep = [c for c in S.columns if sd[c] > 1e-6]
    return {"cols": keep, "mu": {c: float(mu[c]) for c in keep}, "sd": {c: float(sd[c]) for c in keep},
            "dropped": [c for c in S.columns if c not in keep]}


def apply_norm(S, nd):
    X = S[nd["cols"]].copy()
    for c in ORIENT:
        if c in X:
            X[c] = X[c].round(6)
    mu = np.array([nd["mu"][c] for c in nd["cols"]]); sd = np.array([nd["sd"][c] for c in nd["cols"]])
    return ((X.values - mu) / sd).astype(np.float32)


def renc(r):
    """encoding of the (drawn or true) alpha residual [deg] fed to the beta head"""
    return torch.stack([r / 10.0, r / 60.0, torch.tanh(r / 2.0), torch.tanh(r / 10.0)], 1)


class Conv(nn.Sequential):
    def __init__(self, i, o, s=1, groups=1, k=3):
        super().__init__(nn.Conv2d(i, o, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(o), nn.ReLU())


class DWS(nn.Sequential):
    def __init__(self, i, o, s=1):
        super().__init__(Conv(i, i, s, groups=i), Conv(i, o, 1, k=1))


class FastNet(nn.Module):
    def __init__(self, arch, ns, wx=24, wy=16):
        super().__init__()
        self.arch, self.wx, self.wy = arch, wx, wy
        gx, gy = torch.meshgrid(torch.linspace(-1, 1, wx), torch.linspace(-1, 1, wy), indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy])[None])
        q = lambda n: (n + 3) // 4
        if arch == "full":
            self.conv = nn.Sequential(Conv(4, 32), Conv(32, 32), Conv(32, 64, 2), Conv(64, 64), Conv(64, 64, 2)); nf, t = 64 * q(wx) * q(wy), (256, 128)
        elif arch == "narrow":
            self.conv = nn.Sequential(Conv(4, 16), Conv(16, 16), Conv(16, 32, 2), Conv(32, 32), Conv(32, 32, 2)); nf, t = 32 * q(wx) * q(wy), (128, 64)
        elif arch == "dws":
            self.conv = nn.Sequential(Conv(4, 32), DWS(32, 64, 2), DWS(64, 64), DWS(64, 64, 2)); nf, t = 64 * q(wx) * q(wy), (256, 128)
        elif arch == "tiny":
            self.conv = nn.Sequential(Conv(4, 16), Conv(16, 32, 2), Conv(32, 32, 2)); nf, t = 32 * q(wx) * q(wy), (64, 64)
        elif arch == "proj1d":
            c1 = lambda i, o, s=1: nn.Sequential(nn.Conv1d(i, o, 3, s, 1), nn.ReLU())
            self.cx = nn.Sequential(c1(2, 16), c1(16, 16, 2)); self.cy = nn.Sequential(c1(2, 16), c1(16, 16, 2))
            nf, t = 16 * ((wx + 1) // 2) + 16 * ((wy + 1) // 2), (128, 64)
        else:
            raise ValueError(arch)
        self.trunk = nn.Sequential(nn.Linear(nf + ns, t[0]), nn.ReLU(), nn.Linear(t[0], t[1]), nn.ReLU())
        self.point = nn.Linear(t[1], 6)
        self.arhA = nn.Linear(t[1], NB)
        self.arhB = nn.Sequential(nn.Linear(t[1] + 4, 64), nn.ReLU(), nn.Linear(64, NB))

    def crop(self, x):
        X, Y = x.shape[1], x.shape[2]
        if (X, Y) == (self.wx, self.wy):
            return x
        ox, oy = (X - self.wx) // 2, (Y - self.wy) // 2
        return x[:, ox:ox + self.wx, oy:oy + self.wy]

    def features(self, x, s):
        x = self.crop(x).float(); adc = x / 16.0; occ = (x > 0).float()
        if self.arch == "proj1d":
            px = torch.stack([adc.sum(2), occ.sum(2)], 1); py = torch.stack([adc.sum(1), occ.sum(1)], 1)
            f = torch.cat([self.cx(px).flatten(1), self.cy(py).flatten(1)], 1)
        else:
            z = torch.cat([adc[:, None], occ[:, None], self.coords.expand(x.shape[0], -1, -1, -1)], 1)
            f = self.conv(z).flatten(1)
        return self.trunk(torch.cat([f, s], 1))

    def forward(self, x, s):
        """deployment graph for the point estimate: [atanA, atanB, logsigA, logsigB, tx, ty]"""
        return self.point(self.features(x, s))


class A2Export(nn.Module):
    """ONNX graph of the single A2 network: inputs image, scalars, u (n, 4) hash uniforms -> out (n, 8):
    point atanA, atanB [rad], logsig A, B, drawn residual rA, rB [deg], pseudo angle A, B [deg] (unsaturated)."""
    def __init__(self, net):
        super().__init__(); self.net = net
        self.register_buffer("edges", torch.tensor(EDGES, dtype=torch.float32))

    def draw(self, logits, u_bin, u_in):
        cdf = torch.cumsum(torch.softmax(logits, 1), 1)
        k = torch.clamp((cdf < u_bin[:, None]).sum(1), max=NB - 1)
        lo, hi = self.edges[k], self.edges[k + 1]
        return lo + u_in * (hi - lo)

    def forward(self, x, s, u):
        h = self.net.features(x, s); p = self.net.point(h)
        rA = self.draw(self.net.arhA(h), u[:, 0], u[:, 1])
        rB = self.draw(self.net.arhB(torch.cat([h, renc(rA)], 1)), u[:, 2], u[:, 3])
        a = p[:, 0] * 57.29577951308232; b = p[:, 1] * 57.29577951308232
        return torch.cat([p[:, :4], torch.stack([rA, rB, a + rA, b + rB], 1)], 1)


def count_macs(model, *inputs):
    """multiply-accumulates per sample (Conv1d/Conv2d/Linear), batch 1"""
    tot = [0]
    def hk(m, i, o):
        if isinstance(m, nn.Conv2d):
            tot[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Conv1d):
            tot[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            tot[0] += m.in_features * m.out_features
    hs = [mm.register_forward_hook(hk) for mm in model.modules() if isinstance(mm, (nn.Conv1d, nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(*inputs)
    for h in hs:
        h.remove()
    return tot[0]


def nparams(model, prefixes=None):
    return int(sum(p.numel() for n, p in model.named_parameters() if prefixes is None or n.split(".")[0] in prefixes))


def bin_index(r):
    return np.clip(np.searchsorted(EDGES, r, side="right") - 1, 0, NB - 1)


def pit_from_logits(logits, r):
    """PIT = CDF(r) under the histogram (linear inside the bin), numpy"""
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    k = bin_index(r); c = np.cumsum(p, 1)
    below = np.where(k > 0, c[np.arange(len(r)), np.maximum(k - 1, 0)], 0.0)
    frac = np.clip((np.clip(r, EDGES[0], EDGES[-1]) - EDGES[k]) / (EDGES[k + 1] - EDGES[k]), 0, 1)
    return below + frac * p[np.arange(len(r)), k]


def draw_np(logits, u_bin, u_in):
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    c = np.cumsum(p, 1); k = np.minimum((c < u_bin[:, None]).sum(1), NB - 1)
    return EDGES[k] + u_in * (EDGES[k + 1] - EDGES[k])
