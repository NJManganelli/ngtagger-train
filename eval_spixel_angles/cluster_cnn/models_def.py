"""Shared model/feature definitions for train.py and evaluate.py."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

# digitizer tan(theta_L,x) per layer (ntuple tanLAx: 0 on L1 = 3D sensors, 0.202 on L2-L4)
C_LAYER = np.array([0, 0.0, 0.202, 0.202, 0.202], np.float32)
# 2-bit quantisation of the 4-bit digi ADC (ADC 0 = 1750 e, 1 = 3250 e, ... 5 = 9250 e):
# 0 empty | 1: ADC 0-1 (< 4000 e) | 2: ADC 2-4 | 3: ADC >= 5 (> 8500 e)
Q2_EDGES = (2, 5)

BASE_COLS = ["layer", "sizeX", "sizeY", "size", "charge", "adcSum", "truncX", "truncY", "nPxLost"]
VAR_COLS = {"base": [], "small": [], "bla": ["bLocalX", "bLocalY", "bLocalZ", "bMag", "tanLAx", "tanLAy"],
            "pos": ["localX", "localY", "zOutward", "globalR", "globalZ"]}


def scalars(m, variant):
    cols = {"l1": (m.layer == 1), "l2": (m.layer == 2), "l3": (m.layer == 3), "l4": (m.layer == 4)}
    if variant != "small":
        cols.update({"sizeX": m.sizeX, "sizeY": m.sizeY, "size": m["size"], "logq": np.log(m.charge.clip(lower=1)),
                     "adcSum": m.adcSum, "truncX": m.truncX, "truncY": m.truncY, "nPxLost": m.nPxLost})
    for c in VAR_COLS[variant]:
        cols[c] = m[c]
    return pd.DataFrame(cols).astype(np.float32)


class Net(nn.Module):
    """Full CNN: 4 channels (ADC/16, occupancy, x/y coordinates), 5 conv blocks, MLP head."""
    def __init__(self, ns, WX=24, WY=16):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.conv = nn.Sequential(blk(4, 32), blk(32, 32), blk(32, 64, 2), blk(64, 64), blk(64, 64, 2))
        gx, gy = torch.meshgrid(torch.linspace(-1, 1, WX), torch.linspace(-1, 1, WY), indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy])[None])
        nflat = 64 * ((WX + 3) // 4) * ((WY + 3) // 4)
        self.head = nn.Sequential(nn.Linear(nflat + ns, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(self, x, s):
        x = x.float(); adc = x / 16.0; occ = (x > 0).float()
        z = torch.cat([adc[:, None], occ[:, None], self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        o = self.head(torch.cat([self.conv(z).flatten(1), s], 1))
        return torch.cat([F.softplus(o[:, :2]), o[:, 2:]], 1)


class SmallNet(nn.Module):
    """On-sensor-limited net after smart-pixels-ml Full1D_model2 (Conv1D_Full): 2-bit map,
    average projections onto x and y, Conv1D(5, k=3) each, tanh, Dense16-tanh-Dense16-tanh-out.
    One time slice (the CMSSW digitizer has none) + layer one-hot into the first dense."""
    def __init__(self, ns, WX=24, WY=16):
        super().__init__()
        self.cx = nn.Conv1d(1, 5, 3); self.cy = nn.Conv1d(1, 5, 3)
        nflat = 5 * (WX - 2) + 5 * (WY - 2)
        self.head = nn.Sequential(nn.Linear(nflat + ns, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 6))

    def forward(self, x, s):
        x = x.long() - 1                                   # raw ADC, -1 = empty
        q = (x >= 0).float() + (x >= Q2_EDGES[0]).float() + (x >= Q2_EDGES[1]).float()  # 0..3
        px = q.mean(2)[:, None]; py = q.mean(1)[:, None]
        h = torch.tanh(torch.cat([self.cx(px).flatten(1), self.cy(py).flatten(1)], 1))
        o = self.head(torch.cat([h, s], 1))
        return torch.cat([F.softplus(o[:, :2]), o[:, 2:]], 1)


def make(variant, ns):
    return SmallNet(ns) if variant == "small" else Net(ns)
