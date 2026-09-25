"""Cluster CNN: fixed-window ADC map -> (x, y, cotAlpha, cotBeta).

Angles are learned as magnitude + sign of FOLDED variables, because a charge
projection is (nearly) symmetric under direction reversal:
  f_a = cotAlpha + c_layer   (c_layer = digitizer tan(theta_L,x): 0 on L1 (3D), 0.202 on L2-L4;
                              the x-extent is |f_a| * T / pitch, minimal at cotAlpha = -c_layer)
  f_b = cotBeta              (no Lorentz drift along y)
Heads: atan|f| (rad, smooth-L1), sign(f) (BCE), tx/ty = truth position relative to the
bounding-box centre in pixel units (smooth-L1, masked to truth-consistent labels).

Variants (--variant):
  base : image + layer one-hot + sizeX/sizeY/size/charge/truncation flags (SENSOR-LOCAL)
  bla  : base + local B (x,y,z), |B|, tan(theta_L) x/y      (sensor-local calibration)
  pos  : base + localX/localY/zOutward + global r/z          (NOT sensor-local: prompt prior)
  small: on-sensor-limited Conv1D net (models_def.SmallNet), 2-bit map + layer only
--clean : train only on tpChargeFrac >= 0.9
"""
import argparse, os, time, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from models_def import C_LAYER, BASE_COLS, VAR_COLS, scalars, make

p = argparse.ArgumentParser()
p.add_argument("--variant", default="base", choices=["base", "bla", "pos", "small"])
p.add_argument("--clean", action="store_true")
p.add_argument("--data", default="data")
p.add_argument("--epochs", type=int, default=6)
p.add_argument("--ntrain", type=int, default=2_000_000)
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--threads", type=int, default=6)
p.add_argument("--tag", default=None)
p.add_argument("--device", default="mps")
p.add_argument("--lr", type=float, default=2e-3)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(1); np.random.seed(1)
dev = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
tag = a.tag or (a.variant + ("_clean" if a.clean else ""))
OUT = os.path.join(HERE, "models", tag); os.makedirs(OUT, exist_ok=True)  # models/ -> WDMac symlink
D = os.path.join(HERE, a.data)

NEED = sorted(set(BASE_COLS + sum(VAR_COLS.values(), []) + ["linkClass", "tpPt", "tpCharge", "tpLocalCotAlpha",
        "tpLocalCotBeta", "hxOk", "hxDist", "tpChargeFrac", "split", "tx_hx", "ty_hx"]))
meta = pd.read_parquet(os.path.join(D, "meta.parquet"), columns=NEED)
imgs = np.load(os.path.join(D, "images.npy"), mmap_mode="r")
WX, WY = imgs.shape[1:]

def targets(m):
    c = C_LAYER[m.layer.values]
    fa = m.tpLocalCotAlpha.values + c
    fb = m.tpLocalCotBeta.values
    return np.stack([np.arctan(np.abs(fa)), np.arctan(np.abs(fb)), (fa > 0), (fb > 0),
                     m.tx_hx.values, m.ty_hx.values], 1).astype(np.float32)

good = ((meta.linkClass == 2) & (meta.tpPt >= 0.25) & (meta.tpCharge != 0) & (meta.tpLocalCotAlpha > -900)
        & (meta.hxOk == 1) & (meta.hxDist < 1.0))  # truth-consistent labels (helix within 1 cm of the hit)
if a.clean:
    good &= meta.tpChargeFrac >= 0.9
tr_idx = np.where(good & (meta.split == 0))[0]
va_idx = np.where(good & (meta.split == 1))[0]
if len(tr_idx) > a.ntrain:
    tr_idx = np.sort(np.random.choice(tr_idx, a.ntrain, replace=False))
va_idx = np.sort(np.random.choice(va_idx, min(len(va_idx), 200_000), replace=False))
print(f"train {len(tr_idx)} val {len(va_idx)}", flush=True)

S_tr = scalars(meta.iloc[tr_idx], a.variant)
mu, sd = S_tr.mean(), S_tr.std().replace(0, 1).fillna(1)
norm = {"mu": mu.to_dict(), "sd": sd.to_dict(), "cols": list(S_tr.columns), "variant": a.variant}
json.dump(norm, open(os.path.join(OUT, "norm.json"), "w"))

def tensors(idx):
    m = meta.iloc[idx]
    X = torch.from_numpy(np.ascontiguousarray(imgs[idx]))  # uint8
    S = torch.from_numpy(((scalars(m, a.variant) - mu) / sd).values.astype(np.float32))
    Y = torch.from_numpy(targets(m))
    M = torch.from_numpy(((m.hxDist < 0.1) & (m.hxOk == 1)).values.astype(np.float32))  # position-label mask
    return X.to(dev), S.to(dev), Y.to(dev), M.to(dev)

def loss_fn(o, y, msk):
    la = F.smooth_l1_loss(o[:, 0], y[:, 0], beta=0.01) + F.smooth_l1_loss(o[:, 1], y[:, 1], beta=0.01)
    ls = F.binary_cross_entropy_with_logits(o[:, 2], y[:, 2]) + F.binary_cross_entropy_with_logits(o[:, 3], y[:, 3])
    lp = ((F.smooth_l1_loss(o[:, 4], y[:, 4], beta=0.1, reduction="none")
           + F.smooth_l1_loss(o[:, 5], y[:, 5], beta=0.1, reduction="none")) * msk).sum() / msk.sum().clamp(min=1)
    return 20 * la + 0.3 * ls + 0.1 * lp, (la.item(), ls.item(), lp.item())

net = make(a.variant, S_tr.shape[1]).to(dev)
print("params", sum(p.numel() for p in net.parameters()), flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * (len(tr_idx) // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
Xv, Sv, Yv, Mv = tensors(va_idx)
# training data is loaded in chunks to bound memory
CH = 500_000
log = []
t0 = time.time()
for ep in range(a.epochs):
    net.train()
    perm = np.random.permutation(len(tr_idx))
    tot, nb = 0, 0
    for c0 in range(0, len(perm), CH):
        sel = np.sort(tr_idx[perm[c0:c0 + CH]])
        X, S, Y, M = tensors(sel)
        pp = torch.randperm(len(sel), device=dev)
        for b0 in range(0, len(sel) - a.bs + 1, a.bs):
            if nb >= (ep + 1) * (len(tr_idx) // a.bs):
                break
            b = pp[b0:b0 + a.bs]
            o = net(X[b], S[b])
            l, parts = loss_fn(o, Y[b], M[b])
            opt.zero_grad(); l.backward(); opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += l.item(); nb += 1
    net.eval()
    with torch.no_grad():
        vo = torch.cat([net(Xv[i:i + 8192], Sv[i:i + 8192]) for i in range(0, len(va_idx), 8192)])
        vl, vparts = loss_fn(vo, Yv, Mv)
        # quick val metric: sigma_MAD in deg of magnitude, sign acc
        def smad(x): return float(1.4826 * np.median(np.abs(x - np.median(x))))
        vo, Yc = vo.cpu(), Yv.cpu()
        da = np.degrees((vo[:, 0] - Yc[:, 0]).numpy()); db = np.degrees((vo[:, 1] - Yc[:, 1]).numpy())
        sa = ((vo[:, 2] > 0).float() == Yc[:, 2]).float().mean().item(); sb = ((vo[:, 3] > 0).float() == Yc[:, 3]).float().mean().item()
    rec = dict(epoch=ep, train_loss=tot / max(nb, 1), val_loss=vl.item(), val_parts=vparts,
               val_magA_smad_deg=smad(da), val_magB_smad_deg=smad(db), val_signA=sa, val_signB=sb,
               minutes=(time.time() - t0) / 60)
    log.append(rec); print(json.dumps(rec), flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(OUT, "model.pt"))
json.dump(log, open(os.path.join(OUT, "trainlog.json"), "w"), indent=1)
