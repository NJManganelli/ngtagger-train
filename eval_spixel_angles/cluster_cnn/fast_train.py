"""FAST class-C study: train one FastNet (slim backbone, point/sigma head + ARH residual head) on the FIXED class-T training arrays
(fast_build_cache.py, no pT cut), then run it on ALL clusters of the 50 test events (cluster_cnn_v2 test arrays).

usage: fast_train.py --region barrel|disc --arch full|narrow|dws|tiny|proj1d [--wx 24 --wy 16] [--samples 15e6] [--warh 0.1]
Loss = 20 Huber(atan) + 0.2 GNLL(detached point) + 0.1 aux position (as v2_train pos-hubsig) + warh x (CE_A + CE_B) (ARH head,
residual to the DETACHED point, beta conditioned on the TRUE alpha residual = teacher forcing).
Test output FAST/preds/<tag>.parquet: row, pA, pB [deg], sA, sB [deg], dA, dB (ARH draw with the hash uniforms [deg]),
pitA (marginal), pitB (conditional on the true rA) for valid-truth rows.
"""
import argparse, os, time, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from fast_models import FastNet, raw_scalars, fit_norm, apply_norm, renc, bin_index, pit_from_logits, draw_np, count_macs, nparams, NB
from fast_common import FAST, load_test, uniforms, classes

p = argparse.ArgumentParser()
p.add_argument("--region", default="barrel"); p.add_argument("--arch", default="narrow")
p.add_argument("--wx", type=int, default=24); p.add_argument("--wy", type=int, default=16)
p.add_argument("--samples", type=float, default=None); p.add_argument("--bs", type=int, default=1024)
p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--warh", type=float, default=0.1)
p.add_argument("--tag", default=None); p.add_argument("--ntrain", type=int, default=0); p.add_argument("--ntest", type=int, default=0)
a = p.parse_args()
torch.set_num_threads(4); torch.manual_seed(1); np.random.seed(1)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tag = a.tag or f"{a.region}_{a.arch}" + ("" if (a.wx, a.wy) == (24, 16) else f"_w{a.wx}x{a.wy}")
OUT = os.path.join(FAST, "models", tag); os.makedirs(OUT, exist_ok=True)
AR = os.path.join(FAST, "arrays")
SAMPLES = a.samples or (15e6 if a.region == "barrel" else 12e6)
R2D = 57.29577951308232


def targets(m):
    return np.stack([np.arctan(m.tpLocalCotAlpha.values), np.arctan(m.tpLocalCotBeta.values), m.tx_hx.values, m.ty_hx.values], 1).astype(np.float32)


def gnll(mu, ls, y):
    ls = ls.clamp(-7, 3)
    return (0.5 * ((y - mu) / ls.exp()) ** 2 + ls).mean()


def loss_fn(net, x, s, y, msk):
    h = net.features(x, s); o = net.point(h)
    main = 20 * (F.smooth_l1_loss(o[:, 0], y[:, 0], beta=0.01) + F.smooth_l1_loss(o[:, 1], y[:, 1], beta=0.01)) \
        + 0.2 * (gnll(o[:, 0].detach(), o[:, 2], y[:, 0]) + gnll(o[:, 1].detach(), o[:, 3], y[:, 1]))
    lp = ((F.smooth_l1_loss(o[:, 4], y[:, 2], beta=0.1, reduction="none") + F.smooth_l1_loss(o[:, 5], y[:, 3], beta=0.1, reduction="none"))
          * msk).sum() / msk.sum().clamp(min=1)
    rA = (y[:, 0] - o[:, 0].detach()) * R2D; rB = (y[:, 1] - o[:, 1].detach()) * R2D
    kA = torch.bucketize(rA, EDG_T[1:-1], right=True); kB = torch.bucketize(rB, EDG_T[1:-1], right=True)
    ce = F.cross_entropy(net.arhA(h), kA) + F.cross_entropy(net.arhB(torch.cat([h, renc(rA)], 1)), kB)
    return main + 0.1 * lp + a.warh * ce, ce


from fast_models import EDGES
EDG_T = torch.tensor(EDGES, dtype=torch.float32, device=dev)

mtr = pd.read_parquet(os.path.join(AR, f"{a.region}_train_meta.parquet")); itr = np.load(os.path.join(AR, f"{a.region}_train_img.npy"), mmap_mode="r")
if a.ntrain:
    mtr, itr = mtr.iloc[:a.ntrain].reset_index(drop=True), itr[:a.ntrain]
itr = np.ascontiguousarray(itr)
mva = pd.read_parquet(os.path.join(AR, f"{a.region}_val_meta.parquet")); iva = np.load(os.path.join(AR, f"{a.region}_val_img.npy"))
Str = raw_scalars(mtr, a.region); nd = fit_norm(Str)
json.dump({**nd, **vars(a)}, open(os.path.join(OUT, "norm.json"), "w"), indent=1)
ntr = len(mtr); epochs = max(1, int(round(SAMPLES / ntr)))
X_tr = torch.from_numpy(itr); del itr
S_tr = torch.from_numpy(apply_norm(Str, nd)); Y_tr = torch.from_numpy(targets(mtr))
M_tr = torch.from_numpy(((mtr.hxDist < 0.1)).values.astype(np.float32))
Xv = torch.from_numpy(iva[:100_000]).to(dev); Sv = torch.from_numpy(apply_norm(raw_scalars(mva.iloc[:100_000], a.region), nd)).to(dev)
Yv = torch.from_numpy(targets(mva.iloc[:100_000])).to(dev); Mv = torch.from_numpy((mva.hxDist < 0.1).values[:100_000].astype(np.float32)).to(dev)
net = FastNet(a.arch, S_tr.shape[1], a.wx, a.wy)
macs = count_macs(net.eval(), torch.zeros(1, 24, 16), torch.zeros(1, S_tr.shape[1]))
info = dict(tag=tag, ntrain=ntr, epochs=epochs, scalars=nd["cols"], dropped=nd["dropped"], point_macs=macs,
            point_params=nparams(net) - nparams(net, ["arhA", "arhB"]), total_params=nparams(net))
print(json.dumps(info), flush=True)
net = net.to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps = epochs * (ntr // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
t0, CH, log = time.time(), 400_000, []
for ep in range(epochs):
    net.train(); perm = torch.randperm(ntr); tot, tce, nb = 0.0, 0.0, 0
    for c0 in range(0, ntr - a.bs + 1, CH):
        sel = perm[c0:c0 + CH]
        X, S, Y, M = X_tr[sel].to(dev), S_tr[sel].to(dev), Y_tr[sel].to(dev), M_tr[sel].to(dev)
        for b0 in range(0, len(sel) - a.bs + 1, a.bs):
            b = slice(b0, b0 + a.bs)
            l, ce = loss_fn(net, X[b], S[b], Y[b], M[b])
            opt.zero_grad(); l.backward(); opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += l.item(); tce += ce.item(); nb += 1
    net.eval()
    with torch.no_grad():
        vl, vce, oo = [], [], []
        for i in range(0, len(Xv), 8192):
            l, ce = loss_fn(net, Xv[i:i + 8192], Sv[i:i + 8192], Yv[i:i + 8192], Mv[i:i + 8192]); vl.append(l.item()); vce.append(ce.item())
            oo.append(net(Xv[i:i + 8192], Sv[i:i + 8192]).cpu().numpy())
    oo = np.concatenate(oo); yv = Yv.cpu().numpy()
    dA, dB = (oo[:, 0] - yv[:, 0]) * R2D, (oo[:, 1] - yv[:, 1]) * R2D
    rec = dict(epoch=ep, train_loss=tot / nb, train_ce=tce / nb, val_loss=float(np.mean(vl)), val_ce=float(np.mean(vce)),
               val_rmsA=float(np.sqrt(np.mean(dA ** 2))), val_rmsB=float(np.sqrt(np.mean(dB ** 2))),
               val_gt10A=float(np.mean(np.abs(dA) > 10)), val_gt10B=float(np.mean(np.abs(dB) > 10)), minutes=(time.time() - t0) / 60)
    log.append(rec); print(json.dumps(rec), flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(OUT, "model.pt"))
json.dump({"info": info, "log": log}, open(os.path.join(OUT, "trainlog.json"), "w"), indent=1)
del X_tr, S_tr, Y_tr

# ---- test events: all clusters
m = load_test(a.region)
if a.ntest:
    m = m.iloc[:a.ntest].reset_index(drop=True)
img = np.load(os.path.join(__import__("v2_common").V2, "arrays", f"{a.region}_test_img.npy"), mmap_mode="r")
img = img[:len(m)]
S = apply_norm(raw_scalars(m, a.region), nd); U = uniforms(m)
valid, _ = classes(m)
tA = np.where(valid, np.degrees(np.arctan(m.tpLocalCotAlpha.values)), 0.0); tB = np.where(valid, np.degrees(np.arctan(m.tpLocalCotBeta.values)), 0.0)
cols = {k: np.full(len(m), np.nan, np.float32) for k in ("pA", "pB", "sA", "sB", "dA", "dB", "pitA", "pitB")}
with torch.no_grad():
    for i in range(0, len(m), 8192):
        sl = slice(i, i + 8192)
        h = net.features(torch.as_tensor(np.ascontiguousarray(img[sl])).to(dev), torch.as_tensor(S[sl]).to(dev))
        o = net.point(h).cpu().numpy(); lA = net.arhA(h)
        pa, pb = np.degrees(np.clip(o[:, 0], -1.5707, 1.5707)), np.degrees(np.clip(o[:, 1], -1.5707, 1.5707))
        u = U[sl]; lAn = lA.cpu().numpy().astype(np.float64)
        dA = draw_np(lAn, u[:, 0], u[:, 1])
        lB = net.arhB(torch.cat([h, renc(torch.as_tensor(dA, dtype=torch.float32, device=dev))], 1)).cpu().numpy().astype(np.float64)
        dB = draw_np(lB, u[:, 2], u[:, 3])
        rA_t = tA[sl] - pa; rB_t = tB[sl] - pb
        lBt = net.arhB(torch.cat([h, renc(torch.as_tensor(rA_t, dtype=torch.float32, device=dev))], 1)).cpu().numpy().astype(np.float64)
        v = valid[sl]
        cols["pA"][sl], cols["pB"][sl] = pa, pb
        cols["sA"][sl], cols["sB"][sl] = np.degrees(np.exp(np.clip(o[:, 2], -7, 3))), np.degrees(np.exp(np.clip(o[:, 3], -7, 3)))
        cols["dA"][sl], cols["dB"][sl] = dA, dB
        cols["pitA"][sl] = np.where(v, pit_from_logits(lAn, rA_t), np.nan); cols["pitB"][sl] = np.where(v, pit_from_logits(lBt, rB_t), np.nan)
pd.DataFrame({"row": m.row.values, **cols}).to_parquet(os.path.join(FAST, "preds", f"{tag}.parquet"))
print("done", tag, round((time.time() - t0) / 60, 1), "min", flush=True)
