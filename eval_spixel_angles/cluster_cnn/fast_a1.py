"""FAST class-C study, option A1: a learned CONDITIONAL DISTRIBUTION of the donor residual r = (truth - CNN) [deg, angle space]
given the SAME 11 observables as the knn donor (CNN alpha, beta, sigma_alpha, sigma_beta, sizeX, sizeY, log charge, r, |z|,
localX, localY) + the surface one-hot. Replaces the exact 11-D KD-tree 10-NN search.

Models (small MLP trunk 15 -> 128 -> 128):
  arh : autoregressive histogram p(rA | z) p(rB | z, rA) on fast_models.EDGES (156 bins), CE loss (beta head teacher-forced)
  mdn : 8-component bivariate Gaussian mixture with a per-component correlation (the alpha/beta dependence), NLL
Training data = class-T clusters (fixed definition, no pT cut) of the test events, 2-FOLD by event half exactly as the knn
prototype (a cluster of half h is drawn from a model trained on the other half); 10% of the training half's events = early-stop
monitor. Sampling: deterministic, hash uniforms (fast_common.uniforms): arh = bin by u1 + position by u2, then beta the same with
u3, u4; mdn = component by u1, z1 = Phi^-1(u2), z2 = Phi^-1(u3).
PIT (valid-truth rows): alpha marginal, beta CONDITIONAL on the true alpha residual (exact, from the model).
usage: fast_a1.py <region> <cnn source: ref | fast model tag> [arh mdn]
Output FAST/preds/a1_<model>_<cnn>_<region>.parquet: row, dA, dB, pitA, pitB
"""
import os, sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.special import ndtri, ndtr
from fast_models import EDGES, NB, renc, pit_from_logits, draw_np
from fast_common import FAST, V2, load_test, uniforms, classes, groups

torch.set_num_threads(4); torch.manual_seed(3)
R2D = 57.29577951308232; KM = 8


def cnn_source(src, reg, m):
    if src == "ref":
        P = pd.read_parquet(os.path.join(V2, "preds", f"{reg}_pos_hubsig.parquet")).set_index("row").reindex(m.row.values)
        return (np.degrees(np.arctan(P.cotA.values)), np.degrees(np.arctan(P.cotB.values)), P.sigA_deg.values, P.sigB_deg.values)
    P = pd.read_parquet(os.path.join(FAST, "preds", f"{src}.parquet")).set_index("row").reindex(m.row.values)
    return P.pA.values, P.pB.values, P.sA.values, P.sB.values


def observables(m, reg, hA, hB, sA, sB):
    surf, _ = groups(m, reg)
    Z = np.c_[hA, hB, sA, sB, m.sizeX, m.sizeY, np.log(m.charge.clip(lower=1)), m.globalR, np.abs(m.globalZ), m.localX, m.localY]
    oh = np.stack([surf == s for s in (1, 2, 3, 4)], 1).astype(np.float64)
    return np.c_[Z, oh].astype(np.float64)


class Trunk(nn.Module):
    def __init__(self, ni, nh=128):
        super().__init__(); self.f = nn.Sequential(nn.Linear(ni, nh), nn.ReLU(), nn.Linear(nh, nh), nn.ReLU())

    def forward(self, z):
        return self.f(z)


class ARH(nn.Module):
    def __init__(self, ni):
        super().__init__(); self.t = Trunk(ni); self.hA = nn.Linear(128, NB)
        self.hB = nn.Sequential(nn.Linear(132, 64), nn.ReLU(), nn.Linear(64, NB))

    def logitsA(self, h):
        return self.hA(h)

    def logitsB(self, h, rA):
        return self.hB(torch.cat([h, renc(rA)], 1))

    def loss(self, z, r):
        h = self.t(z); E = EDG
        kA = torch.bucketize(r[:, 0], E[1:-1], right=True); kB = torch.bucketize(r[:, 1], E[1:-1], right=True)
        return F.cross_entropy(self.logitsA(h), kA) + F.cross_entropy(self.logitsB(h, r[:, 0]), kB)


class MDN(nn.Module):
    """r in units of 10 deg inside the network"""
    def __init__(self, ni):
        super().__init__(); self.t = Trunk(ni); self.o = nn.Linear(128, KM * 6)

    def params(self, z):
        o = self.o(self.t(z)).view(-1, KM, 6)
        lw = F.log_softmax(o[..., 0], 1); mu = o[..., 1:3]; ls = o[..., 3:5].clamp(-6, 3); rho = 0.98 * torch.tanh(o[..., 5])
        return lw, mu, ls, rho

    def loss(self, z, r):
        lw, mu, ls, rho = self.params(z); x = (r[:, None, :] / 10.0 - mu) / ls.exp()
        q = (x[..., 0] ** 2 - 2 * rho * x[..., 0] * x[..., 1] + x[..., 1] ** 2) / (1 - rho ** 2)
        lp = -0.5 * q - ls.sum(-1) - 0.5 * torch.log(1 - rho ** 2) - np.log(2 * np.pi)
        return -torch.logsumexp(lw + lp, 1).mean()


EDG = torch.tensor(EDGES, dtype=torch.float32)


def fit(kind, Z, R, ev, seed):
    """train on (Z, R); 10% of the events = monitor; return the best-epoch model"""
    rng = np.random.default_rng(seed); ue = np.unique(ev); mon = np.isin(ev, rng.choice(ue, max(1, len(ue) // 10), replace=False))
    zt, rt = torch.tensor(Z[~mon], dtype=torch.float32), torch.tensor(R[~mon], dtype=torch.float32)
    zv, rv = torch.tensor(Z[mon], dtype=torch.float32), torch.tensor(R[mon], dtype=torch.float32)
    net = (ARH if kind == "arh" else MDN)(Z.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20); best, bst, log = 1e9, None, []
    for ep in range(20):
        net.train(); perm = torch.randperm(len(zt))
        for i in range(0, len(zt), 2048):
            b = perm[i:i + 2048]; l = net.loss(zt[b], rt[b]); opt.zero_grad(); l.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = float(np.mean([net.loss(zv[i:i + 65536], rv[i:i + 65536]).item() for i in range(0, len(zv), 65536)]))
        log.append(vl)
        if vl < best:
            best, bst = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bst); net.eval()
    return net, log


def predict(kind, net, Z, U, rtrue, valid):
    z = torch.tensor(Z, dtype=torch.float32); n = len(Z)
    dA, dB, pA, pB = (np.full(n, np.nan) for _ in range(4))
    with torch.no_grad():
        for i in range(0, n, 65536):
            s = slice(i, i + 65536); u = U[s]; v = valid[s]
            if kind == "arh":
                h = net.t(z[s]); lA = net.logitsA(h).double().numpy()
                a = draw_np(lA, u[:, 0], u[:, 1]); lB = net.logitsB(h, torch.tensor(a, dtype=torch.float32)).double().numpy()
                dA[s], dB[s] = a, draw_np(lB, u[:, 2], u[:, 3])
                lBt = net.logitsB(h, torch.tensor(rtrue[s, 0], dtype=torch.float32)).double().numpy()
                pA[s] = np.where(v, pit_from_logits(lA, rtrue[s, 0]), np.nan); pB[s] = np.where(v, pit_from_logits(lBt, rtrue[s, 1]), np.nan)
            else:
                lw, mu, ls, rho = [t.double().numpy() for t in net.params(z[s])]
                w = np.exp(lw); sg = np.exp(ls) * 10; mu = mu * 10; c = np.cumsum(w, 1)
                k = np.minimum((c < u[:, :1]).sum(1), KM - 1); ii = np.arange(len(k))
                z1, z2 = ndtri(np.clip(u[:, 1], 1e-12, 1 - 1e-12)), ndtri(np.clip(u[:, 2], 1e-12, 1 - 1e-12))
                r_ = rho[ii, k]
                dA[s] = mu[ii, k, 0] + sg[ii, k, 0] * z1
                dB[s] = mu[ii, k, 1] + sg[ii, k, 1] * (r_ * z1 + np.sqrt(1 - r_ ** 2) * z2)
                ra, rb = rtrue[s, 0][:, None], rtrue[s, 1][:, None]
                xa = (ra - mu[..., 0]) / sg[..., 0]
                pA[s] = np.where(v, (w * ndtr(xa)).sum(1), np.nan)
                wc = w * np.exp(-0.5 * xa ** 2) / sg[..., 0]; wc = wc / np.maximum(wc.sum(1, keepdims=True), 1e-300)
                mc = mu[..., 1] + rho * sg[..., 1] * xa; sc = sg[..., 1] * np.sqrt(1 - rho ** 2)
                pB[s] = np.where(v, (wc * ndtr((rb - mc) / sc)).sum(1), np.nan)
    return dA, dB, pA, pB


if __name__ == "__main__":
    reg, src = sys.argv[1], sys.argv[2]; kinds = sys.argv[3:] or ["arh", "mdn"]
    m = load_test(reg); valid, clsT = classes(m); t0 = time.time()
    hA, hB, sA, sB = cnn_source(src, reg, m)
    tA = np.where(valid, np.degrees(np.arctan(m.tpLocalCotAlpha.values)), 0.0); tB = np.where(valid, np.degrees(np.arctan(m.tpLocalCotBeta.values)), 0.0)
    Z = observables(m, reg, hA, hB, sA, sB); ok = np.isfinite(Z).all(1)
    mu, sd = Z[clsT & ok].mean(0), Z[clsT & ok].std(0); sd[sd == 0] = 1; Z = np.nan_to_num((Z - mu) / sd)
    R = np.c_[tA - hA, tB - hB]; U = uniforms(m)
    ev = np.sort(m.event.unique()); half = m.event.isin(ev[::2]).values
    for kind in kinds:
        out = {k: np.full(len(m), np.nan) for k in ("dA", "dB", "pitA", "pitB")}; logs = {}
        for hf in (True, False):
            tr = clsT & ok & (half != hf) & np.isfinite(R).all(1)
            net, lg = fit(kind, Z[tr], R[tr], m.event.values[tr], seed=int(hf)); logs[str(hf)] = lg
            tg = np.where(half == hf)[0]
            a, b, pa, pb = predict(kind, net, Z[tg], U[tg], np.nan_to_num(R[tg]), valid[tg])
            out["dA"][tg], out["dB"][tg], out["pitA"][tg], out["pitB"][tg] = a, b, pa, pb
            if hf:
                torch.save(net.state_dict(), os.path.join(FAST, "models", f"a1_{kind}_{src}_{reg}.pt"))
            print(kind, src, reg, "fold", hf, "val loss", [round(x, 4) for x in lg], round((time.time() - t0) / 60, 1), "min", flush=True)
        pd.DataFrame({"row": m.row.values, **{k: v.astype(np.float32) for k, v in out.items()}}).to_parquet(
            os.path.join(FAST, "preds", f"a1_{kind}_{src}_{reg}.parquet"))
        json.dump(logs, open(os.path.join(FAST, "logs", f"a1_{kind}_{src}_{reg}.json"), "w"))
