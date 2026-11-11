"""Cross-fitted closure of the parametrization (PLAN.md section 7b; rules of stats-review-02, finding 4).

  python param_closure.py [--scen AAAA AAAI IAAA] [--allocs ...] [--root DIR] [--files 5 7 8 9 10]
      -> checks/param_closure_<scenario>.json

Cross-fit by event: fold = hash(run, lumi, event) mod 2 over the scan files; the tables are fitted on one fold and
applied to the other, both ways; every scan event is a test event.

Pre-declared material checks, per shipped allocation and scenario:
  core          six primary cells (particle |eta| 0-1.1 / 1.1-1.8 / 1.8-2.4 x pT 2-5 / >= 5 GeV, tracks with a particle),
                d0 and z0: |W68_emulated - W68_actual| > max(0.25 |W68_actual - W68_today|, 0.02 W68_actual, 3 sd)
  prompt tail   share of prompt tracks (|eta| inclusive, pT 2-5 and >= 5 GeV) with |q - truth| > 3 sigma_today, d0 and z0:
                |difference| > max(0.25 |share_actual - share_today|, 0.1 share_actual, 3 sd)
  end to end    prompt efficiency at b/c-daughter efficiency 50, 70, 90 % by |d0| / sigma(d0), and the prompt share with
                |d0| / sigma > 3: difference > max(0.25 of the arm's change from today, 3 sd)
  classifier    (alloc_4444, alloc_6604, alloc_4444_wholesensor) actual arm against emulated, HGB, 5 folds by event:
                (AUC_e - 0.5) > max(0.25 (AUC_0 - 0.5), 0.01, 3 sd), AUC_0 = actual arm against today, judged only where
                AUC_0 - 0.5 >= 0.02; sd never below the Mann-Whitney value
sd: paired event bootstrap (the emulation's own draw included: the draws are tied to the tracks).
Reported only (minor): head-output medians and 16 / 84 % quantiles, the changed share, kappa (pairs of tracks within
Delta R < 0.4 both changed, over the squared changed share).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import param_model as PM

N_BOOT = 100
CLASSIFIER_ALLOCS = ("alloc_4444", "alloc_6604", "alloc_4444_wholesensor")
FIELDS = ("a", "cov_diag", "refit", "inputs", "margins", "same_clusters", "forced_a", "forced_cov_diag", "forced_refit")


def load(root, scen, files, allocs):
    """(truth dict, today dict, {alloc: arm dict}) concatenated over the files."""
    import ext_scan as XS
    root = Path(root)
    T, arms = [], {}
    names = ["implemented_now"] + list(allocs)
    for f in files:
        T.append(dict(np.load(root / "truth" / f"{scen}_f{f:02d}_truth.npz")))
        got = {}
        for u in XS.units_of([f], [scen]):
            p = root / "units" / f"{u['name']}.npz"
            if not (set(u["arms"]) & set(names)) or not p.exists():
                continue
            Z = np.load(p)
            for n in set(u["arms"]) & set(names):
                got[n] = {k: Z[f"{n}__{k}"] for k in FIELDS}
        for n in names:
            arms.setdefault(n, []).append(got[n])
    T = {k: np.concatenate([t[k] for t in T]) for k in T[0]}
    arms = {n: {k: np.concatenate([d[k] for d in v]).astype(np.float64 if v[0][k].dtype.kind == "f" else v[0][k].dtype)
                for k in FIELDS} for n, v in arms.items()}
    today = arms.pop("implemented_now")
    m = T["matched"] & today["refit"] & (T["tp_pt"] > 3)
    kpt = float(np.median(np.abs(today["a"][m, 0]) * T["tp_pt"][m]))
    sgn = float(np.sign(np.median(today["a"][m, 0] * T["tp_charge"][m])))
    today.update(matched=T["matched"], kpt=kpt)
    truth = np.stack([sgn * T["tp_charge"] * kpt / np.maximum(T["tp_pt"], 1e-9), T["tp_phi0"], np.sinh(T["tp_eta"]),
                      T["tp_z0"], T["tp_d0"]], 1).astype(np.float64)
    truth[~T["matched"]] = np.nan
    T["event_key"] = (T["run"] * 1000003 + T["lumi"]) * 1000003 + T["event"]
    return T, today, arms, truth, {"kpt": kpt, "rinv_sign": sgn}


def crossfit(today, arm, truth, D, terc, fold):
    """The emulated arm for every track: tables fitted on the other fold."""
    out = None
    for k in (0, 1):
        tab = PM.fit(today, arm, truth, D, terc, fold != k)          # fitted on the other fold
        em = PM.apply(tab, today, truth, D, terc)
        if out is None:
            out = {key: np.array(v, copy=True) for key, v in em.items()}
        else:
            for key, v in em.items():
                out[key][fold == k] = v[fold == k]
    return out


def boot_weights(ev, n=N_BOOT, seed=20261111):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(ev, return_inverse=True)
    return [rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv] for _ in range(n)]


def wq(x, w, q=0.6827):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][min(np.searchsorted(c, q * c[-1]), len(x) - 1)] if len(x) and c[-1] > 0 else np.nan


def stat_with_sd(fn, W, m):
    """fn(weights) on the tracks m: value and bootstrap sd."""
    val = fn(np.ones(int(m.sum())))
    reps = [fn(w[m].astype(float)) for w in W]
    return float(val), float(np.nanstd(reps))


def closure_of(T, today, arm, em, truth, W):
    res = {"material": [], "core": {}, "prompt_tail": {}, "end_to_end": {}}
    sg = PM.sigma_today(today)
    r = {k: PM.residual(v["a"].astype(np.float64).copy(), truth) for k, v in (("today", today), ("act", arm), ("emu", em))}
    aeta, pt = np.abs(T["tp_eta"]), T["tp_pt"]
    matched = T["matched"]
    for qi, q in ((4, "d0"), (3, "z0")):
        for ie, (elo, ehi) in enumerate(((0, 1.1), (1.1, 1.8), (1.8, 2.4))):
            for ip, (plo, phi) in enumerate(((2, 5), (5, 1e9))):
                m = matched & (aeta >= elo) & (aeta < ehi) & (pt >= plo) & (pt < phi)
                if m.sum() < 200:
                    continue
                x = {k: np.abs(v[m, qi]) * 1e4 for k, v in r.items()}
                w_t, w_a, w_e = (float(np.quantile(x[k], 0.6827)) for k in ("today", "act", "emu"))
                _, sd = stat_with_sd(lambda w: wq(x["emu"], w) - wq(x["act"], w), W, m)
                tol = max(0.25 * abs(w_a - w_t), 0.02 * w_a, 3 * sd)
                key = f"{q}|eta{elo}-{ehi}|pt{plo}-{'inf' if phi > 1e8 else phi}"
                res["core"][key] = {"n": int(m.sum()), "w68_today_um": w_t, "w68_actual_um": w_a, "w68_emulated_um": w_e,
                                    "sd": sd, "tolerance": tol, "pass": bool(abs(w_e - w_a) <= tol)}
                if abs(w_e - w_a) > tol:
                    res["material"].append(f"core {key}: emulated {w_e:.2f} vs actual {w_a:.2f} um (tolerance {tol:.2f})")
        for plo, phi in ((2, 5), (5, 1e9)):
            m = matched & (T["origin"] == 0) & (pt >= plo) & (pt < phi)
            if m.sum() < 200:
                continue
            t = {k: (np.abs(v[m, qi]) > 3 * sg[m, qi]).astype(float) for k, v in r.items()}
            s_t, s_a, s_e = (float(t[k].mean()) for k in ("today", "act", "emu"))
            _, sd = stat_with_sd(lambda w: (np.sum(w * t["emu"]) - np.sum(w * t["act"])) / max(w.sum(), 1), W, m)
            tol = max(0.25 * abs(s_a - s_t), 0.1 * s_a, 3 * sd)
            key = f"{q}|pt{plo}-{'inf' if phi > 1e8 else phi}"
            res["prompt_tail"][key] = {"n": int(m.sum()), "share_today": s_t, "share_actual": s_a, "share_emulated": s_e, "sd": sd,
                                       "tolerance": tol, "pass": bool(abs(s_e - s_a) <= tol)}
            if abs(s_e - s_a) > tol:
                res["material"].append(f"prompt tail {key}: emulated {s_e:.4f} vs actual {s_a:.4f} (tolerance {tol:.4f})")
    # end to end: b/c daughters against prompt by |d0| / sigma(d0)
    sel = matched & (pt >= 2) & (aeta < 2.4)
    hf, pr = sel & (T["origin"] == 1), sel & (T["origin"] == 0)
    sip = {k: np.abs(v["a"][:, 4]) / np.sqrt(np.maximum(v["cov_diag"][:, 4], 1e-30)) for k, v in (("today", today), ("act", arm), ("emu", em))}
    both = hf | pr

    def e2e(k, w, eff=None):
        s, wh, wp = sip[k][both], w * hf[both], w * pr[both]
        if eff is None:
            return float(np.sum(wp * (s > 3)) / max(wp.sum(), 1))
        cut = wq(s[hf[both]], wh[hf[both]], 1 - eff)
        return float(np.sum(wp * (s > cut)) / max(wp.sum(), 1))
    for name, eff in (("prompt_eff_at_bc_50", 0.5), ("prompt_eff_at_bc_70", 0.7), ("prompt_eff_at_bc_90", 0.9), ("prompt_share_sip_gt_3", None)):
        v = {k: e2e(k, np.ones(int(both.sum())), eff) for k in sip}
        _, sd = stat_with_sd(lambda w: e2e("emu", w, eff) - e2e("act", w, eff), W, both)
        tol = max(0.25 * abs(v["act"] - v["today"]), 3 * sd)
        res["end_to_end"][name] = {**v, "sd": sd, "tolerance": tol, "pass": bool(abs(v["emu"] - v["act"]) <= tol)}
        if abs(v["emu"] - v["act"]) > tol:
            res["material"].append(f"end to end {name}: emulated {v['emu']:.4f} vs actual {v['act']:.4f} (tolerance {tol:.4f})")
    # reported only
    res["changed_share"] = {"actual": float((~arm["same_clusters"]).mean()), "emulated": float(em["changed"].mean())}
    hq = lambda x: [float(v) for v in np.quantile(x[np.isfinite(x)], [0.16, 0.5, 0.84])]
    res["head_outputs_q16_q50_q84"] = {h: {"today": hq(today["margins"][:, i]), "actual": hq(arm["margins"][:, i]),
                                           "emulated": hq(em["margins"][:, i])} for i, h in enumerate(PM.HEADS)}
    return res


def features(T, today, x):
    cols = [x["a"], np.log(np.maximum(x["cov_diag"], 1e-30)), x["margins"]]
    if "inputs" in x:
        cols += [np.stack([x["inputs"][:, PM.IN[c]] for c in PM.COUNTS + PM.CHI2], 1)]
    else:
        cols += [np.stack([x[c] for c in PM.COUNTS + PM.CHI2], 1)]
    window = np.where(today["refit"], (np.abs(today["a"][:, 4]) > PM.D0_SPLIT).astype(float), 2.0)
    return np.column_stack(cols + [window, T["origin"].astype(float)])


def classifier_auc(Xa, Xb, ev, seed=1):
    """AUC of an HGB classifier telling Xa from Xb, 5 folds by event (both copies of a track in one fold)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    fold = np.r_[ev, ev] % 5
    p = np.zeros(len(y))
    for k in range(5):
        m = fold == k
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        clf.fit(X[~m], y[~m])
        p[m] = clf.predict_proba(X[m])[:, 1]
    aucs = [roc_auc_score(y[fold == k], p[fold == k]) for k in range(5)]
    n = len(Xa)
    return float(roc_auc_score(y, p)), float(max(np.std(aucs) / np.sqrt(5), np.sqrt((2 * n + 1) / (12.0 * n * n))))


def kappa(T, changed):
    """P(both changed) / P(changed)^2 for pairs of tracks of one event within Delta R < 0.4 (OT pT >= 2 GeV)."""
    ok = T["ot_pt"] >= 2
    ev, eta, phi, ch = T["event_key"][ok], T["ot_eta"][ok], T["ot_phi"][ok], changed[ok]
    o = np.argsort(ev, kind="stable")
    ev, eta, phi, ch = ev[o], eta[o], phi[o], ch[o]
    st = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1, len(ev)]
    both = pairs = 0
    for i in range(len(st) - 1):
        s = slice(st[i], st[i + 1])
        d = np.hypot(eta[s][:, None] - eta[s][None, :], PM.wrap(phi[s][:, None] - phi[s][None, :]))
        m = np.triu(d < 0.4, 1)
        pairs += int(m.sum())
        both += int((m & ch[s][:, None] & ch[s][None, :]).sum())
    p = float(ch.mean())
    return {"pairs": pairs, "changed_share": p, "kappa": (both / pairs) / (p * p) if pairs and p > 0 else float("nan")}


def run(a):
    import ext_scan as XS
    import study as ST
    root = Path(a.root) if a.root else XS.out_root()
    allocs = a.allocs or [x for x in XS.SHIPPED if x != "implemented_now"]
    for scen in a.scen:
        T, today, arms, truth, meta = load(root, scen, a.files, allocs)
        D = PM.draws(T["run"], T["lumi"], T["event"], T["track"])
        fold = (PM._mix(T["event_key"].astype(np.uint64)) % np.uint64(2)).astype(int)
        terc = np.quantile(today["margins"][today["refit"], 0], [1 / 3, 2 / 3])
        W = boot_weights(T["event_key"])
        res = {"scenario": scen, "files": list(a.files), "tracks": int(len(T["run"])), "events": int(len(np.unique(T["event_key"]))),
               "meta": meta, "clean_tercile_edges": [float(x) for x in terc], "allocations": {}, "material": []}
        for al in allocs:
            em = crossfit(today, arms[al], truth, D, terc, fold)
            c = closure_of(T, today, arms[al], em, truth, W)
            if al in CLASSIFIER_ALLOCS and not a.no_classifier:
                ev = np.unique(T["event_key"], return_inverse=True)[1]
                Xa, Xe, X0 = features(T, today, arms[al]), features(T, today, em), features(T, today, today)
                auc_e, sd_e = classifier_auc(Xa, Xe, ev)
                auc_0, _ = classifier_auc(Xa, X0, ev)
                tol = max(0.25 * (auc_0 - 0.5), 0.01, 3 * sd_e)
                judged = auc_0 - 0.5 >= 0.02
                c["classifier"] = {"auc_actual_vs_emulated": auc_e, "sd": sd_e, "auc_actual_vs_today": auc_0, "tolerance": tol,
                                   "judged": bool(judged), "pass": bool((auc_e - 0.5) <= tol or not judged)}
                if judged and (auc_e - 0.5) > tol:
                    c["material"].append(f"classifier: AUC actual vs emulated {auc_e:.3f} (tolerance 0.5 + {tol:.3f}; actual vs today {auc_0:.3f})")
                c["kappa_actual"] = kappa(T, ~arms[al]["same_clusters"])
                c["kappa_emulated"] = kappa(T, em["changed"])
            res["allocations"][al] = c
            res["material"] += [f"{al}: {m}" for m in c["material"]]
            print(scen, al, "material:", len(c["material"]), "| changed", round(c["changed_share"]["actual"], 3),
                  round(c["changed_share"]["emulated"], 3), flush=True)
        out = (Path(a.out) if a.out else ST.LOCAL / "checks") / f"param_closure_{scen}.json"
        json.dump(res, open(out, "w"), indent=1, default=float)
        print(scen, "material findings:", len(res["material"]))


if __name__ == "__main__":
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--allocs", nargs="+", default=None)
    ap.add_argument("--files", type=int, nargs="+", default=list(ST.SCAN_FILES))
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-classifier", action="store_true")
    run(ap.parse_args())
