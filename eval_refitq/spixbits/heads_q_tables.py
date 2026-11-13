"""Tables of the heads retrained per quantization scenario, from heads_q_eval.py's outputs (reading rules of
reviews/stats-review-03-heads-per-quantization.md).

  python heads_q_tables.py [--out FILE.md]   -> results.json beside the evaluation, and the markdown tables

Per detector scenario, window (and both windows together) and quantization scenario, for the head a scanned output
drives (d0 for y and the bending angle beta, z0 for x and alpha; both for the 16-bit allocations), on the test events:
  W a / b / c   effective width exp(L_abs - 1/2) [um] of: a the scenario's own heads, b the five-file "implemented now"
                heads on the scenario's inputs, c those heads on today's inputs (L_abs with the common cap; c is another
                population: today's window routing). For a perfect width head W is the residual's rms.
  G             L(b) - L(a), nats per track, on the scenario's own problem = recalibration [L(b) - L(b')] + retraining
                proper [L(b') - L(a)]; +- sqrt(bootstrap^2 + trial^2 + training-sample^2)
  Gm            the same between the 20-trial means (the stable estimator)
  X, rest       (L_abs(b) - L_abs(a)) / (L_abs(b) - L_abs(c)) and L_abs(a) - L_abs(c): what retraining recovers and
                what stays (which includes the resolution no retraining can recover)
Trend: per scan, the slope of G against bits in each of 200 event-bootstrap replicas (the same event weights for every
quantization scenario); claimed only beyond 3 sd.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import retrain_heads_q as RQ

SCEN = ("AAAA", "AAAI", "IAAA")
WIN = ("prompt", "displaced")
HEADS = ("clean", "d0", "z0", "tanl", "rinv", "phi")
N_BOOT = 200
up = lambda o, lo, hi: [(4, "alloc_4444")] + [(b, f"up_{o}_{b}b") for b in range(5, hi + 1)] + [("float", f"up_{o}_float")]
down = lambda o, lo, hi, ns: ([("not sent", f"down_{o}_notsent")] if ns else []) + [(b, f"down_{o}_{b}b") for b in range(lo, hi + 1)] + [("float", "network_float")]
FAMILIES = [
    ("x upwards from the 4/4/4/4 floor (y, α, β at 4 bits)", ("z0",), up("x", 5, 9)),
    ("y upwards from the 4/4/4/4 floor (x, α, β at 4 bits)", ("d0",), up("y", 5, 11)),
    ("α upwards from the 4/4/4/4 floor (x, y, β at 4 bits)", ("z0",), up("alpha", 5, 7)),
    ("β upwards from the 4/4/4/4 floor (x, y, α at 4 bits)", ("d0",), up("beta", 5, 7)),
    ("x downwards from network float (the other three unquantized)", ("z0",), down("x", 2, 8, False)),
    ("y downwards from network float (the other three unquantized)", ("d0",), down("y", 2, 10, False)),
    ("α downwards from network float (the other three unquantized)", ("z0",), down("alpha", 2, 7, True)),
    ("β downwards from network float (the other three unquantized)", ("d0",), down("beta", 2, 7, True)),
    ("16-bit allocations x/y/α/β", ("d0", "z0"), [(f"{n[6]}/{n[7]}/{n[8]}/{n[9]}", n) for n in (
        "alloc_4444", "alloc_5533", "alloc_6622", "alloc_5623", "alloc_4633", "alloc_4723", "alloc_5722", "alloc_6703", "alloc_6604",
        "alloc_7702", "alloc_8800")]),
]


def mean_of(Z, w, grp, k, weights=None, wins=None):
    """Mean per-track loss per head of model k from the per-event sums, over the windows `wins`, with event weights."""
    s = sum(Z[f"{x}/{grp}/{k}/sum"].astype(np.float64) for x in (wins or (w,)))
    c = sum(Z[f"{x}/{grp}/{k}/count"].astype(np.float64) for x in (wins or (w,)))
    if weights is None:
        return s.sum(0) / np.maximum(c.sum(0), 1)
    return (weights @ s) / np.maximum(weights @ c, 1)


def entry(J, Z, TS, w, W):
    """The numbers of one quantization scenario in one window (w = "prompt", "displaced" or "both")."""
    wins = WIN if w == "both" else (w,)
    m = lambda grp, k, wt=None: mean_of(Z, w, grp, k, wt, wins)
    a, b, bp, bm, const = (m("own", k) for k in ("a", "b", "bp", "bm", "const"))
    G = b - a
    reps = np.array([m("own", "b", wt) - m("own", "a", wt) for wt in W])
    sd_boot = reps.std(0)
    comb = lambda key: np.sqrt(sum((np.array(J["windows"][x]["own"][key]) * J["windows"][x]["n_test"]) ** 2 for x in wins)) / sum(J["windows"][x]["n_test"] for x in wins)
    sd_trial = np.hypot(comb("a_trial_sd"), comb("b_trial_sd"))
    sd_train = np.sqrt(sum(np.array(TS[f"{n}/{x}"]["sd_per_head"]) ** 2 for n in (RQ.REFERENCE, "alloc_4444") for x in wins) / len(wins) ** (2 if w == "both" else 1))
    a_tm = sum(np.array(J["windows"][x]["own"]["a_trial_mean"]) * J["windows"][x]["n_test"] for x in wins) / sum(J["windows"][x]["n_test"] for x in wins)
    A, B, C_, BP = (m("abs", k) for k in ("a", "b", "c", "bp"))
    den = B - C_
    out = {"G": G, "G_recalibration": b - bp, "G_retraining": bp - a, "sd_boot": sd_boot, "sd_trial": sd_trial, "sd_train": sd_train,
           "sd_total": np.sqrt(sd_boot ** 2 + sd_trial ** 2 + sd_train ** 2), "G_trial_means": bm - a_tm,
           "L_abs_a": A, "L_abs_b": B, "L_abs_bp": BP, "L_abs_c": C_, "W_a": np.exp(A - 0.5), "W_b": np.exp(B - 0.5), "W_c": np.exp(C_ - 0.5),
           "dL_const": m("abs", "const") - m("abs", "const_today"), "X": np.where(np.abs(den) > 1e-9, (B - A) / den, np.nan), "rest": A - C_}
    out = {k: [float(x) for x in v] for k, v in out.items()}
    if w != "both":
        e = J["windows"][w]
        f = e["clean_fraction_test"]
        H = float(-(f * np.log(f) + (1 - f) * np.log(1 - f))) if 0 < f < 1 else float("nan")
        out.update(clean_fraction=f, clean_info_over_entropy={k: float((const[0] - v[0]) / H) for k, v in (("a", a), ("b", b), ("bp", bp))},
                   n_test=e["n_test"], n_test_today=e["n_test_today"], recalibration=e["recalibration"])
    out["_reps"] = reps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RQ.ROOT / "tables.md"))
    a = ap.parse_args()
    res, trends, lines, kb = {}, {}, [], {}
    hi = HEADS.index
    for s in SCEN:
        d = RQ.ROOT / "eval" / s
        TS = json.load(open(d / "train_spread.json"))
        names = sorted(p.stem for p in d.glob("*.json") if p.stem not in ("train_spread", RQ.REFERENCE))
        n_ev = len(np.load(d / f"{names[0]}.npz")["event_key"])
        rng = np.random.default_rng(20261113)
        W = [rng.multinomial(n_ev, np.full(n_ev, 1 / n_ev)).astype(np.float64) for _ in range(N_BOOT)]
        res[s] = {w: {} for w in WIN + ("both",)}
        for n in names:
            J, Z = json.load(open(d / f"{n}.json")), np.load(d / f"{n}.npz")
            kb[f"{s}/{n}"] = J["k_bend"]
            for w in WIN + ("both",):
                res[s][w][n] = entry(J, Z, TS, w, W)
            res[s]["migration_" + n] = J["window_migration_test_tracks"]
        for title, heads, items in FAMILIES:                 # slope of G against bits, per replica
            pts = [(b, n) for b, n in items if isinstance(b, int) and n in res[s]["prompt"]]
            for w in WIN + ("both",):
                for h in heads:
                    if len(pts) < 3 or title.startswith("16-bit"):
                        continue
                    x = np.array([b for b, _ in pts], float)
                    Y = np.stack([res[s][w][n]["_reps"][:, hi(h)] for _, n in pts], 1)
                    sl = ((x - x.mean()) * (Y - Y.mean(1, keepdims=True))).sum(1) / ((x - x.mean()) ** 2).sum()
                    y0 = np.array([res[s][w][n]["G"][hi(h)] for _, n in pts])
                    v = float(((x - x.mean()) * (y0 - y0.mean())).sum() / ((x - x.mean()) ** 2).sum())
                    trends[f"{s}|{w}|{h}|{title}"] = {"slope_nats_per_bit": v, "sd": float(sl.std()), "claimed": bool(abs(v) > 3 * sl.std()), "points": len(pts)}
    for s in SCEN:
        for w in WIN + ("both",):
            for n in list(res[s][w]):
                res[s][w][n].pop("_reps")
    kbv = np.array(list(kb.values()))
    json.dump({"results": res, "trends": trends, "k_bend_range": [float(kbv.min()), float(kbv.max())]}, open(RQ.ROOT / "results.json", "w"), indent=1)
    for w in WIN + ("both",):
        lines.append(f"\n## {'Both windows together' if w == 'both' else 'Window: ' + w}\n")
        for title, heads, items in FAMILIES:
            for h in heads:
                i = hi(h)
                lines.append(f"\n**{title} — {h} head**\n")
                lines.append("Cell: W a / b / c [µm]; G ± total (recalibration + retraining proper); X; rest.\n")
                lines.append("| bits | " + " | ".join(SCEN) + " |")
                lines.append("|---|" + "---|" * len(SCEN))
                for lab, n in items:
                    cells = []
                    for s in SCEN:
                        e = res[s][w].get(n)
                        cells.append("" if e is None else f"{e['W_a'][i]:.1f} / {e['W_b'][i]:.1f} / {e['W_c'][i]:.1f}; "
                                     f"{e['G'][i]:+.3f} ± {e['sd_total'][i]:.3f} ({e['G_recalibration'][i]:+.3f} {e['G_retraining'][i]:+.3f}); "
                                     f"X {e['X'][i]:.2f}; rest {e['rest'][i]:+.3f}")
                    lines.append(f"| {lab} | " + " | ".join(cells) + " |")
                t = [f"{s}: {trends[k]['slope_nats_per_bit']:+.4f} ± {trends[k]['sd']:.4f}{' (claimed)' if trends[k]['claimed'] else ''}"
                     for s in SCEN for k in [f"{s}|{w}|{h}|{title}"] if k in trends]
                if t:
                    lines.append("\nSlope of G per bit: " + "; ".join(t) + ".")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("claimed trends:", sum(t["claimed"] for t in trends.values()), "of", len(trends))


if __name__ == "__main__":
    main()
