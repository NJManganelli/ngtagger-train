"""Read-only symmetry check of the disc beta angle errors (WP-C C1b, after J5; the physicist's request, 2026-10-17).

On the FIT files (A10; files 4 and 6 are not read), AAAA (one copy of each event), the current calibrated disc model
(the smooth seed, A16; the fitted disc material scale): every own class-T disc cluster with a clean history (D1's
population). The beta innovation pull is D1's: (m' - h) / sqrt(sigma'^2 + HCH), m' and sigma' from the disc row.

Figures (wpc/c1b/): beta_symmetry (per disc: the pull width and centre against the signed local incidence angle
atan(k), per facing x z side and pooled; the wrong-side-of-zero test: the row at the predicted key, at the true key, and
same-sign clusters only; the wrong-sign fraction), beta_symmetry_mirror (width at +k and -k against |angle| per facing x
side), beta_vs_true_angle (the pooled width against the true beta angle atan(cotBeta)).

Jobs: collect (through ops/spheavy), plots (light).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import arms as AR
import c1b_overlap as OV
import common as CM
import diagnostics as DG
import disc_angles as DA
import gates as G
import seedmodel as SMOD
from common import OP
from ngtagger.spdefs import angles as SA

OUT = CM.OUT / "c1b"
ROWS = OUT / "beta_symmetry_rows.npz"
NEV_FILE = OV.NEV_FILE
N_REP = 100
FAIL_BIN = (-0.2, -0.1)                              # J5's failing key bin (discs 12-14 on file 4)
DISCS = (11, 12, 13, 14)


def collect(a):
    with open(OUT / "scale" / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    geo = OP.load_geometry(CM.GEOMETRY)
    keep = ("surface", "cot_pred", "m_raw", "sig_pub", "hch", "facing", "cot_true", "has", "table_pop", "clean", "mi", "dkeybin")
    parts, t0 = [], time.time()
    for fi, f in enumerate(CM.FIT_FILES):
        path = CM.input_path("AAAA", f)
        cfg = CM.producer_config(path)
        evr = fi * NEV_FILE + CM.event_rank(path)
        tools = AR.Tools(cfg, disc=AR.DiscTools(disc_scale=scale))
        arm = AR.inn_disc(cfg)
        for start, A in CM.iterate(path, AR.branches(path), chunk=10, nev=a.nev):
            ctx = AR.prepare(A, geo)
            R = DG.mark_primary(AR.run_arm(ctx, geo, cfg, "AAAA", arm, tools, own=True)["own"])
            m = R["table_pop"] & R["clean"] & (R["surface"] > 10)
            parts.append({**{k: R[k][m] for k in keep}, "ev": evr[R["ev"][m] + start],
                          "zside": np.sign(geo["origin"][R["mi"][m]][:, 2]).astype(np.int8)})
        print(f"[{time.time() - t0:6.1f}s] AAAA f{f:02d}", flush=True)
    P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(ROWS, scale=scale, **P)
    print(f"{len(P['surface'])} disc rows", flush=True)
    return 0


def deg(k):
    """The angle atan(k) in degrees."""
    return np.degrees(np.arctan(k))


def _pull(P, imp, rows):
    m, s = imp.calibrate("beta", P["m_raw"][:, 1], P["sig_pub"][:, 1], rows)
    with np.errstate(invalid="ignore"):
        return (m - P["cot_pred"][:, 1]) / np.sqrt(s ** 2 + P["hch"][:, 1])


def _stats(x, ev, reps):
    """D1's (centre, width) as judged (the table's 25 %-trimmed mean and weighting width) and the clipped-likelihood
    scale (A18), each with its event-bootstrap sd, and n."""
    ok = np.isfinite(x)
    x, ev = x[ok], ev[ok]
    if len(x) < SA.MIN_FIT:
        return None
    c, w = G._bin_stats(x)
    B = np.array([G._bin_stats(x, r[ev]) for r in reps])
    s, sd_s = SMOD.clip_scale(x, ev, reps)
    return {"n": int(len(x)), "centre": c, "width": w, "sd_c": float(np.nanstd(B[:, 0])), "sd_w": float(np.nanstd(B[:, 1])),
            "scale": s, "sd_s": sd_s}


def plots(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(ROWS)
    P = {k: z[k] for k in z.files if k != "scale"}
    imp = DA.DiscImport()
    hasb = P["has"][:, 1].astype(bool)
    rows_pred = imp.rows(P["surface"], P["cot_pred"][:, 0], P["cot_pred"][:, 1])
    rows_true = imp.rows(P["surface"], P["cot_true"][:, 0], P["cot_true"][:, 1])
    xa, xb = _pull(P, imp, rows_pred), _pull(P, imp, rows_true)
    kp = SA.local_incidence_key(P["cot_pred"][:, 0], P["cot_pred"][:, 1])
    kt = SA.local_incidence_key(P["cot_true"][:, 0], P["cot_true"][:, 1])
    same = np.sign(kp) == np.sign(kt)
    reps = G.replicas(nev=NEV_FILE * len(CM.FIT_FILES), n=N_REP, seed=G.SEED)
    se = SA.signed_edges(SA.DISC_ABS_EDGES)
    nb = len(se) - 1
    lo_d, hi_d = deg(np.clip(se[:-1], -3.0, 3.0)), deg(np.clip(se[1:], -3.0, 3.0))
    ctr = 0.5 * (lo_d + hi_d)
    fb = P["dkeybin"]                                 # D1's bin: the row key of the predicted cots
    INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#d9d8d4", "#efeee9"
    COL = {(1, 1): "#2a78d6", (1, -1): "#eb6834", (-1, 1): "#1baf7a", (-1, -1): "#4a3aa7"}
    POOL = INK

    def style(ax):
        ax.grid(True, color=GRID, lw=0.5, axis="y")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7)
        for e in deg(np.clip(se[1:-1], -3, 3)):
            ax.axvline(e, color=GRID, lw=0.6)
        ax.axvspan(deg(FAIL_BIN[0]), deg(FAIL_BIN[1]), color="#fbe3d6", lw=0, zorder=0)

    def table(x, sel):
        out = []
        for b in range(nb):
            m = sel & hasb & (fb == b)
            out.append(_stats(x[m], P["ev"][m], reps))
        return out

    fig, axs = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
    summary = {}
    for i, S in enumerate(DISCS):
        on = P["surface"] == S
        xp = [float(np.median(deg(kp[on & hasb & (fb == b)]))) if np.any(on & hasb & (fb == b)) else ctr[b]
              for b in range(nb)]                     # a bin's points at its clusters' median angle
        series = {fz: table(xa, on & (P["facing"] == fz[0]) & (P["zside"] == fz[1])) for fz in COL}
        pooled = table(xa, on)
        pb, pc = table(xb, on), table(xa, on & same)
        summary[S] = {"angle": xp, "pooled": pooled, "true_key": pb, "same_sign": pc, "series": {str(k): v for k, v in series.items()}}
        for j in range(4):
            style(axs[i, j])
        ax_w, ax_c, ax_h, ax_f = axs[i]
        for c_ in (ax_w, ax_h):
            c_.axhspan(0.9, 1.1, color=BAND, lw=0, zorder=0)
            c_.axhline(1.0, color=INK2, lw=0.6)
        ax_c.axhspan(-0.25, 0.25, color=BAND, lw=0, zorder=0)
        ax_c.axhline(0.0, color=INK2, lw=0.6)
        for (fa, zs), tab in series.items():
            pts = [(xp[b], t) for b, t in enumerate(tab) if t]
            lab = f"facing {'+' if fa > 0 else '−'}, {'+' if zs > 0 else '−'}z"
            ax_w.errorbar([p[0] for p in pts], [p[1]["width"] for p in pts], yerr=[p[1]["sd_w"] for p in pts], fmt="o-",
                          ms=3, lw=1, color=COL[(fa, zs)], label=lab)
            ax_c.errorbar([p[0] for p in pts], [p[1]["centre"] for p in pts], yerr=[p[1]["sd_c"] for p in pts], fmt="o-",
                          ms=3, lw=1, color=COL[(fa, zs)], label=lab)
        pts = [(xp[b], t) for b, t in enumerate(pooled) if t]
        ax_w.plot([p[0] for p in pts], [p[1]["width"] for p in pts], "s-", color=POOL, lw=2, ms=5, label="pooled")
        ax_w.plot([p[0] for p in pts], [p[1]["scale"] for p in pts], "--", color=POOL, lw=1,
                  label="pooled, clipped-likelihood scale")
        ax_c.plot([p[0] for p in pts], [p[1]["centre"] for p in pts], "s-", color=POOL, lw=2, ms=5, label="pooled")
        for x_, t in pts:
            ax_w.annotate(f"{t['n']}", (x_, t["width"]), textcoords="offset points", xytext=(0, 6), ha="center",
                          fontsize=6, color=INK2)
        for tab, c_, lab in ((pooled, POOL, "(a) row at the predicted key"), (pb, "#2a78d6", "(b) row at the true key"),
                             (pc, "#eb6834", "(c) predicted and true key of the same sign")):
            q = [(xp[b], t) for b, t in enumerate(tab) if t]
            ax_h.errorbar([p[0] for p in q], [p[1]["width"] for p in q], yerr=[p[1]["sd_w"] for p in q], fmt="o-", ms=3,
                          lw=1.4, color=c_, label=lab)
        wrong, moved = [], []
        for b in range(nb):
            m = on & hasb & (fb == b)
            ok = m.sum() >= SA.MIN_FIT
            wrong.append(float(np.mean(~same[m])) if ok else np.nan)
            moved.append(float(np.mean(rows_true[m] != rows_pred[m])) if ok else np.nan)
        ax_f.bar(ctr, moved, width=0.8 * (hi_d - lo_d), color="#2a78d6", label="row at the true key ≠ row at the predicted key")
        ax_f.plot(ctr, wrong, "o", color="#eb6834", ms=4, label="predicted key of the wrong sign")
        kmin = np.min(np.abs(kp[on & hasb]))
        ax_f.text(0.02, 0.95, f"wrong sign: {int(np.sum(~same[on & hasb]))} of {int(np.sum(on & hasb))} clusters\n"
                  f"smallest |predicted angle| {deg(kmin):.1f}°", transform=ax_f.transAxes, va="top", fontsize=7, color=INK2)
        summary[S]["wrong_sign_fraction"], summary[S]["row_change_fraction"] = wrong, moved
        ax_w.set_ylabel(f"ID{S - 10}\nβ pull width")
        ax_c.set_ylabel("β pull centre (bias)")
        ax_h.set_ylabel("β pull width, pooled")
        ax_f.set_ylabel("fraction of the bin's clusters")
        for ax_ in (ax_w, ax_h):
            ax_.set_ylim(0.6, 1.6)
        ax_c.set_ylim(-0.8, 0.8)
        ax_f.set_ylim(0, 0.1)
        if i == 0:
            ax_w.set_title("width per facing × z side (D1's statistic, as judged)")
            ax_c.set_title("centre per facing × z side")
            ax_h.set_title("the wrong-side-of-zero test")
            ax_f.set_title("wrong-sign keys and row changes")
            for ax_ in (ax_w, ax_h, ax_f):
                ax_.legend(fontsize=6, frameon=False, loc="upper left")
        if i == 3:
            for ax_ in axs[i]:
                ax_.set_xlabel("signed local incidence angle atan(k) [deg]")
    fig.text(0.5, 0.005, "canon_v4 AAAA fit files 1, 2, 3, 5, 7-10; the calibrated disc arm (smooth seed, disc material "
             f"scale {float(z['scale']):.2f}); own class-T disc clusters, clean history. Grey band 1 ± 0.1 (width) / "
             "0 ± 0.25 (centre); orange span: J5's failing key bin (−0.2, −0.1); light lines: the table's bin edges; "
             "points at the median angle of the bin's clusters; numbers: clusters per bin.", ha="center", color=INK2,
             fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"beta_symmetry.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # the mirror: the width at k < 0 and k > 0 against |angle|. The sign of k is set by facing x z side (k < 0 where the
    # module normal points along z on that side), so per facing the mirror pairs the two z sides.
    NEG, POS = "#eb6834", "#2a78d6"
    neg_bins = [b for b in range(nb) if se[b + 1] <= 0]
    pos_bins = [b for b in range(nb) if se[b] >= 0]
    pairs = (("facing +", (1, 1), (1, -1)), ("facing −", (-1, -1), (-1, 1)), ("pooled", None, None))
    fig, axs = plt.subplots(4, 3, figsize=(13, 12), sharex=True, sharey=True)
    for i, S in enumerate(DISCS):
        for j, (title, fn, fp) in enumerate(pairs):
            ax = axs[i, j]
            ax.grid(True, color=GRID, lw=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=7)
            ax.axhspan(0.9, 1.1, color=BAND, lw=0, zorder=0)
            ax.axhline(1.0, color=INK2, lw=0.6)
            ax.axvspan(deg(0.1), deg(0.2), color="#fbe3d6", lw=0, zorder=0)
            for e in deg(np.clip(se[se > 0], 0, 3))[:-1]:
                ax.axvline(e, color=GRID, lw=0.6)
            for bins, f_, col, mk in ((neg_bins, fn, NEG, "s--"), (pos_bins, fp, POS, "o-")):
                tab = summary[S]["pooled"] if f_ is None else summary[S]["series"][str(f_)]
                q = [(abs(summary[S]["angle"][b]), tab[b]) for b in bins if tab[b]]
                side = "" if f_ is None else f", {'+' if f_[1] > 0 else '−'}z"
                ax.errorbar([p[0] for p in q], [p[1]["width"] for p in q], yerr=[p[1]["sd_w"] for p in q], fmt=mk, ms=4,
                            lw=1.3, color=col, label=f"k {'<' if bins is neg_bins else '>'} 0{side}")
                for x_, t in q:
                    ax.annotate(f"{t['n']}", (x_, t["width"]), textcoords="offset points",
                                xytext=(0, 6 if bins is pos_bins else -11), ha="center", fontsize=6, color=col)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"ID{S - 10}\nβ pull width (D1's statistic)")
            if i == 3:
                ax.set_xlabel("|signed local incidence angle| [deg]")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
            ax.set_ylim(0.6, 1.6)
    fig.text(0.5, 0.005, "Shaded: J5's failing bin (−0.2, −0.1) and its mirror (0.1, 0.2), |angle| 5.7-11.3°; grey band "
             "1 ± 0.1; light lines: the table's bin edges; points at the median angle of the bin's clusters; numbers: clusters "
             "per bin.", ha="center", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"beta_symmetry_mirror.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # the pooled width against the true beta angle
    tb = np.degrees(np.arctan(P["cot_true"][:, 1]))
    edges = np.arange(-40, 40.1, 4.0)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, S in zip(axs, DISCS):
        ax.grid(True, color=GRID, lw=0.5)
        ax.axhspan(0.9, 1.1, color=BAND, lw=0, zorder=0)
        ax.axhline(1.0, color=INK2, lw=0.6)
        on = (P["surface"] == S) & hasb
        q = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            t = _stats(xa[on & (tb >= lo) & (tb < hi)], P["ev"][on & (tb >= lo) & (tb < hi)], reps)
            if t:
                q.append((0.5 * (lo + hi), t))
        ax.errorbar([p[0] for p in q], [p[1]["width"] for p in q], yerr=[p[1]["sd_w"] for p in q], fmt="s-", color=INK,
                    ms=4, lw=1.5)
        for x_, t in q:
            ax.annotate(f"{t['n']}", (x_, t["width"]), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=6, color=INK2)
        ax.set_title(f"ID{S - 10}")
        ax.set_xlabel("true β angle atan(cotβ) [deg]")
        ax.set_ylim(0.6, 1.6)
    axs[0].set_ylabel("β pull width, pooled (row at the predicted key)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"beta_vs_true_angle.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(OUT / "beta_symmetry.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("collect", "plots"))
    ap.add_argument("--nev", type=int, default=None)
    a = ap.parse_args(argv)
    return {"collect": collect, "plots": plots}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
