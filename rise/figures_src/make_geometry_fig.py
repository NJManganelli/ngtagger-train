"""Schematic for the digiRefit idea slide: OT track projected into TBPX,
per-layer digi windows, KF update near the vertex.

Pure schematic (no data), drawn to real radii: TBPX layers ~3.0/6.8/10.9/16.0 cm,
OT barrel layers 25.0/37.2/52.2/68.7/86.0/108.6 cm.

Run:  pixi run python rise/figures_src/make_geometry_fig.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "figures", "digirefit_idea.png")

IT_R = [3.0, 6.8, 10.9, 16.0]
OT_R = [25.0, 37.2, 52.2, 68.7, 86.0, 108.6]


def helix(r_inv, phi0, d0, s):
    x0, y0 = d0 * np.sin(phi0), -d0 * np.cos(phi0)
    psi = r_inv * s
    x = x0 + (np.sin(phi0 + psi) - np.sin(phi0)) / r_inv
    y = y0 - (np.cos(phi0 + psi) - np.cos(phi0)) / r_inv
    return x, y


def main():
    plt.rcParams.update({"font.size": 16})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 7.0),
                               gridspec_kw={"width_ratios": [1, 1]})
    # ---------------- left: full view ----------------
    th = np.linspace(0, np.pi / 2, 100)
    for r in OT_R:
        a.plot(r * np.cos(th), r * np.sin(th), color="#c8b89a", lw=1.6)
    for r in IT_R:
        a.plot(r * np.cos(th), r * np.sin(th), color="#9a9a9a", lw=1.2, ls=":")
    s = np.linspace(0, 118, 300)
    x, y = helix(-1 / 350.0, np.deg2rad(55), 0.05, s)
    a.plot(x, y, color="#1b6ca8", lw=3)
    # stubs on OT layers
    for r in OT_R:
        i = np.argmin(np.abs(np.hypot(x, y) - r))
        a.plot(x[i], y[i], "s", ms=10, color="#0aa3a3", mec="k", zorder=5)
    a.annotate("outer-tracker stubs\n(the L1 track is fit out here)",
               xy=(x[np.argmin(np.abs(np.hypot(x, y) - 68.7))], y[np.argmin(np.abs(np.hypot(x, y) - 68.7))]),
               xytext=(8, 95), fontsize=15,
               arrowprops=dict(arrowstyle="->", lw=1.5))
    a.annotate("SmartPixels inner tracker\n(TBPX, r < 16 cm)", xy=(12, 9),
               xytext=(40, 18), fontsize=15, arrowprops=dict(arrowstyle="->", lw=1.5))
    a.set_xlim(0, 120); a.set_ylim(0, 120)
    a.set_aspect("equal")
    a.set_xlabel("x [cm]"); a.set_ylabel("y [cm]")
    a.set_title("the lever arm: OT track,\nextrapolated inward", fontsize=17)
    # ---------------- right: IT zoom ----------------
    for r in IT_R:
        b.plot(r * np.cos(th), r * np.sin(th), color="#9a9a9a", lw=2, ls=":")
        b.text(r * np.cos(np.deg2rad(12)) + 0.4, r * np.sin(np.deg2rad(12)), f"L{IT_R.index(r)+1}",
               fontsize=13, color="#666666")
    s = np.linspace(0, 20, 200)
    xs, ys = helix(-1 / 350.0, np.deg2rad(55), 0.05, s)     # seed
    xr, yr = helix(-1 / 350.0, np.deg2rad(55.9), 0.012, s)  # "refit" (shifted for clarity)
    b.plot(xs, ys, color="#1b6ca8", lw=3, label="OT-only seed (extrapolated)")
    b.plot(xr, yr, color="#d95f02", lw=3, ls="--", label="after KF update on IT hits")
    for r in IT_R:
        i = np.argmin(np.abs(np.hypot(xs, ys) - r))
        # window box around the crossing
        b.plot(xs[i], ys[i], "o", ms=6, color="#1b6ca8")
        b.add_patch(plt.Rectangle((xs[i] - 1.1, ys[i] - 1.1), 2.2, 2.2, fill=False,
                                  ec="#2ca02c", lw=2))
        i2 = np.argmin(np.abs(np.hypot(xr, yr) - r))
        b.plot(xr[i2], yr[i2], "*", ms=13, color="#2ca02c", mec="k", zorder=6)
    b.annotate("per-layer digi window:\nreal pixel digis + synthesized\ncluster angles ($\\alpha$, $\\beta$)",
               xy=(7.2, 3.3), xytext=(8.5, -2.6), fontsize=14,
               arrowprops=dict(arrowstyle="->", lw=1.5, color="#2ca02c"))
    b.plot(0.05 * np.sin(np.deg2rad(55)), -0.05 * np.cos(np.deg2rad(55)), "x", ms=12,
           color="k", zorder=7)
    b.annotate("POCA: $d_0$, $z_0$ sharpen here", xy=(0.15, 0.0), xytext=(4.0, -4.6),
               fontsize=14, arrowprops=dict(arrowstyle="->", lw=1.5))
    b.set_xlim(-1, 19); b.set_ylim(-6, 19)
    b.set_aspect("equal")
    b.set_xlabel("x [cm]"); b.set_ylabel("y [cm]")
    b.legend(loc="upper left", fontsize=13)
    b.set_title("Tier-2 digiRefit: 5-par Kalman update\non SmartPixels hit windows", fontsize=17)
    fig.suptitle("digiRefit: take the 5-parameter outer-tracker L1 track, add inner-pixel hits near the vertex",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
