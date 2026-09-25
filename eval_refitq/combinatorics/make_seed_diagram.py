import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rb  = np.array([851,1269,1784,2347,2936,3697])*120/4096     # OL1..OL6
zd  = np.array([2239,2645,3163,3782,4523])*120/2048          # D1..D5
rit = np.array([2.85, 5.95, 10.33, 14.50]); ZIT = 20.0
# REAL Phase-2 Inner Tracker discs, D121 geometry read from CVMFS
# (Geometry/TrackerCommonData/data/PhaseII/Tracker_DD4hep_compatible_2021_02/
#  pixel.xml; z = ZPixelForward 291 mm + the placed offset, radial extents from
#  the ITDisc Tubs). NOT SIMULATED here -- pixelAV has no angle regression for
#  these orientations -- but the seed design must assume they exist.
_off = np.array([-38.0, 31.95, 121.23, 235.2, 380.68, 551.38, 818.42, 1106.0,
                 1459.0, 1718.59, 2016.69, 2359.0])
zitd = (291.0 + _off) / 10.0
ritd_in = np.where(np.arange(12) < 8, 3.048, 6.238)
ritd_out = np.where(np.arange(12) < 8, 16.151, 25.510)
ZB, RD0, RD1 = 120.0, 20.0, 120.0
sh  = np.sinh

def zband(r, e0, e1):
    """z range on a barrel layer of radius r for an eta band, +- the z0 window."""
    z0 = 15.0
    a, b = r*sh(e0), r*sh(e1)
    return max(-ZB, min(a, b) - z0), min(ZB, max(a, b) + z0)

def rband(z, e0, e1):
    """r range on a disk at |z| for an eta band, clipped to the disk."""
    a, b = z/sh(max(e1, 1e-6)), z/sh(max(e0, 1e-6))
    return max(RD0, min(a, b)), min(RD1, max(a, b))

# seed: name, barrel layer indices, disk indices, eta band, colour
SEEDS = [
    ("L1L2", [0, 1], [],     0.00, 2.05, "#1f77b4"),
    ("L2L3", [1, 2], [],     1.14, 1.89, "#ff7f0e"),
    ("L3L4", [2, 3], [],     0.00, 1.36, "#2ca02c"),
    ("L5L6", [4, 5], [],     0.00, 0.95, "#d62728"),
    ("L1D1", [0],    [0],    1.79, 2.27, "#9467bd"),
    ("L2D1", [1],    [0],    1.14, 1.87, "#8c564b"),
    ("D1D2", [],     [0, 1], 1.07, 2.65, "#17becf"),
    ("D3D4", [],     [2, 3], 1.35, 2.90, "#7f7f7f"),
]
# extended triplets: outer PAIR + a third hit further IN (hatched)
TRIPLETS = [
    ("L3L4+L2", [2, 3], [1],  [], 0.00, 1.36, "#2ca02c"),
    ("L5L6+L4", [4, 5], [3],  [], 0.00, 0.95, "#d62728"),
    ("L2L3+D1", [1, 2], [],   [0], 1.14, 1.89, "#ff7f0e"),
    ("D1D2+L2", [],     [1],  [0, 1], 1.07, 2.65, "#17becf"),
]

fig, axes = plt.subplots(3, 1, figsize=(15, 16.0),
                         gridspec_kw=dict(height_ratios=[1, 1, 0.85], hspace=0.30))

def draw_detector(ax):
    for i, r in enumerate(rb):
        ax.plot([-ZB, ZB], [r, r], color="0.3", lw=2.4, solid_capstyle="butt", zorder=3)
        ax.text(0, r+2.2, "OL%d" % (i+1), ha="center", fontsize=7.5, color="0.35")
    for i, z in enumerate(zd):
        for s in (+1, -1):
            ax.plot([s*z, s*z], [RD0, RD1], color="0.55", lw=2.4, zorder=3)
            ax.text(s*z, RD1+3.0, "D%d" % (i+1), ha="center", fontsize=7.5, color="0.5")
    for r in rit:
        ax.plot([-ZIT, ZIT], [r, r], color="#0b5fa5", lw=2.2, zorder=4)
    # the IT discs that exist in the real detector but not in this simulation
    for i in range(12):
        for sgn in (+1, -1):
            ax.plot([sgn*zitd[i]]*2, [ritd_in[i], ritd_out[i]],
                    color="#0b5fa5", lw=1.6, ls=(0, (2, 2)), alpha=0.55, zorder=2)

    for e in (0.5, 1.0, 1.5, 2.0, 2.5):
        for s in (+1, -1):
            zz = np.array([0.0, 300.0]); rr = np.clip(zz/sh(e), 0, RD1)
            ax.plot(s*zz, rr, color="0.88", lw=0.7, ls=":", zorder=1)
        zt = min(RD1*sh(e), 297.0)
        ax.text(zt, min(zt/sh(e), RD1)+2, "%.1f" % e, fontsize=7, color="0.55", ha="center")
    ax.set_xlim(-300, 300); ax.set_ylim(0, 132)
    ax.set_xlabel("z  [cm]"); ax.set_ylabel("r  [cm]")
    ax.grid(alpha=0.12)

# ---- panel 1: the eight real PROMPT seeds -------------------------------
ax = axes[0]; draw_detector(ax)
for k, (nm, bl, dl, e0, e1, col) in enumerate(SEEDS):
    for li in bl:
        r = rb[li]
        for s in (+1, -1):
            lo, hi = zband(r, e0, e1)
            lo_s, hi_s = sorted([s*lo, s*hi])
            ax.add_patch(Rectangle((lo_s, r-2.0), hi_s-lo_s, 4.0, facecolor=col,
                                   alpha=0.5, edgecolor=col, lw=0.8, zorder=5))
    for di in dl:
        z = zd[di]
        for s in (+1, -1):
            r0, r1 = rband(z, e0, e1)
            ax.add_patch(Rectangle((s*z-6.0, r0), 12.0, r1-r0, facecolor=col,
                                   alpha=0.5, edgecolor=col, lw=0.8, zorder=5))
    ax.plot([], [], color=col, lw=6, alpha=0.6,
            label="%s   |$\\eta$| %.2f-%.2f" % (nm, e0, e1))
ax.legend(loc="lower center", ncol=4, fontsize=8, framealpha=0.95,
          bbox_to_anchor=(0.5, 0.005))
ax.set_title("The EIGHT real prompt seeds. Each band covers BOTH layers of the pair, "
             "over the z range that seed is active.\n"
             "SmartPixels IL1-IL4 in blue at r 2.9-14.5 cm, |z|<20 (barrel only: a "
             "pixelAV placeholder, not a final system)", fontsize=10.5, pad=10)

# ---- panel 2: the four EXTENDED triplets --------------------------------
ax = axes[1]; draw_detector(ax)
for nm, pb, tb, dlist, e0, e1, col in TRIPLETS:
    for li in pb:                                  # the seed PAIR: solid
        r = rb[li]
        for s in (+1, -1):
            lo, hi = zband(r, e0, e1); lo_s, hi_s = sorted([s*lo, s*hi])
            ax.add_patch(Rectangle((lo_s, r-2.0), hi_s-lo_s, 4.0, facecolor=col,
                                   alpha=0.5, edgecolor=col, lw=0.8, zorder=5))
    for li in tb:                                  # the THIRD hit: hatched
        r = rb[li]
        for s in (+1, -1):
            lo, hi = zband(r, e0, e1); lo_s, hi_s = sorted([s*lo, s*hi])
            ax.add_patch(Rectangle((lo_s, r-2.0), hi_s-lo_s, 4.0, facecolor="none",
                                   edgecolor=col, lw=1.2, hatch="////", zorder=5))
    for di in dlist:
        z = zd[di]
        for s in (+1, -1):
            r0, r1 = rband(z, e0, e1)
            ax.add_patch(Rectangle((s*z-3.5, r0), 7.0, r1-r0, facecolor="none",
                                   edgecolor=col, lw=1.2, hatch="////", zorder=5))
    ax.plot([], [], color=col, lw=6, alpha=0.6, label="%s  (pair solid, third hatched)" % nm)
ax.legend(loc="lower center", ncol=2, fontsize=8, framealpha=0.95,
          bbox_to_anchor=(0.5, 0.005))
ax.set_title("The FOUR extended (displaced) triplets: an existing seed PAIR on the OUTER "
             "two layers, plus a third hit FURTHER IN.\n"
             "The pair fixes curvature with the long lever arm; the inner hit is where "
             "d0/r is largest, so it is what makes d0 solvable.", fontsize=10.5, pad=10)

# ---- panel 3: the INNER TRACKER region, real D121 geometry --------------
ax = axes[2]
for i, r in enumerate(rit):
    ax.plot([-ZIT, ZIT], [r, r], color="#0b5fa5", lw=3.0, zorder=5)
    ax.text(-ZIT-6, r, "IL%d" % (i+1), ha="right", va="center", fontsize=8,
            color="#0b5fa5")
for i in range(12):
    col = "#17becf" if i < 8 else "#9467bd"
    for sgn in (+1, -1):
        ax.plot([sgn*zitd[i]]*2, [ritd_in[i], ritd_out[i]], color=col, lw=3.0,
                ls=(0, (3, 2)), zorder=4)
    ax.text(zitd[i], ritd_out[i]+0.7, "%d" % (i+1), ha="center", fontsize=6.5,
            color=col)
ax.text(-295, 27.4, "TFPX discs 1-8 (cyan), TEPX 9-12 (purple): the REAL D121 geometry "
        "read from CVMFS.\nDASHED because pixelAV does not simulate them (no angle "
        "regression for these orientations),\nyet the seed design must assume they exist.",
        fontsize=8, color="0.25", va="top")
for e in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    sh_ = np.sinh(e); zz = np.array([0.0, 300.0])
    for sgn in (+1, -1):
        ax.plot(sgn*zz, np.clip(zz/sh_, 0, 26), color="0.8", lw=0.8, ls=":", zorder=1)
    zt = min(26*sh_, 293)
    ax.text(zt, min(zt/sh_, 26)+0.35, "%.1f" % e, fontsize=7, color="0.5", ha="center")
ax.set_xlim(-300, 300); ax.set_ylim(0, 29)
ax.set_xlabel("z  [cm]"); ax.set_ylabel("r  [cm]")
ax.set_title("The Inner Tracker region. The REAL detector keeps >=4 surfaces at every "
             "|eta| out to 4.0\n(|eta|=1.5: two barrel + two discs;  |eta|=2.5: one "
             "barrel + six discs).  Our barrel-only census falls to 2 by 1.5, to 0 by 3.0.",
             fontsize=10.5, pad=10)
ax.grid(alpha=0.12)

fig.savefig("eval_refitq/combinatorics/results/seed_coverage_rz.png", dpi=135,
            bbox_inches="tight")
print("wrote results/seed_coverage_rz.png")
