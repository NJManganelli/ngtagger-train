"""All three variants, both sources, in cot and in degrees.

Top row is the quantity the payload actually stores and CMSSW actually applies:
a residual on cot(angle), dimensionless. Bottom row is the same clusters
translated to the opening angle itself, in degrees, which is the number a
detector person can judge.

The translation is done PER CLUSTER and exactly --

    angle = atan2(1, cot)          so  residual_deg = degrees(angle_pred - angle_true)

-- not by a small-angle factor, because d(angle)/d(cot) = -1/(1+cot^2) varies by
more than 2x over the |cotBeta| range present here, so one global scale factor
would misstate the wide-angle clusters. Note the minus sign in that derivative:
cot decreases with angle, so a POSITIVE cot residual is a NEGATIVE angle
residual. The sign flip between the rows is that, not a bug.
"""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent      # payload JSONs sit beside this file
ROOT = HERE.parent                          # the -vars.parquet dumps sit here
sys.path.insert(0, str(HERE))
from validate_angle_payload_closure import load_parquet, to_spec, robust
from correctionlib import CorrectionSet

BLOCALY_CLAMP, LAYER = -3.81, 1
VAR = ["Conv1D_Full-2bit", "Conv2D_Max-2bit", "Mlp_Slim-2bit"]
COL = {"Conv1D_Full-2bit": "tab:blue", "Conv2D_Max-2bit": "tab:orange",
       "Mlp_Slim-2bit": "tab:green"}
DEG = np.degrees


def ang(cot):
    return np.arctan2(1.0, cot)


fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6))
rows = []
for v in VAR:
    S = to_spec(load_parquet([str(ROOT / f"2t-{v}_optimized-vars.parquet")]))
    cs = CorrectionSet.from_file(str(HERE / f"spix_angle_response_{v}.json"))
    n = len(S["true_alpha"])
    lay = np.full(n, LAYER, float); bly = np.full(n, BLOCALY_CLAMP)
    for j, a in enumerate(("alpha", "beta")):
        if f"resid_{a}" not in S:
            continue
        tru = S[f"true_{a}"]
        r = S[f"resid_{a}"]
        shift = cs.compound[f"spix_angle_{a}_shift"].evaluate(
            lay, S["true_alpha"], S["true_beta"], bly, np.ones(n))
        # degrees: reconstruct both endpoints and difference the angles
        rd = DEG(ang(tru + r) - ang(tru))
        sd = DEG(ang(tru + shift) - ang(tru))
        for i, (x_p, x_a, lim, un) in enumerate(
                ((r, shift, 0.25, ""), (rd, sd, 15.0, " deg"))):
            _, wp = robust(x_p); _, wa = robust(x_a)
            # Both estimators, because they disagree by 1.5-2x and the gap IS
            # the tail: MAD is the core width the payload's sigma can be held
            # to, RMS is the one that carries the outlier rate.
            rp, ra = float(np.std(x_p)), float(np.std(x_a))
            b = np.linspace(-lim, lim, 161)
            ax = axs[i][j]
            ax.hist(x_p, bins=b, histtype="step", density=True, lw=1.6,
                    color=COL[v],
                    label=f"{v} ML parquet: MAD {wp:.4g}{un}, RMS {rp:.4g}{un}")
            ax.hist(x_a, bins=b, histtype="step", density=True, lw=1.2,
                    ls="--", color=COL[v],
                    label=f"{v} as applied: MAD {wa:.4g}{un}, RMS {ra:.4g}{un}")
            if i:
                rows.append((v, a, wp, wa))

for j, a in enumerate(("alpha", "beta")):
    nm = f"cot{a}" + (" (bending angle)" if a == "alpha" else "")
    axs[0][j].set_xlabel(f"spec cot{a} residual, pred - true [unitless]")
    axs[0][j].set_title(nm)
    axs[1][j].set_xlabel(f"spec {a} residual, pred - true [deg]")
    axs[1][j].set_title(nm.replace(f"cot{a}", a) + ", as an opening angle")
    for i in (0, 1):
        ax = axs[i][j]
        ax.set_yscale("log")
        # same number of decades below the peak in both rows -- the degree
        # panels are ~50x lower in density purely because the x unit is ~50x
        # bigger, and giving them a shorter y range hid tails that are in fact
        # just as heavy (verified: >5x-width fraction agrees within 20%).
        ax.set_ylim(*((1e-3, 60) if i == 0 else (2e-5, 1.2)))
        ax.set_ylabel("density [1/bin]")
        ax.grid(alpha=.3)
        ax.legend(fontsize=6.5, loc="lower center")
for i in (0, 1):
    axs[i][1].text(0.02, 0.96, "Mlp_Slim predicts the bending angle only:\n"
                   "no spec beta output, so no curve here",
                   transform=axs[i][1].transAxes, va="top", fontsize=8,
                   color="tab:green")
fig.suptitle("PixelAV angle payload: ML residual (solid) vs payload as applied "
             "in CMSSW (dashed)")
fig.tight_layout()
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "angle_closure_ALL.png"
fig.savefig(out, dpi=130)
print("wrote", out)
for v, a, w, aw in rows:
    print(f"  {v:20s} {a:5s} parquet {w:7.4f} deg  applied {aw:7.4f} deg"
          f"  ratio {aw/w:.3f}")
