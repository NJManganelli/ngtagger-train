"""Seed covariance STRUCTURE check: empirical covariance of (TTTrack - truth at the
POCA) for genuine tracks at high pT (scattering negligible), against the TTTrack
covariance the fit reports. Robust: trim tracks beyond 4 MAD in any parameter.
Truth: nano tp_phi0/tp_d0/tp_z0 (itot_tp), curvature with KPT_CMSSW."""
import glob, numpy as np, uproot, awkward as ak
from ngtagger.truth_helix import KPT_CMSSW
files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root"))
HP = ["rInv", "phi", "tanL", "z0", "d0"]
cov = [f"cov_{HP[i]}_{HP[j]}" for i in range(5) for j in range(i, 5)]
cols = HP + ["genuine", "tp_pt", "tp_phi0", "tp_tanL", "tp_z0", "tp_d0", "tp_charge"] + cov
A = uproot.concatenate([f"{f}:Events" for f in files], [f"L1TTrack_{c}" for c in cols])
N = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(float) for c in cols}
tru = np.stack([N["tp_charge"] * KPT_CMSSW / N["tp_pt"], N["tp_phi0"], N["tp_tanL"], N["tp_z0"], N["tp_d0"]], 1)
d = np.stack([N[p] for p in HP], 1) - tru
d[:, 1] = np.angle(np.exp(1j * d[:, 1]))
C = np.zeros((len(d), 5, 5)); k = 0
for i in range(5):
    for j in range(i, 5):
        C[:, i, j] = C[:, j, i] = N[cov[k]]; k += 1
corr = lambda M: M / np.sqrt(np.outer(np.diag(M), np.diag(M)))
for lo, hi in ((10, 1e9), (20, 1e9), (3, 5)):
    for lab, tl in (("barrel |tanL|<0.8", (0, 0.8)), ("|tanL| 0.8-1.6", (0.8, 1.6))):
        m = (N["genuine"] > 0) & (N["tp_pt"] >= lo) & (N["tp_pt"] < hi) & (np.abs(N["tanL"]) >= tl[0]) & (np.abs(N["tanL"]) < tl[1])
        x = d[m]
        med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0)
        keep = np.all(np.abs(x - med) < 4 * mad, 1)
        E = np.cov(x[keep].T); R = np.median(C[m][keep], 0)
        print(f"== pT [{lo:g},{hi:g}) {lab}: n {m.sum()} (trimmed {keep.sum()})")
        print("   sqrt diag empirical / reported:", " ".join(f"{p}:{np.sqrt(E[i,i]/R[i,i]):.2f}" for i, p in enumerate(HP)))
        ce, cr = corr(E), corr(R)
        for i, j in ((0, 1), (0, 4), (1, 4), (2, 3)):
            print(f"   corr({HP[i]},{HP[j]}): empirical {ce[i,j]:+.3f}  reported {cr[i,j]:+.3f}")
