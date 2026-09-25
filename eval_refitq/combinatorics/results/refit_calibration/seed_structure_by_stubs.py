"""Test 2: does the OT seed error structure depend on the track's stub content?
Per floor group x pT bin x (number of stubs; PS/2S stub mix; missing barrel layer),
the robust (MAD) and variance widths of (TTTrack - truth@POCA) relative to the
reported sigma, and the empirical rInv-phi0 / phi0-d0 correlations."""
import sys, glob, numpy as np, uproot, awkward as ak
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import refit_calibration_fit as F
from ngtagger.truth_helix import KPT_CMSSW
files = sorted(glob.glob("/Users/nmangane/smartpixels/cmssw/work/otstub_arm/itot_tp_f*_100ev.root"))
HP = ["rInv", "phi", "tanL", "z0", "d0"]
cov = [f"cov_{HP[i]}_{HP[j]}" for i in range(5) for j in range(i, 5)]
cols = HP + ["genuine", "tp_pt", "tp_phi0", "tp_tanL", "tp_z0", "tp_d0", "tp_charge", "nStubs", "hitPattern"] + cov
A = uproot.concatenate([f"{f}:Events" for f in files], [f"L1TTrack_{c}" for c in cols])
S = uproot.concatenate([f"{f}:Events" for f in files], ["L1TTrackStub_trackIdx", "L1TTrackStub_layer", "L1TTrackStub_isBarrel"])
N = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(float) for c in cols}
ntrk = ak.to_numpy(ak.num(A["L1TTrack_rInv"])); off = np.r_[0, np.cumsum(ntrk)]
nst = ak.to_numpy(ak.num(S["L1TTrackStub_trackIdx"])); ev = np.repeat(np.arange(len(nst)), nst)
g = off[ev] + ak.to_numpy(ak.flatten(S["L1TTrackStub_trackIdx"])).astype(np.int64)
# barrel OT layers 1-3 are PS modules, 4-6 2S (no isPS column in the nano); barrel tracks only below
isps = (ak.to_numpy(ak.flatten(S["L1TTrackStub_isBarrel"])) > 0) & (ak.to_numpy(ak.flatten(S["L1TTrackStub_layer"])) <= 3)
n_ps = np.bincount(g, weights=isps, minlength=len(N["rInv"]))
n_st = np.bincount(g, minlength=len(n_ps))
d = np.stack([N[p] for p in HP], 1) - np.stack([N["tp_charge"] * KPT_CMSSW / N["tp_pt"], N["tp_phi0"], N["tp_tanL"], N["tp_z0"], N["tp_d0"]], 1)
d[:, 1] = np.angle(np.exp(1j * d[:, 1]))
C = np.zeros((len(d), 5, 5)); k = 0
for i in range(5):
    for j in range(i, 5):
        C[:, i, j] = C[:, j, i] = N[cov[k]]; k += 1
pull = d / np.sqrt(np.stack([C[:, i, i] for i in range(5)], 1))
base = (N["genuine"] > 0) & (np.abs(N["tanL"]) < 0.8)
print("barrel |tanL|<0.8 genuine tracks: pull MAD (robust) / pull RMS-after-4MAD-trim, and empirical corr(rInv,phi0)")
for lo, hi in ((2, 5), (5, 1e9)):
    for lab, m in (("4 stubs", n_st == 4), ("5 stubs", n_st == 5), ("6+ stubs", n_st >= 6),
                   ("PS stubs <=2", n_ps <= 2), ("PS stubs 3", n_ps == 3), ("PS stubs >=4", n_ps >= 4)):
        s = base & m & (N["tp_pt"] >= lo) & (N["tp_pt"] < hi)
        if s.sum() < 200:
            continue
        x = pull[s]
        med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0)
        keep = np.all(np.abs(x - med) < 4 * mad, 1)
        sd = x[keep].std(0)
        cr = np.corrcoef(d[s][keep][:, 0], d[s][keep][:, 1])[0, 1]
        print(f"  pT [{lo:g},{hi:g}) {lab:13s} n {s.sum():6d} | " + " ".join(f"{p} {mad[i]:.2f}/{sd[i]:.2f}" for i, p in enumerate(HP))
              + f" | corr(rInv,phi0) {cr:+.2f}")
