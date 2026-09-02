"""How many sigma is each static search window, per layer and per pT?

Companion to planningAndPatches/v2p7-refit-study-program.md section 3b.

For the x dimension the KF's relinearization term vanishes on the first scalar
update (aLin = a is captured before the k-loop), so r == resX exactly and

    sqrt(S_x) = resX / pullX          (exact for x; approximate for y)

is recoverable from existing nano. S_x = H_x C H_x^T + sigma_meas^2 is the
innovation variance: the statistically correct half-width scale for a search
window. Comparing it to the static per-layer constants shows directly how
mis-sized they are, and how that varies with pT.
"""
import sys
sys.path.insert(0, '/Users/nmangane/smartpixels/ngtagger-train/src')
import awkward as ak, numpy as np, uproot
from ngtagger.train.chi2quant import SENTINEL
import glob as _glob, argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('-i', '--inputs', nargs='+',
                 default=['/Users/nmangane/smartpixels/cmssw/work/spxsmoke/nano/clamp_on_f*.root'],
                 help='nano files or globs (post-guard digiRefit productions)')
_args = _ap.parse_args()
DEFAULT_FILES = [f for p in _args.inputs for f in (sorted(_glob.glob(p)) or [p])]
print(f'files: {len(DEFAULT_FILES)}')

REF, VAR = "L1TTrack", "L1TSmartPixelsTrackDigiRefitAAAA"
HIT = "L1TSmartPixelsRefitHitDigiRefitAAAA"
WIN_RPHI = {1: 0.05, 2: 0.17, 3: 0.5, 4: 0.9}
WIN_Z = {1: 0.45, 2: 0.35, 3: 0.25, 4: 0.2}

ref = uproot.concatenate([f"{f}:Events" for f in DEFAULT_FILES],
                         filter_name=[f"{REF}_pt", f"{REF}_genuine"])
hits = uproot.concatenate([f"{f}:Events" for f in DEFAULT_FILES],
                          filter_name=[f"{HIT}_{c}" for c in
                                       ("trackIdx","layer","resX","pullX","resY","pullY",
                                        "windowMult","selHitClass","hitAccepted")])
counts = ak.to_numpy(ak.num(ref[f"{REF}_pt"]))
off = np.concatenate([[0], np.cumsum(counts)])
pt = ak.to_numpy(ak.flatten(ref[f"{REF}_pt"]))

h = {c: ak.to_numpy(ak.flatten(hits[f"{HIT}_{c}"])) for c in
     ("trackIdx","layer","resX","pullX","resY","pullY","windowMult","selHitClass","hitAccepted")}
ev = np.repeat(np.arange(len(counts)), ak.to_numpy(ak.num(hits[f"{HIT}_trackIdx"])))
g = off[ev] + h["trackIdx"].astype(np.int64)
hit_pt = pt[g]

ok = (h["hitAccepted"] > 0) & (h["pullX"] > SENTINEL) & (np.abs(h["pullX"]) > 1e-9)
sx = np.abs(h["resX"][ok]) / np.abs(h["pullX"][ok])
oky = (h["hitAccepted"] > 0) & (h["pullY"] > SENTINEL) & (np.abs(h["pullY"]) > 1e-9)
sy = np.abs(h["resY"][oky]) / np.abs(h["pullY"][oky])
lay, layy = h["layer"][ok], h["layer"][oky]
ptx = hit_pt[ok]

print("innovation sigma sqrt(S_x) vs the STATIC r-phi window, per layer")
print("  L   n      sqrt(S_x) [um]  p50/p90      window [um]  window/sigma  p90-sigma")
for L in (1,2,3,4):
    m = lay == L
    if m.sum() < 50: continue
    s = sx[m]*1e4  # cm -> um
    w = WIN_RPHI[L]*1e4
    print(f"  {L}  {m.sum():>6}   {np.median(s):>8.1f} / {np.quantile(s,0.9):>8.1f}"
          f"      {w:>8.0f}     {w/np.median(s):>8.1f}x   {w/np.quantile(s,0.9):>6.1f}x")

print("\nsame for z / local-y (approximate: sequential updates leave a relin. term)")
for L in (1,2,3,4):
    m = layy == L
    if m.sum() < 50: continue
    s = sy[m]*1e4
    w = WIN_Z[L]*1e4
    print(f"  {L}  {m.sum():>6}   {np.median(s):>8.1f} / {np.quantile(s,0.9):>8.1f}"
          f"      {w:>8.0f}     {w/np.median(s):>8.1f}x   {w/np.quantile(s,0.9):>6.1f}x")

print("\nsqrt(S_x) [um] by pT bin (the 1/pT scaling a constant window cannot follow)")
bins = [(2,3),(3,5),(5,10),(10,20),(20,1e9)]
print("     pT bin   " + "".join(f"   L{L}" .rjust(9) for L in (1,2,3,4)))
for lo,hi in bins:
    row = f"  {lo:>4}-{hi if hi<1e9 else 'inf':>5}  "
    for L in (1,2,3,4):
        m = (lay == L) & (ptx >= lo) & (ptx < hi)
        row += f"{np.median(sx[m])*1e4:>9.1f}" if m.sum() > 30 else "        -"
    print(row)

print("\nwindow/sigma ratio by pT bin (r-phi): how many sigma the constant really is")
print("     pT bin   " + "".join(f"   L{L}" .rjust(9) for L in (1,2,3,4)))
for lo,hi in bins:
    row = f"  {lo:>4}-{hi if hi<1e9 else 'inf':>5}  "
    for L in (1,2,3,4):
        m = (lay == L) & (ptx >= lo) & (ptx < hi)
        row += f"{WIN_RPHI[L]/np.median(sx[m]):>8.1f}x" if m.sum() > 30 else "        -"
    print(row)

print("\nconsequence: in-window multiplicity and wrong-hit rate per layer")
acc = h["hitAccepted"] > 0
print("  L   <windowMult>  median  frac wrong (otherTP) of accepted")
for L in (1,2,3,4):
    m = acc & (h["layer"] == L)
    wm = h["windowMult"][m]
    wrong = (h["selHitClass"][m] == 1).mean()
    print(f"  {L}    {wm.mean():>8.2f}    {np.median(wm):>5.0f}      {wrong*100:>5.1f}%")
