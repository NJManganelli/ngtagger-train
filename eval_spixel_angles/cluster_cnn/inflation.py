"""Truth-free angle inflation: theta' = theta_pred + a_cell * N(0,1),  a_cell = sqrt(k^2 - 1) * sigma_eff,cell.

sigma_cell = sigma_MAD [deg] of the base-CNN residual (atan space) in cells of
(layer, |pred cot| bin, sizeX bin, sizeY bin), measured on truth-consistent linked test
clusters; cells with < 50 entries fall back to (layer, |pred cot| bin), then to layer.
The residuals are heavier-tailed than a Gaussian, so the naive a = sqrt(k^2-1)*sigma_MAD
over-inflates (sigma_MAD ratio ~1.6 at k = 1.5). a_cell is therefore CALIBRATED per cell
(bisection) so that sigma_MAD(residual + a*N) = k * sigma_MAD(residual) on half the test
events; sigma_eff = a / sqrt(k^2-1) is the Gaussian-equivalent width. The lookup uses only
the prediction and reco quantities, so linked and unlinked clusters are treated identically.
N(0,1) is a deterministic splitmix64 hash of (event, clIdx, salt): reproducible, order-free.

Study record of the k = 1.5 result only: the CMSSW angle source uses the raw CNN output.
"""
import numpy as np

BINS = [0, 0.5, 1.07, 2, 4, np.inf]
SXE, SYE = [2, 3, 4, 6], [2, 3, 5, 8]
SALT = {"A": 0xA1, "B": 0xB2}
K = 1.5


def cells(layer, pred_cot, sizeX, sizeY):
    pb = np.digitize(np.abs(pred_cot), BINS[1:-1])
    sxb = np.digitize(sizeX, SXE); syb = np.digitize(sizeY, SYE)
    return ((layer * 10 + pb) * 10 + sxb) * 10 + syb, layer * 10 + pb, pb


def _splitmix(x):
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hash_normal(event, clidx, salt):
    with np.errstate(over="ignore"):
        key = (np.asarray(event).astype(np.uint64) * np.uint64(1000003)) ^ (np.asarray(clidx).astype(np.uint64) << np.uint64(20)) ^ np.uint64(salt)
    u1 = (_splitmix(key) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    u2 = (_splitmix(key ^ np.uint64(0x5DEECE66D)) >> np.uint64(11)).astype(np.float64) / 2.0 ** 53
    return np.sqrt(-2 * np.log(np.clip(u1, 1e-300, 1))) * np.cos(2 * np.pi * u2)


def lookup(keys_maps):
    """keys_maps: [(key array, dict key -> value), ...] from finest to coarsest; first finite wins."""
    out = np.full(len(keys_maps[0][0]), np.nan)
    for keys, mp in keys_maps:
        v = np.array([mp.get(c, np.nan) for c in keys.tolist()])
        out = np.where(np.isfinite(out), out, v)
    return out


def calibrate_add(r, n, k=K, smad=None):
    """additive width a such that sigma_MAD(r + a*n) = k * sigma_MAD(r) (bisection)."""
    s0 = smad(r); lo, hi = 0.0, 5 * k * s0 + 1e-6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if smad(r + mid * n) < k * s0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def inflate(theta_pred_deg, add_deg, event, clidx, ang):
    return np.clip(theta_pred_deg + add_deg * hash_normal(event, clidx, SALT[ang]), -89.99, 89.99)
