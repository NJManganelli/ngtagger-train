"""Build the data library of the CMSSW SmartPixels-output emulator (angleSource = "emu").

  emu_build_library.py OUT.root

OUT.root (uproot, read by SmartPixelsRecHitProducer through ROOT) holds ONE tree:
  dump   one row per Conv1D_Full-2bit dump row (RESCALED to physical units, CMS frame = pure swap):
         tA, tB = atan(true cot) [deg] (the 32-NN lookup key), rA, rB = atan(pred) - atan(true) [deg]  (float32)
The class-C residual model (A1) and the CNNs are ONNX graphs (emu_export_onnx.py); no donor library is shipped.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import deg

SPDUMP = os.path.expanduser("~/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet")
# the dump stores NORMALISED labels; physical = scale x stored (rescale() in smart-pixels-ml/summary-plots/read-all-models-2s.ipynb)
SF = {"cotA": 6.528905408493022, "cotB": 1.8827422609846889}


def load_dump():
    """CMS frame = pure swap: CMS alpha = dump cotB, CMS beta = dump cotA. r = atan(pred) - atan(true) [deg]."""
    d = pd.read_parquet(SPDUMP)
    tA, pA = SF["cotB"] * d.cotBtrue.values, SF["cotB"] * d.cotB.values
    tB, pB = SF["cotA"] * d.cotAtrue.values, SF["cotA"] * d.cotA.values
    return pd.DataFrame({"tA": deg(tA), "tB": deg(tB), "rA": deg(pA) - deg(tA), "rB": deg(pB) - deg(tB)})


def build(path):
    import uproot
    D = load_dump()
    with uproot.recreate(path, compression=uproot.LZMA(9)) as f:
        f.mktree("dump", {k: np.float32 for k in ("tA", "tB", "rA", "rB")}).extend(
            {k: D[k].values.astype(np.float32) for k in ("tA", "tB", "rA", "rB")})
    print("wrote", path, len(D), "dump rows,", os.path.getsize(path), "bytes", flush=True)


if __name__ == "__main__":
    build(sys.argv[1])
