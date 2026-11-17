"""Copy everything the study reads besides the canon_v5 nanos onto the internal disk (run once, both disks mounted).

  payload/   the v3 calibration set canon_v5 was produced with (v3cmssw/spec/payload: payload set v1), each file's
             sha256 checked against the spec's PROVENANCE.json; checksums.json records them
  network_residuals.npz   the full-MLP quantized network's residuals (prediction - truth, physical units; float32):
             rx, ry [um], rcotA, rcotB, with its true (cotA, cotB) [SmartPixels frame], from the SD card's parquet
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

LOCAL = Path.home() / "smartpixels/scratch_local/spixbits"
SRC = Path("/Volumes/WDMac/smartpixels_scratch/v3cmssw/spec/payload")
PRED = "/Volumes/NJM256GBSD/smartpixels/predictions_full_mlp_4-quantized.parquet"
RATIO = {"x": 1.0038368, "y": 1.0010152, "cotA": 0.9951212, "cotB": 0.9820157}      # file / physical (quant_bits.py)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    dst = LOCAL / "cache" / "payload"
    dst.mkdir(parents=True, exist_ok=True)
    prov = json.load(open(SRC / "PROVENANCE.json"))
    sums = {}
    for f in sorted(SRC.iterdir()):
        shutil.copy2(f, dst / f.name)
        sums[f.name] = {"sha256": sha(dst / f.name), "source": str(f)}
        if f.name in prov and prov[f.name]["sha256"] != sums[f.name]["sha256"]:
            sys.exit(f"{f.name}: sha256 differs from the spec's PROVENANCE.json")
    import pandas as pd
    P = pd.read_parquet(PRED, columns=["x", "xtrue", "y", "ytrue", "cotA", "cotAtrue", "cotB", "cotBtrue"])
    out = {}
    for v in ("x", "y", "cotA", "cotB"):
        out["r" + v] = ((P[v].to_numpy(np.float64) - P[v + "true"].to_numpy(np.float64)) / RATIO[v]).astype(np.float32)
    for v in ("cotA", "cotB"):
        out[v + "_true"] = (P[v + "true"].to_numpy(np.float64) / RATIO[v]).astype(np.float32)
    tmp = LOCAL / "cache" / "network_residuals.tmp.npz"
    np.savez(tmp, **out)
    tmp.rename(LOCAL / "cache" / "network_residuals.npz")
    sums["network_residuals.npz"] = {"sha256": sha(LOCAL / "cache" / "network_residuals.npz"), "source": PRED,
                                     "rows": int(len(P))}
    json.dump(sums, open(LOCAL / "cache" / "checksums.json", "w"), indent=1)
    print(json.dumps({k: v["sha256"][:12] for k, v in sums.items()}, indent=1))


if __name__ == "__main__":
    main()
