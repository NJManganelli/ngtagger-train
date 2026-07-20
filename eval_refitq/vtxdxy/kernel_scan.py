"""FastHisto peak-finder kernel study: two-close-vertices toy scan.

Thin wrapper over ngtagger.train.vtxstudy.run_kernel_scan (the reusable
compute lives in the package). Writes kernel_scan.json + kernel_scan.png to
this directory.

Weakness under study (user-identified): the flat boxcar window can prefer the
midpoint BETWEEN two similarly-hard vertices over either true peak. The scan
throws two vertices at a controlled z-separation and relative hardness and
measures the wrong-vertex / midpoint-pick rate for the flat kernel vs the
tapered kernels (triangular, gaussian, epanechnikov), plus each kernel's
single-vertex resolution cost.

Usage:
  pixi run python eval_refitq/vtxdxy/kernel_scan.py
  ngtagger vtx-study --kernel-scan
"""
from __future__ import annotations

import os

from ngtagger.train.vtxstudy import run_kernel_scan

OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    run_kernel_scan(OUT)


if __name__ == "__main__":
    main()
