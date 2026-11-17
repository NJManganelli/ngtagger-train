"""spixbits analysis: run every step on whatever units exist (one process, one step at a time, under 1 GB).

  python run_all.py                                   the first scan (default directories)
  python run_all.py --reports                         reports only, from the caches
  python run_all.py --scan-dir DIR --tag NAME         another scan (DIR holds units/ and arms.json); its caches,
                                                      results/NAME/ and README_NAME.md are kept apart
  python compare.py NAME                              afterwards: NAME against the first scan, same tracks and replicas

Outputs: ~/smartpixels/doc-archive/spix-bits-refit-study/results/ and README.md beside it. Units are only read.
"""
import argparse
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", action="store_true")
    ap.add_argument("--scan-dir")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if bool(a.scan_dir) != bool(a.tag):
        raise SystemExit("--scan-dir and --tag go together")
    if a.scan_dir:
        os.environ["SPIXBITS_SCAN_DIR"], os.environ["SPIXBITS_TAG"] = a.scan_dir, a.tag
    import core
    import lib
    lib.check_disk()
    if not a.reports:
        core.run(core.SCENARIOS)
    import fixed_points
    import scans
    import near_full
    import readme
    fixed_points.main()
    scans.main()
    near_full.main()
    readme.main()
