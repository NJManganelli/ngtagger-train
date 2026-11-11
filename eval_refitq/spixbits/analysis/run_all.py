"""spixbits analysis: run every step on whatever units exist (one process, one step at a time, under 1 GB).

  python run_all.py            core summaries (new or changed arms only), then all reports and the README draft
  python run_all.py --reports  reports only, from the caches

Outputs: ~/smartpixels/doc-archive/spix-bits-refit-study/results/ and README.md beside it. Units are only read.
"""
import sys

import core
import lib

if __name__ == "__main__":
    lib.check_disk()
    if "--reports" not in sys.argv:
        core.run(core.SCENARIOS)
    import fixed_points
    import scans
    import near_full
    import readme
    fixed_points.main()
    scans.main()
    near_full.main()
    readme.main()
