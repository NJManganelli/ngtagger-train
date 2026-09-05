#!/usr/bin/env python
"""Rename `spx_*` correction keys to `spix_*` inside a correctionlib payload.

The angle-response payloads named their corrections `spx_angle_*`. "spx" does not
communicate anything; "spix" (smart pixels) does. The consumer looks these up by
exact string (`aset->at("spx_angle_alpha_sigma")` in L1SmartPixelsTrackProducer),
so the code-side rename and the payload-side rename must happen together or the
producer throws on load.

This exists so the coupling is a scripted, re-runnable migration rather than a
hand edit: payloads live in several places (the generator's output directory, the
container's working copy, CMSSW `data/`), and any stray pre-rename payload can be
brought forward instead of becoming unloadable.

What it rewrites, and why a blanket token replace is safe here: the token
`spx_angle_` / `spx_pos_` appears in `corrections[].name`,
`compound_corrections[].name`, the `stack` arrays that cross-reference those
names, and the human `description` strings. All four must move together, and the
token is unambiguous -- no other identifier in these payloads contains it.

    # in place, with a .bak
    ./migrate_payload_spx_to_spix.py payload.json

    # check only, non-zero exit if a rename is still needed
    ./migrate_payload_spx_to_spix.py --check payload.json
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import shutil
import sys

# Ordered longest-first so no prefix shadows another.
RENAMES = (("spx_angle_", "spix_angle_"), ("spx_pos_", "spix_pos_"))


def migrate_text(txt):
    n = 0
    for old, new in RENAMES:
        n += txt.count(old)
        txt = txt.replace(old, new)
    return txt, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payloads", nargs="+")
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if any file still needs migration")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    files = [f for p in args.payloads for f in (sorted(_glob.glob(p)) or [p])]
    need = 0
    for f in files:
        raw = open(f).read()
        new, n = migrate_text(raw)
        if not n:
            print(f"  ok        {f}")
            continue
        need += 1
        if args.check:
            print(f"  NEEDS     {f}  ({n} occurrences)")
            continue
        # Validate BOTH sides parse, and that the correction-name set is unchanged
        # apart from the prefix -- a blanket replace on a JSON string is only safe
        # if the result is still the same document shape.
        before, after = json.loads(raw), json.loads(new)
        def names(d):
            return (sorted(c["name"] for c in d.get("corrections", [])),
                    sorted(c["name"] for c in d.get("compound_corrections", [])))
        b_c, b_cc = names(before)
        a_c, a_cc = names(after)
        exp_c = sorted(migrate_text(x)[0] for x in b_c)
        exp_cc = sorted(migrate_text(x)[0] for x in b_cc)
        if a_c != exp_c or a_cc != exp_cc:
            raise SystemExit(f"{f}: correction-name set changed unexpectedly; refusing to write")
        if not args.no_backup:
            shutil.copy2(f, f + ".bak")
        with open(f, "w") as fh:
            fh.write(new)
        print(f"  migrated  {f}  ({n} occurrences, {len(a_c)} corrections, "
              f"{len(a_cc)} compound)")

    if args.check and need:
        print(f"\n{need} payload(s) still use spx_* keys", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
