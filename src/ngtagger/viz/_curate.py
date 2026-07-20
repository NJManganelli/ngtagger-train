"""Curate a small set of illustrative example tracks (5-par prompt framing).

Four archetypes (see docs/refit-replay-viz.md):
  1. clean    : prompt GENUINE track (|d0|<0.03 cm), all 4 TBPX layers hit with
                same-TP hits (selHitClass==0) -- the textbook refit; d0/z0 should
                stay near truth (PV) as IT hits come in.
  2. wrong    : GENUINE track with an other-TP (selHitClass==1) hit picked up on an
                outer layer (L3/L4) -- the KF gets pulled; the wrong-hit story
                (outer layers are wrong-hit-rich). Watch the r-z chi2 inflate.
  3. displaced: GENUINE track with a real nonzero |d0|>0.1 cm -- the B-TAGGING
                relevant case (b-decay tracks are displaced). CAVEAT: on nano_pE
                the GenVtx(PV) d0-truth is ~0 and thus WRONG for these; the
                matched-TP d0 (coming in nano_pF) is the right truth. Flagged.
  4. fake     : unmatched (non-genuine) track with >=3 accepted hits.
"""

from __future__ import annotations


def _hit_classes(tr):
    return {L: tr.hits[L]["selHitClass"] for L in tr.hits}


def curate_tracks(data, n_each=(2, 1, 1, 1), pt_min=2.5):
    """Return a list of (record, archetype, reason) picked deterministically.

    Scans in (event, idx) order so the picks are reproducible. ``n_each`` is the
    number wanted per archetype (clean, wrong, displaced, fake). A 3-tuple is
    accepted for back-compat and maps to (clean, wrong, fake) with 0 displaced.
    """
    if len(n_each) == 3:
        want_clean, want_wrong, want_fake = n_each
        want_disp = 0
    else:
        want_clean, want_wrong, want_disp, want_fake = n_each
    clean, wrong, disp, fake = [], [], [], []

    def _refit_ok(tr):
        return tr.real["AAAA"]["refitPerformed"]

    for tr in data.tracks:
        if tr.seed["pt"] < pt_min or not _refit_ok(tr):
            continue
        cls = _hit_classes(tr)
        n = len(tr.hits)
        d0 = tr.seed["d0"]
        # clean: prompt genuine from the HARD interaction with z0 near GenVtx_z, so
        # GenVtx(PV) is valid truth for BOTH d0 (~0) and z0 -> a faithful resolution
        # illustration (avoid PU tracks whose z0 != the hard PV).
        z0_near_pv = abs(tr.seed["z0"] - tr.genvtx["z"]) < 0.3
        if (len(clean) < want_clean and tr.truth["genuine"] and tr.truth["fromHard"]
                and n == 4 and abs(d0) < 0.03 and z0_near_pv
                and all(cls.get(L) == 0 for L in (1, 2, 3, 4))):
            clean.append((tr, "clean",
                          f"genuine prompt from hard int (pt={tr.seed['pt']:.1f}, d0={d0:+.3f} cm, "
                          f"tpPt={tr.truth['tpPt']:.1f}), all 4 TBPX layers same-TP hits, z0≈PV: "
                          f"the textbook refit -- d0/z0 should track truth (PV) as IT comes in."))
        elif (len(wrong) < want_wrong and tr.truth["genuine"] and n >= 3
              and any(cls.get(L) == 1 for L in (3, 4))
              and any(cls.get(L) == 0 for L in (1, 2))):
            bad = [L for L in (3, 4) if cls.get(L) == 1]
            wrong.append((tr, "wrong",
                          f"genuine (pt={tr.seed['pt']:.1f}) with an other-TP wrong hit on "
                          f"L{'/'.join(map(str, bad))}: the KF is pulled by the wrong outer hit -- "
                          f"why outer layers are wrong-hit-rich (watch the r-z chi2 inflate)."))
        elif (len(disp) < want_disp and tr.truth["genuine"] and n >= 3 and abs(d0) > 0.1):
            disp.append((tr, "displaced",
                         f"genuine DISPLACED (pt={tr.seed['pt']:.1f}, d0={d0:+.3f} cm, "
                         f"{'hardInt' if tr.truth['fromHard'] else 'PU/sec'}): the b-tagging-relevant "
                         f"case -- true d0 is NONZERO, so GenVtx(PV) truth is wrong here; needs "
                         f"matched-TP d0 (nano_pF). Does IT sharpen the real d0?"))
        elif (len(fake) < want_fake and (not tr.truth["genuine"]) and n >= 3):
            fake.append((tr, "fake",
                         f"unmatched/fake track (pt={tr.seed['pt']:.1f}, not genuine) with "
                         f"{n} accepted hits: no consistent parent, best-chi2 selection cherry-picks."))
        if (len(clean) >= want_clean and len(wrong) >= want_wrong
                and len(disp) >= want_disp and len(fake) >= want_fake):
            break
    return clean + wrong + disp + fake
