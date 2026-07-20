"""Curate a small set of illustrative example tracks for the talk.

Three archetypes (see docs/refit-replay-viz.md):
  1. clean  : prompt-ish GENUINE track, all 4 TBPX layers hit with same-TP hits
              (selHitClass==0) -- the textbook refit.
  2. wrong  : GENUINE track with an other-TP (selHitClass==1) hit picked up on an
              outer layer (L3/L4) -- the KF gets pulled; the MS-bulge / wrong-hit
              story (outer layers are wrong-hit-rich).
  3. fake   : unmatched (non-genuine) track with >=3 accepted hits.
"""

from __future__ import annotations


def _hit_classes(tr):
    return {L: tr.hits[L]["selHitClass"] for L in tr.hits}


def curate_tracks(data, n_each=(1, 1, 1), pt_min=2.5):
    """Return a list of (record, archetype, reason) picked deterministically.

    Scans in (event, idx) order so the picks are reproducible. ``n_each`` is the
    number wanted per archetype (clean, wrong, fake).
    """
    want_clean, want_wrong, want_fake = n_each
    clean, wrong, fake = [], [], []
    for tr in data.tracks:
        if tr.seed["pt"] < pt_min:
            continue
        cls = _hit_classes(tr)
        n = len(tr.hits)
        if (len(clean) < want_clean and tr.truth["genuine"] and n == 4
                and all(cls.get(L) == 0 for L in (1, 2, 3, 4))):
            clean.append((tr, "clean",
                          f"genuine prompt (pt={tr.seed['pt']:.1f}, tpPt={tr.truth['tpPt']:.1f}), "
                          f"all 4 TBPX layers same-TP hits: the textbook refit (watch d0/z0 tighten "
                          f"as angles come in)."))
        elif (len(wrong) < want_wrong and tr.truth["genuine"] and n >= 3
              and any(cls.get(L) == 1 for L in (3, 4))
              and any(cls.get(L) == 0 for L in (1, 2))):
            bad = [L for L in (3, 4) if cls.get(L) == 1]
            wrong.append((tr, "wrong",
                          f"genuine (pt={tr.seed['pt']:.1f}) with an other-TP wrong hit on "
                          f"L{'/'.join(map(str, bad))}: the KF is pulled by the wrong outer hit -- "
                          f"why outer layers are wrong-hit-rich (the MS-bulge story)."))
        elif (len(fake) < want_fake and (not tr.truth["genuine"]) and n >= 3):
            fake.append((tr, "fake",
                         f"unmatched/fake track (pt={tr.seed['pt']:.1f}, not genuine) with "
                         f"{n} accepted hits: no consistent parent, best-chi2 selection cherry-picks."))
        if len(clean) >= want_clean and len(wrong) >= want_wrong and len(fake) >= want_fake:
            break
    return clean + wrong + fake
