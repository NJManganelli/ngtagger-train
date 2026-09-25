"""Doublet and triplet seeds as DISTINCT objects, and the 4-layer track rule.

WHY THE DISTINCTION IS NOT COSMETIC. A doublet has two azimuths for three r-phi
unknowns, so it can only get curvature by assuming d0 = 0, and a real impact
parameter biases it by kappa_bias = d0 * (1/rA - 1/rB) / (c * dr):

    IL1+IL2  10.34 /cm      IL3+IL4  1.17 /cm      OL1+OL2  0.19 /cm
    IL2+IL3   2.85 /cm      IL4+OL1  0.49 /cm

a ~50x span that is PURE RADIUS. Covering d0 = 500 um needs the curvature gate
opened by 0.517 against kappa_max = 0.500 at 2 GeV, so an IT doublet labelled
"2 GeV" really admits 0.98 GeV. At OL1+OL2 the same displacement perturbs kappa
by 0.019 and the window absorbs it.

So the OT's choice of doublet seeds is a CONSEQUENCE OF ITS RADII and does not
transfer inward. A triplet has three azimuths for three unknowns: d0 is solved
(solve3), the pT cut is honest, and the projection to further layers runs along
the real trajectory rather than a prompt approximation. Its price is that three
r-phi measurements determining three r-phi parameters leaves ZERO r-phi degrees
of freedom -- there is no chi2 left to reject on, so a displaced triplet's only
discrimination is the single r-z constraint plus the per-cluster angles.

THE 4-LAYER RULE, from L1Trigger/TrackFindingTracklet Setup_cfi.py and
Setup.cc:44:

    MinLayers = 4 , NumSeedingLayers = 2 , kfMinProj = MinLayers - NumSeedingLayers

The OT never picks "how many projections must confirm". It requires FOUR LAYERS
ON THE TRACK, and the seed's arity decides how many the projections still owe:
a doublet owes 2, a triplet owes 1. State.cc:92 uses the same number to abandon
a state the moment reaching four layers becomes arithmetically impossible.

DEVIATION, AND IT IS DELIBERATE. NumSeedingLayers is a single global 2, so
CMSSW applies kfMinProj = 2 to its own triplet seeds too, effectively demanding
five layers of a triplet-seeded track against four of a doublet-seeded one. That
reads as a simplification rather than a designed asymmetry, so MIN_LAYERS is
mirrored as the intent -- four layers total, whatever the arity. Set
strict_cmssw=True to reproduce the shipped behaviour instead.

NOTE what this changes about earlier numbers: the previous seeding required a
pair plus one projected hit, i.e. THREE layers, and called that a found track.
The OT would not emit it. Efficiencies quoted against a 3-layer object were
optimistic relative to the standard, and not uniformly so -- the gap is largest
for seeds whose third layer sits at large radius, where a fourth hit is least
likely.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402

IL = (1, 2, 3, 4)
OT_BARREL = (11, 12, 13, 14, 15, 16)
# L1L2, L2L3, L3L4, L5L6 -- Settings.h:679. There is deliberately no L4L5.
OT_DOUBLETS = ((11, 12), (12, 13), (13, 14), (15, 16))
# L2L3L4 and L4L5L6 -- the design's only barrel triplet seeds, IN CONSTRUCTION
# ORDER: the pair first, the attached third layer last.
#
# THE NAME IS SORTED, THE CONSTRUCTION IS NOT. seedlayers_[8] = {2,3,1} and
# [9] = {4,5,3} (Settings.h:679-690, 0-indexed layers), and ProcessBase.cc:142
# reads them as (layerdisk1, layerdisk2, layerdisk3) = (pair, pair, third). So
# L2L3L4 is the L3L4 pair with an L2 stub attached, and L4L5L6 is the L5L6 pair
# with an L4 stub attached -- CMSSW's own module names say so, TPD_L3L4L2 and
# TPD_L5L6L4. Listing these ascending made the pair L2L3 and L4L5, and L4L5 is
# not a seed at all, so once the real projection menu was enforced the second
# triplet found exactly zero tracks. An ascending tuple is not a harmless
# relabelling here; it builds a different combinatorial set.
OT_TRIPLETS = ((13, 14, 12), (15, 16, 14))
MIN_LAYERS = 4


class Seed:
    __slots__ = ("arity", "layers", "tag", "note", "soft")

    def __init__(self, layers, note="", soft=False):
        self.layers = tuple(layers)
        self.arity = len(self.layers)
        self.tag = "+".join(NAME[L] for L in self.layers) + (" soft" if soft else "")
        self.note = note
        # A DEDICATED SUB-2-GeV RECOVERY SEED, not the same seed run lower.
        # It carries its own pT floor and its own layer rule, and it is a
        # SEPARATE entry in the menu, so the standard seeds are untouched: a
        # 2 GeV track confirmed by two OT hits and no extra IT hit still
        # passes on the standard entry, which it would not if the 3xIT rule
        # had been imposed on the existing seeds. It also makes the soft
        # extension something you switch on and off in the seed list.
        self.soft = bool(soft)

    def __repr__(self):
        kind = 'doublet' if self.arity == 2 else 'triplet'
        return f"<{'soft ' if self.soft else ''}{kind} {self.tag}>"

    def min_proj(self, strict_cmssw=False):
        """Confirmed projections still required to reach a 4-layer track."""
        return MIN_LAYERS - (2 if strict_cmssw else self.arity)


NAME = {1: "IL1", 2: "IL2", 3: "IL3", 4: "IL4",
        11: "OL1", 12: "OL2", 13: "OL3", 14: "OL4", 15: "OL5", 16: "OL6"}


def soft_seeds(seeds):
    """Dedicated sub-2-GeV recovery entries: the IT-only seeds, duplicated.

    Only IT-only seeds: the recovery rule is "3xIT, optionally + 1xOT", so a
    seed that already spends a layer in the OT cannot satisfy it, and the OT
    cannot seed below its stub threshold at all (measured: 4.4% of 1-2 GeV TPs
    reach the >= 4 OT layers an OT track needs, against 42.3% just above it).
    """
    return [Seed(s.layers, note=(s.note + " sub-2GeV recovery").strip(),
                 soft=True)
            for s in seeds if all(L <= 4 for L in s.layers)]


def enumerate_seeds(il_instrumented, ot=OT_BARREL):
    """Every doublet and triplet a given SmartPixels build can form.

    IT-only and OT-only follow their own rules: all IT combinations, because
    that space is the point of the study and has no incumbent design; only the
    shipped OT seeds, because that one does.

    MIXED seeds obey a LOCALITY RULE -- the outermost instrumented IT layers meet
    the innermost OT layers, never IL1+IL2+OL6. Geometrically that is the right
    cut: IL4 -> OL1 is a 1.4x radius step and OL1 -> OL2 is 1.5x, so IL3+IL4+OL1
    and IL4+OL1+OL2 are as radially sensible as the OT's own seeds.

    DEGENERATE CASES ARE KEPT AND LABELLED, not excluded. In a build with only
    IL1 the rule still permits IL1+OL1+OL2, a 7.7x jump from r ~ 3 to r ~ 23, and
    those long-lever seeds measure at ~4e6 candidates/event and 98.8% fake. They
    are left in so the cost and quality columns can condemn them rather than a
    hand-applied filter deciding in advance.
    """
    il = sorted(il_instrumented)
    seeds = []
    for i, a in enumerate(il):                       # IT doublets: all pairs
        for b in il[i + 1:]:
            seeds.append(Seed((a, b)))
    for a, b in OT_DOUBLETS:                         # OT doublets: design only
        if a in ot and b in ot:
            seeds.append(Seed((a, b)))
    for i, a in enumerate(il):                       # IT triplets: all triples
        for j, b in enumerate(il[i + 1:], i + 1):
            for c in il[j + 1:]:
                seeds.append(Seed((a, b, c)))
    for t in OT_TRIPLETS:                            # OT triplets: design only
        if all(x in ot for x in t):
            seeds.append(Seed(t))
    # ---- mixed, under the locality rule ----------------------------------
    if il and ot:
        o1, o2 = ot[0], (ot[1] if len(ot) > 1 else None)
        far = il[-1]
        deg = " DEGENERATE: long lever arm" if far <= 2 else ""
        seeds.append(Seed((far, o1), "mixed doublet" + deg))
        if len(il) > 1:
            seeds.append(Seed((il[-2], far, o1), "mixed triplet" + deg))
        if o2 is not None:
            seeds.append(Seed((far, o1, o2), "mixed triplet" + deg))
    return seeds


def run_seed(U, Q, seed, ptmin, targets, use_angles=True,
             strict_cmssw=False, d0_cm=0.0, min_layers=None,
             min_it_layers=None, min_ot_conf=0):
    """Run one seed and follow it to EVERY target layer.

    A doublet projects to all of them with a d0 = 0 helix. A triplet first
    requires its named third layer, re-solves the helix with solve3 so d0 is
    MEASURED, and then projects along that trajectory -- which is why its
    windows need no d0 allowance where the doublet's do.

    Cost counters are summed over targets, so doublet and triplet costs are
    finally commensurable: the old accounting charged a seed for ONE projection
    where the OT does four.
    """
    allidx = np.arange(len(U["layer"]))
    la, lb = seed.layers[0], seed.layers[1]
    out, gA, gB, kap, cot, z0p, idx = M.it_pairs(
        U, Q, allidx, la, lb, ptmin, use_angles, False, d0_cm, 0.0,
        z_inner=M.ot_vmr_inner_z(seed.layers),
        bend_cut=M.ot_bend_cut(seed.layers))
    if gA is None:
        return None
    d0m = None
    used = [la, lb]
    if seed.arity == 3:
        lc = seed.layers[2]
        # seed_layers is DELIBERATELY NOT PASSED HERE. Attaching the third
        # stub is seed formation, not a projection: the real finder locates it
        # through innerThirdTable_, a middle->third r/z-bin LUT, while
        # rphimatchcut_/zmatchcut_ describe projections to layers the seed does
        # NOT occupy. Handing the projection menu this call asked whether
        # L2L3L4 may project into L2 -- its own third layer -- which its row
        # forbids, so both barrel triplets found exactly zero tracks. The
        # third-stub window therefore stays the derived one; we do not have
        # innerThirdTable_ and will not fake it with the wrong table.
        o3 = M.it_project(U, Q, dict(out), gA, gB, kap, cot, z0p, idx,
                          la, lb, lc, ptmin, False, d0_cm,
                          use_angles=use_angles)
        if "_trip" not in o3:
            return None
        ta, tb, tc = o3["_trip"]
        # d0 is SOLVED here; that is the whole point of the third layer
        _p0, d0m, kap3, ok3 = M.solve3(U["globalR"][ta], U["globalPhi"][ta],
                                       U["globalR"][tb], U["globalPhi"][tb],
                                       U["globalR"][tc], U["globalPhi"][tc])
        keep = ok3 & (np.abs(kap3) <= 1.0 / ptmin)
        gA, gB = ta[keep], tb[keep]
        gC, d0m, kap = tc[keep], d0m[keep], kap3[keep]
        dr = U["globalR"][gB] - U["globalR"][gA]
        sdr = np.where(np.abs(dr) > 0.5, dr, 1e9)
        cot = (U["globalZ"][gB] - U["globalZ"][gA]) / sdr
        z0p = U["globalZ"][gA] - U["globalR"][gA] * cot
        used.append(lc)
        out = {k: v for k, v in o3.items() if k != "_trip"}
        out["seed_objects"] = int(len(gA))
    else:
        gC = None
        out["seed_objects"] = int(len(gA))
    if not len(gA):
        return None
    # ---- follow to every remaining layer --------------------------------
    nconf = np.zeros(len(gA), np.int32)
    nconf_it = np.zeros(len(gA), np.int32)
    nconf_ot = np.zeros(len(gA), np.int32)
    hits = {}
    cand_tot = 0
    for L in targets:
        if L in used:
            continue
        oj = M.it_project(U, Q, {}, gA, gB, kap, cot, z0p, idx, la, lb, L,
                          ptmin, False, d0_cm, use_angles=use_angles,
                          d0_meas=d0m, seed_layers=seed.layers)
        cand_tot += int(oj.get("match_cand", 0))
        if "_pairidx" not in oj:
            continue
        pi = oj["_pairidx"]
        nconf[pi] += 1
        (nconf_ot if L > 10 else nconf_it)[pi] += 1
        hits[L] = (pi, oj["_trip"][2])
    out["cand_all_targets"] = cand_tot
    out["targets_followed"] = len([L for L in targets if L not in used])
    # ---- the 4-layer rule -----------------------------------------------
    # min_layers overrides the 4-layer rule. It has to be settable: the rule is
    # defined against six OT barrel layers, and a soft IT-only configuration may
    # have only three layers in total, where demanding four means demanding a
    # confirmation the geometry cannot supply.
    # A LAYER COUNT THAT DOES NOT CARE WHICH SYSTEM SUPPLIED THE LAYER IS WRONG
    # HERE. With a flat "3 layers" rule and the OT among the targets, an IT
    # DOUBLET completes on a single OT stub -- 2xIT + 1xOT -- which is not a
    # soft-track candidate at all: the IT pair alone cannot constrain d0 and the
    # OT stub sits at 5-10x the radius. MEASURED cost of allowing it, 200
    # events at 1 GeV: the fake fraction rises with the IT lever arm, IL3+IL4
    # 0.034 -> 0.067 but IL1+IL4 0.202 -> 0.557, while every arity-3 seed --
    # already 3xIT, so its OT hit really is optional -- shows no penalty at all
    # (0.029 -> 0.029). The entire effect was the doublets.
    #
    # So the requirement is stated per system: the IT must supply min_it_layers
    # by itself, and OT confirmations are counted separately. min_ot_conf = 0
    # makes the OT hit optional ("3xIT or 3xIT + 1xOT"), 1 makes it mandatory.
    if min_it_layers is not None:
        n_it_seed = sum(1 for L in seed.layers if L <= 4)
        keep = (n_it_seed + nconf_it >= min_it_layers) & (nconf_ot >= min_ot_conf)
        need = max(min_it_layers - n_it_seed, 0)
    elif min_layers is None:
        need = seed.min_proj(strict_cmssw)
        keep = nconf >= need
    else:
        need = max(min_layers - seed.arity, 0)
        keep = nconf >= need
    # SUMS ONLY. Per-chunk counters are added together by acc_add, so a mean or
    # a constant stored here would come back multiplied by the chunk count; the
    # report divides these by n_pairs_rule instead.
    out["conf_it_sum"] = int(nconf_it.sum())
    out["conf_ot_sum"] = int(nconf_ot.sum())
    out["n_pairs_rule"] = int(len(gA))
    out["tracks_before_minlayers"] = int(len(gA))
    out["tracks_to_fit"] = int(keep.sum())
    if not keep.any():
        out["n_layers_median"] = 0.0
        return out
    # median layers ON THE TRACKS THAT SURVIVE, not on every pair: the pairs
    # that fail the rule are by definition short and drag the median under the
    # minimum, which reads as though the rule were not being applied.
    out["n_layers_median"] = float(np.median(seed.arity + nconf[keep]))
    out["_gA"], out["_gB"] = gA[keep], gB[keep]
    out["_gC"] = None if gC is None else gC[keep]
    out["_d0"] = None if d0m is None else d0m[keep]
    out["_nconf"] = nconf[keep]
    # extra layer hits, remapped onto the surviving tracks
    remap = np.full(len(gA), -1, np.int64)
    remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    out["_hits"] = {L: (remap[pi][remap[pi] >= 0], gc[remap[pi] >= 0])
                    for L, (pi, gc) in hits.items()}
    return out


def filter_tracks(out, keep):
    """Restrict a run_seed result to a boolean subset of its tracks.

    Used to apply the KF chi2 acceptance, which can only be evaluated after the
    fit and therefore after the 4-layer rule that run_seed applies.
    """
    o = dict(out)
    o["_gA"], o["_gB"] = out["_gA"][keep], out["_gB"][keep]
    o["_gC"] = None if out.get("_gC") is None else out["_gC"][keep]
    o["_d0"] = None if out.get("_d0") is None else out["_d0"][keep]
    o["_nconf"] = out["_nconf"][keep]
    remap = np.full(len(keep), -1, np.int64)
    remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    o["_hits"] = {L: (remap[rows][remap[rows] >= 0], gc[remap[rows] >= 0])
                  for L, (rows, gc) in out.get("_hits", {}).items()}
    o["tracks_to_fit"] = int(keep.sum())
    return o


def recovered_keys(U, out):
    """TPs recovered by a seed, requiring the SEED's own layers to be truthful.

    A track is credited when every seed layer belongs to the same
    TrackingParticle. Confirmations on projected layers are what the 4-layer
    rule already counted; demanding they be truthful too would measure a
    different thing -- purity of the whole track rather than of the seed.
    """
    ga, gb = out["_gA"], out["_gB"]
    ta = U["tpIdx"][ga]
    real = (ta >= 0) & (ta == U["tpIdx"][gb])
    if out.get("_gC") is not None:
        real &= ta == U["tpIdx"][out["_gC"]]
    if not real.any():
        return np.empty(0, np.int64)
    return np.unique(M.tp_key(U["event"][ga][real], ta[real]))


# ==========================================================================
# duplicate removal
# ==========================================================================
MIN_SHARED_LAYERS = 3      # Settings.h minIndStubs_ = 3

# L1Trigger/TrackFindingTracklet PurgeDuplicate.cc:171, ranks{1,5,2,7,4,3,8,6}
# over seeds L1L2, L2L3, L3L4, L5L6, D1D2, D3D4, L1D1, L2D1 -- lower wins. So
# among barrel doublets the real priority is L1L2 > L3L4 > L2L3 > L5L6, and
# EXTENDED (triplet) seeds are pushed to rank 9, the lowest of all.
OT_SEED_RANK = {(11, 12): 1, (13, 14): 2, (12, 13): 5, (15, 16): 7}


def duplicate_removal(gidx, score, min_shared=MIN_SHARED_LAYERS):
    """Merge-style DR: drop a track sharing >= min_shared IDENTICAL hits with an
    already-accepted track in the same event.

    This is PurgeDuplicate's "merge" criterion with mergeComparison CompareBest
    -- nShareLay counted over layers where both tracks have a hit and it is the
    same hit, merged when nShareLay >= minIndStubs (3).

    PREFERENCE DIVERGES DELIBERATELY. The real code picks the survivor from a
    static seedRank table, which ranks every extended (triplet) seed below every
    prompt doublet regardless of what the individual tracks look like -- and
    carries a comment admitting one of its swaps is unexplained ("The swap here
    reduces the duplicate rate for extended tracking by 1/4. Why???"). Here the
    survivor is the track with the better rank_score, which already folds in the
    layer count and the chi2 the static table only proxies for. That is a
    superset of the information, and it is the natural place for the ranking to
    be used.

    Implementation is by INVERTED INDEX on the cluster indices rather than
    pairwise comparison: a global cluster index identifies a hit uniquely within
    the chunk, so each candidate only has to look at tracks already accepted on
    one of its own ~8 hits. Pairwise would be O(n^2) at ~3,000 tracks/event.

    No event argument is needed: a cluster belongs to exactly one event, so two
    tracks from different events can never share one and cross-event merging is
    impossible by construction.

    THIS IS THE ONE PYTHON ROW LOOP IN THE CENSUS HOT PATH, and it was measured
    rather than assumed. At the real scale of 50,000 tracks per chunk:
        accidental sharing only (worst case)   2.30 s, 97% kept
        realistic duplicates, 4 per particle   0.32 s, 26% kept
        realistic duplicates, 8 per particle   0.29 s, 14% kept
    i.e. 1-10 minutes across a 250-chunk census that takes ~90. Heavy
    duplication makes it CHEAPER, not dearer, because a rejected track
    short-circuits before touching the owner lists.
    If it ever does become hot, the fix is to build the conflict graph with a
    sparse incidence matmul (tracks x clusters) and run the greedy over EDGES
    rather than over every hit. Premature today.
    """
    order = np.argsort(-score, kind="stable")
    keep = np.zeros(len(score), bool)
    owners = {}                      # cluster index -> list of accepted tracks
    from collections import Counter
    for t in order:
        hits = gidx[t][gidx[t] >= 0]
        c = Counter()
        for h in hits:
            o = owners.get(int(h))
            if o:
                c.update(o)
        if c and max(c.values()) >= min_shared:
            continue
        keep[t] = True
        for h in hits:
            owners.setdefault(int(h), []).append(t)
    return keep
