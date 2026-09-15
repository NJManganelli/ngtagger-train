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
# L2L3L4 and L4L5L6 -- the design's only barrel triplet seeds.
OT_TRIPLETS = ((12, 13, 14), (14, 15, 16))
MIN_LAYERS = 4


class Seed:
    __slots__ = ("arity", "layers", "tag", "note")

    def __init__(self, layers, note=""):
        self.layers = tuple(layers)
        self.arity = len(self.layers)
        self.tag = "+".join(NAME[L] for L in self.layers)
        self.note = note

    def __repr__(self):
        return f"<{'doublet' if self.arity == 2 else 'triplet'} {self.tag}>"

    def min_proj(self, strict_cmssw=False):
        """Confirmed projections still required to reach a 4-layer track."""
        return MIN_LAYERS - (2 if strict_cmssw else self.arity)


NAME = {1: "IL1", 2: "IL2", 3: "IL3", 4: "IL4",
        11: "OL1", 12: "OL2", 13: "OL3", 14: "OL4", 15: "OL5", 16: "OL6"}


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
             strict_cmssw=False, d0_cm=0.0):
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
        U, Q, allidx, la, lb, ptmin, use_angles, False, d0_cm, 0.0)
    if gA is None:
        return None
    d0m = None
    used = [la, lb]
    if seed.arity == 3:
        lc = seed.layers[2]
        o3 = M.it_project(U, Q, dict(out), gA, gB, kap, cot, z0p, idx,
                          la, lb, lc, ptmin, False, d0_cm, use_angles=use_angles)
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
    hits = {}
    cand_tot = 0
    for L in targets:
        if L in used:
            continue
        oj = M.it_project(U, Q, {}, gA, gB, kap, cot, z0p, idx, la, lb, L,
                          ptmin, False, d0_cm, use_angles=use_angles,
                          d0_meas=d0m)
        cand_tot += int(oj.get("match_cand", 0))
        if "_pairidx" not in oj:
            continue
        pi = oj["_pairidx"]
        nconf[pi] += 1
        hits[L] = (pi, oj["_trip"][2])
    out["cand_all_targets"] = cand_tot
    out["targets_followed"] = len([L for L in targets if L not in used])
    # ---- the 4-layer rule -----------------------------------------------
    need = seed.min_proj(strict_cmssw)
    keep = nconf >= need
    out["min_proj"] = need
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
