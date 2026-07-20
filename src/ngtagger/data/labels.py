"""Truth labeling of L1 jets from the withGen nano tables, porting the
8-class scheme of upstream TrainTagger (b, charm, light, gluon, taup, taum,
muon, electron) to gen matching done in-pipeline:

  - jets matched to GenJet by deltaR -> hadronFlavour / partonFlavour
  - jets matched to GenVisTau by deltaR -> tau flavour and charge
  - jets matched to prompt gen electrons/muons (GenPart) by deltaR

The regression target follows upstream: gen jet pt for hadronic classes,
visible lepton pt for leptonic classes, ratio clipped to [0.3, 2].
"""

from __future__ import annotations

import awkward as ak
import numpy as np

CLASS_LABELS = ["b", "charm", "light", "gluon", "taup", "taum", "muon", "electron"]

# 3-class jet-charge scheme (model-space study B.2.3): index = sign(parton
# electric charge) + 1, with gluon/unmatched central.
CHARGE_CLASS_LABELS = ["qminus", "neutral", "qplus"]

# statusFlags bit 0: isPrompt (NanoAOD convention)
_IS_PROMPT = 1 << 0


def _delta_r2(eta1, phi1, eta2, phi2):
    dphi = np.pi - abs(np.pi - abs(phi1 - phi2))
    deta = eta1 - eta2
    return deta * deta + dphi * dphi


def _match(jets, other, max_dr: float = 0.4):
    """Index of the closest `other` object within max_dr for each jet (-1 if none),
    plus the matched-object mask. nested=True cartesian gives the (event, jet,
    obj) grouping directly, and stays well-defined for events with zero `other`
    objects (empty inner lists -> argmin None -> unmatched)."""
    j = ak.zip({"eta": jets.eta, "phi": jets.phi})
    o = ak.zip({"eta": other.eta, "phi": other.phi})
    pairs = ak.cartesian({"j": j, "o": o}, axis=1, nested=True)
    idxp = ak.argcartesian({"j": j, "o": o}, axis=1, nested=True)
    dr2 = _delta_r2(pairs.j.eta, pairs.j.phi, pairs.o.eta, pairs.o.phi)  # (event, jet, obj)

    best = ak.argmin(dr2, axis=2, keepdims=True)
    best_dr2 = ak.firsts(dr2[best], axis=2)
    best_idx = ak.firsts(idxp.o[best], axis=2)
    matched = ak.fill_none(best_dr2 < max_dr * max_dr, False)
    idx = ak.fill_none(ak.where(matched, best_idx, -1), -1)
    return idx, matched


def _take(vals2d, idx, matched, default):
    """Gather vals2d[(event, obj)] at the per-jet matched indices; unmatched
    jets get `default` WITHOUT reading the array (index-0 reads into empty
    collections crash for events with zero objects)."""
    return ak.fill_none(vals2d[ak.mask(idx, matched)], default)


def label_jets(jets: ak.Array, gen: dict, max_dr: float = 0.4, gen_pt_min: float = 5.0):
    """Return (class_label int array, one-hot y, target_pt ratio, target_pt_phys,
    keep mask) aligned with jets, mirroring the upstream class definitions."""
    genjets = gen["GenJet"]
    vistaus = gen["GenVisTau"]
    genparts = gen["GenPart"]

    gj_idx, gj_match = _match(jets, genjets, max_dr)
    tau_idx, tau_match = _match(jets, vistaus, max_dr)

    prompt = (genparts.statusFlags & _IS_PROMPT) != 0
    els = genparts[(abs(genparts.pdgId) == 11) & prompt & (genparts.pt > 5)]
    mus = genparts[(abs(genparts.pdgId) == 13) & prompt & (genparts.pt > 5)]
    el_idx, el_match = _match(jets, els, max_dr)
    mu_idx, mu_match = _match(jets, mus, max_dr)

    hflav = _take(genjets.hadronFlavour, gj_idx, gj_match, -1)
    pflav = _take(genjets.partonFlavour, gj_idx, gj_match, 0)
    gj_pt = _take(genjets.pt, gj_idx, gj_match, 0.0)

    tau_charge = _take(vistaus.charge, tau_idx, tau_match, 0)
    tau_pt = _take(vistaus.pt, tau_idx, tau_match, 0.0)
    el_pt = _take(els.pt, el_idx, el_match, 0.0)
    mu_pt = _take(mus.pt, mu_idx, mu_match, 0.0)

    base = gj_match & (gj_pt > gen_pt_min)
    no_lep = ~tau_match & ~el_match & ~mu_match

    conditions = {
        "b": base & no_lep & (hflav == 5),
        "charm": base & no_lep & (hflav == 4),
        "light": base & no_lep & (hflav == 0) & (abs(pflav) <= 3),
        "gluon": base & no_lep & (hflav == 0) & (pflav == 21),
        "taup": base & tau_match & ~el_match & ~mu_match & (tau_charge > 0),
        "taum": base & tau_match & ~el_match & ~mu_match & (tau_charge < 0),
        "muon": mu_match & ~tau_match & ~el_match,
        "electron": el_match & ~tau_match & ~mu_match,
    }

    label = ak.zeros_like(jets.pt, dtype=np.int64) - 1
    for i, name in enumerate(CLASS_LABELS):
        label = ak.where(conditions[name], i, label)

    hadronic = conditions["b"] | conditions["charm"] | conditions["light"] | conditions["gluon"]
    leptonic = conditions["taup"] | conditions["taum"] | conditions["muon"] | conditions["electron"]
    lep_pt = tau_pt + el_pt + mu_pt  # matches are exclusive by construction above

    target_pt_phys = hadronic * gj_pt + leptonic * lep_pt
    ratio = ak.where(jets.pt > 0, target_pt_phys / jets.pt, 0.0)
    target_pt = np.clip(ak.to_numpy(ak.flatten(ratio)), 0.3, 2.0)

    keep = (label >= 0) & (target_pt_phys > gen_pt_min)
    return label, target_pt, target_pt_phys, keep


def parton_charge_class(pflav):
    """GenJet.partonFlavour -> 3-class charge index {0: q-, 1: neutral, 2: q+}.

    Exact pdgId -> class map (study B.2.3: u-type quarks carry +2/3, d-type
    -1/3, antiquarks negated; gluon and unmatched are neutral/central):

        +1 d    -> q-      -1 dbar -> q+
        +2 u    -> q+      -2 ubar -> q-
        +3 s    -> q-      -3 sbar -> q+
        +4 c    -> q+      -4 cbar -> q-
        +5 b    -> q-      -5 bbar -> q+
        +6 t    -> q+      -6 tbar -> q-
        21 g, 0 (unmatched/undefined), anything else -> neutral
    """
    absf = abs(pflav)
    is_quark = (absf >= 1) & (absf <= 6)
    up_type = is_quark & (absf % 2 == 0)  # u, c, t: +2/3; d, s, b: -1/3
    sign = np.sign(pflav)
    qsign = ak.where(up_type, sign, -sign)  # sign of the parton electric charge
    return ak.values_astype(ak.where(is_quark, qsign + 1, 1), np.int64)


def label_jet_charge(jets: ak.Array, gen: dict, max_dr: float = 0.4):
    """Per-jet 3-class charge label from the deltaR-matched GenJet
    partonFlavour, aligned with `jets` (same nesting as label_jets output).
    Additive to the 8-class scheme -- label_jets is untouched; leptonic jets
    also get a charge class here (their charge is already carried by the
    8-class labels; mask downstream if a quark-only sample is wanted)."""
    genjets = gen["GenJet"]
    gj_idx, gj_match = _match(jets, genjets, max_dr)
    pflav = _take(genjets.partonFlavour, gj_idx, gj_match, 0)
    return parton_charge_class(pflav)
