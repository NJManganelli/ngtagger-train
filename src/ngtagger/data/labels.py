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

# statusFlags bit 0: isPrompt (NanoAOD convention)
_IS_PROMPT = 1 << 0


def _delta_r2(eta1, phi1, eta2, phi2):
    dphi = np.pi - abs(np.pi - abs(phi1 - phi2))
    deta = eta1 - eta2
    return deta * deta + dphi * dphi


def _match(jets, other, max_dr: float = 0.4):
    """Index of the closest `other` object within max_dr for each jet (-1 if none),
    plus the matched-object mask, via argcartesian minimization."""
    pairs = ak.argcartesian({"jet": jets.eta, "obj": other.eta}, axis=1)
    dr2 = _delta_r2(
        jets.eta[pairs.jet], jets.phi[pairs.jet], other.eta[pairs.obj], other.phi[pairs.obj]
    )
    # group pairs per jet: cartesian is jet-major with len(other) entries per jet
    n_other = ak.num(other.eta, axis=1)
    n_jets = ak.num(jets.eta, axis=1)
    grouped_dr2 = ak.unflatten(dr2, ak.flatten(ak.broadcast_arrays(n_other, jets.eta)[0]), axis=1)
    grouped_obj = ak.unflatten(pairs.obj, ak.flatten(ak.broadcast_arrays(n_other, jets.eta)[0]), axis=1)

    best = ak.argmin(grouped_dr2, axis=2, keepdims=True)
    best_dr2 = ak.firsts(grouped_dr2[best], axis=2)
    best_idx = ak.firsts(grouped_obj[best], axis=2)
    matched = ak.fill_none(best_dr2 < max_dr * max_dr, False)
    idx = ak.fill_none(ak.where(matched, best_idx, -1), -1)
    # events with zero `other` objects give empty groups -> fill with -1
    return idx, matched


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

    safe_gj = ak.where(gj_idx >= 0, gj_idx, 0)
    hflav = ak.where(gj_match, genjets.hadronFlavour[safe_gj], -1)
    pflav = ak.where(gj_match, genjets.partonFlavour[safe_gj], 0)
    gj_pt = ak.where(gj_match, genjets.pt[safe_gj], 0.0)

    safe_tau = ak.where(tau_idx >= 0, tau_idx, 0)
    tau_charge = ak.where(tau_match, vistaus.charge[safe_tau], 0)
    tau_pt = ak.where(tau_match, vistaus.pt[safe_tau], 0.0)
    safe_el = ak.where(el_idx >= 0, el_idx, 0)
    el_pt = ak.where(el_match, els.pt[safe_el], 0.0)
    safe_mu = ak.where(mu_idx >= 0, mu_idx, 0)
    mu_pt = ak.where(mu_match, mus.pt[safe_mu], 0.0)

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
