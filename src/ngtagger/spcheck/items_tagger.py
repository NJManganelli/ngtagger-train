"""Fast-gate items on the jet-tagging study (plan App. D.1: join uniqueness, the files 1-2
split guard, the label mask; C.4: working points not matched). They import
eval_refitq/tagger_sp from THIS code root and read the study's jet caches of file 3
(tagger_sp/build.py outputs, one per mask) and the event identities of the TTbar files.
The study reads canon_v4 since 2026-10-17; until its jet caches exist (build.py, after the
track-MVA heads are trained) every item is n/a (WAITING)."""
from __future__ import annotations

import os

import numpy as np

from .core import Item, NotApplicable, event_replicas, identity, same_input

TAGGER_DIR = "eval_refitq/tagger_sp"
JETS = "/Volumes/WDMac/smartpixels_scratch/tagger_sp_canon_v4/cache/jets"   # tagger_sp common.OUT (canon_v4)
WAITING = ("no canon_v4 jet cache yet: tagger_sp/build.py trains the track-MVA heads before it writes them; "
           "waiting for the tagger-recipe package")
MASKS = ("IIII", "AAAA", "AAAI", "IAAA")
STOCK_B_CUT = 0.15


def _file_no(path: str) -> int:
    import re

    m = re.search(r"_f(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _jets(mask: str, n: int) -> dict:
    p = os.path.join(JETS, f"ttbar_{mask}_f{n:02d}.npz")
    if not os.path.exists(p):
        raise NotApplicable(WAITING)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _event_ids(path: str) -> np.ndarray:
    import uproot

    t = uproot.open(path)["Events"]
    return np.stack([t[c].array(library="np").astype(np.int64) for c in ("run", "luminosityBlock", "event")], 1)


# ---------------------------------------------------------------- split guard
class SplitGuard(Item):
    id = "tagger.split_guard"
    title = "TTbar files 1-2 (cluster-CNN training events) are never test data; the split is by event identity"
    c4_row = "D.1 tagger item: the files 1-2 split guard"
    known_bad_desc = "common.TTBAR_TRAIN_ONLY emptied (a split that may draw test events from files 1-2)"
    rule_desc = ("common.split_table on the event identities of TTbar files 1-10: no test event in files 1-2; "
                 "round(0.2 x events) test events; the same test events for AAAA and IIII (different entry "
                 "orders); no is_test jet in the cached files 1-2 of any mask; identity with the reference")
    trees = (TAGGER_DIR,)

    def _stat(self, ctx, train_only=None):
        C = ctx.script_module(TAGGER_DIR, "common")
        saved = C.TTBAR_TRAIN_ONLY
        if train_only is not None:
            C.TTBAR_TRAIN_ONLY = train_only
        try:
            out = {"input": ctx.input, "events": ctx.events, "train_only": list(C.TTBAR_TRAIN_ONLY)}
            sets = {}
            for mask in ("AAAA", "IIII"):
                ev = {n: _event_ids(C.path("ttbar", mask, n)) for n in range(1, 11)}
                test = C.split_table("ttbar", ev)
                sets[mask] = {tuple(x) for n in ev for x in ev[n][test[n]]}
                out[f"{mask}_n_test"] = int(sum(v.sum() for v in test.values()))
                out[f"{mask}_n_test_files_1_2"] = int(test[1].sum() + test[2].sum())
                out[f"{mask}_n_events"] = int(sum(len(v) for v in ev.values()))
            out["same_test_events_AAAA_IIII"] = sets["AAAA"] == sets["IIII"]
            out["cached_test_jets_files_1_2"] = int(sum(_jets(m, n)["is_test"].sum() for m in MASKS for n in (1, 2)))
        finally:
            C.TTBAR_TRAIN_ONLY = saved
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, train_only=())

    def judge(self, stat, ref):
        why = []
        for m in ("AAAA", "IIII"):
            if stat[f"{m}_n_test_files_1_2"]:
                why.append(f"{m}: {stat[f'{m}_n_test_files_1_2']} test events in files 1-2")
            if stat[f"{m}_n_test"] != round(0.2 * stat[f"{m}_n_events"]):
                why.append(f"{m}: {stat[f'{m}_n_test']} test events, not 20%")
        if not stat["same_test_events_AAAA_IIII"]:
            why.append("AAAA and IIII test events differ")
        if stat["cached_test_jets_files_1_2"]:
            why.append(f"{stat['cached_test_jets_files_1_2']} cached test jets in files 1-2")
        if same_input(stat, ref) and stat["train_only"] == ref["stat"]["train_only"]:
            why += identity(stat, ref, ["AAAA_n_test", "IIII_n_test"])[1]
        return not why, "files 1-2 train-only, 20% test, identical across masks" if not why else "; ".join(why)


# ---------------------------------------------------------------- label mask
class LabelMask(Item):
    id = "tagger.label_mask"
    title = "jet labels and the kept-jet mask: labelled jets only, classes as defined, one label era"
    c4_row = "D.1 tagger item: the label mask"
    known_bad_desc = "the kept-jet mask without the label and gen-pT requirements (unmatched pile-up jets admitted)"
    rule_desc = ("on the first 5 events of the gate input: labels.label_jets through nano.load_jets equals the "
                 "study's jet cache (kept jets, in order); every kept jet has label 0-7 and gen pT > 5 GeV; the "
                 "b-vs-udsg / c-vs-udsg / tau classes of analyze.PAIRS are those of the definition sheet; every "
                 "file of each mask's training manifest comes from one production era")
    trees = (TAGGER_DIR,)
    PAIRS_SHEET = {"b_vs_udsg": (["b"], ["light", "gluon"]), "c_vs_udsg": (["charm"], ["light", "gluon"])}

    def _stat(self, ctx, keep_unlabelled=False):
        import json

        import awkward as ak

        from ngtagger.data.labels import label_jets
        from ngtagger.data.nano import load_jets

        C = ctx.script_module(TAGGER_DIR, "common")
        AN = ctx.script_module(TAGGER_DIR, "analyze")
        jets, _, gen = load_jets([ctx.input], feature_groups=["baseline"], max_events=ctx.events,
                                 track_table=C.TRACK_TABLE)
        label, _, target_phys, keep = label_jets(jets, gen, max_dr=0.4)
        if keep_unlabelled:                      # every jet kept: unmatched pile-up jets enter the labelled set
            keep = label > -2
        lab = ak.to_numpy(ak.flatten(label))
        kp = ak.to_numpy(ak.flatten(keep))
        tpt = ak.to_numpy(ak.flatten(target_phys))
        nj = ak.to_numpy(ak.num(jets.pt))
        entry = np.repeat(np.arange(len(nj)), nj)
        cache = _jets("AAAA", _file_no(ctx.input)) if "AAAA" in os.path.basename(ctx.input) else None
        out = {"input": ctx.input, "events": ctx.events, "keep_unlabelled": keep_unlabelled,
               "n_jets": int(len(lab)), "n_kept": int(kp.sum()),
               "kept_per_class": np.bincount(lab[kp] + 1, minlength=9).tolist(),
               "n_kept_unlabelled": int(np.sum(kp & (lab < 0))), "n_kept_genpt_le5": int(np.sum(kp & (tpt <= 5.0)))}
        if cache is not None:
            sel = cache["entry"] < ctx.events
            out["cache_equal"] = bool(np.array_equal(cache["label"][sel], lab[kp])
                                      and np.array_equal(cache["entry"][sel], entry[kp]))
        out["pairs_as_sheet"] = all(list(AN.PAIRS[k][:2]) == list(v) for k, v in self.PAIRS_SHEET.items())
        eras = {}
        for m in MASKS:
            p = C.manifest_path(m)
            if os.path.exists(p):
                files = [f[2] for f in json.load(open(p))["files"]]
                eras[m] = sorted({e for e in ("canon_v2", "canon_v3", "canon_v4") if any(f"/{e}" in f for f in files)})
        out["manifest_eras"] = eras
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, keep_unlabelled=True)

    def judge(self, stat, ref):
        why = []
        if stat["n_kept_unlabelled"]:
            why.append(f"{stat['n_kept_unlabelled']} kept jets have no label")
        if stat["n_kept_genpt_le5"]:
            why.append(f"{stat['n_kept_genpt_le5']} kept jets have gen pT <= 5 GeV")
        if stat.get("cache_equal") is False:
            why.append("labels differ from the study's jet cache")
        if not stat["pairs_as_sheet"]:
            why.append("analyze.PAIRS differs from the definition sheet")
        mixed = {m: e for m, e in stat["manifest_eras"].items() if len(e) != 1}
        if mixed:
            why.append(f"manifests mixing eras: {mixed}")
        if same_input(stat, ref) and stat["keep_unlabelled"] == ref["stat"]["keep_unlabelled"]:
            why += identity(stat, ref, ["n_jets", "n_kept", "kept_per_class"])[1]
        return not why, f"{stat['n_kept']} labelled jets, cache identical, one era" if not why else "; ".join(why)


# ---------------------------------------------------------------- join keys
class TaggerJoinKeys(Item):
    id = "tagger.join_keys"
    title = "tagger joins: event identities unique per file and study-wide; the across-arm jet key reported"
    c4_row = "ambiguous join (tagger side; D.1 join uniqueness)"
    known_bad_desc = "one event identity duplicated in file 3's identity array"
    rule_desc = ("(run, lumi, event) unique in every TTbar file of every mask; common.event_key unique over the "
                 "study; the (event, label, truth pT) key of the jets common to all arms REPORTED with its "
                 "duplicate count (not unique: two L1 jets matched to one gen object), identity with the reference")
    trees = (TAGGER_DIR,)

    def _stat(self, ctx, dup_event=False):
        C = ctx.script_module(TAGGER_DIR, "common")
        out = {"input": ctx.input, "events": ctx.events, "dup_event_injected": dup_event,
               "files_with_duplicate_ids": 0, "event_key_duplicates": 0, "jet_key_duplicates": {}}
        keys = []
        for m in MASKS:
            for n in range(1, 11):
                ev = _event_ids(C.path("ttbar", m, n)) if m in ("AAAA", "IIII") else _jets(m, n)["evid"]
                if dup_event and m == "AAAA" and n == _file_no(ctx.input):
                    ev = np.vstack([ev, ev[:1]])
                out["files_with_duplicate_ids"] += int(len(np.unique(ev, axis=0)) != len(ev))
                if m == "AAAA":
                    keys.append(C.event_key(0, n, ev))
            jd = 0
            for n in range(1, 11):
                d = _jets(m, n)
                k = np.column_stack([d["evid"][d["entry"]], d["label"], d["truth_pt"]])
                jd += len(k) - len(np.unique(k, axis=0))
            out["jet_key_duplicates"][m] = int(jd)
        kk = np.concatenate(keys)
        out["event_key_duplicates"] = int(len(kk) - len(np.unique(kk)))
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, dup_event=True)

    def judge(self, stat, ref):
        why = []
        if stat["files_with_duplicate_ids"]:
            why.append(f"{stat['files_with_duplicate_ids']} files with duplicate event ids")
        if stat["event_key_duplicates"]:
            why.append(f"{stat['event_key_duplicates']} duplicate study event keys")
        if same_input(stat, ref) and stat["dup_event_injected"] == ref["stat"]["dup_event_injected"]:
            why += identity(stat, ref, ["jet_key_duplicates"])[1]
        return not why, (f"event keys unique; across-arm jet key NOT unique (reported): {stat['jet_key_duplicates']}"
                         if not why else "; ".join(why))


# ---------------------------------------------------------------- matched working points
class MatchedWP(Item):
    id = "tagger.matched_wp"
    title = "matched working points: the realised background fraction equals the target in every arm and replica"
    c4_row = "working points not matched"
    known_bad_desc = "the current analyze.thresholds / frac_above (jets tied at the threshold dropped)"
    rule_desc = ("stock bTagScore of the file-3 jet caches, all four arms, b vs udsg in the study acceptance; "
                 "targets: the menu point (IIII udsg fraction above 0.15 on these jets), 10 %, 1 %; 300 event-"
                 "bootstrap replicas with thresholds re-derived per replica: max |realised - target| <= 1e-9; "
                 "identity of the nominal efficiencies with the reference")
    trees = (TAGGER_DIR,)
    N_REP = 300

    def _arms(self, ctx):
        n = _file_no(ctx.input)
        arms = {}
        for m in MASKS:
            d = _jets(m, n)
            acc = (d["pt"] > 15.0) & (np.abs(d["eta"]) < 2.4)
            arms[m] = {"score": d["stock_bTagScore"][acc].astype(np.float64), "label": d["label"][acc],
                       "entry": d["entry"][acc], "n_events": len(d["evid"])}
        return arms

    def _stat(self, ctx, use_analyze=False):
        from ngtagger.definitions import matched_wp as MW

        arms = self._arms(ctx)
        I = arms["IIII"]
        bkg_I = np.isin(I["label"], (2, 3))
        menu = float(np.mean(I["score"][bkg_I] > STOCK_B_CUT))
        targets = {"menu": menu, "10%": 0.10, "1%": 0.01}
        reps = event_replicas(I["n_events"], self.N_REP, 20261009).astype(np.float64)
        AN = ctx.script_module(TAGGER_DIR, "analyze") if use_analyze else None
        out = {"input": ctx.input, "events": "all 100 (jet caches)", "method": "analyze.thresholds/frac_above (KNOWN-BAD)"
               if use_analyze else "definitions.matched_wp (tie split)", "targets": targets,
               "max_abs_realised_minus_target": 0.0, "nominal": {}}
        worst = 0.0
        for m, a in arms.items():
            sig, bkg = a["label"] == 0, np.isin(a["label"], (2, 3))
            for name, tgt in targets.items():
                W = np.vstack([np.ones(a["n_events"]), reps])[:, a["entry"]]
                if use_analyze:
                    thr = AN.thresholds(a["score"][bkg], W[:, bkg], tgt)
                    real = AN.frac_above(a["score"][bkg], W[:, bkg], thr)
                    eff = AN.frac_above(a["score"][sig], W[:, sig], thr)
                else:
                    real = np.empty(len(W))
                    eff = np.empty(len(W))
                    for r in range(len(W)):
                        p = MW.matched_point(a["score"][sig], a["score"][bkg], tgt, W[r, sig], W[r, bkg])
                        real[r], eff[r] = p["realised_background"], p["efficiency"]
                worst = max(worst, float(np.nanmax(np.abs(real - tgt))))
                out["nominal"][f"{m}@{name}"] = {"efficiency": float(eff[0]), "realised_background": float(real[0])}
        out["max_abs_realised_minus_target"] = worst
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, use_analyze=True)

    def judge(self, stat, ref):
        why = []
        if not stat["max_abs_realised_minus_target"] <= 1e-9:
            why.append(f"realised background off the target by up to {stat['max_abs_realised_minus_target']:.4f}")
        if same_input(stat, ref) and stat["method"] == ref["stat"]["method"]:
            why += identity(stat, ref, ["nominal"])[1]
        return not why, ("realised = target in every arm and replica" if not why else "; ".join(why))


ITEMS = [SplitGuard(), LabelMask(), TaggerJoinKeys(), MatchedWP()]
