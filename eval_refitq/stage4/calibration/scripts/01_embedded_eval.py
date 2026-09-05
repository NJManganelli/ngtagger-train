"""Study 1: embedded (centrally-trained hls4ml SC4 NG) tagger AUC on OUR
gen-match-labeled jets, per view, on (a) all labeled jets and (b) exactly the
Stage-4 test split (seed 12345, test_fraction 0.2 — replicated from
ngtagger.train.trainer.prepare_dataset and verified against the stage-4
prediction dumps).

Alignment is EXACT/index-level: the embedded scores are branches of the very
jet table (L1puppiJetSC4NG) the training pipeline reads, so the labeled-jet
selection (keep mask) and split permutation apply row-for-row.

Outputs (incremental):
  calibration/embedded_eval.json           per-view AUCs + verification + pairing
  calibration/dumps/embedded_<view>.npz    per-jet aligned arrays for reuse
"""
from __future__ import annotations

import json
import os
import sys

import awkward as ak
import numpy as np
import uproot
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(CAL), "scripts"))  # stage4/scripts
import run_matrix  # noqa: E402  (importable: main() is __main__-guarded)

from ngtagger.data.labels import CLASS_LABELS, label_jets  # noqa: E402
from ngtagger.data.nano import load_jets  # noqa: E402

# embedded-score branch -> our class label, in CLASS_LABELS order
SCORE_FIELDS = {
    "b": "bTagScore", "charm": "cTagScore", "light": "udsTagScore",
    "gluon": "gTagScore", "taup": "tau_pTagScore", "taum": "tau_nTagScore",
    "muon": "muTagScore", "electron": "eTagScore",
}
SPLIT_SEED, TEST_FRACTION = 12345, 0.2  # from run_matrix.run_cell
OUT_JSON = os.path.join(CAL, "embedded_eval.json")
N_BOOT = 1000


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def per_class_auc(y: np.ndarray, probs: np.ndarray):
    """y: int labels; probs: (n, 8). Returns ({class: {auc, se_hm, n_pos}}, macro)."""
    out, vals = {}, []
    for i, name in enumerate(CLASS_LABELS):
        pos = y == i
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            out[name] = {"auc": None, "se_hm": None, "n_pos": n_pos}
            continue
        a = float(roc_auc_score(pos, probs[:, i]))
        out[name] = {"auc": a, "se_hm": hanley_mcneil_se(a, n_pos, n_neg),
                     "n_pos": n_pos}
        vals.append(a)
    return out, (float(np.mean(vals)) if vals else None)


def macro_auc(y, probs):
    _, m = per_class_auc(y, probs)
    return m


def boot_macro(y, probs, n_boot=N_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = macro_auc(y[idx], probs[idx])
        if m is not None:
            vals.append(m)
    return float(np.std(vals))


def load_view(view: str):
    """Replicate prepare_dataset's jet selection & split; keep embedded scores."""
    files = run_matrix.files_for(view)
    jets, constituents, gen = load_jets(files, n_const=16, feature_groups=["baseline"])
    label, _tpt, _tptp, keep = label_jets(jets, gen, max_dr=0.4)

    # per-jet event number + file index (event-major flatten order == uproot order)
    ev_no, file_idx = [], []
    for fi, f in enumerate(files):
        with uproot.open(f + ":Events") as t:
            e = t["event"].array(library="np")
        ev_no.append(e)
        file_idx.append(np.full(len(e), fi))
    ev_no = np.concatenate(ev_no)
    file_idx = np.concatenate(file_idx)
    nj = ak.to_numpy(ak.num(jets.pt, axis=1))
    assert len(nj) == len(ev_no), "event count mismatch"
    jet_ev = np.repeat(ev_no, nj)
    jet_file = np.repeat(file_idx, nj)

    flat = lambda a: ak.to_numpy(ak.flatten(a))
    kmask = flat(keep).astype(bool)
    y = flat(label)[kmask].astype(int)
    probs = np.stack([flat(jets[SCORE_FIELDS[c]]) for c in CLASS_LABELS], axis=1)[kmask]
    d = {
        "y": y, "probs": probs.astype(np.float64),
        "pt": flat(jets.pt)[kmask], "eta": flat(jets.eta)[kmask],
        "phi": flat(jets.phi)[kmask],
        "event": jet_ev[kmask], "file": jet_file[kmask],
        "n_events": len(ev_no),
    }
    # replicate the split (depends only on n and seed)
    n = len(y)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n)
    n_test = int(n * TEST_FRACTION)
    is_test = np.zeros(n, bool)
    is_test[idx[:n_test]] = True
    d["is_test"] = is_test
    d["test_order"] = idx[:n_test]  # order used by the stage-4 dumps
    return d


def verify_split(view: str, d) -> dict:
    """Cross-check reconstructed test split against a stage-4 pred dump."""
    dump = os.path.join(os.path.dirname(CAL), "pred_dumps", f"{view}__baseline__s1.npz")
    with np.load(dump, allow_pickle=True) as f:
        kin_pt, y_true = f["kin_pt"], f["y_true"]
    ours_pt = d["pt"][d["test_order"]]
    ours_y = d["y"][d["test_order"]]
    ok_n = len(kin_pt) == len(ours_pt)
    ok_pt = ok_n and bool(np.allclose(kin_pt, ours_pt.astype(np.float32), atol=1e-4))
    ok_y = ok_n and bool((y_true == ours_y).all())
    return {"n_dump": int(len(kin_pt)), "n_ours": int(len(ours_pt)),
            "pt_match": ok_pt, "y_match": ok_y}


def pair_views(dA, dB, max_dr=0.2):
    """Match labeled jets across two views by (file, event) + closest deltaR.

    Greedy one-to-one: for each A jet, nearest B jet in the same event within
    max_dr; B jets used at most once (ambiguities counted)."""
    keyB = {}
    for j, (f, e) in enumerate(zip(dB["file"], dB["event"])):
        keyB.setdefault((f, e), []).append(j)
    iA, iB, n_multi = [], [], 0
    usedB = set()
    for i, (f, e) in enumerate(zip(dA["file"], dA["event"])):
        cand = keyB.get((f, e), [])
        if not cand:
            continue
        deta = dA["eta"][i] - dB["eta"][cand]
        dphi = np.pi - np.abs(np.pi - np.abs(dA["phi"][i] - dB["phi"][cand]))
        dr2 = deta**2 + dphi**2
        order = np.argsort(dr2)
        inside = dr2 < max_dr**2
        if inside.sum() > 1:
            n_multi += 1
        for o in order:
            if not inside[o]:
                break
            j = cand[o]
            if j not in usedB:
                iA.append(i)
                iB.append(j)
                usedB.add(j)
                break
    return np.array(iA), np.array(iB), n_multi


def paired_boot_delta(yA, pA, yB, pB, n_boot=N_BOOT, seed=11):
    """Bootstrap over matched jet pairs: delta macro + per-class deltas."""
    rng = np.random.default_rng(seed)
    n = len(yA)
    dm, dc = [], {c: [] for c in CLASS_LABELS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aA, mA = per_class_auc(yA[idx], pA[idx])
        aB, mB = per_class_auc(yB[idx], pB[idx])
        if mA is not None and mB is not None:
            dm.append(mA - mB)
        for c in CLASS_LABELS:
            if aA[c]["auc"] is not None and aB[c]["auc"] is not None:
                dc[c].append(aA[c]["auc"] - aB[c]["auc"])
    out = {"macro_delta_std": float(np.std(dm)),
           "per_class_delta_std": {c: (float(np.std(v)) if len(v) > 50 else None)
                                   for c, v in dc.items()}}
    return out


def main():
    res = {"class_labels": list(CLASS_LABELS), "split_seed": SPLIT_SEED,
           "test_fraction": TEST_FRACTION, "views": {}}
    data = {}
    for view in ["1111", "1100", "0000"]:
        print(f"[load] {view}", flush=True)
        d = load_view(view)
        data[view] = d
        ssum = d["probs"].sum(axis=1)
        ver = verify_split(view, d)
        all_pc, all_macro = per_class_auc(d["y"], d["probs"])
        te = d["is_test"]
        te_pc, te_macro = per_class_auc(d["y"][te], d["probs"][te])
        v = {
            "n_labeled_jets": int(len(d["y"])), "n_events": d["n_events"],
            "labeled_jets_per_event": float(len(d["y"]) / d["n_events"]),
            "class_counts_all": {c: int((d["y"] == i).sum())
                                 for i, c in enumerate(CLASS_LABELS)},
            "score_sum_mean": float(ssum.mean()), "score_sum_std": float(ssum.std()),
            "score_sum_min": float(ssum.min()), "score_sum_max": float(ssum.max()),
            "split_verification_vs_stage4_dump": ver,
            "embedded_auc_all": {"macro": all_macro,
                                 "macro_boot_std": boot_macro(d["y"], d["probs"]),
                                 "per_class": all_pc},
            "embedded_auc_test": {"macro": te_macro, "n_test": int(te.sum()),
                                  "macro_boot_std": boot_macro(d["y"][te], d["probs"][te]),
                                  "per_class": te_pc},
        }
        res["views"][view] = v
        np.savez_compressed(
            os.path.join(CAL, "dumps", f"embedded_{view}.npz"),
            y=d["y"], probs=d["probs"], pt=d["pt"], eta=d["eta"], phi=d["phi"],
            event=d["event"], file_idx=d["file"], is_test=d["is_test"],
            test_order=d["test_order"],
            class_labels=np.asarray(CLASS_LABELS, dtype=object))
        with open(OUT_JSON, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[done] {view}: macro(all)={all_macro:.4f} macro(test)={te_macro:.4f} "
              f"verify={ver}", flush=True)

    # cross-view pairing: embedded@1111 vs embedded@0000 on event-matched jets
    dA, dB = data["1111"], data["0000"]
    iA, iB, n_multi = pair_views(dA, dB)
    same_label = dA["y"][iA] == dB["y"][iB]
    pairing = {
        "n_1111": int(len(dA["y"])), "n_0000": int(len(dB["y"])),
        "n_matched": int(len(iA)),
        "match_rate_1111": float(len(iA) / len(dA["y"])),
        "match_rate_0000": float(len(iA) / len(dB["y"])),
        "n_ambiguous_multimatch": int(n_multi),
        "label_agreement": float(same_label.mean()),
    }
    # paired comparison on matched jets with AGREEING labels (clean pairing)
    sel = same_label
    yA, pA = dA["y"][iA][sel], dA["probs"][iA][sel]
    yB, pB = dB["y"][iB][sel], dB["probs"][iB][sel]
    aA, mA = per_class_auc(yA, pA)
    aB, mB = per_class_auc(yB, pB)
    boot = paired_boot_delta(yA, pA, yB, pB)
    pairing["n_label_agree"] = int(sel.sum())
    pairing["embedded_1111_macro_on_matched"] = mA
    pairing["embedded_0000_macro_on_matched"] = mB
    pairing["macro_delta_1111_minus_0000"] = (mA - mB) if mA and mB else None
    pairing["macro_delta_boot_std"] = boot["macro_delta_std"]
    pairing["per_class_delta"] = {
        c: {"delta": (aA[c]["auc"] - aB[c]["auc"])
            if aA[c]["auc"] is not None and aB[c]["auc"] is not None else None,
            "boot_std": boot["per_class_delta_std"][c]}
        for c in CLASS_LABELS}
    res["pairing_1111_vs_0000"] = pairing
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[pairing] matched {pairing['n_matched']} "
          f"(rate {pairing['match_rate_1111']:.3f}), "
          f"delta={pairing['macro_delta_1111_minus_0000']}", flush=True)


if __name__ == "__main__":
    main()
