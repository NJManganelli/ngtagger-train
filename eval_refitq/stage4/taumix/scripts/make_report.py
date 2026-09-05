"""Render taumix/REPORT.md from taumix_summary.json (+ stage-4 anchors)."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAUMIX = os.path.dirname(HERE)
STAGE4 = os.path.dirname(TAUMIX)

with open(os.path.join(TAUMIX, "taumix_summary.json")) as f:
    S = json.load(f)
with open(os.path.join(STAGE4, "stage4_summary.json")) as f:
    S4 = json.load(f)
CLASSES = S["meta"]["class_labels"]
VIEWS = ["1111", "0000"]


def fmt(v, pm=None, nd=3):
    if v is None:
        return "—"
    s = f"{v:.{nd}f}"
    if pm is not None:
        s += f"±{pm:.{nd}f}"
    return s


def cell(key, setname):
    return S["cells"][key]["agg"][setname]


def tau_row(label, a, se_key="se_hm"):
    tp, tm = a["taup"], a["taum"]
    return (f"| {label} | {fmt(tp['mean'], tp['std'])} (HM ±{tp[se_key]:.3f}, "
            f"n={tp['n_pos']}) | {fmt(tm['mean'], tm['std'])} (HM ±{tm[se_key]:.3f}, "
            f"n={tm['n_pos']}) | {fmt(a['macro_mean'], a['macro_std'])} |")


def emb_tau_row(label, e):
    tp = e["per_class"]["taup"]
    tm = e["per_class"]["taum"]
    return (f"| {label} | {fmt(tp['auc'])} (HM ±{tp['se_hm']:.3f}, n={tp['n_pos']}) "
            f"| {fmt(tm['auc'])} (HM ±{tm['se_hm']:.3f}, n={tm['n_pos']}) "
            f"| {fmt(e['macro'])} |")


L = []
L.append("# Tau-mixing study (taumix): QQToHToTauTau unified nanos in the stage-4 jet-tagger training\n")
L.append("Two 50-event QQToHToTauTau RelVal files produced as unified coherent nanos "
         "(1111_coopt + 0000_baseline per file) and mixed into the stage-4 "
         "training (10 ttbar files) per view. All metrics: one-vs-rest AUC, "
         "3 seeds (mean±seed std), Hanley–McNeil SE (HM) for per-class AUCs.\n")

L.append("## Provenance / production notes\n")
L.append("- " + S["meta"]["provenance_note"] + "\n")
L.append("- Split: " + S["meta"]["split_note"] + "\n")

L.append("## Jet inventory (htt files, per view)\n")
L.append("| view | htt labeled jets | " + " | ".join(CLASSES) + " | taus/event |")
L.append("|---|---|" + "---|" * len(CLASSES) + "---|")
for v in VIEWS:
    inv = S["inventory"][v]
    cc = inv["htt_class_counts"]
    L.append(f"| {v} | {inv['htt_labeled_jets']} (from {inv['n_htt_events']} events) | "
             + " | ".join(str(cc[c]) for c in CLASSES)
             + f" | {inv['htt_taus_per_event']:.2f} |")
L.append("")

L.append("## Split (per-class train/test counts, mixed dataset)\n")
for v in VIEWS:
    sp = S["split"][v]
    L.append(f"**view {v}** — original-split verification vs stage-4 dump: "
             f"`{sp['orig_split_verification']}`\n")
    L.append("| | " + " | ".join(CLASSES) + " |")
    L.append("|---|" + "---|" * len(CLASSES))
    for row, dd in (("train", sp["per_class_train"]), ("test", sp["per_class_test"]),
                    ("test (htt part)", sp["per_class_test_htt_part"])):
        L.append(f"| {row} | " + " | ".join(str(dd[c]) for c in CLASSES) + " |")
    L.append("")

L.append("## Tau AUCs on the mixed stratified test set\n")
for v in VIEWS:
    L.append(f"**view {v}**\n")
    L.append("| model | taup AUC | taum AUC | macro AUC |")
    L.append("|---|---|---|---|")
    L.append(tau_row("ttbar-only (ctrl, same test set)", cell(f"{v}__baseline_ttonly_ctrl", "mixed_test")))
    L.append(tau_row("tau-mixed", cell(f"{v}__baseline_taumix", "mixed_test")))
    if f"{v}__refitbdt_taumix" in S["cells"]:
        L.append(tau_row("tau-mixed +refitbdt", cell(f"{v}__refitbdt_taumix", "mixed_test")))
    L.append(emb_tau_row("embedded NG tagger", S["embedded"][v]["mixed_test"]))
    a4 = S4["cells"][f"{v}__baseline"]
    L.append(f"| stage-4 anchor (ttbar-only train+test) | "
             f"{a4['best_per_flavor_auc']['taup']:.3f} | "
             f"{a4['best_per_flavor_auc']['taum']:.3f} | "
             f"{a4['best_macro_auc']:.3f} |")
    L.append("")

L.append("## Continuity: original stage-4 ttbar-only test set (no leakage — "
         "frozen out of mixed training)\n")
for v in VIEWS:
    L.append(f"**view {v}**\n")
    L.append("| model | taup AUC | taum AUC | macro AUC |")
    L.append("|---|---|---|---|")
    L.append(tau_row("ttbar-only (ctrl)", cell(f"{v}__baseline_ttonly_ctrl", "orig_ttbar_test")))
    L.append(tau_row("tau-mixed", cell(f"{v}__baseline_taumix", "orig_ttbar_test")))
    L.append(emb_tau_row("embedded NG tagger", S["embedded"][v]["orig_ttbar_test"]))
    L.append("")

L.append("## Class/pt weighting check (trainer `class_pt_weights`, onlyclass)\n")
for v in VIEWS:
    w = S["class_weights"][v]
    L.append(f"**view {v}** (weight = b-count / class-count on the train set)\n")
    L.append("| train set | " + " | ".join(CLASSES) + " |")
    L.append("|---|" + "---|" * len(CLASSES))
    for row in ("ttbar_only_train", "mixed_train"):
        L.append(f"| {row} | " + " | ".join(
            fmt(w[row].get(c), nd=2) for c in CLASSES) + " |")
    L.append("")

L.append("## Verdicts\n")


def tau_avg(a):
    return 0.5 * (a["taup"]["mean"] + a["taum"]["mean"])


mx11 = cell("1111__baseline_taumix", "mixed_test")
ct11 = cell("1111__baseline_ttonly_ctrl", "mixed_test")
mx00 = cell("0000__baseline_taumix", "mixed_test")
ct00 = cell("0000__baseline_ttonly_ctrl", "mixed_test")
ot00m = cell("0000__baseline_taumix", "orig_ttbar_test")
ot00c = cell("0000__baseline_ttonly_ctrl", "orig_ttbar_test")
L.append(
    "1. **Does tau mixing improve tau performance?** View-dependent. At **0000** "
    f"yes for taum: {ct00['taum']['mean']:.3f}±{ct00['taum']['std']:.3f} → "
    f"{mx00['taum']['mean']:.3f}±{mx00['taum']['std']:.3f} on the same mixed test set "
    f"(Δ=+{mx00['taum']['mean']-ct00['taum']['mean']:.3f}, ~2.4σ by seed spread; the gain "
    f"transfers to the original ttbar test set: {ot00c['taum']['mean']:.3f} → "
    f"{ot00m['taum']['mean']:.3f}); 0000 macro +{mx00['macro_mean']-ct00['macro_mean']:.3f}. "
    f"At **1111** no resolvable change (taup {ct11['taup']['mean']:.3f} → "
    f"{mx11['taup']['mean']:.3f}, taum {ct11['taum']['mean']:.3f} → "
    f"{mx11['taum']['mean']:.3f}; all Δ within seed spread). The 1111 view already "
    "learned taus adequately from ttbar; 0000 was tau-training-starved.\n")
L.append(
    "2. **Does the refit tau gain (1111 vs 0000) sharpen with real tau stats?** No — "
    "it SHRINKS. Stage-4 anchors suggested tau refit gains of +0.17/+0.16 "
    "(taup/taum). With tau-mixed training and ~2x tau test stats: taup "
    f"+{mx11['taup']['mean']-mx00['taup']['mean']:.3f} (persists), taum "
    f"{mx11['taum']['mean']-mx00['taum']['mean']:+.3f} (gone). Mean tau AUC difference "
    f"{tau_avg(mx11)-tau_avg(mx00):+.3f}, not resolvable at these SEs (~0.065/class). "
    "A large part of the stage-4 'refit tau gain' was 0000's trainability deficit at "
    "~50 training taus, not refit information per se.\n")
if "1111__refitbdt_taumix" in S["cells"]:
    rb = cell("1111__refitbdt_taumix", "mixed_test")
    L.append(
        "3. **Does the refit-BDT tau hint survive?** Not at this scale: 1111 "
        f"+refitbdt taup {rb['taup']['mean']:.3f}±{rb['taup']['std']:.3f}, taum "
        f"{rb['taum']['mean']:.3f}±{rb['taum']['std']:.3f} vs baseline taumix "
        f"{mx11['taup']['mean']:.3f}/{mx11['taum']['mean']:.3f} — no lift beyond "
        "seed spread.\n")
e11 = S["embedded"]["1111"]["mixed_test"]
e00 = S["embedded"]["0000"]["mixed_test"]
L.append(
    "4. **Does scratch-with-tau-mixing close the tau gap to the embedded model?** "
    f"Partially. 0000 taum closes fully ({mx00['taum']['mean']:.3f} vs embedded "
    f"{e00['per_class']['taum']['auc']:.3f}); 0000 taup does not "
    f"({mx00['taup']['mean']:.3f} vs {e00['per_class']['taup']['auc']:.3f}, gap ~0.2). "
    f"1111 taus remain ~0.04–0.05 below embedded ({mx11['taup']['mean']:.3f}/"
    f"{mx11['taum']['mean']:.3f} vs {e11['per_class']['taup']['auc']:.3f}/"
    f"{e11['per_class']['taum']['auc']:.3f}), within ~1 HM SE. Macro gap to embedded "
    f"is ~0.02 ({mx11['macro_mean']:.3f}/{mx00['macro_mean']:.3f} vs "
    f"{e11['macro']:.3f}/{e00['macro']:.3f}).\n")

L.append("## Honest accounting / caveats\n")
inv1 = S["inventory"]["1111"]
L.append(
    f"- Tau yield vs expectation: {inv1['htt_tau_jets']} labeled tau jets / "
    f"{inv1['n_htt_events']} events (0.96/evt) vs ~1.3/evt naive expectation; "
    "GenVisTau rate is 1.19/evt, so ~81% of visible taus yield a labeled jet — "
    "gen-matching is healthy; the shortfall is jet-pt/acceptance, not labeling.\n"
    "- Tau statistics roughly DOUBLE (test taus 21→41 at 1111, 25→44 at 0000; "
    "per-class HM SE ±0.08–0.09 → ±0.065). Resolvable at this level: tau effects "
    "of Δ≳0.1 (direction + machinery validation); paired same-test-set deltas down "
    "to ~0.05 via seed spread. NOT resolvable: 0.01-level tau claims — those need "
    "the ≥5k-tau campaign.\n"
    "- Class weighting (`class_pt_weights`, onlyclass): tau weights fall ~22–27 → "
    "~14–16 with mixing; the VBF-like light influx moves the light weight 0.93→0.87 "
    "and gluon 1.20→1.12 (<8% shift for non-tau classes) — behavior sensible, no "
    "pathologies. Note the stage-4 matrix protocol (replicated here) trains "
    "UNWEIGHTED; the weights are what `run_training` would apply.\n"
    "- htt-only test subset is much harder for every model incl. embedded "
    "(macros 0.56–0.68): different topology (VBF light jets, tau-rich, b-poor) and "
    "tiny (77 jets) — treat per-subset numbers as indicative only.\n"
    "- Cross-provenance training set (see provenance note above): old-code ttbar "
    "nanos + new-HashPRNG htt nanos, defensible per A/B KS tests but recorded.\n"
    "- ttbar block of the split keeps the original stage-4 test membership rather "
    "than a fully re-stratified split — this is what makes the continuity re-score "
    "leakage-free; htt block is stratified-by-class (20%).\n")

L.append("## Plots\n")
L.append("- `plots/per_class_auc.png` — per-class AUC comparison\n"
         "- `plots/tau_roc_overlay.png` — tau ROC overlay (mixed vs ttbar-only vs embedded)\n")

with open(os.path.join(TAUMIX, "REPORT.md"), "w") as f:
    f.write("\n".join(L))
print("REPORT.md written")
