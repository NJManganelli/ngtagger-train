"""Read stage4_summary.json and print the Stage-4 report tables:
per-view x per-feature macro/per-flavor AUC + seed spread, marginal
contributions of refit-BDT and vertex-dxy, refit-views vs 0000 baseline,
and a feature-ablation ranking."""
import json
import os

OUT = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/stage4"
SUMMARY = os.path.join(OUT, "stage4_summary.json")
FLAV = ["b", "charm", "light", "gluon", "taup", "taum", "muon", "electron"]

with open(SUMMARY) as f:
    S = json.load(f)
cells = S["cells"]


def g(key, field, default=None):
    return cells.get(key, {}).get(field, default)


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "  -  "


FEATS = [("baseline", "baseline"), ("refitbdt", "+refitBDT"),
         ("vertexdxy", "+vtxDxy"), ("both", "+both")]
VIEWS = ["1111", "1100", "0000"]

print("=" * 78)
print("STAGE-4 MATRIX: macro AUC (best-of-seed) [mean +/- std over seeds] | acc")
print("=" * 78)
hdr = f"{'view':6}"
for _, lab in FEATS:
    hdr += f"{lab:>26}"
print(hdr)
for v in VIEWS:
    row = f"{v:6}"
    for fk, _ in FEATS:
        key = f"{v}__{fk}"
        if key not in cells:
            row += f"{'N/A':>26}"
            continue
        b = g(key, "best_macro_auc")
        mean = g(key, "macro_auc_mean")
        std = g(key, "macro_auc_std")
        acc = g(key, "best_acc")
        cell = f"{fmt(b)} [{fmt(mean)}+/-{fmt(std,3)}]|{fmt(acc,2)}"
        row += f"{cell:>26}"
    print(row)

print("\n" + "=" * 78)
print("PER-FLAVOR AUC (best seed) — b / charm / light / gluon are the physics targets")
print("=" * 78)
for v in VIEWS:
    print(f"\n--- view {v} (activeSP {g(f'{v}__baseline','activeSP')}) ---")
    print(f"  {'featureset':14}" + "".join(f"{fl:>9}" for fl in FLAV))
    for fk, lab in FEATS:
        key = f"{v}__{fk}"
        if key not in cells:
            continue
        pf = g(key, "best_per_flavor_auc", {})
        print(f"  {lab:14}" + "".join(f"{fmt(pf.get(fl)):>9}" for fl in FLAV))

print("\n" + "=" * 78)
print("MARGINAL CONTRIBUTIONS (delta best macro AUC vs that view's baseline)")
print("=" * 78)
for v in VIEWS:
    base = g(f"{v}__baseline", "best_macro_auc")
    if base is None:
        continue
    print(f"\nview {v}: baseline macro AUC = {fmt(base)}")
    for fk, lab in [("refitbdt", "+refitBDT"), ("vertexdxy", "+vtxDxy"), ("both", "+both")]:
        key = f"{v}__{fk}"
        if key not in cells:
            continue
        val = g(key, "best_macro_auc")
        if val is not None:
            d = val - base
            print(f"   {lab:12}: {fmt(val)}  (delta {d:+.4f})")
    # per-flavor b-tag delta specifically
    pb = g(f"{v}__baseline", "best_per_flavor_auc", {}).get("b")
    for fk, lab in [("refitbdt", "+refitBDT"), ("vertexdxy", "+vtxDxy"), ("both", "+both")]:
        key = f"{v}__{fk}"
        pv = g(key, "best_per_flavor_auc", {}).get("b") if key in cells else None
        if pb is not None and pv is not None:
            print(f"      b-AUC {lab:10}: {fmt(pv)} (delta {pv-pb:+.4f}) vs base b-AUC {fmt(pb)}")

print("\n" + "=" * 78)
print("REFIT-VIEWS vs OT-ONLY BASELINE (does IT refit downstream help the tagger)")
print("=" * 78)
base0 = g("0000__baseline", "best_macro_auc")
print(f"0000 (OT-only) baseline macro AUC: {fmt(base0)}")
for v in ["1111", "1100"]:
    b = g(f"{v}__baseline", "best_macro_auc")
    if b is not None and base0 is not None:
        print(f"  {v} baseline macro AUC: {fmt(b)} (delta vs 0000 {b-base0:+.4f})")
    both = g(f"{v}__both", "best_macro_auc")
    if both is not None and base0 is not None:
        print(f"  {v} +both      macro AUC: {fmt(both)} (delta vs 0000 baseline {both-base0:+.4f})")

print("\n" + "=" * 78)
print("FEATURE-SET RANKING (all cells by best macro AUC)")
print("=" * 78)
ranked = sorted([(k, v.get("best_macro_auc") or 0) for k, v in cells.items()],
                key=lambda kv: -kv[1])
for k, a in ranked:
    print(f"  {fmt(a)}  {k}   (n_feat={g(k,'n_features')}, n_test={g(k,'n_test')})")

# arch/head confirmation
print("\n" + "=" * 78)
print("ARCH/HEAD RUN CONFIRMATION")
print("=" * 78)
ch = "1111__both__chargehead"
if ch in cells:
    print(f"  charge-head model RAN: {ch} macro AUC {fmt(g(ch,'best_macro_auc'))} "
          f"acc {fmt(g(ch,'best_acc'),2)} (charge_head={g(ch,'charge_head')})")
else:
    print("  charge-head cell not present yet")
