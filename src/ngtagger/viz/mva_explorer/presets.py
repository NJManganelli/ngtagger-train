"""First-class regression presets exported by default (Panel 1)."""

from __future__ import annotations

import os

from ngtagger.viz.mva_explorer.correctionlib_ingest import (
    export_generic_dataset,
    export_structured_smear,
)

SMARTPIXELS = "/Users/nmangane/smartpixels"

TKLAYOUT_JSON = os.path.join(
    SMARTPIXELS, "tkLayoutRedux/tkLayout/spixel_smear_tklayout_trigger_MS.json.gz")
CALV1_JSON = os.path.join(
    SMARTPIXELS, "cmssw/spixel_smear_all_configs_barrel_CalV1_v2p1_compound.json")
DIGIREFIT_JSONS = {
    "smarthit_true": os.path.join(SMARTPIXELS, "cmssw/work/spxsmoke/smarthit_true_v4fixed.json"),
    "smarthit_fake": os.path.join(SMARTPIXELS, "cmssw/work/spxsmoke/smarthit_fake_v4fixed.json"),
    "spx_angle": os.path.join(
        SMARTPIXELS, "cmssw/work/spxsmoke/spx_angle_response_Conv1D_Full-2bit_v4fixed.json"),
}

# pretty axis labels for the known preset inputs (transform-aware: the payloads
# fold eta/d0 to absolute values before binning)
AXIS_LABELS = {
    "pt_tp": "TP pT [GeV]", "eta_tp": "TP |eta|",
    "z0_tp": "TP z0 [cm]", "d0_tp": "TP |d0| [cm]",
    "abs_eta": "|eta|", "layer": "TBPX layer",
    "cotAlpha": "cot(alpha)", "cotBeta": "cot(beta)",
    "bLocalY": "local-Y edge flag",
}


def _label_axes(meta: dict) -> dict:
    def lab(ax):
        ax["label"] = AXIS_LABELS.get(ax["name"], ax["name"])
        return ax
    if meta["type"] == "structured":
        meta["axes"] = [lab(a) for a in meta["axes"]]
    else:
        for c in meta["corrections"]:
            c["axes"] = [lab(a) for a in c["axes"]]
    return meta


def export_tklayout(out_dir: str, json_path: str = TKLAYOUT_JSON) -> dict:
    meta = export_structured_smear(
        json_path, "reg_tklayout", "tkLayout trigger-window MS smearing", out_dir)
    return _rewrite(_label_axes(meta), out_dir)


def export_calv1(out_dir: str, json_path: str = CALV1_JSON) -> dict:
    meta = export_structured_smear(
        json_path, "reg_calv1", "CalV1 barrel compound smearing (v2p1)", out_dir)
    return _rewrite(_label_axes(meta), out_dir)


def export_digirefit(out_dir: str, jsons: dict | None = None) -> list[dict]:
    metas = []
    titles = {
        "smarthit_true": "digiRefit v4fixed: smarthit true-hit payload",
        "smarthit_fake": "digiRefit v4fixed: smarthit fake-hit payload",
        "spx_angle": "digiRefit v4fixed: angle response (Conv1D Full-2bit)",
    }
    for key, path in (jsons or DIGIREFIT_JSONS).items():
        meta = export_generic_dataset(
            path, f"reg_{key}", titles.get(key, key), out_dir)
        metas.append(_rewrite(_label_axes(meta), out_dir))
    return metas


def _rewrite(meta: dict, out_dir: str) -> dict:
    import json as _json

    with open(os.path.join(out_dir, f"{meta['id']}_meta.json"), "w") as f:
        _json.dump(meta, f)
    return meta


def export_all_regressions(out_dir: str) -> list[dict]:
    metas = [export_tklayout(out_dir), export_calv1(out_dir)]
    metas += export_digirefit(out_dir)
    return metas
