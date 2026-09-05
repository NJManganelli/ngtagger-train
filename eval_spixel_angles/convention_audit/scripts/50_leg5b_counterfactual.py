#!/usr/bin/env python3
"""Leg 5b: counterfactual A/B. Build the WRONG (no-swap) payload variant in scratch
(without editing the extractor), then evaluate BOTH payloads over the real CMSSW-side
angle sample (analyzer ntuple) and quantify the response difference."""
import importlib.util, json, os, sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUD = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/convention_audit"
SCR = os.path.join(AUD, "scratch"); os.makedirs(SCR, exist_ok=True)
ev = json.load(open(os.path.join(AUD, "evidence.json")))

EXTRACTOR = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/extract_pixelav_angle_payload.py"
spec = importlib.util.spec_from_file_location("xp", EXTRACTOR)
xp = importlib.util.module_from_spec(spec); spec.loader.exec_module(xp)

SRC = ["/Users/nmangane/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet"]
cf_path = os.path.join(SCR, "counterfactual_noswap_Conv1D_Full-2bit.json")
if not os.path.exists(cf_path):
    xp.SWAP_ALPHA_BETA = False  # runtime flip; extractor file untouched
    import pyarrow.parquet as pq
    have = set(f.name for f in pq.ParquetFile(SRC[0]).schema_arrow)
    sa, sb = xp.ALPHA_BETA_MAPPING(have, have, have)
    acc_a, acc_b, diag = xp.accumulate_files(SRC, sa, sb)
    cset = xp.build_payload("COUNTERFACTUAL-noswap-Conv1D_Full-2bit", SRC, acc_a, acc_b, diag)
    open(cf_path, "w").write(cset.json(exclude_unset=True))
    xp.SWAP_ALPHA_BETA = True

import correctionlib
DEPLOYED = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/spx_angle_response_Conv1D_Full-2bit_v4fixed.json"
cs_impl = correctionlib.CorrectionSet.from_file(DEPLOYED)
cs_cf = correctionlib.CorrectionSet.from_file(cf_path)

f = uproot.open("/Users/nmangane/smartpixels/cmssw/work/spxsmoke/payload_v4fixed_numEvent300.root")
t = f["smartPixelsPayloadAnalyzer/crossings"]
a = t.arrays(["layer", "trk_cotAlpha", "trk_cotBeta", "b_localy"], library="np")
lay = a["layer"].astype(np.int64); ca = a["trk_cotAlpha"].astype(float)
cb = a["trk_cotBeta"].astype(float); by = a["b_localy"].astype(float)
n = len(ca)

def ev_all(cset):
    out = {}
    for k in ["spx_angle_alpha_sigma", "spx_angle_alpha_bias",
              "spx_angle_beta_sigma", "spx_angle_beta_bias"]:
        c = cset[k]
        out[k] = np.array([c.evaluate(int(l), float(x), float(y), float(b))
                           for l, x, y, b in zip(lay, ca, cb, by)])
    return out

r_impl = ev_all(cs_impl); r_cf = ev_all(cs_cf)

def cmp(k):
    ri, rc = r_impl[k], r_cf[k]
    if "sigma" in k:
        lr = np.abs(np.log(rc / ri))
        return dict(impl_med=float(np.median(ri)), cf_med=float(np.median(rc)),
                    med_abs_log_ratio=float(np.median(lr)),
                    frac_diff_gt20pct=float(np.mean(np.abs(rc / ri - 1) > 0.2)))
    return dict(impl_med=float(np.median(ri)), cf_med=float(np.median(rc)),
                med_abs_diff=float(np.median(np.abs(rc - ri))),
                max_abs_diff=float(np.max(np.abs(rc - ri))))

summary = {k: cmp(k) for k in r_impl}

# clamp fractions on real angle sample (source coverage q01..q99 windows)
clamp = dict(
    alpha_outside_payload_domain=float(np.mean(np.abs(ca) > 0.6)),
    beta_outside_payload_domain=float(np.mean(np.abs(cb) > 6.0)),
    beta_outside_source_coverage=float(np.mean(np.abs(cb) > 1.07)),  # both mappings restricted
    alpha_outside_source_coverage_impl=float(np.mean((ca < -0.99) | (ca > 0.66))),
)

# response curves along alpha (beta fixed 0.15) and along beta (alpha fixed 0.0)
agrid = np.linspace(-0.6, 0.6, 241)
def curve(cset, name, agrid, beta):
    c = cset[name]
    return np.array([c.evaluate(1, float(x), beta, -3.811) for x in agrid])
bgrid = np.linspace(-6, 6, 241)
def curveb(cset, name, bgrid):
    c = cset[name]
    return np.array([c.evaluate(1, 0.05, float(y), -3.811) for y in bgrid])

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(agrid, curve(cs_impl, "spx_angle_alpha_sigma", agrid, 0.15), label="implemented (swap)")
axs[0, 0].plot(agrid, curve(cs_cf, "spx_angle_alpha_sigma", agrid, 0.15), "--", label="counterfactual (no-swap)")
axs[0, 0].set_title("alpha_sigma vs cotAlpha (L1, cotBeta=0.15)"); axs[0, 0].legend()
axs[0, 1].plot(agrid, curve(cs_impl, "spx_angle_alpha_bias", agrid, 0.15), label="implemented")
axs[0, 1].plot(agrid, curve(cs_cf, "spx_angle_alpha_bias", agrid, 0.15), "--", label="counterfactual")
axs[0, 1].set_title("alpha_bias vs cotAlpha"); axs[0, 1].axhline(0, color="k", lw=.5)
axs[1, 0].hist(r_impl["spx_angle_alpha_sigma"], bins=80, histtype="step", label="impl")
axs[1, 0].hist(r_cf["spx_angle_alpha_sigma"], bins=80, histtype="step", label="cf")
axs[1, 0].set_yscale("log"); axs[1, 0].set_title("assigned alpha_sigma over real tracks"); axs[1, 0].legend()
axs[1, 1].plot(bgrid, curveb(cs_impl, "spx_angle_beta_sigma", bgrid), label="impl")
axs[1, 1].plot(bgrid, curveb(cs_cf, "spx_angle_beta_sigma", bgrid), "--", label="cf")
axs[1, 1].set_title("beta_sigma vs cotBeta (alpha=0.05)"); axs[1, 1].legend()
for ax in axs.flat: ax.grid(alpha=.3)
fig.suptitle("Leg 5b: implemented vs counterfactual(no-swap) payload on real CMSSW angles")
fig.tight_layout()
fig.savefig(os.path.join(AUD, "plots", "leg5b_counterfactual_ab.png"), dpi=120)

ev["leg5b_counterfactual"] = dict(n_eval=n, summary=summary, clamp_fractions=clamp,
                                  counterfactual_payload=cf_path)
json.dump(ev, open(os.path.join(AUD, "evidence.json"), "w"), indent=1, default=float)
for k, v in summary.items():
    print(k, {kk: round(vv, 5) for kk, vv in v.items()})
print("clamp:", {k: round(v, 4) for k, v in clamp.items()})
