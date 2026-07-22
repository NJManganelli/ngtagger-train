"""Empirical probes of correctionlib 2.9.0 HashPRNG semantics. Writes findings json."""
import json, numpy as np
import correctionlib, correctionlib.schemav2 as cs

F = {}
def prng(name, inputs, dist="stdnormal"):
    return cs.Correction(name=name, version=1, inputs=[
        cs.Variable(name=n, type=t) for n, t in inputs],
        output=cs.Variable(name="rng", type="real"),
        data=cs.HashPRNG(nodetype="hashprng", inputs=[n for n, _ in inputs], distribution=dist))

# 1. int input as entropy?
try:
    c = prng("p_int", [("layer", "int"), ("x", "real")])
    cset = cs.CorrectionSet(schema_version=2, corrections=[c])
    ev = correctionlib.CorrectionSet.from_string(cset.json())
    v1 = ev["p_int"].evaluate(1, 0.5); v2 = ev["p_int"].evaluate(2, 0.5)
    F["int_entropy"] = {"works": True, "layer1": v1, "layer2": v2, "distinct": v1 != v2}
except Exception as e:
    F["int_entropy"] = {"works": False, "error": str(e)}

# 2. two identical prng nodes (different correction names) -> same value?
ca = prng("pa", [("x", "real"), ("y", "real")])
cb = prng("pb", [("x", "real"), ("y", "real")])
crev = prng("prev", [("y", "real"), ("x", "real")])  # reversed declaration order
cflat = prng("pflat", [("x", "real"), ("y", "real")], dist="stdflat")
ev = correctionlib.CorrectionSet.from_string(cs.CorrectionSet(schema_version=2, corrections=[ca, cb, crev, cflat]).json())
va = ev["pa"].evaluate(0.3, 1.7); vb = ev["pb"].evaluate(0.3, 1.7)
vrev = ev["prev"].evaluate(1.7, 0.3)  # same by-name values, reversed positional order
F["same_inputs_same_value"] = {"va": va, "vb": vb, "identical": va == vb}
F["order_sensitivity"] = {"v_fwd": va, "v_rev": vrev, "differs": va != vrev}
# flat vs normal from same entropy: comonotone? sample correlation over grid
xs = np.linspace(-0.5, 0.5, 2001)
zn = np.array([ev["pa"].evaluate(float(x), 1.7) for x in xs])
zf = np.array([ev["pflat"].evaluate(float(x), 1.7) for x in xs])
from scipy.stats import norm, spearmanr
F["flat_vs_normal_same_entropy"] = {"spearman": float(spearmanr(zf, zn).statistic),
                                    "max_abs_diff_phi": float(np.max(np.abs(norm.cdf(zn) - zf)))}
# distribution sanity for stdnormal over dense entropy grid
F["stdnormal_stats"] = {"mean": float(zn.mean()), "std": float(zn.std()), "n": len(zn)}

# 3. compound: [sigma(layer,x,y), prng(x,y)] output_op "*" — by-name routing, int in compound
sigma = cs.Correction(name="sig", version=1, inputs=[
    cs.Variable(name="layer", type="int"), cs.Variable(name="x", type="real"), cs.Variable(name="y", type="real")],
    output=cs.Variable(name="sig", type="real"),
    data=cs.Category(nodetype="category", input="layer", content=[
        cs.CategoryItem(key=1, value=2.0), cs.CategoryItem(key=2, value=3.0)]))
try:
    comp = cs.CompoundCorrection(name="smear", inputs=[
        cs.Variable(name="layer", type="int"), cs.Variable(name="x", type="real"), cs.Variable(name="y", type="real")],
        output=cs.Variable(name="smear", type="real"), inputs_update=[], input_op="*", output_op="*",
        stack=["sig", "pa"])
    full = cs.CorrectionSet(schema_version=2, corrections=[sigma, ca], compound_corrections=[comp])
    ev2 = correctionlib.CorrectionSet.from_string(full.json())
    vs = ev2.compound["smear"].evaluate(1, 0.3, 1.7)
    vprng = ev2["pa"].evaluate(0.3, 1.7)
    F["compound"] = {"works": True, "value": vs, "sigma_times_prng": 2.0 * vprng,
                     "matches": abs(vs - 2.0 * vprng) < 1e-12}
except Exception as e:
    F["compound"] = {"works": False, "error": str(e)}

# 4. does prng WITH layer int included work inside compound?
try:
    cint = prng("pint2", [("layer", "int"), ("x", "real"), ("y", "real")])
    comp2 = cs.CompoundCorrection(name="smear2", inputs=[
        cs.Variable(name="layer", type="int"), cs.Variable(name="x", type="real"), cs.Variable(name="y", type="real")],
        output=cs.Variable(name="smear2", type="real"), inputs_update=[], input_op="*", output_op="*",
        stack=["sig", "pint2"])
    full2 = cs.CorrectionSet(schema_version=2, corrections=[sigma, cint], compound_corrections=[comp2])
    ev3 = correctionlib.CorrectionSet.from_string(full2.json())
    v1 = ev3.compound["smear2"].evaluate(1, 0.3, 1.7); v2 = ev3.compound["smear2"].evaluate(2, 0.3, 1.7)
    F["compound_int_entropy"] = {"works": True, "layer1": v1, "layer2": v2,
                                 "layer_changes_throw": abs(v1 / 2.0) != abs(v2 / 3.0)}
except Exception as e:
    F["compound_int_entropy"] = {"works": False, "error": str(e)}

out = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/hashprng_ab/probe_findings.json"
json.dump(F, open(out, "w"), indent=1)
print(json.dumps(F, indent=1))
