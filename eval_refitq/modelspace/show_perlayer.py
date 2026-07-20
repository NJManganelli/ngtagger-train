import json
import sys

cfg = sys.argv[1] if len(sys.argv) > 1 else "AAAA"
d = json.load(open(f"eval_refitq/modelspace/perlayer_results_{cfg}.json"))
print("gain top25 (spec17+fullL, seed0):")
for k, v in d["xgb_gain_top25_fullL_seed0"].items():
    print(f"  {k:24s} {v:9.1f}")
print("logistic pull2 layer weights:")
for k, v in d["logistic_layer_weights_pull2"].items():
    print(f"  {k:16s} {v:+.3f}")
print("MI |pull|:")
for k, v in d["mutual_info_abs_pull"].items():
    print(f"  {k:16s} {v:.4f}")
