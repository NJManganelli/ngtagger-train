"""Fast-gate items on the shared definitions (ngtagger.definitions)."""
from __future__ import annotations

import numpy as np

from .core import Item, identity, same_input

UNIT_SCALES = (1e-4, 1.0, 1e4)      # cm -> um is x 1e4
SCALE_TOL = 1e-9


class EstimatorUnits(Item):
    id = "estimator.units"
    title = "the B.1 core gives the same physical width in any unit (no magnitude cuts inside the estimator)"
    c4_row = "(new, 2026-10-09) unit-dependent sentinel cut in the estimator, found by the P0.2 audit"
    known_bad_desc = ("the census pull table's original core (drops |x| >= 1e3, absolute 1e-5 tolerance), "
                      "applied to the same residuals in cm, in um and in 1e-4 cm")
    rule_desc = ("on the OT track z0 and d0 residuals to their TP (L1TTrack, tpIdx >= 0 as the validity mask) of "
                 "the gate input: core(k x) / k equal for k = 1e-4, 1, 1e4 to 1e-9 relative; identity of the cm "
                 "widths with the reference")

    def _stat(self, ctx, estimator_name="core"):
        import awkward as ak

        from ngtagger.definitions import estimators as E

        est = E.core if estimator_name == "core" else E.core_pull_table
        A = ctx.arrays(["L1TTrack_tpIdx", "L1TTrack_z0", "L1TTrack_tp_z0", "L1TTrack_d0", "L1TTrack_tp_d0"])
        valid = ak.to_numpy(ak.flatten(A["L1TTrack_tpIdx"])) >= 0
        out = {"input": ctx.input, "events": ctx.events, "estimator": estimator_name, "n_valid": int(valid.sum())}
        for p in ("z0", "d0"):
            x = (ak.to_numpy(ak.flatten(A[f"L1TTrack_{p}"])) - ak.to_numpy(ak.flatten(A[f"L1TTrack_tp_{p}"])))
            x = np.asarray(x, np.float64)[valid]
            s = E.scale_invariance(x, scales=UNIT_SCALES, estimator=est)
            out[p] = {"rms0_cm_by_scale": s["rms0_over_k"], "rel_spread": s["rel_spread"]}
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, "core_pull_table")

    def judge(self, stat, ref):
        why = [f"{p}: widths in cm from scales {UNIT_SCALES}: {stat[p]['rms0_cm_by_scale']} (spread "
               f"{stat[p]['rel_spread']:.3g})" for p in ("z0", "d0") if not stat[p]["rel_spread"] <= SCALE_TOL]
        if same_input(stat, ref) and stat["estimator"] == ref["stat"]["estimator"]:
            why += identity(stat, ref, ["n_valid", "z0", "d0"])[1]
        return not why, (f"z0 core {stat['z0']['rms0_cm_by_scale'][1] * 1e4:.0f} um, d0 core "
                         f"{stat['d0']['rms0_cm_by_scale'][1] * 1e4:.0f} um in every unit") if not why else "; ".join(why)


ITEMS = [EstimatorUnits()]
