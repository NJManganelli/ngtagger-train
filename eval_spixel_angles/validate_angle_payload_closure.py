"""Closure test: the ML residual the payload was fitted from vs the payload as applied.

TWO DISTRIBUTIONS, ONE PLOT, AND THEY MUST AGREE.

  fitted-from : the regression-eval parquet the extractor consumed. Residual in
                SPEC convention, pred - true, after the alpha/beta swap.
  as-applied  : the payload evaluated exactly as SmartPixelsRecHitProducer does
                --  shift = sigma(layer, cotAlpha, cotBeta, bLocalY) * prng
                            + bias, via the compound `spix_angle_*_shift` with
                    prngAcc seeded at 1.0 -- at the same true angles.

Agreement validates the extraction. Disagreement localises it: a width mismatch
is a sigma-fit or binning problem, an offset is the bias sign (the parquet
column is true - pred while the spec wants pred - true), and a mismatch only
outside |cotBeta| ~ 1.07 is the known source-coverage clamp rather than a bug.

The same run also prints the parquet's own residual widths per variant, which is
the number to compare a NEW training against: regenerate its -vars.parquet and
run this with --parquet pointing at it.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The one place the frame swap is allowed to live, reused rather than restated.
from extract_pixelav_angle_payload import SWAP_ALPHA_BETA  # noqa: E402

BLOCALY_CLAMP = -3.81          # the payload's single degenerate bLocalY bin
LAYER = 1                      # layer axis is a duplicated fit; any layer works


def robust(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan"), float("nan")
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def payload_beta_edges(path, name="spix_angle_alpha_sigma"):
    """The non-negative half of the payload's own cotBeta axis, as band edges.

    Comparing the fitted width against the payload inside the payload's own
    bins is the only way a ratio means "this bin is mis-fitted" rather than
    "these two bands happen to share one bin". Read from the JSON, since the
    compiled Correction object does not expose its node tree.
    """
    d = json.loads(Path(path).read_text())
    node = [c for c in d["corrections"] if c["name"] == name][0]["data"]
    while isinstance(node, dict) and node.get("nodetype") == "category":
        node = node["content"][0]["value"]
    if not (isinstance(node, dict) and node.get("nodetype") == "multibinning"):
        return [0.0, 0.3, 1.0, 1e9]
    ax = node["edges"][node["inputs"].index("cotBeta")]
    return sorted({abs(float(v)) for v in ax}) + [1e9]


def load_parquet(paths):
    """Load whatever angle columns a dump actually has.

    THE VARIANTS ARE NOT UNIFORM. Mlp_Slim-2bit carries only cotB (spec alpha,
    the bending angle) over 1.4M rows and no per-cluster NN sigma, while the
    Conv variants carry both angles over 108k rows. The extractor handles this
    with its own pick() helper; asking for a fixed column list here turned a
    single-angle dump into an unreadable file.
    """
    want = ["cotA", "cotB", "cotAtrue", "cotBtrue",
            "residuals_cotA", "residuals_cotB", "sigmacotA", "sigmacotB"]
    frames, have = [], None
    for p in paths:
        names = set(pq.ParquetFile(p).schema_arrow.names)
        got = [c for c in want if c in names]
        if have is None:
            have = got
        elif have != got:
            raise SystemExit(f"{p} has columns {got} but an earlier file had "
                             f"{have}; mixing dumps of different shape would "
                             f"silently concatenate different quantities")
        t = pq.read_table(p, columns=got)
        frames.append({c: np.asarray(t[c], float) for c in got})
    return {c: np.concatenate([f[c] for f in frames]) for c in have}


def to_spec(D):
    """Source columns -> spec convention, and the residual sign flip.

    spec alpha (bending) <- source cotB ; spec beta <- source cotA, per the
    coordinate relation cotAlpha_cms = +cotBeta_pix. The parquet residual is
    (true - pred); the spec residual is (pred - true).
    """
    sa, sb = ("cotB", "cotA") if SWAP_ALPHA_BETA else ("cotA", "cotB")
    out = {}
    for spec, src in (("alpha", sa), ("beta", sb)):
        if f"{src}true" not in D or f"residuals_{src}" not in D:
            continue
        out[f"true_{spec}"] = D[f"{src}true"]
        out[f"resid_{spec}"] = -D[f"residuals_{src}"]
    if "true_alpha" not in out:
        raise SystemExit("this dump has no spec-alpha columns; nothing to test")
    # The payload's sigma/bias are functions of BOTH true angles. A dump that
    # carries only one of them cannot address the other axis, so it is held at
    # zero and the result is valid only along the axis the dump covers -- said
    # here rather than quietly evaluating at an arbitrary point.
    if "true_beta" not in out:
        out["true_beta"] = np.zeros_like(out["true_alpha"])
        out["beta_axis_absent"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True,
                    help="regression-eval -vars.parquet file(s) or globs")
    ap.add_argument("--payload", required=True, help="spix_angle_response_*.json")
    ap.add_argument("-o", "--outdir", default=None)
    a = ap.parse_args()

    paths = sorted({p for g in a.parquet for p in glob.glob(g)} or set())
    if not paths:
        raise SystemExit(f"no parquet matched {a.parquet}")
    D = load_parquet(paths)
    S = to_spec(D)
    n = len(S["true_alpha"])
    print(f"{n:,} clusters from {len(paths)} file(s)")
    print(f"frame swap applied: {SWAP_ALPHA_BETA}  "
          f"(spec alpha <- source cot{'B' if SWAP_ALPHA_BETA else 'A'})")

    # the residual-column sign claim, checked rather than trusted
    sa = "cotB" if SWAP_ALPHA_BETA else "cotA"
    chk = D["residuals_" + sa] - (D[sa + "true"] - D[sa])
    print(f"residual column == (true - pred)?  max |diff| {np.abs(chk).max():.2e}")

    from correctionlib import CorrectionSet
    cs = CorrectionSet.from_file(a.payload)

    lay = np.full(n, LAYER, dtype=np.int32)
    bly = np.full(n, BLOCALY_CLAMP)
    res = {"payload": a.payload, "parquet": paths, "n": int(n),
           "swap_alpha_beta": bool(SWAP_ALPHA_BETA)}

    if S.get("beta_axis_absent"):
        print("NOTE: this dump has no spec-beta angle; the payload's cotBeta "
              "axis is held at 0 and only the alpha result is meaningful")
    print()
    beta_edges = payload_beta_edges(a.payload)
    print(f"{'angle':>6}{'source':>14}{'median':>11}{'width':>11}"
          f"{'payload sigma':>15}{'ratio':>8}")
    for ang in ("alpha", "beta"):
        if f"resid_{ang}" not in S:
            print(f"{ang:>6}{'absent from this dump':>30}")
            continue
        r = S[f"resid_{ang}"]
        med, wid = robust(r)
        sig = cs[f"spix_angle_{ang}_sigma"].evaluate(
            lay, S["true_alpha"], S["true_beta"], bly)
        bias = cs[f"spix_angle_{ang}_bias"].evaluate(
            lay, S["true_alpha"], S["true_beta"], bly)
        # as CMSSW does it: the compound shift, prngAcc seeded at 1.0
        shift = cs.compound[f"spix_angle_{ang}_shift"].evaluate(
            lay, S["true_alpha"], S["true_beta"], bly, np.ones(n))
        amed, awid = robust(shift)
        print(f"{ang:>6}{'ML parquet':>14}{med:11.5f}{wid:11.5f}"
              f"{np.median(sig):15.5f}{wid / max(np.median(sig), 1e-12):8.3f}")
        print(f"{'':>6}{'as applied':>14}{amed:11.5f}{awid:11.5f}"
              f"{'(bias ' + format(np.median(bias), '+.5f') + ')':>15}"
              f"{awid / max(wid, 1e-12):8.3f}")
        res[ang] = {"parquet_median": med, "parquet_width": wid,
                    "payload_sigma_median": float(np.median(sig)),
                    "payload_bias_median": float(np.median(bias)),
                    "applied_median": amed, "applied_width": awid,
                    "width_ratio_applied_over_parquet":
                        awid / max(wid, 1e-12)}

        # Matching cores is not matching distributions. The payload is Gaussian
        # by construction (sigma * normal draw + bias), so it cannot reproduce
        # a non-Gaussian tail however well the width is fitted. Quantify the
        # tail rather than leave it to the eye.
        print(f"{'':>6}{'tail |r| >':>16}{'parquet':>10}{'applied':>10}"
              f"{'gaussian':>10}")
        tail = {}
        for k, g in ((3, 2.700e-3), (5, 5.733e-7), (10, 1.524e-23)):
            cut = k * wid
            fp = float(np.mean(np.abs(r - med) > cut))
            fa = float(np.mean(np.abs(shift - amed) > cut))
            print(f"{'':>6}{str(k) + ' x width':>16}{fp:10.2e}{fa:10.2e}"
                  f"{g:10.2e}")
            tail[f"{k}x"] = {"parquet_frac": fp, "applied_frac": fa,
                             "gaussian_frac": g}
        res[ang]["tails"] = tail

        # Bands taken from the payload's OWN cotBeta edges, so each line is
        # one payload bin. Inventing band edges instead made two lines report
        # the same sigma and look like a payload defect.
        edges = beta_edges
        key = np.abs(S["true_beta"])
        # app/pq, not parquet/sigma: the payload's sigma varies with cotAlpha
        # inside each band, so the median sigma is not the width of the
        # mixture and comparing against it manufactures a disagreement.
        print(f"{'':>6}{'|cotBeta| band':>16}{'n':>8}{'parquet':>10}"
              f"{'med sigma':>10}{'applied':>10}{'app/pq':>8}")
        band = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (key >= lo) & (key < hi)
            if m.sum() < 50:
                continue
            _, w = robust(r[m])
            _, wa = robust(shift[m])
            ps = float(np.median(sig[m]))
            lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
            print(f"{'':>6}{lab:>16}{int(m.sum()):8d}{w:10.5f}"
                  f"{ps:10.5f}{wa:10.5f}{wa / max(w, 1e-12):8.3f}")
            band[lab] = {"n": int(m.sum()), "parquet_width": w,
                         "payload_sigma_median": ps, "applied_width": wa,
                         "ratio_applied_over_parquet": wa / max(w, 1e-12)}
        res[ang]["by_cotbeta"] = band
        print()

    if a.outdir:
        od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
        tag = Path(a.payload).stem.replace("spix_angle_response_", "")
        (od / f"angle_closure_{tag}.json").write_text(json.dumps(res, indent=1))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, ang in zip(axs, ("alpha", "beta")):
                if f"resid_{ang}" not in S:
                    ax.axis("off")
                    ax.text(0.05, 0.5, f"cot{ang} absent from this dump",
                            fontsize=9)
                    continue
                r = S[f"resid_{ang}"]
                shift = cs.compound[f"spix_angle_{ang}_shift"].evaluate(
                    lay, S["true_alpha"], S["true_beta"], bly, np.ones(n))
                lim = float(np.nanpercentile(np.abs(r), 99.5))
                b = np.linspace(-lim, lim, 90)
                # MAD and RMS both, as in the combined figure: the gap between
                # them is the tail the Gaussian payload cannot reproduce.
                ax.hist(r, bins=b, histtype="step", density=True,
                        label=f"ML parquet (fitted from): MAD "
                              f"{robust(r)[1]:.4g}, RMS {np.std(r):.4g}")
                ax.hist(shift, bins=b, histtype="step", density=True,
                        label=f"payload as applied: MAD "
                              f"{robust(shift)[1]:.4g}, RMS {np.std(shift):.4g}")
                ax.set_yscale("log")
                ax.set_xlabel(f"spec cot{ang} residual, pred - true [unitless]")
                ax.set_ylabel("density [1/bin]")
                ax.legend(fontsize=7)
                ax.grid(alpha=.3)
                ax.set_title(f"cot{ang}: extraction closure", fontsize=10)
            fig.suptitle(f"angle payload closure -- {tag}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            p = od / f"angle_closure_{tag}.png"
            fig.savefig(p, dpi=130, bbox_inches="tight")
            print(f"wrote {p}")
        except ImportError:
            pass
        print(f"wrote {od / f'angle_closure_{tag}.json'}")


if __name__ == "__main__":
    main()
