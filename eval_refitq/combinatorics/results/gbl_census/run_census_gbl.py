"""Per-track GBL / forward-only features for every census track row.

Census PINNED (never globbed): cache-tp-incl/tpcensus_038cb9d6de3d851e -- the
PRE-menu-update seed menu. Shards results/gbl_census/chunk_NNNNN.npz hold, per
seed k, feat_k (ntrack, n_features) aligned row for row with the census trk_k.

  --verify   chunk 0 only: reproduce the census fit from the stored hits, check
             forward_filter against kf_run, gbl_fit (kinks off, HO off) against
             kf_run(ho=False), and a toy closure of the kink model.
"""
import sys, os, json, time, argparse, resource
import numpy as np
sys.path.insert(0, "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics")
import census_gbl as G           # noqa: E402
import kf_emulation as KF        # noqa: E402
import tp_findability as TF      # noqa: E402
import tracklet_topology_cost as M   # noqa: E402

CENSUS = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/cache-tp-incl/tpcensus_038cb9d6de3d851e"
OUT = "/Users/nmangane/smartpixels/ngtagger-train/eval_refitq/combinatorics/results/gbl_census"
meta = json.load(open(os.path.join(CENSUS, "manifest.json")))
SEEDS = [tuple(s) for s in meta["seeds"]]
LPOS = {int(L): i for i, L in enumerate(KF.LAYER_ORDER)}
D0P = meta["config"]["kf_opts"].get("d0_prior_cm", KF.D0_PRIOR_CM)
SIGZ = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
TCOL = {c: j for j, c in enumerate(KF.TRACK_COLS)}
INPUTS = [i["path"] for i in meta["inputs"]]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS: bytes


def seed_trip(k, gidx):
    la, lb = SEEDS[k][0], SEEDS[k][1]
    ga, gb = gidx[:, LPOS[la]], gidx[:, LPOS[lb]]
    return ga, gb, gb


def process(U, Q, shard, verify=False):
    rmed = G._layer_radius(U)
    out, checks = {}, {}
    for key in shard.files:
        if not key.startswith("hit_"):
            continue
        k = int(key[4:])
        gidx = shard[key].astype(np.int64)
        if not len(gidx):
            continue
        trip = seed_trip(k, gidx)
        inp = G.census_inputs(U, Q, trip, gidx, D0P)
        fwd = G.forward_filter(inp)
        gbl = G.gbl_fit(inp, fwd["x"], rmed, kinks=True)
        pa, pb, ua = G.angle_pulls_local(inp, gbl["x"], gbl)
        f = G.features(inp, fwd, gbl, pa, pb, ua)
        out[f"feat_{k}"] = np.stack([f[n] for n in G.FEATURE_NAMES], 1).astype(np.float32)
        if verify:
            trk = shard[f"trk_{k}"]
            ff = KF.fit_tracks(U, Q, trip, gidx=gidx, use_angles=False, d0_prior_cm=D0P)
            nh = np.maximum(ff["nhit"], 1)
            rec = {"chi2_rphi_per_layer": ff["chi2_rphi"] / nh, "chi2_rz_per_layer": ff["chi2_rz"] / nh,
                   "d0": ff["d0"], "z0": ff["z0"], "inv_pt": np.abs(ff["kappa"])}
            for c, v in rec.items():
                d = np.abs(v.astype(np.float32) - trk[:, TCOL[c]])
                checks.setdefault(f"census row {c}: max |recomputed - stored|", []).append(float(np.nanmax(d)))
            checks.setdefault("forward vs kf_run chi2_rphi max |d|", []).append(float(np.max(np.abs(fwd["chi2_rphi"] - ff["chi2_rphi"]))))
            checks.setdefault("forward vs kf_run chi2_rz max |d|", []).append(float(np.max(np.abs(fwd["chi2_rz"] - ff["chi2_rz"]))))
            checks.setdefault("forward vs kf_run d0 max |d| (cm)", []).append(float(np.max(np.abs(fwd["x"][:, 2] - ff["d0"]))))
            # equivalence: HO off in both, kinks off, per-hit MS variance kept
            ref = KF.kf_run(inp["R"], inp["m0"], inp["m1"], inp["v0"], inp["v1"], inp["valid"],
                            d0_prior_cm=D0P, ho=False)
            g0 = G.gbl_fit(inp, np.zeros((len(gidx), 5)), rmed, kinks=False, ho=False, n_iter=1)
            kref = np.stack([-M.C_BEND * ref[0], ref[1], ref[2], ref[3], ref[4]], 1)
            sd = np.sqrt(np.stack([ref[5] * M.C_BEND ** 2, ref[15], ref[6], ref[7], ref[8]], 1))
            ok = ref[10]
            checks.setdefault("GBL(no kinks,HO off) vs kf_run(HO off): max |d|/sigma", []).append(
                float(np.max(np.abs(g0["x"][ok] - kref[ok]) / sd[ok])) if ok.any() else 0.0)
    return out, checks


def toy(U, Q, shard, rng):
    """Hits generated from the GBL model itself on real census geometry."""
    rmed = G._layer_radius(U)
    k = max((int(x[4:]) for x in shard.files if x.startswith("hit_")), key=lambda k: len(shard[f"hit_{k}"]))
    gidx = shard[f"hit_{k}"].astype(np.int64)
    inp = G.census_inputs(U, Q, seed_trip(k, gidx), gidx, D0P)
    fwd = G.forward_filter(inp)
    x_true = fwd["x"].copy()
    g = G.gbl_fit(inp, x_true, rmed, kinks=True, ho=False, n_iter=1)   # for the kink geometry
    nt = len(gidx)
    r = np.maximum(inp["R"], 1e-3)
    sec2 = 1 + x_true[:, 3] ** 2; pt = 1 / np.maximum(np.abs(x_true[:, 0] / M.C_BEND), 1e-6)
    th2 = np.stack([G.RF.highland_theta2(np.full(nt, s[2]), pt * np.sqrt(sec2), x_true[:, 3]) for s in G.scatterers(rmed)], 1)
    kT = rng.standard_normal(th2.shape) * np.sqrt(th2); kL = rng.standard_normal(th2.shape) * np.sqrt(th2)
    m0 = x_true[:, [0]] * r + x_true[:, [1]] - x_true[:, [2]] / r + np.einsum("tls,ts->tl", g["dT"], kT)
    m1 = x_true[:, [3]] * r + x_true[:, [4]] + np.einsum("tls,ts->tl", g["dL"], kL)
    m0 += rng.standard_normal(r.shape) * np.sqrt(inp["v0_res"]); m1 += rng.standard_normal(r.shape) * np.sqrt(inp["v1"])
    ti = dict(inp); ti["m0"] = np.where(inp["valid"], m0, 0.0); ti["m1"] = np.where(inp["valid"], m1, 0.0)
    ft = G.forward_filter(ti, ho=False)
    gt = G.gbl_fit(ti, ft["x"], rmed, kinks=True, ho=False, n_iter=2)
    v = inp["valid"] & (inp["valid"].sum(1) >= 4)[:, None]
    mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))
    pr = gt["prob"][inp["valid"].sum(1) >= 4]
    return {"toy tracks": int((inp["valid"].sum(1) >= 4).sum()),
            "toy smoothed pull MAD r-phi": float(mad(gt["pull_rphi"][v])), "toy smoothed pull MAD z": float(mad(gt["pull_rz"][v])),
            "toy smoothed pull RMS r-phi": float(np.std(gt["pull_rphi"][v])), "toy smoothed pull RMS z": float(np.std(gt["pull_rz"][v])),
            "toy P(chi2) deciles (0.1 each if flat)": np.histogram(pr, bins=10, range=(0, 1))[0].tolist() and
            [round(x / len(pr), 3) for x in np.histogram(pr, bins=10, range=(0, 1))[0]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--chunks", type=int, default=None, help="stop after this many chunks")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    json.dump({"census": CENSUS, "census_key": meta["cache_key"], "seed_menu": "pre-menu-update (2026-09-24)",
               "features": list(G.FEATURE_NAMES), "inputs": INPUTS},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    t0 = time.time()
    for ci, ((U, Q), ev) in enumerate(TF.unified_chunks(",".join(INPUTS), None, meta["config"]["chunk"], SIGZ, None)):
        shard = np.load(os.path.join(CENSUS, f"chunk_{ci:05d}.npz"))
        if a.verify:
            _, checks = process(U, Q, shard, verify=True)
            for c, v in checks.items():
                print(f"  {c}: {max(v):.3g}")
            for c, v in toy(U, Q, shard, np.random.default_rng(5)).items():
                print(f"  {c}: {v}")
            return
        dst = os.path.join(OUT, f"chunk_{ci:05d}.npz")
        if not os.path.exists(dst):
            feats, _ = process(U, Q, shard)
            np.savez_compressed(dst, **feats)
        if ci % 10 == 0:
            print(f"chunk {ci:4d}  {time.time()-t0:7.0f} s  peak RSS {rss_gb():.2f} GB", flush=True)
        if a.chunks is not None and ci + 1 >= a.chunks:
            break
    print(f"done {ci+1} chunks in {time.time()-t0:.0f} s, peak RSS {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
