"""ngtagger command line interface.

  ngtagger train      -c configs/deepset_hgq2.yaml -i nano1.root nano2.root -o output/run1
  ngtagger multishot  -c configs/deepset_hgq2.yaml -i nano.root -o output/ms -n 5 -p 2
  ngtagger export     -m output/ms/best -o firmware/
  ngtagger inspect-nano nano.root
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def main(argv=None):
    parser = ArgumentParser(prog="ngtagger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="single training run")
    p_train.add_argument("-c", "--config", required=True)
    p_train.add_argument("-i", "--inputs", nargs="+", required=True, help="L1PFTrkNano root files")
    p_train.add_argument("-o", "--output", required=True)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--max-events", type=int, default=None)

    p_ms = sub.add_parser("multishot", help="multi-seed training, keep best")
    p_ms.add_argument("-c", "--config", required=True)
    p_ms.add_argument("-i", "--inputs", nargs="+", required=True)
    p_ms.add_argument("-o", "--output", required=True)
    p_ms.add_argument("-n", "--n-shots", type=int, default=5)
    p_ms.add_argument("-p", "--parallel", type=int, default=None,
                      help="concurrent shots (default: auto per machine)")
    p_ms.add_argument("--max-events", type=int, default=None)

    p_exp = sub.add_parser("export", help="hls4ml firmware conversion")
    p_exp.add_argument("-m", "--model", required=True, help="trained model directory")
    p_exp.add_argument("-o", "--output", default="firmware")
    p_exp.add_argument("--build", action="store_true", help="run HLS synthesis")
    p_exp.add_argument("--emulator-repo", default=None,
                       help="cms-hls4ml L1TSC4NGJetModel checkout to package into")
    p_exp.add_argument("--version-tag", default="L1TSC4NGJetModel_dev")

    p_tq = sub.add_parser("train-trkquality", help="retrain the track-quality GBDT")
    p_tq.add_argument("-i", "--inputs", nargs="+", required=True, help="withGen L1TrkNano files")
    p_tq.add_argument("-o", "--output", required=True)
    p_tq.add_argument("--track-table", default="L1TTrack", help="L1TTrack or L1TExtTrack")
    p_tq.add_argument("--label", default="genuine", choices=["genuine", "looselyGenuine"])
    p_tq.add_argument("--max-events", type=int, default=None)
    p_tq.add_argument("--conifer", action="store_true", help="export conifer model json (+cpp project)")

    p_rq = sub.add_parser("train-refitquality",
                          help="refit-aware track-quality BDT study (SmartPixels digiRefit)")
    p_rq.add_argument("-i", "--inputs", nargs="+", required=True,
                      help="withGen SmartPixels digiRefit nano files")
    p_rq.add_argument("-o", "--output", required=True)
    p_rq.add_argument("--tier", choices=["A", "B", "C", "D", "matrix"], default="matrix",
                      help="single tier, or 'matrix' for the full 13-cell A/B/C/D x config run")
    p_rq.add_argument("--config", choices=["AIII", "AAII", "AAAI", "AAAA"], default="AAAA",
                      help="digiRefit config (ignored for tier A; used for a single B/C/D run)")
    p_rq.add_argument("--track-table", default="L1TTrack", help="L1TTrack or L1TExtTrack")
    p_rq.add_argument("--label", default="genuine", choices=["genuine", "looselyGenuine"])
    p_rq.add_argument("--max-events", type=int, default=None)
    p_rq.add_argument("--seed", type=int, default=0)
    p_rq.add_argument("--conifer", action="store_true",
                      help="export conifer model json for the trained cell(s)")
    p_rq.add_argument("--export-conifer", action="store_true",
                      help="SPEC-ORDER path: train the producer-contract REFIT_BDT_FEATURES "
                           "model and export a deployable conifer JSON. Uses --config and "
                           "--spec-version; ignores --tier.")
    p_rq.add_argument("--spec-version", type=int, choices=[0, 1], default=0,
                      help="SPEC-ORDER feature vector version: 0 = v0 (17 features), "
                           "1 = v1 (24 features: v0 + the classic-7 TrackQuality hw features "
                           "of the input track). The producer selects the assembly by n_features.")
    p_rq.add_argument("--conifer-name", default="refitq_conifer_v0",
                      help="basename for the SPEC-ORDER conifer JSON + metadata")
    p_rq.add_argument("--provenance", default="",
                      help="free-text provenance recorded in the SPEC-ORDER model metadata")

    p_vtx = sub.add_parser("train-nnvtx", help="retrain the E2E NNVtx + association networks")
    p_vtx.add_argument("-i", "--inputs", nargs="+", required=True, help="withGen L1PFTrkNano files")
    p_vtx.add_argument("-o", "--output", required=True)
    p_vtx.add_argument("--track-table", default="L1TTrack")
    p_vtx.add_argument("--extra-features", nargs="*", default=[],
                       help="additional branch names or computed features (see COMPUTED_FEATURES)")
    p_vtx.add_argument("--epochs", type=int, default=30)
    p_vtx.add_argument("--max-events", type=int, default=None)

    p_dv = sub.add_parser("train-dispvtx", help="retrain the displaced-vertex GBDT")
    p_dv.add_argument("-i", "--inputs", nargs="+", required=True, help="withGen L1TrkNano files")
    p_dv.add_argument("-o", "--output", required=True)
    p_dv.add_argument("--max-events", type=int, default=None)
    p_dv.add_argument("--conifer", action="store_true")

    p_vs = sub.add_parser("vtx-study",
                          help="fastHisto vertex (dx, dy) / peak-finder kernel study")
    mode = p_vs.add_mutually_exclusive_group(required=True)
    mode.add_argument("--kernel-scan", action="store_true",
                      help="two-close-vertices kernel scan (flat vs tapered)")
    mode.add_argument("--realdata", nargs="+", metavar="FILE",
                      help="real-data (dx, dy) smoke over extended-track nano file(s)")
    p_vs.add_argument("--outdir", default="eval_refitq/vtxdxy")
    p_vs.add_argument("--track-table", default="L1TExtTrack",
                      help="extended (5-par) track table for --realdata")
    p_vs.add_argument("--d0-gate", type=float, default=0.15,
                      help="|d0| prompt-track gate [cm] for --realdata")
    p_vs.add_argument("--seed", type=int, default=0, help="--kernel-scan toy seed")
    p_vs.add_argument("--no-plot", action="store_true", help="skip PNG output")

    p_ins = sub.add_parser("inspect-nano", help="print tagger-relevant tables of a nano file")
    p_ins.add_argument("file")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from ngtagger.train.trainer import run_training

        run_training(args.config, args.inputs, args.output, seed=args.seed,
                     max_events=args.max_events)
    elif args.cmd == "multishot":
        from ngtagger.train.multishot import run_multishot

        run_multishot(args.config, args.inputs, args.output, n_shots=args.n_shots,
                      parallel=args.parallel, max_events=args.max_events)
    elif args.cmd == "export":
        from ngtagger.models.base import ModelRegistry
        import json

        with open(os.path.join(args.model, "meta.json")) as f:
            meta = json.load(f)
        model = ModelRegistry.create(meta["config"]["model"], args.model)
        model.load()
        model.firmware_convert(args.output, build=args.build)
        if args.emulator_repo:
            from ngtagger.export.hls4ml_export import package_for_emulator

            project = model.firmware_config.get("project_name", "L1TSC4NGJetModel")
            package_for_emulator(os.path.join(args.output, project),
                                 args.emulator_repo, args.version_tag)
    elif args.cmd == "train-trkquality":
        from ngtagger.train.trkquality import export_conifer, train_trkquality

        train_trkquality(args.inputs, args.output, track_table=args.track_table,
                         label=args.label, max_events=args.max_events)
        if args.conifer:
            export_conifer(args.output)
    elif args.cmd == "train-refitquality":
        from ngtagger.train.refitquality import (
            export_conifer, load_refit_tables, train_matrix, train_one)

        if getattr(args, "export_conifer", False):
            from ngtagger.train.refitquality import train_refitq_spec

            train_refitq_spec(args.inputs, args.output, config=args.config,
                              track_table=args.track_table, label=args.label,
                              max_events=args.max_events, seed=args.seed,
                              conifer_name=args.conifer_name, provenance=args.provenance,
                              spec_version=args.spec_version)
        elif args.tier == "matrix":
            train_matrix(args.inputs, args.output, track_table=args.track_table,
                         label=args.label, max_events=args.max_events, seed=args.seed)
            if args.conifer:
                for tag in ["A", "B-AIII", "B-AAII", "B-AAAI", "B-AAAA",
                            "C-AIII", "C-AAII", "C-AAAI", "C-AAAA",
                            "D-AIII", "D-AAII", "D-AAAI", "D-AAAA"]:
                    export_conifer(args.output, tag)
        else:
            ref, var, hits = load_refit_tables(args.inputs, args.config,
                                               args.track_table, args.max_events)
            train_one(ref, var, hits, args.tier, args.config, args.output,
                      label=args.label, seed=args.seed)
            if args.conifer:
                tag = "A" if args.tier == "A" else f"{args.tier}-{args.config}"
                export_conifer(args.output, tag)
    elif args.cmd == "train-nnvtx":
        from ngtagger.train.nnvtx import train_nnvtx

        train_nnvtx(args.inputs, args.output, track_table=args.track_table,
                    extra_features=args.extra_features, max_events=args.max_events,
                    epochs=args.epochs)
    elif args.cmd == "train-dispvtx":
        from ngtagger.train.dispvtx import export_conifer, train_dispvtx

        train_dispvtx(args.inputs, args.output, max_events=args.max_events)
        if args.conifer:
            export_conifer(args.output)
    elif args.cmd == "vtx-study":
        from ngtagger.train.vtxstudy import run_kernel_scan, run_vertex_dxy_smoke

        if args.kernel_scan:
            run_kernel_scan(args.outdir, seed=args.seed, make_plot=not args.no_plot)
        else:
            run_vertex_dxy_smoke(args.realdata, args.outdir, track_table=args.track_table,
                                 d0_gate=args.d0_gate, make_plot=not args.no_plot)
    elif args.cmd == "inspect-nano":
        import uproot

        with uproot.open(f"{args.file}:Events") as tree:
            for prefix in ("L1puppiJetSC4NG", "L1SC4NGJetCands", "L1ExtPuppiCand",
                           "L1PuppiCand", "L1TTrack", "L1TExtTrack", "L1HGCCluster",
                           "GenJet", "GenVisTau"):
                branches = [b for b in tree.keys() if b.startswith(prefix + "_") or b == "n" + prefix]
                print(f"{prefix}: {len(branches)} branches")
                for b in sorted(branches):
                    print("   ", b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
