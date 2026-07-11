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
