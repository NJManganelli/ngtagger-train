"""ngtagger command line interface.

  ngtagger train-jetmulticlass       -c configs/deepset_hgq2.yaml -i nano1.root nano2.root -o output/run1
  ngtagger multishot-jetmulticlass  -c configs/deepset_hgq2.yaml -i nano.root -o output/ms -n 5 -p 2
  ngtagger train-jetmulticlass-tabfm -i nano.root -o output/tabfm
  ngtagger export                   -m output/ms/best -o firmware/
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

    p_train = sub.add_parser("train-jetmulticlass",
                             help="single training run of the multiclass jet tagger")
    p_train.add_argument("-c", "--config", required=True)
    p_train.add_argument("-i", "--inputs", nargs="+", required=True, help="L1PFTrkNano root files")
    p_train.add_argument("-o", "--output", required=True)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--max-events", type=int, default=None)

    p_ms = sub.add_parser("multishot-jetmulticlass",
                          help="multi-seed training of the multiclass jet tagger, keep best")
    p_ms.add_argument("-c", "--config", required=True)
    p_ms.add_argument("-i", "--inputs", nargs="+", required=True)
    p_ms.add_argument("-o", "--output", required=True)
    p_ms.add_argument("-n", "--n-shots", type=int, default=5)
    p_ms.add_argument("-p", "--parallel", type=int, default=None,
                      help="concurrent shots (default: auto per machine)")
    p_ms.add_argument("--max-events", type=int, default=None)

    p_jt = sub.add_parser("train-jetmulticlass-tabfm",
                          help="TabFM baselines for the jet tagger: 8-class flavour, "
                               "pt-ratio regression, 3-class charge")
    p_jt.add_argument("-i", "--inputs", nargs="+", required=True, help="withGen nano files")
    p_jt.add_argument("-o", "--output", required=True)
    p_jt.add_argument("--n-const", type=int, default=8,
                      help="constituents per jet (flattened slot-major for the tabular model). "
                           "Default 8, NOT the deepset's 16: the shipped TabFM checkpoint "
                           "returns all-NaN predictions at 320 flattened features (16 const) "
                           "while 160 (8 const) is clean")
    p_jt.add_argument("--feature-groups", nargs="*", default=None,
                      help="feature groups (same vocabulary as the deepset configs)")
    p_jt.add_argument("--heads", nargs="+", default=["flavour", "pt", "charge"],
                      choices=["flavour", "pt", "charge"], help="which heads to fit")
    p_jt.add_argument("--max-context", type=int, default=4096,
                      help="rows shown to TabFM as in-context 'training' data")
    p_jt.add_argument("--max-eval", type=int, default=8192, help="evaluation rows")
    p_jt.add_argument("--unbalanced-context", action="store_true",
                      help="sample classification contexts at natural class proportions")
    p_jt.add_argument("--unbalanced-eval", action="store_true",
                      help="evaluate at natural class proportions instead of class-balanced")
    p_jt.add_argument("--n-estimators", type=int, default=8, help="TabFM ensemble members")
    p_jt.add_argument("--gen-match-dr", type=float, default=0.4, help="jet-radius gen matching cone")
    p_jt.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p_jt.add_argument("--max-events", type=int, default=None)
    p_jt.add_argument("--seed", type=int, default=0)

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
                      help="BDT-input ablation tier: A = classic-7 TrackQuality baseline "
                           "(reference hw track word only, no refit info); B = A + refit "
                           "counters/occupancy, position pulls/residuals and refit-kick "
                           "deltas (no angles, no chi2); C = B + alpha (bending-angle) "
                           "features + chi2IncRPhiTot; D = C + beta features + "
                           "chi2IncRZTot. 'matrix' runs one A cell + B/C/D per config")
    p_rq.add_argument("--config", choices=["AIII", "AAII", "AAAI", "AAAA"], default="AAAA",
                      help="digiRefit config (ignored for tier A; used for a single B/C/D run)")
    p_rq.add_argument("--track-table", default="L1TTrack",
                      help="reference track table (L1TTrack or L1TExtTrack) for legacy "
                           "single-storage nano; for simultaneous-storage nano the "
                           "SmartPixels scenario track table (e.g. "
                           "L1TSmartPixelsTrackDigiRefitAAII) together with "
                           "--crossref-track-table")
    p_rq.add_argument("--crossref-track-table", default=None,
                      help="simultaneous-storage nano only: the nominal track table "
                           "(L1TTrack, or L1TExtTrack for Ext scenario tables) that "
                           "supplies the seed hw word, nStubs and TP-truth columns the "
                           "SmartPixels scenario tables do not carry; rows are linked "
                           "implicitly by index (scenario track i == nominal track i)")
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

    for _name, _help in (
        ("train-refitquality-tabfm",
         "TabFM (tabular foundation model) genuine-vs-fake baseline on the refit features"),
        ("train-refitquality-tabfmmulticlass",
         "TabFM track-ORIGIN classification (electron/muon/pion/kaon/proton + combinatorial fakes)"),
    ):
      _p = sub.add_parser(_name, help=_help)
      _p.add_argument("-i", "--inputs", nargs="+", required=True,
                      help="withGen SmartPixels digiRefit nano files")
      _p.add_argument("-o", "--output", required=True)
      _p.add_argument("--config", choices=["AIII", "AAII", "AAAI", "AAAA"], default="AAII",
                      help="digiRefit config (selects the variant/hit tables)")
      _p.add_argument("--tier", choices=["A", "B", "C", "D"], default="D",
                      help="feature tier (same construction as train-refitquality; "
                           "D = full refit visibility)")
      _p.add_argument("--track-table", default="L1TTrack",
                      help="reference track table, or the SmartPixels scenario table "
                           "together with --crossref-track-table")
      _p.add_argument("--crossref-track-table", default=None,
                      help="simultaneous-storage nano: the nominal table supplying the "
                           "seed hw word, nStubs and TP-truth columns")
      _p.add_argument("--label", default="looselyGenuine",
                      choices=["genuine", "looselyGenuine"],
                      help="positive class for the binary mode (ignored for multiclass)")
      _p.add_argument("--max-context", type=int, default=4096,
                      help="rows shown to TabFM as in-context 'training' data")
      _p.add_argument("--max-eval", type=int, default=8192, help="evaluation rows")
      _p.add_argument("--unbalanced-eval", action="store_true",
                      help="evaluate at natural class proportions instead of class-balanced "
                           "(balanced keeps every rare-class row; metrics are within-class "
                           "so balancing does not bias AUC or the fake rates)")
      _p.add_argument("--unbalanced-context", action="store_true",
                      help="sample the context at natural class proportions instead of "
                           "class-balanced (fakes are ~1%% of tracks, so balancing is the default)")
      _p.add_argument("--n-estimators", type=int, default=8,
                      help="TabFM ensemble members (cost scales with this)")
      _p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
      _p.add_argument("--max-events", type=int, default=None)
      _p.add_argument("--seed", type=int, default=0)

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

    if args.cmd == "train-jetmulticlass":
        from ngtagger.train.trainer import run_training

        run_training(args.config, args.inputs, args.output, seed=args.seed,
                     max_events=args.max_events)
    elif args.cmd == "multishot-jetmulticlass":
        from ngtagger.train.multishot import run_multishot

        run_multishot(args.config, args.inputs, args.output, n_shots=args.n_shots,
                      parallel=args.parallel, max_events=args.max_events)
    elif args.cmd == "train-jetmulticlass-tabfm":
        from ngtagger.train.tabfm_tagger import train_tagger_tabfm

        train_tagger_tabfm(args.inputs, args.output, n_const=args.n_const,
                           feature_groups=args.feature_groups, heads=tuple(args.heads),
                           max_context=args.max_context, max_eval=args.max_eval,
                           balanced_context=not args.unbalanced_context,
                           balanced_eval=not args.unbalanced_eval,
                           n_estimators=args.n_estimators, gen_match_dr=args.gen_match_dr,
                           device=args.device, max_events=args.max_events, seed=args.seed)
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
                              spec_version=args.spec_version,
                              crossref_track_table=args.crossref_track_table)
        elif args.tier == "matrix":
            _, meta_all = train_matrix(args.inputs, args.output, track_table=args.track_table,
                                       label=args.label, max_events=args.max_events, seed=args.seed,
                                       crossref_track_table=args.crossref_track_table)
            if args.conifer:
                for tag in meta_all:
                    export_conifer(args.output, tag)
        else:
            ref, var, hits = load_refit_tables(args.inputs, args.config,
                                               args.track_table, args.max_events,
                                               crossref_track_table=args.crossref_track_table)
            ext = "Ext" if "Ext" in args.track_table else ""
            tables = {"ref": args.crossref_track_table or args.track_table,
                      "var": f"L1TSmartPixels{ext}TrackDigiRefit{args.config}"}
            train_one(ref, var, hits, args.tier, args.config, args.output,
                      label=args.label, seed=args.seed, tables=tables)
            if args.tier == "A" and args.crossref_track_table:
                # simultaneous storage: also train the scenario-hw baseline
                train_one(ref, var, hits, "A", args.config, args.output,
                          label=args.label, seed=args.seed, hw_source="var",
                          tables=tables)
            if args.conifer:
                tags = ["A" if args.tier == "A" else f"{args.tier}-{args.config}"]
                if args.tier == "A" and args.crossref_track_table:
                    tags.append(f"A-spx-{args.config}")
                for tag in tags:
                    export_conifer(args.output, tag)
    elif args.cmd in ("train-refitquality-tabfm", "train-refitquality-tabfmmulticlass"):
        from ngtagger.train.tabfm_refitq import train_refitq_tabfm

        train_refitq_tabfm(args.inputs, args.output, config=args.config,
                           track_table=args.track_table,
                           crossref_track_table=args.crossref_track_table,
                           tier=args.tier, label=args.label,
                           multiclass=(args.cmd == "train-refitquality-tabfmmulticlass"),
                           max_context=args.max_context, max_eval=args.max_eval,
                           balanced_context=not args.unbalanced_context,
                           balanced_eval=not args.unbalanced_eval,
                           n_estimators=args.n_estimators, device=args.device,
                           max_events=args.max_events, seed=args.seed)
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
