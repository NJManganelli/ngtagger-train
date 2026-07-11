"""Firmware conversion: HGQ2 keras model -> hls4ml (Vitis backend,
distributed_arithmetic strategy via da4ml), mirroring upstream TrainTagger
Keras_v3 firmware_convert. Also stubs conifer (BDT alternative, Vector_Trees
line) and the cms-hls4ml emulator packaging used by the L1Nano variants.
"""

from __future__ import annotations

import os
import shutil


def convert_hgq2_model(tag_model, firmware_dir: str, build: bool = False):
    """Convert a trained HGQ2 model to an hls4ml Vitis project."""
    import hls4ml

    fw = tag_model.firmware_config
    project_name = fw.get("project_name", "L1TSC4NGJetModel")
    outdir = os.path.join(firmware_dir, project_name)
    shutil.rmtree(outdir, ignore_errors=True)

    config = hls4ml.utils.config_from_keras_model(tag_model.model, granularity="name")
    config["Model"]["Strategy"] = "distributed_arithmetic"  # da4ml
    config["Model"]["ReuseFactor"] = 1
    config["IOType"] = "io_parallel"

    hls_model = hls4ml.converters.convert_from_keras_model(
        tag_model.model,
        backend=fw.get("backend", "Vitis"),
        project_name=project_name,
        clock_period=fw.get("clock_period", 2.5),
        hls_config=config,
        output_dir=outdir,
        part=fw.get("fpga_part", "xcvu13p-flga2577-2-e"),
    )
    hls_model.compile()

    # bit-accuracy check against the keras model on random inputs
    import numpy as np

    x = np.random.default_rng(0).normal(size=(256, *tag_model.model.input_shape[1:])).astype("float32")
    y_keras = tag_model.model.predict(x, verbose=0)
    y_hls = hls_model.predict(x)
    y_keras0 = y_keras[0] if isinstance(y_keras, (list, tuple)) else y_keras
    y_hls0 = y_hls[0] if isinstance(y_hls, (list, tuple)) else y_hls
    max_diff = float(abs(np.asarray(y_keras0) - np.asarray(y_hls0)).max())
    print(f"hls4ml compile OK; max |keras - hls| on random inputs: {max_diff:.6f}")

    if build:
        hls_model.build(csim=False, synth=True, vsynth=False)
    return hls_model


def package_for_emulator(firmware_dir: str, emulator_repo: str, version_tag: str):
    """Package the generated HLS project for the cms-hls4ml emulator external
    (L1TSC4NGJetModel), so the trained model plugs into the L1Nano variants.

    Steps (see https://github.com/cms-hls4ml/L1TSC4NGJetModel):
      1. copy the hls4ml project (firmware/ headers + weights) into the
         emulator repo's model directory named `{version_tag}`
      2. implement/point the hls4mlEmulator Model wrapper at the new project
      3. `make && make install` inside a CMSSW area (hls4mlEmulatorExtras)
      4. set process.l1tSC4NGJetProducer.l1tSC4NGJetModelPath = version_tag

    This is a documented stub: the copy is performed, the wrapper build is
    left to the CMSSW-side environment (scram tools required).
    """
    src = firmware_dir
    dst = os.path.join(emulator_repo, version_tag)
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"copied HLS project into {dst}; build the emulator wrapper in a CMSSW area "
          f"(hls4mlEmulatorExtras: make && make install), then set "
          f"l1tSC4NGJetModelPath='{version_tag}'")


def convert_conifer_stub(*args, **kwargs):
    """Placeholder for BDT (conifer / ydf) alternatives explored on the
    upstream Vector_Trees branch; intentionally unimplemented."""
    raise NotImplementedError("conifer/BDT export not yet implemented (upstream Vector_Trees line)")
