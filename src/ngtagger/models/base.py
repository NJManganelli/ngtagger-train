"""TagModel base class + registry, following the upstream TrainTagger
JetTagModel/JetModelFactory pattern (Keras_v3 line) but slimmed for the
nano-direct pipeline."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import numpy as np
import yaml


class ModelRegistry:
    _models: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def deco(klass):
            cls._models[name] = klass
            return klass
        return deco

    @classmethod
    def create(cls, name: str, output_dir: str) -> "TagModel":
        if name not in cls._models:
            raise KeyError(f"unknown model '{name}'; known: {sorted(cls._models)}")
        return cls._models[name](output_dir)


class TagModel(ABC):
    """Common interface: build/compile/fit/save/load + firmware conversion."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.model = None
        self.history = None
        self.config: dict = {}
        self.class_labels: list[str] = []
        self.feature_names: list[str] = []
        self.output_id_name = "jet_id_output"
        self.output_pt_name = "pT_output"
        self.output_charge_name = "charge_output"

    def load_yaml(self, path: str):
        with open(path) as f:
            self.config = yaml.safe_load(f)

    @property
    def model_config(self):
        return self.config.get("model_config", {})

    @property
    def training_config(self):
        return self.config.get("training_config", {})

    @property
    def firmware_config(self):
        return self.config.get("firmware_config", {})

    @abstractmethod
    def build(self, input_shape: tuple, n_classes: int): ...

    @abstractmethod
    def compile(self): ...

    @abstractmethod
    def fit(self, X, y, pt_target, sample_weight=None, validation_split=0.1, seed: int = 0,
            y_charge=None): ...

    def constraint_metrics(self) -> dict:
        """Final MDMM multiplier state for logging; {} unless the model
        trained with a constraints section (see train/mdmm.py)."""
        return {}

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save(os.path.join(self.output_dir, "model.keras"))
        meta = {
            "class_labels": self.class_labels,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self):
        import keras

        self.model = keras.models.load_model(os.path.join(self.output_dir, "model.keras"))
        with open(os.path.join(self.output_dir, "meta.json")) as f:
            meta = json.load(f)
        self.class_labels = meta["class_labels"]
        self.feature_names = meta["feature_names"]
        self.config = meta["config"]

    def predict(self, X):
        return self.model.predict(X, batch_size=4096, verbose=0)

    def best_val_loss(self) -> float:
        if self.history is None:
            return np.inf
        return float(np.min(self.history.history.get("val_loss", [np.inf])))

    def firmware_convert(self, firmware_dir: str, build: bool = False):
        from ngtagger.export.hls4ml_export import convert_hgq2_model

        return convert_hgq2_model(self, firmware_dir, build=build)
