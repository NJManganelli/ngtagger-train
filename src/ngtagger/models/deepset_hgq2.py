"""HGQ2 (Keras 3) DeepSet tagger, following upstream TrainTagger Keras_v3
DeepSetModelHGQ2: per-particle QConv1D phi network -> average pool ->
classification + pt-regression QDense heads, EBOPs-regularized
quantization-aware training with distributed-arithmetic-friendly layout."""

from __future__ import annotations

import keras
import numpy as np
from keras.layers import Activation, GlobalAveragePooling1D, Input

from ngtagger.models.base import ModelRegistry, TagModel


def _hgq_scopes(beta0: float = 1e-8):
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.constraints import MinMax
    from hgq.regularizers import MonoL1

    scope_weights = QuantizerConfigScope(
        default_q_type="kbi", b0=7, i0=0, overflow_mode="wrap",
        fr=MonoL1(1.0e-8), ir=MonoL1(1.0e-8),
    )
    scope_data = QuantizerConfigScope(
        default_q_type="kif", place="datalane", overflow_mode="wrap",
        f0=7, fr=MonoL1(1.0e-8), ic=MinMax(0, 12),
    )
    scope_layers = LayerConfigScope(enable_ebops=True, heterogeneous_axis=None, beta0=beta0)
    return scope_weights, scope_data, scope_layers


@ModelRegistry.register("DeepSetHGQ2")
class DeepSetHGQ2(TagModel):
    def build(self, input_shape: tuple, n_classes: int):
        from hgq.layers import QBatchNormalization, QConv1D, QDense

        cfg = self.model_config
        s0, s1, s2 = _hgq_scopes(cfg.get("beta", 1e-8))
        with s0, s1, s2:
            n_const, n_feat = input_shape
            inputs = Input(shape=(n_const, n_feat), name="model_input")
            main = QBatchNormalization(name="norm_input")(inputs)

            for i, width in enumerate(cfg.get("conv1d_layers", [10, 10])):
                main = QConv1D(
                    filters=width, kernel_size=1, activation="relu",
                    parallelization_factor=cfg.get("conv1d_parallelisation_factor", [n_const] * 8)[i],
                    name=f"Conv1D_{i + 1}",
                )(main)
            main = GlobalAveragePooling1D(name="avgpool")(main)

            jet_id = main
            for i, width in enumerate(cfg.get("classification_layers", [32, 16])):
                jet_id = QDense(width, activation="relu", name=f"Dense_{i + 1}_jetID")(jet_id)
            jet_id = QDense(n_classes, activation="relu", name="Dense_out_jetID")(jet_id)
            jet_id = Activation("linear", name=self.output_id_name)(jet_id)

            pt = main
            for i, width in enumerate(cfg.get("regression_layers", [10])):
                pt = QDense(width, activation="relu", name=f"Dense_{i + 1}_pT")(pt)
            pt = QDense(1, name=self.output_pt_name)(pt)

            outputs = [jet_id, pt]
            if cfg.get("charge_layers"):
                # 3-class charge head (study B.2.3): {q-, neutral/gluon, q+},
                # 2-bit output field; shares the pooled trunk like the others.
                charge = main
                for i, width in enumerate(cfg["charge_layers"]):
                    charge = QDense(width, activation="relu", name=f"Dense_{i + 1}_charge")(charge)
                charge = QDense(3, activation="relu", name="Dense_out_charge")(charge)
                charge = Activation("linear", name=self.output_charge_name)(charge)
                outputs.append(charge)

            self.model = keras.Model(inputs=inputs, outputs=outputs)

    @property
    def has_charge_head(self) -> bool:
        return bool(self.model_config.get("charge_layers"))

    def _output_names(self) -> list[str]:
        names = [self.output_id_name, self.output_pt_name]
        if self.has_charge_head:
            names.append(self.output_charge_name)
        return names

    def compile(self):
        tc = self.training_config
        opt = keras.optimizers.Adam(learning_rate=tc.get("learning_rate", 1e-2))
        names = self._output_names()
        losses = {
            self.output_id_name: keras.losses.CategoricalCrossentropy(from_logits=True),
            self.output_pt_name: keras.losses.LogCosh(),
        }
        metrics = {self.output_id_name: ["categorical_accuracy"]}
        loss_weights = list(tc.get("loss_weights", [1.0, 1.0]))
        if self.has_charge_head:
            losses[self.output_charge_name] = keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics[self.output_charge_name] = ["categorical_accuracy"]
            if len(loss_weights) < len(names):
                loss_weights.append(tc.get("charge_loss_weight", 1.0))
        if len(loss_weights) != len(names):
            raise ValueError(f"loss_weights has {len(loss_weights)} entries for outputs {names}")

        self._train_model = self.model
        self._mdmm_callbacks = []
        if self.config.get("constraints"):
            from ngtagger.train.mdmm import attach_mdmm

            self._train_model, self._mdmm_callbacks = attach_mdmm(
                self.model, self.config["constraints"], names, self.class_labels or None
            )
        self._train_model.compile(
            optimizer=opt,
            loss=losses,
            loss_weights=dict(zip(names, loss_weights)),
            metrics=metrics,
        )

    def constraint_metrics(self) -> dict:
        from ngtagger.train.mdmm import collect_constraint_metrics

        return collect_constraint_metrics(getattr(self, "_train_model", None),
                                          getattr(self, "_mdmm_callbacks", []))

    def _callbacks(self):
        tc = self.training_config
        cbs = [
            keras.callbacks.EarlyStopping(
                patience=tc.get("EarlyStopping_patience", 10), restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=tc.get("ReduceLROnPlateau_factor", 0.5),
                patience=tc.get("ReduceLROnPlateau_patience", 5),
                min_lr=tc.get("ReduceLROnPlateau_min_lr", 1e-5),
            ),
        ]
        try:
            from hgq.utils.sugar import FreeEBOPs

            cbs.append(FreeEBOPs())
        except ImportError:
            pass
        return cbs

    def fit(self, X, y, pt_target, sample_weight=None, validation_split=0.1, seed: int = 0,
            y_charge=None):
        keras.utils.set_random_seed(seed)
        tc = self.training_config
        targets = {self.output_id_name: y, self.output_pt_name: pt_target}
        if self.has_charge_head:
            if y_charge is None:
                raise ValueError(
                    "model_config.charge_layers enables the charge head but no "
                    "y_charge labels were provided (prepare_dataset supplies "
                    "charge_train; see data/labels.py label_jet_charge)"
                )
            targets[self.output_charge_name] = y_charge
        train_model = getattr(self, "_train_model", None) or self.model
        self.history = train_model.fit(
            X,
            targets,
            sample_weight=sample_weight,
            epochs=tc.get("epochs", 100),
            batch_size=tc.get("batch_size", 2048),
            validation_split=validation_split,
            callbacks=self._callbacks() + getattr(self, "_mdmm_callbacks", []),
            verbose=self.config.get("run_config", {}).get("verbose", 2),
        )
        return self.history
