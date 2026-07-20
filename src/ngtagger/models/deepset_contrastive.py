"""Contrastive (SimCLR) DeepSet tagger, following the upstream TrainTagger
embedding/embedding_kv3 line (FloatingDeepSetEmbeddingModel):

  stage 1: float encoder + projection head pre-trained with NT-Xent on
           augmented jet views (constituent dropout + pt smearing)
  stage 2: projection head dropped; quantized (HGQ2) classification and
           regression heads fine-tuned on top of the (frozen-then-unfrozen)
           encoder for the hardware model.

The two-stage scheme is the local-minima mitigation for QAT: the embedding
initializes the quantized fine-tune far from the poor minima that cold
quantized training falls into.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops
from keras.layers import Conv1D, Dense, GlobalAveragePooling1D, Input, Activation

from ngtagger.models.base import ModelRegistry, TagModel
from ngtagger.models.deepset_hgq2 import _hgq_scopes
from ngtagger.models.losses import augment_constituents, nt_xent


class _SimCLRModel(keras.Model):
    def __init__(self, encoder, projector, temperature=0.5, **kw):
        kw.setdefault("name", "simclr")
        super().__init__(**kw)
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, x):
        return self.projector(self.encoder(x))

    def compute_loss_pair(self, x1, x2):
        z1 = self(x1)
        z2 = self(x2)
        z1 = z1 / (ops.norm(z1, axis=1, keepdims=True) + 1e-8)
        z2 = z2 / (ops.norm(z2, axis=1, keepdims=True) + 1e-8)
        return nt_xent(z1, z2, self.temperature)

    def train_step(self, data):
        import tensorflow as tf

        x1, x2 = data
        with tf.GradientTape() as tape:
            loss = self.compute_loss_pair(x1, x2)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


@ModelRegistry.register("DeepSetContrastive")
class DeepSetContrastive(TagModel):
    def build(self, input_shape: tuple, n_classes: int):
        cfg = self.model_config
        if cfg.get("charge_layers"):
            raise NotImplementedError(
                "charge head is only implemented for DeepSetHGQ2 (drop "
                "model_config.charge_layers or switch models)"
            )
        n_const, n_feat = input_shape
        self._input_shape = input_shape
        self._n_classes = n_classes

        # float encoder for embedding pre-training
        enc_in = Input(shape=(n_const, n_feat))
        h = enc_in
        for i, width in enumerate(cfg.get("conv1d_layers", [16, 16])):
            h = Conv1D(width, 1, activation="relu", name=f"enc_conv_{i}")(h)
        h = GlobalAveragePooling1D(name="enc_pool")(h)
        self.encoder = keras.Model(enc_in, h, name="encoder")

        proj_in = Input(shape=(h.shape[-1],))
        p = Dense(cfg.get("projection_dims", 16), activation="relu")(proj_in)
        p = Dense(cfg.get("projection_dims", 16))(p)
        self.projector = keras.Model(proj_in, p, name="projector")

        self.simclr = _SimCLRModel(self.encoder, self.projector,
                                   temperature=cfg.get("temperature", 0.5))

    def _build_finetune_head(self):
        """Quantized heads on top of the pre-trained encoder (HGQ2)."""
        from hgq.layers import QConv1D, QDense, QBatchNormalization

        cfg = self.model_config
        n_const, n_feat = self._input_shape
        s0, s1, s2 = _hgq_scopes(cfg.get("beta", 1e-8))
        with s0, s1, s2:
            inputs = Input(shape=(n_const, n_feat), name="model_input")
            main = QBatchNormalization(name="norm_input")(inputs)
            for i, width in enumerate(cfg.get("conv1d_layers", [16, 16])):
                main = QConv1D(width, 1, activation="relu", name=f"Conv1D_{i + 1}")(main)
            main = GlobalAveragePooling1D(name="avgpool")(main)

            jet_id = main
            for i, width in enumerate(cfg.get("classification_layers", [32, 16])):
                jet_id = QDense(width, activation="relu", name=f"Dense_{i + 1}_jetID")(jet_id)
            jet_id = QDense(self._n_classes, activation="relu", name="Dense_out_jetID")(jet_id)
            jet_id = Activation("linear", name=self.output_id_name)(jet_id)

            pt = main
            for i, width in enumerate(cfg.get("regression_layers", [10])):
                pt = QDense(width, activation="relu", name=f"Dense_{i + 1}_pT")(pt)
            pt = QDense(1, name=self.output_pt_name)(pt)

            self.model = keras.Model(inputs, [jet_id, pt])

        # transfer float encoder weights into the quantized conv stack
        for i in range(len(cfg.get("conv1d_layers", [16, 16]))):
            src = self.encoder.get_layer(f"enc_conv_{i}")
            dst = self.model.get_layer(f"Conv1D_{i + 1}")
            try:
                dst.set_weights(src.get_weights() + dst.get_weights()[len(src.get_weights()):])
            except (ValueError, IndexError):
                # HGQ2 layers carry extra quantizer variables; copy kernel/bias only
                dst_w = dst.get_weights()
                dst_w[0] = src.get_weights()[0]
                if len(src.get_weights()) > 1:
                    dst_w[1] = src.get_weights()[1]
                dst.set_weights(dst_w)

    def compile(self):
        tc = self.training_config
        self.simclr.compile(optimizer=keras.optimizers.Adam(tc.get("embedding_lr", 1e-2)))

    def fit(self, X, y, pt_target, sample_weight=None, validation_split=0.1, seed: int = 0,
            y_charge=None):
        # y_charge accepted for interface parity; no charge head here (guarded in build)
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        tc = self.training_config

        # ---- stage 1: contrastive embedding pre-training on augmented views
        isfilled_idx = self.feature_names.index("isfilled") if "isfilled" in self.feature_names else None
        v1 = augment_constituents(X, rng, isfilled_index=isfilled_idx)
        v2 = augment_constituents(X, rng, isfilled_index=isfilled_idx)
        self.simclr.fit(
            _pair_dataset(v1, v2, tc.get("batch_size", 2048)),
            epochs=tc.get("embedding_epochs", 20),
            verbose=self.config.get("run_config", {}).get("verbose", 2),
        )

        # ---- stage 2: quantized fine-tune with the standard supervised losses
        self._build_finetune_head()
        opt = keras.optimizers.Adam(learning_rate=tc.get("learning_rate", 1e-2))
        self.model.compile(
            optimizer=opt,
            loss={
                self.output_id_name: keras.losses.CategoricalCrossentropy(from_logits=True),
                self.output_pt_name: keras.losses.LogCosh(),
            },
            loss_weights=dict(zip([self.output_id_name, self.output_pt_name],
                                  tc.get("loss_weights", [1.0, 1.0]))),
            metrics={self.output_id_name: ["categorical_accuracy"]},
        )
        self.history = self.model.fit(
            X,
            {self.output_id_name: y, self.output_pt_name: pt_target},
            sample_weight=sample_weight,
            epochs=tc.get("finetuning_epochs", 100),
            batch_size=tc.get("batch_size", 2048),
            validation_split=validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=tc.get("EarlyStopping_patience", 10),
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=tc.get("ReduceLROnPlateau_factor", 0.5),
                                                  patience=tc.get("ReduceLROnPlateau_patience", 5),
                                                  min_lr=tc.get("ReduceLROnPlateau_min_lr", 1e-5)),
            ],
            verbose=self.config.get("run_config", {}).get("verbose", 2),
        )
        return self.history


def _pair_dataset(v1, v2, batch_size):
    """tf.data dataset yielding (view1, view2) batches for the SimCLR
    train_step."""
    import tensorflow as tf

    return tf.data.Dataset.from_tensor_slices((v1, v2)).shuffle(len(v1)).batch(batch_size, drop_remainder=True)
