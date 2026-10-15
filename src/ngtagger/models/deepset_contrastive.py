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
    """Backend-neutral SimCLR wrapper (tensorflow / jax / torch).

    The pair loss is expressed by overriding ``compute_loss`` rather than by
    hand-rolling a ``tf.GradientTape`` training step, so gradients come from
    Keras 3's built-in per-backend ``train_step``. The two augmented views are
    passed as a single two-element input and the normalized embeddings are
    returned stacked on axis 1, which keeps the whole path inside ``keras.ops``.
    """

    def __init__(self, encoder, projector, temperature=0.5, **kw):
        kw.setdefault("name", "simclr")
        super().__init__(**kw)
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature

    def embed(self, x):
        """L2-normalized projection head output for one view."""
        z = self.projector(self.encoder(x))
        return z / (ops.norm(z, axis=1, keepdims=True) + 1e-8)

    def call(self, inputs):
        x1, x2 = inputs
        return ops.stack([self.embed(x1), self.embed(x2)], axis=1)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs):
        # y / sample_weight are unused: NT-Xent is self-supervised, the targets
        # supplied by _PairSequence are placeholders.
        del x, y, sample_weight, kwargs
        return nt_xent(y_pred[:, 0], y_pred[:, 1], self.temperature)


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
            _PairSequence(v1, v2, tc.get("batch_size", 2048), seed=seed),
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


class _PairSequence(keras.utils.PyDataset):
    """Backend-neutral ((view1, view2), placeholder_y) batch feeder.

    Replaces the former tf.data pipeline. Incomplete trailing batches are
    dropped, matching the old drop_remainder=True: NT-Xent normalizes over the
    in-batch negatives, so a short final batch would shift the loss scale.
    """

    def __init__(self, v1, v2, batch_size, seed=0, **kw):
        super().__init__(**kw)
        self.v1 = v1
        self.v2 = v2
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(len(v1))

    def __len__(self):
        return len(self.v1) // self.batch_size

    def __getitem__(self, idx):
        sel = self._order[idx * self.batch_size:(idx + 1) * self.batch_size]
        placeholder_y = np.zeros((len(sel), 1), dtype="float32")
        return (self.v1[sel], self.v2[sel]), placeholder_y

    def on_epoch_end(self):
        self._order = self._rng.permutation(len(self.v1))
