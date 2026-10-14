"""DeepSetHGQ2 with LINEAR class logits ("linout" recipe of the tagger_sp studies).

The framework's DeepSetHGQ2 applies a ReLU to the class logits (Dense_out_jetID) before the
from-logits cross-entropy. A jet whose eight pre-activations are all negative gets logits that
are exactly 0 -> a uniform softmax (1/8 per class) -> b-vs-udsg D = 1/3 exactly, and the ReLU
passes no gradient, so training never recovers it. In the ladder recipe 0-50% of the test jets
of a seed are such "dead" jets and the matched-WP thresholds tie at D = 1/3 (eff and fake rate
then collapse, e.g. to 0 at 10% / 1%). This model is the framework's, built unchanged, with
only that output ReLU removed (linear logits); everything else in the recipe is the ladder's.
Importing this module registers it ("DeepSetHGQ2LinOut", used by tagger_sp_linout.yaml).
"""
from __future__ import annotations

import keras

from ngtagger.models.base import ModelRegistry
from ngtagger.models.deepset_hgq2 import DeepSetHGQ2

OUT_LAYER = "Dense_out_jetID"


@ModelRegistry.register("DeepSetHGQ2LinOut")
class DeepSetHGQ2LinOut(DeepSetHGQ2):
    def build(self, input_shape: tuple, n_classes: int):
        super().build(input_shape, n_classes)
        # HGQ's QDense applies self.activation in call(), so this takes effect in the graph
        self.model.get_layer(OUT_LAYER).activation = keras.activations.linear
