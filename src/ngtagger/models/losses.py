"""Contrastive losses (backend-agnostic via keras.ops), following the
upstream TrainTagger embedding branches (SimCLR NT-Xent)."""

from __future__ import annotations

import keras
from keras import ops


def nt_xent(z1, z2, temperature: float = 0.5):
    """SimCLR NT-Xent loss over a batch of positive embedding pairs.

    z1, z2: (N, D) L2-normalized embeddings of two augmented views.
    """
    z = ops.concatenate([z1, z2], axis=0)  # (2N, D)
    sim = ops.matmul(z, ops.transpose(z)) / temperature  # (2N, 2N)

    n = ops.shape(z1)[0]
    mask = ops.eye(2 * n) * 1e9
    sim = sim - mask  # remove self-similarity from the softmax

    positives = ops.concatenate(
        [ops.diagonal(sim, offset=n), ops.diagonal(sim, offset=-n)], axis=0
    )  # (2N,)

    denom = ops.logsumexp(sim, axis=1)
    loss = -(positives - denom)
    return ops.mean(loss)


def augment_constituents(X, rng, dropout_prob: float = 0.1, pt_smear: float = 0.05,
                         pt_index: int = 0, isfilled_index: int | None = None):
    """Cheap jet-level augmentations for contrastive views on the
    (N, n_const, n_feat) tensor: random constituent dropout + pt smearing.
    Operates in numpy (pre-batching)."""
    import numpy as np

    X = X.copy()
    n, n_const, _ = X.shape
    drop = rng.random((n, n_const)) < dropout_prob
    if isfilled_index is not None:
        drop &= X[:, :, isfilled_index] > 0
    X[drop] = 0.0
    smear = rng.normal(1.0, pt_smear, size=(n, n_const))
    X[:, :, pt_index] *= np.where(X[:, :, pt_index] > 0, smear, 1.0)
    return X
