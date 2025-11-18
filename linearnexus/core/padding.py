"""Padding/unpadding utilities for variable-length sequences."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def compute_unpadded_indices(mask: Array) -> Tuple[Array, Array]:
    """Returns indices and prefix sums for packed variable-length batches.

    Args:
        mask: Binary tensor [batch, seq] where 1 marks valid tokens.
    Returns:
        Tuple of (indices, cu_seqlens) suitable for packed representations.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be rank-2 [batch, seq]")
    batch, seq = mask.shape
    flat_indices = jnp.arange(batch * seq, dtype=jnp.int32)
    mask_flat = mask.reshape(-1)
    indices = flat_indices[mask_flat.astype(bool)]
    counts = jnp.sum(mask, axis=1, dtype=jnp.int32)
    cu_seqlens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)]
    )
    return indices, cu_seqlens


def unpad(x: Array, indices: Array) -> Array:
    """Gathers valid tokens according to packed indices."""

    if x.ndim < 2:
        raise ValueError("x must be at least 2D (batch, seq, ...)")
    batch, seq = x.shape[:2]
    flat = x.reshape(batch * seq, *x.shape[2:])
    return flat[indices]


def pad(x_flat: Array, indices: Array, batch: int, seq: int) -> Array:
    """Scatters packed tokens back to padded layout."""

    if x_flat.shape[0] != indices.shape[0]:
        raise ValueError("x_flat length must match indices length")
    output_shape = (batch * seq,) + x_flat.shape[1:]
    output = jnp.zeros(output_shape, dtype=x_flat.dtype)
    output = output.at[indices].set(x_flat)
    return output.reshape((batch, seq) + x_flat.shape[1:])
