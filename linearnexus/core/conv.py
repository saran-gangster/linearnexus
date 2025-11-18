"""Convolution helpers shared across LinearNexus layers."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

Array = jax.Array


def depthwise_conv1d_causal(
    inputs: Array,
    weight: Array,
    bias: Optional[Array],
    *,
    cache: Optional[Array] = None,
) -> tuple[Array, Array]:
    """Depthwise causal conv1d with optional cache warm-start.

    Args:
        inputs: Tensor shaped [batch, seq, channels].
        weight: Depthwise weights shaped [kernel, channels].
        bias: Optional bias shaped [channels].
        cache: Optional tensor storing the last ``kernel - 1`` timesteps to seed
            the convolution cache. Shape must be [batch, kernel - 1, channels].
    Returns:
        Tuple of (conv_output, new_cache) where conv_output matches ``inputs`` and
        new_cache stores the final ``kernel - 1`` activations for reuse.
    """

    batch_size, seq_len, channels = inputs.shape
    kernel_size = weight.shape[0]
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    if kernel_size == 1:
        conv_output = inputs
        if bias is not None:
            conv_output = conv_output + bias
        new_cache = jnp.zeros((batch_size, 0, channels), dtype=inputs.dtype)
        return conv_output, new_cache

    if cache is None:
        cache = jnp.zeros((batch_size, kernel_size - 1, channels), dtype=inputs.dtype)

    if cache.shape != (batch_size, kernel_size - 1, channels):
        raise ValueError(
            "cache shape must be (batch, kernel-1, channels); got"
            f" {cache.shape} for kernel_size={kernel_size}"
        )

    weight_dw = weight[:, None, :]
    conv_input = jnp.concatenate([cache, inputs], axis=1)
    conv_output = jax.lax.conv_general_dilated(
        conv_input,
        weight_dw,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=channels,
    )
    conv_output = conv_output[:, -seq_len:, :]
    if bias is not None:
        conv_output = conv_output + bias
    new_cache = conv_input[:, -(kernel_size - 1) :, :]
    return conv_output, new_cache
