"""Gating helpers shared across selective-state layers."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def low_rank_project(
    x: Array,
    kernel: Callable[[Array], Array],
    *,
    activation: Callable[[Array], Array] = jax.nn.silu,
) -> Array:
    """Applies a low-rank projection + activation used for gates.

    Args:
        x: Input tensor [batch, seq, hidden].
        kernel: Callable implementing the low-rank linear layers (e.g., nnx.Sequential).
        activation: Activation applied after the kernel (default SiLU).
    Returns:
        Projected tensor with activation applied.
    """

    projected = kernel(x)
    return activation(projected)


def normalize_gate_logits(
    logits: Array,
    *,
    normalizer: float = 1.0,
    clip_range: tuple[float, float] | None = None,
) -> Array:
    """Normalizes gate logits to stabilize selective SSM updates."""

    gate = jax.nn.log_sigmoid(logits) / normalizer
    if clip_range is not None:
        gate = jnp.clip(gate, clip_range[0], clip_range[1])
    return gate
