"""Cache/state utilities shared by LinearNexus layers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass
class RecurrentState:
    """Container that tracks recurrent selective-SSM state."""

    value: Array  # [batch, channels, state_size]

    @classmethod
    def zeros(
        cls,
        *,
        batch_size: int,
        channels: int,
        state_size: int,
        dtype: jnp.dtype,
    ) -> "RecurrentState":
        return cls(value=jnp.zeros((batch_size, channels, state_size), dtype=dtype))

    def update(self, new_value: Array) -> "RecurrentState":
        return RecurrentState(value=new_value)


@dataclass
class ConvState:
    """Cache for causal depthwise convolutions."""

    buffer: Array  # [batch, kernel_size - 1, channels]

    @classmethod
    def zeros(
        cls,
        *,
        batch_size: int,
        kernel_size: int,
        channels: int,
        dtype: jnp.dtype,
    ) -> "ConvState":
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if kernel_size == 1:
            return cls(buffer=jnp.zeros((batch_size, 0, channels), dtype=dtype))
        return cls(buffer=jnp.zeros((batch_size, kernel_size - 1, channels), dtype=dtype))

    def update(self, new_buffer: Array) -> "ConvState":
        return ConvState(buffer=new_buffer)
