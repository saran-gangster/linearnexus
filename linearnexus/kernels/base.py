"""Kernel protocols shared across LinearNexus."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Tuple

try:  # Optional during doc builds; runtime requires JAX
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("JAX is required to use LinearNexus kernels") from exc

Array = jax.Array


@dataclass
class GridConfig:
    """Tunable grid description for chunked kernels."""

    block_shape: tuple[int, ...]
    num_programs: tuple[int, ...]


class KernelMode(str, Enum):
    """Supported kernel execution strategies."""

    CHUNK = "chunk"
    RECURRENT = "recurrent"


class SelectiveKernelProtocol(Protocol):
    """Common protocol all selective kernels must implement."""

    def forward_chunk(
        self,
        params,
        inputs,
        state,
        *,
        chunk_size: int,
    ) -> Tuple[jax.Array, object]:
        """Chunked forward pass.

        Args:
            params: Kernel-specific parameter container.
            inputs: Kernel inputs (activations, deltas, gates, etc.).
            state: Kernel state to seed recurrence.
            chunk_size: Preferred chunk size for processing.
        Returns:
            Tuple of (output, updated_state).
        """

    def forward_recurrent(self, params, inputs, state) -> Tuple[jax.Array, object]:
        """Recurrent single-step forward pass."""

    def get_grid_config(
        self,
        *,
        batch_size: int,
        seq_len: int,
        feature_dim: int,
    ) -> GridConfig:
        """Return launch/grid preferences for the kernel."""


@dataclass
class KernelNumericConfig:
    """Lightweight numeric knobs used by kernels."""

    dtype: jnp.dtype = jnp.float32
    eps: float = 1e-5
