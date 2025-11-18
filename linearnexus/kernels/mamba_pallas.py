"""Pallas-based selective scan kernel for Mamba layers."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

try:  # pragma: no cover - optional dependency in CPU-only envs
    from jax.experimental import pallas as pl
except ImportError:  # pragma: no cover - surfaced when Pallas is unavailable
    pl = None  # type: ignore[assignment]

from .base import GridConfig, KernelMode, SelectiveKernelProtocol
from .mamba_reference import MambaKernelInputs, MambaKernelParams, MambaKernelState

PALLAS_AVAILABLE: bool = pl is not None

Array = jax.Array


def _mamba_pallas_kernel(
    hidden_ref,
    delta_ref,
    gate_ref,
    B_ref,
    C_ref,
    a_ref,
    d_ref,
    state_ref,
    out_ref,
    state_out_ref,
):
    """Single-program selective scan executed over one (batch, channel) pair."""

    batch_idx = pl.program_id(axis=0)
    channel_idx = pl.program_id(axis=1)

    hidden = hidden_ref[batch_idx, channel_idx, :]
    delta = delta_ref[batch_idx, channel_idx, :]
    gate = gate_ref[batch_idx, channel_idx, :]
    state = state_ref[batch_idx, channel_idx, :]
    B = B_ref[batch_idx, :, :]
    C = C_ref[batch_idx, :, :]
    a = a_ref[channel_idx, :]
    d = d_ref[channel_idx]

    def step(carry: Array, inputs: Tuple[Array, Array, Array, Array, Array]):
        ssm_state = carry
        hidden_t, delta_t, gate_t, B_t, C_t = inputs
        discrete_A = jnp.exp(a * delta_t)
        discrete_B = delta_t * B_t
        deltaB_u = discrete_B * hidden_t
        ssm_state = discrete_A * ssm_state + deltaB_u
        y = jnp.dot(ssm_state, C_t)
        y = y + d * hidden_t
        y = y * gate_t
        return ssm_state, y

    scan_inputs = (hidden, delta, gate, B, C)
    final_state, outputs = jax.lax.scan(step, state, scan_inputs)

    out_ref[batch_idx, channel_idx, :] = outputs
    state_out_ref[batch_idx, channel_idx, :] = final_state


class MambaPallasKernel(SelectiveKernelProtocol):
    """Pallas-accelerated selective scan kernel."""

    def __init__(self, *, mode: KernelMode = KernelMode.CHUNK, dtype: jnp.dtype = jnp.float32):
        if pl is None:
            raise ImportError("jax.experimental.pallas is required to instantiate MambaPallasKernel")
        self.mode = mode
        self.dtype = dtype

    def _pad_sequence(self, tensor: Array, pad: int, axis: int = -1) -> Array:
        if pad == 0:
            return tensor
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[axis] = (0, pad)
        return jnp.pad(tensor, pad_width)

    def forward_chunk(
        self,
        params: MambaKernelParams,
        inputs: MambaKernelInputs,
        state: Optional[MambaKernelState],
        *,
        chunk_size: int,
    ) -> tuple[Array, MambaKernelState]:
        hidden = inputs.hidden.astype(self.dtype)
        delta = inputs.delta.astype(self.dtype)
        gate = inputs.gate.astype(self.dtype)
        B = inputs.B.astype(self.dtype)
        C = inputs.C.astype(self.dtype)

        batch_size, intermediate_size, seq_len = hidden.shape
        ssm_state_dim = params.a_log.shape[-1]
        chunk_size = max(1, chunk_size)
        pad = (-(seq_len)) % chunk_size

        hidden = self._pad_sequence(hidden, pad, axis=-1)
        delta = self._pad_sequence(delta, pad, axis=-1)
        gate = self._pad_sequence(gate, pad, axis=-1)
        B = self._pad_sequence(B, pad, axis=1)
        C = self._pad_sequence(C, pad, axis=1)

        padded_len = hidden.shape[-1]
        if state is None:
            state = MambaKernelState.zeros(batch_size, intermediate_size, ssm_state_dim, self.dtype)
        ssm_state = state.ssm.astype(self.dtype)

        if padded_len == 0:
            return hidden[:, :, :0], state

        a = -jnp.exp(params.a_log.astype(self.dtype))
        d = params.d.astype(self.dtype)

        out_shape = (
            jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
            jax.ShapeDtypeStruct(ssm_state.shape, ssm_state.dtype),
        )

        outputs, final_state = pl.pallas_call(
            _mamba_pallas_kernel,
            out_shape=out_shape,
            grid=(batch_size, intermediate_size),
        )(hidden, delta, gate, B, C, a, d, ssm_state)

        outputs = outputs[:, :, :seq_len]
        return outputs, MambaKernelState(ssm=final_state)

    def forward_recurrent(
        self,
        params: MambaKernelParams,
        inputs: MambaKernelInputs,
        state: Optional[MambaKernelState],
    ) -> tuple[Array, MambaKernelState]:
        return self.forward_chunk(params, inputs, state, chunk_size=1)

    def get_grid_config(
        self,
        *,
        batch_size: int,
        seq_len: int,
        feature_dim: int,
    ) -> GridConfig:
        programs_seq = max(1, seq_len // 128)
        return GridConfig(block_shape=(feature_dim,), num_programs=(batch_size, feature_dim * programs_seq))
