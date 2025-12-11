"""Reference selective scan kernel for Mamba-style layers.

This is a pure JAX implementation used as a correctness reference.
It provides both chunk-based (training) and recurrent (inference) modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from ..core.mode import KernelMode


@dataclass
class MambaKernelParams:
    """Kernel parameters derived from the layer weights.
    
    Attributes:
        a_log: Log of the diagonal state matrix, shape [intermediate, ssm_state].
        d: Skip connection weights, shape [intermediate].
    """

    a_log: jax.Array  # [intermediate, ssm_state]
    d: jax.Array  # [intermediate]


@dataclass
class MambaKernelInputs:
    """Inputs required by the selective scan kernel.
    
    Attributes:
        hidden: Input after in-projection, shape [batch, intermediate, seq].
        delta: Time step parameter, shape [batch, intermediate, seq].
        B: Input-dependent B matrix, shape [batch, seq, ssm_state].
        C: Input-dependent C matrix, shape [batch, seq, ssm_state].
        gate: Gating values, shape [batch, intermediate, seq].
    """

    hidden: jax.Array  # [batch, intermediate, seq]
    delta: jax.Array  # [batch, intermediate, seq]
    B: jax.Array  # [batch, seq, ssm_state]
    C: jax.Array  # [batch, seq, ssm_state]
    gate: jax.Array  # [batch, intermediate, seq]


@dataclass
class MambaKernelState:
    """State carried between invocations for recurrent decoding.
    
    Attributes:
        ssm: The SSM hidden state, shape [batch, intermediate, ssm_state].
    """

    ssm: jax.Array  # [batch, intermediate, ssm_state]

    @classmethod
    def zeros(cls, batch_size: int, intermediate: int, ssm_state: int, dtype: jnp.dtype) -> "MambaKernelState":
        """Create zero-initialized state."""
        return cls(
            ssm=jnp.zeros((batch_size, intermediate, ssm_state), dtype=dtype),
        )


class MambaReferenceKernel:
    """Pure JAX selective scan kernel used as a correctness reference.
    
    This kernel implements the selective scan algorithm from the Mamba paper
    using standard JAX operations (lax.scan). It supports both chunk-based
    processing for parallel training and token-by-token recurrent inference.
    
    Args:
        mode: Default processing mode (CHUNK or RECURRENT).
        dtype: Computation dtype (default: float32).
    """

    def __init__(self, *, mode: KernelMode = KernelMode.CHUNK, dtype: jnp.dtype = jnp.float32):
        self.mode = mode
        self.dtype = dtype

    def forward_chunk(
        self,
        params: MambaKernelParams,
        inputs: MambaKernelInputs,
        state: Optional[MambaKernelState],
        *,
        chunk_size: int,
    ) -> tuple[jax.Array, MambaKernelState]:
        hidden = inputs.hidden.astype(self.dtype)
        delta = inputs.delta.astype(self.dtype)
        gate = inputs.gate.astype(self.dtype)
        B = inputs.B.astype(self.dtype)
        C = inputs.C.astype(self.dtype)

        batch_size, intermediate, seq_len = hidden.shape
        ssm_state_dim = params.a_log.shape[-1]
        state = state or MambaKernelState.zeros(batch_size, intermediate, ssm_state_dim, self.dtype)

        num_chunks = math.ceil(seq_len / chunk_size)
        pad = num_chunks * chunk_size - seq_len
        if pad > 0:
            hidden = jnp.pad(hidden, ((0, 0), (0, 0), (0, pad)))
            delta = jnp.pad(delta, ((0, 0), (0, 0), (0, pad)))
            gate = jnp.pad(gate, ((0, 0), (0, 0), (0, pad)))
            B = jnp.pad(B, ((0, 0), (0, pad), (0, 0)))
            C = jnp.pad(C, ((0, 0), (0, pad), (0, 0)))

        hidden_chunks = (
            hidden.reshape(batch_size, intermediate, num_chunks, chunk_size)
            .transpose(2, 0, 1, 3)
        )
        delta_chunks = (
            delta.reshape(batch_size, intermediate, num_chunks, chunk_size)
            .transpose(2, 0, 1, 3)
        )
        gate_chunks = (
            gate.reshape(batch_size, intermediate, num_chunks, chunk_size)
            .transpose(2, 0, 1, 3)
        )
        B_chunks = (
            B.reshape(batch_size, num_chunks, chunk_size, ssm_state_dim)
            .transpose(1, 0, 2, 3)
        )
        C_chunks = (
            C.reshape(batch_size, num_chunks, chunk_size, ssm_state_dim)
            .transpose(1, 0, 2, 3)
        )

        a = -jnp.exp(params.a_log.astype(self.dtype))  # [intermediate, ssm_state]
        d = params.d.astype(self.dtype)

        def chunk_scan(carry, chunk_inputs):
            ssm_state = carry
            hidden_chunk, delta_chunk, gate_chunk, B_chunk, C_chunk = chunk_inputs

            def step(carry_inner, step_inputs):
                ssm_state_inner = carry_inner
                hidden_t, delta_t, gate_t, B_t, C_t = step_inputs
                discrete_A = jnp.exp(a[None, :, :] * delta_t[:, :, None])
                discrete_B = delta_t[:, :, None] * B_t[:, None, :]
                deltaB_u = discrete_B * hidden_t[:, :, None]
                ssm_state_inner = discrete_A * ssm_state_inner + deltaB_u
                y = jnp.einsum("bis,bs->bi", ssm_state_inner, C_t)
                y = y + d[None, :] * hidden_t
                y = y * gate_t
                return ssm_state_inner, y

            scan_inputs = (
                hidden_chunk.transpose(2, 0, 1),
                delta_chunk.transpose(2, 0, 1),
                gate_chunk.transpose(2, 0, 1),
                B_chunk.transpose(1, 0, 2),
                C_chunk.transpose(1, 0, 2),
            )
            carry_out, outputs = jax.lax.scan(
                step,
                ssm_state,
                scan_inputs,
            )
            outputs = outputs.transpose(1, 2, 0)
            return carry_out, outputs

        chunk_inputs = (
            hidden_chunks,
            delta_chunks,
            gate_chunks,
            B_chunks,
            C_chunks,
        )

        final_state, chunk_outputs = jax.lax.scan(
            chunk_scan,
            state.ssm,
            chunk_inputs,
        )
        outputs = (
            chunk_outputs.transpose(1, 2, 0, 3)
            .reshape(batch_size, intermediate, num_chunks * chunk_size)
        )
        outputs = outputs[:, :, :seq_len]
        return outputs, MambaKernelState(ssm=final_state)

    def forward_recurrent(
        self,
        params: MambaKernelParams,
        inputs: MambaKernelInputs,
        state: Optional[MambaKernelState],
    ) -> tuple[jax.Array, MambaKernelState]:
        outputs, new_state = self.forward_chunk(
            params,
            inputs,
            state,
            chunk_size=1,
        )
        return outputs, new_state
