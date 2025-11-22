"""TPU-optimized selective scan kernel for Mamba using Pallas Mosaic backend.

Targets TPU v5e and later with vectorized operations and VMEM scratch memory.
Implements the selective SSM recurrence with chunked processing for efficient
parallel training and autoregressive inference.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.lib import xla_bridge

try:  # pragma: no cover - optional TPU-specific dependency
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]
    pltpu = None  # type: ignore[assignment]

from .base import GridConfig, KernelMode, SelectiveKernelProtocol
from .mamba_reference import MambaKernelInputs, MambaKernelParams, MambaKernelState

# TPU architecture constants
NUM_LANES = 128  # Vector width for TPU v5e
MIN_BLOCK_SIZE = 128  # Minimum efficient tile size

Array = jax.Array


def _next_power_of_two(value: int) -> int:
    """Round up to next power of two."""
    return 1 << (value - 1).bit_length()


def _check_tpu_available() -> Tuple[bool, str]:
    """Check TPU availability and ensure libtpu build is fresh enough."""

    if pl is None or pltpu is None:
        return False, "Pallas TPU backend not installed"

    try:
        gpu_devices = jax.devices()
    except Exception as exc:  # pragma: no cover - defensive guard
        return False, f"Unable to query JAX devices: {exc}"

    tpu_devices = [device for device in gpu_devices if device.platform == "tpu"]
    if not tpu_devices:
        return False, "No TPU devices detected"

    # Check libtpu build freshness (Pallas TPU requires a recent runtime)
    try:
        backend = xla_bridge.get_backend("tpu")
        platform_version = getattr(backend, "platform_version", "").strip()
    except Exception as exc:  # pragma: no cover
        return False, f"Unable to query TPU backend version: {exc}"

    if platform_version:
        match = re.search(r"Built on (\w{3}) (\d{1,2}) (\d{4})", platform_version)
        if match:
            month_str, day_str, year_str = match.groups()
            try:
                build_date = datetime.strptime(
                    f"{month_str} {day_str} {year_str}", "%b %d %Y"
                ).date()
                now = datetime.now(timezone.utc).date()
                if (now - build_date).days > 31:
                    return (
                        False,
                        "TPU runtime is older than 31 days; upgrade libtpu ("
                        f"build {build_date.isoformat()}).",
                    )
            except ValueError:  # pragma: no cover - unexpected format
                pass

    return True, ""


TPU_AVAILABLE, TPU_AVAILABILITY_MESSAGE = _check_tpu_available()


def _mamba_tpu_kernel(
    hidden_ref,      # [batch, intermediate, seq_len]
    delta_ref,       # [batch, intermediate, seq_len]
    gate_ref,        # [batch, intermediate, seq_len]
    B_ref,           # [batch, seq_len, state_size]
    C_ref,           # [batch, seq_len, state_size]
    a_ref,           # [intermediate, state_size]
    d_ref,           # [intermediate]
    state_ref,       # [batch, intermediate, state_size]
    out_ref,         # [batch, intermediate, seq_len]
    state_out_ref,   # [batch, intermediate, state_size]
    acc_scratch_ref, # VMEM scratch for accumulator
):
    """TPU kernel for selective scan with vectorized operations.
    
    Grid: (batch_size, intermediate_size)
    Each program processes one (batch, channel) pair across the entire sequence.
    
    Uses VMEM scratch memory for efficient accumulation and TPU vector operations
    for 128-lane parallelism.
    """
    batch_idx = pl.program_id(axis=0)
    channel_idx = pl.program_id(axis=1)
    
    seq_len = hidden_ref.shape[2]
    state_size = a_ref.shape[1]
    
    # Load initial state from HBM to VMEM scratch
    ssm_state = state_ref[batch_idx, channel_idx, :]  # [state_size]
    acc_scratch_ref[:] = ssm_state  # Copy to VMEM for fast updates
    
    # Load channel-specific parameters (constant across sequence)
    a = a_ref[channel_idx, :]       # [state_size]
    d_scalar = d_ref[channel_idx]   # scalar
    
    # Process sequence sequentially (recurrent dependency)
    # Note: seq_len should be aligned to NUM_LANES for vectorization efficiency
    def scan_step(t, _):
        """Single timestep of selective SSM."""
        # Load timestep inputs from HBM
        hidden_t = hidden_ref[batch_idx, channel_idx, t]  # scalar
        delta_t = delta_ref[batch_idx, channel_idx, t]    # scalar
        gate_t = gate_ref[batch_idx, channel_idx, t]      # scalar
        B_t = B_ref[batch_idx, t, :]                       # [state_size]
        C_t = C_ref[batch_idx, t, :]                       # [state_size]
        
        # Load current state from VMEM
        current_state = acc_scratch_ref[:]  # [state_size]
        
        # Selective scan recurrence (vectorized over state_size)
        discrete_A = jnp.exp(a * delta_t)           # [state_size]
        discrete_B = delta_t * B_t                  # [state_size]
        deltaB_u = discrete_B * hidden_t            # [state_size]
        new_state = discrete_A * current_state + deltaB_u  # [state_size]
        
        # Write updated state back to VMEM
        acc_scratch_ref[:] = new_state
        
        # Output projection (manual inner product for TPU efficiency)
        # Use sum instead of dot to avoid matmul overhead for small vectors
        y = jnp.sum(new_state * C_t)  # scalar
        y = y + d_scalar * hidden_t   # scalar
        y = y * gate_t                # scalar
        
        # Write output to HBM
        out_ref[batch_idx, channel_idx, t] = y
        
        return None
    
    # Sequential scan over sequence dimension
    # unroll=False for long sequences to avoid code size explosion
    lax.fori_loop(0, seq_len, scan_step, None, unroll=False)
    
    # Write final state from VMEM back to HBM
    state_out_ref[batch_idx, channel_idx, :] = acc_scratch_ref[:]


class MambaTPUKernel(SelectiveKernelProtocol):
    """TPU-accelerated selective scan kernel using Pallas Mosaic backend.
    
    Optimized for TPU v5e with:
    - VMEM scratch memory for state accumulation (avoids HBM round-trips)
    - Vectorized operations over 128-lane vector units
    - Sequential processing for recurrent state dependencies
    - Power-of-two padding for efficient memory access
    """

    def __init__(self, *, mode: KernelMode = KernelMode.CHUNK, dtype: jnp.dtype = jnp.float32):
        if pl is None or pltpu is None:
            raise ImportError(
                "jax.experimental.pallas and pallas.tpu are required for MambaTPUKernel"
            )
        if not TPU_AVAILABLE:
            reason = TPU_AVAILABILITY_MESSAGE or "TPU backend unavailable"
            raise RuntimeError(
                "MambaTPUKernel requires a TPU backend: "
                f"{reason}"
            )
        self.mode = mode
        self.dtype = dtype

    def _pad_axis(self, tensor: Array, pad: int, axis: int = -1) -> Array:
        """Pad tensor along specified axis with zeros."""
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
        """Execute selective scan on TPU with chunked processing.
        
        Args:
            params: Kernel parameters (a_log, d)
            inputs: Input tensors (hidden, delta, gate, B, C)
            state: Initial SSM state (or None for zero initialization)
            chunk_size: Chunk size for processing (used for padding alignment)
        
        Returns:
            (outputs, final_state): Processed sequence and updated SSM state
        """
        # Cast inputs to kernel dtype
        hidden = inputs.hidden.astype(self.dtype)
        delta = inputs.delta.astype(self.dtype)
        gate = inputs.gate.astype(self.dtype)
        B = inputs.B.astype(self.dtype)
        C = inputs.C.astype(self.dtype)

        batch_size, intermediate_size, seq_len = hidden.shape
        ssm_state_dim = params.a_log.shape[-1]
        chunk_size = max(1, chunk_size)
        
        # Pad sequence to chunk_size multiple
        chunk_pad = (-(seq_len)) % max(1, chunk_size)
        hidden = self._pad_axis(hidden, chunk_pad, axis=-1)
        delta = self._pad_axis(delta, chunk_pad, axis=-1)
        gate = self._pad_axis(gate, chunk_pad, axis=-1)
        B = self._pad_axis(B, chunk_pad, axis=1)
        C = self._pad_axis(C, chunk_pad, axis=1)

        padded_len = hidden.shape[-1]
        
        # Pad state dimension to NUM_LANES for vectorization efficiency
        # TPU vector units work best with multiples of 128
        power_two_state = max(NUM_LANES, _next_power_of_two(ssm_state_dim))
        state_pad = power_two_state - ssm_state_dim

        # Initialize or pad state
        if state is None:
            ssm_state = jnp.zeros((batch_size, intermediate_size, power_two_state), dtype=self.dtype)
        else:
            ssm_state = state.ssm.astype(self.dtype)
            if state_pad:
                ssm_state = self._pad_axis(ssm_state, state_pad, axis=-1)

        # Pad B, C to match state dimension
        if state_pad:
            B = self._pad_axis(B, state_pad, axis=2)
            C = self._pad_axis(C, state_pad, axis=2)

        # Handle empty sequence edge case
        if padded_len == 0:
            trimmed_state = ssm_state[:, :, :ssm_state_dim]
            return hidden[:, :, :0], MambaKernelState(ssm=trimmed_state)

        # Pad sequence to NUM_LANES for vectorization
        # TPU requires alignment for efficient memory access
        aligned_len = ((padded_len + NUM_LANES - 1) // NUM_LANES) * NUM_LANES
        extra_pad = aligned_len - padded_len
        if extra_pad:
            hidden = self._pad_axis(hidden, extra_pad, axis=-1)
            delta = self._pad_axis(delta, extra_pad, axis=-1)
            gate = self._pad_axis(gate, extra_pad, axis=-1)
            B = self._pad_axis(B, extra_pad, axis=1)
            C = self._pad_axis(C, extra_pad, axis=1)
            padded_len = aligned_len

        # Prepare parameters
        a = -jnp.exp(params.a_log.astype(self.dtype))
        if state_pad:
            a = self._pad_axis(a, state_pad, axis=-1)
        d = params.d.astype(self.dtype)

        # Define output shapes
        out_shape = (
            jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
            jax.ShapeDtypeStruct(ssm_state.shape, ssm_state.dtype),
        )

        # Define VMEM scratch memory for state accumulation
        # This avoids repeated HBM reads/writes during sequence processing
        scratch_shapes = [
            pltpu.VMEM((power_two_state,), self.dtype),  # State accumulator
        ]

        # Invoke TPU kernel with Mosaic backend
        # Grid: (batch_size, intermediate_size) - one program per (batch, channel)
        # dimension_semantics: both dimensions are parallel (independent work)
        outputs, final_state = pl.pallas_call(
            _mamba_tpu_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,  # No metadata to prefetch
                grid=(batch_size, intermediate_size),
                in_specs=[
                    pl.BlockSpec((1, 1, padded_len), lambda b, c: (b, c, 0)),  # hidden
                    pl.BlockSpec((1, 1, padded_len), lambda b, c: (b, c, 0)),  # delta
                    pl.BlockSpec((1, 1, padded_len), lambda b, c: (b, c, 0)),  # gate
                    pl.BlockSpec((1, padded_len, power_two_state), lambda b, c: (b, 0, 0)),  # B
                    pl.BlockSpec((1, padded_len, power_two_state), lambda b, c: (b, 0, 0)),  # C
                    pl.BlockSpec((1, power_two_state), lambda b, c: (c, 0)),   # a
                    pl.BlockSpec((1,), lambda b, c: (c,)),                     # d
                    pl.BlockSpec((1, 1, power_two_state), lambda b, c: (b, c, 0)),  # state_in
                ],
                out_specs=[
                    pl.BlockSpec((1, 1, padded_len), lambda b, c: (b, c, 0)),  # out
                    pl.BlockSpec((1, 1, power_two_state), lambda b, c: (b, c, 0)),  # state_out
                ],
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # Both grid dimensions are parallel (independent programs)
                dimension_semantics=("parallel", "parallel"),
            ),
        )(hidden, delta, gate, B, C, a, d, ssm_state)

        # Trim padding from outputs
        outputs = outputs[:, :, :seq_len]
        trimmed_state = final_state[:, :, :ssm_state_dim]
        return outputs, MambaKernelState(ssm=trimmed_state)

    def forward_recurrent(
        self,
        params: MambaKernelParams,
        inputs: MambaKernelInputs,
        state: Optional[MambaKernelState],
    ) -> tuple[Array, MambaKernelState]:
        """Single-step recurrent mode (autoregressive inference).
        
        Delegates to forward_chunk with chunk_size=1 for simplicity.
        TPU kernel handles arbitrary sequence lengths efficiently.
        """
        return self.forward_chunk(params, inputs, state, chunk_size=1)

    def get_grid_config(
        self,
        *,
        batch_size: int,
        seq_len: int,
        feature_dim: int,
    ) -> GridConfig:
        """Return grid configuration for kernel launch.
        
        TPU kernel uses 2D grid over (batch, channels) with full sequence
        processed per program.
        """
        return GridConfig(
            block_shape=(NUM_LANES,),  # Vector width
            num_programs=(batch_size, feature_dim)
        )
