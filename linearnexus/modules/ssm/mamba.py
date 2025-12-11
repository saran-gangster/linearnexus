"""Mamba block implementation with selective SSM.

Implements the Mamba architecture (Gu & Dao, 2023) which uses selective
state-space models for efficient sequence modeling. Key features:
- Input-dependent SSM parameters (selectivity)
- O(n) complexity for training
- Efficient recurrent form for generation

This is a pure JAX reference implementation. Custom Pallas kernels
for GPU/TPU acceleration are planned for future releases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.core import ConvState, depthwise_conv1d_causal
from linearnexus.modules.common import RMSNorm, MLP, get_norm, _get_activation

Array = jax.Array


# -----------------------------------------------------------------------------
# Mamba State
# -----------------------------------------------------------------------------

@dataclass
class MambaState:
    """State for Mamba block during autoregressive generation.
    
    Bundles the convolutional cache and SSM recurrent state.
    
    Attributes:
        conv_state: Buffer for causal convolution [batch, kernel-1, intermediate]
        ssm_state: SSM hidden state [batch, intermediate, state_size]
        position: Current sequence position
    """
    conv_state: Array   # [batch, kernel_size - 1, intermediate]
    ssm_state: Array    # [batch, intermediate, state_size]
    position: int
    
    @classmethod
    def zeros(
        cls,
        batch_size: int,
        intermediate_size: int,
        conv_kernel: int,
        state_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> "MambaState":
        """Create zero-initialized state."""
        conv_buffer_len = max(0, conv_kernel - 1)
        return cls(
            conv_state=jnp.zeros((batch_size, conv_buffer_len, intermediate_size), dtype=dtype),
            ssm_state=jnp.zeros((batch_size, intermediate_size, state_size), dtype=dtype),
            position=0,
        )


# -----------------------------------------------------------------------------
# Selective SSM Kernel (Pure JAX Reference)
# -----------------------------------------------------------------------------

def selective_scan_ref(
    hidden: Array,      # [batch, intermediate, seq]
    delta: Array,       # [batch, intermediate, seq]
    A: Array,           # [intermediate, state_size]
    B: Array,           # [batch, seq, state_size]
    C: Array,           # [batch, seq, state_size]
    D: Array,           # [intermediate]
    gate: Array,        # [batch, intermediate, seq]
    ssm_state: Optional[Array] = None,  # [batch, intermediate, state_size]
    chunk_size: int = 64,
) -> Tuple[Array, Array]:
    """Pure JAX reference implementation of selective scan.
    
    Implements the selective SSM recurrence:
        h_t = A_t * h_{t-1} + B_t * x_t
        y_t = C_t @ h_t + D * x_t
        
    Where A_t = exp(A * delta_t), B_t = delta_t * B_input_t
    
    Uses chunked processing for efficient parallel training while
    maintaining correct sequential state updates.
    
    Args:
        hidden: Input after convolution [batch, intermediate, seq]
        delta: Time step values [batch, intermediate, seq]
        A: Log-space SSM decay matrix [intermediate, state_size]
        B: Input-dependent SSM input matrix [batch, seq, state_size]
        C: Input-dependent SSM output matrix [batch, seq, state_size]
        D: Skip connection weights [intermediate]
        gate: Output gating [batch, intermediate, seq]
        ssm_state: Initial SSM state [batch, intermediate, state_size]
        chunk_size: Chunk size for processing
        
    Returns:
        Tuple of (output [batch, intermediate, seq], final_state [batch, intermediate, state_size])
    """
    batch_size, intermediate, seq_len = hidden.shape
    state_size = A.shape[1]
    
    # Initialize state if not provided
    if ssm_state is None:
        ssm_state = jnp.zeros((batch_size, intermediate, state_size), dtype=hidden.dtype)
    
    # Handle empty sequence
    if seq_len == 0:
        return hidden, ssm_state
    
    # Pad sequence to multiple of chunk_size
    num_chunks = math.ceil(seq_len / chunk_size)
    pad_len = num_chunks * chunk_size - seq_len
    
    if pad_len > 0:
        hidden = jnp.pad(hidden, ((0, 0), (0, 0), (0, pad_len)))
        delta = jnp.pad(delta, ((0, 0), (0, 0), (0, pad_len)))
        gate = jnp.pad(gate, ((0, 0), (0, 0), (0, pad_len)))
        B = jnp.pad(B, ((0, 0), (0, pad_len), (0, 0)))
        C = jnp.pad(C, ((0, 0), (0, pad_len), (0, 0)))
    
    # Reshape into chunks: [num_chunks, batch, intermediate/state, chunk_size]
    hidden_chunks = hidden.reshape(batch_size, intermediate, num_chunks, chunk_size).transpose(2, 0, 1, 3)
    delta_chunks = delta.reshape(batch_size, intermediate, num_chunks, chunk_size).transpose(2, 0, 1, 3)
    gate_chunks = gate.reshape(batch_size, intermediate, num_chunks, chunk_size).transpose(2, 0, 1, 3)
    B_chunks = B.reshape(batch_size, num_chunks, chunk_size, state_size).transpose(1, 0, 2, 3)
    C_chunks = C.reshape(batch_size, num_chunks, chunk_size, state_size).transpose(1, 0, 2, 3)
    
    def process_chunk(carry, chunk_inputs):
        """Process one chunk of the sequence."""
        current_state = carry
        hidden_c, delta_c, gate_c, B_c, C_c = chunk_inputs
        
        def step(state, step_inputs):
            """Single timestep of selective SSM."""
            h_t, delta_t, gate_t, B_t, C_t = step_inputs
            
            # Discretization: A_discrete = exp(A * delta)
            discrete_A = jnp.exp(A[None, :, :] * delta_t[:, :, None])  # [batch, intermediate, state]
            
            # Input matrix: B_discrete = delta * B
            discrete_B = delta_t[:, :, None] * B_t[:, None, :]  # [batch, intermediate, state]
            
            # State update: h = A_discrete * h + B_discrete * x
            deltaB_u = discrete_B * h_t[:, :, None]  # [batch, intermediate, state]
            new_state = discrete_A * state + deltaB_u  # [batch, intermediate, state]
            
            # Output: y = C @ h + D * x
            y = jnp.einsum("bis,bs->bi", new_state, C_t)  # [batch, intermediate]
            y = y + D[None, :] * h_t  # Skip connection
            y = y * gate_t  # Output gating
            
            return new_state, y
        
        # Prepare timestep inputs: [chunk_size, batch, ...]
        step_inputs = (
            hidden_c.transpose(2, 0, 1),   # [chunk, batch, intermediate]
            delta_c.transpose(2, 0, 1),    # [chunk, batch, intermediate]
            gate_c.transpose(2, 0, 1),     # [chunk, batch, intermediate]
            B_c.transpose(1, 0, 2),        # [chunk, batch, state]
            C_c.transpose(1, 0, 2),        # [chunk, batch, state]
        )
        
        final_state, outputs = jax.lax.scan(step, current_state, step_inputs)
        outputs = outputs.transpose(1, 2, 0)  # [batch, intermediate, chunk]
        
        return final_state, outputs
    
    chunk_inputs = (hidden_chunks, delta_chunks, gate_chunks, B_chunks, C_chunks)
    final_state, chunk_outputs = jax.lax.scan(process_chunk, ssm_state, chunk_inputs)
    
    # Reshape outputs: [num_chunks, batch, intermediate, chunk] -> [batch, intermediate, seq]
    outputs = chunk_outputs.transpose(1, 2, 0, 3).reshape(batch_size, intermediate, -1)
    outputs = outputs[:, :, :seq_len]  # Remove padding
    
    return outputs, final_state


# -----------------------------------------------------------------------------
# Mamba Block
# -----------------------------------------------------------------------------

class MambaBlock(nnx.Module):
    """Mamba block with selective state-space model.
    
    Architecture:
        x -> Norm -> InProj -> [hidden, gate]
        hidden -> Conv1D -> Activation -> SSM -> gate * SSM_out
        gate -> Activation
        out = OutProj(gated_output)
    
    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Internal SSM dimension (typically 2x hidden_size).
        state_size: SSM state dimension (typically 16).
        conv_kernel: Causal convolution kernel size (typically 4).
        time_step_rank: Rank of time-step projection (typically hidden_size).
        use_conv_bias: Whether to use bias in convolution.
        use_bias: Whether to use bias in linear projections.
        activation: Activation function ("silu", "gelu", etc.).
        chunk_size: Chunk size for training (trades memory for speed).
        norm_type: Normalization type ("rmsnorm" or "layernorm").
        norm_eps: Normalization epsilon.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        *,
        state_size: int = 16,
        conv_kernel: int = 4,
        time_step_rank: Optional[int] = None,
        use_conv_bias: bool = True,
        use_bias: bool = True,
        activation: str = "silu",
        chunk_size: int = 64,
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs,
    ):
        # Config defaults
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or (hidden_size * 2)
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.time_step_rank = time_step_rank or hidden_size
        self.chunk_size = chunk_size
        
        self.activation = _get_activation(activation)
        
        # Input normalization
        self.norm = get_norm(norm_type, hidden_size, eps=norm_eps, rngs=rngs)
        
        # Projections
        self.in_proj = nnx.Linear(
            hidden_size,
            self.intermediate_size * 2,  # hidden + gate
            use_bias=use_bias,
            rngs=rngs,
        )
        
        # x_proj: project to (time_step_rank, B, C)
        self.x_proj = nnx.Linear(
            self.intermediate_size,
            self.time_step_rank + state_size * 2,
            use_bias=False,
            rngs=rngs,
        )
        
        # Time-step projection
        self.dt_proj = nnx.Linear(
            self.time_step_rank,
            self.intermediate_size,
            use_bias=True,
            rngs=rngs,
        )
        
        # Output projection
        self.out_proj = nnx.Linear(
            self.intermediate_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )
        
        # Depthwise convolution weights
        conv_init_scale = 1.0 / math.sqrt(conv_kernel * self.intermediate_size)
        self.conv_weight = nnx.Param(
            jax.random.normal(rngs.params(), (conv_kernel, self.intermediate_size)) * conv_init_scale
        )
        if use_conv_bias:
            self.conv_bias = nnx.Param(jnp.zeros((self.intermediate_size,)))
        else:
            self.conv_bias = None
        
        # SSM parameters
        # A: initialized as arange (like in original Mamba)
        a_init = jnp.log(jnp.arange(1, state_size + 1, dtype=jnp.float32))
        self.A_log = nnx.Param(jnp.tile(a_init[None, :], (self.intermediate_size, 1)))
        
        # D: skip connection, initialized to ones
        self.D = nnx.Param(jnp.ones((self.intermediate_size,)))
    
    def __call__(
        self,
        x: Array,
        *,
        state: Optional[MambaState] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, Optional[MambaState]]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            state: Optional MambaState for generation.
            mask: Optional attention mask [batch, seq] (applied to input).
            mode: "chunk" for training, "recurrent" for generation.
                  If None, auto-selects based on sequence length.
            
        Returns:
            Tuple of (output [batch, seq, hidden_size], new_state).
        """
        batch_size, seq_len, _ = x.shape
        dtype = x.dtype
        
        # Auto-select mode
        if mode is None:
            mode = "chunk" if seq_len > 1 else "recurrent"
        
        # Initialize state if needed
        if state is None:
            state = MambaState.zeros(
                batch_size,
                self.intermediate_size,
                self.conv_kernel,
                self.state_size,
                dtype,
            )
        
        # Apply input mask if provided
        if mask is not None:
            x = x * mask[..., None]
        
        # Pre-norm
        x = self.norm(x)
        
        # Input projection: [batch, seq, hidden] -> [batch, seq, 2 * intermediate]
        projected = self.in_proj(x)
        hidden, gate = jnp.split(projected, 2, axis=-1)  # Each [batch, seq, intermediate]
        
        # Apply mask after projection too
        if mask is not None:
            hidden = hidden * mask[..., None]
            gate = gate * mask[..., None]
        
        # Causal convolution
        conv_out, new_conv_cache = depthwise_conv1d_causal(
            hidden,
            self.conv_weight.value,
            self.conv_bias.value if self.conv_bias is not None else None,
            cache=state.conv_state,
        )
        conv_out = self.activation(conv_out)  # [batch, seq, intermediate]
        
        # Apply mask after conv
        if mask is not None:
            conv_out = conv_out * mask[..., None]
        
        # Project to SSM parameters
        x_proj_out = self.x_proj(conv_out)  # [batch, seq, time_step_rank + 2*state_size]
        
        # Split into time_step, B, C
        dt_rank = self.time_step_rank
        state_size = self.state_size
        time_step = x_proj_out[..., :dt_rank]
        B = x_proj_out[..., dt_rank:dt_rank + state_size]
        C = x_proj_out[..., dt_rank + state_size:]
        
        # Time-step projection and softplus
        delta = jax.nn.softplus(self.dt_proj(time_step))  # [batch, seq, intermediate]
        
        # Prepare SSM inputs (kernel expects [batch, intermediate, seq])
        hidden_ssm = conv_out.transpose(0, 2, 1)
        gate_ssm = self.activation(gate).transpose(0, 2, 1)
        delta_ssm = delta.transpose(0, 2, 1)
        
        # SSM parameters
        A = -jnp.exp(self.A_log.value)  # [intermediate, state_size]
        
        # Run selective scan
        ssm_out, new_ssm_state = selective_scan_ref(
            hidden_ssm,
            delta_ssm,
            A,
            B,
            C,
            self.D.value,
            gate_ssm,
            ssm_state=state.ssm_state,
            chunk_size=self.chunk_size if mode == "chunk" else 1,
        )
        
        # Transpose back: [batch, intermediate, seq] -> [batch, seq, intermediate]
        ssm_out = ssm_out.transpose(0, 2, 1)
        
        # Output projection
        output = self.out_proj(ssm_out)
        
        # Update state
        new_state = MambaState(
            conv_state=new_conv_cache,
            ssm_state=new_ssm_state,
            position=state.position + seq_len,
        )
        
        return output, new_state
    
    def init_state(
        self,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> MambaState:
        """Initialize generation state."""
        return MambaState.zeros(
            batch_size,
            self.intermediate_size,
            self.conv_kernel,
            self.state_size,
            dtype,
        )
