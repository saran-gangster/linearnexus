"""Mamba2 block implementation with State Space Duality (SSD).

Implements Mamba2 architecture (Dao & Gu, 2024) which introduces:
- Multi-head SSM structure (parallel heads like attention)
- State Space Duality: O(N) recurrent form = O(N²) quadratic form
- Chunk-based parallel computation for training
- Grouped states (n_groups) for efficiency
- Gated RMSNorm after SSM

Key differences from Mamba1:
- A is per-head (not per-channel): [num_heads] vs [intermediate_size, state_size]
- B/C are computed after conv (not via separate x_proj)
- Multi-head structure: hidden_size = num_heads * head_dim
- Uses SSD algorithm instead of selective scan

This is a pure JAX reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.core import ConvState, depthwise_conv1d_causal
from linearnexus.modules.common import RMSNorm, get_norm, _get_activation

Array = jax.Array


# -----------------------------------------------------------------------------
# Mamba2 State
# -----------------------------------------------------------------------------

@dataclass
class Mamba2State:
    """State for Mamba2 block during autoregressive generation.
    
    Attributes:
        conv_state: Buffer for causal convolution [batch, kernel-1, conv_dim]
        ssm_state: SSM hidden state [batch, num_heads, head_dim, state_size]
        position: Current sequence position
    """
    conv_state: Array   # [batch, kernel_size - 1, conv_dim]
    ssm_state: Array    # [batch, num_heads, head_dim, state_size]
    position: int
    
    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel: int,
        conv_dim: int,
        state_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> "Mamba2State":
        """Create zero-initialized state."""
        conv_buffer_len = max(0, conv_kernel - 1)
        return cls(
            conv_state=jnp.zeros((batch_size, conv_buffer_len, conv_dim), dtype=dtype),
            ssm_state=jnp.zeros((batch_size, num_heads, head_dim, state_size), dtype=dtype),
            position=0,
        )


# -----------------------------------------------------------------------------
# Gated RMSNorm (for Mamba2)
# -----------------------------------------------------------------------------

class RMSNormGated(nnx.Module):
    """RMSNorm with gating, used in Mamba2.
    
    Applies: norm(x) * gate (where gate is SiLU-activated)
    
    Args:
        dim: Hidden dimension to normalize.
        eps: Small constant for numerical stability.
        norm_before_gate: If True, normalize before gating. If False, gate first.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
        rngs: nnx.Rngs,
    ):
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nnx.Param(jnp.ones((dim,)))
    
    def __call__(self, x: Array, gate: Array) -> Array:
        """Apply gated RMS normalization.
        
        Args:
            x: Input tensor of shape [..., dim]
            gate: Gate tensor of shape [..., dim]
            
        Returns:
            Gated normalized tensor of same shape.
        """
        # RMS normalize
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms * self.weight.value
        
        # Apply gating with SiLU
        gate_act = jax.nn.silu(gate)
        
        if self.norm_before_gate:
            return x_norm * gate_act
        else:
            return x_norm * gate_act


# -----------------------------------------------------------------------------
# SSD (State Space Duality) Kernel - Pure JAX Reference
# -----------------------------------------------------------------------------

def segment_sum(input_tensor: Array) -> Array:
    """Compute segment sum for SSD algorithm.
    
    Creates a lower triangular cumulative sum matrix for computing
    inter-position state decay factors.
    
    Args:
        input_tensor: Shape [..., chunk_size]
        
    Returns:
        Segment sum matrix of shape [..., chunk_size, chunk_size]
    """
    chunk_size = input_tensor.shape[-1]
    
    # Expand to [..., chunk_size, chunk_size]
    expanded = jnp.broadcast_to(
        input_tensor[..., None],
        input_tensor.shape + (chunk_size,)
    )
    
    # Create lower triangular mask (below diagonal)
    mask_below = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    masked = jnp.where(mask_below, expanded, 0.0)
    
    # Cumulative sum along the second-to-last axis
    cumsum = jnp.cumsum(masked, axis=-2)
    
    # Keep only lower triangular part (including diagonal)
    mask_lower = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)
    result = jnp.where(mask_lower, cumsum, -jnp.inf)
    
    return result


def ssd_chunk_scan(
    hidden_states: Array,   # [batch, seq, num_heads, head_dim]
    dt: Array,              # [batch, seq, num_heads]
    A: Array,               # [num_heads]
    B: Array,               # [batch, seq, n_groups, state_size]
    C: Array,               # [batch, seq, n_groups, state_size]
    D: Array,               # [num_heads]
    *,
    chunk_size: int = 256,
    n_groups: int = 1,
    ssm_state: Optional[Array] = None,  # [batch, num_heads, head_dim, state_size]
    dt_bias: Optional[Array] = None,    # [num_heads]
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
) -> Tuple[Array, Array]:
    """SSD (State Space Duality) chunk-based scan.
    
    Implements the chunk-parallel SSM computation from Mamba2.
    This uses the quadratic form within chunks and linear recurrence
    between chunks for efficient parallel training.
    
    Args:
        hidden_states: Input [batch, seq, num_heads, head_dim]
        dt: Time deltas [batch, seq, num_heads]
        A: Log-space decay (negative) [num_heads]
        B: SSM input matrix [batch, seq, n_groups, state_size]
        C: SSM output matrix [batch, seq, n_groups, state_size]
        D: Skip connection [num_heads]
        chunk_size: Size of chunks for parallel processing
        n_groups: Number of groups for B/C
        ssm_state: Initial SSM state
        dt_bias: Bias added to dt before softplus
        dt_min: Minimum dt value (for initialization scaling)
        dt_max: Maximum dt value (for initialization scaling)
        dt_limit: Clamp dt to this range after softplus
        
    Returns:
        Tuple of (output [batch, seq, num_heads, head_dim], 
                  final_state [batch, num_heads, head_dim, state_size])
    """
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    state_size = B.shape[-1]
    dtype = hidden_states.dtype
    
    # Apply dt_bias and softplus
    if dt_bias is not None:
        dt = dt + dt_bias[None, None, :]
    dt = jax.nn.softplus(dt)
    dt = jnp.clip(dt, dt_limit[0], dt_limit[1])
    
    # Expand B and C if using grouped states
    # [batch, seq, n_groups, state_size] -> [batch, seq, num_heads, state_size]
    heads_per_group = num_heads // n_groups
    B = jnp.repeat(B, heads_per_group, axis=2)
    C = jnp.repeat(C, heads_per_group, axis=2)
    
    # Initialize state if not provided
    if ssm_state is None:
        ssm_state = jnp.zeros((batch_size, num_heads, head_dim, state_size), dtype=dtype)
    
    # Handle empty sequence
    if seq_len == 0:
        output = jnp.zeros((batch_size, 0, num_heads, head_dim), dtype=dtype)
        return output, ssm_state
    
    # Pad to multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    
    def pad_along_seq(x):
        """Pad tensor along sequence axis (axis=1)."""
        if pad_size == 0:
            return x
        pad_width = [(0, 0)] * x.ndim
        pad_width[1] = (0, pad_size)
        return jnp.pad(x, pad_width, mode='constant', constant_values=0)
    
    # D residual (skip connection): [batch, seq, num_heads, head_dim]
    D_residual = hidden_states * D[None, None, :, None]
    D_residual = pad_along_seq(D_residual)
    
    # Discretize: x_discrete = x * dt
    hidden_discrete = hidden_states * dt[..., None]  # [batch, seq, num_heads, head_dim]
    hidden_discrete = pad_along_seq(hidden_discrete)
    
    # A * dt for decay computation
    A_dt = A[None, None, :] * dt  # [batch, seq, num_heads]
    A_dt = pad_along_seq(A_dt)
    
    B = pad_along_seq(B)
    C = pad_along_seq(C)
    
    padded_len = seq_len + pad_size
    num_chunks = padded_len // chunk_size
    
    # Reshape into chunks: [batch, num_chunks, chunk_size, ...]
    hidden_chunks = hidden_discrete.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)
    A_dt_chunks = A_dt.reshape(batch_size, num_chunks, chunk_size, num_heads)
    B_chunks = B.reshape(batch_size, num_chunks, chunk_size, num_heads, state_size)
    C_chunks = C.reshape(batch_size, num_chunks, chunk_size, num_heads, state_size)
    D_res_chunks = D_residual.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)
    
    # Permute A_dt for cumsum: [batch, num_chunks, num_heads, chunk_size]
    A_dt_perm = A_dt_chunks.transpose(0, 1, 3, 2)  # [batch, nc, heads, cs]
    A_cumsum = jnp.cumsum(A_dt_perm, axis=-1)  # [batch, nc, heads, cs]
    
    # 1. INTRA-CHUNK (diagonal blocks): quadratic attention-like computation
    # Compute L = exp(segment_sum(A)) for causal masking within chunk
    # segment_sum output: [batch, nc, heads, cs, cs]
    L = jnp.exp(segment_sum(A_dt_perm))  # [batch, nc, heads, cs, cs]
    
    # G = einsum over state dimension: C[l] @ B[s]^T
    # C_chunks: [batch, nc, cs, heads, state], B_chunks: [batch, nc, cs, heads, state]
    # Want G: [batch, nc, heads, cs_l, cs_s] where we contract over state
    # Use 'l' for query position, 'k' for key position, 'n' for state
    G = jnp.einsum('bclhn,bckhn->bhclk', C_chunks, B_chunks)  # [batch, heads, cs_l, cs_k]
    # Expand to include num_chunks dimension properly
    # Actually let's reshape: C_chunks is [batch, nc, cs, heads, state]
    # b=batch, c=num_chunks, l=chunk_pos (query), k=chunk_pos (key), h=heads, n=state
    G = jnp.einsum('bclhn,bckhn->bchlk', C_chunks, B_chunks)  # [batch, nc, heads, cs, cs]
    
    # M = G * L (apply causal mask via decay)
    M = G * L  # [batch, nc, heads, cs, cs]
    
    # Y_diag = M @ hidden (intra-chunk output)
    # M: [batch, nc, heads, cs_l, cs_k], hidden_chunks: [batch, nc, cs, heads, dim]
    # We need hidden as [batch, nc, heads, cs, dim]
    hidden_perm = hidden_chunks.transpose(0, 1, 3, 2, 4)  # [batch, nc, heads, cs, dim]
    # [batch, nc, heads, cs_l, cs_k] @ [batch, nc, heads, cs_k, dim] -> [batch, nc, heads, cs_l, dim]
    Y_diag = jnp.einsum('bchlk,bchkd->bchld', M, hidden_perm)  # [batch, nc, heads, cs, dim]
    
    # 2. INTER-CHUNK STATES: compute state at end of each chunk
    # decay_states = exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)  # [batch, nc, heads, cs]
    
    # B_decay = B * decay_states
    # B_chunks: [batch, nc, cs, heads, state], decay_states: [batch, nc, heads, cs]
    B_decay = B_chunks * decay_states.transpose(0, 1, 3, 2)[..., None]  # [batch, nc, cs, heads, state]
    
    # states = sum over chunk of (B_decay * hidden)
    # B_decay: [batch, nc, cs, heads, state], hidden_chunks: [batch, nc, cs, heads, dim]
    # Contract over chunk_size, result: [batch, nc, heads, dim, state]
    states = jnp.einsum('bnchs,bnchd->bnhds', B_decay, hidden_chunks)  # [batch, nc, heads, dim, state]
    
    # 3. CHUNK-TO-CHUNK RECURRENCE
    # Prepend initial state: [batch, 1, heads, dim, state]
    previous_states = ssm_state[:, None, :, :, :]  # [batch, 1, heads, dim, state]
    states_with_init = jnp.concatenate([previous_states, states], axis=1)  # [batch, nc+1, heads, dim, state]
    
    # Compute inter-chunk decay
    A_chunk_end = A_cumsum[..., -1]  # [batch, nc, heads]
    A_chunk_end_padded = jnp.pad(A_chunk_end, ((0, 0), (1, 0), (0, 0)))  # [batch, nc+1, heads]
    
    # segment_sum expects [..., chunk_size], output is [..., chunk_size, chunk_size]
    # Here chunk_size is nc+1
    decay_chunk = jnp.exp(segment_sum(A_chunk_end_padded.transpose(0, 2, 1)))  # [batch, heads, nc+1, nc+1]
    decay_chunk = decay_chunk.transpose(0, 2, 3, 1)  # [batch, nc+1, nc+1, heads]
    
    # new_states = decay_chunk @ states_with_init
    # [batch, nc+1, nc+1, heads] x [batch, nc+1, heads, dim, state] -> [batch, nc+1, heads, dim, state]
    new_states = jnp.einsum('bnmh,bmhds->bnhds', decay_chunk, states_with_init)
    
    # Extract states for each chunk (excluding the last which is final state)
    chunk_states = new_states[:, :-1, :, :, :]  # [batch, nc, heads, dim, state]
    final_ssm_state = new_states[:, -1, :, :, :]  # [batch, heads, dim, state]
    
    # 4. STATE -> OUTPUT (off-diagonal blocks)
    state_decay_out = jnp.exp(A_cumsum)  # [batch, nc, heads, cs]
    
    # Y_off = C @ chunk_states * decay
    # C_chunks: [batch, nc, cs, heads, state], chunk_states: [batch, nc, heads, dim, state]
    # Contract over state, result: [batch, nc, cs, heads, dim]
    Y_off = jnp.einsum('bnchs,bnhds->bnchd', C_chunks, chunk_states)
    # Apply decay: [batch, nc, cs, heads, dim] * [batch, nc, heads, cs] (transposed)
    Y_off = Y_off * state_decay_out.transpose(0, 1, 3, 2)[..., None]  # [batch, nc, cs, heads, dim]
    
    # 5. COMBINE
    # Y_diag is [batch, nc, heads, cs, dim], need to transpose to [batch, nc, cs, heads, dim]
    Y_diag = Y_diag.transpose(0, 1, 3, 2, 4)  # [batch, nc, cs, heads, dim]
    Y = Y_diag + Y_off + D_res_chunks  # [batch, nc, cs, heads, dim]
    
    # Reshape back: [batch, padded_seq, heads, dim]
    output = Y.reshape(batch_size, padded_len, num_heads, head_dim)
    
    # Remove padding
    output = output[:, :seq_len, :, :]
    
    return output, final_ssm_state


def ssd_recurrent_step(
    hidden_state: Array,     # [batch, num_heads, head_dim]
    dt: Array,               # [batch, num_heads]
    A: Array,                # [num_heads]
    B: Array,                # [batch, n_groups, state_size]
    C: Array,                # [batch, n_groups, state_size]
    D: Array,                # [num_heads]
    ssm_state: Array,        # [batch, num_heads, head_dim, state_size]
    *,
    n_groups: int = 1,
    dt_bias: Optional[Array] = None,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
) -> Tuple[Array, Array]:
    """Single-step recurrent update for Mamba2.
    
    Used during autoregressive generation (token-by-token).
    
    Args:
        hidden_state: Input [batch, num_heads, head_dim]
        dt: Time delta [batch, num_heads]
        A: Decay parameter [num_heads]
        B: SSM input [batch, n_groups, state_size]
        C: SSM output [batch, n_groups, state_size]
        D: Skip connection [num_heads]
        ssm_state: Current state [batch, num_heads, head_dim, state_size]
        n_groups: Number of groups
        dt_bias: Bias for dt
        dt_limit: Range to clamp dt
        
    Returns:
        Tuple of (output [batch, num_heads, head_dim], new_state)
    """
    batch_size, num_heads, head_dim = hidden_state.shape
    state_size = B.shape[-1]
    
    # Apply dt_bias and softplus
    if dt_bias is not None:
        dt = dt + dt_bias[None, :]
    dt = jax.nn.softplus(dt)
    dt = jnp.clip(dt, dt_limit[0], dt_limit[1])
    
    # Expand dt for head_dim: [batch, num_heads, head_dim]
    dt_expanded = dt[..., None].repeat(head_dim, axis=-1)
    
    # Expand B and C if using grouped states
    heads_per_group = num_heads // n_groups
    # [batch, n_groups, state_size] -> [batch, num_heads, state_size]
    B = jnp.repeat(B, heads_per_group, axis=1)
    C = jnp.repeat(C, heads_per_group, axis=1)
    
    # Discretization
    # dA = exp(A * dt): [batch, num_heads, head_dim, state_size]
    A_expanded = A[None, :, None, None]  # [1, heads, 1, 1]
    dt_for_A = dt[..., None, None]  # [batch, heads, 1, 1]
    dA = jnp.exp(A_expanded * dt_for_A)
    dA = jnp.broadcast_to(dA, (batch_size, num_heads, head_dim, state_size))
    
    # dB = dt * B: [batch, num_heads, head_dim, state_size]
    dB = dt_expanded[..., None] * B[:, :, None, :]
    
    # dBx = dB * x: [batch, num_heads, head_dim, state_size]
    dBx = dB * hidden_state[..., None]
    
    # State update: new_state = dA * state + dBx
    new_ssm_state = dA * ssm_state + dBx
    
    # Output: y = C @ state + D * x
    # [batch, heads, dim, state] @ [batch, heads, state] -> [batch, heads, dim]
    y = jnp.einsum('bhds,bhs->bhd', new_ssm_state, C)
    
    # D skip connection
    D_expanded = D[None, :, None]  # [1, heads, 1]
    y = y + D_expanded * hidden_state
    
    return y, new_ssm_state


# -----------------------------------------------------------------------------
# Mamba2 Block
# -----------------------------------------------------------------------------

class Mamba2Block(nnx.Module):
    """Mamba2 block with State Space Duality.
    
    Architecture:
        x -> Norm -> InProj -> [z, hidden_BC, dt]
        hidden_BC -> Conv1D -> Activation -> split(hidden, B, C)
        SSM(hidden, dt, A, B, C, D) -> gated_norm(*, z) -> OutProj
    
    Key differences from Mamba1:
    - Multi-head structure (num_heads * head_dim = intermediate_size)
    - A is per-head, not per-channel
    - B/C come from conv output, not separate x_proj
    - Uses SSD algorithm for chunk-parallel training
    
    Args:
        hidden_size: Input/output dimension.
        num_heads: Number of SSM heads.
        head_dim: Dimension per head (intermediate = num_heads * head_dim).
        state_size: SSM state dimension (typically 128 for Mamba2).
        expand: Expansion factor for intermediate size.
        n_groups: Number of groups for B/C (GQA-style efficiency).
        conv_kernel: Causal convolution kernel size.
        use_conv_bias: Whether to use bias in convolution.
        activation: Activation function.
        chunk_size: Chunk size for training.
        time_step_limit: Range to clamp dt.
        use_bias: Whether to use bias in linear projections.
        norm_type: Normalization type.
        norm_eps: Normalization epsilon.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        head_dim: int = 64,
        state_size: int = 128,
        expand: int = 2,
        n_groups: int = 1,
        conv_kernel: int = 4,
        use_conv_bias: bool = False,
        activation: str = "silu",
        chunk_size: int = 256,
        time_step_limit: Tuple[float, float] = (0.0, float("inf")),
        use_bias: bool = True,
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        norm_eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        # Config
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.intermediate_size = num_heads * head_dim
        self.expand = expand
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        
        self.activation = _get_activation(activation)
        self.activation_name = activation
        
        # Convolution dimension: intermediate + 2 * n_groups * state_size (for B and C)
        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size
        
        # Input normalization
        self.norm = get_norm(norm_type, hidden_size, eps=norm_eps, rngs=rngs)
        
        # Input projection outputs:
        # - gate (z): intermediate_size
        # - hidden_B_C: conv_dim (goes through conv, then split)
        # - dt: num_heads
        projection_size = self.intermediate_size + self.conv_dim + num_heads
        self.in_proj = nnx.Linear(
            hidden_size,
            projection_size,
            use_bias=use_bias,
            rngs=rngs,
        )
        
        # Depthwise convolution weights for hidden_B_C
        conv_init_scale = 1.0 / math.sqrt(conv_kernel * self.conv_dim)
        self.conv_weight = nnx.Param(
            jax.random.normal(rngs.params(), (conv_kernel, self.conv_dim)) * conv_init_scale
        )
        if use_conv_bias:
            self.conv_bias = nnx.Param(jnp.zeros((self.conv_dim,)))
        else:
            self.conv_bias = None
        
        # SSM parameters
        # A: per-head decay, initialized as log(arange)
        # Note: A is stored as log and made negative during computation
        A_init = jnp.log(jnp.arange(1, num_heads + 1, dtype=jnp.float32))
        self.A_log = nnx.Param(A_init)
        
        # dt_bias: per-head bias for time step
        self.dt_bias = nnx.Param(jnp.ones((num_heads,)))
        
        # D: skip connection, per-head
        self.D = nnx.Param(jnp.ones((num_heads,)))
        
        # Gated RMSNorm (Mamba2 specific)
        self.ssm_norm = RMSNormGated(
            self.intermediate_size,
            eps=norm_eps,
            norm_before_gate=False,
            rngs=rngs,
        )
        
        # Output projection
        self.out_proj = nnx.Linear(
            self.intermediate_size,
            hidden_size,
            use_bias=use_bias,
            rngs=rngs,
        )
    
    def __call__(
        self,
        x: Array,
        *,
        state: Optional[Mamba2State] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, Optional[Mamba2State]]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            state: Optional Mamba2State for generation.
            mask: Optional attention mask [batch, seq].
            mode: "chunk" for training, "recurrent" for generation.
            
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
            state = Mamba2State.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.conv_kernel,
                self.conv_dim,
                self.state_size,
                dtype,
            )
        
        # Apply input mask if provided
        if mask is not None:
            x = x * mask[..., None]
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # Input projection: [batch, seq, hidden] -> [batch, seq, proj_size]
        projected = self.in_proj(x_norm)
        
        # Split into gate, hidden_B_C, and dt
        gate = projected[..., :self.intermediate_size]
        hidden_B_C = projected[..., self.intermediate_size:self.intermediate_size + self.conv_dim]
        dt = projected[..., -self.num_heads:]
        
        # Causal convolution on hidden_B_C
        conv_out, new_conv_cache = depthwise_conv1d_causal(
            hidden_B_C,
            self.conv_weight.value,
            self.conv_bias.value if self.conv_bias is not None else None,
            cache=state.conv_state,
        )
        conv_out = self.activation(conv_out)  # [batch, seq, conv_dim]
        
        # Apply mask after conv
        if mask is not None:
            conv_out = conv_out * mask[..., None]
        
        # Split conv output into hidden, B, C
        groups_state_size = self.n_groups * self.state_size
        hidden_states = conv_out[..., :self.intermediate_size]
        B = conv_out[..., self.intermediate_size:self.intermediate_size + groups_state_size]
        C = conv_out[..., self.intermediate_size + groups_state_size:]
        
        # Reshape for SSM
        # hidden: [batch, seq, intermediate] -> [batch, seq, num_heads, head_dim]
        hidden_states = hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # B, C: [batch, seq, n_groups * state_size] -> [batch, seq, n_groups, state_size]
        B = B.reshape(batch_size, seq_len, self.n_groups, self.state_size)
        C = C.reshape(batch_size, seq_len, self.n_groups, self.state_size)
        
        # SSM parameters
        A = -jnp.exp(self.A_log.value)  # [num_heads], negative
        
        if mode == "chunk":
            # Chunk-parallel SSD
            ssm_out, new_ssm_state = ssd_chunk_scan(
                hidden_states,
                dt,
                A,
                B,
                C,
                self.D.value,
                chunk_size=self.chunk_size,
                n_groups=self.n_groups,
                ssm_state=state.ssm_state,
                dt_bias=self.dt_bias.value,
                dt_limit=self.time_step_limit,
            )
        else:
            # Recurrent mode (token-by-token)
            # Process one step at a time
            outputs = []
            current_state = state.ssm_state
            
            for t in range(seq_len):
                h_t = hidden_states[:, t, :, :]  # [batch, heads, dim]
                dt_t = dt[:, t, :]  # [batch, heads]
                B_t = B[:, t, :, :]  # [batch, groups, state]
                C_t = C[:, t, :, :]  # [batch, groups, state]
                
                out_t, current_state = ssd_recurrent_step(
                    h_t, dt_t, A, B_t, C_t, self.D.value,
                    current_state,
                    n_groups=self.n_groups,
                    dt_bias=self.dt_bias.value,
                    dt_limit=self.time_step_limit,
                )
                outputs.append(out_t)
            
            ssm_out = jnp.stack(outputs, axis=1)  # [batch, seq, heads, dim]
            new_ssm_state = current_state
        
        # Reshape SSM output: [batch, seq, heads, dim] -> [batch, seq, intermediate]
        ssm_out = ssm_out.reshape(batch_size, seq_len, self.intermediate_size)
        
        # Apply gated normalization
        normed_out = self.ssm_norm(ssm_out, gate)
        
        # Output projection
        output = self.out_proj(normed_out.astype(dtype))
        
        # Update state
        new_state = Mamba2State(
            conv_state=new_conv_cache,
            ssm_state=new_ssm_state,
            position=state.position + seq_len,
        )
        
        return output, new_state
    
    def init_state(
        self,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> Mamba2State:
        """Initialize generation state."""
        return Mamba2State.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.conv_kernel,
            self.conv_dim,
            self.state_size,
            dtype,
        )
