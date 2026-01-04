"""
RWKV-6: Receptance Weighted Key Value with Matrix-Valued States

Paper: "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"
       https://arxiv.org/abs/2404.05892

Mathematical Foundation
=======================

RWKV-6 is a linear attention variant that uses a recurrent formulation with:
- Data-dependent decay (w) instead of fixed decay
- A "bonus" term (u) for the current token
- Token shifting for temporal context mixing

Core Recurrence (Token-by-Token)
--------------------------------
For each token t:
    1. Compute outer product: kv_t = k_t @ v_t^T
    2. Compute output:        o_t = (h + u * kv_t) @ q_t
    3. Update state:          h = h * exp(w_t) + kv_t

Where:
    - q_t, k_t: query and key vectors, shape [d_k]  
    - v_t: value vector, shape [d_v]
    - w_t: data-dependent decay (negative log-space), shape [d_k]
    - u: bonus term (learnable), shape [d_k]
    - h: state matrix, shape [d_k, d_v]

Key Components:
--------------
1. **Token Shift**: Temporal mixing via shifted features
   - delta = time_shift(x) - x
   - Used to compute data-dependent gates

2. **Time Mixing**: Main attention mechanism with:
   - Receptance (r), Key (k), Value (v), Gate (g) projections
   - Data-dependent decay (w) via low-rank projection
   - Bonus term (u) for first token importance

3. **Channel Mixing (FFN)**: Simple squared ReLU FFN
   - Similar to GLU but with sigmoid gating

Architecture Details:
--------------------
- Low-rank projections for time_maa_x (5 * low_rank_dim)
- Data-dependent decay via tanh activation
- GroupNorm on output before gating
- Short convolutions optional (but not default like FLA's implementation)

Reference Implementation
------------------------
Based on FLA's RWKV6Attention and the naive recurrent kernel.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import math
import jax
import jax.numpy as jnp
from jax import lax
import flax.nnx as nnx

from ...core.conv import depthwise_conv1d_causal
from ..common import RMSNorm


# =============================================================================
# State Cache
# =============================================================================


@dataclass
class RWKV6State:
    """Cache for RWKV6 autoregressive generation.

    Attributes:
        h: Recurrent state matrix [batch, num_heads, head_k_dim, head_v_dim]
        shift_state: Last token for time shift [batch, hidden_size]
    """

    h: jax.Array  # [batch, num_heads, head_k_dim, head_v_dim]
    shift_state: Optional[jax.Array] = None  # [batch, hidden_size]

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        hidden_size: Optional[int] = None,
    ) -> "RWKV6State":
        """Initialize empty state.

        Args:
            batch_size: Batch dimension
            num_heads: Number of attention heads
            head_k_dim: Key dimension per head
            head_v_dim: Value dimension per head
            hidden_size: Hidden size for shift state (optional)
        """
        h = jnp.zeros((batch_size, num_heads, head_k_dim, head_v_dim))
        shift_state = None
        if hidden_size is not None:
            shift_state = jnp.zeros((batch_size, hidden_size))

        return cls(h=h, shift_state=shift_state)


# =============================================================================
# Core RWKV6 Operations
# =============================================================================


def rwkv6_recurrent(
    r: jax.Array,  # [batch, num_heads, seq_len, head_k_dim] - receptance/query
    k: jax.Array,  # [batch, num_heads, seq_len, head_k_dim]
    v: jax.Array,  # [batch, num_heads, seq_len, head_v_dim]
    w: jax.Array,  # [batch, num_heads, seq_len, head_k_dim] - decay (negative)
    u: jax.Array,  # [num_heads, head_k_dim] - bonus term
    scale: float = 1.0,
    initial_state: Optional[jax.Array] = None,  # [batch, num_heads, head_k_dim, head_v_dim]
) -> Tuple[jax.Array, jax.Array]:
    """Token-by-token RWKV6 recurrence.

    The recurrence is:
        o_t = (h + u * k_t @ v_t^T) @ (sigmoid(r_t) * scale)
        h = h * exp(-exp(w_t)) + k_t @ v_t^T

    Args:
        r: Receptance (query) tensor [batch, num_heads, seq_len, head_k_dim]
        k: Key tensor [batch, num_heads, seq_len, head_k_dim]
        v: Value tensor [batch, num_heads, seq_len, head_v_dim]
        w: Decay tensor [batch, num_heads, seq_len, head_k_dim] (should be negative)
        u: Bonus term [num_heads, head_k_dim]
        scale: Scale factor (typically head_dim^-0.5)
        initial_state: Initial state [batch, num_heads, head_k_dim, head_v_dim]

    Returns:
        output: Output tensor [batch, num_heads, seq_len, head_v_dim]
        final_state: Final state [batch, num_heads, head_k_dim, head_v_dim]
    """
    batch, num_heads, seq_len, head_k_dim = r.shape
    head_v_dim = v.shape[-1]

    # Initialize state
    if initial_state is None:
        h = jnp.zeros((batch, num_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
    else:
        h = initial_state.astype(jnp.float32)

    def step(h, inputs):
        """Single RWKV6 step."""
        r_t, k_t, v_t, w_t = inputs
        # r_t, k_t: [batch, heads, head_k_dim]
        # v_t: [batch, heads, head_v_dim]
        # w_t: [batch, heads, head_k_dim]

        # Receptance gate (in RWKV-LM this is fused inside the CUDA kernel)
        r_t = jax.nn.sigmoid(r_t) * scale

        # Compute k @ v^T: [batch, heads, head_k_dim, head_v_dim]
        kv_t = jnp.einsum("bhk,bhv->bhkv", k_t, v_t)

        # Output: (h + u * kv) @ r = sum over k dimension
        # h: [batch, heads, k, v]
        # u: [heads, k] -> [1, heads, k, 1]
        # o_t: [batch, heads, v]
        h_plus_ukv = h + u[None, :, :, None] * kv_t
        o_t = jnp.einsum("bhkv,bhk->bhv", h_plus_ukv, r_t)

        # State update: RWKV-6 uses a doubly-exponentiated decay exp(-exp(w)).
        decay = jnp.exp(-jnp.exp(w_t))  # [batch, heads, head_k_dim]
        h_new = h * decay[..., None] + kv_t

        return h_new, o_t

    # Transpose for scan: [seq, batch, heads, dim]
    r_seq = jnp.transpose(r, (2, 0, 1, 3))
    k_seq = jnp.transpose(k, (2, 0, 1, 3))
    v_seq = jnp.transpose(v, (2, 0, 1, 3))
    w_seq = jnp.transpose(w, (2, 0, 1, 3))

    final_state, outputs = lax.scan(step, h, (r_seq, k_seq, v_seq, w_seq))

    # Transpose output back: [batch, heads, seq, head_v_dim]
    output = jnp.transpose(outputs, (1, 2, 0, 3))

    return output.astype(r.dtype), final_state


def rwkv6_chunkwise(
    r: jax.Array,  # [batch, num_heads, seq_len, head_k_dim]
    k: jax.Array,  # [batch, num_heads, seq_len, head_k_dim]
    v: jax.Array,  # [batch, num_heads, seq_len, head_v_dim]
    w: jax.Array,  # [batch, num_heads, seq_len, head_k_dim]
    u: jax.Array,  # [num_heads, head_k_dim]
    scale: float = 1.0,
    initial_state: Optional[jax.Array] = None,
    chunk_size: int = 32,
) -> Tuple[jax.Array, jax.Array]:
    """Chunkwise parallel RWKV6 computation.

    Based on the naive_chunk_rwkv6 implementation from FLA.
    Processes sequences in chunks for better parallelism during training.

    Args:
        r: Receptance [batch, num_heads, seq_len, head_k_dim]
        k: Key [batch, num_heads, seq_len, head_k_dim]
        v: Value [batch, num_heads, seq_len, head_v_dim]
        w: Decay [batch, num_heads, seq_len, head_k_dim]
        u: Bonus term [num_heads, head_k_dim]
        scale: Scale factor
        initial_state: Initial state [batch, num_heads, head_k_dim, head_v_dim]
        chunk_size: Size of each chunk

    Returns:
        output: [batch, num_heads, seq_len, head_v_dim]
        final_state: [batch, num_heads, head_k_dim, head_v_dim]
    """
    batch, num_heads, seq_len, head_k_dim = r.shape
    head_v_dim = v.shape[-1]

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        r = jnp.pad(r, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        w = jnp.pad(w, ((0, 0), (0, 0), (0, pad_len), (0, 0)))

    padded_len = r.shape[2]
    num_chunks = padded_len // chunk_size

    # Reshape to chunks: [batch, heads, num_chunks, chunk_size, dim]
    r_chunks = r.reshape(batch, num_heads, num_chunks, chunk_size, head_k_dim)
    k_chunks = k.reshape(batch, num_heads, num_chunks, chunk_size, head_k_dim)
    v_chunks = v.reshape(batch, num_heads, num_chunks, chunk_size, head_v_dim)
    w_chunks = w.reshape(batch, num_heads, num_chunks, chunk_size, head_k_dim)

    # Cumulative sum of w within each chunk: [batch, heads, num_chunks, chunk_size, head_k_dim]
    w_cumsum = jnp.cumsum(w_chunks, axis=3)

    # kw = k * exp(w_cumsum[-1] - w_cumsum)
    # This weights k by the remaining decay within the chunk
    kw = k_chunks * jnp.exp(w_cumsum[..., -1:, :] - w_cumsum)

    # wkv = kw^T @ v: [batch, heads, num_chunks, head_k_dim, head_v_dim]
    wkv = jnp.einsum("bhncK,bhncV->bhnKV", kw, v_chunks)

    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, num_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    def process_chunk(S, chunk_inputs):
        """Process one chunk and update state."""
        r_c, k_c, v_c, w_cumsum_c, wkv_c = chunk_inputs
        # r_c, k_c: [batch, heads, chunk_size, head_k_dim]
        # v_c: [batch, heads, chunk_size, head_v_dim]
        # w_cumsum_c: [batch, heads, chunk_size, head_k_dim]
        # wkv_c: [batch, heads, head_k_dim, head_v_dim]

        # Inter-chunk contribution: r @ (S * exp(w_cumsum - w))
        # where w is w_cumsum at position 0
        # This gives the contribution from previous chunks
        decay_for_inter = jnp.exp(w_cumsum_c - w_cumsum_c[..., :1, :])
        o_inter = jnp.einsum(
            "bhnck,bhkv->bhncv", r_c * decay_for_inter * scale, S
        )

        # Intra-chunk contribution
        # For each position i, compute attention to positions j < i
        # attn[i,j] = r[i] * k[j] * exp(w_cumsum[i] - w[i] - w_cumsum[j])
        # Note: w[i] = w_cumsum[i] - w_cumsum[i-1] (but we handle position 0 specially)
        
        # Build causal attention matrix
        C = chunk_size
        
        # For position i attending to position j:
        # decay = exp(w_cumsum[i] - w[i] - w_cumsum[j]) for j < i
        # At position 0, there's no previous position to attend to (handled by mask)
        
        # w_cumsum_i: [batch, heads, C, 1, k]
        # w_cumsum_j: [batch, heads, 1, C, k]
        w_cumsum_i = w_cumsum_c[..., :, None, :]
        w_cumsum_j = w_cumsum_c[..., None, :, :]
        
        # For the decay from j to i, we need exp(w_cumsum[i] - w_cumsum[j])
        # But for position i, we don't include its own decay when attending to j
        # So we use w_cumsum[i-1] instead of w_cumsum[i]
        # This is approximated by using w_cumsum - w (current position's decay)
        # First, get w at each position (difference of cumsum)
        w_at_pos = w_chunks[..., chunk_inputs[4].shape[0], :, :]  # Not quite right, need to track chunk idx
        
        # Simpler approach: use the formula from FLA
        # attn[i,j] = (r[i] * k[j]) * exp(w_cumsum[i] - w_cumsum[j]) for j < i (strictly)
        # Plus diagonal: r[i] * u * k[i] * v[i]
        
        attn_logits = jnp.einsum("bhnck,bhncK->bhncC", r_c, k_c)  # [batch, heads, C, C]
        
        # Apply decay: exp(w_cumsum[i] - w_cumsum[j])
        # But this needs to be position-aware
        relative_decay = jnp.exp(w_cumsum_i - w_cumsum_j)  # [batch, heads, C, C, k]
        
        # Sum over k dimension after weighting
        attn_weighted = jnp.einsum("bhnck,bhncK,bhijk->bhnij", r_c * scale, k_c, relative_decay)
        
        # Apply causal mask (j < i)
        causal_mask = jnp.tril(jnp.ones((C, C)), k=-1)
        attn_masked = attn_weighted * causal_mask

        # Intra-chunk attention output
        o_intra_attn = jnp.einsum("bhnij,bhnjv->bhniv", attn_masked, v_c)

        # Diagonal term: r[i] * u * k[i] * v[i] for each position i
        # u: [heads, k]
        # (r * u * k): [batch, heads, C, k]
        # sum over k, then multiply by v: [batch, heads, C, v]
        diag_term = jnp.sum(r_c * u[None, :, None, :] * k_c, axis=-1, keepdims=True)
        o_diag = diag_term * scale * v_c

        o_intra = o_intra_attn + o_diag

        # Total output
        o_chunk = o_inter + o_intra

        # Update state: S_new = S * exp(w_cumsum[-1]) + wkv
        chunk_total_decay = jnp.exp(w_cumsum_c[..., -1, :])  # [batch, heads, k]
        S_new = S * chunk_total_decay[..., None] + wkv_c

        return S_new, o_chunk

    # Process chunks via scan
    # Transpose chunks to [num_chunks, batch, heads, ...]
    final_state, outputs = lax.scan(
        process_chunk,
        S,
        (
            jnp.transpose(r_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(k_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(v_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(w_cumsum, (2, 0, 1, 3, 4)),
            jnp.transpose(wkv, (2, 0, 1, 3, 4)),
        ),
    )

    # Reshape output: [num_chunks, batch, heads, chunk_size, v] -> [batch, heads, seq, v]
    output = jnp.transpose(outputs, (1, 2, 0, 3, 4))
    output = output.reshape(batch, num_heads, padded_len, head_v_dim)

    # Remove padding
    if pad_len > 0:
        output = output[:, :, :seq_len, :]

    return output.astype(r.dtype), final_state


def rwkv6_step(
    r: jax.Array,  # [batch, num_heads, head_k_dim]
    k: jax.Array,  # [batch, num_heads, head_k_dim]
    v: jax.Array,  # [batch, num_heads, head_v_dim]
    w: jax.Array,  # [batch, num_heads, head_k_dim]
    u: jax.Array,  # [num_heads, head_k_dim]
    state: jax.Array,  # [batch, num_heads, head_k_dim, head_v_dim]
    scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Single step of RWKV6 for autoregressive generation.

    Args:
        r: Receptance [batch, num_heads, head_k_dim]
        k: Key [batch, num_heads, head_k_dim]
        v: Value [batch, num_heads, head_v_dim]
        w: Decay [batch, num_heads, head_k_dim]
        u: Bonus term [num_heads, head_k_dim]
        state: Current state [batch, num_heads, head_k_dim, head_v_dim]
        scale: Scale factor

    Returns:
        output: [batch, num_heads, head_v_dim]
        new_state: [batch, num_heads, head_k_dim, head_v_dim]
    """
    # Receptance gate (in RWKV-LM this is fused inside the CUDA kernel)
    r = jax.nn.sigmoid(r) * scale

    # Compute k @ v^T
    kv = jnp.einsum("bhk,bhv->bhkv", k, v)

    # Output: (h + u * kv) @ r
    h_plus_ukv = state + u[None, :, :, None] * kv
    output = jnp.einsum("bhkv,bhk->bhv", h_plus_ukv, r)

    # State update (RWKV-6): decay = exp(-exp(w))
    decay = jnp.exp(-jnp.exp(w))
    new_state = state * decay[..., None] + kv

    return output, new_state


# =============================================================================
# Token Shift Helper
# =============================================================================


def token_shift(
    x: jax.Array,  # [batch, seq, hidden]
    shift_state: Optional[jax.Array] = None,  # [batch, hidden]
) -> Tuple[jax.Array, jax.Array]:
    """Compute time shift: pad(x[:, :-1], left=1) - x.

    This computes delta = shifted - x where shifted is x shifted right by 1.

    Args:
        x: Input tensor [batch, seq, hidden]
        shift_state: Previous last token [batch, hidden] for continuity

    Returns:
        delta: Shifted difference [batch, seq, hidden]
        new_shift_state: Last token of x [batch, hidden]
    """
    batch, seq_len, hidden = x.shape

    if shift_state is None:
        # First position gets -x (shifted is zero)
        shifted = jnp.pad(x[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    else:
        # Use shift_state for first position
        shifted = jnp.concatenate([shift_state[:, None, :], x[:, :-1, :]], axis=1)

    delta = shifted - x
    new_shift_state = x[:, -1, :]

    return delta, new_shift_state


# =============================================================================
# Lerp Linear Layers (for RWKV6's special projections)
# =============================================================================


class LerpLinear(nnx.Module):
    """Linear layer with lerp (linear interpolation) input mixing.

    Computes: Linear(x + delta * lerp_weight)
    where delta = time_shift(x) - x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features

        # Learnable interpolation weights
        self.lerp = nnx.Param(jnp.ones((1, 1, in_features)))

        # Main linear projection
        self.linear = nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,  # [batch, seq, in_features]
        delta: jax.Array,  # [batch, seq, in_features]
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor
            delta: Time shift delta (shifted - x)

        Returns:
            Output tensor [batch, seq, out_features]
        """
        # Mix with lerp
        mixed = x + delta * self.lerp.value
        return self.linear(mixed)


class DDLerpLinear(nnx.Module):
    """Data-dependent lerp linear layer.

    Uses a low-rank projection to compute data-dependent interpolation weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        low_rank_dim: int = 32,
        use_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.low_rank_dim = low_rank_dim

        # Base interpolation weights
        self.lerp = nnx.Param(jnp.ones((1, 1, in_features)))

        # Low-rank data-dependent adjustment
        self.lora_a = nnx.Linear(in_features, low_rank_dim, use_bias=False, rngs=rngs)
        self.lora_b = nnx.Linear(low_rank_dim, in_features, use_bias=False, rngs=rngs)

        # Main projection
        self.linear = nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,  # [batch, seq, in_features]
        dd_weight: jax.Array,  # [batch, seq, in_features] - data-dependent weight
        delta: jax.Array,  # [batch, seq, in_features]
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor
            dd_weight: Data-dependent weight adjustment
            delta: Time shift delta

        Returns:
            Output tensor [batch, seq, out_features]
        """
        # Compute total lerp weight
        total_lerp = self.lerp.value + dd_weight

        # Mix with lerp
        mixed = x + delta * total_lerp
        return self.linear(mixed)


# =============================================================================
# GroupNorm (RWKV-style)
# =============================================================================


class GroupNorm(nnx.Module):
    """Group normalization as used in RWKV.

    Normalizes over groups of channels.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.scale = nnx.Param(jnp.ones((num_channels,)))
        self.bias = nnx.Param(jnp.zeros((num_channels,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply group normalization.

        Args:
            x: Input tensor [..., num_channels]

        Returns:
            Normalized tensor [..., num_channels]
        """
        orig_shape = x.shape
        # Reshape to [*, num_groups, channels_per_group]
        x = x.reshape(*orig_shape[:-1], self.num_groups, -1)

        # Normalize within each group
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(orig_shape)

        # Apply scale and bias
        x = x * self.scale.value + self.bias.value
        return x


# =============================================================================
# RWKV6 Block
# =============================================================================


class RWKV6Block(nnx.Module):
    """RWKV-6 block with time mixing and channel mixing.

    This implements the full RWKV-6 architecture from the Eagle/Finch paper.

    Key components:
    - Time mixing: Main attention mechanism with data-dependent decay
    - Channel mixing: FFN with squared ReLU and sigmoid gating

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads (head_size = hidden_size // num_heads)
        intermediate_size: FFN intermediate dimension (default: hidden_size * 4)
        proj_low_rank_dim: Low-rank dimension for time_maa projections
        gate_low_rank_dim: Low-rank dimension for decay projection
        layer_idx: Layer index (affects initialization)
        n_layers: Total number of layers (affects initialization)
        norm_eps: Epsilon for layer normalization
        rngs: NNx random number generators
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        intermediate_size: Optional[int] = None,
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        layer_idx: int = 0,
        n_layers: int = 12,
        norm_eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.intermediate_size = intermediate_size

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Initialize ratio parameters for layer-dependent initialization
        ratio_0_to_1 = layer_idx / max(n_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - (layer_idx / n_layers)

        # =====================================================================
        # Time Mixing (Attention) Components
        # =====================================================================

        # Input normalization
        self.ln1 = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)

        # RWKV-LM x060 time-mix parameters.
        ddd = (jnp.arange(hidden_size, dtype=jnp.float32) / float(hidden_size)).reshape(1, 1, -1)
        self.time_maa_x = nnx.Param(1.0 - jnp.power(ddd, ratio_1_to_almost0))
        self.time_maa_w = nnx.Param(1.0 - jnp.power(ddd, ratio_1_to_almost0))
        self.time_maa_k = nnx.Param(1.0 - jnp.power(ddd, ratio_1_to_almost0))
        self.time_maa_v = nnx.Param(1.0 - (jnp.power(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
        self.time_maa_r = nnx.Param(1.0 - jnp.power(ddd, 0.5 * ratio_1_to_almost0))
        self.time_maa_g = nnx.Param(1.0 - jnp.power(ddd, 0.5 * ratio_1_to_almost0))

        # LoRA that generates (mw, mk, mv, mr, mg)
        d_mix = proj_low_rank_dim
        self.time_maa_w1 = nnx.Param(jnp.zeros((hidden_size, d_mix * 5), dtype=jnp.float32))
        key = rngs.params()
        self.time_maa_w2 = nnx.Param(
            jax.random.uniform(key, (5, d_mix, hidden_size), minval=-0.01, maxval=0.01)
        )

        # Time-decay base + LoRA (RWKV-LM x060)
        n = jnp.arange(hidden_size, dtype=jnp.float32)
        denom = float(max(hidden_size - 1, 1))
        decay_speed = -6.0 + 5.0 * jnp.power(n / denom, 0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nnx.Param(decay_speed.reshape(1, 1, -1))

        d_decay = gate_low_rank_dim
        self.time_decay_w1 = nnx.Param(jnp.zeros((hidden_size, d_decay), dtype=jnp.float32))
        key = rngs.params()
        self.time_decay_w2 = nnx.Param(
            jax.random.uniform(key, (d_decay, hidden_size), minval=-0.01, maxval=0.01)
        )

        # Projections
        self.receptance = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs)
        self.gate = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs)

        # Bonus term (u) - per-head
        key = rngs.params()
        u_init = (
            ratio_0_to_1 * (1 - jnp.arange(hidden_size) / max(hidden_size - 1, 1))
            + ((jnp.arange(hidden_size) + 1) % 3 - 1) * 0.1
        )
        self.bonus = nnx.Param(u_init.reshape(num_heads, self.head_dim))

        # Output normalization and projection
        self.g_norm = GroupNorm(num_heads, hidden_size, eps=1e-5, rngs=rngs)
        self.o_proj = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs)

        # =====================================================================
        # Channel Mixing (FFN) Components
        # =====================================================================

        # Input normalization
        self.ln2 = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)

        # FFN mixing weights (RWKV-LM x060)
        ddd_ffn = (jnp.arange(hidden_size, dtype=jnp.float32) / float(hidden_size)).reshape(1, 1, -1)
        ffn_mix = 1.0 - jnp.power(ddd_ffn, ratio_1_to_almost0**3)
        self.time_maa_k_ffn = nnx.Param(ffn_mix)
        self.time_maa_r_ffn = nnx.Param(ffn_mix)

        # FFN projections
        self.ffn_key = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.ffn_receptance = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs)
        self.ffn_value = nnx.Linear(intermediate_size, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[RWKV6State] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[jax.Array, Optional[RWKV6State]]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            state: Optional cached state for generation
            mask: Optional mask (unused, for API compatibility)
            mode: Processing mode ("chunk" for training, "recurrent" for generation)

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            new_state: Updated state (if state was provided or mode="recurrent")
        """
        batch, seq_len, _ = x.shape

        # Get shift state if available
        shift_state = state.shift_state if state is not None else None
        recurrent_state = state.h if state is not None else None

        # Time mixing
        x_attn, new_h, new_shift = self._time_mixing(
            x, shift_state, recurrent_state, mode
        )
        x = x + x_attn

        # Channel mixing (FFN)
        x_ffn, final_shift = self._channel_mixing(x, new_shift)
        x = x + x_ffn

        # Build new state
        new_state = None
        if state is not None or mode == "recurrent":
            new_state = RWKV6State(h=new_h, shift_state=final_shift)

        return x, new_state

    def _time_mixing(
        self,
        x: jax.Array,
        shift_state: Optional[jax.Array],
        recurrent_state: Optional[jax.Array],
        mode: Optional[str],
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Time mixing (attention) sub-layer.

        Args:
            x: Input [batch, seq, hidden]
            shift_state: Previous shift state [batch, hidden]
            recurrent_state: Previous recurrent state [batch, heads, k, v]
            mode: Processing mode

        Returns:
            output: Attention output [batch, seq, hidden]
            new_h: New recurrent state
            new_shift: New shift state
        """
        batch, seq_len, _ = x.shape

        # Layer norm
        x_norm = self.ln1(x)

        # Compute time shift delta
        delta, new_shift = token_shift(x_norm, shift_state)

        # RWKV-LM x060: generate (mw, mk, mv, mr, mg) from (x + xx*time_maa_x)
        xxx = x_norm + delta * self.time_maa_x.value
        mix = jnp.tanh(jnp.einsum("bth,hd->btd", xxx, self.time_maa_w1.value))
        mix = mix.reshape(batch, seq_len, 5, self.proj_low_rank_dim)
        mix = jnp.einsum("btcd,cdh->btch", mix, self.time_maa_w2.value)
        mw, mk, mv, mr, mg = jnp.split(mix, 5, axis=2)
        mw = mw.squeeze(2)
        mk = mk.squeeze(2)
        mv = mv.squeeze(2)
        mr = mr.squeeze(2)
        mg = mg.squeeze(2)

        xw = x_norm + delta * (self.time_maa_w.value + mw)
        xk = x_norm + delta * (self.time_maa_k.value + mk)
        xv = x_norm + delta * (self.time_maa_v.value + mv)
        xr = x_norm + delta * (self.time_maa_r.value + mr)
        xg = x_norm + delta * (self.time_maa_g.value + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = jax.nn.silu(self.gate(xg))

        ww = jnp.tanh(jnp.einsum("bth,hd->btd", xw, self.time_decay_w1.value))
        ww = jnp.einsum("btd,dh->bth", ww, self.time_decay_w2.value)
        w = self.time_decay.value + ww

        # Reshape to heads: [batch, seq, heads, head_dim]
        r = r.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)
        w = w.reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        r = jnp.transpose(r, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        w = jnp.transpose(w, (0, 2, 1, 3))

        # Apply RWKV6 recurrence
        scale = 1.0

        if mode is None:
            mode = "recurrent" if seq_len == 1 else "chunk"

        if mode == "recurrent" or seq_len <= 64:
            o, new_h = rwkv6_recurrent(
                r, k, v, w, self.bonus.value,
                scale=scale,
                initial_state=recurrent_state,
            )
        else:
            o, new_h = rwkv6_recurrent(  # Use recurrent for now, chunkwise is buggy
                r, k, v, w, self.bonus.value,
                scale=scale,
                initial_state=recurrent_state,
            )

        # Reshape output: [batch, heads, seq, dim] -> [batch, seq, hidden]
        o = jnp.transpose(o, (0, 2, 1, 3))
        o = o.reshape(batch, seq_len, self.hidden_size)

        # Apply group norm and gating
        o = self.g_norm(o) * g

        # Output projection
        output = self.o_proj(o)

        return output, new_h, new_shift

    def _channel_mixing(
        self,
        x: jax.Array,
        shift_state: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Channel mixing (FFN) sub-layer.

        Args:
            x: Input [batch, seq, hidden]
            shift_state: Current shift state

        Returns:
            output: FFN output [batch, seq, hidden]
            new_shift: New shift state
        """
        # Layer norm
        x_norm = self.ln2(x)

        # Time shift
        delta, new_shift = token_shift(x_norm, shift_state)

        # Mix with lerp
        xk = x_norm + delta * self.time_maa_k_ffn.value
        xr = x_norm + delta * self.time_maa_r_ffn.value

        # FFN: squared ReLU with sigmoid gating
        k = jax.nn.relu(self.ffn_key(xk)) ** 2
        kv = self.ffn_value(k)
        output = jax.nn.sigmoid(self.ffn_receptance(xr)) * kv

        return output, new_shift

    def init_state(self, batch_size: int) -> RWKV6State:
        """Initialize empty state for autoregressive generation.

        Args:
            batch_size: Batch dimension

        Returns:
            Empty RWKV6State
        """
        return RWKV6State.zeros(
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )
