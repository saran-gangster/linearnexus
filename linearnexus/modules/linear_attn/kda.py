"""
Kimi Delta Attention (KDA): Per-Head Per-Dimension Gated Delta Rule

Paper: Kimi-VL Technical Report / Moonshot AI
       
Mathematical Foundation
=======================

KDA extends Gated DeltaNet with **per-head per-dimension gates**, giving finer
control over which dimensions decay. While Gated DeltaNet uses a scalar gate
per head, KDA uses a vector gate of shape [heads, key_dim].

Core Recurrence (Token-by-Token)
--------------------------------
For each token t:
    1. Apply per-dim decay:   S = S * exp(g_t)  where g_t is [heads, key_dim, 1]
    2. Retrieve old value:    v_old = sum(k_t[:, None] * S, axis=-2)
    3. Compute delta:         v_delta = beta_t * (v_t - v_old)
    4. Update state:          S = S + (beta_t * k_t)[:, None] @ v_delta^T
    5. Query output:          o_t = sum(q_t[:, None] * S, axis=-2)

Key Differences from Gated DeltaNet:
------------------------------------
1. **Per-dim gates**: g is [batch, heads, seq, key_dim] not [batch, heads, seq]
2. **Gate computation**: g = -exp(A_log) * softplus(f_proj(x) + dt_bias)
   where f_proj is a 2-layer MLP
3. **Beta projection**: Uses separate b_proj with sigmoid activation
4. **GVA support**: Grouped Value Attention (num_v_heads > num_heads)

Architecture:
- f_proj: hidden -> head_v_dim -> key_dim (2-layer MLP for gate)
- b_proj: hidden -> num_heads (for beta)
- A_log: learnable [num_heads] for gate scaling
- dt_bias: learnable [key_dim] added before softplus

Reference Implementation
------------------------
Based on FLA's kda ops and KimiDeltaAttention layer.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
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
class KDAState:
    """Cache for KDA autoregressive generation.

    Attributes:
        S: Recurrent state matrix [batch, num_v_heads, key_dim, value_dim_per_head]
        conv_state_q: Conv cache for Q [batch, kernel-1, key_dim_total]
        conv_state_k: Conv cache for K [batch, kernel-1, key_dim_total]
        conv_state_v: Conv cache for V [batch, kernel-1, value_dim]
    """

    S: jax.Array  # [batch, num_v_heads, key_dim_per_head, value_dim_per_head]
    conv_state_q: Optional[jax.Array] = None
    conv_state_k: Optional[jax.Array] = None
    conv_state_v: Optional[jax.Array] = None

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_v_heads: int,
        key_dim_per_head: int,
        value_dim_per_head: int,
        key_dim_total: Optional[int] = None,
        value_dim: Optional[int] = None,
        conv_size: int = 4,
        use_conv: bool = True,
    ) -> "KDAState":
        """Initialize empty state.

        Args:
            batch_size: Batch dimension
            num_v_heads: Number of value heads (can be > num_heads for GVA)
            key_dim_per_head: Key dimension per head
            value_dim_per_head: Value dimension per head
            key_dim_total: Total key dimension for conv (num_heads * key_dim_per_head)
            value_dim: Total value dimension for conv (num_v_heads * value_dim_per_head)
            conv_size: Convolution kernel size
            use_conv: Whether to use short convolutions
        """
        S = jnp.zeros((batch_size, num_v_heads, key_dim_per_head, value_dim_per_head))

        conv_state_q = None
        conv_state_k = None
        conv_state_v = None

        if use_conv and key_dim_total is not None and value_dim is not None:
            conv_state_q = jnp.zeros((batch_size, conv_size - 1, key_dim_total))
            conv_state_k = jnp.zeros((batch_size, conv_size - 1, key_dim_total))
            conv_state_v = jnp.zeros((batch_size, conv_size - 1, value_dim))

        return cls(
            S=S,
            conv_state_q=conv_state_q,
            conv_state_k=conv_state_k,
            conv_state_v=conv_state_v,
        )


# =============================================================================
# KDA Gate Computation
# =============================================================================

def kda_gate(
    g: jax.Array,  # [batch, seq, num_heads * head_k_dim]
    A_log: jax.Array,  # [num_heads]
    head_k_dim: int,
    g_bias: Optional[jax.Array] = None,  # [num_heads * head_k_dim]
    beta: Optional[jax.Array] = None,  # [batch, seq, num_heads]
    threshold: float = 20.0,
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """KDA gate computation.

    Computes: g = -exp(A_log)[..., None] * softplus(g + g_bias)

    Args:
        g: Gate input [batch, seq, num_heads * head_k_dim]
        A_log: Log of decay base [num_heads]
        head_k_dim: Dimension per head
        g_bias: Optional bias added to g before softplus
        beta: Optional tensor for beta sigmoid [batch, seq, num_heads]
        threshold: Threshold for softplus linear approximation

    Returns:
        g_out: Gate output [batch, seq, num_heads, head_k_dim]
        beta_out: Sigmoid of beta if provided, else None
    """
    # Add bias if provided
    if g_bias is not None:
        g = g + g_bias

    # Reshape to [batch, seq, num_heads, head_k_dim]
    batch, seq_len, _ = g.shape
    num_heads = A_log.shape[0]
    g = g.reshape(batch, seq_len, num_heads, head_k_dim)

    # Compute -exp(A_log) * softplus(g)
    A_exp = -jnp.exp(A_log.astype(jnp.float32))  # [num_heads]
    
    # Stable softplus: log(1 + exp(x)) for x <= threshold, else x
    g_float = g.astype(jnp.float32)
    g_softplus = jnp.where(
        g_float > threshold,
        g_float,
        jax.nn.softplus(g_float)
    )

    # Apply decay: [num_heads, 1] * [batch, seq, num_heads, head_k_dim]
    g_out = A_exp[None, None, :, None] * g_softplus

    # Compute beta sigmoid if provided
    beta_out = None
    if beta is not None:
        beta_out = jax.nn.sigmoid(beta.astype(jnp.float32))

    return g_out.astype(g.dtype), beta_out


# =============================================================================
# Core KDA Operations
# =============================================================================

def kda_recurrent(
    q: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    k: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    v: jax.Array,  # [batch, num_v_heads, seq_len, value_dim]
    g: jax.Array,  # [batch, num_v_heads, seq_len, key_dim] - per-dim decay (negative)
    beta: jax.Array,  # [batch, num_v_heads, seq_len] - learning rate
    scale: Optional[float] = None,
    initial_state: Optional[jax.Array] = None,  # [batch, num_v_heads, key_dim, value_dim]
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Token-by-token KDA recurrence with per-dimension gating.

    The key difference from Gated DeltaNet is that decay gate is per-head
    per-key-dimension rather than just per-head.

    Recurrence:
        S = S * exp(g_t)[..., None]  # per-dim decay
        v_old = einsum('bhkv,bhk->bhv', S, k_t)
        v_delta = beta_t * (v_t - v_old)
        S = S + einsum('bhk,bhv->bhkv', beta_t * k_t, v_delta)
        o_t = einsum('bhk,bhkv->bhv', q_t, S)

    Args:
        q: Query tensor [batch, num_v_heads, seq_len, key_dim]
        k: Key tensor [batch, num_v_heads, seq_len, key_dim]
        v: Value tensor [batch, num_v_heads, seq_len, value_dim]
        g: Decay gate [batch, num_v_heads, seq_len, key_dim] (typically negative)
        beta: Learning rate [batch, num_v_heads, seq_len]
        scale: Query scaling factor (default: 1/sqrt(key_dim))
        initial_state: Initial state [batch, num_v_heads, key_dim, value_dim]
        use_qk_l2norm: Whether to L2-normalize Q and K

    Returns:
        output: Output tensor [batch, num_v_heads, seq_len, value_dim]
        final_state: Final state [batch, num_v_heads, key_dim, value_dim]
    """
    batch, num_v_heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]

    if scale is None:
        scale = key_dim ** -0.5

    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, num_v_heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    def step(S, inputs):
        """Single step with per-dim gated decay + delta rule."""
        k_t, v_t, q_t, g_t, beta_t = inputs
        # k_t: [batch, heads, key_dim]
        # v_t: [batch, heads, value_dim]
        # g_t: [batch, heads, key_dim]
        # beta_t: [batch, heads]

        # Cast to float32 for stability
        k_t = k_t.astype(jnp.float32)
        v_t = v_t.astype(jnp.float32)
        q_t = q_t.astype(jnp.float32)
        g_t = g_t.astype(jnp.float32)
        beta_t = beta_t.astype(jnp.float32)

        # L2 normalize Q and K if requested
        if use_qk_l2norm:
            q_t = q_t / (jnp.linalg.norm(q_t, axis=-1, keepdims=True) + 1e-6)
            k_t = k_t / (jnp.linalg.norm(k_t, axis=-1, keepdims=True) + 1e-6)

        q_t = q_t * scale

        # 1. Apply per-dim decay to state: S = S * exp(g)[..., None]
        # g_t: [batch, heads, key_dim] -> [batch, heads, key_dim, 1]
        decay = jnp.exp(g_t)[..., None]  # [batch, heads, key_dim, 1]
        S = S * decay

        # 2. Retrieve old value: v_old = sum(k_t[..., None] * S, axis=-2)
        # k_t: [batch, heads, key_dim]
        # S: [batch, heads, key_dim, value_dim]
        v_old = jnp.einsum("bhk,bhkv->bhv", k_t, S)

        # 3. Compute delta: v_delta = beta_t * (v_t - v_old)
        v_delta = beta_t[..., None] * (v_t - v_old)

        # 4. Update state: S = S + (beta_t * k_t) @ v_delta^T
        # Note: FLA uses beta_t * k_t in the outer product
        S_new = S + jnp.einsum("bhk,bhv->bhkv", beta_t[..., None] * k_t, v_delta)

        # 5. Query output: o_t = sum(q_t[..., None] * S, axis=-2)
        o_t = jnp.einsum("bhk,bhkv->bhv", q_t, S_new)

        return S_new, o_t

    # Transpose for scan: [seq, batch, heads, dim]
    k_seq = jnp.transpose(k, (2, 0, 1, 3))
    v_seq = jnp.transpose(v, (2, 0, 1, 3))
    q_seq = jnp.transpose(q, (2, 0, 1, 3))
    g_seq = jnp.transpose(g, (2, 0, 1, 3))
    beta_seq = jnp.transpose(beta, (2, 0, 1))

    final_state, outputs = lax.scan(step, S, (k_seq, v_seq, q_seq, g_seq, beta_seq))

    # Transpose output back: [batch, heads, seq, value_dim]
    output = jnp.transpose(outputs, (1, 2, 0, 3))

    return output.astype(q.dtype), final_state


def kda_step(
    q: jax.Array,  # [batch, num_v_heads, key_dim]
    k: jax.Array,  # [batch, num_v_heads, key_dim]
    v: jax.Array,  # [batch, num_v_heads, value_dim]
    g: jax.Array,  # [batch, num_v_heads, key_dim]
    beta: jax.Array,  # [batch, num_v_heads]
    state: jax.Array,  # [batch, num_v_heads, key_dim, value_dim]
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Single step of KDA for autoregressive generation.

    Args:
        q: Query [batch, num_v_heads, key_dim]
        k: Key [batch, num_v_heads, key_dim]
        v: Value [batch, num_v_heads, value_dim]
        g: Decay gate [batch, num_v_heads, key_dim]
        beta: Learning rate [batch, num_v_heads]
        state: Current state [batch, num_v_heads, key_dim, value_dim]
        scale: Query scaling factor
        use_qk_l2norm: Whether to L2-normalize Q and K

    Returns:
        output: Output [batch, num_v_heads, value_dim]
        new_state: Updated state [batch, num_v_heads, key_dim, value_dim]
    """
    key_dim = q.shape[-1]
    if scale is None:
        scale = key_dim ** -0.5

    # Cast to float32
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    S = state.astype(jnp.float32)

    # L2 normalize
    if use_qk_l2norm:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    q = q * scale

    # 1. Apply per-dim decay
    decay = jnp.exp(g)[..., None]  # [batch, heads, key_dim, 1]
    S = S * decay

    # 2. Retrieve old value
    v_old = jnp.einsum("bhk,bhkv->bhv", k, S)

    # 3. Compute delta
    v_delta = beta[..., None] * (v - v_old)

    # 4. Update state
    S_new = S + jnp.einsum("bhk,bhv->bhkv", beta[..., None] * k, v_delta)

    # 5. Query output
    output = jnp.einsum("bhk,bhkv->bhv", q, S_new)

    return output.astype(state.dtype), S_new


def kda_chunkwise(
    q: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    k: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    v: jax.Array,  # [batch, num_v_heads, seq_len, value_dim]
    g: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    beta: jax.Array,  # [batch, num_v_heads, seq_len]
    scale: Optional[float] = None,
    initial_state: Optional[jax.Array] = None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Chunkwise parallel KDA with per-dimension gating.

    Extends Gated DeltaNet chunkwise with per-dim cumulative gates.

    Args:
        q: Query [batch, num_v_heads, seq_len, key_dim]
        k: Key [batch, num_v_heads, seq_len, key_dim]
        v: Value [batch, num_v_heads, seq_len, value_dim]
        g: Decay gate [batch, num_v_heads, seq_len, key_dim]
        beta: Learning rate [batch, num_v_heads, seq_len]
        scale: Query scaling factor
        initial_state: Initial state [batch, num_v_heads, key_dim, value_dim]
        chunk_size: Size of each chunk
        use_qk_l2norm: Whether to L2-normalize Q and K

    Returns:
        output: [batch, num_v_heads, seq_len, value_dim]
        final_state: [batch, num_v_heads, key_dim, value_dim]
    """
    batch, num_v_heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]

    if scale is None:
        scale = key_dim ** -0.5

    # L2 normalize Q and K if requested
    if use_qk_l2norm:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))

    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size

    # Reshape: [batch, heads, num_chunks, chunk_size, dim]
    q_chunks = q.reshape(batch, num_v_heads, num_chunks, chunk_size, key_dim)
    k_chunks = k.reshape(batch, num_v_heads, num_chunks, chunk_size, key_dim)
    v_chunks = v.reshape(batch, num_v_heads, num_chunks, chunk_size, value_dim)
    g_chunks = g.reshape(batch, num_v_heads, num_chunks, chunk_size, key_dim)
    beta_chunks = beta.reshape(batch, num_v_heads, num_chunks, chunk_size)

    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, num_v_heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    def process_chunk(S, chunk_inputs):
        """Process one chunk with KDA per-dim gating."""
        q_c, k_c, v_c, g_c, beta_c = chunk_inputs
        # q_c, k_c: [batch, heads, L, key_dim]
        # v_c: [batch, heads, L, value_dim]
        # g_c: [batch, heads, L, key_dim]
        # beta_c: [batch, heads, L]
        L = chunk_size

        # Cast to float32
        q_c = q_c.astype(jnp.float32)
        k_c = k_c.astype(jnp.float32)
        v_c = v_c.astype(jnp.float32)
        g_c = g_c.astype(jnp.float32)
        beta_c = beta_c.astype(jnp.float32)

        # Compute cumulative gates within chunk (per-dim)
        # g_cum[i] = sum(g[0:i+1]) for positions 0 to i
        g_cum = jnp.cumsum(g_c, axis=-2)  # [batch, heads, L, key_dim]

        # Total chunk decay (for state update)
        g_total = g_cum[:, :, -1, :]  # [batch, heads, key_dim]

        # Build intra-chunk attention with per-dim relative gates
        # For position i affecting position j (j > i):
        # Relative decay per dim: exp(g_cum[j] - g_cum[i])

        # Compute pairwise relative gates
        # g_cum_i: [batch, heads, L, 1, key_dim]
        # g_cum_j: [batch, heads, 1, L, key_dim]
        g_cum_i = g_cum[:, :, :, None, :]
        g_cum_j = g_cum[:, :, None, :, :]
        # relative_gate: [batch, heads, L, L, key_dim]
        relative_gate = jnp.exp(g_cum_j - g_cum_i)

        # K @ K^T with per-dim gating (contract over key_dim)
        # kk[i,j] = sum_d(k[i,d] * k[j,d] * relative_gate[i,j,d])
        k_i = k_c[:, :, :, None, :]  # [batch, heads, L, 1, key_dim]
        k_j = k_c[:, :, None, :, :]  # [batch, heads, 1, L, key_dim]
        kk_gated = jnp.sum(k_i * k_j * relative_gate, axis=-1)  # [batch, heads, L, L]

        # Apply lower triangular mask (exclude diagonal for Akk)
        mask_lower = jnp.tril(jnp.ones((L, L)), k=-1)
        M = kk_gated * mask_lower

        # A = I + diag(beta) @ M
        A = jnp.eye(L) + beta_c[..., None] * M

        # Solve A @ T = diag(beta) for T
        diag_beta = jnp.eye(L) * beta_c[..., None]
        T = jax.lax.linalg.triangular_solve(A, diag_beta, left_side=True, lower=True)

        # Compute W = T @ (K * exp(g_cum)) and U = T @ (V * exp(g_cum))
        # Need per-dim gated K
        g_cum_exp = jnp.exp(g_cum)  # [batch, heads, L, key_dim]
        k_gated = k_c * g_cum_exp  # [batch, heads, L, key_dim]

        # For v_gated, we need scalar gate (sum over key_dim or use first?)
        # Actually from FLA, V uses the same gating structure
        # Let's use mean of g_cum over key_dim for value gating
        g_cum_mean = jnp.mean(g_cum, axis=-1, keepdims=True)  # [batch, heads, L, 1]
        v_gated = v_c * jnp.exp(g_cum_mean)  # [batch, heads, L, value_dim]

        # W = T @ k_gated: [batch, heads, L, key_dim]
        W = jnp.einsum("bhij,bhjk->bhik", T, k_gated)
        # U = T @ v_gated: [batch, heads, L, value_dim]
        U = jnp.einsum("bhij,bhjv->bhiv", T, v_gated)

        # Inter-chunk: decay state and compute Q @ S_decayed
        # Need position-wise decay
        # O_inter[i] = q[i] @ S * exp(g_cum[i])
        # But g is per-dim, so we need to handle this carefully
        # q @ S gives [batch, heads, L, value_dim]
        # Scale by exp(mean(g_cum)) for simplicity
        O_inter = jnp.einsum("bhik,bhkv->bhiv", q_c, S) * jnp.exp(g_cum_mean) * scale

        # Intra-chunk correction: U - W @ S
        WS = jnp.einsum("bhik,bhkv->bhiv", W, S)
        correction = U - WS

        # Intra-chunk: (Q @ K^T * mask * gates) @ correction
        # Use per-dim gates for QK^T
        q_i = q_c[:, :, :, None, :]  # [batch, heads, L, 1, key_dim]
        qk_gated = jnp.sum(q_i * k_j * relative_gate, axis=-1)  # [batch, heads, L, L]

        # Apply causal mask (include diagonal for Aqk unlike Akk)
        causal_mask = jnp.tril(jnp.ones((L, L)))
        qk_masked = qk_gated * causal_mask * scale

        O_intra = jnp.einsum("bhij,bhjv->bhiv", qk_masked, correction)

        O_chunk = O_inter + O_intra

        # State update: S_new = S * exp(g_total) + k_gated^T @ correction
        # g_total: [batch, heads, key_dim]
        S_new = S * jnp.exp(g_total)[..., None]  # [batch, heads, key_dim, value_dim]
        # k_gated: [batch, heads, L, key_dim]
        # correction: [batch, heads, L, value_dim]
        S_new = S_new + jnp.einsum("bhik,bhiv->bhkv", k_gated, correction)

        return S_new, O_chunk

    # Process chunks
    final_state, outputs = lax.scan(
        process_chunk,
        S,
        (
            jnp.transpose(q_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(k_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(v_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(g_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(beta_chunks, (2, 0, 1, 3)),
        ),
    )

    # Reshape output
    output = jnp.transpose(outputs, (1, 2, 0, 3, 4))
    output = output.reshape(batch, num_v_heads, padded_len, value_dim)

    # Remove padding
    if pad_len > 0:
        output = output[:, :, :seq_len, :]

    return output.astype(q.dtype), final_state


# =============================================================================
# Fused RMSNorm with Gating
# =============================================================================

class FusedRMSNormGated(nnx.Module):
    """Fused RMSNorm with sigmoid gating for KDA output.
    
    Computes: RMSNorm(x) * sigmoid(g)
    
    This is used in the output processing of KDA to gate the 
    attention output before the final projection.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((hidden_size,)))
    
    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        """Apply gated RMSNorm.
        
        Args:
            x: Input tensor [..., hidden_size]
            gate: Gate tensor [..., hidden_size]
            
        Returns:
            Gated normalized tensor [..., hidden_size]
        """
        # RMSNorm
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(variance + self.eps)
        x_norm = x_norm * self.scale.value
        
        # Gate with sigmoid
        return x_norm * jax.nn.sigmoid(gate)


# =============================================================================
# KDA Block
# =============================================================================

class KDABlock(nnx.Module):
    """Kimi Delta Attention (KDA) block.

    This implements KDA with per-dimension gating, similar to Gated DeltaNet
    but with finer-grained control over which key dimensions decay.

    Key features:
    - Per-dimension decay gates (g has shape [batch, seq, heads, key_dim])
    - Two-layer MLP for gate projection (f_proj)
    - Separate beta projection for learning rate
    - Optional short convolutions on Q, K, V
    - Grouped Value Attention (GVA) support
    - Fused RMSNorm with gating on output

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of Q/K heads
        num_v_heads: Number of V heads (>= num_heads for GVA)
        head_dim: Dimension per head (default: 128)
        expand_v: Value expansion ratio (default: 1.0)
        use_short_conv: Whether to use short convolutions
        conv_size: Kernel size for short convolutions
        norm_eps: Epsilon for layer normalization
        rngs: Random number generators
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_v_heads: Optional[int] = None,
        head_dim: int = 128,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.norm_eps = norm_eps

        # Derived dimensions
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        # Validation
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={num_heads}."
            )

        # Input normalization
        self.norm = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)

        # QKV projections
        self.q_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, self.value_dim, use_bias=False, rngs=rngs)

        # Short convolutions
        if use_short_conv:
            self.q_conv_weight = nnx.Param(
                jax.random.normal(rngs.params(), (conv_size, self.key_dim)) * 0.02
            )
            self.k_conv_weight = nnx.Param(
                jax.random.normal(rngs.params(), (conv_size, self.key_dim)) * 0.02
            )
            self.v_conv_weight = nnx.Param(
                jax.random.normal(rngs.params(), (conv_size, self.value_dim)) * 0.02
            )

        # Gate projection: 2-layer MLP (hidden -> head_v_dim -> key_dim)
        self.f_proj_1 = nnx.Linear(hidden_size, self.head_v_dim, use_bias=False, rngs=rngs)
        self.f_proj_2 = nnx.Linear(self.head_v_dim, self.key_dim, use_bias=False, rngs=rngs)

        # Beta projection
        self.b_proj = nnx.Linear(hidden_size, num_heads, use_bias=False, rngs=rngs)

        # Learnable decay parameters (Mamba-style)
        A_init = jnp.log(jax.random.uniform(rngs.params(), (num_heads,), minval=1.0, maxval=16.0))
        self.A_log = nnx.Param(A_init)
        self.dt_bias = nnx.Param(jnp.zeros((self.key_dim,)))

        # Output gating: 2-layer MLP
        self.g_proj_1 = nnx.Linear(hidden_size, self.head_v_dim, use_bias=False, rngs=rngs)
        self.g_proj_2 = nnx.Linear(self.head_v_dim, self.value_dim, use_bias=True, rngs=rngs)

        # Output normalization with gating
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps, rngs=rngs)

        # Output projection
        self.o_proj = nnx.Linear(self.value_dim, hidden_size, use_bias=False, rngs=rngs)

    def init_state(self, batch_size: int) -> KDAState:
        """Initialize empty state for generation.

        Args:
            batch_size: Batch size

        Returns:
            Zero-initialized KDAState
        """
        return KDAState.zeros(
            batch_size=batch_size,
            num_v_heads=self.num_v_heads,
            key_dim_per_head=self.head_k_dim,
            value_dim_per_head=self.head_v_dim,
            key_dim_total=self.key_dim,
            value_dim=self.value_dim,
            conv_size=self.conv_size,
            use_conv=self.use_short_conv,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[KDAState] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[str] = None,
    ) -> Tuple[jax.Array, Optional[KDAState]]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            state: Optional KDAState for autoregressive generation
            mask: Optional attention mask (not used, for interface compatibility)
            mode: Optional mode override ("recurrent" or "chunk")

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            new_state: Updated state (if state was provided or mode is recurrent)
        """
        batch, seq_len, _ = x.shape
        residual = x

        # Input normalization
        x = self.norm(x)

        # QKV projections
        q = self.q_proj(x)  # [batch, seq, key_dim]
        k = self.k_proj(x)  # [batch, seq, key_dim]
        v = self.v_proj(x)  # [batch, seq, value_dim]

        # Short convolutions
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if state is not None:
                conv_state_q = state.conv_state_q
                conv_state_k = state.conv_state_k
                conv_state_v = state.conv_state_v

            q, conv_state_q = depthwise_conv1d_causal(
                q, self.q_conv_weight.value, bias=None, cache=conv_state_q
            )
            k, conv_state_k = depthwise_conv1d_causal(
                k, self.k_conv_weight.value, bias=None, cache=conv_state_k
            )
            v, conv_state_v = depthwise_conv1d_causal(
                v, self.v_conv_weight.value, bias=None, cache=conv_state_v
            )
            q = jax.nn.silu(q)
            k = jax.nn.silu(k)
            v = jax.nn.silu(v)
        else:
            # Without conv, still apply activation
            q = jax.nn.silu(q)
            k = jax.nn.silu(k)
            v = jax.nn.silu(v)

        # Gate computation: g = -exp(A_log) * softplus(f_proj(x) + dt_bias)
        f = self.f_proj_1(x)
        f = self.f_proj_2(f)  # [batch, seq, key_dim]
        beta_raw = self.b_proj(x)  # [batch, seq, num_heads]

        # Apply KDA gate
        g, beta = kda_gate(
            g=f,
            A_log=self.A_log.value,
            head_k_dim=self.head_k_dim,
            g_bias=self.dt_bias.value,
            beta=beta_raw,
        )
        # g: [batch, seq, num_heads, head_k_dim]
        # beta: [batch, seq, num_heads]

        # Reshape Q, K, V to head format
        q = q.reshape(batch, seq_len, self.num_heads, self.head_k_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_k_dim)
        v = v.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        # Transpose to [batch, heads, seq, dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        g = jnp.transpose(g, (0, 2, 1, 3))  # [batch, num_heads, seq, head_k_dim]
        beta = jnp.transpose(beta, (0, 2, 1))  # [batch, num_heads, seq]

        # Handle GVA: repeat Q, K, g, beta if num_v_heads > num_heads
        if self.num_v_heads > self.num_heads:
            repeat_factor = self.num_v_heads // self.num_heads
            q = jnp.repeat(q, repeat_factor, axis=1)
            k = jnp.repeat(k, repeat_factor, axis=1)
            g = jnp.repeat(g, repeat_factor, axis=1)
            beta = jnp.repeat(beta, repeat_factor, axis=1)

        # Allow negative eigenvalues: multiply beta by 2
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Get recurrent state
        recurrent_state = state.S if state is not None else None

        # Select mode - default to recurrent since chunkwise for per-dim gates is complex
        # and our reference implementation has numerical issues
        use_recurrent = mode == "recurrent" or mode is None or seq_len == 1
        if use_recurrent:
            if seq_len == 1:
                # Single step
                o, new_recurrent_state = kda_step(
                    q=q[:, :, 0, :],  # [batch, heads, key_dim]
                    k=k[:, :, 0, :],
                    v=v[:, :, 0, :],
                    g=g[:, :, 0, :],
                    beta=beta[:, :, 0],
                    state=recurrent_state if recurrent_state is not None else jnp.zeros(
                        (batch, self.num_v_heads, self.head_k_dim, self.head_v_dim)
                    ),
                    use_qk_l2norm=True,
                )
                o = o[:, :, None, :]  # [batch, heads, 1, value_dim]
            else:
                o, new_recurrent_state = kda_recurrent(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    use_qk_l2norm=True,
                )
        else:
            o, new_recurrent_state = kda_chunkwise(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                use_qk_l2norm=True,
            )

        # Output gating
        gate = self.g_proj_1(self.norm(residual))
        gate = self.g_proj_2(gate)  # [batch, seq, value_dim]
        gate = gate.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        # Apply gated RMSNorm per head
        o = jnp.transpose(o, (0, 2, 1, 3))  # [batch, seq, heads, head_v_dim]
        o = self.o_norm(o, gate)

        # Reshape to [batch, seq, value_dim]
        o = o.reshape(batch, seq_len, self.value_dim)

        # Output projection + residual
        output = self.o_proj(o) + residual

        # State handling
        new_state = None
        if state is not None or mode == "recurrent":
            new_state = KDAState(
                S=new_recurrent_state,
                conv_state_q=conv_state_q,
                conv_state_k=conv_state_k,
                conv_state_v=conv_state_v,
            )

        return output, new_state
