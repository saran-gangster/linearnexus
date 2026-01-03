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
    3. Compute delta:         v_delta = (v_t - v_old)
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
        v_delta = (v_t - v_old)
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

        # 3. Compute delta: v_delta = (v_t - v_old)
        v_delta = v_t - v_old

        # 4. Update state: S = S + (beta_t * k_t) @ v_delta^T
        # IMPORTANT: beta is applied exactly once (matches FLA reference).
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

    # 3. Compute delta (beta is applied in the outer-product update)
    v_delta = v - v_old

    # 4. Update state (beta applied exactly once)
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
    """Chunkwise parallel KDA (faithful port of FLA's naive_chunk_kda).

    This implements the same algorithm as:
      [examples/fla/ops/kda/naive.py](examples/fla/ops/kda/naive.py)

    Notes:
    - This is a correctness-oriented implementation (uses O(BT^3) work per chunk).
    - Internally computes in float32 for stability.
    - Expects g to be per-token log-decay (typically negative); the kernel uses
      cumulative sums of g within each chunk.
    """
    batch, num_v_heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]

    if scale is None:
        scale = key_dim ** -0.5

    # L2 normalize Q and K if requested (matches FLA's optional in-kernel l2norm)
    if use_qk_l2norm:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    # Scale queries (matches reference)
    q = q * scale

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))

    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size
    BT = chunk_size

    # Reshape to chunks: [B, H, N, BT, ...]
    q = q.reshape(batch, num_v_heads, num_chunks, BT, key_dim)
    k = k.reshape(batch, num_v_heads, num_chunks, BT, key_dim)
    v = v.reshape(batch, num_v_heads, num_chunks, BT, value_dim)
    g = g.reshape(batch, num_v_heads, num_chunks, BT, key_dim)
    beta = beta.reshape(batch, num_v_heads, num_chunks, BT)

    # Compute in float32
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Cumulative gates within each chunk: g <- cumsum(g)
    # Prefer cumsum to avoid associative_scan lowering IR bloat.
    g = lax.cumsum(g, axis=3)  # [B, H, N, BT, K]

    # ---------------------------------------------------------------------
    # Precompute Aqk and A (called Akk in FLA) per chunk.
    #
    # FLA Triton intra-chunk kernels avoid materializing [BT, BT, K] and avoid
    # overflow by using a blockwise factorization with a per-row-block gate
    # reference g_ref:
    #   exp(g_row - g_col) = exp(g_row - g_ref) * exp(g_ref - g_col)
    # For causal pairs (row >= col) and chunk-local cumsum gates, both
    # differences are <= 0, so the exponentials are in (0, 1] and never
    # overflow.
    # ---------------------------------------------------------------------
    exp_g = jnp.exp(g)  # [B, H, N, BT, K] (safe: g is typically <= 0)

    # Tile within each chunk (BC=16 as in FLA kernels)
    BC = 16
    if BT % BC != 0:
        raise ValueError(f"chunk_size (BT={BT}) must be divisible by BC={BC}.")
    NC = BT // BC

    # Masks within a BCxBC tile
    tril_inclusive = jnp.tril(jnp.ones((BC, BC), dtype=bool), k=0)
    tril_strict = jnp.tril(jnp.ones((BC, BC), dtype=bool), k=-1)

    def _exp_leq0(x: jax.Array) -> jax.Array:
        return jnp.exp(jnp.minimum(x, 0.0)).astype(jnp.float32)

    def _masked_exp(d: jax.Array, mask: jax.Array) -> jax.Array:
        """Exponentiate with non-causal entries forced to zero.

        Uses `exp(-inf) = 0` rather than compute-then-mask.
        """
        neg_inf = jnp.array(-jnp.inf, dtype=jnp.float32)
        return jnp.exp(jnp.where(mask, d, neg_inf)).astype(jnp.float32)

    def build_m_and_aqk(
        q_all: jax.Array,
        k_all: jax.Array,
        g_all: jax.Array,
        beta_all: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Build M (unit lower-triangular) and Aqk for all chunks.

        Shapes:
            q_all, k_all, g_all: [B, H, N, BT, K]
            beta_all: [B, H, N, BT]
            Returns:
                M:   [B, H, N, BT, BT]
                Aqk: [B, H, N, BT, BT]

        Uses lax.fori_loop over BC blocks (no Python unrolling) while keeping
        chunks (N) batched for GPU parallelism.
        """
        # Build blockwise tensors and reshape once at the end.
        # This avoids repeated dynamic_update_slice into huge [BT, BT] buffers.
        B_, H_, N_, _, K_ = q_all.shape

        eye_bc = jnp.eye(BC, dtype=jnp.float32)[None, None, None, :, :]
        zeros_bc = jnp.zeros((B_, H_, N_, BC, BC), dtype=jnp.float32)

        def diag_block(
            g_rows: jax.Array,
            q_rows: jax.Array,
            k_rows: jax.Array,
            beta_rows: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            d = g_rows[:, :, :, :, None, :] - g_rows[:, :, :, None, :, :]
            exp_rel_strict = _masked_exp(d, tril_strict[None, None, None, :, :, None])
            exp_rel_incl = _masked_exp(d, tril_inclusive[None, None, None, :, :, None])

            A0 = jnp.sum(
                (k_rows[:, :, :, :, None, :] * k_rows[:, :, :, None, :, :]) * exp_rel_strict,
                axis=-1,
            )
            Aqk = jnp.sum(
                (q_rows[:, :, :, :, None, :] * k_rows[:, :, :, None, :, :]) * exp_rel_incl,
                axis=-1,
            )
            A0 = A0 * beta_rows[:, :, :, :, None]
            return eye_bc + A0, Aqk

        def offdiag_block(
            g_rows: jax.Array,
            q_rows: jax.Array,
            k_rows: jax.Array,
            beta_rows: jax.Array,
            g_cols: jax.Array,
            k_cols: jax.Array,
            g_ref: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            row_scale = _exp_leq0(g_rows - g_ref[:, :, :, None, :])
            col_scale = _exp_leq0(g_ref[:, :, :, None, :] - g_cols)

            k_row_scaled = k_rows * row_scale
            k_col_scaled = k_cols * col_scale
            q_row_scaled = q_rows * row_scale

            A0 = jnp.matmul(k_row_scaled, jnp.swapaxes(k_col_scaled, -1, -2))
            A0 = A0 * beta_rows[:, :, :, :, None]
            Aqk = jnp.matmul(q_row_scaled, jnp.swapaxes(k_col_scaled, -1, -2))
            return A0, Aqk

        m_rows = []
        aqk_rows = []
        for i in range(NC):
            r0 = i * BC
            g_rows = g_all[:, :, :, r0 : r0 + BC, :]
            q_rows = q_all[:, :, :, r0 : r0 + BC, :]
            k_rows = k_all[:, :, :, r0 : r0 + BC, :]
            beta_rows = beta_all[:, :, :, r0 : r0 + BC]
            g_ref = g_all[:, :, :, r0, :]  # [B, H, N, K]

            m_blocks = []
            aqk_blocks = []
            for j in range(NC):
                if j == i:
                    m_blk, aqk_blk = diag_block(g_rows, q_rows, k_rows, beta_rows)
                elif j < i:
                    c0 = j * BC
                    g_cols = g_all[:, :, :, c0 : c0 + BC, :]
                    k_cols = k_all[:, :, :, c0 : c0 + BC, :]
                    m_blk, aqk_blk = offdiag_block(
                        g_rows,
                        q_rows,
                        k_rows,
                        beta_rows,
                        g_cols,
                        k_cols,
                        g_ref,
                    )
                else:
                    m_blk, aqk_blk = zeros_bc, zeros_bc

                m_blocks.append(m_blk)
                aqk_blocks.append(aqk_blk)

            m_rows.append(jnp.concatenate(m_blocks, axis=-1))
            aqk_rows.append(jnp.concatenate(aqk_blocks, axis=-1))

        M_full = jnp.concatenate(m_rows, axis=-2)
        Aqk_full = jnp.concatenate(aqk_rows, axis=-2)
        return M_full, Aqk_full

    M, Aqk = build_m_and_aqk(q, k, g, beta)

    # Packed RHS triangular solve for all chunks (keeps N batched; avoids materializing A).
    rhs_w = beta[..., None] * (k * exp_g)  # [B, H, N, BT, K]
    rhs_u = beta[..., None] * v            # [B, H, N, BT, V]
    rhs = jnp.concatenate([rhs_w, rhs_u], axis=-1)
    solved = jax.lax.linalg.triangular_solve(
        M,
        rhs,
        left_side=True,
        lower=True,
        unit_diagonal=True,
    )
    w = solved[..., :key_dim]
    u = solved[..., key_dim:]

    # ---------------------------------------------------------------------
    # Chunk scan: update recurrent state and produce outputs.
    # ---------------------------------------------------------------------
    if initial_state is None:
        S0 = jnp.zeros((batch, num_v_heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S0 = initial_state.astype(jnp.float32)

    def step(S, inputs):
        q_i, k_i, u_i, g_i, w_i, Aqk_i, exp_g_i = inputs
        # Shapes:
        # q_i: [B, H, BT, K]
        # k_i: [B, H, BT, K]
        # u_i: [B, H, BT, V]
        # g_i: [B, H, BT, K] (cumulative)
        # w_i: [B, H, BT, K]
        # Aqk_i: [B, H, BT, BT]
        # exp_g_i: [B, H, BT, K]

        # v_i = u_i - w_i @ S
        WS = jnp.einsum("bhck,bhkv->bhcv", w_i, S)
        v_i = u_i - WS

        # o = (q_i * exp(g_i)) @ S + Aqk @ v_i
        o_inter = jnp.einsum("bhck,bhkv->bhcv", q_i * exp_g_i, S)
        o_intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_i, v_i)
        o = o_inter + o_intra

        # State update (reuse exp_g_last from precomputed exp(g)).
        exp_g_last = exp_g_i[:, :, -1, :]  # [B, H, K]
        S_new = S * exp_g_last[..., None]

        # exp(g_last - g_t) = exp_g_last / exp_g_t, with safe division.
        ratio = jnp.where(exp_g_i > 0.0, exp_g_last[:, :, None, :] / exp_g_i, 0.0)
        kg_update = ratio * k_i  # [B, H, BT, K]
        S_new = S_new + jnp.einsum(
            "bhkc,bhcv->bhkv",
            jnp.transpose(kg_update, (0, 1, 3, 2)),
            v_i,
        )

        return S_new, o

    # Scan over chunks (N axis)
    q_scan = jnp.transpose(q, (2, 0, 1, 3, 4))
    k_scan = jnp.transpose(k, (2, 0, 1, 3, 4))
    u_scan = jnp.transpose(u, (2, 0, 1, 3, 4))
    g_scan = jnp.transpose(g, (2, 0, 1, 3, 4))
    w_scan = jnp.transpose(w, (2, 0, 1, 3, 4))
    Aqk_scan = jnp.transpose(Aqk, (2, 0, 1, 3, 4))
    exp_g_scan = jnp.transpose(exp_g, (2, 0, 1, 3, 4))

    final_state, o_chunks = lax.scan(step, S0, (q_scan, k_scan, u_scan, g_scan, w_scan, Aqk_scan, exp_g_scan))

    # Reassemble outputs: [N, B, H, BT, V] -> [B, H, N*BT, V]
    o_chunks = jnp.transpose(o_chunks, (1, 2, 0, 3, 4)).reshape(batch, num_v_heads, padded_len, value_dim)
    if pad_len:
        o_chunks = o_chunks[:, :, :seq_len, :]

    return o_chunks.astype(v.dtype), final_state


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

        # Mode selection (match project convention: chunk for training, recurrent for generation)
        if mode is None:
            mode = "recurrent" if seq_len == 1 else "chunk"

        if mode == "recurrent" or seq_len == 1:
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
