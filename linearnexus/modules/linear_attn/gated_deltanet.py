"""
Gated DeltaNet: Combining Mamba2's Gating with Delta Rule

Paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
       https://arxiv.org/abs/2412.06464

Mathematical Foundation
=======================

Gated DeltaNet extends DeltaNet by adding an exponential decay gate before the
delta rule update. This allows the model to learn when to forget old associations,
similar to Mamba2's gating mechanism.

Core Recurrence (Token-by-Token)
--------------------------------
For each token t:
    1. Apply decay:        S_decayed = exp(g_t) * S_{t-1}
    2. Retrieve old value: v_old = S_decayed @ k_t  
    3. Compute delta:      v_delta = beta_t * (v_t - v_old)
    4. Update state:       S_t = S_decayed + k_t @ v_delta^T
    5. Query output:       o_t = q_t @ S_t

Where:
    - q_t, k_t: query and key vectors, shape [d_k]  
    - v_t: value vector, shape [d_v]
    - beta_t: learning rate in [0, 1] or [0, 2] if allow_neg_eigval
    - g_t: decay gate (typically negative), computed as:
           g_t = -A_log.exp() * softplus(a_t + dt_bias)
    - S_t: state matrix, shape [d_k, d_v]

Matrix Form:
    S_t = exp(g_t) * S_{t-1} + k_t @ (beta_t * (v_t - exp(g_t)*S_{t-1} @ k_t))^T

Key Differences from DeltaNet:
------------------------------
1. **Decay Gate**: State is decayed before delta update, allowing forgetting
2. **Grouped Value Attention (GVA)**: Q/K can have fewer heads than V
   - num_heads (for Q/K) <= num_v_heads (for V)
   - Q/K heads are repeated to match V heads
3. **Dual Projections**: Uses a_proj (for decay) + b_proj (for beta) instead of single beta_proj
4. **L2 Normalization**: Q and K are L2-normalized for stability
5. **Parameter Allocation**: Similar to Mamba2 (~6*hidden_size² total)

Architecture Details:
--------------------
- expand_v=2: Value dimension expanded (like Mamba2)
- head_dim=256: Fixed head dimension
- Short convolutions on Q, K, V for local context
- Optional output gating with FusedRMSNormGated

Reference Implementation
------------------------
Based on FLA's gated_delta_rule ops and gated_deltanet layer.
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
class GatedDeltaNetState:
    """Cache for Gated DeltaNet autoregressive generation.

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
    ) -> "GatedDeltaNetState":
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
# Core Gated Delta Rule Operations
# =============================================================================


def gated_delta_rule_recurrent(
    q: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    k: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    v: jax.Array,  # [batch, num_v_heads, seq_len, value_dim]
    g: jax.Array,  # [batch, num_v_heads, seq_len] - decay (typically negative)
    beta: jax.Array,  # [batch, num_v_heads, seq_len] - learning rate
    scale: Optional[float] = None,
    initial_state: Optional[jax.Array] = None,  # [batch, num_v_heads, key_dim, value_dim]
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Token-by-token gated delta rule recurrence.

    The key difference from DeltaNet is the decay gate applied before the
    delta rule update: S = exp(g) * S + k @ (beta * (v - exp(g)*S @ k))^T

    Args:
        q: Query tensor [batch, num_v_heads, seq_len, key_dim]
        k: Key tensor [batch, num_v_heads, seq_len, key_dim]
        v: Value tensor [batch, num_v_heads, seq_len, value_dim]
        g: Decay gate [batch, num_v_heads, seq_len] (typically negative)
        beta: Learning rate [batch, num_v_heads, seq_len]
        scale: Query scaling factor (default: 1.0, assumes pre-scaled)
        initial_state: Initial state [batch, num_v_heads, key_dim, value_dim]
        use_qk_l2norm: Whether to L2-normalize Q and K

    Returns:
        output: Output tensor [batch, num_v_heads, seq_len, value_dim]
        final_state: Final state [batch, num_v_heads, key_dim, value_dim]
    """
    batch, num_v_heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    orig_dtype = q.dtype

    if scale is None:
        scale = key_dim**-0.5

    # Compute recurrence in float32 like FLA kernels for stability.
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, num_v_heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    def step(S, inputs):
        """Single step with gated decay + delta rule."""
        k_t, v_t, q_t, g_t, beta_t = inputs
        # k_t: [batch, heads, key_dim]
        # v_t: [batch, heads, value_dim]
        # g_t, beta_t: [batch, heads]

        # L2 normalize Q and K if requested
        if use_qk_l2norm:
            q_t = q_t / (jnp.linalg.norm(q_t, axis=-1, keepdims=True) + 1e-6)
            k_t = k_t / (jnp.linalg.norm(k_t, axis=-1, keepdims=True) + 1e-6)
        
        q_t = q_t * jnp.asarray(scale, dtype=jnp.float32)

        # 1. Apply decay to state: S = exp(g) * S
        decay = jnp.exp(g_t)[..., None, None]  # [batch, heads, 1, 1]
        S_decayed = S * decay

        # 2. Retrieve old value: v_old = S_decayed @ k
        v_old = jnp.einsum("bhkv,bhk->bhv", S_decayed, k_t)

        # 3. Compute delta: v_delta = beta * (v - v_old)
        v_delta = beta_t[..., None] * (v_t - v_old)

        # 4. Update state: S = S_decayed + k @ v_delta^T
        S_new = S_decayed + jnp.einsum("bhk,bhv->bhkv", k_t, v_delta)

        # 5. Query output: o = q @ S
        o_t = jnp.einsum("bhk,bhkv->bhv", q_t, S_new)

        return S_new, o_t

    # Transpose for scan: [seq, batch, heads, dim]
    k_seq = jnp.transpose(k, (2, 0, 1, 3))
    v_seq = jnp.transpose(v, (2, 0, 1, 3))
    q_seq = jnp.transpose(q, (2, 0, 1, 3))
    g_seq = jnp.transpose(g, (2, 0, 1))
    beta_seq = jnp.transpose(beta, (2, 0, 1))

    final_state, outputs = lax.scan(step, S, (k_seq, v_seq, q_seq, g_seq, beta_seq))

    # Transpose output back: [batch, heads, seq, value_dim]
    output = jnp.transpose(outputs, (1, 2, 0, 3))

    return output.astype(orig_dtype), final_state


def gated_delta_rule_chunkwise(
    q: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    k: jax.Array,  # [batch, num_v_heads, seq_len, key_dim]
    v: jax.Array,  # [batch, num_v_heads, seq_len, value_dim]
    g: jax.Array,  # [batch, num_v_heads, seq_len]
    beta: jax.Array,  # [batch, num_v_heads, seq_len]
    scale: Optional[float] = None,
    initial_state: Optional[jax.Array] = None,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Chunkwise parallel gated delta rule (WY-style).

    This is a chunk-parallel algorithm (triangular solve per chunk + scan across
    chunks) modeled after the chunkwise implementation used in FLA, and closely
    patterned after this repo's KDA implementation in
    `linearnexus/modules/linear_attn/kda.py`.

    Args:
        q: Query [batch, num_v_heads, seq_len, key_dim]
        k: Key [batch, num_v_heads, seq_len, key_dim]
        v: Value [batch, num_v_heads, seq_len, value_dim]
        g: Decay gate [batch, num_v_heads, seq_len]
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
    orig_dtype = q.dtype

    if scale is None:
        scale = key_dim**-0.5

    # L2 normalize Q and K if requested
    if use_qk_l2norm:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    # Scale queries (matches reference kernels)
    q = q * jnp.asarray(scale, dtype=jnp.float32)

    # Float32 compute like FLA kernels.
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_len)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))

    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size
    BT = chunk_size

    # Canonical layout: chunk-major [N, B, H, BT, ...] (matches kda_chunkwise)
    q = q.reshape(batch, num_v_heads, num_chunks, BT, key_dim)
    k = k.reshape(batch, num_v_heads, num_chunks, BT, key_dim)
    v = v.reshape(batch, num_v_heads, num_chunks, BT, value_dim)
    g = g.reshape(batch, num_v_heads, num_chunks, BT)
    beta = beta.reshape(batch, num_v_heads, num_chunks, BT)

    q = jnp.transpose(q, (2, 0, 1, 3, 4))
    k = jnp.transpose(k, (2, 0, 1, 3, 4))
    v = jnp.transpose(v, (2, 0, 1, 3, 4))
    g = jnp.transpose(g, (2, 0, 1, 3))
    beta = jnp.transpose(beta, (2, 0, 1, 3))

    # Cumulative gates within each chunk: G_t = cumsum(g_t)
    g = lax.cumsum(g, axis=3)  # [N, B, H, BT]
    exp_g = jnp.exp(g)  # [N, B, H, BT] (safe: g typically <= 0)

    # Tile within each chunk (BC=16 as in FLA kernels / kda_chunkwise)
    BC = 16
    if BT % BC != 0:
        raise ValueError(f"chunk_size (BT={BT}) must be divisible by BC={BC}.")
    NC = BT // BC

    tril_inclusive = jnp.tril(jnp.ones((BC, BC), dtype=bool), k=0)
    tril_strict = jnp.tril(jnp.ones((BC, BC), dtype=bool), k=-1)

    def _exp_leq0(x: jax.Array) -> jax.Array:
        return jnp.exp(jnp.minimum(x, 0.0)).astype(jnp.float32)

    def _masked_exp(d: jax.Array, mask: jax.Array) -> jax.Array:
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
            q_all, k_all: [N, B, H, BT, K]
            g_all:        [N, B, H, BT]
            beta_all:     [N, B, H, BT]
        Returns:
            M:   [N, B, H, BT, BT]
            Aqk: [N, B, H, BT, BT]
        """
        N_, B_, H_, _, _ = q_all.shape
        eye_bc = jnp.eye(BC, dtype=jnp.float32)[None, None, None, :, :]
        zeros_bc = jnp.zeros((N_, B_, H_, BC, BC), dtype=jnp.float32)

        def diag_block(
            g_rows: jax.Array,
            q_rows: jax.Array,
            k_rows: jax.Array,
            beta_rows: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            # g_rows: [N,B,H,BC]
            d = g_rows[:, :, :, :, None] - g_rows[:, :, :, None, :]
            exp_rel_strict = _masked_exp(d, tril_strict[None, None, None, :, :])
            exp_rel_incl = _masked_exp(d, tril_inclusive[None, None, None, :, :])

            # A0[i,j] = sum_k k_i[k] * k_j[k] * exp(g_i - g_j) for i>j
            A0 = jnp.sum(
                (k_rows[:, :, :, :, None, :] * k_rows[:, :, :, None, :, :])
                * exp_rel_strict[..., None],
                axis=-1,
            )
            Aqk = jnp.sum(
                (q_rows[:, :, :, :, None, :] * k_rows[:, :, :, None, :, :])
                * exp_rel_incl[..., None],
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
            # g_ref: [N,B,H]
            row_scale = _exp_leq0(g_rows - g_ref[:, :, :, None])
            col_scale = _exp_leq0(g_ref[:, :, :, None] - g_cols)

            k_row_scaled = k_rows * row_scale[..., None]
            k_col_scaled = k_cols * col_scale[..., None]
            q_row_scaled = q_rows * row_scale[..., None]

            A0 = jnp.matmul(k_row_scaled, jnp.swapaxes(k_col_scaled, -1, -2))
            A0 = A0 * beta_rows[:, :, :, :, None]
            Aqk = jnp.matmul(q_row_scaled, jnp.swapaxes(k_col_scaled, -1, -2))
            return A0, Aqk

        m_rows = []
        aqk_rows = []
        for i in range(NC):
            r0 = i * BC
            g_rows = g_all[:, :, :, r0 : r0 + BC]
            q_rows = q_all[:, :, :, r0 : r0 + BC, :]
            k_rows = k_all[:, :, :, r0 : r0 + BC, :]
            beta_rows = beta_all[:, :, :, r0 : r0 + BC]
            g_ref = g_all[:, :, :, r0]  # [N,B,H]

            m_blocks = []
            aqk_blocks = []
            for j in range(NC):
                if j == i:
                    m_blk, aqk_blk = diag_block(g_rows, q_rows, k_rows, beta_rows)
                elif j < i:
                    c0 = j * BC
                    g_cols = g_all[:, :, :, c0 : c0 + BC]
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

    # Packed RHS triangular solve for all chunks.
    rhs_w = beta[..., None] * (k * exp_g[..., None])  # [N,B,H,BT,K]
    rhs_u = beta[..., None] * v  # [N,B,H,BT,V]
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
        # q_i: [B,H,BT,K]
        # k_i: [B,H,BT,K]
        # u_i: [B,H,BT,V]
        # g_i: [B,H,BT] (cumulative)
        # w_i: [B,H,BT,K]
        # Aqk_i: [B,H,BT,BT]
        # exp_g_i: [B,H,BT]

        # v_i = u_i - w_i @ S
        WS = jnp.einsum("bhck,bhkv->bhcv", w_i, S)
        v_i = u_i - WS

        # o = (q_i * exp(g_i)) @ S + Aqk @ v_i
        o_inter = jnp.einsum("bhck,bhkv->bhcv", q_i * exp_g_i[..., None], S)
        o_intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_i, v_i)
        o = o_inter + o_intra

        # State update
        exp_g_last = exp_g_i[:, :, -1]  # [B,H]
        S_new = S * exp_g_last[..., None, None]

        # exp(g_last - g_t) = exp_g_last / exp_g_t
        ratio = jnp.where(exp_g_i > 0.0, exp_g_last[:, :, None] / exp_g_i, 0.0)
        kg_update = ratio[..., None] * k_i  # [B,H,BT,K]
        S_new = S_new + jnp.einsum(
            "bhkc,bhcv->bhkv",
            jnp.transpose(kg_update, (0, 1, 3, 2)),
            v_i,
        )

        return S_new, o

    final_state, o_chunks = lax.scan(step, S0, (q, k, u, g, w, Aqk, exp_g))

    # Reassemble outputs: [N,B,H,BT,V] -> [B,H,N*BT,V]
    o_chunks = jnp.transpose(o_chunks, (1, 2, 0, 3, 4)).reshape(
        batch, num_v_heads, padded_len, value_dim
    )

    # Remove padding
    if pad_len > 0:
        o_chunks = o_chunks[:, :, :seq_len, :]

    return o_chunks.astype(orig_dtype), final_state


def gated_delta_rule_step(
    q: jax.Array,  # [batch, num_v_heads, key_dim]
    k: jax.Array,  # [batch, num_v_heads, key_dim]
    v: jax.Array,  # [batch, num_v_heads, value_dim]
    g: jax.Array,  # [batch, num_v_heads]
    beta: jax.Array,  # [batch, num_v_heads]
    state: jax.Array,  # [batch, num_v_heads, key_dim, value_dim]
    scale: Optional[float] = None,
    use_qk_l2norm: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Single step of gated delta rule for autoregressive generation.

    Args:
        q: Query [batch, num_v_heads, key_dim]
        k: Key [batch, num_v_heads, key_dim]
        v: Value [batch, num_v_heads, value_dim]
        g: Decay gate [batch, num_v_heads]
        beta: Learning rate [batch, num_v_heads]
        state: Current state [batch, num_v_heads, key_dim, value_dim]
        scale: Query scaling factor
        use_qk_l2norm: Whether to L2-normalize Q and K

    Returns:
        output: [batch, num_v_heads, value_dim]
        new_state: [batch, num_v_heads, key_dim, value_dim]
    """
    orig_dtype = q.dtype
    key_dim = q.shape[-1]
    if scale is None:
        scale = key_dim**-0.5

    # Float32 compute like FLA kernels.
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    state = state.astype(jnp.float32)

    # L2 normalize Q and K
    if use_qk_l2norm:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    
    q = q * jnp.asarray(scale, dtype=jnp.float32)

    # 1. Decay state
    decay = jnp.exp(g)[..., None, None]
    S_decayed = state * decay

    # 2. Retrieve and update
    v_old = jnp.einsum("bhkv,bhk->bhv", S_decayed, k)
    v_delta = beta[..., None] * (v - v_old)
    new_state = S_decayed + jnp.einsum("bhk,bhv->bhkv", k, v_delta)

    # 3. Query
    output = jnp.einsum("bhk,bhkv->bhv", q, new_state)

    return output.astype(orig_dtype), new_state


# =============================================================================
# Gated DeltaNet Block
# =============================================================================


class GatedDeltaNetBlock(nnx.Module):
    """Gated DeltaNet block combining Mamba2 gating with delta rule.

    This implements the Gated Delta Network from "Gated Delta Networks:
    Improving Mamba2 with Delta Rule" (arXiv:2412.06464).

    Key features:
    - Exponential decay gate (like Mamba2) before delta rule update
    - Grouped Value Attention (GVA) with potentially more V heads than Q/K
    - L2 normalization on Q and K for stability
    - Short convolutions on Q, K, V for local context
    - Parameter allocation similar to Mamba2 (~6*hidden² total)

    Args:
        hidden_size: Model dimension
        num_heads: Number of heads for Q/K
        num_v_heads: Number of heads for V (GVA if > num_heads)
        head_dim: Dimension per Q/K head (default: 256)
        expand_v: Value dimension expansion factor (default: 2)
        use_short_conv: Whether to use short convolutions
        conv_size: Kernel size for short convolutions
        use_gate: Whether to use output gating
        allow_neg_eigval: If True, beta can be in [0, 2] instead of [0, 1]
        use_qk_l2norm: Whether to L2-normalize Q and K
        norm_eps: Epsilon for layer normalization
        rngs: NNx random number generators
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        num_v_heads: Optional[int] = None,
        head_dim: int = 256,
        expand_v: float = 2.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        use_qk_l2norm: bool = True,
        norm_eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.use_qk_l2norm = use_qk_l2norm

        # Compute dimensions
        self.key_dim = num_heads * head_dim  # Total Q/K dimension
        self.head_v_dim = int(head_dim * expand_v)  # V dimension per head
        self.value_dim = self.num_v_heads * self.head_v_dim  # Total V dimension

        # Validate GVA constraint
        assert self.num_v_heads % self.num_heads == 0, \
            f"num_v_heads ({self.num_v_heads}) must be divisible by num_heads ({self.num_heads})"

        # Input normalization
        self.norm = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)

        # Projections
        self.q_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, self.value_dim, use_bias=False, rngs=rngs)

        # Decay parameters (a_proj + A_log + dt_bias)
        self.a_proj = nnx.Linear(hidden_size, self.num_v_heads, use_bias=False, rngs=rngs)
        
        # Initialize A_log uniform in [1e-4, 16], store log for stability
        # Note: minval must be > 0 to avoid log(0) = -inf
        key = rngs.params()
        A = jax.random.uniform(key, (self.num_v_heads,), minval=1e-4, maxval=16.0)
        self.A_log = nnx.Param(jnp.log(A))

        # Initialize dt_bias using Mamba2's scheme
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        key = rngs.params()
        dt = jnp.exp(
            jax.random.uniform(key, (self.num_v_heads,))
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = jnp.clip(dt, dt_init_floor)
        # Inverse softplus
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_bias = nnx.Param(inv_dt)

        # Beta projection
        self.b_proj = nnx.Linear(hidden_size, self.num_v_heads, use_bias=False, rngs=rngs)

        # Short convolutions
        if use_short_conv:
            # Conv weights: [kernel_size, channels]
            key = rngs.params()
            self.q_conv_weight = nnx.Param(
                jax.random.normal(key, (conv_size, self.key_dim)) * 0.02
            )
            key = rngs.params()
            self.k_conv_weight = nnx.Param(
                jax.random.normal(key, (conv_size, self.key_dim)) * 0.02
            )
            key = rngs.params()
            self.v_conv_weight = nnx.Param(
                jax.random.normal(key, (conv_size, self.value_dim)) * 0.02
            )

        # Output gating and projection
        if use_gate:
            self.g_proj = nnx.Linear(hidden_size, self.value_dim, use_bias=False, rngs=rngs)
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, rngs=rngs)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, rngs=rngs)
            
        self.o_proj = nnx.Linear(self.value_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[GatedDeltaNetState] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[jax.Array, Optional[GatedDeltaNetState]]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            state: Optional cached state for generation
            mask: Optional attention mask (unused, for API compatibility)
            mode: Processing mode ("chunk" for training, "recurrent" for generation)

        Returns:
            output: Output tensor [batch, seq_len, hidden_size]
            new_state: Updated state (if state was provided or mode="recurrent")
        """
        batch, seq_len, _ = x.shape
        residual = x

        # Normalize input
        x = self.norm(x)

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, key_dim]
        k = self.k_proj(x)  # [batch, seq, key_dim]
        v = self.v_proj(x)  # [batch, seq, value_dim]

        # Get conv states if available
        conv_state_q = state.conv_state_q if state is not None else None
        conv_state_k = state.conv_state_k if state is not None else None
        conv_state_v = state.conv_state_v if state is not None else None

        # Apply short convolutions + SiLU
        if self.use_short_conv:
            q, conv_state_q = depthwise_conv1d_causal(
                q, self.q_conv_weight.value, bias=None, cache=conv_state_q
            )
            q = jax.nn.silu(q)
            
            k, conv_state_k = depthwise_conv1d_causal(
                k, self.k_conv_weight.value, bias=None, cache=conv_state_k
            )
            k = jax.nn.silu(k)
            
            v, conv_state_v = depthwise_conv1d_causal(
                v, self.v_conv_weight.value, bias=None, cache=conv_state_v
            )
            v = jax.nn.silu(v)
        else:
            q = jax.nn.silu(q)
            k = jax.nn.silu(k)
            v = jax.nn.silu(v)

        # Reshape to heads
        # Q, K: [batch, seq, num_heads, head_dim]
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        # V: [batch, seq, num_v_heads, head_v_dim]
        v = v.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        # Transpose to [batch, heads, seq, dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply GVA: repeat Q/K heads to match V heads
        if self.num_v_heads > self.num_heads:
            repeat_factor = self.num_v_heads // self.num_heads
            q = jnp.repeat(q, repeat_factor, axis=1)
            k = jnp.repeat(k, repeat_factor, axis=1)

        # Compute beta (learning rate) - use x (already normalized hidden_states)
        beta = jax.nn.sigmoid(self.b_proj(x).astype(jnp.float32))  # [batch, seq, num_v_heads]
        if self.allow_neg_eigval:
            beta = beta * 2.0
        beta = jnp.transpose(beta, (0, 2, 1))  # [batch, num_v_heads, seq]

        # Compute decay gate: g = -A * softplus(a + dt_bias)
        # Note: Use x (normalized hidden_states), matching FLA reference
        a = self.a_proj(x).astype(jnp.float32)  # [batch, seq, num_v_heads]
        g = -jnp.exp(self.A_log.value.astype(jnp.float32)) * jax.nn.softplus(
            a + self.dt_bias.value.astype(jnp.float32)
        )
        g = jnp.transpose(g, (0, 2, 1))  # [batch, num_v_heads, seq]

        # Get recurrent state
        recurrent_state = state.S if state is not None else None

        # Select mode
        if mode is None:
            mode = "recurrent" if seq_len == 1 else "chunk"

        # Apply gated delta rule
        scale = self.head_dim ** -0.5
        
        # Use recurrent mode for short sequences, chunkwise for longer
        # Chunkwise now uses token-by-token scan within each chunk (numerically stable)
        if mode == "recurrent" or seq_len <= 32:
            o, new_recurrent_state = gated_delta_rule_recurrent(
                q, k, v, g, beta,
                scale=scale,
                initial_state=recurrent_state,
                use_qk_l2norm=self.use_qk_l2norm,
            )
        else:
            o, new_recurrent_state = gated_delta_rule_chunkwise(
                q, k, v, g, beta,
                scale=scale,
                initial_state=recurrent_state,
                chunk_size=64,
                use_qk_l2norm=self.use_qk_l2norm,
            )

        # Apply output gating
        if self.use_gate:
            # g_out: [batch, seq, num_v_heads, head_v_dim]
            # Note: Use x (normalized hidden_states), matching FLA reference
            g_out = self.g_proj(x)
            g_out = g_out.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
            # o: [batch, num_v_heads, seq, head_v_dim] -> [batch, seq, num_v_heads, head_v_dim]
            o = jnp.transpose(o, (0, 2, 1, 3))
            # Gated RMSNorm: norm(o) * sigmoid(g)
            o = self.o_norm(o) * jax.nn.sigmoid(g_out)
        else:
            o = jnp.transpose(o, (0, 2, 1, 3))
            o = self.o_norm(o)

        # Reshape and project output
        o = o.reshape(batch, seq_len, self.value_dim)
        output = self.o_proj(o)

        # Residual connection
        output = output + residual

        # Build new state
        new_state = None
        if state is not None or mode == "recurrent":
            new_state = GatedDeltaNetState(
                S=new_recurrent_state,
                conv_state_q=conv_state_q,
                conv_state_k=conv_state_k,
                conv_state_v=conv_state_v,
            )

        return output, new_state

    def init_state(self, batch_size: int) -> GatedDeltaNetState:
        """Initialize empty state for autoregressive generation.

        Args:
            batch_size: Batch dimension

        Returns:
            Empty GatedDeltaNetState
        """
        return GatedDeltaNetState.zeros(
            batch_size=batch_size,
            num_v_heads=self.num_v_heads,
            key_dim_per_head=self.head_dim,
            value_dim_per_head=self.head_v_dim,
            key_dim_total=self.key_dim,
            value_dim=self.value_dim,
            conv_size=self.conv_size,
            use_conv=self.use_short_conv,
        )
