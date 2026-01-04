"""
DeltaNet: Linear Attention with the Delta Rule

Paper: "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
       https://arxiv.org/abs/2406.06484

Mathematical Foundation
=======================

DeltaNet uses the delta rule to update a linear attention state matrix S.
Unlike standard linear attention (S_t = S_{t-1} + k_t v_t^T), DeltaNet first
removes the old value associated with k_t before adding the new one.

Core Recurrence (Token-by-Token)
--------------------------------
For each token t:
    1. Retrieve old value: v_old = S_{t-1} @ k_t
    2. Compute delta:      v_delta = beta_t * (v_t - v_old)
    3. Update state:       S_t = S_{t-1} + k_t @ v_delta^T
    4. Query output:       o_t = q_t @ S_t

Where:
    - q_t, k_t: query and key vectors, shape [d_k]
    - v_t: value vector, shape [d_v]
    - beta_t: scalar learning rate in [0, 1], typically sigmoid(linear(x))
    - S_t: state matrix, shape [d_k, d_v]

Matrix Form (Equivalent):
    S_t = (I - beta_t * k_t k_t^T) S_{t-1} + beta_t * k_t v_t^T

This formulation removes the contribution of k_t from the old state before
adding the new k_t -> v_t association, making it a proper "update" rather
than just accumulation.

Chunkwise Parallel Form (WY Representation)
-------------------------------------------
For efficient training, we process chunks in parallel using the WY decomposition.
Given a chunk of L tokens with Q, K, V matrices [L, d]:

1. Build lower triangular matrix: A = I + diag(beta) @ tril(K K^T, -1)
2. Solve for correction:          T = A^{-1} @ diag(beta)
3. Compute corrected terms:       W = T @ K,  U = T @ V

Intra-chunk output (parallel):
    O_intra = (Q K^T ⊙ causal_mask) @ (U - W @ S_0)

State update for next chunk:
    S_{new} = S_0 + K^T @ (U - W @ S_0)

Full output:
    O = Q @ S_0 + O_intra

Reference Implementation
------------------------
This follows the exact formulation from FLA's naive.py:
    for i in range(seq_len):
        v_new = v[i] - (S * k[i][..., None]).sum(-2)  # S @ k (retrieve)
        v_new = beta[i] * v_new                        # scale by learning rate
        S = S + k[i][:, None] * v_new[None, :]         # outer product update
        o[i] = (q[i][:, None] * S).sum(-2)             # q @ S
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal
import jax
import jax.numpy as jnp
from jax import lax
import flax.nnx as nnx

from ...core.conv import depthwise_conv1d_causal


# =============================================================================
# State Cache
# =============================================================================

@dataclass
class DeltaNetState:
    """Cache for DeltaNet autoregressive generation.

    Attributes:
        S: Recurrent state matrix [batch, heads, key_dim, value_dim]
        conv_state: Optional conv state [batch, kernel-1, channels]
    """

    S: jax.Array  # [batch, heads, key_dim, value_dim]
    conv_state: Optional[jax.Array] = None  # [batch, kernel-1, channels]

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        conv_channels: Optional[int] = None,
        conv_size: int = 4,
    ) -> "DeltaNetState":
        """Initialize empty state.

        Args:
            batch_size: Batch dimension
            num_heads: Number of attention heads
            key_dim: Dimension of keys per head
            value_dim: Dimension of values per head
            conv_channels: Number of channels for conv state (None = no conv)
            conv_size: Convolution kernel size
        """
        S = jnp.zeros((batch_size, num_heads, key_dim, value_dim))
        conv_state = None
        if conv_channels is not None:
            # Conv cache shape: [batch, kernel-1, channels]
            conv_state = jnp.zeros((batch_size, conv_size - 1, conv_channels))
        return cls(S=S, conv_state=conv_state)


# =============================================================================
# Core Delta Rule Operations
# =============================================================================


def delta_rule_recurrent(
    q: jax.Array,  # [batch, heads, seq_len, key_dim]
    k: jax.Array,  # [batch, heads, seq_len, key_dim]
    v: jax.Array,  # [batch, heads, seq_len, value_dim]
    beta: jax.Array,  # [batch, heads, seq_len]
    initial_state: Optional[jax.Array] = None,  # [batch, heads, key_dim, value_dim]
    scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Token-by-token delta rule recurrence (Reference Implementation).

    This follows the exact formulation from FLA's naive.py.

    Args:
        q: Query tensor [batch, heads, seq_len, key_dim]
        k: Key tensor [batch, heads, seq_len, key_dim]
        v: Value tensor [batch, heads, seq_len, value_dim]
        beta: Learning rate [batch, heads, seq_len]
        initial_state: Initial state [batch, heads, key_dim, value_dim]

    Returns:
        output: Output tensor [batch, heads, seq_len, value_dim]
        final_state: Final state [batch, heads, key_dim, value_dim]
    """
    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    orig_dtype = q.dtype

    if scale is None:
        scale = key_dim**-0.5

    # Handle beta shape
    if beta.ndim == 4:
        beta = beta.squeeze(-1)

    # Compute in float32 for numerical stability (matches FLA kernels).
    q = q.astype(jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    def step(S, inputs):
        """Single step: S @ k -> delta -> update -> query."""
        k_t, v_t, q_t, beta_t = inputs
        # k_t: [batch, heads, key_dim]
        # v_t: [batch, heads, value_dim]
        # beta_t: [batch, heads]

        # 1. Retrieve old value: v_old = S @ k_t
        v_old = jnp.einsum("bhkv,bhk->bhv", S, k_t)

        # 2. Compute delta: v_delta = beta * (v - v_old)
        v_delta = beta_t[..., None] * (v_t - v_old)

        # 3. Update state: S = S + k @ v_delta^T
        S_new = S + jnp.einsum("bhk,bhv->bhkv", k_t, v_delta)

        # 4. Query output: o = q @ S
        o_t = jnp.einsum("bhk,bhkv->bhv", q_t, S_new)

        return S_new, o_t

    # Transpose for scan: [seq, batch, heads, dim]
    k_seq = jnp.transpose(k, (2, 0, 1, 3))
    v_seq = jnp.transpose(v, (2, 0, 1, 3))
    q_seq = jnp.transpose(q, (2, 0, 1, 3))
    beta_seq = jnp.transpose(beta, (2, 0, 1))

    final_state, outputs = lax.scan(step, S, (k_seq, v_seq, q_seq, beta_seq))

    # Transpose output back: [batch, heads, seq, value_dim]
    output = jnp.transpose(outputs, (1, 2, 0, 3))

    return output.astype(orig_dtype), final_state


def delta_rule_chunkwise(
    q: jax.Array,  # [batch, heads, seq_len, key_dim]
    k: jax.Array,  # [batch, heads, seq_len, key_dim]
    v: jax.Array,  # [batch, heads, seq_len, value_dim]
    beta: jax.Array,  # [batch, heads, seq_len]
    initial_state: Optional[jax.Array] = None,
    chunk_size: int = 64,
    scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Chunkwise parallel delta rule using WY representation.

    For each chunk:
        This matches the Flash-Linear-Attention (FLA) kernels:

        1. Build strictly-lower A where A[i, j] = beta[i] * <k[i], k[j]> for i > j
        2. Compute A_inv = (I + A)^{-1}
        3. Compute w = A_inv @ (beta * k), u = A_inv @ (beta * v)
        4. v_new = u - w @ S_0
        5. O = q @ S_0 + (q k^T ⊙ causal) @ v_new
        6. S_new = S_0 + k^T @ v_new

    Args:
        q: Query [batch, heads, seq_len, key_dim]
        k: Key [batch, heads, seq_len, key_dim]
        v: Value [batch, heads, seq_len, value_dim]
        beta: Learning rate [batch, heads, seq_len]
        initial_state: Initial state [batch, heads, key_dim, value_dim]
        chunk_size: Size of each chunk

    Returns:
        output: [batch, heads, seq_len, value_dim]
        final_state: [batch, heads, key_dim, value_dim]
    """
    batch, heads, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    orig_dtype = q.dtype

    if scale is None:
        scale = key_dim**-0.5

    if beta.ndim == 4:
        beta = beta.squeeze(-1)

    # Compute in float32 for numerical stability and apply query scaling.
    q = q.astype(jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_len)))

    padded_len = q.shape[2]
    num_chunks = padded_len // chunk_size

    # Reshape: [batch, heads, num_chunks, chunk_size, dim]
    q_chunks = q.reshape(batch, heads, num_chunks, chunk_size, key_dim)
    k_chunks = k.reshape(batch, heads, num_chunks, chunk_size, key_dim)
    v_chunks = v.reshape(batch, heads, num_chunks, chunk_size, value_dim)
    beta_chunks = beta.reshape(batch, heads, num_chunks, chunk_size)

    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    eye_L = jnp.eye(chunk_size, dtype=jnp.float32)
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))

    def process_chunk(S, chunk_inputs):
        """Process one chunk using WY decomposition."""
        q_c, k_c, v_c, beta_c = chunk_inputs

        # A = tril(beta * (K K^T), -1)  (strictly lower)
        kk = jnp.einsum("bhik,bhjk->bhij", k_c, k_c)
        A_strict = jnp.tril(beta_c[..., :, None] * kk, k=-1)

        # A_inv = (I + A)^{-1}
        A_full = eye_L + A_strict
        eye_batched = jnp.broadcast_to(eye_L, A_full.shape)
        A_inv = jax.lax.linalg.triangular_solve(
            A_full,
            eye_batched,
            left_side=True,
            lower=True,
            transpose_a=False,
            conjugate_a=False,
            unit_diagonal=False,
        )

        # w = A_inv @ (beta * k), u = A_inv @ (beta * v)
        k_beta = k_c * beta_c[..., :, None]
        v_beta = v_c * beta_c[..., :, None]
        w = jnp.einsum("bhij,bhjk->bhik", A_inv, k_beta)
        u = jnp.einsum("bhij,bhjv->bhiv", A_inv, v_beta)

        # v_new = u - w @ S
        wS = jnp.einsum("bhik,bhkv->bhiv", w, S)
        v_new = u - wS

        # O = q @ S + (q k^T ⊙ causal) @ v_new
        O_inter = jnp.einsum("bhik,bhkv->bhiv", q_c, S)
        qk = jnp.einsum("bhik,bhjk->bhij", q_c, k_c)
        O_intra = jnp.einsum("bhij,bhjv->bhiv", qk * causal_mask, v_new)
        O_chunk = O_inter + O_intra

        # S_new = S + k^T @ v_new
        S_new = S + jnp.einsum("bhik,bhiv->bhkv", k_c, v_new)

        return S_new, O_chunk

    # Process chunks
    final_state, outputs = lax.scan(
        process_chunk,
        S,
        (
            jnp.transpose(q_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(k_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(v_chunks, (2, 0, 1, 3, 4)),
            jnp.transpose(beta_chunks, (2, 0, 1, 3)),
        ),
    )

    # Reshape output
    output = jnp.transpose(outputs, (1, 2, 0, 3, 4))
    output = output.reshape(batch, heads, padded_len, value_dim)

    # Remove padding
    if pad_len > 0:
        output = output[:, :, :seq_len, :]

    return output.astype(orig_dtype), final_state


def delta_rule_step(
    q: jax.Array,  # [batch, heads, key_dim]
    k: jax.Array,  # [batch, heads, key_dim]
    v: jax.Array,  # [batch, heads, value_dim]
    beta: jax.Array,  # [batch, heads]
    state: jax.Array,  # [batch, heads, key_dim, value_dim]
    scale: Optional[float] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Single step of delta rule for autoregressive generation.

    Args:
        q: Query [batch, heads, key_dim]
        k: Key [batch, heads, key_dim]
        v: Value [batch, heads, value_dim]
        beta: Learning rate [batch, heads]
        state: Current state [batch, heads, key_dim, value_dim]

    Returns:
        output: [batch, heads, value_dim]
        new_state: [batch, heads, key_dim, value_dim]
    """
    if beta.ndim == 3:
        beta = beta.squeeze(-1)

    key_dim = q.shape[-1]
    if scale is None:
        scale = key_dim**-0.5

    orig_dtype = q.dtype
    q = q.astype(jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    state = state.astype(jnp.float32)

    v_old = jnp.einsum("bhkv,bhk->bhv", state, k)
    v_delta = beta[..., None] * (v - v_old)
    new_state = state + jnp.einsum("bhk,bhv->bhkv", k, v_delta)
    output = jnp.einsum("bhk,bhkv->bhv", q, new_state)

    return output.astype(orig_dtype), new_state


# =============================================================================
# DeltaNet Block
# =============================================================================


class DeltaNetBlock(nnx.Module):
    """DeltaNet attention block with projections and optional gating.

    Architecture:
        1. Input normalization (RMSNorm)
        2. Project to Q, K, V, Beta (and optional Gate)
        3. Optional short convolution on Q, K, V
        4. Apply activation and normalization to Q, K
        5. Delta rule attention (chunk or recurrent based on mode)
        6. Optional output gating
        7. Output projection + residual

    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        expand_k: Key dimension expansion (default 1.0)
        expand_v: Value dimension expansion (default 1.0)
        use_beta: Whether to use learnable beta (default True)
        use_gate: Whether to use output gating (default False)
        use_short_conv: Whether to use short conv on Q/K/V (default True)
        conv_size: Convolution kernel size (default 4)
        conv_bias: Whether to use bias in conv (default False)
        qk_activation: Activation for Q/K ("silu", "relu", etc.)
        qk_norm: Normalization for Q/K ("l2", "sum", or None)
        chunk_size: Chunk size for parallel computation (default 64)
        norm_type: Norm type for input ("rmsnorm" or "layernorm")
        norm_eps: Epsilon for normalization
        rngs: PRNG keys for initialization
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_beta: bool = True,
        use_gate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        qk_activation: str = "silu",
        qk_norm: str = "l2",
        chunk_size: int = 64,
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        norm_eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.qk_activation = qk_activation
        self.qk_norm_type = qk_norm

        # Compute dimensions
        head_dim = hidden_size // num_heads
        self.key_dim = int(head_dim * expand_k)
        self.value_dim = int(head_dim * expand_v)

        qk_total = num_heads * self.key_dim
        v_total = num_heads * self.value_dim

        self.qk_total = qk_total
        self.v_total = v_total

        # Input normalization
        from ..common import RMSNorm

        self.norm = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)

        # Output normalization (per-head value dim), matching FLA's o_norm behavior.
        self.o_norm = RMSNorm(self.value_dim, eps=norm_eps, rngs=rngs)

        # Projections
        self.qkv_proj = nnx.Linear(
            hidden_size, qk_total + qk_total + v_total, use_bias=False, rngs=rngs
        )

        # Beta projection (learning rate)
        self.use_beta = use_beta
        if use_beta:
            self.beta_proj = nnx.Linear(hidden_size, num_heads, use_bias=False, rngs=rngs)

        # Gate projection
        self.use_gate = use_gate
        if use_gate:
            self.gate_proj = nnx.Linear(hidden_size, v_total, use_bias=False, rngs=rngs)

        # Short convolution
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            conv_channels = qk_total + qk_total + v_total
            # Match PyTorch Conv1d default init (kaiming_uniform_ with a=sqrt(5)):
            # uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) where fan_in = kernel_size for depthwise conv.
            conv_bound = float(conv_size) ** -0.5
            self.conv_weight = nnx.Param(
                jax.random.uniform(
                    rngs.params(),
                    (conv_size, conv_channels),
                    minval=-conv_bound,
                    maxval=conv_bound,
                ).astype(jnp.float32)
            )
            if conv_bias:
                self.conv_bias_param = nnx.Param(
                    jax.random.uniform(
                        rngs.params(),
                        (conv_channels,),
                        minval=-conv_bound,
                        maxval=conv_bound,
                    ).astype(jnp.float32)
                )
            else:
                self.conv_bias_param = None

        # Q/K normalization
        if qk_norm == "l2":
            self.q_norm = RMSNorm(self.key_dim, eps=norm_eps, rngs=rngs)
            self.k_norm = RMSNorm(self.key_dim, eps=norm_eps, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

        # Output projection
        self.out_proj = nnx.Linear(v_total, hidden_size, use_bias=False, rngs=rngs)

    def _apply_activation(self, x: jax.Array) -> jax.Array:
        """Apply activation function to Q/K."""
        if self.qk_activation == "silu":
            return jax.nn.silu(x)
        elif self.qk_activation == "relu":
            return jax.nn.relu(x)
        elif self.qk_activation == "elu":
            return jax.nn.elu(x) + 1
        elif self.qk_activation == "identity":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.qk_activation}")

    def _apply_norm(self, q: jax.Array, k: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Apply normalization to Q/K."""
        if self.qk_norm_type == "l2":
            q_f32 = q.astype(jnp.float32)
            k_f32 = k.astype(jnp.float32)
            q = q_f32 / (jnp.linalg.norm(q_f32, axis=-1, keepdims=True) + 1e-6)
            k = k_f32 / (jnp.linalg.norm(k_f32, axis=-1, keepdims=True) + 1e-6)
        elif self.qk_norm_type == "sum":
            # Match FLA's sum_norm: divide by sum (no abs).
            q_f32 = q.astype(jnp.float32)
            k_f32 = k.astype(jnp.float32)
            q = q_f32 / (jnp.sum(q_f32, axis=-1, keepdims=True) + 1e-6)
            k = k_f32 / (jnp.sum(k_f32, axis=-1, keepdims=True) + 1e-6)
        return q, k

    def __call__(
        self,
        x: jax.Array,  # [batch, seq_len, hidden]
        *,
        state: Optional[DeltaNetState] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[str] = None,
    ) -> Tuple[jax.Array, Optional[DeltaNetState]]:
        """Forward pass.

        Args:
            x: Input [batch, seq_len, hidden]
            state: Optional cached state for generation
            mask: Unused (delta rule is inherently causal)
            mode: "chunk" for training, "recurrent" for generation

        Returns:
            output: [batch, seq_len, hidden]
            new_state: Updated state
        """
        batch, seq_len, hidden = x.shape
        heads = self.num_heads
        key_dim = self.key_dim
        value_dim = self.value_dim

        if mode is None:
            mode = "recurrent" if state is not None else "chunk"

        # Input norm
        normed = self.norm(x)

        # Project to Q, K, V
        qkv = self.qkv_proj(normed)
        q, k, v = jnp.split(qkv, [self.qk_total, 2 * self.qk_total], axis=-1)

        # Short conv
        new_conv_state = None
        if self.use_short_conv:
            # depthwise_conv1d_causal expects [batch, seq, channels]
            qkv_cat = jnp.concatenate([q, k, v], axis=-1)  # [batch, seq, channels]

            # Get cached conv state
            conv_cache = state.conv_state if state is not None else None
            
            # Get bias
            conv_bias = self.conv_bias_param.value if self.conv_bias_param is not None else None

            # Apply causal conv
            qkv_cat, new_conv_state = depthwise_conv1d_causal(
                qkv_cat, self.conv_weight.value, conv_bias, cache=conv_cache
            )

            q, k, v = jnp.split(qkv_cat, [self.qk_total, 2 * self.qk_total], axis=-1)

        # Apply activation
        q = self._apply_activation(q)
        k = self._apply_activation(k)
        # Match FLA: values always use SiLU.
        v = jax.nn.silu(v)

        # Reshape to multi-head
        q = q.reshape(batch, seq_len, heads, key_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, heads, key_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, heads, value_dim).transpose(0, 2, 1, 3)

        # Apply norm
        q, k = self._apply_norm(q, k)

        # Compute beta
        if self.use_beta:
            beta = self.beta_proj(normed)  # [batch, seq, heads]
            beta = jax.nn.sigmoid(beta)
            beta = beta.transpose(0, 2, 1)  # [batch, heads, seq]
        else:
            beta = jnp.ones((batch, heads, seq_len))

        # Initial state
        initial_S = state.S if state is not None else None

        # Delta rule attention (kernels apply q scaling internally like FLA: scale=1/sqrt(key_dim)).
        if mode == "chunk" and seq_len > 1:
            output, final_S = delta_rule_chunkwise(
                q,
                k,
                v,
                beta,
                initial_state=initial_S,
                chunk_size=self.chunk_size,
            )
        else:
            if seq_len == 1:
                init_S = (
                    initial_S
                    if initial_S is not None
                    else jnp.zeros((batch, heads, key_dim, value_dim), dtype=q.dtype)
                )
                out, final_S = delta_rule_step(
                    q.squeeze(2),
                    k.squeeze(2),
                    v.squeeze(2),
                    beta.squeeze(2),
                    init_S,
                )
                output = out[:, :, None, :]
            else:
                output, final_S = delta_rule_recurrent(
                    q, k, v, beta, initial_state=initial_S
                )

        # Output norm + optional gate (closer to FLA layer semantics).
        # output currently: [batch, heads, seq, value_dim]
        output = output.transpose(0, 2, 1, 3)  # [batch, seq, heads, value_dim]
        output = self.o_norm(output)
        if self.use_gate:
            gate = jax.nn.sigmoid(self.gate_proj(normed)).reshape(batch, seq_len, heads, value_dim)
            output = output * gate

        # Flatten: [batch, seq, heads, value_dim] -> [batch, seq, v_total]
        output = output.reshape(batch, seq_len, self.v_total)

        # Output projection
        output = self.out_proj(output)

        # Residual
        output = output + x

        # Build new state
        new_state = None
        if state is not None or mode == "recurrent":
            new_state = DeltaNetState(S=final_S, conv_state=new_conv_state)

        return output, new_state

    def init_state(self, batch_size: int) -> DeltaNetState:
        """Initialize empty state for autoregressive generation."""
        conv_channels = None
        if self.use_short_conv:
            conv_channels = self.qk_total * 2 + self.v_total

        return DeltaNetState.zeros(
            batch_size=batch_size,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            conv_channels=conv_channels,
            conv_size=self.conv_size,
        )
