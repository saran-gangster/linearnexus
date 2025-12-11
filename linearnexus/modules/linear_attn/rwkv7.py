"""
RWKV-7 (Goose): Receptance Weighted Key Value with DPLR Transition

Paper: "RWKV-7 Goose" - Diagonal Plus Low Rank transition matrices

Mathematical Foundation
=======================

RWKV-7 uses a Diagonal-Plus-Low-Rank (DPLR) transition matrix instead of 
the simpler diagonal transition in RWKV-6. This allows richer state dynamics
while maintaining linear complexity.

Core Recurrence (DPLR - Diagonal Plus Low Rank)
-----------------------------------------------
    S_t = S_{t-1} @ (D_t + a_t @ b_t^T) + v_t @ k_t^T

Where:
    - D_t = Diag(exp(w_t)) is the diagonal decay matrix
    - a_t = -kk_t (negative L2-normalized key)
    - b_t = kk_t * a_t (scaled normalized key for low-rank component)
    - The term (D_t + a_t @ b_t^T) is the DPLR transition matrix

Expanded form:
    1. h = exp(w) * h + b * (a^T @ h)   [DPLR transition]
    2. h = h + k @ v^T                   [Add new key-value]  
    3. o = h^T @ r                       [Query the state]

Key Differences from RWKV-6:
----------------------------
1. **Transition Matrix**: DPLR instead of diagonal-only
2. **Low-rank component**: Adds `b * (a^T @ h)` term
3. **kk normalization**: L2-normalized keys used for stability
4. **Output correction**: Additional correction term with r_k parameter
5. **v_first mechanism**: First layer's v is propagated and interpolated

Architecture Details:
--------------------
- Uses LoRA-style low-rank projections for w, a, g, v
- L2 normalization on keys (kk)
- k_k and k_a parameters for key scaling
- r_k parameter for output correction term
- GroupNorm on output before gating
- Separate x_* lerp parameters for each projection

Reference Implementation
------------------------
Based on FLA's RWKV7Attention and the DPLR recurrent kernel.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import math
import jax
import jax.numpy as jnp
from jax import lax
import flax.nnx as nnx

from ..common import RMSNorm
from .rwkv6 import token_shift, GroupNorm, LerpLinear


# =============================================================================
# State Cache
# =============================================================================


@dataclass
class RWKV7State:
    """Cache for RWKV7 autoregressive generation.

    Attributes:
        h: Recurrent state matrix [batch, num_heads, head_k_dim, head_v_dim]
        shift_state: Last token for time shift [batch, hidden_size]
        v_first: First layer's v for cross-layer interpolation [batch, seq, hidden_size]
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
    ) -> "RWKV7State":
        """Initialize empty state.

        Args:
            batch_size: Batch dimension
            num_heads: Number of attention heads
            head_k_dim: Key dimension per head
            head_v_dim: Value dimension per head
            hidden_size: Hidden size for shift state (optional)

        Returns:
            RWKV7State with zero-initialized arrays
        """
        h = jnp.zeros((batch_size, num_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
        shift_state = None
        if hidden_size is not None:
            shift_state = jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)

        return cls(h=h, shift_state=shift_state)


# =============================================================================
# DPLR Recurrent Kernel (Core Algorithm)
# =============================================================================


def dplr_delta_rule_recurrent(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    scale: float = 1.0,
    initial_state: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """DPLR (Diagonal Plus Low Rank) delta rule recurrence.

    This is the core RWKV-7 recurrence:
        S_t = S_{t-1} @ (D_t + a_t @ b_t^T) + v_t @ k_t^T
    
    Expanded to:
        h = exp(gk) * h + b * (a^T @ h)
        h = h + k @ v^T
        o = h^T @ q

    Args:
        q: Query tensor [batch, heads, seq, head_dim] (receptance r)
        k: Key tensor [batch, heads, seq, head_dim]
        v: Value tensor [batch, heads, seq, head_v_dim]
        a: Low-rank 'a' term [batch, heads, seq, head_dim] (typically -kk)
        b: Low-rank 'b' term [batch, heads, seq, head_dim] (typically kk * a_scalar)
        gk: Log decay [batch, heads, seq, head_dim] (w in RWKV)
        scale: Query scale factor (default 1.0)
        initial_state: Initial hidden state [batch, heads, head_dim, head_v_dim]

    Returns:
        output: [batch, heads, seq, head_v_dim]
        final_state: [batch, heads, head_dim, head_v_dim]
    """
    batch, heads, seq_len, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    dtype = q.dtype

    # Initialize state
    if initial_state is None:
        h = jnp.zeros((batch, heads, head_k_dim, head_v_dim), dtype=jnp.float32)
    else:
        h = initial_state.astype(jnp.float32)

    # Scale query
    q = q.astype(jnp.float32) * scale
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    gk = gk.astype(jnp.float32)

    def step(h, inputs):
        """Single step of DPLR recurrence."""
        q_t, k_t, v_t, a_t, b_t, gk_t = inputs
        # q_t, k_t, a_t, b_t, gk_t: [batch, heads, head_k_dim]
        # v_t: [batch, heads, head_v_dim]
        # h: [batch, heads, head_k_dim, head_v_dim]

        # 1. DPLR transition: h = exp(gk) * h + b * (a^T @ h)
        # exp(gk): [batch, heads, head_k_dim]
        # a^T @ h: [batch, heads, head_k_dim] @ [batch, heads, head_k_dim, head_v_dim] -> [batch, heads, head_v_dim]
        decay = jnp.exp(gk_t)  # [batch, heads, head_k_dim]
        a_h = jnp.einsum('bhk,bhkv->bhv', a_t, h)  # [batch, heads, head_v_dim]
        h_new = decay[..., None] * h + b_t[..., None] * a_h[..., None, :]
        # decay[..., None]: [batch, heads, head_k_dim, 1]
        # b_t[..., None]: [batch, heads, head_k_dim, 1]
        # a_h[..., None, :]: [batch, heads, 1, head_v_dim]
        # h_new: [batch, heads, head_k_dim, head_v_dim]

        # 2. Add new key-value: h = h + k @ v^T
        kv = jnp.einsum('bhk,bhv->bhkv', k_t, v_t)  # [batch, heads, head_k_dim, head_v_dim]
        h_new = h_new + kv

        # 3. Query the state: o = h^T @ q
        o_t = jnp.einsum('bhkv,bhk->bhv', h_new, q_t)  # [batch, heads, head_v_dim]

        return h_new, o_t

    # Transpose for scan: [seq, batch, heads, dim]
    inputs = (
        jnp.transpose(q, (2, 0, 1, 3)),   # [seq, batch, heads, head_k_dim]
        jnp.transpose(k, (2, 0, 1, 3)),   # [seq, batch, heads, head_k_dim]
        jnp.transpose(v, (2, 0, 1, 3)),   # [seq, batch, heads, head_v_dim]
        jnp.transpose(a, (2, 0, 1, 3)),   # [seq, batch, heads, head_k_dim]
        jnp.transpose(b, (2, 0, 1, 3)),   # [seq, batch, heads, head_k_dim]
        jnp.transpose(gk, (2, 0, 1, 3)),  # [seq, batch, heads, head_k_dim]
    )

    final_state, outputs = lax.scan(step, h, inputs)

    # outputs: [seq, batch, heads, head_v_dim] -> [batch, heads, seq, head_v_dim]
    outputs = jnp.transpose(outputs, (1, 2, 0, 3))

    return outputs.astype(dtype), final_state


def dplr_delta_rule_step(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    state: jax.Array,
    scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Single step of DPLR delta rule for autoregressive generation.

    Args:
        q: Query [batch, heads, head_k_dim]
        k: Key [batch, heads, head_k_dim]
        v: Value [batch, heads, head_v_dim]
        a: Low-rank 'a' term [batch, heads, head_k_dim]
        b: Low-rank 'b' term [batch, heads, head_k_dim]
        gk: Log decay [batch, heads, head_k_dim]
        state: Hidden state [batch, heads, head_k_dim, head_v_dim]
        scale: Query scale factor

    Returns:
        output: [batch, heads, head_v_dim]
        new_state: [batch, heads, head_k_dim, head_v_dim]
    """
    dtype = q.dtype
    
    q = q.astype(jnp.float32) * scale
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    gk = gk.astype(jnp.float32)
    h = state.astype(jnp.float32)

    # 1. DPLR transition
    decay = jnp.exp(gk)
    a_h = jnp.einsum('bhk,bhkv->bhv', a, h)
    h_new = decay[..., None] * h + b[..., None] * a_h[..., None, :]

    # 2. Add new key-value
    kv = jnp.einsum('bhk,bhv->bhkv', k, v)
    h_new = h_new + kv

    # 3. Query the state
    o = jnp.einsum('bhkv,bhk->bhv', h_new, q)

    return o.astype(dtype), h_new


# =============================================================================
# LoRA (Low-Rank Adaptation) Layer
# =============================================================================


class LoRA(nnx.Module):
    """Low-Rank Adaptation layer with optional activation.

    Used in RWKV-7 for:
    - w_lora: decay projection (tanh activation)
    - a_lora: low-rank 'a' projection (no activation)
    - g_lora: gate projection (sigmoid activation)
    - v_lora: value interpolation (sigmoid activation)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        low_rank_dim: int = 32,
        activation: Optional[str] = None,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            low_rank_dim: Low-rank dimension
            activation: Activation function ('tanh', 'sigmoid', or None)
            use_bias: Whether to use bias on the up projection
            rngs: RNG keys
        """
        self.in_features = in_features
        self.out_features = out_features
        self.low_rank_dim = low_rank_dim
        self.activation = activation

        # Down projection (no bias)
        self.down = nnx.Linear(in_features, low_rank_dim, use_bias=False, rngs=rngs)
        # Up projection (with optional bias)
        self.up = nnx.Linear(low_rank_dim, out_features, use_bias=use_bias, rngs=rngs)

        # Initialize weights
        key = rngs.params()
        # Down projection: small init
        self.down.kernel.value = jax.random.normal(key, (in_features, low_rank_dim)) * 0.01
        # Up projection: zero init for residual-friendly behavior
        key, subkey = jax.random.split(key)
        self.up.kernel.value = jnp.zeros((low_rank_dim, out_features))
        if use_bias:
            self.up.bias.value = jnp.zeros((out_features,))

    def set_bias_value(self, value: jax.Array):
        """Set the bias of the up projection."""
        if hasattr(self.up, 'bias'):
            self.up.bias.value = value

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input [batch, seq, in_features]

        Returns:
            Output [batch, seq, out_features]
        """
        x = self.down(x)
        if self.activation == 'tanh':
            x = jnp.tanh(x)
        x = self.up(x)
        if self.activation == 'sigmoid':
            x = jax.nn.sigmoid(x)
        return x


# =============================================================================
# L2 Normalization
# =============================================================================


def l2_norm(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    """L2 normalize along the last dimension.

    Args:
        x: Input tensor
        eps: Small constant for numerical stability

    Returns:
        L2-normalized tensor
    """
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


# =============================================================================
# Output Correction
# =============================================================================


def gate_output_correction(
    o: jax.Array,
    r: jax.Array,
    k: jax.Array,
    r_k: jax.Array,
    v: jax.Array,
    g: jax.Array,
) -> jax.Array:
    """Apply output correction and gating.

    correction_term = ((r * k * r_k).sum(-1, keepdim=True) * v).view(o.shape)
    output = (o + correction_term) * g

    Args:
        o: Output from attention [batch, seq, hidden_size]
        r: Receptance [batch, seq, num_heads, head_dim]
        k: Key [batch, seq, num_heads, head_dim]
        r_k: R_k parameter [num_heads, head_dim]
        v: Value [batch, seq, num_heads, head_v_dim]
        g: Gate [batch, seq, hidden_size]

    Returns:
        Corrected and gated output [batch, seq, hidden_size]
    """
    batch, seq_len, num_heads, head_dim = r.shape
    head_v_dim = v.shape[-1]

    # Unsqueeze r_k for broadcasting
    r_k_expanded = r_k[None, None, :, :]  # [1, 1, num_heads, head_dim]

    # Compute correction scalar: (r * k * r_k).sum(-1, keepdim=True)
    correction_scalar = jnp.sum(r * k * r_k_expanded, axis=-1, keepdims=True)
    # correction_scalar: [batch, seq, num_heads, 1]

    # correction_term = correction_scalar * v
    correction_term = correction_scalar * v  # [batch, seq, num_heads, head_v_dim]

    # Reshape to match o
    correction_term = correction_term.reshape(batch, seq_len, -1)

    # Apply correction and gating
    output = (o + correction_term) * g

    return output


# =============================================================================
# RWKV7 Attention Block
# =============================================================================


class RWKV7Block(nnx.Module):
    """RWKV-7 attention block with DPLR transition.

    This implements the full RWKV-7 architecture including:
    - Token shift for temporal mixing
    - LoRA projections for w, a, g, v
    - L2-normalized keys
    - DPLR recurrence
    - Output correction with r_k
    - GroupNorm on output
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: Optional[int] = None,
        head_dim: int = 64,
        value_dim: Optional[int] = None,
        decay_low_rank_dim: Optional[int] = None,
        gate_low_rank_dim: Optional[int] = None,
        a_low_rank_dim: Optional[int] = None,
        v_low_rank_dim: Optional[int] = None,
        layer_idx: int = 0,
        n_layers: int = 24,
        norm_eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize RWKV7 block.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads (if None, derived from head_dim)
            head_dim: Dimension per head (default 64)
            value_dim: Value dimension (default = hidden_size)
            decay_low_rank_dim: Low-rank dim for decay (w) projection
            gate_low_rank_dim: Low-rank dim for gate (g) projection
            a_low_rank_dim: Low-rank dim for 'a' projection
            v_low_rank_dim: Low-rank dim for v interpolation
            layer_idx: Layer index for layer-dependent initialization
            n_layers: Total number of layers
            norm_eps: Epsilon for normalization
            rngs: RNG keys
        """
        self.hidden_size = hidden_size
        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size

        # Compute head dimensions
        if num_heads is not None:
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
        else:
            self.head_dim = head_dim
            self.num_heads = hidden_size // head_dim

        self.head_v_dim = self.value_dim // self.num_heads
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.norm_eps = norm_eps

        # Compute low-rank dimensions (following RWKV-7 formula)
        factor = self.head_dim / 64

        if decay_low_rank_dim is None:
            decay_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        self.decay_low_rank_dim = decay_low_rank_dim

        if gate_low_rank_dim is None:
            gate_low_rank_dim = max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
        self.gate_low_rank_dim = gate_low_rank_dim

        if a_low_rank_dim is None:
            a_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
        self.a_low_rank_dim = a_low_rank_dim

        if v_low_rank_dim is None:
            v_low_rank_dim = max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))
        self.v_low_rank_dim = v_low_rank_dim

        # Input normalization
        self.norm = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)

        # Token shift lerp parameters (x_*)
        self.x_r = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.x_w = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.x_k = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.x_v = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.x_a = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.x_g = nnx.Param(jnp.zeros((1, 1, hidden_size)))

        # Key scaling parameters
        self.k_k = nnx.Param(jnp.zeros(self.key_dim))
        self.k_a = nnx.Param(jnp.ones(self.key_dim) * 1.02)
        self.r_k = nnx.Param(jnp.ones((self.num_heads, self.head_dim)) * -0.04)

        # Projections
        self.r_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, self.value_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(self.value_dim, hidden_size, use_bias=False, rngs=rngs)

        # LoRA projections
        self.w_lora = LoRA(hidden_size, self.key_dim, decay_low_rank_dim, activation='tanh', rngs=rngs)
        self.a_lora = LoRA(hidden_size, self.key_dim, a_low_rank_dim, activation=None, rngs=rngs)
        self.g_lora = LoRA(hidden_size, self.value_dim, gate_low_rank_dim, activation='sigmoid', use_bias=False, rngs=rngs)

        # v_lora only for non-first layers (for v_first interpolation)
        self.use_v_lora = (layer_idx != 0)
        if self.use_v_lora:
            self.v_lora = LoRA(hidden_size, self.value_dim, v_low_rank_dim, activation=None, rngs=rngs)

        # Output GroupNorm
        self.g_norm = GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=self.head_dim * norm_eps,
            rngs=rngs,
        )

        # Initialize weights
        self._initialize_weights(rngs)

    def _initialize_weights(self, rngs: nnx.Rngs):
        """Initialize weights with layer-dependent values."""
        key = rngs.params()

        if self.n_layers > 1:
            ratio_0_to_1 = self.layer_idx / (self.n_layers - 1)
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.n_layers)
        else:
            ratio_0_to_1 = 0.0
            ratio_1_to_almost0 = 1.0

        # Position-based initialization
        n = jnp.arange(self.hidden_size)
        linear = n / (self.hidden_size - 1) - 0.5
        ddd = n / self.hidden_size

        # Zigzag pattern
        zigzag = ((n % self.head_dim) - ((self.head_dim - 1) / 2)) / ((self.head_dim - 1) / 2)
        zigzag = zigzag * jnp.abs(zigzag)

        # Decay initialization
        www = -6 + 6 * jnp.power(n / (self.hidden_size - 1), 1 + ratio_0_to_1 ** 0.3)

        # Initialize x_* parameters
        self.x_r.value = (1.0 - jnp.power(ddd, 0.2 * ratio_1_to_almost0))[None, None, :]
        self.x_w.value = (1.0 - jnp.power(ddd, 0.9 * ratio_1_to_almost0))[None, None, :]
        self.x_k.value = (1.0 - jnp.power(ddd, 0.7 * ratio_1_to_almost0))[None, None, :]
        self.x_v.value = (1.0 - jnp.power(ddd, 0.7 * ratio_1_to_almost0))[None, None, :]
        self.x_a.value = (1.0 - jnp.power(ddd, 0.9 * ratio_1_to_almost0))[None, None, :]
        self.x_g.value = (1.0 - jnp.power(ddd, 0.2 * ratio_1_to_almost0))[None, None, :]

        # k_k initialization
        self.k_k.value = 0.71 - linear * 0.1

        # w_lora bias: www + 0.5 + zigzag*2.5
        # (0.5 comes from softplus approximation)
        self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)

        # a_lora bias
        self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)

        # v_lora bias (non-first layers)
        if self.use_v_lora:
            self.v_lora.set_bias_value(0.73 - linear * 0.4)

        # GroupNorm scale (RWKV uses layer-dependent init)
        self.g_norm.scale.value = jnp.ones(self.value_dim) * ((self.layer_idx + 1) / self.n_layers) ** 0.7

        # Orthogonal initialization for projections
        key, subkey = jax.random.split(key)
        self.r_proj.kernel.value = self._orthogonal_init(subkey, (self.hidden_size, self.key_dim), gain=1.0)
        key, subkey = jax.random.split(key)
        self.k_proj.kernel.value = self._orthogonal_init(subkey, (self.hidden_size, self.key_dim), gain=0.1)
        key, subkey = jax.random.split(key)
        self.v_proj.kernel.value = self._orthogonal_init(subkey, (self.hidden_size, self.value_dim), gain=1.0)
        # o_proj: zero init
        self.o_proj.kernel.value = jnp.zeros((self.value_dim, self.hidden_size))

    def _orthogonal_init(self, key: jax.Array, shape: Tuple[int, int], gain: float = 1.0) -> jax.Array:
        """Initialize with orthogonal matrix."""
        m, n = shape
        if m >= n:
            # QR decomposition of random matrix
            q, _ = jnp.linalg.qr(jax.random.normal(key, (m, n)))
            return q * gain
        else:
            q, _ = jnp.linalg.qr(jax.random.normal(key, (n, m)))
            return q.T * gain

    def init_state(self, batch_size: int) -> RWKV7State:
        """Initialize empty state for generation.

        Args:
            batch_size: Batch size

        Returns:
            Zero-initialized RWKV7State
        """
        return RWKV7State.zeros(
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_v_dim,
            hidden_size=self.hidden_size,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[RWKV7State] = None,
        v_first: Optional[jax.Array] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[jax.Array, Optional[RWKV7State], Optional[jax.Array]]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq, hidden_size]
            state: Optional cached state for generation
            v_first: First layer's v for interpolation (cross-layer)
            mask: Optional attention mask (not used in linear attention)
            mode: Computation mode ('chunk' for training, 'recurrent' for generation)

        Returns:
            output: Output tensor [batch, seq, hidden_size]
            new_state: Updated state (if state was provided or mode='recurrent')
            v_first_out: This layer's v_first (for first layer) or passed through
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Normalize input
        x = self.norm(x)

        # Token shift
        if state is not None and state.shift_state is not None:
            shift_cache = state.shift_state
        else:
            shift_cache = None

        shifted, new_shift_state = token_shift(x, shift_cache)
        delta = shifted - x

        # Compute lerp'd inputs: x + delta * x_*
        xr = x + delta * self.x_r.value
        xw = x + delta * self.x_w.value
        xk = x + delta * self.x_k.value
        xv = x + delta * self.x_v.value
        xa = x + delta * self.x_a.value
        xg = x + delta * self.x_g.value

        # Projections
        r = self.r_proj(xr)  # Receptance (query)
        k = self.k_proj(xk)  # Key
        v = self.v_proj(xv)  # Value

        # Decay: w = -0.6065 * sigmoid(w_lora(xw))
        # Note: -0.6065 ≈ log(0.5) - ensures decay is in reasonable range
        w = -0.6065306597126334 * jax.nn.sigmoid(self.w_lora(xw))

        # v_first interpolation for non-first layers
        v_first_out = v_first
        if self.layer_idx == 0:
            v_first_out = v
        elif self.use_v_lora and v_first is not None:
            # Interpolate: v = lerp(v, v_first, sigmoid(v_lora(xv)))
            v_mix = jax.nn.sigmoid(self.v_lora(xv))
            v = v + v_mix * (v_first - v)

        # Low-rank 'a' term: a = sigmoid(a_lora(xa))
        a = jax.nn.sigmoid(self.a_lora(xa))

        # Gate
        g = self.g_lora(xg)

        # L2-normalized key: kk = l2_norm(k * k_k)
        # k * k_k: element-wise scaling
        k_scaled = k * self.k_k.value
        kk = k_scaled.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        kk = l2_norm(kk)

        # Key update: k = k + k * (a - 1) * k_a
        # Equivalent to: k = k * (1 + (a - 1) * k_a)
        k = k + k * (a - 1) * self.k_a.value

        # Reshape to heads
        r = r.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        w = w.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)
        a_for_kernel = a.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # DPLR parameters:
        # a = -kk (negative normalized key)
        # b = kk * a (scaled normalized key)
        dplr_a = -kk
        dplr_b = kk * a_for_kernel

        # Transpose for kernel: [batch, heads, seq, dim]
        r = jnp.transpose(r, (0, 2, 1, 3))
        w = jnp.transpose(w, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        dplr_a = jnp.transpose(dplr_a, (0, 2, 1, 3))
        dplr_b = jnp.transpose(dplr_b, (0, 2, 1, 3))

        # Get recurrent state
        recurrent_state = state.h if state is not None else None

        # Apply DPLR kernel
        if mode == "recurrent" or seq_len == 1:
            # Single token mode
            if seq_len == 1:
                o, new_h = dplr_delta_rule_step(
                    q=r[:, :, 0, :],
                    k=k[:, :, 0, :],
                    v=v[:, :, 0, :],
                    a=dplr_a[:, :, 0, :],
                    b=dplr_b[:, :, 0, :],
                    gk=w[:, :, 0, :],
                    state=recurrent_state if recurrent_state is not None else jnp.zeros(
                        (batch_size, self.num_heads, self.head_dim, self.head_v_dim)
                    ),
                    scale=1.0,
                )
                o = o[:, :, None, :]  # Add seq dim back
            else:
                o, new_h = dplr_delta_rule_recurrent(
                    q=r, k=k, v=v, a=dplr_a, b=dplr_b, gk=w,
                    scale=1.0,
                    initial_state=recurrent_state,
                )
        else:
            # Chunk mode (training)
            o, new_h = dplr_delta_rule_recurrent(
                q=r, k=k, v=v, a=dplr_a, b=dplr_b, gk=w,
                scale=1.0,
                initial_state=recurrent_state,
            )

        # Reshape output: [batch, heads, seq, head_v_dim] -> [batch, seq, heads, head_v_dim]
        o = jnp.transpose(o, (0, 2, 1, 3))

        # GroupNorm
        o_flat = o.reshape(batch_size * seq_len, self.value_dim)
        o_normed = self.g_norm(o_flat.reshape(batch_size * seq_len, 1, self.value_dim))
        o = o_normed.reshape(batch_size, seq_len, self.value_dim)

        # Output correction with r_k
        # Need r, k in [batch, seq, heads, dim] format
        r_for_correction = jnp.transpose(r, (0, 2, 1, 3))  # [batch, seq, heads, head_dim]
        k_for_correction = jnp.transpose(k, (0, 2, 1, 3))  # [batch, seq, heads, head_dim]
        v_for_correction = jnp.transpose(v, (0, 2, 1, 3))  # [batch, seq, heads, head_v_dim]

        o = gate_output_correction(o, r_for_correction, k_for_correction, self.r_k.value, v_for_correction, g)

        # Output projection + residual
        output = self.o_proj(o) + residual

        # Build new state
        new_state = None
        if state is not None or mode == "recurrent":
            new_state = RWKV7State(
                h=new_h,
                shift_state=new_shift_state,
            )

        return output, new_state, v_first_out


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RWKV7State",
    "RWKV7Block",
    "dplr_delta_rule_recurrent",
    "dplr_delta_rule_step",
    "LoRA",
    "l2_norm",
    "gate_output_correction",
]
