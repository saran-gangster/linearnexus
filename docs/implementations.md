# Implementation Reports

This document contains detailed implementation reports for new architecture blocks added to LinearNexus.

---

## DeltaNet: Linear Attention with the Delta Rule

**Paper:** [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)

**Implementation Date:** December 2025

### Mathematical Foundation

DeltaNet uses the delta rule to update a linear attention state matrix $S$. Unlike standard linear attention ($S_t = S_{t-1} + k_t v_t^T$), DeltaNet first removes the old value associated with $k_t$ before adding the new one.

#### Core Recurrence (Token-by-Token)

For each token $t$:

1. **Retrieve old value:** $v_{old} = S_{t-1} k_t$
2. **Compute delta:** $v_{\delta} = \beta_t (v_t - v_{old})$
3. **Update state:** $S_t = S_{t-1} + k_t v_{\delta}^T$
4. **Query output:** $o_t = q_t S_t$

Where:
- $q_t, k_t$: query and key vectors, shape `[d_k]`
- $v_t$: value vector, shape `[d_v]`
- $\beta_t$: scalar learning rate in `[0, 1]`, typically `sigmoid(linear(x))`
- $S_t$: state matrix, shape `[d_k, d_v]`

This is equivalent to the matrix form:

$$S_t = (I - \beta_t k_t k_t^T) S_{t-1} + \beta_t k_t v_t^T$$

This formulation removes the contribution of $k_t$ from the old state before adding the new $k_t \rightarrow v_t$ association, making it a proper "update" rather than just accumulation.

#### Chunkwise Parallel Form (WY Representation)

For efficient training, we process chunks in parallel using the WY decomposition. Given a chunk of $L$ tokens with $Q, K, V$ matrices `[L, d]`:

1. Build lower triangular matrix: $A = I + \text{diag}(\beta) \cdot \text{tril}(K K^T, -1)$
2. Solve for correction: $T = A^{-1} \text{diag}(\beta)$ (lower triangular solve)
3. Compute corrected terms: $W = T K$, $U = T V$

**Intra-chunk output (parallel):**
$$O_{intra} = (Q K^T \odot \text{causal\_mask}) (U - W S_0)$$

**State update for next chunk:**
$$S_{new} = S_0 + K^T (U - W S_0)$$

**Full output:**
$$O = Q S_0 + O_{intra}$$

### Implementation Details

| Component | File | Description |
|-----------|------|-------------|
| **Core Kernels** | `modules/linear_attn/delta_net.py` | `delta_rule_recurrent`, `delta_rule_chunkwise`, `delta_rule_step` |
| **Block** | `modules/linear_attn/delta_net.py` | `DeltaNetBlock` - Full module with projections, conv, gating |
| **State Cache** | `modules/linear_attn/delta_net.py` | `DeltaNetState` - Cache for recurrent state and conv |
| **Model Integration** | `models.py` | Presets: `DELTANET_SMALL`, `DELTANET_MEDIUM`, `DELTA_HYBRID_SMALL` |

### Block Architecture

```
Input x [batch, seq, hidden]
    │
    ▼
┌─────────────────┐
│   RMSNorm       │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│QKV Proj│ │Beta   │
└───┬───┘ │Proj   │
    │     └───┬───┘
    ▼         │
┌───────┐     │
│Short  │     │
│Conv1D │     │
│(opt)  │     │
└───┬───┘     │
    │         │
    ▼         │
┌───────┐     │
│SiLU + │     │
│L2 Norm│     │
└───┬───┘     │
    │         │
    ▼         ▼
┌─────────────────┐
│  Delta Rule     │
│  Attention      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gate (opt)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Proj    │
└────────┬────────┘
         │
         ▼
    + Residual
         │
         ▼
Output [batch, seq, hidden]
```

### Key Features

- ✅ **Recurrent mode** for token-by-token generation with O(1) memory per step
- ✅ **Chunkwise parallel mode** for efficient training with O(L²) within chunks
- ✅ **Step function** for optimized single-token inference
- ✅ **Optional short convolution** on Q, K, V for local context
- ✅ **Optional output gating** for improved expressiveness
- ✅ **Configurable Q/K activation** (silu, relu, elu, identity)
- ✅ **Configurable Q/K normalization** (l2, sum, none)
- ✅ **Learnable beta** (learning rate per head per token)
- ✅ **Full integration** with LMModel and hybrid architectures

### Configuration Options

```python
# In ModelConfig
deltanet_heads: int          # Number of attention heads (default: n_heads)
deltanet_expand_k: float     # Key expansion ratio (default: 1.0)
deltanet_expand_v: float     # Value expansion ratio (default: 1.0)
deltanet_use_beta: bool      # Learnable beta (default: True)
deltanet_use_gate: bool      # Output gating (default: False)
deltanet_use_short_conv: bool # Short convolutions (default: True)
deltanet_qk_activation: str  # Activation for Q/K (default: "silu")
deltanet_qk_norm: str        # Normalization for Q/K (default: "l2")
deltanet_chunk_size: int     # Chunk size for chunkwise (default: 64)
```

### Usage Examples

```python
from linearnexus import create_model, LMModel
from linearnexus.models import ModelConfig
import flax.nnx as nnx

# Pure DeltaNet model
config, _ = create_model("deltanet-small")
model = LMModel(config, rngs=nnx.Rngs(0))

# Hybrid DeltaNet + Attention
config = ModelConfig(
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["deltanet", "deltanet", "deltanet", "attention"],
)
model = LMModel(config, rngs=nnx.Rngs(0))

# Custom DeltaNet block
from linearnexus.modules.linear_attn import DeltaNetBlock

block = DeltaNetBlock(
    hidden_size=768,
    num_heads=12,
    expand_k=1.0,
    expand_v=1.0,
    use_gate=True,
    use_short_conv=True,
    rngs=nnx.Rngs(0),
)
```

### Test Coverage

18 tests covering:
- Kernel shape correctness for recurrent, chunkwise, and step functions
- **Recurrent vs chunkwise parity** (rtol=1e-3, atol=1e-3)
- Step function matching first step of recurrent
- State caching and continuation
- Initial state handling
- Beta scaling behavior (β=0 → no update, β=1 → full update)
- Block forward pass with various configurations
- Model integration (pure DeltaNet and hybrid architectures)
- Deterministic output verification

### Reference Implementation

Based on [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention):
- `examples/fla/layers/delta_net.py` - PyTorch layer implementation
- `examples/fla/ops/delta_rule/naive.py` - Reference recurrent kernel
- `examples/fla/ops/delta_rule/README.md` - WY representation derivation

---

## Gated DeltaNet: Combining Mamba2 Gating with Delta Rule

**Paper:** [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

**Implementation Date:** December 2025

### Mathematical Foundation

Gated DeltaNet extends DeltaNet by adding an exponential decay gate before the delta rule update. This allows the model to learn when to forget old associations, similar to Mamba2's gating mechanism.

#### Core Recurrence (Token-by-Token)

For each token $t$:

1. **Apply decay:** $S_{decayed} = \exp(g_t) \cdot S_{t-1}$
2. **Retrieve old value:** $v_{old} = S_{decayed} \cdot k_t$
3. **Compute delta:** $v_{\delta} = \beta_t (v_t - v_{old})$
4. **Update state:** $S_t = S_{decayed} + k_t v_{\delta}^T$
5. **Query output:** $o_t = q_t S_t$

Where:
- $q_t, k_t$: query and key vectors, shape `[d_k]`
- $v_t$: value vector, shape `[d_v]`
- $\beta_t$: learning rate in `[0, 1]` or `[0, 2]` if `allow_neg_eigval`
- $g_t$: decay gate (typically negative), computed as:
  $$g_t = -A_{log}.\exp() \cdot \text{softplus}(a_t + dt_{bias})$$
- $S_t$: state matrix, shape `[d_k, d_v]`

This is equivalent to the matrix form:

$$S_t = \exp(g_t) \cdot S_{t-1} + k_t (\beta_t (v_t - \exp(g_t) S_{t-1} k_t))^T$$

#### Key Differences from DeltaNet

| Aspect | DeltaNet | Gated DeltaNet |
|--------|----------|----------------|
| **State decay** | None | $\exp(g_t)$ before update |
| **Beta projection** | Single `beta_proj` | Dual `a_proj` + `b_proj` |
| **Decay parameters** | N/A | `A_log`, `dt_bias` (Mamba2-style) |
| **Q/K normalization** | Optional (l2/sum) | Always L2 normalized |
| **GVA (Grouped Value Attention)** | No | Yes (`num_v_heads >= num_heads`) |
| **Head dimension** | Variable | Fixed at 256 (default) |
| **Parameter count** | ~4-5x hidden² | ~6x hidden² (like Mamba2) |

#### Grouped Value Attention (GVA)

Gated DeltaNet supports different numbers of heads for Q/K vs V:
- `num_heads`: Number of Q/K heads
- `num_v_heads`: Number of V heads (can be >= `num_heads`)

When `num_v_heads > num_heads`, Q/K heads are repeated to match V heads:
```python
if num_v_heads > num_heads:
    repeat_factor = num_v_heads // num_heads
    q = jnp.repeat(q, repeat_factor, axis=1)  # [batch, num_v_heads, seq, dim]
    k = jnp.repeat(k, repeat_factor, axis=1)
```

This enables more expressive value representations while keeping Q/K computation efficient.

### Implementation Details

| Component | File | Description |
|-----------|------|-------------|
| **Core Kernels** | `modules/linear_attn/gated_deltanet.py` | `gated_delta_rule_recurrent`, `gated_delta_rule_chunkwise`, `gated_delta_rule_step` |
| **Block** | `modules/linear_attn/gated_deltanet.py` | `GatedDeltaNetBlock` - Full module with projections, conv, gating |
| **State Cache** | `modules/linear_attn/gated_deltanet.py` | `GatedDeltaNetState` - Cache for recurrent state and 3 conv caches (Q, K, V) |
| **Model Integration** | `models.py` | Presets: `GATED_DELTANET_SMALL`, `GATED_DELTANET_MEDIUM`, `GATED_DELTA_HYBRID_SMALL` |

### Block Architecture

```
Input x [batch, seq, hidden]
    │
    ▼
┌─────────────────┐
│   RMSNorm       │
└────────┬────────┘
         │
    ┌────┴────────────┬────────────┐
    ▼                 ▼            ▼
┌───────┐       ┌─────────┐  ┌─────────┐
│QKV    │       │a_proj   │  │b_proj   │
│Proj   │       │(decay)  │  │(beta)   │
└───┬───┘       └────┬────┘  └────┬────┘
    │                │            │
    ▼                │            │
┌───────┐            │            │
│Short  │            │            │
│Conv1D │            │            │
│(Q,K,V)│            │            │
└───┬───┘            │            │
    │                │            │
    ▼                ▼            ▼
┌───────┐       ┌─────────┐  ┌─────────┐
│SiLU   │       │-A*      │  │sigmoid  │
│       │       │softplus │  │(*2 opt) │
└───┬───┘       │(a+bias) │  └────┬────┘
    │           └────┬────┘       │
    │                │            │
    │      g (decay) │     beta   │
    │                │            │
    ▼                ▼            ▼
┌─────────────────────────────────────┐
│     Gated Delta Rule Attention      │
│  S = exp(g)*S + k @ (beta*(v-Sk))^T │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────┐
│  Gated RMSNorm  │  ← g_proj (optional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Proj    │
└────────┬────────┘
         │
         ▼
    + Residual
         │
         ▼
Output [batch, seq, hidden]
```

### Key Features

- ✅ **Exponential decay gate** - Learnable forgetting mechanism from Mamba2
- ✅ **Grouped Value Attention (GVA)** - Different head counts for Q/K vs V
- ✅ **Dual projection** - Separate `a_proj` (decay) and `b_proj` (beta)
- ✅ **L2 normalization** - Stable Q/K normalization by default
- ✅ **Mamba2-style initialization** - `A_log` and `dt_bias` parameter initialization
- ✅ **Separate conv caches** - Individual cache for Q, K, V convolutions
- ✅ **Output gating** - FusedRMSNormGated-style output
- ✅ **allow_neg_eigval** - Beta in `[0, 2]` for negative eigenvalues
- ✅ **Recurrent and chunk modes** - Efficient training and inference
- ✅ **Full model integration** - Works with LMModel and hybrid architectures

### Configuration Options

```python
# In ModelConfig
gated_deltanet_heads: int             # Number of Q/K heads
gated_deltanet_v_heads: int           # Number of V heads (GVA if > heads)
gated_deltanet_head_dim: int          # Head dimension (default: 256)
gated_deltanet_expand_v: float        # Value expansion factor (default: 2.0)
gated_deltanet_use_short_conv: bool   # Short convolutions (default: True)
gated_deltanet_use_gate: bool         # Output gating (default: True)
gated_deltanet_allow_neg_eigval: bool # Beta in [0, 2] (default: False)
gated_deltanet_use_qk_l2norm: bool    # L2 normalize Q and K (default: True)
```

### Usage Examples

```python
from linearnexus import create_model, LMModel
from linearnexus.models import ModelConfig, GATED_DELTANET_SMALL
import flax.nnx as nnx

# Pure Gated DeltaNet model
model = LMModel(GATED_DELTANET_SMALL, rngs=nnx.Rngs(0))

# Custom configuration
config = ModelConfig(
    hidden_size=768,
    n_layers=24,
    block_pattern=["gated_deltanet"],
    gated_deltanet_heads=3,       # 768 / 256 = 3
    gated_deltanet_v_heads=3,
    gated_deltanet_head_dim=256,
    gated_deltanet_expand_v=2.0,
    gated_deltanet_use_gate=True,
)
model = LMModel(config, rngs=nnx.Rngs(0))

# Hybrid Gated DeltaNet + Attention (every 4th is attention)
config = ModelConfig(
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["gated_deltanet", "gated_deltanet", "gated_deltanet", "attention"],
    gated_deltanet_heads=3,
)
model = LMModel(config, rngs=nnx.Rngs(0))

# Direct block usage
from linearnexus.modules.linear_attn import GatedDeltaNetBlock

block = GatedDeltaNetBlock(
    hidden_size=768,
    num_heads=3,
    num_v_heads=6,       # GVA: 2x more V heads
    head_dim=256,
    expand_v=2.0,
    use_gate=True,
    use_short_conv=True,
    rngs=nnx.Rngs(0),
)
```

### Test Coverage

23 tests covering:
- State cache initialization (basic and with conv)
- Kernel shape correctness for recurrent, chunkwise, and step functions
- Decay gate behavior (g=0 → no decay, large negative g → strong forgetting)
- Initial state handling
- Chunkwise padding for non-aligned sequences
- Step function matching first step of recurrent
- Block forward pass with various configurations
- GVA mode (more V heads than Q/K heads)
- Without convolution, without gating
- `allow_neg_eigval` mode
- Residual connection verification
- Sequential multi-step generation
- Model integration (pure and hybrid architectures)

### Implementation Notes

1. **Chunkwise algorithm**: The chunkwise implementation uses cumulative decay gates within chunks, which can have numerical differences from the recurrent mode. Both produce finite, reasonable outputs but may not match exactly.

2. **Decay gate computation**: 
   ```python
   g = -jnp.exp(A_log) * jax.nn.softplus(a_proj(x) + dt_bias)
   ```
   This ensures `g` is always negative (state decays), with learned magnitude.

3. **State cache**: Unlike DeltaNet's single conv cache, Gated DeltaNet maintains three separate conv caches for Q, K, and V, as they have different dimensions when using GVA.

### Reference Implementation

Based on [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention):
- `examples/fla/layers/gated_deltanet.py` - PyTorch layer implementation
- `examples/fla/ops/gated_delta_rule/fused_recurrent.py` - Triton recurrent kernel
- `examples/fla/ops/gated_delta_rule/chunk.py` - Chunkwise kernel with WY representation

---

## RWKV-6: Eagle and Finch Linear Attention

**Paper:** [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) & [Eagle and Finch: RWKV with Matrix-Valued States](https://arxiv.org/abs/2404.05892)

**Implementation Date:** December 2025

### Mathematical Foundation

RWKV-6 is a linear attention mechanism that achieves O(1) inference complexity through a recurrent formulation with data-dependent decay. Unlike standard linear attention, RWKV-6 uses matrix-valued states and token shifting for temporal context mixing.

#### Core Recurrence (Time Mixing)

For each token $t$:

$$h_t = \text{diag}(\exp(w_t)) \cdot h_{t-1} + k_t \otimes v_t$$
$$o_t = r_t^T \cdot h_t$$

Where:
- $r_t$: Receptance (query-like), shape `[d_k]`
- $k_t, v_t$: Key and value vectors
- $w_t$: Data-dependent decay (negative values for actual decay), shape `[d_k]`
- $u$: Bonus term for current token, shape `[d_k]`
- $h_t$: State matrix, shape `[d_k, d_v]`

The full output includes the bonus term:
$$o_t = r_t^T \cdot (h_t + u \odot (k_t \otimes v_t))$$

#### Token Shift (Key Difference from Other Linear Attention)

RWKV uses **token shifting** instead of short convolutions:
```python
def token_shift(x, shift_state=None):
    # x_shifted[t] = x[t-1]
    x_shifted = jnp.concatenate([shift_state or zeros, x[:, :-1, :]], axis=1)
    return x_shifted, x[:, -1, :]  # Cache last token
```

Token shift creates a "delta" signal used for data-dependent interpolation:
```python
delta = x_shifted - x
x_mixed = x + delta * mu  # mu is learnable interpolation weight
```

#### Data-Dependent Interpolation (LerpLinear)

RWKV-6 uses learnable linear interpolation between current and shifted tokens:
```python
class LerpLinear:
    def __call__(self, x, delta):
        # Equivalent to Linear(lerp(x, x_shifted, mu))
        return self.linear(x + delta * self.mu)
```

For deeper layers, **DDLerpLinear** (Data-Dependent Lerp) adds input-dependent interpolation via low-rank projection.

### Implementation Details

| Component | File | Description |
|-----------|------|-------------|
| **Core Kernels** | `modules/linear_attn/rwkv6.py` | `rwkv6_recurrent`, `rwkv6_step`, `rwkv6_chunkwise` |
| **Helpers** | `modules/linear_attn/rwkv6.py` | `token_shift`, `LerpLinear`, `DDLerpLinear`, `GroupNorm` |
| **Block** | `modules/linear_attn/rwkv6.py` | `RWKV6Block` - Full module with time mixing + channel mixing |
| **State Cache** | `modules/linear_attn/rwkv6.py` | `RWKV6State` - Cache for recurrent state and shift state |
| **Model Integration** | `models.py` | Presets: `RWKV6_SMALL`, `RWKV6_MEDIUM`, `RWKV6_HYBRID_SMALL` |

### Block Architecture

```
Input x [batch, seq, hidden]
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
┌───────────────────────────────────────────┐  │
│             TIME MIXING                    │  │
├───────────────────────────────────────────┤  │
│ 1. Token Shift: x_shift = shift(x)        │  │
│ 2. delta = x_shift - x                    │  │
│ 3. LerpLinear projections:                │  │
│    r = LerpLinear_r(x, delta)             │  │
│    k = LerpLinear_k(x, delta)             │  │
│    v = LerpLinear_v(x, delta)             │  │
│    w = DDLerpLinear_w(x, delta)           │  │
│    g = LerpLinear_g(x, delta)             │  │
│ 4. RWKV6 recurrence:                      │  │
│    h = exp(w)*h + k⊗v                     │  │
│    o = r^T*(h + u*(k⊗v))                  │  │
│ 5. GroupNorm + SiLU gate                  │  │
│ 6. Output projection                      │  │
└───────────────────┬───────────────────────┘  │
                    │                          │
                    + ─────────────────────────┘
                    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
┌───────────────────────────────────────────┐  │
│            CHANNEL MIXING                  │  │
├───────────────────────────────────────────┤  │
│ 1. Token Shift (separate state)           │  │
│ 2. LerpLinear projections:                │  │
│    k = LerpLinear_ck(x, delta)            │  │
│ 3. Squared ReLU: k = relu(k)²             │  │
│ 4. Value projection                       │  │
└───────────────────┬───────────────────────┘  │
                    │                          │
                    + ─────────────────────────┘
                    │
                    ▼
Output [batch, seq, hidden]
```

### Key Features

- ✅ **O(1) memory per step** - True linear complexity for inference
- ✅ **Token shifting** - Simple, effective temporal mixing without convolutions
- ✅ **Data-dependent interpolation** - LerpLinear and DDLerpLinear for adaptive mixing
- ✅ **Layer-dependent initialization** - Decay, bonus, and other params vary by layer depth
- ✅ **Two sub-blocks** - Time mixing (attention) + Channel mixing (FFN)
- ✅ **GroupNorm** - Head-wise normalization on output
- ✅ **Recurrent and chunkwise** - Both modes for efficient training/inference
- ✅ **Full model integration** - Works with LMModel and hybrid architectures

### Configuration Options

```python
# In ModelConfig
rwkv6_heads: int              # Number of heads (default: hidden_size // 64)
rwkv6_head_dim: int           # Head dimension (default: 64)
rwkv6_intermediate_size: int  # FFN intermediate size (default: hidden_size * 3.5)
rwkv6_decay_low_rank_dim: int # Low-rank dim for decay (default: 64)
```

### Layer-Dependent Initialization

RWKV-6 uses layer index for special parameter initialization:
```python
ratio_0_to_1 = layer_idx / (n_layers - 1)
ratio_1_to_almost_0 = 1.0 - (layer_idx / n_layers)

# Decay: deeper layers decay faster
decay_speed = jnp.power(h / (num_heads - 1), 4) * (1 - ratio_1_to_almost_0)

# Bonus (u): attenuates with depth
u = 1.0 - jnp.power(h / (num_heads - 1), 4 - 2 * ratio_1_to_almost_0)
```

### Test Coverage

22 tests covering:
- State initialization (basic and with shift state)
- Token shift functionality (basic, with state, single token)
- Kernel shapes and outputs
- Step function matching first recurrent step
- Decay behavior (negative w → decay)
- Block forward pass with various configurations
- Sequential multi-step generation
- Different layer indices
- Model integration (pure and hybrid)
- Numerical stability with long sequences

### Reference Implementation

Based on [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention):
- `examples/fla/layers/rwkv6.py` - PyTorch layer implementation
- `examples/fla/ops/rwkv6/recurrent_naive.py` - Reference recurrence
- `examples/fla/modules/token_shift.py` - Token shift utilities

---

## RWKV-7 (Goose): DPLR Linear Attention

**Paper:** [RWKV-7 "Goose": DPLR Linear Attention](https://github.com/BlinkDL/RWKV-LM)

**Implementation Date:** December 2025

### Mathematical Foundation

RWKV-7 introduces **DPLR (Diagonal Plus Low Rank)** transition matrices, extending RWKV-6 with a low-rank state update term. This enables more expressive state transitions while maintaining linear complexity.

#### Core Recurrence (DPLR Delta Rule)

For each token $t$:

$$h_t = h_{t-1} \cdot (D_t + a_t b_t^T) + v_t k_t^T$$

Where:
- $D_t = \text{diag}(\exp(w_t))$: Diagonal decay matrix
- $a_t, b_t$: Low-rank update vectors, enabling non-diagonal transitions
- $h_t$: State matrix, shape `[d_k, d_v]`

The output is computed as:
$$o_t = h_t \cdot q_t$$

This is equivalent to the expanded form:
$$h_t = \exp(w_t) \odot h_{t-1} + a_t \cdot (b_t^T h_{t-1}) + v_t k_t^T$$

#### Key Differences from RWKV-6

| Aspect | RWKV-6 | RWKV-7 |
|--------|--------|--------|
| **Transition** | Diagonal only: $D_t h_{t-1}$ | DPLR: $D_t h_{t-1} + a_t (b_t^T h_{t-1})$ |
| **State update** | $+ k_t \otimes v_t$ | $+ v_t k_t^T$ (note: swapped order) |
| **Low-rank term** | None | $a_t b_t^T$ for non-diagonal dynamics |
| **Projections** | LerpLinear | LoRA-style low-rank projections |
| **v_first mechanism** | None | Layer 0's v propagates to all layers |
| **K normalization** | None | L2-normalized $k_k$ for stability |

#### DPLR Decomposition

In RWKV-7, the low-rank terms are computed as:
```python
k_k = l2_normalize(k)              # Normalized key for state queries
a = -k_k                           # Low-rank component a
b = k_k * sigmoid(a_proj(x))       # Low-rank component b
```

This specific form relates to the delta rule - the low-rank update allows the model to learn when and how to modify specific parts of the state.

#### v_first Mechanism

RWKV-7 introduces cross-layer value propagation:
```python
# In layer 0:
v_first = v                        # Store first layer's v

# In subsequent layers:
v = v + (v_first - v) * lerp_v     # Interpolate with layer 0's v
```

This allows information from the first layer to influence all subsequent layers, creating a residual-like connection for values.

#### Gate Output Correction

RWKV-7 uses a special gate correction mechanism:
```python
def gate_output_correction(y, v, g):
    """Correct output with value-based correction.
    
    y: raw attention output
    v: value vectors  
    g: gate (typically k_g from projection)
    """
    y_corrected = y + (jnp.abs(v) - v) * jnp.sign(g)
    return y_corrected
```

### Implementation Details

| Component | File | Description |
|-----------|------|-------------|
| **Core Kernels** | `modules/linear_attn/rwkv7.py` | `dplr_delta_rule_recurrent`, `dplr_step`, `dplr_chunkwise` |
| **Helpers** | `modules/linear_attn/rwkv7.py` | `LoRA`, `l2_normalize`, `gate_output_correction`, `k_update` |
| **Block** | `modules/linear_attn/rwkv7.py` | `RWKV7Block` - Full module with time mixing + channel mixing |
| **State Cache** | `modules/linear_attn/rwkv7.py` | `RWKV7State` - Cache for recurrent state and shift state |
| **Model Integration** | `models.py` | Presets: `RWKV7_SMALL`, `RWKV7_MEDIUM`, `RWKV7_HYBRID_SMALL` |

### Block Architecture

```
Input x [batch, seq, hidden]
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
┌───────────────────────────────────────────┐  │
│             TIME MIXING                    │  │
├───────────────────────────────────────────┤  │
│ 1. Token Shift: x_shift, delta            │  │
│ 2. LoRA projections (low-rank):           │  │
│    w = -softplus(LoRA_w(x, delta))        │  │
│    a = sigmoid(LoRA_a(x, delta))          │  │
│    g = sigmoid(LoRA_g(x, delta))          │  │
│    v = (opt) v + lerp_v*(v_first - v)     │  │
│ 3. Compute k_k = l2_norm(k)               │  │
│    a_dplr = -k_k                          │  │
│    b_dplr = k_k * a                       │  │
│ 4. DPLR recurrence:                       │  │
│    h = exp(w)*h + a*(b^T*h) + v*k^T       │  │
│    o = h @ q                              │  │
│ 5. Gate correction:                       │  │
│    y = y + (|v| - v) * sign(k_g)          │  │
│ 6. GroupNorm + Gate                       │  │
│ 7. Output projection                      │  │
└───────────────────┬───────────────────────┘  │
                    │                          │
                    + ─────────────────────────┘
                    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          │
┌───────────────────────────────────────────┐  │
│            CHANNEL MIXING                  │  │
├───────────────────────────────────────────┤  │
│ 1. Token Shift                            │  │
│ 2. LerpLinear projection: k               │  │
│ 3. Squared ReLU: k = relu(k)²             │  │
│ 4. Value projection                       │  │
└───────────────────┬───────────────────────┘  │
                    │                          │
                    + ─────────────────────────┘
                    │
                    ▼
Output [batch, seq, hidden], v_first_out
```

### Key Features

- ✅ **DPLR transitions** - Diagonal + Low Rank state updates for expressiveness
- ✅ **LoRA-style projections** - Low-rank factorized parameter efficiency
- ✅ **v_first propagation** - Cross-layer value information flow
- ✅ **Gate output correction** - Specialized output correction mechanism
- ✅ **L2-normalized keys** - Numerical stability via key normalization
- ✅ **Layer-dependent initialization** - Sophisticated init based on layer depth
- ✅ **Orthogonal initialization** - For key projections
- ✅ **Recurrent and chunkwise modes** - Efficient training and inference
- ✅ **Full model integration** - Works with LMModel and handles v_first across blocks

### Configuration Options

```python
# In ModelConfig
rwkv7_heads: int                  # Number of heads (default: hidden_size // 64)
rwkv7_head_dim: int               # Head dimension (default: 64)
rwkv7_intermediate_size: int      # FFN intermediate size (default: hidden_size * 3.5)
rwkv7_decay_low_rank_dim: int     # LoRA dim for decay (default: 64)
rwkv7_gate_low_rank_dim: int      # LoRA dim for gate (default: 64)
rwkv7_a_low_rank_dim: int         # LoRA dim for a projection (default: 64)
rwkv7_v_low_rank_dim: int         # LoRA dim for v lerp (default: 32)
```

### LoRA Projections

RWKV-7 uses low-rank projections for parameter efficiency:
```python
class LoRA(nnx.Module):
    """Low-rank projection with activation and optional bias."""
    
    def __init__(self, input_dim, output_dim, low_rank_dim, activation="tanh"):
        self.down = Linear(input_dim, low_rank_dim)   # Down project
        self.up = Linear(low_rank_dim, output_dim)    # Up project
        self.mu = Param(zeros(input_dim))             # Learnable interpolation
        self.bias = Param(zeros(output_dim))          # Output bias
    
    def __call__(self, x, delta):
        x_mixed = x + delta * self.mu
        return self.bias + self.up(activation(self.down(x_mixed)))
```

### v_first Handling in LMModel

The model forward pass handles v_first propagation across layers:
```python
def __call__(self, tokens, states=None, mode=None):
    x = self.embed(tokens)
    v_first = None  # Will be set by first RWKV7 block
    
    for i, block in enumerate(self.blocks):
        if isinstance(block, RWKV7Block):
            x, state, v_first_out = block(x, state=state, v_first=v_first, ...)
            if v_first is None:
                v_first = v_first_out  # Capture from layer 0
        else:
            x, state = block(x, state=state, ...)
    ...
```

### Test Coverage

29 tests covering:
- State initialization (shape and dtype)
- DPLR kernel shapes and finiteness
- Step function matching first recurrent step
- Initial state propagation
- LoRA module functionality (shape, activations, bias setting)
- L2 normalization behavior
- Gate output correction (shape and finiteness)
- Block forward pass (shape and finiteness)
- Sequential generation with state
- v_first propagation across blocks
- init_state method
- Model integration (pure RWKV7, hybrid, presets)
- Sequential generation at model level
- Numerical stability (long sequences, state accumulation)
- DPLR decay effect verification
- Edge cases (batch_size=1, seq_len=1, different layer indices)

### Reference Implementation

Based on [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention):
- `examples/fla/layers/rwkv7.py` - PyTorch layer implementation
- `examples/fla/ops/rwkv7/fused_recurrent.py` - Triton recurrent kernel
- `examples/fla/ops/generalized_delta_rule/dplr/` - DPLR algorithm implementations

---

## KDA: Kimi Delta Attention (Per-Dimension Gated Linear Attention)

**Paper:** [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

**Implementation Date:** December 2025

### Mathematical Foundation

KDA (Kimi Delta Attention) extends Gated DeltaNet with **per-head per-key-dimension decay gates**. Unlike Gated DeltaNet where decay gate $g$ is a scalar per head, KDA makes it a vector matching the key dimension, enabling finer-grained control over which key dimensions decay.

#### Core Recurrence (Token-by-Token)

For each token $t$:

1. **Apply per-dim decay:** $S_t = S_{t-1} \odot \exp(g_t)_{[..., \text{None}]}$
2. **Retrieve old value:** $v_{old} = k_t^T S_t$
3. **Compute delta:** $v_{\delta} = \beta_t (v_t - v_{old})$
4. **Update state:** $S_t = S_t + (\beta_t k_t) \otimes v_{\delta}$
5. **Query output:** $o_t = q_t^T S_t$

Where:
- $q_t, k_t$: query and key vectors, shape `[d_k]`
- $v_t$: value vector, shape `[d_v]`
- $\beta_t$: scalar learning rate in `[0, 1]` (or `[0, 2]` with `allow_neg_eigval`)
- $g_t$: **per-dimension decay gate**, shape `[d_k]` (negative values)
- $S_t$: state matrix, shape `[d_k, d_v]`

#### Gate Computation

KDA computes decay gates via a two-layer MLP:
$$g = -\exp(A_{log}) \cdot \text{softplus}(f_{proj}(x) + dt_{bias})$$

Where:
- $f_{proj}$: Two-layer MLP: `hidden → head_v_dim → key_dim`
- $A_{log}$: Learnable log-decay base per head
- $dt_{bias}$: Learnable bias per dimension

#### Key Differences from Gated DeltaNet

| Aspect | Gated DeltaNet | KDA |
|--------|----------------|-----|
| **Decay gate shape** | `[batch, heads, seq]` | `[batch, heads, seq, key_dim]` |
| **Gate projection** | Single `a_proj` | Two-layer MLP (`f_proj_1`, `f_proj_2`) |
| **Gate computation** | Per-head decay | Per-head per-dimension decay |
| **Output normalization** | RMSNorm + gate | FusedRMSNormGated (combined) |
| **Output gate projection** | Single layer | Two-layer MLP (`g_proj_1`, `g_proj_2`) |
| **Expressiveness** | Per-head forgetting | Selective dimension-wise forgetting |

#### Why Per-Dimension Gates Matter

Per-dimension gating allows the model to:
1. **Selectively forget** specific key dimensions while retaining others
2. **Learn dimension-specific decay patterns** (e.g., position info vs semantic info)
3. **Better handle long-range dependencies** for some dimensions while local for others

### Implementation Details

| Component | File | Description |
|-----------|------|-------------|
| **Core Kernels** | `modules/linear_attn/kda.py` | `kda_recurrent`, `kda_chunkwise`, `kda_step` |
| **Gate Function** | `modules/linear_attn/kda.py` | `kda_gate` - Per-dim gate computation |
| **Output Norm** | `modules/linear_attn/kda.py` | `FusedRMSNormGated` - RMSNorm with sigmoid gating |
| **Block** | `modules/linear_attn/kda.py` | `KDABlock` - Full module with projections, conv, gating |
| **State Cache** | `modules/linear_attn/kda.py` | `KDAState` - Cache for recurrent state and conv caches |
| **Model Integration** | `models.py` | Presets: `KDA_SMALL`, `KDA_MEDIUM`, `KDA_HYBRID_SMALL` |

### Block Architecture

```
Input x [batch, seq, hidden]
    │
    ▼
┌─────────────────┐
│   RMSNorm       │
└────────┬────────┘
         │
    ┌────┴─────────────────┬────────────┐
    ▼                      ▼            ▼
┌───────┐           ┌───────────┐  ┌─────────┐
│QKV    │           │f_proj_1   │  │b_proj   │
│Proj   │           │(hidden→   │  │(beta)   │
│       │           │head_v_dim)│  └────┬────┘
└───┬───┘           └─────┬─────┘       │
    │                     ▼             │
    ▼               ┌───────────┐       │
┌───────┐           │f_proj_2   │       │
│Short  │           │(→key_dim) │       │
│Conv1D │           └─────┬─────┘       │
│(Q,K,V)│                 │             │
│(opt)  │                 ▼             │
└───┬───┘           ┌───────────┐       │
    │               │-exp(A_log)│       │
    ▼               │*softplus  │       │
┌───────┐           │(f+dt_bias)│       │
│SiLU   │           └─────┬─────┘       │
│       │                 │             │
└───┬───┘                 │             ▼
    │           g [b,s,h,k_dim] beta [b,s,h]
    │                 │             │
    ▼                 ▼             ▼
┌─────────────────────────────────────────┐
│      KDA: Per-Dimension Gated Delta     │
│  S = S * exp(g)[...,None]               │
│  S = S + (beta*k) @ (v - Sk)^T          │
│  o = q @ S                              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│        FusedRMSNormGated               │
│  o = RMSNorm(o) * sigmoid(g_proj(x))   │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌─────────────────┐
│  Output Proj    │
└────────┬────────┘
         │
         ▼
    + Residual
         │
         ▼
Output [batch, seq, hidden]
```

### Key Features

- ✅ **Per-dimension decay gates** - Fine-grained control over what to forget
- ✅ **Two-layer MLP for gates** - More expressive gate computation
- ✅ **Grouped Value Attention (GVA)** - `num_v_heads >= num_heads` support
- ✅ **FusedRMSNormGated** - Combined normalization and gating
- ✅ **L2 normalization** - Stable Q/K normalization by default
- ✅ **Mamba2-style parameters** - `A_log` and `dt_bias` initialization
- ✅ **Optional short convolutions** - On Q, K, V for local context
- ✅ **allow_neg_eigval** - Beta in `[0, 2]` for negative eigenvalues
- ✅ **Recurrent mode by default** - Stable numerical behavior
- ✅ **Full model integration** - Works with LMModel and hybrid architectures

### Configuration Options

```python
# In ModelConfig
kda_heads: int              # Number of Q/K heads (default: hidden_size // 128)
kda_v_heads: int            # Number of V heads (GVA if > heads)
kda_head_dim: int           # Head dimension (default: 128)
kda_expand_v: float         # Value expansion factor (default: 1.0)
kda_use_short_conv: bool    # Short convolutions (default: True)
kda_allow_neg_eigval: bool  # Beta in [0, 2] (default: False)
kda_use_qk_l2norm: bool     # L2 normalize Q and K (default: True)
```

### FusedRMSNormGated

KDA uses a specialized output normalization:
```python
class FusedRMSNormGated(nnx.Module):
    """RMSNorm fused with sigmoid gating."""
    
    def __call__(self, x, gate):
        # RMSNorm
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(variance + eps) * scale
        # Gate with sigmoid
        return x_norm * jax.nn.sigmoid(gate)
```

This is more parameter-efficient than separate norm + gate layers.

### Usage Examples

```python
from linearnexus import create_model, LMModel
from linearnexus.models import ModelConfig, KDA_SMALL
import flax.nnx as nnx

# Pure KDA model
model = LMModel(KDA_SMALL, rngs=nnx.Rngs(0))

# Via preset
config, _ = create_model("kda-small")
model = LMModel(config, rngs=nnx.Rngs(0))

# Custom configuration
config = ModelConfig(
    hidden_size=768,
    n_layers=24,
    block_pattern=["kda"],
    kda_heads=6,           # 768 / 128 = 6
    kda_v_heads=6,
    kda_head_dim=128,
    kda_expand_v=1.0,
)
model = LMModel(config, rngs=nnx.Rngs(0))

# Hybrid KDA + Attention (every 4th is attention)
config = ModelConfig(
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["kda", "kda", "kda", "attention"],
    kda_heads=6,
)
model = LMModel(config, rngs=nnx.Rngs(0))

# Direct block usage with GVA
from linearnexus.modules.linear_attn import KDABlock

block = KDABlock(
    hidden_size=768,
    num_heads=6,
    num_v_heads=12,      # GVA: 2x more V heads
    head_dim=128,
    expand_v=1.0,
    use_short_conv=True,
    rngs=nnx.Rngs(0),
)
```

### Implementation Notes

1. **Recurrent mode by default**: The block defaults to recurrent mode (even for multi-token inputs) because the chunkwise algorithm for per-dimension gates is complex and our reference implementation has numerical issues. Recurrent mode is numerically stable and efficient via JAX's `lax.scan`.

2. **Gate computation**: The two-layer MLP for gate projection allows the model to learn complex per-dimension decay patterns:
   ```python
   f = f_proj_2(f_proj_1(x))  # hidden → head_v_dim → key_dim
   g = -exp(A_log) * softplus(f + dt_bias)  # Always negative
   ```

3. **Memory efficiency**: Per-dimension gates increase the gate tensor size from `[batch, heads, seq]` to `[batch, heads, seq, key_dim]`, but this is acceptable given the expressive benefits.

4. **Numerical stability**: L2 normalization on Q/K with epsilon `1e-6` ensures stable dot products.

### Test Coverage

34 tests covering:
- State initialization (with and without conv)
- Gate computation (shape, negative values, bias effect, beta sigmoid)
- Kernel shapes and outputs (recurrent, chunkwise, step)
- Step function matching first recurrent step
- Initial state propagation
- FusedRMSNormGated (shape, finiteness, gate effect)
- Block forward pass (shape, finiteness)
- Sequential generation with state caching
- GVA mode (more V heads than Q/K heads)
- Without convolution configurations
- `allow_neg_eigval` mode
- init_state method
- Residual connection verification
- Model integration (pure KDA, hybrid, presets)
- Sequential generation at model level
- Numerical stability (long sequences, state accumulation, decay effect)
- Edge cases (batch_size=1, seq_len=1, expand_v factor)

### Reference Implementation

Based on [flash-linear-attention (FLA)](https://github.com/fla-org/flash-linear-attention):
- `examples/fla/layers/kda.py` - PyTorch layer implementation
- `examples/fla/ops/kda/naive.py` - Reference recurrent and chunk kernels
- `examples/fla/ops/kda/fused_recurrent.py` - Triton recurrent kernel
- `examples/fla/ops/kda/gate.py` - Gate computation

---

*Future implementations will be documented below.*

