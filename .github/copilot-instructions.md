# LinearNexus AI Coding Agent Instructions

LinearNexus is a **minimal LLM training framework in JAX** — nanoGPT-style simplicity with multi-architecture support. Train GPT, Mamba, or hybrid models from scratch with custom optimizers (AdamW, Muon, Sophia) and modern training paradigms (SFT, GRPO, PPO).

Built on JAX + Flax NNx for clean, functional design.

## Core Architecture

**Layered abstraction** with unified block protocol:
```
models.py (LMModel, configs) → modules/ (blocks) → core/ (utilities) → kernels/ (compute)
```

- **`models.py`**: `LMModel`, `ModelConfig`, presets (`GPT_SMALL`, `MAMBA_SMALL`, `JAMBA_SMALL`). Uses `block_pattern` to define architecture (e.g., `["attention"]`, `["mamba"]`, `["mamba", "mamba", "attention"]`).
- **`modules/`**: Neural network building blocks implementing unified interface:
  - `attention/`: `CausalSelfAttention`, `AttentionBlock`, `KVCache`, `MultiHeadLatentAttention` (MLA)
  - `ssm/`: `MambaBlock`, `Mamba2Block`, `MambaState`, `Mamba2State`
  - `linear_attn/`: `DeltaNetBlock`, `GatedDeltaNetBlock`, `RWKV6Block` and states
  - `common.py`: `MLP`, `RMSNorm`, `Embedding`, `RotaryEmbedding`
  - `sparse/`, `hybrid/`: Planned for future phases
- **`core/`**: Shared utilities (`ConvState`, `RecurrentState`, `depthwise_conv1d_causal`, `KernelMode`, `ConfigBase`)
- **`kernels/`**: Compute primitives (`MambaReferenceKernel` with `forward_chunk()` and `forward_recurrent()`)
- **`train/`**: Training loops (`SFTTrainer`, `GRPOTrainer`, `PPOTrainer`)
- **`optim.py`**: Custom optimizers (`adamw`, `muon`, `sophia`)
- **`data.py`**: Tokenizers (`CharTokenizer`, `BPETokenizer`), `TextDataset`, `DataLoader`
- **`generate.py`**: Text generation with KV/SSM state caching

**Key Pattern**: All blocks implement the same interface:
```python
def __call__(self, x, *, state=None, mask=None, mode=None) -> tuple[output, new_state]
```
This allows `LMModel` to iterate blocks uniformly, enabling GPT, Mamba, or hybrid models via `ModelConfig.block_pattern`.

## JAX/Flax NNx Specifics

- **Immutability**: JAX arrays are immutable. Use `array.at[idx].set(val)` not `array[idx] = val`.
- **PRNG keys**: Always split keys explicitly: `key, subkey = jax.random.split(key)`. Never reuse the same key.
- **NNx parameters**: `nnx.Param` for trainable weights. Access via `.value`.
- **State threading**: Blocks return `(output, new_state)` tuples. Thread state through sequential calls for autoregressive generation.
- **Type hints**: Required. Use `jax.Array` for array types, `Optional[T]` for nullable args.

## Code Patterns

### Block Structure (Unified Interface)
```python
class MyBlock(nnx.Module):
    def __init__(self, config: MyConfig, rngs: nnx.Rngs):
        self.norm = RMSNorm(config.hidden_size)
        self.proj = nnx.Linear(..., rngs=rngs)
    
    def __call__(self, x, *, state=None, mask=None, mode=None):
        # 1. Normalize input
        # 2. Compute (attention/SSM/etc.)
        # 3. Residual connection
        return output, new_state
    
    def init_state(self, batch_size: int) -> BlockState:
        return MyState.zeros(batch_size, ...)
```

### Adding New Blocks
1. Implement `YourBlock(nnx.Module)` with `__call__` and `init_state` (see `docs/adding_new_blocks.md`)
2. Add exports in `modules/your_category/__init__.py`
3. Register block type in `_create_block` in `models.py`
4. Add tests under `tests/` to validate shape, behavior, and caching

### Testing Requirements
- **Shape correctness**: Verify `[batch, seq, hidden]` flows correctly
- **State caching**: Test incremental generation with cached state
- **Numerical parity**: Chunk vs recurrent modes should align (rtol=1e-4)
- **Use small shapes**: `hidden_size=64, seq_len=32` for fast iteration

## File Organization & Entry Points

**Critical paths**:
```
linearnexus/
├── models.py           # LMModel, ModelConfig, create_model(), presets
├── train/              # SFTTrainer, GRPOTrainer, PPOTrainer
├── optim.py            # adamw, muon, sophia, create_optimizer()
├── data.py             # Tokenizers, TextDataset, DataLoader
├── generate.py         # generate(), complete(), batch_generate()
├── checkpoint.py       # CheckpointManager, save/load utilities
├── modules/
│   ├── common.py       # MLP, RMSNorm, Embedding, RotaryEmbedding
│   ├── attention/
│   │   ├── causal.py   # CausalSelfAttention, AttentionBlock, KVCache
│   │   └── mla.py      # MultiHeadLatentAttention, MLABlock, MLACache
│   └── ssm/
│       └── mamba.py    # MambaBlock, MambaState, selective_scan_ref
├── core/
│   ├── cache.py        # ConvState, RecurrentState
│   ├── conv.py         # depthwise_conv1d_causal
│   ├── mode.py         # KernelMode, select_mode
│   └── config.py       # ConfigBase dataclass
└── kernels/
    └── mamba_reference.py  # MambaReferenceKernel (pure JAX)
```

**CLI tools**:
- `train_lm.py`: Training script with model/optimizer selection
- `sample.py`: Text generation from checkpoints

## Common Gotchas

1. **Layout mismatches**: Public APIs use `[batch, seq, hidden]`. Kernels/SSM internals may use `[batch, intermediate, seq]`. Always transpose explicitly and comment shapes.
2. **State initialization**: Use `KVCache.zeros()`, `MambaState.zeros()`, or block's `init_state()` method. Never create states manually.
3. **Mode selection**: `chunk` mode for training (parallel), `recurrent` mode for generation (token-by-token with caching).
4. **Discretization**: Mamba's `A_discrete = exp(a_log * delta)` is input-dependent per token. Never cache.
5. **Depthwise conv**: Use `feature_group_count=channels` in `conv_general_dilated` for efficiency.

## Development Workflow

**Setup** (first time):
```bash
pip install -e .           # Core dependencies (JAX, Flax, etc.)
pip install -e .[dev]      # Adds pytest, black, ruff
pip install -e .[full]     # Adds tiktoken, wandb, tqdm
```

**Test cycle** (fast feedback):
```bash
pytest -q                              # All tests
pytest tests/test_mla.py -v            # Single test file
pytest tests/test_comprehensive.py -v  # Full model tests
```

**Training** (quick validation):
```bash
python train_lm.py --model gpt-small --download-shakespeare --max-steps 100 --batch-size 2
python train_lm.py --model mamba-small --optimizer muon --lr 1e-3
python train_lm.py --model jamba-small  # Hybrid model
```

**Generation**:
```bash
python sample.py --checkpoint checkpoints/step_100 --prompt "Hello" --max-tokens 50
```

**Debug mode** (when tracing errors):
```python
import jax
jax.config.update('jax_disable_jit', True)
```

**Typical workflow**:
1. Implement block in `modules/` with unified interface
2. Register in `models.py` `_create_block()`
3. Add tests under `tests/`
4. Validate with training script

## Documentation Standards

- **Docstrings**: Required for public APIs. Include shape annotations in Args/Returns.
- **Inline comments**: Annotate tensor shapes at every transformation: `# [batch, seq, hidden]`
- **Tests**: Include one-line comments explaining what each test validates.

## Model Presets

```python
from linearnexus import create_model, LMModel
import flax.nnx as nnx

# GPT-2 small style
config, _ = create_model("gpt-small")
model = LMModel(config, rngs=nnx.Rngs(0))

# Mamba (pure SSM)
config, _ = create_model("mamba-small")

# Jamba-style hybrid (every 8th layer is attention)
config, _ = create_model("jamba-small")

# Custom hybrid pattern
from linearnexus.models import ModelConfig
config = ModelConfig(
    hidden_size=768,
    n_layers=12,
    block_pattern=["mamba", "mamba", "attention"],  # Repeats to fill n_layers
)
```

## Reference Implementations (`examples/fla/`)

The `examples/fla/` directory contains **PyTorch reference implementations** from [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for linear/subquadratic attention mechanisms. **Consult these when implementing new architecture blocks** — they provide battle-tested implementations to port to JAX/Flax NNx.

### Structure
```
examples/fla/
├── layers/          # High-level attention modules (nn.Module wrappers)
│   ├── gla.py       # Gated Linear Attention
│   ├── delta_net.py # DeltaNet (delta rule)
│   ├── mamba.py     # Mamba SSM
│   ├── mamba2.py    # Mamba-2
│   ├── rwkv6.py     # RWKV-6
│   ├── rwkv7.py     # RWKV-7
│   ├── linear_attn.py # Basic linear attention
│   ├── based.py     # Based linear attention
│   ├── rebased.py   # ReBased attention
│   ├── hgrn.py      # HGRN (Hierarchical Gated RNN)
│   ├── hgrn2.py     # HGRN-2
│   ├── gsa.py       # Gated Slot Attention
│   ├── mla.py       # Multi-head Latent Attention
│   ├── nsa.py       # Native Sparse Attention
│   └── ...          # 30+ attention variants
│
├── ops/             # Low-level kernel implementations
│   ├── gla/         # chunk_gla, fused_chunk_gla, fused_recurrent_gla
│   ├── delta_rule/  # chunk_delta_rule, fused_recurrent_delta_rule
│   ├── retention/   # chunk_retention, fused_recurrent_retention
│   ├── rwkv6/       # chunk_rwkv6, fused_recurrent_rwkv6
│   ├── linear_attn/ # chunk_linear_attn, fused_recurrent_linear_attn
│   └── ...          # Triton kernels for each mechanism
│
├── modules/         # Shared components
│   ├── rotary.py    # Rotary embeddings
│   ├── convolution.py # Short convolutions
│   ├── layernorm.py # Various norm implementations
│   ├── mlp.py       # MLP variants
│   └── activations.py # Activation functions
│
└── models/          # Full model implementations
    ├── gla/         # GLA transformer
    ├── delta_net/   # DeltaNet model
    ├── mamba/       # Mamba model
    ├── retnet/      # RetNet
    └── ...          # 25+ complete models
```

### How to Use When Adding New Blocks

1. **Find the reference**: Look in `examples/fla/layers/<mechanism>.py` for the PyTorch layer
2. **Study the ops**: Check `examples/fla/ops/<mechanism>/` for chunk/recurrent kernels
3. **Port to JAX**: Convert PyTorch patterns to JAX/Flax NNx following our block protocol

**Example**: To implement GLA (Gated Linear Attention):
```python
# 1. Reference: examples/fla/layers/gla.py (GatedLinearAttention class)
# 2. Ops: examples/fla/ops/gla/ (chunk_gla, fused_recurrent_gla)
# 3. Key components to port:
#    - Q/K/V projections with expand_k, expand_v
#    - Optional short convolution (ShortConvolution)
#    - Gate projection with low-rank decomposition
#    - chunk_gla kernel → JAX lax.scan implementation
```

### Available Mechanisms (Priority for LinearNexus)

| Mechanism | Reference | Status | Description |
|-----------|-----------|--------|-------------|
| **Attention** | native | ✅ Implemented | Standard causal self-attention with KV cache |
| **MLA** | `layers/mla.py` | ✅ Implemented | Multi-head Latent Attention (DeepSeek) |
| **Mamba** | `layers/mamba.py` | ✅ Implemented | Mamba SSM with selective scan |
| **Mamba-2** | `layers/mamba2.py` | ✅ Implemented | Mamba-2 with SSD |
| **DeltaNet** | `layers/delta_net.py` | ✅ Implemented | Delta rule for linear transformers |
| **Gated DeltaNet** | `layers/gated_deltanet.py` | ✅ Implemented | DeltaNet with multiplicative gating |
| **RWKV-6** | `layers/rwkv6.py` | ✅ Implemented | RWKV-6 with token shift and time mixing |
| **RWKV-7** | `layers/rwkv7.py` | ✅ Implemented | RWKV-7 "Goose" with DPLR transitions |
| **GLA** | `layers/gla.py` | 🔲 Planned | Gated Linear Attention with hardware-efficient training |
| **RetNet** | `layers/multiscale_retention.py` | 🔲 Planned | Multi-scale retention |
| **Based** | `layers/based.py` | 🔲 Planned | Based linear attention |
| **HGRN** | `layers/hgrn.py`, `hgrn2.py` | 🔲 Planned | Hierarchical Gated RNN |
| **Lightning** | `ops/lightning_attn/` | 🔲 Planned | Lightning attention kernels |

### Porting Checklist

When porting from FLA reference to LinearNexus:
1. [ ] Convert `nn.Module` → `nnx.Module`
2. [ ] Convert `nn.Linear` → `nnx.Linear` (note: kernel vs weight transpose)
3. [ ] Replace PyTorch ops with JAX equivalents (`torch.einsum` → `jnp.einsum`)
4. [ ] Implement `__call__(x, *, state=None, mask=None, mode=None)` interface
5. [ ] Add `init_state(batch_size)` method returning appropriate cache
6. [ ] Convert Triton kernels to pure JAX (`lax.scan`) for reference impl
7. [ ] Test chunk vs recurrent parity (rtol=1e-4)

**RWKV-specific additions:**
8. [ ] Implement token shift helper with state caching
9. [ ] Add LerpLinear/DDLerpLinear layers if using data-dependent interpolation
10. [ ] Pass `layer_idx` through `_create_block()` for layer-dependent init
11. [ ] Implement both time mixing AND channel mixing sub-blocks
12. [ ] Use GroupNorm instead of RMSNorm for output normalization (if needed)

## Porting FLA to LinearNexus: Best Practices & Methodology

This section provides a systematic approach for implementing new architecture blocks from the FLA reference implementations. Based on lessons learned from porting **DeltaNet** and **Gated DeltaNet**.

### Phase 1: Understand the Math First

**CRITICAL**: Before writing any code, fully understand the mathematical formulation.

1. **Read the paper** or find the core equations in `examples/fla/ops/<mechanism>/README.md`
2. **Identify the recurrence relation**: What is the token-by-token update rule?
   - DeltaNet: `S_t = S_{t-1} + k_t @ (β_t * (v_t - S_{t-1} @ k_t))^T`
   - Gated DeltaNet: `S_t = exp(g_t) * S_{t-1} + k_t @ (β_t * (v_t - exp(g_t)*S_{t-1} @ k_t))^T`
3. **Identify the parallel/chunk form**: How is it parallelized for training?
   - Look for WY decomposition, matrix inversions, or scan patterns
4. **Write down tensor shapes** at each step before coding

**Key files to read in FLA:**
```
examples/fla/ops/<mechanism>/
├── naive.py          # ← START HERE if exists (cleanest reference)
├── fused_recurrent.py # ← If no naive.py, read the Triton kernel loop
├── README.md         # Mathematical derivation (if exists)
├── chunk.py          # Chunkwise parallel implementation
└── __init__.py       # Public API
```

**Pro tip**: If there's no `naive.py`, look at the Triton kernel in `fused_recurrent.py`. The innermost loop (`for _ in range(0, T):`) contains the recurrence relation in clear form.

### Phase 2: Study the FLA Reference Implementation

**Reading order** (most important first):

1. **`ops/<mechanism>/naive.py`** or **`fused_recurrent.py`**: Reference kernel
   - This is the ground truth for correctness
   - In Triton kernels, look for the main loop - it shows the exact recurrence
   - Example from Gated DeltaNet Triton:
     ```python
     b_h *= exp(b_g)                              # Decay state
     b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))  # Delta
     b_h += b_k[:, None] * b_v                    # Update
     b_o = tl.sum(b_h * b_q[:, None], 0)          # Query
     ```

2. **`layers/<mechanism>.py`**: High-level module
   - Shows full architecture: projections, convolutions, gating
   - **Pay attention to**:
     - `__init__`: All learnable parameters and their initialization
     - `forward`: The exact order of operations
     - Special parameters like `A_log`, `dt_bias` (Mamba-style decay)
   - Identifies configurable options (expand_k, expand_v, use_gate, num_v_heads, etc.)

3. **`ops/<mechanism>/chunk.py`**: Chunkwise implementation
   - Shows how to parallelize the recurrence
   - **Warning**: Chunkwise with gating (cumulative gates) is complex
   - For gated mechanisms, consider using recurrent-only for initial implementation

**Key patterns to identify:**
- Input projections (Q, K, V, beta/b_proj, a_proj for decay, gate)
- Short convolutions (common in linear attention - typically on Q, K, V separately)
- Activation functions (silu after conv is standard)
- Normalization (L2 norm on Q/K is common for stability)
- Output gating and projections
- **GVA (Grouped Value Attention)**: Different head counts for Q/K vs V

### Phase 3: Implementation Order

**Do NOT try to implement everything at once.** Follow this order:

#### Step 1: Core Kernel (Recurrent) - THE MOST IMPORTANT
```python
def mechanism_recurrent(q, k, v, ..., initial_state=None):
    """Token-by-token reference implementation.
    
    Port directly from naive.py or fused_recurrent.py Triton loop:
    - torch.Tensor → jax.Array
    - for loop → lax.scan
    - torch.einsum → jnp.einsum
    - .clone() → (not needed in JAX, arrays are immutable)
    - tl.load/tl.store → direct array access
    
    ALWAYS compute in float32 for numerical stability,
    then cast output back to input dtype.
    """
    # Initialize state
    if initial_state is None:
        S = jnp.zeros((batch, heads, key_dim, value_dim), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)
    
    def step(S, inputs):
        k_t, v_t, q_t, ... = inputs
        # Exact recurrence from FLA
        ...
        return S_new, o_t
    
    # Transpose for scan: [seq, batch, heads, dim]
    final_state, outputs = lax.scan(step, S, inputs_transposed)
    return output.astype(q.dtype), final_state
```

#### Step 2: Step Function (for generation) - DERIVE FROM RECURRENT
```python
def mechanism_step(q, k, v, ..., state):
    """Single token update - literally one iteration of recurrent loop.
    
    This should be extractable directly from the step() function above.
    """
```

#### Step 3: Core Kernel (Chunkwise) - OPTIONAL FOR GATED MECHANISMS
```python
def mechanism_chunkwise(q, k, v, ..., chunk_size=64):
    """Parallel chunk implementation.
    
    WARNING: For mechanisms with cumulative gates (like Gated DeltaNet),
    the chunkwise algorithm is complex and may have numerical differences.
    
    Consider skipping this initially and using recurrent for all modes.
    The recurrent mode is fast enough for training with JAX's scan.
    """
```

#### Step 4: State Cache
```python
@dataclass
class MechanismState:
    S: jax.Array                          # Recurrent state [batch, heads, k_dim, v_dim]
    conv_state_q: Optional[jax.Array]     # Conv cache for Q [batch, kernel-1, q_dim]
    conv_state_k: Optional[jax.Array]     # Conv cache for K
    conv_state_v: Optional[jax.Array]     # Conv cache for V
    
    @classmethod
    def zeros(cls, batch_size, num_heads, key_dim, value_dim, 
              key_dim_total=None, value_dim_total=None, conv_size=4, use_conv=True):
        """Factory for empty state.
        
        IMPORTANT: Include ALL cache dimensions needed for generation.
        Separate conv caches per projection is cleaner than combined.
        """
```

#### Step 5: Full Block
```python
class MechanismBlock(nnx.Module):
    def __init__(self, hidden_size, num_heads, ..., *, rngs: nnx.Rngs):
        # IMPORTANT: rngs must be keyword-only for compatibility with _create_block()
        
        # 1. Input norm (required)
        self.norm = RMSNorm(hidden_size, eps=norm_eps, rngs=rngs)
        
        # 2. Projections
        self.q_proj = nnx.Linear(hidden_size, key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, value_dim, use_bias=False, rngs=rngs)
        
        # 3. Mechanism-specific params (beta, decay, etc.)
        # Use nnx.Param for learnable scalars/vectors
        self.A_log = nnx.Param(jnp.log(init_A))  # Example: decay parameter
        
        # 4. Conv weights (if using short conv)
        if use_short_conv:
            self.q_conv_weight = nnx.Param(jax.random.normal(key, (conv_size, key_dim)) * 0.02)
        
        # 5. Output (norm + gate + proj)
        self.o_norm = RMSNorm(head_v_dim, eps=norm_eps, rngs=rngs)
        self.o_proj = nnx.Linear(value_dim, hidden_size, use_bias=False, rngs=rngs)
        
    def __call__(self, x, *, state=None, mask=None, mode=None):
        batch, seq_len, _ = x.shape
        residual = x
        
        # 1. Normalize
        x = self.norm(x)
        
        # 2. Project
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 3. Short conv (if enabled) - BEFORE activation
        if self.use_short_conv:
            q, conv_q = depthwise_conv1d_causal(q, self.q_conv_weight.value, cache=...)
            q = jax.nn.silu(q)  # Activation AFTER conv
            # Same for k, v
        
        # 4. Reshape to heads: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        # 5. GVA: Repeat Q/K heads if needed
        if num_v_heads > num_heads:
            q = jnp.repeat(q, num_v_heads // num_heads, axis=1)
        
        # 6. Compute beta, decay, etc. from projections
        beta = jax.nn.sigmoid(self.b_proj(x))
        
        # 7. Apply kernel
        if mode == "recurrent" or seq_len == 1:
            o, new_S = mechanism_recurrent(q, k, v, ...)
        else:
            o, new_S = mechanism_chunkwise(q, k, v, ...)
        
        # 8. Output processing
        o = o.transpose(0, 2, 1, 3)  # [batch, seq, heads, dim]
        if self.use_gate:
            g = self.g_proj(self.norm(residual))  # Gate from ORIGINAL input
            o = self.o_norm(o) * jax.nn.sigmoid(g)
        o = o.reshape(batch, seq_len, -1)
        output = self.o_proj(o) + residual  # Residual connection
        
        # 9. State handling
        new_state = None
        if state is not None or mode == "recurrent":
            new_state = MechanismState(S=new_S, ...)
        
        return output, new_state
```

### Phase 4: Critical Conversion Patterns

#### PyTorch → JAX Tensor Operations
```python
# PyTorch                          # JAX
x.clone()                       →  x  # Arrays are immutable
x[idx] = val                    →  x.at[idx].set(val)
torch.einsum('bhld,bhmd->bhlm') →  jnp.einsum('bhld,bhmd->bhlm')
x.unsqueeze(-1)                 →  x[..., None]
x.squeeze(-1)                   →  x.squeeze(-1)  # Same!
torch.tril(x)                   →  jnp.tril(x)
torch.zeros_like(x)             →  jnp.zeros_like(x)
x.new_zeros(shape)              →  jnp.zeros(shape, dtype=x.dtype)
F.softplus(x)                   →  jax.nn.softplus(x)
F.silu(x)                       →  jax.nn.silu(x)
x.sigmoid()                     →  jax.nn.sigmoid(x)
torch.exp(x)                    →  jnp.exp(x)
x.float()                       →  x.astype(jnp.float32)
```

#### Triton → JAX Patterns
```python
# Triton                           # JAX
tl.load(p_q, mask=mask)         →  q_t = q[..., t, :]  # Direct indexing
tl.store(p_o, b_o)              →  (handled by scan output)
tl.sum(b_h * b_k[:, None], 0)   →  jnp.einsum('bhkv,bhk->bhv', S, k_t)
b_h *= exp(b_g)                 →  S = S * jnp.exp(g_t)[..., None, None]
b_k[:, None] * b_v              →  jnp.einsum('bhk,bhv->bhkv', k_t, v_delta)
```

#### Loop → lax.scan Pattern
```python
# PyTorch / Triton pattern
S = initial_state
for t in range(seq_len):
    k_t, v_t, q_t = k[:,:,t], v[:,:,t], q[:,:,t]
    # ... update S ...
    o_t = query(q_t, S)
    outputs.append(o_t)

# JAX equivalent
def step(S, inputs):
    k_t, v_t, q_t, ... = inputs
    # ... update S (same math) ...
    o_t = query(q_t, S)
    return S_new, o_t

# IMPORTANT: Transpose so seq is first axis for scan
inputs = (
    jnp.transpose(k, (2, 0, 1, 3)),  # [seq, batch, heads, dim]
    jnp.transpose(v, (2, 0, 1, 3)),
    jnp.transpose(q, (2, 0, 1, 3)),
    ...
)
final_state, outputs = lax.scan(step, S, inputs)
output = jnp.transpose(outputs, (1, 2, 0, 3))  # Back to [batch, heads, seq, dim]
```

#### LinearNexus Core Utilities
```python
# Use existing utilities instead of reimplementing:
from linearnexus.core.conv import depthwise_conv1d_causal
from linearnexus.modules.common import RMSNorm, MLP

# depthwise_conv1d_causal signature:
# inputs: [batch, seq, channels]
# weight: [kernel_size, channels]  
# bias: Optional[channels]
# cache: Optional[batch, kernel-1, channels]
# Returns: (output, new_cache)

# IMPORTANT: This is DIFFERENT from PyTorch's conv shapes!
# PyTorch conv: [batch, channels, seq], weight: [out, in, kernel]
# LinearNexus:  [batch, seq, channels], weight: [kernel, channels]
```

### Phase 5: Common Pitfalls & Solutions

#### 1. Shape Mismatches (MOST COMMON ISSUE)
**Problem**: FLA uses `[batch, heads, seq, dim]` internally, but LinearNexus public API uses `[batch, seq, hidden]`.

**Solution**: Transpose explicitly at block boundaries:
```python
# After projection, before kernel:
q = q.reshape(batch, seq, heads, dim)
q = jnp.transpose(q, (0, 2, 1, 3))  # → [batch, heads, seq, dim]

# After kernel, before output projection:
output = jnp.transpose(output, (0, 2, 1, 3))  # → [batch, seq, heads, dim]
output = output.reshape(batch, seq, heads * dim)  # → [batch, seq, hidden]
```

#### 2. RMSNorm Requires `rngs`
**Problem**: `RMSNorm(dim)` fails because it needs `rngs` parameter.

**Solution**: Always pass `rngs`:
```python
self.norm = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
```

#### 3. Conv Cache Shape
**Problem**: `depthwise_conv1d_causal` expects specific shapes.

**Solution**: Check `core/conv.py` for exact signature:
- inputs: `[batch, seq, channels]` (NOT `[batch, channels, seq]`)
- weight: `[kernel_size, channels]` (NOT `[channels, 1, kernel_size]`)
- cache: `[batch, kernel-1, channels]`

#### 4. Block Interface Compatibility
**Problem**: Block must work with `models.py` `_create_block()`.

**Solution**: Use keyword-only `rngs` argument:
```python
def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    expand_k: float = 1.0,
    # ... other params ...
    *,
    rngs: nnx.Rngs,  # ← keyword-only, MUST be named 'rngs'
):
```

#### 5. State Return Convention
**Problem**: When to return state vs None?

**Solution**: Follow this pattern:
```python
# In __call__:
new_state = None
if state is not None or mode == "recurrent":
    new_state = MechanismState(S=final_S, conv_state_q=..., conv_state_k=..., conv_state_v=...)
return output, new_state
```

#### 6. Numerical Stability
**Problem**: Cumulative products/sums can explode or vanish.

**Solution**:
- Always compute recurrence in `float32`, cast output back to input dtype
- L2-normalize Q and K for linear attention mechanisms
- Use `+ 1e-6` in normalizations to avoid division by zero
- Small random init for conv weights (`* 0.02`)

#### 7. Gating from Original Input
**Problem**: Output gate should be computed from original input, not normalized.

**Solution**:
```python
def __call__(self, x, ...):
    residual = x
    x = self.norm(x)
    # ... compute o ...
    if self.use_gate:
        g = self.g_proj(self.norm(residual))  # Gate from ORIGINAL normalized
        o = self.o_norm(o) * jax.nn.sigmoid(g)
```

#### 8. Chunkwise vs Recurrent Parity
**Problem**: Chunkwise algorithm may not match recurrent exactly, especially with gating.

**Solution**:
- For mechanisms with cumulative gates, chunkwise is complex
- Test that both produce finite outputs (no NaN/Inf)
- Use recurrent for generation, chunkwise for training
- Relaxed tolerance is OK: `rtol=0.1` for gated mechanisms

#### 9. Einsum Index Mismatches
**Problem**: FLA reference uses different index conventions. Copy-pasting einsum strings causes shape errors.

**Example bug**:
```python
# FLA PyTorch (indices don't map to JAX reshape)
x_proj = torch.einsum('bsnr,hsr->bsnh', x_proj, self.x_proj_up.weight)

# WRONG JAX port (kept same indices, but tensor shapes differ):
x_proj = jnp.einsum('bsnr,hsr->bsnh', x_proj, self.x_proj_up.value)
# Error: einsum subscript 's' has size 4 for operand 1, but size 16 for operand 0

# CORRECT: Analyze actual shapes and adjust indices:
# x_proj: [batch, seq, num_heads, low_rank] → use 'btnr' (t=seq, n=heads, r=rank)
# weight: [heads, head_dim, low_rank] → use 'hnr'
# output: [batch, seq, heads, head_dim] → 'btnh'
x_proj = jnp.einsum('btnr,hnr->btnh', x_proj, self.x_proj_up.value)
```

**Solution**:
1. **Print shapes** before einsum: `print(f"x_proj: {x_proj.shape}, weight: {weight.shape}")`
2. **Map dimensions explicitly**: Write out what each index means
3. **Don't trust FLA indices** - they may use different reshape conventions
4. **Test immediately** after adding einsum with small inputs

#### 10. Preset Override Compatibility
**Problem**: `create_model("preset", ..., hidden_size=64)` may create incompatible configs.

**Example**:
```python
# RWKV6_SMALL preset has rwkv6_heads=12, hidden_size=768
# User overrides hidden_size without overriding rwkv6_heads:
model = create_model("rwkv6-small", hidden_size=64, n_heads=2)  # FAILS!
# Error: hidden_size (64) must be divisible by num_heads (12)
```

**Solution**:
1. Presets should use derived defaults when possible:
```python
def __post_init__(self):
    if self.rwkv6_heads is None:
        self.rwkv6_heads = max(1, self.hidden_size // 64)  # Auto-derive
```

2. Tests should use **compatible** overrides:
```python
# WRONG: mixing incompatible overrides
model = create_model("rwkv6-small", hidden_size=64, n_heads=2)

# RIGHT: only override compatible params, or use custom config
model = create_model("rwkv6-small", n_layers=2)  # Keep hidden_size/heads
# OR
config = ModelConfig(hidden_size=64, rwkv6_heads=2, ...)  # Full custom
```

3. Document which params must be overridden together in preset docstrings

#### 11. Module-Specific FLA Files
**Problem**: Not all mechanisms follow the same FLA file structure.

**RWKV-specific files**:
```
examples/fla/
├── modules/token_shift.py      # ← Token shift utilities (CRITICAL for RWKV)
├── layers/rwkv6.py             # High-level layer
└── ops/rwkv6/
    ├── recurrent_naive.py      # ← Reference recurrence (START HERE)
    ├── chunk_naive.py          # Chunkwise reference
    └── fused_recurrent.py      # Triton kernel
```

**Solution**: Check for mechanism-specific modules:
- `modules/token_shift.py` for RWKV
- `modules/short_conv.py` for GLA/DeltaNet
- `modules/rotary.py` for attention variants

### Phase 6: Testing Strategy

#### Test Order (progressive complexity):
1. **State initialization**: `State.zeros()` produces correct shapes
2. **Kernel shapes**: Verify input/output dimensions
3. **Recurrent basic behavior**: g=0 → no decay, β=0 → no update, β=1 → updates
4. **Step function**: Single step matches first step of recurrent
5. **Recurrent vs chunkwise**: Both produce finite outputs, similar magnitude
6. **Block forward**: Full block with projections
7. **State caching**: Multi-step generation maintains consistency
8. **Model integration**: Works in LMModel with hybrid patterns

#### Test Template:
```python
class TestMechanismKernels:
    def test_recurrent_output_shape(self, key):
        """Verify output shapes from recurrent kernel."""
        q = jax.random.normal(key, (batch, heads, seq, dim))
        ...
        output, state = mechanism_recurrent(q, k, v, ...)
        assert output.shape == (batch, heads, seq, value_dim)
        assert state.shape == (batch, heads, key_dim, value_dim)
    
    def test_step_matches_recurrent_first_step(self, key):
        """Single step should match first step of recurrent."""
        # Single token inputs
        q = jax.random.normal(key, (batch, heads, key_dim))
        ...
        out_step, state_step = mechanism_step(q, k, v, ..., state=zeros)
        
        # Recurrent with seq_len=1
        out_rec, state_rec = mechanism_recurrent(q[:,:,None,:], ..., initial_state=zeros)
        
        assert jnp.allclose(out_step, out_rec[:,:,0,:], rtol=1e-4)
        assert jnp.allclose(state_step, state_rec, rtol=1e-4)
    
    def test_outputs_are_finite(self, key):
        """Both recurrent and chunkwise should produce finite outputs."""
        ...
        out_rec, _ = mechanism_recurrent(q, k, v, ...)
        out_chunk, _ = mechanism_chunkwise(q, k, v, ...)
        
        assert jnp.all(jnp.isfinite(out_rec))
        assert jnp.all(jnp.isfinite(out_chunk))

class TestMechanismBlock:
    def test_forward_shape(self, key):
        block = MechanismBlock(hidden_size=64, num_heads=2, ..., rngs=nnx.Rngs(0))
        x = jax.random.normal(key, (batch, seq, 64))
        output, state = block(x)
        assert output.shape == x.shape
    
    def test_sequential_generation(self, key):
        """Multi-step autoregressive generation."""
        block = MechanismBlock(...)
        state = block.init_state(batch)
        
        for i in range(5):
            x = jax.random.normal(key, (batch, 1, hidden_size))
            output, state = block(x, state=state, mode="recurrent")
            assert output.shape == (batch, 1, hidden_size)
            assert state is not None

class TestModelIntegration:
    def test_model_with_mechanism(self, key):
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layers=2,
            block_pattern=["mechanism"],
            ...
        )
        model = LMModel(config, rngs=nnx.Rngs(0))
        tokens = jax.random.randint(key, (2, 16), 0, 100)
        logits, _ = model(tokens)
        assert logits.shape == (2, 16, 100)
```

### Phase 7: Integration with models.py

1. **Import in models.py**:
```python
from linearnexus.modules.linear_attn import (
    DeltaNetBlock, DeltaNetState,
    GatedDeltaNetBlock, GatedDeltaNetState,  # Add new imports
)
```

2. **Update BlockState type alias**:
```python
BlockState = Union[KVCache, MambaState, Mamba2State, DeltaNetState, GatedDeltaNetState, None]
```

3. **Add config parameters** to `ModelConfig`:
```python
# Group related params with clear naming
mechanism_heads: Optional[int] = None         # Number of Q/K heads
mechanism_v_heads: Optional[int] = None       # Number of V heads (for GVA)
mechanism_head_dim: int = 256                 # Head dimension
mechanism_expand_v: float = 2.0               # Value expansion
mechanism_use_short_conv: bool = True
mechanism_use_gate: bool = True
mechanism_special_param: float = 0.0          # Mechanism-specific
```

4. **Update `__post_init__`** for defaults:
```python
def __post_init__(self):
    ...
    if self.mechanism_heads is None:
        self.mechanism_heads = max(1, self.hidden_size // self.mechanism_head_dim)
    if self.mechanism_v_heads is None:
        self.mechanism_v_heads = self.mechanism_heads
```

5. **Register in `_create_block()`**:
```python
elif block_type == "mechanism":
    return MechanismBlock(
        hidden_size=config.hidden_size,
        num_heads=config.mechanism_heads,
        num_v_heads=config.mechanism_v_heads,
        head_dim=config.mechanism_head_dim,
        expand_v=config.mechanism_expand_v,
        use_short_conv=config.mechanism_use_short_conv,
        conv_size=config.conv_kernel,
        use_gate=config.mechanism_use_gate,
        norm_eps=config.norm_eps,
        rngs=rngs,
    )
```

6. **Update error message**:
```python
else:
    raise ValueError(
        f"Unknown block type '{block_type}'. "
        f"Supported: 'attention', 'mamba', 'mamba2', 'deltanet', 'gated_deltanet', 'mechanism'"
    )
```

7. **Add presets**:
```python
MECHANISM_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    block_pattern=["mechanism"],
    mechanism_heads=3,
    mechanism_v_heads=3,
    mechanism_head_dim=256,
    mechanism_expand_v=2.0,
    mechanism_use_gate=True,
)

# Hybrid preset
MECHANISM_HYBRID_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["mechanism", "mechanism", "mechanism", "attention"],
    mechanism_heads=3,
    mechanism_head_dim=256,
)
```

### RWKV-Style Architectures: Special Patterns

RWKV architectures (RWKV-6, RWKV-7) have unique patterns not found in typical linear attention:

#### 1. Token Shift (Temporal Mixing)
RWKV uses "token shift" instead of short convolutions - a simpler approach to temporal context:

```python
def token_shift(x: jax.Array, shift_state: Optional[jax.Array] = None):
    """Shift tokens by 1 position for temporal mixing.
    
    Args:
        x: [batch, seq, hidden]
        shift_state: Previous last token [batch, hidden] for generation
    
    Returns:
        shifted: x shifted right by 1 (first token uses shift_state or zeros)
        new_state: x[:, -1, :] to cache for next call
    """
    if shift_state is None:
        shift_state = jnp.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype)
    
    # Prepend cached state, drop last token
    shifted = jnp.concatenate([shift_state[:, None, :], x[:, :-1, :]], axis=1)
    new_state = x[:, -1, :]  # Cache last token
    return shifted, new_state
```

The pattern `delta = shifted - x` gives a "difference" signal used for data-dependent mixing.

#### 2. Lerp (Linear Interpolation) Layers
RWKV uses learnable interpolation between current and shifted tokens:

```python
class LerpLinear(nnx.Module):
    """Linear layer with input interpolation: out = Linear(lerp(x, shifted, mu))"""
    
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
        self.mu = nnx.Param(jnp.zeros(in_features))  # Learnable interpolation
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
    
    def __call__(self, x, delta):
        # delta = shifted - x
        interpolated = x + delta * self.mu.value  # Equivalent to lerp(x, shifted, mu)
        return self.linear(interpolated)
```

**Data-dependent lerp** adds a low-rank projection for input-dependent interpolation:

```python
class DDLerpLinear(nnx.Module):
    """Data-dependent lerp: mu is computed from input via low-rank projection."""
    
    def __init__(self, in_features, out_features, low_rank_dim=32, *, rngs: nnx.Rngs):
        self.mu = nnx.Param(jnp.zeros(in_features))
        self.low_rank_down = nnx.Linear(in_features, low_rank_dim, use_bias=False, rngs=rngs)
        self.low_rank_up = nnx.Linear(low_rank_dim, in_features, use_bias=False, rngs=rngs)
        self.linear = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
    
    def __call__(self, x, delta):
        # Data-dependent interpolation factor
        mu = self.mu.value + self.low_rank_up(jax.nn.tanh(self.low_rank_down(x)))
        interpolated = x + delta * mu
        return self.linear(interpolated)
```

#### 3. Layer-Dependent Initialization
RWKV uses layer index for special parameter initialization (decay patterns, etc.):

```python
def __init__(self, hidden_size, num_heads, layer_idx, n_layers, *, rngs: nnx.Rngs):
    # Decay initialization varies by layer depth
    ratio_0_to_1 = layer_idx / (n_layers - 1) if n_layers > 1 else 0.0
    ratio_1_to_almost_0 = 1.0 - (layer_idx / n_layers)
    
    # Time decay: deeper layers decay faster
    h = jnp.arange(num_heads)
    decay_speed = jnp.power(h / (num_heads - 1), 4) * (1 - ratio_1_to_almost_0)
    # ... use decay_speed for w initialization
```

**IMPORTANT**: Pass `layer_idx` through `_create_block()`:

```python
# In models.py _create_block():
elif block_type == "rwkv6":
    return RWKV6Block(
        hidden_size=config.hidden_size,
        num_heads=config.rwkv6_heads,
        layer_idx=layer_idx,    # ← CRITICAL: must pass layer index
        n_layers=config.n_layers,
        ...
        rngs=rngs,
    )
```

#### 4. RWKV Recurrence Pattern
RWKV's recurrence is simpler than DeltaNet but has a "bonus" term:

```python
def rwkv6_recurrent(r, k, v, w, u, initial_state=None):
    """RWKV-6 recurrence: o_t = (h + u*kv_t) @ r_t, h = h*exp(w_t) + kv_t
    
    Args:
        r: Receptance [batch, heads, seq, head_dim] - like "query"
        k: Key [batch, heads, seq, head_dim]
        v: Value [batch, heads, seq, head_dim]
        w: Time decay [batch, heads, seq, head_dim] (NEGATIVE values for decay)
        u: Bonus term [heads, head_dim] - adds current kv directly
    
    Key insight: w should be NEGATIVE (e.g., -exp(w_log)) so exp(w) < 1 for decay.
    """
    def step(h, inputs):
        r_t, k_t, v_t, w_t = inputs
        kv_t = jnp.einsum('bhk,bhv->bhkv', k_t, v_t)  # Outer product
        
        # Output: blend state with current kv via bonus
        o_t = jnp.einsum('bhkv,bhk->bhv', h + u[None, :, :, None] * kv_t, r_t)
        
        # State update: decay then add
        h_new = h * jnp.exp(w_t)[..., None] + kv_t
        return h_new, o_t
    
    final_state, outputs = lax.scan(step, initial_state, inputs_transposed)
    return outputs, final_state
```

#### 5. Separate Time Mixing and Channel Mixing
RWKV blocks have TWO sub-blocks (unlike single-path linear attention):

```python
class RWKV6Block(nnx.Module):
    def __init__(self, ...):
        # Time mixing (attention-like)
        self.time_mixing_norm = RMSNorm(...)
        self.receptance_proj = LerpLinear(...)  # R = "query"
        self.key_proj = LerpLinear(...)
        self.value_proj = LerpLinear(...)
        self.gate_proj = LerpLinear(...)
        self.output_proj = nnx.Linear(...)
        
        # Channel mixing (FFN-like)
        self.channel_mixing_norm = RMSNorm(...)
        self.channel_key_proj = LerpLinear(...)
        self.channel_value_proj = nnx.Linear(...)
    
    def __call__(self, x, *, state=None, ...):
        # 1. Time mixing (with token shift)
        x_shifted, new_shift_state = token_shift(x, state.shift_state if state else None)
        delta = x_shifted - x
        x = x + self._time_mixing(self.time_mixing_norm(x), delta, state)
        
        # 2. Channel mixing (also uses token shift)
        x_shifted2, _ = token_shift(x, ...)  # Or reuse shift
        delta2 = x_shifted2 - x
        x = x + self._channel_mixing(self.channel_mixing_norm(x), delta2)
        
        return x, new_state
    
    def _channel_mixing(self, x, delta):
        """Squared ReLU FFN with token shift mixing."""
        k = self.channel_key_proj(x, delta)
        k = jnp.square(jax.nn.relu(k))  # Squared ReLU activation
        return self.channel_value_proj(k)
```

#### 6. RWKV State Structure
RWKV state caches both the recurrent state AND the token shift state:

```python
@dataclass  
class RWKV6State:
    h: jax.Array              # Recurrent state [batch, heads, head_dim, head_dim]
    shift_state: jax.Array    # Last token for shifting [batch, hidden_size]
    
    @classmethod
    def zeros(cls, batch_size, num_heads, head_dim, hidden_size):
        return cls(
            h=jnp.zeros((batch_size, num_heads, head_dim, head_dim)),
            shift_state=jnp.zeros((batch_size, hidden_size)),
        )
```

#### 7. GroupNorm for Head-wise Normalization
RWKV uses GroupNorm (groups=heads) instead of LayerNorm for output normalization:

```python
class GroupNorm(nnx.Module):
    """Group normalization with learnable affine parameters."""
    
    def __init__(self, num_groups, num_channels, eps=1e-5, *, rngs: nnx.Rngs):
        self.num_groups = num_groups
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(num_channels))  # Note: 'scale' not 'weight'
        self.bias = nnx.Param(jnp.zeros(num_channels))
    
    def __call__(self, x):
        # x: [batch, seq, channels] → reshape to groups → normalize → reshape back
        ...
```

#### 8. RWKV-7 DPLR Specific Patterns

RWKV-7 adds **Diagonal Plus Low Rank (DPLR)** transitions and uses LoRA-style projections:

```python
# DPLR Recurrence: h = exp(w)*h + a*(b^T*h) + v*k^T
def dplr_delta_rule_recurrent(q, k, v, a, b, gk, initial_state=None):
    """DPLR recurrence with low-rank state update.
    
    Key difference from RWKV-6: the term a*(b^T*h) allows non-diagonal 
    state transitions, making the model more expressive.
    """
    def step(h, inputs):
        q_t, k_t, v_t, a_t, b_t, gk_t = inputs
        
        # 1. Diagonal decay
        h = h * jnp.exp(gk_t)[..., None]
        
        # 2. Low-rank update: h += a * (b^T @ h)
        b_h = jnp.einsum('bhk,bhkv->bhv', b_t, h)  # [batch, heads, v_dim]
        h = h + jnp.einsum('bhk,bhv->bhkv', a_t, b_h)
        
        # 3. Outer product update
        h = h + jnp.einsum('bhv,bhk->bhkv', v_t, k_t)
        
        # 4. Query
        o_t = jnp.einsum('bhkv,bhk->bhv', h, q_t)
        return h, o_t
    
    final_state, outputs = lax.scan(step, initial_state, inputs_transposed)
    return outputs, final_state
```

**LoRA Projections** (low-rank for parameter efficiency):
```python
class LoRA(nnx.Module):
    """Low-rank projection: bias + up(activation(down(x + delta*mu)))"""
    
    def __init__(self, input_dim, output_dim, low_rank_dim, activation="tanh"):
        self.down = nnx.Linear(input_dim, low_rank_dim, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(low_rank_dim, output_dim, use_bias=False, rngs=rngs)
        self.mu = nnx.Param(jnp.zeros(input_dim))   # Token shift interpolation
        self.bias = nnx.Param(jnp.zeros(output_dim))
        self.activation = activation
    
    def __call__(self, x, delta):
        x_mixed = x + delta * self.mu.value
        h = self.down(x_mixed)
        if self.activation == "tanh":
            h = jnp.tanh(h)
        elif self.activation == "sigmoid":
            h = jax.nn.sigmoid(h)
        return self.bias.value + self.up(h)
```

**v_first Mechanism** (cross-layer value propagation):
```python
# In LMModel.__call__:
v_first = None
for i, block in enumerate(self.blocks):
    if isinstance(block, RWKV7Block):
        x, state, v_first_out = block(x, state=state, v_first=v_first, ...)
        if v_first is None:
            v_first = v_first_out  # Capture from layer 0
    else:
        x, state = block(x, state=state, ...)

# In RWKV7Block:
def __call__(self, x, *, state=None, v_first=None, ...):
    ...
    v = self.v_proj(x_normed)
    
    # v_first propagation (after layer 0)
    v_first_out = v if self.layer_idx == 0 else None
    if v_first is not None and self.use_v_lora:
        v_lerp = self.v_lora(x_normed, delta)  # Interpolation weight
        v = v + (v_first - v) * v_lerp
    
    return output, new_state, v_first_out if self.layer_idx == 0 else v_first
```

**Gate Output Correction**:
```python
def gate_output_correction(y, v, g):
    """RWKV-7 output correction with value-based adjustment."""
    return y + (jnp.abs(v) - v) * jnp.sign(g)
```

### Quick Reference: Implementation Checklist

```
□ Phase 1: Math Understanding
  □ Read paper/README for equations
  □ Read FLA Triton kernel loop for exact recurrence
  □ Write down recurrence relation (with and without gating)
  □ Document tensor shapes at each step

□ Phase 2: Study FLA Reference
  □ Read ops/<mechanism>/fused_recurrent.py (or naive.py)
  □ Read layers/<mechanism>.py __init__ and forward
  □ List ALL parameters (projections, A_log, dt_bias, etc.)
  □ Note activation/normalization choices
  □ Identify GVA (grouped value attention) if present

□ Phase 3: Implementation (in order!)
  □ MechanismState dataclass with zeros() factory
  □ mechanism_recurrent() using lax.scan
  □ mechanism_step() extracted from recurrent
  □ mechanism_chunkwise() (optional for gated)
  □ MechanismBlock with full architecture

□ Phase 4: Integration
  □ Export in modules/linear_attn/__init__.py
  □ Import in models.py
  □ Update BlockState type alias
  □ Add config params to ModelConfig
  □ Update __post_init__ for defaults
  □ Register in _create_block()
  □ Add presets (MECHANISM_SMALL, etc.)

□ Phase 5: Testing
  □ State zeros() shape tests
  □ Kernel shape tests
  □ Step matches first recurrent step
  □ Outputs are finite (no NaN/Inf)
  □ Block forward pass
  □ Sequential generation with state
  □ Model integration (pure and hybrid)

□ Phase 6: Documentation
  □ Module docstring with math formulation
  □ Shape annotations in all function docstrings
  □ Implementation report in docs/implementations.md
```

## Documentation Resources (`docs/`)

Consult these **before** implementing features:

- **`docs/architecture_overview.md`**: System design, data flow, block protocol explanation
- **`docs/repository_overview.md`**: Complete developer guide with module responsibilities
- **`docs/adding_new_blocks.md`**: Step-by-step guide for implementing new block types
- **`docs/training_guide.md`**: Training paradigms (SFT, GRPO, PPO) and optimizer usage
- **`docs/checkpointing_guide.md`**: Saving/loading models and training state
- **`docs/concepts.md`**: Key concepts and design decisions
- **`docs/flax_nnx_glossary.md`**: Quick lookup for NNx terms
- **`docs/flax_nnx_quick_reference.md`**: Practical NNx patterns
- **`docs/why_flax_nnx.md`**: Design philosophy and comparison with Linen

## Dependencies

```
jax>=0.4.28
jaxlib>=0.4.28
flax>=0.8.3
optax>=0.2.3
numpy>=1.26.0
```

---

*Focus*: nanoGPT-style simplicity, multi-architecture support, clean functional design.
