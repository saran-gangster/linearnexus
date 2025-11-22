# LinearNexus Technical Reference

**Quick reference for developers working on LinearNexus**

---

## Quick Links

- [README](README.md) - Project overview
- [ROADMAP](ROADMAP.md) - Development timeline
- [ARCHITECTURE](ARCHITECTURE.md) - System design
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [JAX Pallas Docs](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Pallas GPU Pipeline](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)

---

## Key Concepts Cheat Sheet

### Linear Attention Complexity

**Standard Attention**: O(n²d) time, O(n²) memory
```python
scores = q @ k.T  # [n, n] matrix - quadratic!
attn = softmax(scores)
output = attn @ v
```

**Linear Attention**: O(nd²) time, O(d²) memory
```python
# Associativity: (q @ k.T) @ v == q @ (k.T @ v)
state = k.T @ v  # [d, d] matrix - constant w.r.t. sequence length!
output = q @ state
```

### Pallas Memory Hierarchy

```
GPU:
  GMEM (HBM) [80GB on A100]
    ↕ ~300 GB/s
  SMEM (Shared Memory) [128-256 KB]
    ↕ ~10 TB/s
  Registers [256 KB per SM]
    ↕ ~20 TB/s
  Compute (Tensor Cores)

TPU:
  HBM [High Bandwidth Memory]
    ↕
  VMEM (Vector Memory) [32 MB]
    ↕
  MXU (Matrix Unit) [128x128 BF16/cycle]
```

---

## Core Utilities Quick Reference

### Cache Management (`linearnexus/core/cache.py`)

```python
from linearnexus.core.cache import RecurrentState, ConvState

# Initialize recurrent state
state = RecurrentState.zeros(batch=2, channels=256, state_size=16, dtype=jnp.float32)

# Initialize conv cache for depthwise conv
conv_cache = ConvState.zeros(batch=2, kernel_size=4, channels=256, dtype=jnp.float32)

# Functional update (no mutation)
new_state = state.update(new_ssm_state)
```

### Depthwise Convolution (`linearnexus/core/conv.py`)

```python
from linearnexus.core.conv import depthwise_conv1d_causal

# Causal conv with caching for autoregressive generation
output, new_cache = depthwise_conv1d_causal(
    inputs,      # [batch, seq, channels]
    weight,      # [kernel_size, channels]
    bias,        # [channels] or None
    cache=cache  # [batch, kernel_size-1, channels] or None
)
# Output: [batch, seq, channels], new_cache: [batch, kernel_size-1, channels]
```

### Padding for Variable-Length Sequences (`linearnexus/core/padding.py`)

```python
from linearnexus.core.padding import compute_unpadded_indices, unpad, pad

# Given a binary mask [batch, seq] (1=valid, 0=padding)
indices, cu_seqlens = compute_unpadded_indices(mask)

# Flatten to packed representation
x_packed = unpad(x, indices)  # [total_valid_tokens, hidden]

# Restore to padded shape after processing
x_padded = pad(x_packed, indices, batch_size, seq_len)
```

### Mode Selection (`linearnexus/core/mode.py`)

```python
from linearnexus.core.mode import select_mode
from linearnexus.kernels.base import KernelMode

# Automatically choose chunk vs recurrent based on sequence length
mode = select_mode(seq_len, threshold=64)
# Returns KernelMode.RECURRENT if seq_len <= 64, else KernelMode.CHUNK
```

### Feature Registry (`linearnexus/registry.py`)

```python
from linearnexus.registry import KERNEL_REGISTRY, LAYER_REGISTRY

# Look up kernel by name
kernel_cls = KERNEL_REGISTRY["mamba:reference"]

# Look up layer and config
layer_cls, config_cls = LAYER_REGISTRY["mamba"]

# Iterate all registered features for testing
for name, (layer_cls, config_cls) in LAYER_REGISTRY.items():
    print(f"Testing {name}...")
    run_parity_tests(layer_cls, config_cls)
```

---

## Attention Mechanisms Overview

### RetNet (Multi-Scale Retention)

**Key Idea**: Exponential decay retention mechanism

```python
# Decay factor
γ = exp(-log(32) / head_id)  # Different decay per head

# Chunk processing
retention = softmax(q @ k.T) * decay_mask
output = retention @ v
```

**Modes**:
- `parallel`: Full retention matrix (training)
- `chunk`: Block-wise processing (training)
- `recurrent`: State-based (inference)

**Paper**: [Retentive Network: A Successor to Transformer](https://arxiv.org/abs/2307.08621)

### GLA (Gated Linear Attention)

**Key Idea**: Data-dependent gating for information flow

```python
# Gate computation (log-space for numerical stability)
g = -softplus(gate_proj(x))  # Negative log-space

# Cumulative sum
g_cumsum = cumsum(g, axis=1)

# Gated attention
state = exp(g_cumsum[:, :, None]) * (k.T @ v)
output = q @ state
```

**Modes**:
- `chunk`: Block-wise with inter-chunk recurrence
- `fused_chunk`: Memory-optimized chunk
- `fused_recurrent`: Fast inference

**Paper**: [Gated Linear Attention Transformers](https://arxiv.org/abs/2312.06635)

### DeltaNet

**Key Idea**: Delta rule for improved state updates

```python
# Delta rule update
numerator = beta * v - (beta * q) @ state
denominator = 1 + beta * (k @ q.T)
delta = numerator / denominator

# State update
state = state + k.T @ delta
output = q @ state
```

**Features**:
- Beta parameter (learnable or fixed)
- QK normalization (L2, LayerNorm, etc.)
- Activation variants (SiLU, ReLU, ELU)

**Paper**: [Parallelizing Linear Transformers with Delta Rule](https://arxiv.org/abs/2406.06484)

---

## Pallas Programming Model

### Basic Kernel Structure

```python
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

def my_kernel(x_ref, y_ref, o_ref):
    """
    Args:
        *_ref: Reference types (Ref) for input/output
        Read: x_ref[:] or x_ref[indices]
        Write: o_ref[:] = value
    """
    # Read from SMEM/registers
    x = x_ref[:]
    y = y_ref[:]
    
    # Compute
    result = x + y
    
    # Write to output
    o_ref[:] = result

# Execute kernel
def add_arrays(x, y):
    return pl.pallas_call(
        my_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((block_size,), lambda i: i),
            pl.BlockSpec((block_size,), lambda i: i),
        ],
        out_specs=pl.BlockSpec((block_size,), lambda i: i),
        grid=(num_blocks,),
    )(x, y)
```

### BlockSpec: Memory Transforms

```python
# Identity transform
pl.BlockSpec(
    block_shape=(128, 64),  # Shape of each block
    index_map=lambda i, j: (i, j),  # Grid indices → Block indices
)

# Tiling transform (GPU-specific)
plgpu.BlockSpec(
    block_shape=(128, 64),
    index_map=lambda i, j: (i, j),
    transforms=(
        plgpu.TilingTransform((8, 8)),  # Tile for tensor cores
        plgpu.SwizzleTransform(128),     # Avoid bank conflicts
    )
)
```

### Grid Execution

```python
# 1D grid
grid=(num_blocks,)
# program_id(0) ∈ [0, num_blocks)

# 2D grid
grid=(blocks_m, blocks_n)
# program_id(0) ∈ [0, blocks_m), program_id(1) ∈ [0, blocks_n)

# 3D grid
grid=(blocks_batch, blocks_m, blocks_n)
# Similar for 3 dimensions
```

---

## Performance Optimization Patterns

### 1. Software Pipelining

**Goal**: Overlap memory transfers with computation

```python
pipeline = plgpu.emit_pipeline(
    pipeline_body,
    in_specs=[input_specs],
    grid=(num_chunks,),
    max_concurrent_steps=2,  # Number of stages in flight
    delay_release=1,         # Keep buffers longer for async ops
)
```

**Pattern**:
```
Iteration 0: Load chunk 0 → SMEM
Iteration 1: Load chunk 1 | Compute chunk 0
Iteration 2: Load chunk 2 | Compute chunk 1 | Store chunk 0
...
```

### 2. Warp Specialization (Hopper GPUs)

**Goal**: Separate memory and compute warpgroups

```python
plgpu.emit_pipeline_warp_specialized(
    kernel_body,
    in_specs=input_specs,
    grid=(num_chunks,),
    num_compute_wgs=2,       # Number of compute warpgroups
    memory_registers=40,     # Registers for memory warpgroup
    wg_axis="wg",           # Warpgroup axis name
    compute_context=compute_fn,
)
```

**Benefits**:
- Better scheduling flexibility
- Higher tensor core utilization
- 1.3-1.5x speedup on H100

### 3. Kernel Fusion

**Pattern**: Fuse operations to reduce memory traffic

```python
# Unfused: 3 memory round-trips
q = x @ w_q
q = layernorm(q)
q = silu(q)

# Fused: 1 memory round-trip
def fused_qproj_kernel(x_ref, w_q_ref, q_ref):
    x = x_ref[:]
    q = x @ w_q_ref[:]
    q = inline_layernorm(q)  # Compute without writing to memory
    q = inline_silu(q)
    q_ref[:] = q
```

**Common Fusions**:
- Linear + LayerNorm + Activation
- Softmax + Masking
- Linear + CrossEntropy (for LM head)
- QKV projection together

### 4. Tiling Strategy

**GPU (Mosaic)**:
```python
# Typical tile sizes
TILE_M = 128  # Along sequence dimension
TILE_N = 128  # Along head dimension
TILE_K = 64   # Along contraction dimension

# Constrained by SMEM (128-256 KB)
smem_usage = (
    TILE_M * TILE_K +  # Q tile
    TILE_K * TILE_N +  # K tile  
    TILE_M * TILE_N    # V tile & output
) * sizeof(dtype)
```

**TPU (Mosaic)**:
```python
# Larger tiles for TPU
TILE_M = 512
TILE_N = 512
TILE_K = 128

# Constrained by VMEM (32 MB)
# TPU benefits from larger tiles
```

---

## Common Kernel Patterns

### Pattern 1: Tiled Matrix Multiplication

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles
        a = tl.load(a_ptr + offsets_a)
        b = tl.load(b_ptr + offsets_b)
        
        # Compute partial product
        acc += tl.dot(a, b)
    
    # Store result
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offsets_c, c)
```

### Pattern 2: Fused Softmax

```python
def fused_softmax_kernel(x_ref, o_ref):
    """Numerically stable softmax."""
    x = x_ref[:]
    
    # Find max (for numerical stability)
    x_max = jnp.max(x, axis=-1, keepdims=True)
    
    # Subtract max and exponentiate
    x_exp = jnp.exp(x - x_max)
    
    # Normalize
    x_sum = jnp.sum(x_exp, axis=-1, keepdims=True)
    o_ref[:] = x_exp / x_sum
```

### Pattern 3: Chunk Attention

```python
def chunk_attention_kernel(
    q_ref, k_ref, v_ref, h_ref, o_ref,
    chunk_size: int
):
    chunk_id = pl.program_id(0)
    
    # Load current chunk
    start = chunk_id * chunk_size
    end = start + chunk_size
    
    q_chunk = q_ref[start:end]
    k_chunk = k_ref[start:end]
    v_chunk = v_ref[start:end]
    
    # Intra-chunk attention (causal)
    scores = q_chunk @ k_chunk.T
    mask = jnp.tril(jnp.ones((chunk_size, chunk_size)))
    scores = jnp.where(mask, scores, float('-inf'))
    attn = jax.nn.softmax(scores)
    intra = attn @ v_chunk
    
    # Inter-chunk contribution from previous state
    if chunk_id > 0:
        h_prev = h_ref[chunk_id - 1]
        inter = q_chunk @ h_prev
    else:
        inter = 0
    
    # Update state
    h_curr = update_state(h_prev if chunk_id > 0 else None, k_chunk, v_chunk)
    h_ref[chunk_id] = h_curr
    
    # Combine contributions
    o_ref[start:end] = intra + inter
```

---

## Testing Patterns

### Shared Test Helpers (`tests/helpers/parity.py`)

```python
from tests.helpers.parity import assert_chunk_recurrent_parity, assert_mask_behavior

def test_mamba_parity():
    """Test chunk vs recurrent mode alignment."""
    layer = MambaLayer(rngs, config)
    inputs = jax.random.normal(key, (2, 16, 256))
    
    # Shared helper handles both modes and comparison
    assert_chunk_recurrent_parity(layer, inputs, rtol=1e-4, atol=1e-4)

def test_mamba_masking():
    """Test that masked tokens are zeroed out."""
    layer = MambaLayer(rngs, config)
    inputs = jax.random.normal(key, (2, 16, 256))
    mask = jnp.array([[1, 1, 1, 0, 0, ...], [1, 1, 1, 1, 0, ...]])
    
    # Shared helper validates mask behavior
    assert_mask_behavior(layer, inputs, mask)
```

### Registry-Driven Testing

```python
import pytest
from linearnexus.registry import LAYER_REGISTRY
from tests.helpers.parity import assert_chunk_recurrent_parity

@pytest.mark.parametrize("feature_name", LAYER_REGISTRY.keys())
def test_all_layers_parity(feature_name):
    """Automatically test parity for all registered layers."""
    layer_cls, config_cls = LAYER_REGISTRY[feature_name]
    
    config = config_cls(hidden_size=64, ...)
    layer = layer_cls(nnx.Rngs(0), config)
    inputs = jax.random.normal(jax.random.PRNGKey(42), (2, 16, 64))
    
    assert_chunk_recurrent_parity(layer, inputs, rtol=1e-4, atol=1e-4)
```

### Kernel Correctness Tests

```python
import pytest
import jax
import jax.numpy as jnp
import numpy as np

class TestMambaKernel:
    """Test suite for Mamba kernel."""
    
    @pytest.fixture
    def setup(self):
        """Setup test data."""
        batch_size = 2
        seq_len = 16
        intermediate = 32
        state_size = 8
        
        key = jax.random.PRNGKey(0)
        hidden = jax.random.normal(key, (batch_size, intermediate, seq_len))
        delta = jax.random.normal(key, (batch_size, intermediate, seq_len))
        B = jax.random.normal(key, (batch_size, seq_len, state_size))
        C = jax.random.normal(key, (batch_size, seq_len, state_size))
        gate = jax.random.normal(key, (batch_size, intermediate, seq_len))
        
        return {'hidden': hidden, 'delta': delta, 'B': B, 'C': C, 'gate': gate}
    
    def test_pallas_vs_reference(self, setup):
        """Test Pallas kernel against pure JAX reference."""
        from linearnexus.kernels.mamba_reference import MambaReferenceKernel
        from linearnexus.kernels.mamba_pallas import MambaPallasKernel  # future
        
        ref_kernel = MambaReferenceKernel()
        pallas_kernel = MambaPallasKernel()
        
        # Run both kernels
        ref_output, _ = ref_kernel.forward_chunk(params, inputs, state, chunk_size=8)
        pallas_output, _ = pallas_kernel.forward_chunk(params, inputs, state, chunk_size=8)
        
        # Compare
        np.testing.assert_allclose(
            pallas_output, ref_output,
            rtol=1e-4, atol=1e-4
        )
    
    def test_gradient(self, setup):
        """Test gradient correctness."""
        # Finite differences
        def fn(params):
            return retnet_kernel(**params).sum()
        
        grad_kernel = jax.grad(fn)(setup)
        grad_numerical = numerical_gradient(fn, setup)
        
        for key in grad_kernel:
            np.testing.assert_allclose(
                grad_kernel[key], grad_numerical[key],
                rtol=1e-3, atol=1e-3
            )
    
    @pytest.mark.benchmark
    def test_performance(self, setup, benchmark):
        """Benchmark kernel performance."""
        # Warm up
        for _ in range(10):
            _ = retnet_kernel(**setup)
        
        # Benchmark
        result = benchmark(retnet_kernel, **setup)
        
        # Check performance targets
        assert result.mean < TARGET_LATENCY_MS
```

### Benchmark Template

```python
import time
import jax

def benchmark_kernel(
    kernel_fn,
    inputs,
    num_warmup=10,
    num_iterations=100
):
    """Benchmark a kernel."""
    # Warm up
    for _ in range(num_warmup):
        _ = kernel_fn(*inputs)
    
    # Synchronize (for GPU)
    jax.block_until_ready(_)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = kernel_fn(*inputs)
    jax.block_until_ready(output)
    end = time.perf_counter()
    
    # Compute metrics
    mean_time = (end - start) / num_iterations
    throughput = compute_throughput(inputs, mean_time)
    
    return {
        'mean_time_ms': mean_time * 1000,
        'throughput': throughput,
    }
```

---

## Debugging Tips

### 1. Use Emulation Mode

```python
# Run kernel on CPU for debugging
with jax.experimental.enable_x64():
    output = kernel_fn(*inputs)  # Runs on CPU/XLA
```

### 2. Print Inside Kernels

```python
def debug_kernel(x_ref, o_ref):
    x = x_ref[:]
    
    # Debug print (works in emulation mode)
    jax.debug.print("x shape: {}, x[0]: {}", x.shape, x[0])
    
    o_ref[:] = x * 2
```

### 3. Validate Shapes

```python
def validate_shapes(q, k, v):
    """Validate input shapes before kernel."""
    assert q.shape == k.shape, "Q and K must have same shape"
    assert q.shape[-1] == v.shape[-1], "Q and V must have same head_dim"
    assert q.ndim == 4, "Expected 4D input [batch, seq, heads, dim]"
```

### 4. Check Numerical Stability

```python
def check_numerical_stability(output):
    """Check for NaN/Inf in output."""
    has_nan = jnp.any(jnp.isnan(output))
    has_inf = jnp.any(jnp.isinf(output))
    
    if has_nan or has_inf:
        raise ValueError(f"Numerical instability: NaN={has_nan}, Inf={has_inf}")
```

---

## Useful JAX Snippets

### Automatic Batching (vmap)

```python
# Function for single input
def attention_single(q, k, v):
    return attention_kernel(q, k, v)

# Automatically batch over first dimension
attention_batched = jax.vmap(attention_single, in_axes=0, out_axes=0)

# Usage
q_batch = jnp.array([...])  # [batch, seq, ...]
output = attention_batched(q_batch, k_batch, v_batch)
```

### Gradient Checkpointing

```python
from jax import checkpoint

@checkpoint
def expensive_layer(x, params):
    """Recompute in backward pass instead of storing."""
    return compute_expensive_operation(x, params)
```

### Custom Gradient

```python
from jax import custom_vjp

@custom_vjp
def custom_attention(q, k, v):
    """Attention with custom backward pass."""
    return attention_forward(q, k, v)

def attention_fwd(q, k, v):
    output = attention_forward(q, k, v)
    residuals = (q, k, v)  # Save for backward
    return output, residuals

def attention_bwd(residuals, g):
    q, k, v = residuals
    # Custom gradient computation
    dq, dk, dv = custom_attention_backward(g, q, k, v)
    return dq, dk, dv

custom_attention.defvjp(attention_fwd, attention_bwd)
```

---

## Avoiding Common Pitfalls (Learned from FLA)

### Pitfall 1: Duplicating Cross-Cutting Logic

**Problem**: Copying padding, gating, and caching code into every layer.

**Solution**: Use `linearnexus/core/` utilities:
```python
# Bad: Reimplementing conv in each layer
class MyLayer(nnx.Module):
    def __call__(self, x):
        # Custom conv implementation...
        pass

# Good: Reuse core utility
from linearnexus.core.conv import depthwise_conv1d_causal

class MyLayer(nnx.Module):
    def __call__(self, x, *, state=None):
        conv_out, new_cache = depthwise_conv1d_causal(
            x, self.conv_weight.value, self.conv_bias.value, cache=state.conv_buffer
        )
```

### Pitfall 2: Manual Test Maintenance

**Problem**: Writing separate parity/gradient tests for each feature.

**Solution**: Use registry + shared helpers:
```python
# Automatically test all registered layers
@pytest.mark.parametrize("feature_name", LAYER_REGISTRY.keys())
def test_parity(feature_name):
    layer_cls, config_cls = LAYER_REGISTRY[feature_name]
    assert_chunk_recurrent_parity(layer_cls(rngs, config_cls()), inputs)
```

### Pitfall 3: Config Duplication and Drift

**Problem**: Each model config redefines the same fields with different defaults.

**Solution**: Use `ConfigBase` for shared fields:
```python
from linearnexus.core.config import ConfigBase

@dataclass
class MambaConfig(ConfigBase):
    # Inherited: chunk_size, dtype, init_scale
    # Mamba-specific:
    state_size: int = 16
    conv_kernel: int = 4
```

### Pitfall 4: Implicit Feature Discovery

**Problem**: Tests and benchmarks hardcode imports; adding a feature requires updating many files.

**Solution**: Register features explicitly:
```python
# registry.py
LAYER_REGISTRY = {
    "mamba": (MambaLayer, MambaConfig),
    "gla": (GLALayer, GLAConfig),  # New feature auto-tested
}
```

## Common Issues & Solutions

### Issue 1: Out of Memory

**Symptom**: OOM during training
**Solutions**:
- Reduce batch size or sequence length
- Enable gradient checkpointing
- Use smaller tile sizes in kernels
- Enable mixed precision (bfloat16)

### Issue 2: Numerical Instability

**Symptom**: NaN/Inf in outputs or gradients
**Solutions**:
- Use log-space for exponentials
- Subtract max before softmax
- Clip gradients
- Use float32 for accumulation (even if bfloat16 inputs)

### Issue 3: Slow Performance

**Symptom**: Lower than expected throughput
**Solutions**:
- Profile with JAX profiler
- Check memory bandwidth utilization
- Increase tile sizes (if memory allows)
- Enable kernel fusion
- Use software pipelining

### Issue 4: Incorrect Gradients

**Symptom**: Gradients don't match finite differences
**Solutions**:
- Test with smaller inputs first
- Use `jax.grad` instead of custom VJP initially
- Check for in-place operations
- Validate against reference implementation

---

## Profiling Commands

```bash
# JAX profiling with Pallas GPU kernels
JAX_PLATFORMS=gpu python -m jax.profiler trace \
    --output_dir=/tmp/jax-profile \
    examples/run_mamba_reference.py

# JAX profiling with Pallas TPU kernels
JAX_PLATFORMS=tpu python -m jax.profiler trace \
    --output_dir=/tmp/jax-profile \
    examples/run_mamba_reference.py

# View in TensorBoard
tensorboard --logdir=/tmp/jax-profile

# Profile specific layer
python -c "
from linearnexus.registry import LAYER_REGISTRY
from linearnexus.tools.profile import profile_layer

layer_cls, config_cls = LAYER_REGISTRY['mamba']
profile_layer(layer_cls, config_cls(), batch=2, seq=128)
"

# NVIDIA Nsight Compute (for GPU kernels)
ncu --set full \
    --export profile.ncu-rep \
    python train_script.py

# Memory profiling
python -m memory_profiler train_script.py
```

---

## Resources

### Papers
- RetNet: https://arxiv.org/abs/2307.08621
- GLA: https://arxiv.org/abs/2312.06635
- DeltaNet: https://arxiv.org/abs/2406.06484
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Flash Attention: https://arxiv.org/abs/2205.14135

### Documentation
- JAX: https://jax.readthedocs.io/
- Pallas: https://docs.jax.dev/en/latest/pallas/index.html
- Flax: https://flax.readthedocs.io/
- Optax: https://optax.readthedocs.io/

### Repositories
- flash-linear-attention: https://github.com/fla-org/flash-linear-attention
- JAX: https://github.com/google/jax
- Triton: https://github.com/openai/triton

---

**Last Updated**: November 17, 2024
