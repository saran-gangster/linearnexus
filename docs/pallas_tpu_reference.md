# Pallas TPU Kernel Programming Reference

**Comprehensive guide to writing custom TPU kernels with JAX Pallas targeting TPU v5e and later.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [TPU Architecture Fundamentals](#tpu-architecture-fundamentals)
3. [Memory Hierarchy & VMEM](#memory-hierarchy--vmem)
4. [Writing TPU Kernels](#writing-tpu-kernels)
5. [Grid and Indexing Patterns](#grid-and-indexing-patterns)
6. [TPU-Specific Primitives](#tpu-specific-primitives)
7. [FlashAttention on TPU](#flashattention-on-tpu)
8. [Splash Attention (Sparse)](#splash-attention-sparse)
9. [Common Patterns](#common-patterns)
10. [Performance Optimization](#performance-optimization)
11. [Debugging](#debugging)

---

## Introduction

### What Makes TPU Different?

TPUs use a fundamentally different architecture than GPUs:

**GPU (CUDA/Mosaic)**:
- Warp-based execution (32/64 threads)
- Explicit shared memory (SMEM)
- TensorCores for matrix operations
- NVLINK for multi-GPU communication

**TPU (Pallas TPU)**:
- **Vector architecture**: 128 lanes per core
- **VMEM (Vector Memory)**: Scratchpad for intermediate results
- **MXU (Matrix Unit)**: Systolic array for matmul
- **ICI (Inter-Chip Interconnect)**: High-bandwidth all-to-all communication

### Key Constants

```python
NUM_LANES = 128        # Vector width
NUM_SUBLANES = 8       # Sub-vector granularity
MIN_BLOCK_SIZE = 128   # Minimum efficient tile size
```

> **Runtime requirement:** Pallas TPU kernels only lower when `libtpu` is up to date. Keep
> the TPU runtime within ~30 days of the current release (upgrade your VM image or
> install the latest `libtpu` wheel) or JAX will raise `RuntimeError: Pallas TPU requires
> a libtpu version that's at most a month old` during lowering.

**Critical**: All block sizes must be multiples of `NUM_LANES` (128) for efficient vectorization.

---

## TPU Architecture Fundamentals

### Hardware Layout

```
┌─────────────────────────────────────┐
│ TensorCore (per core)               │
│  - Vector Unit (128 lanes)          │  ← Processes 128 elements/cycle
│  - MXU (Matrix Unit)                │  ← Systolic array for matmul
├─────────────────────────────────────┤
│ VMEM (Vector Memory, ~16MB)         │  ← pltpu.VMEM
├─────────────────────────────────────┤
│ HBM (High-Bandwidth Memory, ~16GB)  │  ← Global memory
└─────────────────────────────────────┘
```

**Multi-core organization** (e.g., TPU v5e has 1-2 cores per chip, 4 chips per host):
- Each core executes independently
- ICI connects cores within and across chips
- `dimension_semantics` controls parallelism vs sequential execution

### Memory Bandwidth

| Memory Type | Bandwidth | Latency | Use Case |
|-------------|-----------|---------|----------|
| VMEM | ~2 TB/s | ~10 cycles | Accumulation, tiling |
| HBM (on-chip) | ~900 GB/s | ~100 cycles | Main data storage |
| ICI (inter-chip) | ~100 GB/s/link | ~1μs | Multi-core collectives |

---

## Memory Hierarchy & VMEM

### VMEM: TPU's Scratchpad

Unlike GPU's SMEM (explicitly allocated per block), TPU uses **VMEM** for intermediate values:

```python
# Allocate VMEM scratch (similar to GPU SMEM, but per-core)
scratch_shapes = [
    pltpu.VMEM((block_q, head_dim), jnp.float32),  # Accumulator
    pltpu.VMEM((block_q, MIN_BLOCK_SIZE), jnp.float32),  # Softmax stats
]

def kernel(..., acc_scratch_ref, stats_scratch_ref):
    # Write to VMEM (not visible to other cores)
    acc_scratch_ref[...] = jnp.zeros_like(acc_scratch_ref)
    
    # Accumulate in VMEM
    acc_scratch_ref[:] += some_computation()
    
    # Read from VMEM
    result = acc_scratch_ref[...]
```

**VMEM vs SMEM**:
- **VMEM**: Per-core, auto-managed, no barriers needed within single core
- **SMEM**: Per-block (GPU), requires explicit barriers for synchronization
- **VMEM size**: ~16MB (much larger than GPU SMEM's ~100KB)

### Data Layout Constraints

TPU vector operations require specific layouts:

```python
# Good: Minormost dimension = NUM_LANES (128)
data = jnp.ones((batch, seq_len, 128))  # ✓ Vectorizable

# Bad: Minormost dimension not aligned
data = jnp.ones((batch, seq_len, 127))  # ✗ Inefficient

# Broadcasting for vectorization
q_segment_ids = jax.lax.broadcast_in_dim(
    segment_ids.q,              # [batch, seq_len]
    (batch, seq_len, NUM_LANES),  # Broadcast to lanes
    (0, 1)
)
```

**`pltpu.repeat`**: Efficient lane replication (compiled to hardware repeat, not copy):

```python
# Repeat value across lanes for vectorized comparison
q_ids = pltpu.repeat(
    segment_ids_ref[:],  # [block_q, NUM_LANES]
    repeats=block_k // NUM_LANES,  # Repeat N times
    axis=1  # Along lane dimension
)  # Result: [block_q, block_k]
```

---

## Writing TPU Kernels

### Basic Kernel Structure

```python
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def add_kernel(x_ref, y_ref, o_ref):
    """Element-wise addition on TPU."""
    # BlockSpec determines how grid indices map to data slices
    batch_idx = pl.program_id(0)
    seq_idx = pl.program_id(1)
    
    # Load tiles (automatically from HBM to VMEM)
    x = x_ref[batch_idx, seq_idx, :]
    y = y_ref[batch_idx, seq_idx, :]
    
    # Compute (in VMEM)
    o_ref[batch_idx, seq_idx, :] = x + y

# Invoke kernel
def add_vectors(x, y):
    batch_size, seq_len, hidden_dim = x.shape
    block_size = 128
    
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch_size, seq_len // block_size),
        in_specs=[
            pl.BlockSpec((1, block_size, hidden_dim), lambda b, s: (b, s, 0)),
            pl.BlockSpec((1, block_size, hidden_dim), lambda b, s: (b, s, 0))
        ],
        out_specs=pl.BlockSpec((1, block_size, hidden_dim), lambda b, s: (b, s, 0))
    )(x, y)
```

### PrefetchScalarGridSpec (TPU-specific)

TPU kernels use `PrefetchScalarGridSpec` for efficient data movement:

```python
pl.pallas_call(
    kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,  # Number of scalar values to prefetch (metadata)
        grid=(num_heads, q_blocks, kv_blocks),
        in_specs=[...],
        out_specs=[...],
        scratch_shapes=[...]  # VMEM allocations
    ),
    compiler_params=pltpu.CompilerParams(
        # Control parallelism:
        # "parallel": distribute across cores (data parallelism)
        # "arbitrary": sequential execution (reduction, stateful loops)
        dimension_semantics=("parallel", "arbitrary", "arbitrary")
    ),
    out_shape=...
)(inputs...)
```

**`num_scalar_prefetch`**: Prefetches small metadata arrays (masks, indices) before kernel launch, hiding latency.

---

## Grid and Indexing Patterns

### Multi-dimensional Grids

```python
grid = (batch_size, num_heads, q_seq_len // block_q, kv_seq_len // block_kv)

def index_map(batch_idx, head_idx, q_idx, kv_idx):
    # Map grid coordinates to data coordinates
    return (batch_idx, head_idx, q_idx, 0)

pl.BlockSpec((1, 1, block_q, head_dim), index_map)
```

**Grid iteration**:
```python
def kernel(...):
    batch_idx = pl.program_id(0)  # Current batch
    head_idx = pl.program_id(1)   # Current head
    q_idx = pl.program_id(2)      # Current Q block
    kv_idx = pl.program_id(3)     # Current KV block
```

### Conditional Execution with `@pl.when`

```python
@pl.when(kv_idx == 0)
def initialize():
    """Runs only for first KV block per Q block."""
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

@pl.when(kv_idx == (kv_seq_len // block_kv) - 1)
def finalize():
    """Runs only for last KV block per Q block."""
    o_ref[...] = (acc_scratch_ref[...] / l_scratch_ref[...]).astype(o_ref.dtype)
```

### Causal Masking Grid Optimization

```python
def below_or_on_diag(q_idx, q_block_size, kv_idx, kv_block_size):
    """Check if KV block is on or below diagonal (causal attention)."""
    # Bottom-left corner of Q block vs top-left corner of KV block
    return ((q_idx + 1) * q_block_size - 1) >= (kv_idx * kv_block_size)

if causal:
    should_run = below_or_on_diag(q_idx, block_q, kv_idx, block_kv)
else:
    should_run = True

@pl.when(should_run)
def compute_block():
    # Only process blocks that aren't masked out entirely
    ...
```

---

## TPU-Specific Primitives

### Matmul: `lax.dot_general`

TPU's MXU (Matrix Unit) is a systolic array optimized for matrix multiplication:

```python
# Standard matmul: C = A @ B
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # B is transposed
qk = lax.dot_general(
    q,  # [block_q, head_dim]
    k,  # [block_kv, head_dim]
    TRANS_B_DIM_NUMBERS,
    preferred_element_type=jnp.float32  # Accumulate in FP32
)  # Result: [block_q, block_kv]

# No transpose (for V matmul)
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
o = lax.dot_general(
    p,  # [block_q, block_kv]
    v,  # [block_kv, head_dim]
    NN_DIM_NUMBERS,
    preferred_element_type=jnp.float32
)  # Result: [block_q, head_dim]
```

**Performance tips**:
- Use `preferred_element_type=jnp.float32` for accuracy (BF16/FP16 inputs → FP32 accumulator)
- Tile sizes should be multiples of 128 for optimal MXU utilization
- Non-transposed access is faster than transposed (memory layout matters)

### Broadcasting and Masking

```python
# Causal mask generation (efficient on TPU vector units)
mask_shape = (block_q, block_kv)
row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)  # [0,1,2,...] per row
row_ids += q_idx * block_q  # Global row offset

col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)  # [0,1,2,...] per col
col_ids += kv_idx * block_kv  # Global col offset

causal_mask = col_ids <= row_ids  # [block_q, block_kv] bool

# Apply mask (use jnp.where, not direct indexing)
logits = jnp.where(causal_mask, logits, mask_value)
```

### Segment IDs (Masking Across Sequences)

```python
# Prevent attention between different segments in packed sequences
q_segment_ids = segment_ids.q[:, :, None]  # [batch, q_seq, 1]
kv_segment_ids = segment_ids.kv[:, None, :]  # [batch, 1, kv_seq]

segment_mask = (q_segment_ids == kv_segment_ids)  # [batch, q_seq, kv_seq]

# Combine with causal mask
final_mask = jnp.logical_and(causal_mask, segment_mask)
logits = jnp.where(final_mask, logits, mask_value)
```

---

## FlashAttention on TPU

### Architecture Overview

TPU FlashAttention differs from GPU version due to vector architecture:

```
Grid: (batch, num_heads, q_blocks, kv_blocks)
Memory: VMEM scratch for (m_i, l_i, acc) per Q block
Execution: Sequential over KV blocks (online softmax)
```

**Key differences from GPU**:
- No warp specialization (single-threaded per grid point)
- VMEM replaces SMEM (larger, no barriers within core)
- MXU instead of TensorCores (systolic array)

### Forward Pass: Online Softmax

```python
def flash_attention_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,  # Optional attention bias
    segment_ids_q_ref,
    segment_ids_kv_ref,
    o_tile_ref,
    l_ref,  # Log-sum-exp numerator (optional)
    m_ref,  # Log-sum-exp max (optional)
    m_scratch_ref,  # VMEM: running max
    l_scratch_ref,  # VMEM: running sum
    acc_scratch_ref,  # VMEM: running output accumulator
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
    kv_seq_idx = pl.program_id(3)
    
    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        """Initialize accumulators for first KV block."""
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        acc_scratch_ref[...] = jnp.zeros_like(acc_scratch_ref)
    
    # Load Q (constant across KV loop)
    q = q_tile_ref[batch_idx]  # [block_q, head_dim]
    
    # Online softmax loop over KV tiles
    @pl.loop(0, block_k_major, step=block_k, unroll=True)
    def kv_loop(start_k):
        m_prev = m_scratch_ref[batch_idx]  # [block_q, NUM_LANES]
        l_prev = l_scratch_ref[batch_idx]
        
        # QK matmul
        k = k_tile_ref[batch_idx, pl.dslice(start_k, block_k), :]
        s = lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
        
        if sm_scale != 1.0:
            s *= sm_scale
        
        # Apply masks (causal, segment, attention bias)
        s = apply_masks(s, ...)
        
        # Online softmax update
        m_curr = jnp.max(s, axis=1)[:, None]  # [block_q, 1]
        m_next = jnp.maximum(m_prev, m_curr)  # [block_q, NUM_LANES]
        
        # Expand to block_k lanes using pltpu.repeat
        block_k_repeats = block_k // MIN_BLOCK_SIZE
        p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, axis=1))
        
        # Rescale previous accumulator
        alpha = jnp.exp(m_prev - m_next)
        l_corr = alpha * l_prev
        l_next = jnp.sum(p, axis=1)[:, None] + l_corr
        
        # Update running statistics
        m_scratch_ref[batch_idx] = m_next
        l_scratch_ref[batch_idx] = l_next
        
        # Update output accumulator
        head_dim_repeats = head_dim // MIN_BLOCK_SIZE
        l_next_inv = 1.0 / l_next
        acc_scratch_ref[batch_idx] *= pltpu.repeat(l_corr * l_next_inv, head_dim_repeats, 1)
        
        # PV matmul
        v = v_tile_ref[batch_idx, pl.dslice(start_k, block_k), :]
        o_curr = lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
        acc_scratch_ref[batch_idx] += o_curr * pltpu.repeat(l_next_inv, head_dim_repeats, 1)
    
    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        """Finalize and write output."""
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)
```

**Critical optimizations**:
1. **`pltpu.repeat`**: Broadcasts scalars to vectors without copy (hardware instruction)
2. **VMEM accumulators**: Avoid repeated HBM reads/writes
3. **`unroll=True`**: Loop unrolling for inner tile loops improves pipelining
4. **`pl.when` guards**: Conditional execution without divergence overhead

### Backward Pass: Dual Kernels

Like GPU FlashAttention, TPU uses two kernels:

**Kernel 1: Compute dK, dV** (iterate over Q blocks):

```python
def flash_attention_dkv_kernel(...):
    q_seq_index = pl.program_id(3)  # Grid dimension is Q
    kv_seq_index = pl.program_id(2)  # Fixed KV block
    
    @pl.when(q_seq_index == 0)
    def init_grads():
        dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
        dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)
    
    # Recompute forward pass quantities
    q = q_tile_ref[...]
    k = k_tile_ref[...]
    v = v_tile_ref[...]
    do = do_tile_ref[...]  # Gradient from output
    l = l_tile_ref[...]    # Saved from forward
    m = m_tile_ref[...]    # Saved from forward
    di = di_tile_ref[...]  # Precomputed: sum(do * o, axis=-1)
    
    # Recompute attention weights
    logits = lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, ...)
    logits = apply_masks(logits, ...)
    p = jnp.exp(logits - m) / l  # [block_q, block_kv]
    
    # dV = P.T @ dO
    dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv_scratch_ref[...] += dv.astype(dv_scratch_ref.dtype)
    
    # Softmax gradient: dS = P * (dP - di)
    dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, ...)
    ds = (dp - pltpu.repeat(di, block_kv // MIN_BLOCK_SIZE, axis=1)) * p
    
    if sm_scale != 1.0:
        ds *= sm_scale
    
    # dK = dS.T @ Q
    dk = lax.dot(ds.T.astype(q.dtype), q, preferred_element_type=jnp.float32)
    dk_scratch_ref[...] += dk.astype(dk_scratch_ref.dtype)
    
    @pl.when(q_seq_index == q_seq_len // block_q - 1)
    def write_grads():
        dk_tile_ref[...] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)
        dv_tile_ref[...] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
```

**Kernel 2: Compute dQ** (iterate over KV blocks):

```python
def flash_attention_dq_kernel(...):
    kv_seq_index = pl.program_id(3)  # Grid dimension is KV
    q_seq_index = pl.program_id(2)   # Fixed Q block
    
    @pl.when(kv_seq_index == 0)
    def init_grad():
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
    
    # Recompute forward quantities (same as dkv kernel)
    # ...
    
    # Softmax gradient
    ds = (dp - pltpu.repeat(di, block_kv // MIN_BLOCK_SIZE, axis=1)) * p
    if sm_scale != 1.0:
        ds *= sm_scale
    
    # dQ = dS @ K
    dq = lax.dot(ds.astype(k.dtype), k, preferred_element_type=jnp.float32)
    dq_scratch_ref[...] += dq.astype(dq_scratch_ref.dtype)
    
    @pl.when(kv_seq_index == kv_seq_len // block_kv - 1)
    def write_grad():
        dq_tile_ref[...] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
```

**Why two kernels?**
- dK/dV: Each KV gradient depends on **all** Q blocks → iterate over Q
- dQ: Each Q gradient depends on **all** KV blocks → iterate over KV
- Allows streaming one dimension while accumulating the other

---

## Splash Attention (Sparse)

### Motivation

FlashAttention processes **all** KV blocks sequentially. For sparse attention (e.g., local, block-sparse), we can skip irrelevant blocks:

```
Standard FlashAttention:
Q block 0: compute with KV [0, 1, 2, 3, 4, 5, 6, 7]
Q block 1: compute with KV [0, 1, 2, 3, 4, 5, 6, 7]
...

Splash Attention (local window=2):
Q block 0: compute with KV [0, 1]      ← Skip blocks 2-7
Q block 1: compute with KV [1, 2]      ← Skip blocks 0, 3-7
Q block 2: compute with KV [2, 3]
...
```

### Mask Metadata

Splash uses three metadata arrays to describe sparsity:

```python
@dataclasses.dataclass
class MaskInfo(NamedTuple):
    data_next: jax.Array | None   # [heads, q_blocks, kv_iters] → next KV block index
    block_mask: jax.Array | None   # [heads, q_blocks, kv_iters] → is block nonzero?
    mask_next: jax.Array | None    # [heads, q_blocks, kv_iters] → fine-grained mask index
    partial_mask_blocks: jax.Array | None  # [heads, q_blocks, kv_blocks, block_q, block_kv] → element-wise mask
    q_sequence: jax.Array | None   # [q_seq_len] → sequence indices for mask_function
```

**`data_next`**: Maps iteration index to actual KV block index (skips zeros):
```python
# Example: local attention (window_size=2)
# data_next[head=0, q_block=3, iter=0] = 2  ← First nonzero KV block for Q block 3
# data_next[head=0, q_block=3, iter=1] = 3  ← Second nonzero KV block
```

**`block_mask`**: Binary mask (0=skip, 1=compute, 2=compute+apply element-wise mask):
```python
if block_mask[h, i, j] == 0:
    # Skip this iteration entirely (no computation)
    pass
elif block_mask[h, i, j] == 1:
    # Compute but don't apply fine-grained mask
    compute_block()
elif block_mask[h, i, j] == 2:
    # Compute and apply partial_mask_blocks
    compute_block_with_mask()
```

### Prefetch Pattern

```python
def _next_nonzero(h, i, j, data_next_ref, block_mask_ref, m_next_ref=None):
    """Fetch next KV block index and mask status."""
    if data_next_ref is None:
        # Dense attention: just use j as KV index
        return j, None, True, True
    
    # Sparse attention: look up actual KV block
    is_nonzero = block_mask_ref[h, i, j] > 0
    should_not_mask = block_mask_ref[h, i, j] != 1  # Apply fine mask if == 2
    next_j = data_next_ref[h, i, j].astype(jnp.int32)
    next_m = m_next_ref[h, i, j].astype(jnp.int32) if m_next_ref is not None else None
    
    return next_j, next_m, is_nonzero, should_not_mask
```

**Usage in kernel**:
```python
kv_index, mask_index, should_run, should_not_mask = _next_nonzero(
    h, i, j, data_next_ref, block_mask_ref, mask_next_ref
)

@pl.when(should_run)
def compute():
    # Fetch KV block at `kv_index` (not `j`!)
    k = k_ref[batch, head, kv_index, :]
    v = v_ref[batch, head, kv_index, :]
    
    # Apply fine-grained mask if needed
    if should_not_mask:
        mask = partial_mask_blocks[mask_index, :, :]
        logits = jnp.where(mask, logits, mask_value)
```

### Dynamic vs Static Masks

**Static masks** (numpy arrays): Compiled into kernel, zero runtime cost
```python
mask = np.array([...])  # Boolean mask known at trace time
splash_kernel = make_splash_mha(mask, block_sizes=...)
out = splash_kernel(q, k, v)
```

**Dynamic masks** (jax.Array): Passed as input, flexible but slower
```python
mask = jnp.array([...])  # Mask can change between calls
splash_kernel = make_splash_mha(mask, block_sizes=...)
out = splash_kernel(q, k, v, mask)  # Mask is input
```

---

## Common Patterns

### Block Size Selection

```python
@dataclasses.dataclass(frozen=True)
class BlockSizes:
    # Forward pass
    block_q: int           # Query tile size
    block_k_major: int     # KV major tile (loaded from HBM)
    block_k: int           # KV minor tile (compute granularity)
    block_b: int           # Batch tile
    
    # Backward pass (dK/dV kernel)
    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None
    
    # Backward pass (dQ kernel)
    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

# Example: TPU v5e
block_sizes = BlockSizes(
    block_q=128,
    block_k_major=128,
    block_k=128,
    block_b=1,
    block_q_major_dkv=128,
    block_k_major_dkv=128,
    block_k_dkv=128,
    block_q_dkv=128,
    block_k_major_dq=128,
    block_k_dq=128,
    block_q_dq=128,
)
```

**Tuning guidelines**:
- All block sizes must be multiples of `MIN_BLOCK_SIZE` (128)
- Larger blocks → better MXU utilization, more VMEM usage
- Forward: `block_k_major` = `block_k` for simplicity (no nested tiling)
- Backward: May use different sizes to balance memory/compute

### Segment IDs (Packed Sequences)

```python
# Prevent cross-attention between segments in batch
segment_ids = SegmentIds(
    q=jnp.array([0, 0, 0, 1, 1, 1]),   # First 3 tokens = segment 0, next 3 = segment 1
    kv=jnp.array([0, 0, 0, 1, 1, 1])
)

# In kernel, broadcast to lanes
q_segment_ids = jax.lax.broadcast_in_dim(
    segment_ids.q, (batch, q_seq_len, NUM_LANES), (0, 1)
)
kv_segment_ids = jax.lax.broadcast_in_dim(
    segment_ids.kv, (batch, NUM_SUBLANES, kv_seq_len), (0, 2)
)

# Vectorized comparison
mask = jnp.equal(q_segment_ids, kv_segment_ids)  # [batch, q_seq, kv_seq]
logits = jnp.where(mask, logits, mask_value)
```

### Causal Masking with Early Termination

```python
if causal:
    # Q block i only needs KV blocks [0, ..., i]
    block_max_kv_steps = pl.cdiv(
        (q_seq_idx + 1) * block_q,  # Max token index in Q block
        block_kv
    )
else:
    block_max_kv_steps = kv_seq_len // block_kv

# Only iterate over needed KV blocks
@pl.loop(0, block_max_kv_steps)
def kv_loop(kv_idx):
    # Compute attention for this KV block
    ...
```

---

## Performance Optimization

### Dimension Semantics

```python
compiler_params=pltpu.CompilerParams(
    dimension_semantics=(
        "parallel",    # Distribute across cores (batch)
        "parallel",    # Distribute across cores (heads)
        "arbitrary",   # Sequential (Q blocks, stateful)
        "arbitrary",   # Sequential (KV blocks, reduction)
    )
)
```

**Parallel**: Independent work, can run on separate cores
**Arbitrary**: Sequential execution, may involve reductions or state

**Common patterns**:
- Forward: `("parallel", "parallel", "parallel", "arbitrary")` — distribute batch/heads, sequential over KV
- Backward dKV: `("parallel", "parallel", "parallel", "arbitrary")` — distribute batch/heads, reduce over Q
- Backward dQ: `("parallel", "parallel", "arbitrary", "arbitrary")` — distribute batch/heads, reduce over KV

### Cost Estimation (Optional)

```python
def _fwd_cost_estimate(q, k, v, ..., kernel_inputs_specs, kernel_outputs_specs):
    # Estimate FLOPs for autotuning
    body_cost = pl.estimate_cost(reference_impl, q, k, v, ...)
    
    input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
    
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes
    )

pl.pallas_call(
    kernel,
    ...,
    cost_estimate=_fwd_cost_estimate(...)
)
```

### VMEM Usage Optimization

```python
# Bad: Allocate full sequence in VMEM
vmem_scratch = pltpu.VMEM((batch_size, seq_len, hidden_dim), jnp.float32)  # May OOM!

# Good: Allocate only tile size
vmem_scratch = pltpu.VMEM((block_q, hidden_dim), jnp.float32)  # Reuse per tile
```

### Loop Unrolling

```python
# Unroll small loops for better pipelining
lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)
```

---

## Debugging

### Environment Variables

```bash
# Enable XLA dumps
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"

# TPU-specific debug flags
export TPU_DEBUG=1
export TPU_STDERR_LOG_LEVEL=0  # 0=INFO, 1=WARNING, 2=ERROR
```

### Interpret Mode

```python
# Run kernel on CPU for debugging (slow but visible stack traces)
result = pl.pallas_call(
    kernel,
    ...,
    interpret=True
)(inputs...)
```

### Common Errors

**"Block size must be a multiple of NUM_LANES"**
```python
# Fix: Ensure all dimensions divisible by 128
block_q = 128  # ✓
block_q = 127  # ✗ Error!
```

**"Grid dimension mismatch"**
```python
# Fix: Ensure grid matches index_map expectations
grid = (batch, heads, q_blocks, kv_blocks)

def index_map(b, h, q, kv):  # Must accept 4 args if grid has 4 dims
    return (b, h, q, 0)
```

**"VMEM out of memory"**
```python
# Fix: Reduce tile sizes or scratch allocations
block_q = 64  # Down from 128
scratch_shapes = [pltpu.VMEM((block_q, head_dim), jnp.float32)]  # Smaller tiles
```

**"NaN/Inf in backward pass"**
```python
# Check softmax statistics are saved correctly
assert l_ref.dtype == jnp.float32  # Use FP32 for statistics
assert m_ref.dtype == jnp.float32

# Verify di computation
di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
```

---

## Comparison: TPU vs GPU FlashAttention

| Aspect | TPU (Pallas) | GPU (Mosaic) |
|--------|--------------|--------------|
| **Memory** | VMEM (16MB, no barriers) | SMEM (100KB, explicit barriers) |
| **Compute** | MXU (systolic array) | TensorCore (wgmma) |
| **Threads** | 128 lanes (vector) | 32/64 threads (warp) |
| **Parallelism** | `dimension_semantics` | Warp specialization |
| **Block size** | Multiple of 128 | Multiple of 64 |
| **Barriers** | Not needed (single-core) | Critical (k_consumed, v_consumed) |
| **Masking** | `jnp.where` (vectorized) | WGMMA layout-aware |
| **Pipeline** | Loop unrolling | `emit_pipeline_warp_specialized` |

---

## Resources

- **Pallas TPU Guide**: https://docs.jax.dev/en/latest/pallas/tpu/quickstart.html
- **Splash Attention Paper**: https://arxiv.org/abs/2406.xxxxx
- **JAX Pallas API**: https://docs.jax.dev/en/latest/pallas/index.html
- **TPU v5e Specs**: https://cloud.google.com/tpu/docs/v5e

---

**Last Updated**: November 23, 2025
