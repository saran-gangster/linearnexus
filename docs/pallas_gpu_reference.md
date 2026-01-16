# Pallas GPU Kernel Programming Reference

**Comprehensive guide to writing custom GPU kernels with JAX Pallas targeting Triton and Mosaic GPU.**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Memory Hierarchy & Spaces](#memory-hierarchy--spaces)
4. [Writing Your First Kernel](#writing-your-first-kernel)
5. [Grid and BlockSpec](#grid-and-blockspec)
6. [Mosaic GPU API](#mosaic-gpu-api)
7. [Pipelining](#pipelining)
8. [Synchronization Primitives](#synchronization-primitives)
9. [TensorCore Operations](#tensorcore-operations)
10. [Common Patterns](#common-patterns)
11. [Debugging](#debugging)

---

## Introduction

### What is Pallas?

Pallas is JAX's extension for writing custom GPU and TPU kernels using a Triton-like programming model. It provides:

- **Array-based programming**: Write kernels using NumPy-style operations
- **Hardware abstraction**: Same code can target GPUs (Triton/Mosaic) and TPUs (Mosaic)
- **JAX integration**: Compose with JAX transformations (vmap, grad, jit)
- **Low-level control**: Access to memory hierarchy and specialized hardware units

### Design Philosophy

```
High-level abstractions → Pallas → Backend-specific IR → Hardware
    (JAX/NumPy)                    (Triton/Mosaic)      (GPU/TPU)
```

**Key Differences from Regular JAX:**
- Use `Ref` types (mutable references) instead of immutable arrays
- Explicit memory management (GMEM ↔ SMEM ↔ registers)
- Grid-based parallel execution model
- Restricted primitive set (no dynamic shapes, limited collectives)

---

## Core Concepts

### Reference Types (`Ref`)

`Ref`s are mutable memory references that replace immutable JAX arrays in kernels.

```python
def kernel(x_ref, y_ref, o_ref):
    # x_ref, y_ref: input Refs (read-only by convention)
    # o_ref: output Ref (write-only by convention)
    
    # Read from Ref → get jax.Array in registers
    x = x_ref[:]  # or x_ref[...] 
    y = y_ref[:]
    
    # Compute (operates on registers)
    result = x + y
    
    # Write to Ref
    o_ref[:] = result
```

**Indexing Patterns:**
```python
# Slicing
x = x_ref[0, 2:5, :]

# Advanced indexing (auto-broadcasts)
idx = jnp.arange(2)[:, None]
x = x_ref[idx, :]

# Dynamic slicing with pl.ds
x = pl.load(x_ref, (0, pl.ds(start=2, size=3), slice(None)))
```

### Kernel Invocation with `pallas_call`

```python
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        add_kernel,  # Kernel function
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),  # Output spec
        grid=(1,),  # Iteration space
        in_specs=[pl.BlockSpec(x.shape, lambda i: (0,))],  # Optional
        out_specs=pl.BlockSpec(x.shape, lambda i: (0,))    # Optional
    )(x, y)
```

**Execution Semantics:**
```python
# Conceptually equivalent to:
for indices in itertools.product(*[range(g) for g in grid]):
    transformed_inputs = [spec.transform(arg, indices) for arg, spec in zip(inputs, in_specs)]
    transformed_outputs = [spec.transform(out, indices) for out, spec in zip(outputs, out_specs)]
    kernel(*transformed_inputs, *transformed_outputs)
```

---

## Memory Hierarchy & Spaces

### GPU Memory Hierarchy

```
┌─────────────────────────────────────────────────┐
│ Registers (fastest, ~256KB per SM)             │  ← Immediate operands
├─────────────────────────────────────────────────┤
│ Shared Memory / L1 Cache (~100MB)              │  ← plgpu.SMEM
├─────────────────────────────────────────────────┤
│ L2 Cache (~50-100MB, implicit)                 │  ← Hardware-managed
├─────────────────────────────────────────────────┤
│ Global Memory / HBM (10s-100s GB)              │  ← plgpu.GMEM
└─────────────────────────────────────────────────┘
```

**Blackwell adds:**
- **Tensor Memory (TMEM)**: Explicit register space for TensorCore accumulators (~same size as registers)

### Memory Space APIs

```python
# Mosaic GPU
import jax.experimental.pallas.mosaic_gpu as plgpu

# Allocate in specific memory spaces
scratch_shapes = {
    'gmem_buf': plgpu.GMEM((128, 128), jnp.float16),  # Global memory
    'smem_buf': plgpu.SMEM((128, 128), jnp.float16),  # Shared memory
    'tmem_acc': plgpu.TMEM((128, 128), jnp.float32),  # Tensor memory (Blackwell)
}

# BlockSpec with memory_space
pl.BlockSpec(
    block_shape=(128, 128),
    index_map=lambda i: (i, 0),
    memory_space=plgpu.SMEM  # Place inputs in SMEM
)
```

### Data Movement Costs

| Operation | Latency | Bandwidth | Use Case |
|-----------|---------|-----------|----------|
| Register access | 1 cycle | ~20 TB/s | Immediate compute |
| SMEM load/store | ~10 cycles | ~10 TB/s | Shared data, TensorCore operands |
| L2 hit | ~100 cycles | ~5 TB/s | Recently accessed data |
| GMEM access | ~500 cycles | ~2 TB/s | Main data transfers |

**Rule of Thumb**: Maximize compute intensity (FLOPs per byte transferred) by reusing data in faster memory.

---

## Writing Your First Kernel

### Example 1: Element-wise Addition

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    """Element-wise addition kernel."""
    # Load from SMEM → registers
    x = x_ref[:]
    y = y_ref[:]
    
    # Compute
    o_ref[:] = x + y

# Invoke with pallas_call
@jax.jit
def add_vectors(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

# Usage
x = jnp.arange(8, dtype=jnp.float32)
y = jnp.ones(8, dtype=jnp.float32)
result = add_vectors(x, y)
```

### Example 2: Blocked Matrix Multiplication

```python
def matmul_kernel(x_ref, y_ref, o_ref):
    """Naive blocked matmul: o = x @ y."""
    # x_ref: (tile_m, K), y_ref: (K, tile_n), o_ref: (tile_m, tile_n)
    o_ref[...] = x_ref[...] @ y_ref[...]

def matmul(x, y, block_m=128, block_n=128):
    m, k = x.shape
    _, n = y.shape
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // block_m, n // block_n),
        in_specs=[
            pl.BlockSpec((block_m, k), lambda i, j: (i, 0)),  # Row blocks of x
            pl.BlockSpec((k, block_n), lambda i, j: (0, j))   # Column blocks of y
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))
    )(x, y)
```

---

## Grid and BlockSpec

### Grid: Iteration Space

The `grid` defines a multi-dimensional iteration space. Each grid point executes the kernel **once**.

```python
grid = (M, N, K)  # 3D grid with M*N*K total invocations

# Equivalent to:
for i in range(M):
    for j in range(N):
        for k in range(K):
            kernel(...)  # Can run in parallel on GPU
```

**Querying Grid Position:**
```python
def kernel(...):
    i = pl.program_id(axis=0)  # Index along first grid dimension
    j = pl.program_id(axis=1)  # Index along second grid dimension
    
    grid_size_i = pl.num_programs(axis=0)  # Total size of first dimension
```

### BlockSpec: Data Slicing

`BlockSpec` maps grid indices to data slices.

```python
pl.BlockSpec(
    block_shape: tuple[int | None, ...],  # Shape of slice (None = squeeze)
    index_map: Callable[[*grid_indices], tuple[int, ...]],  # Grid idx → block idx
    memory_space: MemorySpace = SMEM  # Where to place data
)
```

**Example: Tiled Matrix Multiply**
```python
# Compute C = A @ B with (128, 128) output tiles
grid = (M // 128, N // 128)

in_specs = [
    # A: each program gets a (128, K) row block
    pl.BlockSpec((128, K), lambda i, j: (i, 0)),
    
    # B: each program gets a (K, 128) column block
    pl.BlockSpec((K, 128), lambda i, j: (0, j))
]

out_specs = pl.BlockSpec((128, 128), lambda i, j: (i, j))
```

**Advanced: Element Indexing Mode**
```python
# Unblocked indexing (returns element indices, not block indices)
pl.BlockSpec(
    (pl.Element(128), pl.Element(128)),  # Element-wise indexing
    lambda i, j: (i * 128, j * 128)     # Returns element offsets
)
```

---

## Mosaic GPU API

### Core Primitives

#### Asynchronous Copies

**GMEM → SMEM (TMA - Tensor Memory Accelerator):**
```python
def kernel(x_gmem, o_gmem, x_smem, barrier):
    # Async copy from global to shared memory
    plgpu.copy_gmem_to_smem(
        x_gmem,              # Source in GMEM
        x_smem,              # Destination in SMEM
        barrier,             # Completion barrier
        collective_axes=None # Optional: multicast across cluster
    )
    
    # Wait for copy to complete
    plgpu.barrier_wait(barrier)
    
    # Now safe to use x_smem
    result = jnp.exp(x_smem[...])
    
    # Copy result back to GMEM
    plgpu.commit_smem()  # Ensure SMEM writes visible to TMA
    plgpu.copy_smem_to_gmem(x_smem, o_gmem)
    plgpu.wait_smem_to_gmem(0)  # Wait for write to complete
```

**Collective Copies (Block Clusters):**
```python
# Multicast same data to multiple blocks in cluster
plgpu.copy_gmem_to_smem(
    x_gmem, x_smem, barrier,
    collective_axes="cluster"  # All blocks get same data
)

# Partitioned copy (each block gets half)
plgpu.copy_gmem_to_smem(
    x_gmem,  # Shape: (M, N)
    x_smem,  # Shape: (M, N//2) - each block gets half
    barrier,
    collective_axes="cluster",
    partitioned_axis=1  # Split along axis 1
)
```

#### Memory Synchronization

```python
# Make SMEM writes visible to async hardware units
plgpu.commit_smem()

# Example: ensure TMA sees updated SMEM
smem_ref[...] = value
plgpu.commit_smem()
plgpu.copy_smem_to_gmem(smem_ref, gmem_ref)
```

---

## Pipelining

### Concept: Overlapping Compute and Memory

```
Time →
Step 0: [Load A[0]] [ Compute ]  [ Store ]
Step 1:            [Load A[1]] [Compute] [Store]
Step 2:                       [Load A[2]] [Compute] [Store]
```

### Mosaic GPU: `emit_pipeline`

```python
def matmul_pipelined(a, b):
    m, k = a.shape
    _, n = b.shape
    tile_m, tile_n, tile_k = 128, 128, 64
    
    def kernel(a_gmem, b_gmem, o_gmem, acc):
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)
        
        def pipeline_step(ki, a_smem, b_smem):
            # TensorCore matmul (async)
            plgpu.wgmma(acc, a_smem, b_smem)
            plgpu.wgmma_wait(1)  # Wait for previous step
        
        # Pipeline over K dimension
        swizzle = 128
        transforms = (
            plgpu.TilingTransform((8, swizzle // 2)),
            plgpu.SwizzleTransform(swizzle)
        )
        
        plgpu.emit_pipeline(
            pipeline_step,
            grid=(k // tile_k,),
            in_specs=[
                plgpu.BlockSpec((tile_m, tile_k), lambda ki: (pid_m, ki), transforms=transforms),
                plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, pid_n), transforms=transforms)
            ],
            max_concurrent_steps=2,  # Double-buffer
            delay_release=1          # Don't overwrite until step i+1 completes
        )
        
        # Store result
        o_smem[...] = acc[...].astype(a.dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[...])
        plgpu.wait_smem_to_gmem(0)
    
    return plgpu.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=(m // tile_m, n // tile_n),
        scratch_shapes={
            'o_smem': plgpu.SMEM((tile_m, tile_n), a.dtype),
            'acc': plgpu.ACC((tile_m, tile_n), jnp.float32)
        }
    )(a, b)
```

**Key Parameters:**
- `max_concurrent_steps`: Number of overlapping iterations (typically 2-6)
- `delay_release=1`: Required when not awaiting async ops immediately (e.g., WGMMA)

### Warp-Specialized Pipeline: `emit_pipeline_warp_specialized`

For complex kernels, dedicate separate warpgroups to memory movement vs computation. The memory warpgroup issues TMA transfers while compute warpgroups crunch numbers, maximizing overlap.

```python
def warp_specialized_attention(q_gmem, k_gmem, v_gmem, out_gmem):
    """FlashAttention-style pipeline with 2 compute + 1 memory warpgroup."""
    
    def compute_context(pipeline_callback):
        """Runs in compute warpgroups (wg_idx=0,1)."""
        wg_idx = lax.axis_index("wg")
        q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q
        
        # Load Q tile (not pipelined)
        qo_smem = qo_smem2.at[wg_idx]
        plgpu.copy_gmem_to_smem(
            q_gmem.at[batch, pl.ds(q_seq_base, block_q), q_head],
            qo_smem,
            q_barriers.at[wg_idx]
        )
        plgpu.barrier_wait(q_barriers.at[wg_idx])
        
        # Initialize accumulator
        acc = plgpu.layout_cast(
            jnp.full((block_q, head_dim), 0, jnp.float32),
            plgpu.Layout.WGMMA
        )
        m_i = jnp.full((block_q,), -jnp.inf, jnp.float32)
        l_i = jnp.full((block_q,), 0, jnp.float32)
        
        # Run pipeline with stateful carry
        acc, m_i, l_i = pipeline_callback((acc, m_i, l_i))
        
        # Normalize & store
        acc /= lax.broadcast_in_dim(l_i, (block_q, head_dim), [0])
        qo_smem[...] = acc.astype(dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(qo_smem, out_gmem.at[...])
        plgpu.wait_smem_to_gmem(0)
    
    def kv_step(kv_step_idx, k_smem, v_smem, 
                k_consumed_barrier, v_consumed_barrier, carry):
        """Pipeline step: process one KV tile (runs in compute warpgroups)."""
        acc, m_i, l_i = carry
        slot = lax.rem(kv_step_idx, max_concurrent_steps)
        qo_smem = qo_smem2.at[lax.axis_index("wg")]
        
        # QK matmul
        def compute_qk(acc_ref):
            plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem, (1, 0)))
            return acc_ref[...]
        
        qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
        plgpu.barrier_arrive(k_consumed_barrier)  # Release K buffer
        
        # Online softmax (FlashAttention algorithm)
        log2e = math.log2(math.e)
        m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)
        alpha = jnp.exp2(m_i - m_ij)
        m_i = m_ij
        p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
        acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
        l_i *= alpha
        l_i += p.sum(axis=1)
        
        # PV matmul
        def compute_pv(acc_ref):
            plgpu.wgmma(acc_ref, p.astype(dtype), v_smem)
        
        acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
        plgpu.barrier_arrive(v_consumed_barrier)  # Release V buffer
        
        return acc, m_i, l_i
    
    # Create warp-specialized pipeline
    pipeline = plgpu.emit_pipeline_warp_specialized(
        kv_step,
        grid=(kv_seq_len // block_kv,),
        in_specs=[
            plgpu.BlockSpec(  # K
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=[tiling, swizzle]
            ),
            plgpu.BlockSpec(  # V
                block_shape=(block_kv, head_dim),
                index_map=lambda i: (i, 0),
                transforms=[tiling, swizzle]
            ),
        ],
        max_concurrent_steps=2,
        num_compute_wgs=2,        # 2 compute warpgroups
        memory_registers=40,       # Memory warpgroup uses fewer registers
        wg_axis="wg",
        manual_consumed_barriers=True,  # We manually signal consumed barriers
        compute_context=compute_context
    )
    
    # Execute pipeline (memory warpgroup automatically streams K/V)
    k_ref_slice = k_gmem.at[batch, :, kv_head, :]
    v_ref_slice = v_gmem.at[batch, :, kv_head, :]
    pipeline(k_ref_slice, v_ref_slice)

# Invoke with 3 warpgroups (2 compute + 1 memory)
plgpu.kernel(
    attention_kernel,
    num_threads=3,
    thread_name="wg",
    ...
)(q, k, v)
```

**Key Differences from Standard Pipeline**:

1. **`compute_context`**: Function that wraps the pipeline call. Runs **only in compute warpgroups**. Use for initialization and epilogue (storing results).

2. **`num_compute_wgs`**: Number of compute warpgroups (default 1). Each processes disjoint work (e.g., different query tiles). Memory warpgroup streams data for all.

3. **`memory_registers`**: Register budget for memory warpgroup (typically 40-80). Lower than compute warpgroups (200-240) since it only issues TMA transfers.

4. **`manual_consumed_barriers=True`**: Pipeline step must explicitly signal when done with buffers via `plgpu.barrier_arrive(k_consumed_barrier)`. Memory warpgroup waits on these before recycling slots.

5. **`wg_axis`**: Name of thread axis (must match `thread_name` in `plgpu.kernel`). Used to index warpgroup-specific data (e.g., `qo_smem2.at[wg_idx]`).

**Memory Warpgroup Auto-Generated Code**:
```python
# Implicitly generated by emit_pipeline_warp_specialized
@pl.when(wg_idx == num_compute_wgs)  # Last warpgroup
def _memory_wg():
    plgpu.set_max_registers(memory_registers, action="decrease")
    
    # Prologue: fill pipeline
    for i in range(max_concurrent_steps):
        plgpu.copy_gmem_to_smem(
            k_ref.at[BlockSpec.index_map(i)],
            k_smem.at[i],
            k_load_barriers.at[i]
        )
        plgpu.copy_gmem_to_smem(
            v_ref.at[BlockSpec.index_map(i)],
            v_smem.at[i],
            v_load_barriers.at[i]
        )
    
    # Main loop: double-buffer
    @pl.loop(0, grid_size - max_concurrent_steps)
    def _loop(step):
        slot = lax.rem(step, max_concurrent_steps)
        next_idx = step + max_concurrent_steps
        
        # Wait for compute to finish with this slot
        plgpu.barrier_wait(k_consumed_barriers.at[slot])
        plgpu.copy_gmem_to_smem(
            k_ref.at[BlockSpec.index_map(next_idx)],
            k_smem.at[slot],
            k_load_barriers.at[slot]
        )
        
        plgpu.barrier_wait(v_consumed_barriers.at[slot])
        plgpu.copy_gmem_to_smem(
            v_ref.at[BlockSpec.index_map(next_idx)],
            v_smem.at[slot],
            v_load_barriers.at[slot]
        )
```

**Advantages**:
- **Perfect overlap**: Memory transfers happen while compute is busy (hides TMA latency)
- **Register pressure relief**: Compute warpgroups don't hold TMA state
- **Scalability**: Add more compute warpgroups without changing memory warpgroup logic
- **Determinism**: Memory operations are isolated, easier to reason about barrier lifetimes

**When to Use**:
- Kernels with high compute intensity (many FLOPs per byte loaded)
- Large tiles where TMA latency dominates (>10μs per transfer)
- Multi-warpgroup compute (e.g., processing different output tiles in parallel)

**When NOT to Use**:
- Simple element-wise kernels (overhead exceeds benefit)
- Memory-bound kernels (compute can't hide transfer latency anyway)
- Small tiles (TMA overhead ~1μs, not worth specialization)

---

## Synchronization Primitives

### Barriers

```python
# Allocate barriers
scratch_shapes = {
    'barrier': plgpu.Barrier(num_arrivals=1, num_barriers=2)  # 2 barriers for double-buffering
}

def kernel(barrier):
    slot = i % 2  # Alternate between 2 barriers
    
    # Async copy arrives on barrier
    plgpu.copy_gmem_to_smem(src, dst, barrier.at[slot])
    
    # Wait for completion
    plgpu.barrier_wait(barrier.at[slot])
```

**Barrier Rules (CRITICAL):**
1. Each barrier must see equal number of `arrive` and `wait` operations over its lifetime
2. No two arrivals without a `wait` in between (corrupts state)
3. All threads that `wait` must observe **every** completion

### Semaphores (Cross-Device)

```python
# Inter-GPU communication over NVLINK
def exchange_kernel(x_ref, y_ref, done_sem):
    other_dev = 1 - lax.axis_index("x")
    neighbor_ref = plgpu.remote_ref(y_ref, other_dev)
    
    # Write to remote device
    neighbor_ref[...] = x_ref[...]
    pl.semaphore_signal(done_sem, device_id=other_dev)
    
    # Wait for remote write to complete
    pl.semaphore_wait(done_sem)

# Allocate semaphore
scratch_shapes = {'done_sem': plgpu.Semaphore.REGULAR}
```

---

## TensorCore Operations

### Hopper: `wgmma` (Warpgroup Matrix Multiply-Accumulate)

```python
# Allocate accumulator
scratch_shapes = {'acc': plgpu.ACC((128, 128), jnp.float32)}

def kernel(a_smem, b_smem, acc):
    # Issue async matmul (runs on TensorCore)
    plgpu.wgmma(
        acc,      # Accumulator ref (M, N)
        a_smem,   # Left operand (M, K) in SMEM
        b_smem    # Right operand (K, N) in SMEM
    )
    
    # Synchronize (wait for all but last N operations)
    plgpu.wgmma_wait(0)  # Wait for all to complete
    
    # Read result (implicitly waits)
    result = acc[...]
```

**Requirements:**
- `M` divisible by 64
- `N` divisible by 8, ≤256
- `K` = multiple of (swizzle / dtype_size)
- Operands in SMEM with correct transforms (tiling + swizzle)

### Blackwell: `tcgen05_mma`

```python
# Use TMEM for accumulator
scratch_shapes = {
    'acc_tmem': plgpu.TMEM((128, 256), jnp.float32, collective=True),
    'mma_barrier': plgpu.Barrier(orders_tensor_core=True)
}

def kernel(a_smem, b_smem, acc_tmem, barrier):
    # Collective MMA (2 blocks in cluster)
    plgpu.tcgen05_mma(
        acc_tmem,
        a_smem,
        b_smem,
        barrier,
        accumulate=False,
        collective_axis="cluster"  # Collaborate across 2 SMs
    )
    
    # Wait for completion
    plgpu.barrier_wait(barrier)
    
    # Read from TMEM (async!)
    result = plgpu.async_load_tmem(acc_tmem)
    plgpu.wait_load_tmem()  # Must wait before reusing TMEM
```

**Blackwell Features:**
- **Collective MMA**: 2 SMs share `B` operand, each computes half of result
- **TMEM**: Explicit tensor memory for accumulators
- **Larger shapes**: M=128/256, N≤512 (non-collective), K more flexible

---

## Common Patterns

### Warp Specialization (Hopper+)

Dedicate warps to different tasks (memory vs compute).

```python
def kernel(...):
    @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
    def per_warp():
        warp_id = lax.axis_index("warp")
        
        @pl.when(warp_id == 0)
        def memory_warp():
            # Fetch data from GMEM
            for ki in range(k_iters):
                slot = ki % num_buffers
                plgpu.copy_gmem_to_smem(a_gmem.at[...], a_smem.at[slot], barrier.at[slot])
        
        @pl.when(warp_id == 1)
        def compute_warp():
            # Consume data, issue TensorCore ops
            for ki in range(k_iters):
                slot = ki % num_buffers
                plgpu.barrier_wait(barrier.at[slot])
                plgpu.wgmma(acc, a_smem.at[slot], b_smem.at[slot])
```

### Persistent Kernels

Launch grid smaller than work size, each block processes multiple tiles.

```python
num_sms = jax.extend.backend.get_default_device().core_count
grid = (num_sms // 2,)  # Launch fewer blocks than work items

@plgpu.nd_loop((m_iters, n_iters), collective_axes="grid")
def tile_loop(loop_info):
    m_idx, n_idx = loop_info.index
    # Process tile (m_idx, n_idx)
    ...
```

### Grid Tiling (L2 Cache Optimization)

Reorder grid traversal to improve cache locality.

```python
# Bad: Sequential over M, then N
# grid iteration: (0,0), (0,1), ..., (0,N-1), (1,0), (1,1), ...

# Good: Snake pattern within tiles
m_idx, n_idx = plgpu.planar_snake(
    linear_idx,
    grid_shape=(m_iters, n_iters),
    minor_dim=1,        # N is fastest-changing
    tile_width=8        # Process 8 N-blocks before switching M
)
# Iteration: (0,0), (0,1), ..., (0,7), (1,7), (1,6), ..., (1,0), (2,0), ...
```

---

## Debugging

### Environment Variables

```bash
# Dump compilation logs
export MOSAIC_GPU_DUMP_PTXAS=1

# Dump generated code
export MOSAIC_GPU_DUMP_PTX=1       # PTX intermediate
export MOSAIC_GPU_DUMP_SASS=1      # Final machine code

# MLIR passes
export MOSAIC_GPU_DUMP_MLIR_PASSES=1

# Save to directory
export MOSAIC_GPU_DUMP_TO=/path/to/dir
```

### In-Kernel Debugging

```python
# Use jax.debug.print (recommended)
def kernel(x_ref):
    value = x_ref[0]
    jax.debug.print("Value at [0]: {}", value)

# Interpret mode (runs on CPU, easier to debug)
result = pl.pallas_call(
    kernel,
    out_shape=...,
    interpret=True  # Disable JIT, run sequentially
)(x)
```

### Common Errors

**"WGMMA operation failed"**
- Check operand shapes (M%64, N%8, K%swizzle)
- Verify SMEM transforms (TilingTransform + SwizzleTransform)
- Ensure operands in SMEM, not registers

**"Barrier corruption"**
- Two arrivals without `wait` between
- Unequal arrive/wait counts
- Thread only waits on some completions (must wait on all)

**"Register spills"**
- Reduce `max_concurrent_steps`
- Decrease tile sizes
- Use `memory_registers` in warp-specialized pipelines

**"TMA alignment error"**
- BlockSpec shapes must align with swizzle size (typically 128 bytes)
- For f16: `head_dim % (128 / 2) == 0` → `head_dim % 64 == 0`
- Transforms must be `(TilingTransform, SwizzleTransform)` in that order

**"Barrier deadlock"**
- Ensure all arrivals have matching waits across **all** execution paths
- Use `@pl.when` carefully: barriers in conditional blocks must balance
- In pipelines: prologue arrivals = main loop arrivals = epilogue arrivals

**"Numerical Instability (NaN/Inf)"**
- FlashAttention-style softmax: use base-2 (`log2e`, `exp2`) not natural log
- Check for division by zero in normalization (`acc / l_i`)
- Verify initial `m_i = -inf` and `l_i = 0` for online aggregation

### Backward Pass Pitfalls

When implementing `custom_vjp` for attention kernels:

```python
@partial(jax.custom_vjp, nondiff_argnums=(3,))
def attention(q, k, v, config: TuningConfig):
    return _attention_forward(q, k, v, config, save_residuals=False)

def _attention_fwd(q, k, v, config: TuningConfig):
    """Forward pass: save residuals for backward."""
    out, (lse,) = _attention_forward(q, k, v, config, save_residuals=True)
    return out, (q, k, v, out, lse)  # Save all inputs + residuals

def _attention_bwd(config: TuningConfig, res, do):
    """Backward pass: compute dQ, dK, dV."""
    q, k, v, out, lse = res
    
    # Precompute delta = sum(dO * O) for softmax derivative
    delta = jnp.einsum('bqhd,bqhd->bhq', 
                       out.astype(jnp.float32), 
                       do.astype(jnp.float32))
    
    # dQ kernel: pipeline over KV tiles
    dq = dq_kernel(q, k, v, do, lse, delta, config)
    
    # dK/dV kernel: pipeline over query tiles
    dk, dv = dkv_kernel(q, k, v, do, lse, delta, config)
    
    # Handle GQA: sum gradients over query head groups
    if num_q_heads > num_kv_heads:
        q_heads_per_kv_head = num_q_heads // num_kv_heads
        sum_shape = (*k.shape[:-1], q_heads_per_kv_head, head_dim)
        dk = dk.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dk.dtype)
        dv = dv.reshape(sum_shape).astype(jnp.float32).sum(axis=-2).astype(dv.dtype)
    
    return dq, dk, dv

attention.defvjp(_attention_fwd, _attention_bwd)
```

**Common Mistakes**:

1. **Forgetting to save `lse` (log-sum-exp)**:
   - Forward: `lse = m_i + log2(l_i)` (after final iteration)
   - Backward: Needed to recompute `P = exp(S - lse)` without materializing full attention matrix
   - Shape: `(batch, num_q_heads, q_seq_len)` (note: heads and seq swapped vs Q/K/V)

2. **Wrong `delta` computation**:
   - Correct: `delta[i] = sum_d(dO[i,d] * O[i,d])` (per-token scalar)
   - Wrong: `delta = dO @ O.T` (produces matrix, not vector)
   - Use in backward: `dS[i,j] = P[i,j] * (dP[i,j] - delta[i])`

3. **Mismatched tensor layouts**:
   - Forward: Q/K/V are `(batch, seq, heads, dim)`
   - Backward: lse/delta are `(batch, heads, seq)` (heads and seq swapped!)
   - Solution: Use `Layout.WGMMA_ROW` vs `Layout.WGMMA_COL` when loading scalars

4. **Not handling GQA (Grouped Query Attention)**:
   - Forward: `num_kv_heads < num_q_heads`, each KV head serves multiple Q heads
   - Backward: Each KV tile receives gradients from **all** Q heads in its group
   - Must accumulate: `dk[kv_head] = sum(dq[q_head * q_heads_per_kv_head : (q_head+1) * q_heads_per_kv_head])`

5. **Incorrect barrier counts in dual kernels**:
   - dQ kernel: pipelines over KV (like forward)
   - dK/dV kernel: pipelines over Q (transposed!)
   - Each needs its own barrier allocation matching its grid size
   - Don't reuse forward pass barriers (different tile counts)

**Testing Backward Passes**:
```python
def test_attention_gradients():
    """Verify custom VJP matches autodiff."""
    def attention_reference(q, k, v):
        """Reference implementation (no custom_vjp)."""
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum('bhqk,bkhd->bqhd', weights, v)
    
    # Test forward
    out_custom = attention(q, k, v, config)
    out_ref = attention_reference(q, k, v)
    np.testing.assert_allclose(out_custom, out_ref, rtol=1e-3, atol=1e-3)
    
    # Test backward
    def loss(fn, q, k, v):
        return fn(q, k, v).sum()
    
    grad_custom = jax.grad(loss, argnums=(0, 1, 2))(attention, q, k, v, config)
    grad_ref = jax.grad(loss, argnums=(0, 1, 2))(attention_reference, q, k, v)
    
    for g_custom, g_ref in zip(grad_custom, grad_ref):
        np.testing.assert_allclose(g_custom, g_ref, rtol=1e-2, atol=1e-2)
```

---

## Example: Optimized Matmul (Blackwell)

Complete example from Pallas docs (simplified):

```python
def matmul_blackwell(a, b, config: TuningConfig):
    m, k = a.shape
    _, n = b.shape
    tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
    cluster_tile_m, cluster_tile_n = 2 * tile_m, 2 * tile_n
    
    def kernel(a_gmem, b_gmem, out_gmem, a_smem, b_smem, acc_tmem, ...):
        wg_idx = lax.axis_index("wg")
        is_lead_block = lax.axis_index("cluster") == 0
        
        @plgpu.nd_loop((m // cluster_tile_m, n // cluster_tile_n))
        def mn_loop(loop_info):
            m_idx, n_idx = loop_info.index
            acc_slot = loop_info.local_index % 2
            
            # Compute with warp specialization
            @pl.when(wg_idx == 0)
            def compute_wg():
                @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
                def per_warp():
                    warp_id = lax.axis_index("warp")
                    
                    @pl.when(warp_id == 0)
                    def memory():
                        for ki in range(k // tile_k):
                            slot = ki % max_steps
                            plgpu.copy_gmem_to_smem(
                                a_gmem.at[...], a_smem.at[slot], load_barriers.at[slot],
                                collective_axes="cluster", partitioned_axis=0
                            )
                            plgpu.copy_gmem_to_smem(
                                b_gmem.at[...], b_smem.at[slot], load_barriers.at[slot],
                                collective_axes="cluster", partitioned_axis=1
                            )
                    
                    @pl.when(warp_id == 1 and is_lead_block)
                    def compute():
                        for ki in range(k // tile_k):
                            slot = ki % max_steps
                            plgpu.barrier_wait(load_barriers.at[slot])
                            plgpu.tcgen05_mma(
                                acc_tmem.at[:, acc_slot * cluster_tile_n:...],
                                a_smem.at[slot], b_smem.at[slot],
                                consumed_barriers.at[slot],
                                accumulate=(ki > 0),
                                collective_axis="cluster"
                            )
            
            # Epilogue with second warpgroup
            @pl.when(wg_idx == 1)
            def store_wg():
                plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
                for ni in range(cluster_tile_n // epilogue_tile_n):
                    ni_slice = pl.ds(ni * epilogue_tile_n, epilogue_tile_n)
                    acc_smem.at[ni % 2] = plgpu.async_load_tmem(acc_tmem.at[:, ni_slice])
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(acc_smem.at[ni % 2], out_gmem.at[...])
                plgpu.wait_load_tmem()
    
    num_sms = jax.extend.backend.get_default_device().core_count
    return plgpu.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=(num_sms // 2,),
        cluster=(2,),
        num_threads=2,
        scratch_shapes={...}
    )(a, b)
```

**Key Techniques:**
1. Warp specialization (memory + compute warpgroups)
2. Collective MMA (2 SMs share B operand)
3. Persistent kernel (fewer blocks than work)
4. Double-buffered TMEM accumulators
5. Grid tiling for L2 locality

---

## Resources

- **Pallas Quickstart**: https://docs.jax.dev/en/latest/pallas/quickstart.html
- **Mosaic GPU Reference**: https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- **Pipelining Guide**: https://docs.jax.dev/en/latest/pallas/pipelining.html
- **Blackwell Matmul Tutorial**: https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html
- **Design Doc**: https://docs.jax.dev/en/latest/pallas/design/design.html

---

**Last Updated**: November 18, 2025
