# Pallas GPU Kernel Glossary

**Terminology reference for JAX Pallas GPU kernel programming.**

---

## Core Concepts

### **Pallas**
JAX extension for writing custom GPU and TPU kernels using a Triton-like programming model. Provides low-level hardware access while maintaining JAX composability.

### **Ref** (Reference)
Mutable memory handle used in kernels (vs immutable JAX arrays). Syntax: `value = ref[:]` (read), `ref[:] = value` (write). Enables in-place operations required for performance on GPU/TPU.

### **pallas_call**
Higher-order function to invoke a kernel function over a grid. Signature:
```python
pl.pallas_call(
    kernel_fn,
    out_shape,
    grid=(M, N),
    in_specs=[...],
    out_specs=...
)
```

### **Grid**
Iteration space defining how many times the kernel executes. Example: `grid=(4, 8)` runs kernel 32 times with indices (0,0), (0,1), ..., (3,7). Can run in parallel or sequentially depending on hardware.

### **BlockSpec**
Specification for how to slice inputs/outputs for each grid iteration. Components:
- `block_shape`: Shape of the slice
- `index_map`: Function mapping grid indices to data indices
- `memory_space`: Where to place data (GMEM, SMEM)

---

## Memory Hierarchy

### **GMEM** (Global Memory)
Main GPU memory (HBM). Slowest (2-3 TB/s) but largest (10s-100s GB). Use for primary storage of inputs/outputs. Access pattern: `plgpu.GMEM(shape, dtype)`.

### **SMEM** (Shared Memory)
On-chip shared memory visible to all threads in a block (~100MB total, ~64KB per SM). Fast (~10 TB/s) but limited. Use for shared data, TensorCore operands. Access: `plgpu.SMEM(shape, dtype)`.

### **TMEM** (Tensor Memory) — Blackwell only
Explicit register space for TensorCore accumulators. Same speed as registers but separate from general-purpose registers. Enables overlapping compute and epilogue. Access: `plgpu.TMEM(shape, dtype, collective=True)`.

### **Registers**
Fastest memory (~20 TB/s), private to each thread. Typically implicit (loaded via `ref[:]`), but can specify via `memory_space=plgpu.Registers`.

### **L2 Cache**
Hardware-managed cache between SMEM and GMEM (~50-100MB). Transparent to programmer but critical for performance (optimize access patterns for L2 reuse).

---

## GPU Hardware

### **SM** (Streaming Multiprocessor)
Independent GPU compute unit. Hopper H100 has 132 SMs. Each SM has:
- Shared memory / L1 cache
- Multiple warpgroups
- Register file
- Scheduling units

### **Warp**
Group of 32 CUDA threads executing in lockstep (SIMT). Single instruction stream, multiple data. Threads within warp share instruction fetch/decode.

### **Warpgroup** (Hopper+)
Collection of 4 warps (128 threads) that can collaborate. Unit of work for TensorCore operations (`wgmma`). Blackwell uses 2 warpgroups (256 threads) for `tcgen05_mma`.

### **Block** (Thread Block)
Kernel invocation launched for one grid cell. Contains 1-4 warpgroups (up to 512 threads on Hopper). In Pallas: controlled by `num_threads` parameter in `plgpu.kernel()`.

### **Cluster** (Thread Block Cluster)
Group of 2-8 blocks sharing resources (SMEM via distributed shared memory). Enables collective operations. In Pallas: `cluster=(2,)` creates 2-block cluster.

---

## Synchronization

### **Barrier**
Synchronization primitive for coordinating threads/operations. Usage:
- `plgpu.barrier_arrive()`: Signal completion
- `plgpu.barrier_wait()`: Wait for all arrivals
- Critical: equal arrivals and waits over barrier's lifetime

### **ClusterBarrier**
Barrier spanning all blocks in a cluster. Required for synchronizing collective operations. Allocated with `num_arrivals=cluster_size`.

### **Semaphore**
Cross-device synchronization for multi-GPU communication. Types:
- `Semaphore.REGULAR`: Basic signaling
- `Semaphore.MEMORY`: Synchronizes memory visibility across devices

### **plgpu.commit_smem()**
Memory fence ensuring SMEM writes are visible to async hardware (TMA, TensorCore). Required before async copies reading updated SMEM data.

---

## TensorCore Operations

### **TensorCore**
Specialized matrix multiply hardware. 4th-gen on Hopper, 5th-gen on Blackwell. Operates asynchronously (issue operation, wait later). Peak: ~1000 TFLOPS on H100.

### **wgmma** (Warpgroup Matrix Multiply-Accumulate) — Hopper
TensorCore operation for matmul. Syntax:
```python
plgpu.wgmma(acc_ref, a_smem, b_smem)
plgpu.wgmma_wait(0)  # Wait for completion
```
Requirements: M%64, N%8, K divisible by swizzle, operands in SMEM with transforms.

### **tcgen05_mma** (5th-gen TensorCore MMA) — Blackwell
Next-gen TensorCore operation supporting collective MMA (2 SMs share operand) and TMEM accumulators. Syntax:
```python
plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, barrier, collective_axis="cluster")
plgpu.barrier_wait(barrier)
```

### **ACC** / **Accumulator**
Register space for TensorCore results. Hopper: implicit (part of RF). Blackwell: explicit TMEM. Layout: `plgpu.Layout.WGMMA` or `plgpu.Layout.WGMMA_ROW_MAJOR`.

---

## Memory Transforms

### **TilingTransform**
Reshapes data to improve TensorCore efficiency. Example: `TilingTransform((8, 64))` converts `(128, 128)` → `(16, 2, 8, 64)`. Aligns with wgmma input requirements.

### **SwizzleTransform**
Reorders SMEM addresses to avoid bank conflicts. Example: `SwizzleTransform(128)` for 128-byte swizzle. Common sizes: 32, 64, 128 bytes. Apply after tiling.

### **Transforms Pipeline**
Always apply in order: `TilingTransform` → `SwizzleTransform`. Specified in `BlockSpec`:
```python
transforms = (
    plgpu.TilingTransform((8, swizzle // 2)),
    plgpu.SwizzleTransform(swizzle)
)
```

---

## Asynchronous Operations

### **TMA** (Tensor Memory Accelerator)
Hardware unit for async GMEM ↔ SMEM copies. Operates independently of thread execution. Functions:
- `plgpu.copy_gmem_to_smem(src, dst, barrier)`
- `plgpu.copy_smem_to_gmem(src, dst)`

### **Asynchronous Copy**
Non-blocking data transfer overlapping with compute. Pattern:
1. Issue copy (`copy_gmem_to_smem`)
2. Do other work
3. Wait for completion (`barrier_wait`)

### **Collective Copy**
TMA copy where multiple blocks in cluster share operand. Modes:
- **Multicast**: All blocks get identical data
- **Partitioned**: Each block gets a slice (via `partitioned_axis`)

### **Copy Ordering**
- `wait_smem_to_gmem(N)`: Wait until at most N stores remain
- `wait_read_only=True`: Only wait for read-after-write hazards, allow write-after-write to proceed

---

## Pipelining

### **emit_pipeline**
High-level API for overlapping compute and memory. Automatically manages:
- Double/triple buffering
- Barrier allocation
- Prologue/epilogue insertion
Example:
```python
plgpu.emit_pipeline(
    step_fn,
    grid=(K_iters,),
    in_specs=[...],
    max_concurrent_steps=2
)
```

### **emit_pipeline_warp_specialized**
Pipeline with dedicated memory warp(s) and compute warp(s). Memory warp fetches data while compute warp issues TensorCore operations. Enables deeper pipelines (6+ stages).

### **max_concurrent_steps**
Number of overlapping pipeline iterations (buffering factor). Higher = more overlap but more registers/SMEM. Typical: 2-6.

### **delay_release**
For async ops (WGMMA, collective MMA), tells pipeline not to overwrite buffer until N steps later. Example: `delay_release=1` means don't reuse buffer until step `i+1` completes.

### **Software Pipelining**
Technique to overlap memory latency with compute. Pattern:
```
for i:
    if i+1 < N: prefetch(i+1)
    compute(i)
```

---

## Program Querying

### **pl.program_id(axis)**
Returns grid index for current kernel invocation along `axis` dimension. Example: for `grid=(M, N)`, `program_id(0)` ∈ [0, M).

### **pl.num_programs(axis)**
Returns total grid size along `axis`. Example: for `grid=(4, 8)`, `num_programs(1)` = 8.

### **lax.axis_index(name)**
Multi-controller index (for multi-GPU, warp specialization). Example: `lax.axis_index("cluster")` for block index within cluster.

### **lax.axis_size(name)**
Multi-controller size. Example: `lax.axis_size("wg")` for number of warpgroups.

---

## Control Flow

### **@pl.when(condition)**
Conditional execution in kernel. Only threads where `condition=True` execute body. Example:
```python
@pl.when(pl.program_id(0) == 0)
def first_block():
    # Only runs for block 0
```

### **@pl.core_map(mesh)**
Map function over mesh axes (warps, blocks). Example:
```python
@pl.core_map(plgpu.WarpMesh(axis_name="warp"))
def per_warp():
    warp_id = lax.axis_index("warp")
```

### **@plgpu.nd_loop(shape)**
Loop over multi-dimensional iteration space. Supports collective axes (distribute work across blocks/warps). Example:
```python
@plgpu.nd_loop((M, N), collective_axes="grid")
def loop_body(loop_info):
    i, j = loop_info.index  # Current iteration
    slot = loop_info.local_index % 2  # For buffer rotation
```

### **@plgpu.dynamic_scheduling_loop**
Work-stealing loop for load balancing (Blackwell). Hardware dynamically assigns iterations to blocks. Use for irregular workloads.

---

## Advanced Patterns

### **Warp Specialization**
Assign different tasks to different warps (memory vs compute). Improves occupancy by freeing compute warp threads from memory operations. Implemented via `emit_pipeline_warp_specialized` or manual `core_map`.

### **Persistent Kernel**
Launch fewer blocks than work items, each block processes multiple tiles. Amortizes initialization costs, enables overlapping epilogue with next iteration. Use `nd_loop` with `collective_axes="grid"`.

### **Grid Tiling**
Reorder grid traversal to improve L2 cache hit rate. Use `plgpu.planar_snake()` for snake-pattern iteration. Example: process blocks (0,0)-(0,7), (1,7)-(1,0), (2,0)-(2,7) instead of row-major order.

### **Collective MMA**
TensorCore operation where 2 SMs collaborate on single matmul, sharing one operand. Doubles effective TensorCore utilization. Blackwell-specific via `tcgen05_mma(..., collective_axis="cluster")`.

### **Double Buffering**
Allocate 2 buffers, fetch into one while computing on the other. Classic pipelining technique. Use `max_concurrent_steps=2` in `emit_pipeline`.

---

## Backends

### **Triton**
GPU backend for Pallas (Nvidia/AMD). Lower-level, older, supports Hopper but not all Blackwell features. Use for broad compatibility.

### **Mosaic GPU**
Newer GPU backend for Pallas. Required for Blackwell features (TMEM, tcgen05_mma, ClusterBarrier). Import: `jax.experimental.pallas.mosaic_gpu as plgpu`.

### **Mosaic (TPU)**
TPU backend for Pallas. Different memory model (HBM, VMEM, SMEM). Not covered in this GPU glossary.

---

## Shapes & Indexing

### **pl.ds (dynamic slice)**
Slice specification for BlockSpec. Syntax: `pl.ds(start, size)`. Example: `x_ref[pl.ds(i*128, 128)]` loads 128 elements starting at `i*128`.

### **pl.Element**
Marks BlockSpec dimension as element-wise indexing (not block indexing). Use when `index_map` returns element offsets. Example:
```python
pl.BlockSpec((pl.Element(128),), lambda i: (i * 128,))
```

### **Block Shape**
Tile size processed by each kernel invocation. Example: `(128, 128)` for 128×128 output tile. Trade-off: larger = more reuse but more registers/SMEM.

---

## Debugging

### **interpret=True**
Run kernel on CPU without JIT compilation. Enables standard Python debugging (breakpoints, print). Use for development, switch to `interpret=False` for production.

### **jax.debug.print**
Print from inside kernel. Syntax: `jax.debug.print("Value: {}", x)`. Works in JIT-compiled code (outputs to stdout).

### **MOSAIC_GPU_DUMP_***
Environment variables to dump compilation artifacts:
- `MOSAIC_GPU_DUMP_PTX=1`: PTX intermediate representation
- `MOSAIC_GPU_DUMP_SASS=1`: Final assembly code
- `MOSAIC_GPU_DUMP_PTXAS=1`: Compilation logs (register usage, spills)

### **optimization_barrier**
Prevent compiler from reordering operations across boundary. Use to enforce ordering of async ops. Syntax: `value = pl.optimization_barrier(value)`.

---

## Multi-GPU

### **plgpu.remote_ref**
Reference to memory on another GPU. Syntax: `remote = plgpu.remote_ref(local_ref, device_id)`. Enables direct NVLINK writes.

### **pl.semaphore_signal / pl.semaphore_wait**
Cross-device synchronization. Pattern:
```python
pl.semaphore_signal(sem, device_id=other_dev)  # Signal neighbor
pl.semaphore_wait(sem)                         # Wait for neighbor's signal
```

### **NVLINK**
Direct GPU-to-GPU interconnect (up to 900 GB/s on H100). Used by collective copies and remote_ref. Bypass host CPU for efficient multi-GPU communication.

### **Ring All-Gather**
Communication pattern where each GPU sends to neighbor in ring topology. Use with matmul to overlap communication with compute. Example: compute on local shard while receiving next shard.

---

## Performance Metrics

### **Occupancy**
Fraction of GPU resources (SMs, warps) actively used. Higher = better utilization. Trade-off: larger tiles use more SMEM/registers → lower occupancy but better per-block efficiency.

### **Compute Intensity**
FLOPs per byte transferred. Goal: maximize reuse of data fetched from GMEM. Matmul with tiling: O(B³) FLOPs for O(B²) data → intensity = O(B).

### **Roofline**
Performance model: `achieved_FLOPS = min(peak_FLOPS, bandwidth * intensity)`. Memory-bound: limited by bandwidth. Compute-bound: limited by FLOPS.

### **cuBLAS**
Nvidia's optimized BLAS library (baseline for matmul performance). Pallas kernels aim for 60-100% of cuBLAS on matmul. Example: Blackwell tutorial achieves 69% with warp specialization + tiling.

---

## Architecture Specifics

### **Hopper (H100)**
Nvidia GPU architecture (2022). Features:
- 4th-gen TensorCore (wgmma)
- Thread block clusters
- TMA for async copies
- 80-90GB HBM3

### **Blackwell (B100)**
Next-gen Nvidia architecture (2024). Additions:
- 5th-gen TensorCore (tcgen05_mma)
- Explicit TMEM
- Collective MMA (2 SMs)
- Improved pipelining

---

## Configuration Types

### **TuningConfig**
Hyperparameters for kernel. Example fields:
- `tile_m`, `tile_n`, `tile_k`: Block sizes
- `max_concurrent_steps`: Pipeline depth
- `grid_tile_width`: L2 tiling parameter

### **scratch_shapes**
Temporary buffers allocated by Pallas. Dictionary mapping names to memory space + shape. Example:
```python
scratch_shapes = {
    'smem_buf': plgpu.SMEM((128, 128), jnp.float16),
    'barrier': plgpu.Barrier(num_barriers=2)
}
```

### **Layout**
Data layout in memory. Types:
- `plgpu.Layout.WGMMA`: TensorCore-compatible layout
- `plgpu.Layout.WGMMA_ROW_MAJOR`: Row-major variant

---

## JAX Integration

### **jax.jit + pallas_call**
Pallas kernels compose with JAX transformations. Example:
```python
@jax.jit
def model(x):
    x = pl.pallas_call(custom_kernel, ...)(x)
    return jax.nn.relu(x)
```

### **jax.grad**
Pallas kernels support automatic differentiation (with limitations). Forward mode usually works; reverse mode requires careful state management.

### **jax.vmap**
Batch parallelism over Pallas kernels. Example: `jax.vmap(pallas_fn)(batched_input)` adds batch dimension to grid.

### **jax.shard_map**
Multi-device mapping (for multi-GPU). Syntax:
```python
jax.shard_map(pallas_fn, mesh=mesh, in_specs=P("x"), out_specs=P("x"))
```

---

## Common Acronyms

- **MMA**: Matrix Multiply-Accumulate
- **TMA**: Tensor Memory Accelerator
- **CTA**: Cooperative Thread Array (thread block)
- **SMEM**: Shared Memory
- **GMEM**: Global Memory
- **HBM**: High Bandwidth Memory
- **RF**: Register File
- **SM**: Streaming Multiprocessor
- **PTX**: Parallel Thread Execution (Nvidia IR)
- **SASS**: Shader Assembly (Nvidia machine code)

---

**Last Updated**: November 18, 2025
