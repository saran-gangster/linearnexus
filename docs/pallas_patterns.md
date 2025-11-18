# Pallas Common Patterns & Recipes

**Practical patterns for GPU kernel development with JAX Pallas.**

---

## Table of Contents

1. [Kernel Templates](#kernel-templates)
2. [Memory Management](#memory-management)
3. [Pipelining Patterns](#pipelining-patterns)
4. [Synchronization Patterns](#synchronization-patterns)
5. [Performance Optimization](#performance-optimization)
6. [Multi-GPU Patterns](#multi-gpu-patterns)

---

## Kernel Templates

### Element-wise Operation

```python
def elementwise_kernel_template(activation_fn):
    """Template for fused elementwise kernels."""
    def kernel(x_ref, y_ref, o_ref):
        x = x_ref[:]
        y = y_ref[:]
        result = activation_fn(x + y)
        o_ref[:] = result
    return kernel

# Usage
relu_add = elementwise_kernel_template(jax.nn.relu)
gelu_add = elementwise_kernel_template(jax.nn.gelu)

@jax.jit
def fused_add_relu(x, y):
    return pl.pallas_call(
        relu_add,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)
```

### Reduction Kernel

```python
def reduction_kernel(x_ref, o_ref):
    """Reduce over last grid dimension."""
    # Initialize on first iteration
    @pl.when(pl.program_id(2) == 0)
    def init():
        o_ref[...] = jnp.zeros_like(o_ref)
    
    # Accumulate
    o_ref[...] += x_ref[...]

def reduce_sum(x, block_size=(256, 256)):
    reduction_size, *out_shape = x.shape
    grid = (*[s // b for s, b in zip(out_shape, block_size)], reduction_size)
    
    return pl.pallas_call(
        reduction_kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
        grid=grid,
        in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
        out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j))
    )(x)
```

### Parameterized Kernel

```python
from functools import partial

def matmul_kernel(x_ref, y_ref, o_ref, *, activation, block_k):
    """Matmul with fused activation."""
    acc = jnp.zeros((x_ref.shape[0], y_ref.shape[1]), jnp.float32)
    
    # Reduction over K dimension
    for k in range(x_ref.shape[1] // block_k):
        x = x_ref[:, k*block_k:(k+1)*block_k]
        y = y_ref[k*block_k:(k+1)*block_k, :]
        acc += x @ y
    
    o_ref[:, :] = activation(acc).astype(o_ref.dtype)

def matmul_fused(x, y, activation=jax.nn.relu, block_shape=(128, 128, 64)):
    block_m, block_n, block_k = block_shape
    
    return pl.pallas_call(
        partial(matmul_kernel, activation=activation, block_k=block_k),
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        grid=(x.shape[0] // block_m, y.shape[1] // block_n),
        in_specs=[
            pl.BlockSpec((block_m, x.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((y.shape[0], block_n), lambda i, j: (0, j))
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j))
    )(x, y)
```

---

## Memory Management

### Buffer Reuse with Barriers

```python
def pipelined_kernel_with_reuse(x_gmem, o_gmem, buffers, barriers):
    """Triple-buffered pipeline."""
    num_steps = 100
    buffering = 3
    
    def fetch(i, slot):
        slice_i = pl.ds(i * tile_size, tile_size)
        plgpu.copy_gmem_to_smem(
            x_gmem.at[slice_i],
            buffers.at[slot],
            barriers.at[slot]
        )
    
    # Prologue: fill pipeline
    for slot in range(buffering):
        fetch(slot, slot)
    
    def body(i, _):
        slot = jax.lax.rem(i, buffering)
        
        # Wait for this buffer to be ready
        plgpu.barrier_wait(barriers.at[slot])
        
        # Consume buffer
        process(buffers.at[slot])
        
        # Fetch next (if not done)
        next_i = i + buffering
        @pl.when(next_i < num_steps)
        def _():
            fetch(next_i, slot)
    
    jax.lax.fori_loop(0, num_steps, body, None)

# Allocate scratch
scratch_shapes = {
    'buffers': plgpu.SMEM((3, tile_size), dtype),  # Triple buffer
    'barriers': plgpu.Barrier(num_barriers=3)
}
```

### TMEM Management (Blackwell)

```python
def tmem_double_buffer(acc_tmem_slots, mma_barrier, store_barrier):
    """Double-buffered TMEM for overlapping MMA and epilogue."""
    
    @plgpu.nd_loop((num_output_tiles,))
    def tile_loop(loop_info):
        acc_slot = loop_info.local_index % 2
        acc_tmem = acc_tmem_slots.at[:, acc_slot * tile_n:(acc_slot+1) * tile_n]
        
        # Wait for previous epilogue to finish (except first 2 tiles)
        @pl.when(loop_info.local_index >= 2)
        def _():
            plgpu.barrier_wait(store_barrier.at[acc_slot])
        
        # Compute into this TMEM half
        for ki in range(k_iters):
            plgpu.tcgen05_mma(acc_tmem, a_smem, b_smem, ...)
        plgpu.tcgen05_commit_arrive(mma_barrier.at[acc_slot])
        
        # Epilogue: can overlap with next tile's MMA
        plgpu.barrier_wait(mma_barrier.at[acc_slot])
        result = plgpu.async_load_tmem(acc_tmem)
        # ... store result ...
        plgpu.wait_load_tmem()
        plgpu.barrier_arrive(store_barrier.at[acc_slot])

# Allocate
scratch_shapes = {
    'acc_tmem': plgpu.TMEM((tile_m, 2 * tile_n), jnp.float32),  # Double-buffered
    'mma_barrier': plgpu.Barrier(num_barriers=2, orders_tensor_core=True),
    'store_barrier': plgpu.ClusterBarrier(num_barriers=2, orders_tensor_core=True)
}
```

---

## Pipelining Patterns

### Basic Double-Buffered Pipeline

```python
def pipeline_template(a_gmem, b_gmem, o_gmem, acc):
    """Standard compute/memory pipeline."""
    def step(ki, a_smem, b_smem):
        # Compute on data loaded in previous step
        plgpu.wgmma(acc, a_smem, b_smem)
        plgpu.wgmma_wait(1)  # Wait for previous WGMMA
    
    plgpu.emit_pipeline(
        step,
        grid=(k_iters,),
        in_specs=[
            plgpu.BlockSpec((tile_m, tile_k), lambda ki: (pid_m, ki), transforms=...),
            plgpu.BlockSpec((tile_k, tile_n), lambda ki: (ki, pid_n), transforms=...)
        ],
        max_concurrent_steps=2,  # Double-buffer
        delay_release=1          # Don't overwrite until compute done
    )
```

### Warp-Specialized Pipeline

```python
def warp_specialized_template(a_gmem, b_gmem, o_gmem, o_smem):
    """Dedicate warps to memory vs compute."""
    
    def compute_context(pipeline):
        # Initialize accumulator in compute threads only
        acc = plgpu.layout_cast(
            jnp.full((tile_m, tile_n), 0, jnp.float32),
            plgpu.Layout.WGMMA
        )
        final_acc = pipeline(acc)  # Run pipeline with carry
        o_smem[:, wg_slice] = final_acc.astype(dtype)
    
    def step(ki, a_smem, b_smem, acc_carry):
        # Compute threads process data fetched by memory thread
        def do_wgmma(acc_ref):
            plgpu.wgmma(acc_ref, a_smem, b_smem.at[:, wg_slice])
        return pl.run_state(do_wgmma)(plgpu.ACC.init(acc_carry))
    
    pipeline = plgpu.emit_pipeline_warp_specialized(
        step,
        grid=(k_iters,),
        in_specs=[...],
        compute_context=compute_context,
        max_concurrent_steps=2,
        num_compute_wgs=2,      # 2 compute warps
        memory_registers=40,    # Registers for memory warp
        wg_axis="wg"
    )
    pipeline(a_gmem, b_gmem)

# Call with num_threads=3 (2 compute + 1 memory)
plgpu.kernel(kernel, num_threads=3, thread_name="wg")(...)
```

### Pipeline with Callback

```python
def pipeline_with_callback(a_gmem, b_gmem, send_ref):
    """Inject custom logic into pipeline steps."""
    
    def send_tile(m_idx, n_idx, k_idx, a_smem, b_smem):
        # Send left operand to neighbor device while computing
        @pl.when(n_idx == 0)  # Only send once per left operand
        def _():
            m_slice = pl.ds(m_idx * tile_m, tile_m)
            k_slice = pl.ds(k_idx * tile_k, tile_k)
            plgpu.copy_smem_to_gmem(a_smem, send_ref.at[m_slice, k_slice])
            plgpu.wait_smem_to_gmem(1, wait_read_only=True)
    
    hopper_matmul_mgpu.kernel(
        a_gmem, b_gmem, o_gmem, ...,
        pipeline_callback=functools.partial(send_tile, send_ref=send_ref),
        delay_release=1
    )
```

---

## Synchronization Patterns

### Producer-Consumer with Barriers

```python
def producer_consumer(queue, produced, consumed):
    """Two threads communicate via shared buffer."""
    tid = lax.axis_index("thread")
    buffering = 3
    num_items = 100
    
    @pl.when(tid == 0)
    def producer():
        def body(i, _):
            slot = jax.lax.rem(i, buffering)
            
            # Wait for consumer to finish with this slot
            @pl.when(i >= buffering)
            def _():
                plgpu.barrier_wait(consumed.at[slot])
            
            # Produce item
            queue[slot] = generate_item(i)
            plgpu.barrier_arrive(produced.at[slot])
        
        jax.lax.fori_loop(0, num_items, body, None)
    
    @pl.when(tid == 1)
    def consumer():
        def body(i, _):
            slot = jax.lax.rem(i, buffering)
            
            # Wait for item to be ready
            plgpu.barrier_wait(produced.at[slot])
            
            # Consume
            process(queue[slot])
            plgpu.barrier_arrive(consumed.at[slot])
        
        jax.lax.fori_loop(0, num_items, body, None)

# Allocate
scratch_shapes = {
    'queue': plgpu.SMEM((3, item_size), dtype),
    'produced': plgpu.Barrier(num_barriers=3),
    'consumed': plgpu.Barrier(num_barriers=3)
}
```

### Cluster-Wide Synchronization

```python
def cluster_sync_pattern(acc_tmem, data_smem, cluster_barrier):
    """Synchronize resource reuse across cluster."""
    
    # Block 0 issues collective MMA for both blocks
    is_leader = lax.axis_index("cluster") == 0
    @pl.when(is_leader)
    def _():
        plgpu.tcgen05_mma(
            acc_tmem, a_smem, b_smem, mma_barrier,
            collective_axis="cluster"
        )
    
    # Both blocks wait for MMA
    plgpu.barrier_wait(mma_barrier)
    
    # Both blocks read their share of accumulator
    local_result = plgpu.async_load_tmem(acc_tmem)
    do_something(local_result)
    plgpu.wait_load_tmem()
    
    # Cluster barrier: ensure both blocks done reading before reusing TMEM
    plgpu.barrier_arrive(cluster_barrier)
    plgpu.barrier_wait(cluster_barrier)
    
    # Now safe to reuse acc_tmem for next MMA
```

---

## Performance Optimization

### Grid Tiling for L2 Locality

```python
def optimized_grid_order(m_iters, n_iters, config):
    """Snake pattern to improve L2 hit rate."""
    
    @plgpu.nd_loop((m_iters * n_iters,), collective_axes="grid")
    def tile_loop(loop_info):
        (linear_idx,) = loop_info.index
        
        # Convert linear index to 2D with snake pattern
        m_idx, n_idx = plgpu.planar_snake(
            linear_idx,
            grid_shape=(m_iters, n_iters),
            minor_dim=config.grid_minor_dim,  # 0 or 1
            tile_width=config.grid_tile_width # e.g., 8
        )
        
        # Process tile (m_idx, n_idx)
        # Adjacent tiles in iteration order now access similar data
```

**Explanation**: Without tiling, blocks (0,0)-(0,N-1) run before (1,0), so LHS blocks can't be reused from L2. With tiling (width=8), we process (0,0)-(0,7), (1,7)-(1,0), (2,0)-(2,7), etc., keeping both operands warm in L2.

### Occupancy vs Throughput Trade-offs

```python
# Low occupancy (more SMEM per block) → better for memory-bound kernels
config_high_smem = TuningConfig(
    tile_m=256, tile_n=256, tile_k=128,
    max_concurrent_steps=6  # More buffers = more SMEM
)

# High occupancy (less SMEM per block) → better for compute-bound kernels
config_low_smem = TuningConfig(
    tile_m=128, tile_n=128, tile_k=64,
    max_concurrent_steps=2  # Fewer buffers = less SMEM
)

# Rule of thumb: if kernel has register spills, reduce max_concurrent_steps
```

### Persistent Kernels

```python
def persistent_kernel_pattern():
    """Launch grid < work size, loop over tiles per block."""
    
    num_sms = jax.extend.backend.get_default_device().core_count
    work_items = (m_iters, n_iters)
    
    # Launch one block per SM (or per cluster)
    grid = (num_sms // cluster_size,)
    
    def kernel(...):
        # Each block processes multiple work items
        @plgpu.nd_loop(work_items, collective_axes="grid")
        def tile_loop(loop_info):
            m_idx, n_idx = loop_info.index
            # Process tile
            ...
    
    return plgpu.kernel(
        kernel,
        grid=grid,
        cluster=(cluster_size,),
        ...
    )(...)
```

**Benefits**:
- Amortize block initialization costs
- Enable overlapping epilogue with next tile's compute
- Better for small tiles or many tiles

### Dynamic Work Scheduling (Blackwell)

```python
def dynamic_scheduling_pattern():
    """Work stealing for load balancing."""
    
    @plgpu.dynamic_scheduling_loop(
        grid_names=("m", "n"),
        thread_axis="wg"  # Required for multi-threaded kernels
    )
    def tile_body(loop_info):
        m_idx, n_idx = loop_info.index
        # Process tile dynamically assigned by hardware
        ...
    
    # Grid specifies logical work size, not physical blocks launched
    plgpu.kernel(
        tile_body,
        grid=(m_iters, n_iters),  # Logical work
        ...
    )(...)
```

**Use case**: Uneven work distribution (e.g., sparse matrices, dynamic shapes).

---

## Multi-GPU Patterns

### Ring All-Gather with Compute Overlap

```python
def all_gather_matmul(lhs, rhs, axis_name):
    """AllGather(lhs) @ rhs with overlapped communication."""
    
    def kernel(lhs_local, rhs, out, scratch, done_sem):
        dev_id = lax.axis_index(axis_name)
        num_devs = lax.axis_size(axis_name)
        send_to = (dev_id - 1) % num_devs
        
        def step(i, lhs_source):
            # Compute with current shard
            out_slice = pl.ds((dev_id + i) % num_devs * m_shard, m_shard)
            matmul_block(lhs_source, rhs, out.at[out_slice])
            
            # Send to next device (while computing)
            @pl.when(i + 1 < num_devs)
            def _():
                next_slot = i
                send_ref = plgpu.remote_ref(scratch.at[next_slot], send_to)
                async_send(lhs_source, send_ref)
                pl.semaphore_signal(done_sem, device_id=send_to)
            
            # Wait for next shard to arrive
            @pl.when(i + 1 < num_devs)
            def _():
                pl.semaphore_wait(done_sem, value=(i+1) * num_sms, decrement=False)
        
        # First step: use local shard
        step(0, lhs_local)
        
        # Remaining steps: use received shards
        for i in range(1, num_devs):
            step(i, scratch.at[i-1])
    
    return jax.shard_map(
        lambda lhs: plgpu.kernel(
            kernel,
            out_shape=[
                jax.ShapeDtypeStruct((num_devs * m_shard, n), dtype),
                jax.ShapeDtypeStruct((num_devs - 1, m_shard, k), dtype)  # scratch
            ],
            scratch_shapes={'done_sem': plgpu.Semaphore.REGULAR}
        )(lhs, rhs),
        mesh=mesh,
        in_specs=P("x", None),
        out_specs=P(None, "x"),
        check_vma=False
    )(lhs)
```

### NVLINK Direct Memory Access

```python
def nvlink_exchange(x_ref, y_ref, done_sem):
    """Exchange data between GPUs over NVLINK."""
    other_dev = 1 - lax.axis_index("x")
    
    # Get reference to neighbor's memory
    neighbor_y = plgpu.remote_ref(y_ref, other_dev)
    
    # Asynchronous write to remote device
    plgpu.copy_smem_to_gmem(x_smem, neighbor_y)
    plgpu.wait_smem_to_gmem(0)
    
    # Signal completion
    pl.semaphore_signal(done_sem, device_id=other_dev)
    
    # Wait for neighbor to write to our memory
    pl.semaphore_wait(done_sem)
```

---

## Advanced Patterns

### Custom Type Support

```python
# Use pytree to pass custom types through pallas_call
@dataclass
class MyState:
    weights: jax.Array
    bias: jax.Array

def kernel(state_refs, x_ref, o_ref):
    # state_refs is a pytree of Refs
    w = state_refs.weights[:]
    b = state_refs.bias[:]
    x = x_ref[:]
    o_ref[:] = x @ w + b

def apply(state: MyState, x):
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], state.bias.shape[0]), x.dtype),
        # in_specs is a pytree matching (state, x) structure
        in_specs=[MyState(
            weights=pl.BlockSpec(...),
            bias=pl.BlockSpec(...)
        ), pl.BlockSpec(...)]
    )(state, x)
```

### Mixed Precision Compute

```python
def mixed_precision_matmul(x_ref, y_ref, o_ref):
    """Compute in FP32, accumulate in FP32, output in FP16."""
    # Inputs are FP16 in SMEM
    x_f16 = x_ref[:]
    y_f16 = y_ref[:]
    
    # MMA operates in FP32 internally (on Tensor Cores)
    acc_f32 = x_f16.astype(jnp.float32) @ y_f16.astype(jnp.float32)
    
    # Convert back to FP16 for output
    o_ref[:] = acc_f32.astype(jnp.float16)
```

---

## Testing & Validation Patterns

### Chunk vs Recurrent Parity Test

```python
def test_kernel_modes():
    """Verify chunk and recurrent modes produce same results."""
    
    # Chunk mode (parallel over K)
    def chunk_kernel(x_ref, y_ref, o_ref):
        # ... implementation with pipeline over K ...
        pass
    
    # Recurrent mode (sequential over K)
    def recurrent_kernel(x_ref, y_ref, o_ref, state_ref):
        # ... implementation with explicit loop over K ...
        pass
    
    # Test
    chunk_out = pl.pallas_call(chunk_kernel, ...)(x, y)
    recurrent_out = pl.pallas_call(recurrent_kernel, ...)(x, y, init_state)
    
    np.testing.assert_allclose(chunk_out, recurrent_out, rtol=1e-4, atol=1e-4)
```

### Interpret Mode for Debugging

```python
@partial(jax.jit, static_argnames=['interpret'])
def debug_kernel(x, interpret=False):
    return pl.pallas_call(
        kernel_fn,
        out_shape=...,
        interpret=interpret  # Run on CPU, easier to debug
    )(x)

# Development: interpret=True (slow but debuggable)
result = debug_kernel(x, interpret=True)

# Production: interpret=False (JIT compile to GPU)
result = debug_kernel(x, interpret=False)
```

---

## Resources

- **Pallas Examples**: https://github.com/jax-ml/jax/tree/main/tests/pallas
- **Collective Matmul**: https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html
- **Pipelining Guide**: https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html
- **Design Philosophy**: https://docs.jax.dev/en/latest/pallas/design/design.html

---

**Last Updated**: November 18, 2025
