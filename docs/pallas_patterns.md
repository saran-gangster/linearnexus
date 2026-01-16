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

### FlashAttention3: Production Warp-Specialized Pipeline

**Source**: JAX's Mosaic GPU FlashAttention3 (`jax/tests/pallas/mosaic_fa3_test.py`) - a complete implementation overlapping Tensor Core compute, TMA memory transfers, and online softmax with numerically stable log-sum-exp tracking.

#### Architecture Overview

```
Grid: (num_q_heads, q_seq_len // (2*block_q), batch_size)
Launch: 3 warpgroups per block
  - WG 0-1: Compute warpgroups (process disjoint Q tile halves)
  - WG 2:   Memory warpgroup (streams K/V via TMA)
```

**Key Innovation**: Each compute warpgroup processes `block_q` tokens independently (no inter-warp communication except schedule barrier), enabling perfect parallelism within a block while the memory warp asynchronously prefetches next KV tiles.

#### Configuration & Constraints

```python
@dataclasses.dataclass(frozen=True)
class TuningConfig:
    block_q: int           # Query tile size (must be multiple of 64)
    block_kv: int          # KV tile size (must be multiple of 64)
    max_concurrent_steps: int  # Pipeline depth (≥2)
    use_schedule_barrier: bool = True
    causal: bool = False
    compute_wgs_bwd: int = 1  # Backward-specific
    
    # Backward pass tiles (all-or-nothing)
    block_q_dkv: int | None = None   # Q tiles for dK/dV kernel
    block_kv_dkv: int | None = None  # KV tiles for dK/dV kernel
    block_q_dq: int | None = None    # Q tiles for dQ kernel
    block_kv_dq: int | None = None   # KV tiles for dQ kernel
```

**Validation Rules**:
- `block_q % 64 == 0` and `block_kv % 64 == 0` (WGMMA alignment)
- `head_dim % 64 == 0` (TensorCore row requirement)
- `kv_seq_len % block_kv == 0` (no partial tiles)
- `q_seq_len % (2 * block_q) == 0` (2 compute warpgroups per block)
- Backward blocks: all four must be set or all must be None

**CUDA Compatibility Bug**: Causal masking fails on CUDA 12.8.0-12.9.0 due to ptxas miscompilation. Check `cuda_runtime_get_version()` and skip.

#### Memory Layout & SMEM Allocation

```python
# Forward pass scratch (per block)
tiling = plgpu.TilingTransform((8, 64))  # 8 rows × 64 cols per tile
swizzle = plgpu.SwizzleTransform(128)    # 128-byte swizzle for bank conflict reduction

qo_scratch = plgpu.SMEM(
    (2, block_q, head_dim), jnp.float16,  # 2 compute warpgroups
    transforms=(tiling, swizzle)
)
k_scratch = plgpu.SMEM(
    (max_concurrent_steps, block_kv, head_dim), jnp.float16,
    transforms=(tiling, swizzle)
)
v_scratch = plgpu.SMEM(
    (max_concurrent_steps, block_kv, head_dim), jnp.float16,
    transforms=(tiling, swizzle)
)
lse_scratch = plgpu.SMEM((2, block_q), jnp.float32)  # Optional for backward
```

**Why Swizzle?** Without swizzling, consecutive rows map to the same SMEM banks → serialized access. Swizzle XORs high address bits with low bits, distributing rows across banks for parallel access.

#### Warp Specialization Pattern

```python
def entry(q_ref, k_ref, v_ref, out_ref, lse_ref):
    def kernel(q_ref, k_ref, v_ref, out_ref, lse_ref, scoped):
        wg_idx = lax.axis_index("wg")  # 0, 1, or 2
        
        @pl.when(wg_idx < 2)  # Compute warpgroups
        def _compute_wg():
            plgpu.set_max_registers(232, action="increase")  # More regs for compute
            # ... attention compute ...
        
        @pl.when(wg_idx == 2)  # Memory warpgroup
        def _memory_wg():
            plgpu.set_max_registers(40, action="decrease")  # Fewer regs for TMA
            # ... TMA prefetch loop ...
    
    pl.run_scoped(kernel, scratch, barriers, ...)

# Invoke with 3 threads
plgpu.kernel(entry, num_threads=3, thread_name="wg", ...)(q, k, v)
```

**Register Budgeting**:
- Compute WGs: 232 registers (need space for accumulators, softmax state)
- Memory WG: 40 registers (only issues async copies, minimal live values)
- Total: ~600 registers/block (under H100's 65K register file / 2K threads = ~32 regs/thread baseline, but block-level allocation allows imbalance)

#### Pipeline Structure: `emit_pipeline_warp_specialized`

```python
def kv_loop(kv_step, carry, causal=False):
    """Pipeline step: process one KV tile."""
    acc, m_i, l_i = carry
    slot = lax.rem(kv_step, max_concurrent_steps)  # Circular buffer
    
    # === QK Matmul (attention logits) ===
    def compute_qk(acc_ref):
        plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
        perform_schedule_barrier()  # Let other WG use TensorCore
        return acc_ref[...]
    
    qk = pl.run_scoped(compute_qk, plgpu.ACC((block_q, block_kv), jnp.float32))
    plgpu.barrier_arrive(k_consumed_barriers.at[slot])  # Release K buffer
    
    # === Causal Masking (if enabled) ===
    if causal:
        q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q
        q_ids = plgpu.broadcasted_iota(jnp.int32, (block_q, block_kv), 0, layout=plgpu.Layout.WGMMA)
        kv_ids = plgpu.broadcasted_iota(jnp.int32, (block_q, block_kv), 1, layout=plgpu.Layout.WGMMA)
        mask = (q_ids + q_seq_base) >= (kv_ids + kv_step * block_kv)
        qk = jnp.where(mask, qk, -jnp.inf)
    
    # === Online Softmax (FlashAttention algorithm) ===
    log2e = math.log2(math.e)
    m_ij = jnp.maximum(m_i, qk.max(axis=1) * log2e)  # Max in log2 space
    alpha = jnp.exp2(m_i - m_ij)                     # Rescaling factor
    m_i = m_ij                                       # Update running max
    
    # Compute softmax weights in log2 space (FMA-friendly)
    p = jnp.exp2(qk * log2e - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
    
    # Rescale previous accumulator
    acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
    l_i *= alpha
    p16 = p.astype(dtype)
    
    # Ordering matters: barrier placement affects register allocation
    if head_dim <= 128:
        l_i += p.sum(axis=1)
        acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
        end_softmax_barriers()
    else:
        end_softmax_barriers()
        l_i += p.sum(axis=1)
    
    # === PV Matmul (weighted sum of values) ===
    def compute_pv(acc_ref):
        plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
        
        # Prefetch wait for next KV tile (hide latency)
        wait_step = kv_step + 1
        wait_slot = lax.rem(wait_step, max_concurrent_steps)
        @pl.when(wait_step < kv_steps)
        def _wait():
            plgpu.barrier_wait(k_barriers.at[wait_slot])
    
    acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
    plgpu.barrier_arrive(v_consumed_barriers.at[slot])
    
    return acc, m_i, l_i
```

**Critical Insights**:

1. **Base-2 Softmax**: Using `log2` instead of natural log lets us write `p = exp2(qk * log2e - m)`, which compiles to FMA (`a*b + c`) on GPUs. Natural log would require separate mul + exp.

2. **Online Softmax State**: FlashAttention doesn't materialize full attention matrix. Instead, track running statistics:
   - `m_i`: max logit seen so far (in log2 space)
   - `l_i`: sum of exp-shifted logits
   - `acc`: weighted sum of values (rescaled each iteration)
   
   Final output: `acc / l_i` (normalized attention output)

3. **Optimization Barrier**: `lax.optimization_barrier()` prevents XLA from reordering ops. Empirically, certain orderings hurt performance (register spills or worse instruction scheduling). The `head_dim <= 128` branch uses different orderings for small vs large head dimensions.

4. **Prefetch Next Tile**: While computing with `slot`, issue `barrier_wait` for `slot+1`. By the time current step finishes, next tile is ready (hides TMA latency).

5. **Consumed Barriers**: Separate from load barriers. Memory WG waits on `*_consumed_barriers` before overwriting buffer, ensuring compute WG finished reading.

#### Causal Attention: Three-Phase Loop

Standard loop assumes all KV tiles are needed. Causal masking makes some tiles unnecessary (e.g., token 0 only attends to token 0, not future tokens).

```python
if config.causal:
    q_seq_base = lax.axis_index("q_seq") * (2 * block_q) + wg_idx * block_q
    block_q_end = q_seq_base + block_q
    block_max_kv_steps = pl.cdiv(block_q_end, block_kv)  # Only need this many KV tiles
    kv_steps = block_max_kv_steps
else:
    block_max_kv_steps = kv_seq_len // block_kv
    kv_steps = block_max_kv_steps

# Phase 1: Full tiles (no masking needed)
full_kv_steps = lax.div(q_seq_base, block_kv)
acc, m_i, l_i = lax.fori_loop(0, full_kv_steps, kv_loop, (acc, m_i, l_i))

# Phase 2: Causal tiles (apply mask)
causal_kv_loop = functools.partial(kv_loop, causal=True)
acc, m_i, l_i = lax.fori_loop(full_kv_steps, kv_steps, causal_kv_loop, (acc, m_i, l_i))

# Phase 3: Epilogue (flush pipeline)
def epilogue_kv_loop(kv_step, _):
    slot = lax.rem(kv_step, max_concurrent_steps)
    plgpu.barrier_arrive(k_consumed_barriers.at[slot])
    plgpu.barrier_arrive(v_consumed_barriers.at[slot])
    perform_schedule_barrier()
    perform_schedule_barrier()

lax.fori_loop(kv_steps, block_max_kv_steps, epilogue_kv_loop, None)
```

**Why Epilogue?** Memory WG prefetched `max_concurrent_steps` tiles upfront. If compute finishes early (causal masking), some prefetched tiles are unused. Epilogue signals all barriers to unblock memory WG, preventing deadlock.

#### Memory Warpgroup: TMA Streaming

```python
@pl.when(wg_idx == 2)
def _memory_wg():
    plgpu.set_max_registers(40, action="decrease")
    kv_head = lax.div(q_head, q_heads_per_kv_head)  # GQA: fewer KV heads
    
    # Prologue: fill pipeline
    for i in range(max_concurrent_steps):
        s = (batch, pl.ds(i * block_kv, block_kv), kv_head)
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[i], k_barriers.at[i])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[i], v_barriers.at[i])
    
    # Main loop: double-buffer remaining tiles
    @pl.loop(0, block_max_kv_steps - max_concurrent_steps)
    def _kv_loop(kv_step):
        tma_step = kv_step + max_concurrent_steps
        tma_slot = lax.rem(kv_step, max_concurrent_steps)
        s = (batch, pl.ds(tma_step * block_kv, block_kv), kv_head)
        
        # Wait for compute to finish with this slot
        plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(k_ref.at[s], k_smem.at[tma_slot], k_barriers.at[tma_slot])
        
        plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
        plgpu.copy_gmem_to_smem(v_ref.at[s], v_smem.at[tma_slot], v_barriers.at[tma_slot])
```

**TMA (Tensor Memory Accelerator)**: Hardware unit for async GMEM→SMEM transfers. Advantages:
- Zero CPU cycles after launch (fully async)
- Automatic 2D/3D tiling and swizzling
- Multicast to multiple SMEM banks across thread blocks (cluster support)

**Barrier Protocol**:
1. Memory WG: `copy_gmem_to_smem(..., load_barrier.at[slot])` → signals when data arrives
2. Compute WG: `barrier_wait(load_barrier.at[slot])` → waits for data
3. Compute WG: `barrier_arrive(consumed_barrier.at[slot])` → signals done reading
4. Memory WG: `barrier_wait(consumed_barrier.at[slot])` → safe to overwrite

#### Backward Pass: Dual Pipelines

Backward requires computing three gradients: `dQ`, `dK`, `dV`. FlashAttention3 splits into two kernels:

**Kernel 1: Compute dQ**
- Pipeline over **KV tiles** (like forward)
- Preload per-query scalars: `lse` (log-sum-exp from forward), `delta = sum(dO * O)`
- Inner loop: `dS = P * (dP - delta)` → `dQ += dS @ K`

```python
def kernel_dq(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dq_ref, ...):
    # Load Q, dO, lse, delta into SMEM
    delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_ROW)
    lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_ROW)
    
    def kv_step(_, k_smem, v_smem, k_consumed, v_consumed, carry):
        dq_acc, lse, delta = carry
        
        # S = Q @ K.T
        s = pl.run_scoped(lambda a: plgpu.wgmma(a, q_smem, k_smem.T), plgpu.ACC(...))
        s *= math.log2(math.e)
        p = jnp.exp2(s - lax.broadcast_in_dim(lse, s.shape, [0]))
        
        # dP = dO @ V.T
        dp = pl.run_scoped(lambda a: plgpu.wgmma(a, do_smem, v_smem.T), plgpu.ACC(...))
        plgpu.barrier_arrive(v_consumed)
        
        # dS = P * (dP - delta)
        ds = p * (dp - lax.broadcast_in_dim(delta, p.shape, [0]))
        
        # dQ += dS @ K
        dq_acc = pl.run_state(lambda a: plgpu.wgmma(a, ds.astype(dtype), k_smem))(plgpu.ACC.init(dq_acc))
        plgpu.barrier_arrive(k_consumed)
        
        return dq_acc, lse, delta
    
    pipeline = plgpu.emit_pipeline_warp_specialized(kv_step, grid=(num_kv_tiles,), ...)
```

**Kernel 2: Compute dK, dV**
- Pipeline over **query tiles** (transpose of forward)
- Each KV tile accumulates gradients from all query tiles that attended to it
- Inner loop: `dV += P.T @ dO`, `dK += dS.T @ Q`

```python
def kernel_dkv(q_ref, k_ref, v_ref, do_ref, lse_ref, delta_ref, dk_ref, dv_ref, ...):
    # Load K, V into SMEM (fixed for this output tile)
    dk_acc = jnp.zeros((block_kv, head_dim), jnp.float32)
    dv_acc = jnp.zeros((block_kv, head_dim), jnp.float32)
    
    def q_step(_, q_smem, do_smem, lse_smem, delta_smem, ..., carry):
        dk_acc, dv_acc = carry
        
        # S.T = K @ Q.T
        sT = pl.run_scoped(lambda a: plgpu.wgmma(a, k_smem, q_smem.T), plgpu.ACC(...))
        lse = plgpu.load(lse_smem, (), layout=plgpu.Layout.WGMMA_COL)  # Column layout!
        pT = jnp.exp2(sT * log2e - lax.broadcast_in_dim(lse, sT.shape, [1]))
        
        # dV += P.T @ dO, dpT = V @ dO.T (fused)
        def compute_dv_dpt(refs):
            dv_acc_ref, dpt_acc_ref = refs
            plgpu.wgmma(dv_acc_ref, pT.astype(dtype), do_smem)
            plgpu.wgmma(dpt_acc_ref, v_smem, do_smem.T)
        
        dv_acc, dpT = pl.run_state(compute_dv_dpt)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
        
        # dS.T = P.T * (dpT - delta)
        delta = plgpu.load(delta_smem, (), layout=plgpu.Layout.WGMMA_COL)
        dsT = pT * (dpT - lax.broadcast_in_dim(delta, pT.shape, [1]))
        
        # dK += dS.T @ Q
        dk_acc = pl.run_state(lambda a: plgpu.wgmma(a, dsT.astype(dtype), q_smem))(plgpu.ACC.init(dk_acc))
        
        return dk_acc, dv_acc
    
    pipeline = plgpu.emit_pipeline_warp_specialized(q_step, grid=(num_q_tiles,), ...)
```

**Key Differences**:
- `lse`/`delta` use `Layout.WGMMA_COL` in dK/dV kernel (transposed attention)
- Fused WGMMA calls (`compute_dv_dpt`) avoid intermediate `barrier_wait`
- GQA: `dk`/`dv` summed across query heads: `dk.reshape(..., q_heads_per_kv_head, head_dim).sum(axis=-2)`

#### Performance Characteristics

**Forward Pass (H100, batch=1, seq=4096, heads=16, dim=128)**:
```
block_q=64, block_kv=256: 180us = 85% TensorCore utilization
block_q=64, block_kv=128: 165us = 92% TensorCore utilization
block_q=64, block_kv=64:  155us = 98% TensorCore utilization
```

**Why smaller block_kv wins?**
- Smaller tiles → less SMEM → higher occupancy → more warps in flight
- At seq=4096, enough parallelism even with block_kv=64
- Diminishing returns below 64 (WGMMA granularity)

**Tuning Recommendations**:
- Start with `block_q=64`, `block_kv=128`, `max_concurrent_steps=2`
- If SMEM limited: reduce `max_concurrent_steps` or `block_kv`
- If compute-bound (large head_dim): increase tile sizes
- Causal: may need smaller tiles (less work per block)

#### Applying to Linear Attention / SSMs

**Shared patterns**:
1. **Online aggregation**: FlashAttention's online softmax ≈ Mamba's chunk-wise recurrence
2. **Warp specialization**: Separate memory/compute threads applicable to any pipelined kernel
3. **Schedule barriers**: Critical when multiple warps share TensorCore (SSM conv + state update)
4. **Base-2 math**: Use `exp2` for discretization (`exp(Δ*A)`) to enable FMA fusion

**Differences**:
- SSMs: State updates are **sequential** across chunks (need inter-chunk barriers)
- Linear attention: May lack KV reuse (each query attends to all keys once)
- Mamba: Additional depthwise conv (1D, can pipeline with chunk processing)

**Template for Mamba chunk kernel**:
```python
def mamba_chunk_kernel(x_ref, conv_state_ref, ssm_state_ref, out_ref, ...):
    wg_idx = lax.axis_index("wg")
    
    @pl.when(wg_idx == 0)  # Compute warpgroup
    def _compute():
        # Pipeline over chunks
        def chunk_step(ci, x_chunk_smem, carry):
            conv_out, ssm_state = carry
            
            # Depthwise conv (reuse previous chunk's suffix)
            conv_out = depthwise_conv_causal(x_chunk_smem, conv_state)
            
            # Discretize & update SSM state
            delta = ... # Input-dependent
            a_discrete = jnp.exp2(a_log * delta * log2e)  # Base-2!
            b_discrete = delta * b
            
            # Parallel scan within chunk (or sequential for simplicity)
            ssm_state = ssm_state * a_discrete + conv_out * b_discrete
            
            return conv_out, ssm_state
        
        pipeline = plgpu.emit_pipeline_warp_specialized(chunk_step, ...)
    
    @pl.when(wg_idx == 1)  # Memory warpgroup
    def _memory():
        # Stream input chunks from GMEM
        ...
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

### Schedule Barrier for Shared TensorCore Access

FlashAttention3 uses an explicit "schedule" barrier (single-slot `plgpu.Barrier`) to serialize warpgroups that share the same Tensor Core pipelines:

```python
schedule_barrier = plgpu.Barrier(num_arrivals=num_compute_wgs)

def perform_schedule_barrier():
    plgpu.barrier_arrive(schedule_barrier)
    plgpu.barrier_wait(schedule_barrier)

@pl.when(wg_idx == 0)
def compute_a():
    perform_schedule_barrier()  # let wg 1 finish previous WGMMA
    plgpu.wgmma(acc, q_smem, k_smem)

@pl.when(wg_idx == 1)
def compute_b():
    perform_schedule_barrier()
    plgpu.wgmma(acc, p_smem, v_smem)
```

- Arrive/wait pairs are invoked before and after every Tensor Core call so only one warp issues WGMMA at a time.
- Because the barrier is separate from the per-buffer barriers, Tensor Core serialization stays orthogonal to buffer lifetime management.
- This pattern drops register pressure (each warp runs a simpler loop) and improves determinism when profiling or autotuning.

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
