# Pallas GPU Backend Status (Phase 0)

**Last updated**: November 23, 2025  
**Status**: ❌ **Disabled** – Awaiting Phase 1+ handwritten Triton kernel

---

## Summary

`MambaPallasKernel` (`linearnexus/kernels/mamba_pallas.py`) is **currently unsupported** and raises `NotImplementedError` on instantiation. The Pallas/Triton backend has fundamental limitations that prevent implementing Mamba's selective scan algorithm within the `pl.pallas_call` abstraction.

**Use `MambaReferenceKernel` instead** for all Phase 0 work. GPU acceleration will be added in Phase 1+ via handwritten Triton kernel code.

---

## Technical Root Causes

### 1. No `lax.scan` with Extensive Inputs
Triton's Pallas lowering only supports `fori_loop`-style scans where `num_extensive == 0`:

```python
# ❌ NOT SUPPORTED (extensive inputs = arrays being scanned over)
final_state, outputs = jax.lax.scan(step, state, (hidden, delta, gate, B, C))

# ✅ SUPPORTED (but still hits issue #2 below)
final_state = jax.lax.fori_loop(0, seq_len, step, state)
```

**Error**: `NotImplementedError: num_extensive > 0` during Triton lowering.

### 2. No `dynamic_slice` Primitive
Even with `fori_loop`, indexing arrays with loop variables generates `dynamic_slice`:

```python
def step(t, state):
    hidden_t = hidden[t]  # ❌ Generates dynamic_slice[slice_sizes=(1,)]
    # ...
```

**Error**: `NotImplementedError: Unimplemented primitive in Pallas GPU lowering: dynamic_slice`

JAX traces `hidden[t]` (where `t` is dynamic) as:
```
r:f32[1] = dynamic_slice[slice_sizes=(1,)] hidden t
s:f32[] = squeeze[dimensions=(0,)] r
```

Triton has no lowering rule for `dynamic_slice` because it expects **static indexing** or **ref-based loads** (`hidden_ref[batch, channel, t]`), but even ref indexing with dynamic `t` generates `dynamic_slice` in the current Pallas implementation.

### 3. Sequential Dependencies
Mamba's selective scan is **inherently sequential**: each timestep's state depends on the previous timestep's output. This conflicts with GPU parallelism and Triton's design for embarrassingly parallel workloads.

---

## What We Tried

### Attempt 1: `lax.scan` with Array Inputs
```python
def step(state, inputs):
    hidden_t, delta_t, gate_t, B_t, C_t = inputs
    # ...selective scan math...
    return new_state, y

scan_inputs = (hidden, delta, gate, B, C)
final_state, outputs = jax.lax.scan(step, state, scan_inputs)
```

**Result**: ❌ `NotImplementedError: num_extensive = 6` (Triton only supports `fori_loop`-style scans).

### Attempt 2: `fori_loop` with Array Indexing
```python
def step(t, state):
    hidden_t = hidden[t]  # Load from array
    # ...
    return new_state

final_state = jax.lax.fori_loop(0, seq_len, step, state)
```

**Result**: ❌ `NotImplementedError: dynamic_slice` (array indexing with dynamic `t` unsupported).

### Attempt 3: Direct Ref Indexing
```python
def step(t, state):
    hidden_t = hidden_ref[batch_idx, channel_idx, t]  # Load from ref
    # ...
```

**Result**: ❌ `NotImplementedError: dynamic_slice` (ref indexing with dynamic loop variable still generates dynamic slicing internally).

### Attempt 4: Python `for` Loop Unrolling
```python
for t in range(seq_len):
    process_timestep(t)
```

**Result**: ❌ Works for **very small** `seq_len` (Pythonfor unrolls at trace time) but:
- Explodes compilation time for realistic sequences (seq_len=64+ → 64 separate Triton kernels)
- Still generates `dynamic_slice` for non-constant `t` when traced

---

## Phase 1+ Solution: Handwritten Triton Kernel

The Pallas abstraction (`pl.pallas_call`) is too high-level for Mamba's sequential scan. Phase 1+ will:

1. **Write raw Triton kernel code** (embedded as string in Python) implementing the chunk-based selective scan algorithm from the Mamba paper.
2. **Use Triton's `tl.load`/`tl.store` directly** with pointer arithmetic for sequential timestep processing.
3. **Bypass Pallas entirely** for the inner loop, invoking compiled Triton code via `jax.extend.backend.get_backend().compile()`.
4. **Maintain Pallas for outer orchestration** (grid definition, memory management) but drop into raw Triton for the sequential scan body.

**Reference**: [flash-linear-attention Triton kernels](https://github.com/sustcsonglin/flash-linear-attention/tree/main/fla/ops/triton) provide production examples of hand-optimized Mamba/GLA scan kernels.

---

## Current State (Phase 0)

- ✅ **`MambaReferenceKernel`**: Pure JAX implementation using `lax.scan`. Works on CPU/GPU, numerically correct, used in all tests.
- ❌ **`MambaPallasKernel`**: Raises `NotImplementedError` on instantiation. Tests skipped via `@pytest.mark.skip`.
- ✅ **`MambaLayer`**: Supports `kernel_backend="reference"` (default), `"pallas"` (raises error), `"auto"` (picks reference).
- ✅ **Tests**: `test_mamba_layer.py` passes with reference kernel. `test_mamba_kernels.py` and `test_pallas_backend_*` properly skipped.

---

## FAQ

**Q: Can we use Pallas at all for Mamba?**  
A: Yes, but only for **non-sequential** parts (grid parallelism, memory management, outer loop orchestration). The inner selective scan timestep loop requires handwritten Triton.

**Q: Why not just use the reference kernel on GPU?**  
A: The reference kernel works but is slow (~10x slower than optimized kernels). Research at scale requires custom kernels. Phase 0 focuses on correctness; Phase 1+ adds performance.

**Q: When will GPU kernels land?**  
A: Phase 1 (Weeks 3-6), gated on securing compute access (see `ROADMAP.md`). Handwritten Triton kernels require GPU profiling/tuning impossible on constrained hardware.

**Q: Can I use Pallas for other attention mechanisms (RetNet, GLA)?**  
A: Possibly. Mechanisms with less stringent sequential dependencies (e.g., parallel prefix scan formulations) may fit Pallas better. We'll evaluate case-by-case in Phase 3.

---

## References

- [JAX Pallas Triton Lowering](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/lowering.py#L2486) – `_scan_lowering_rule` shows `num_extensive` check
- [Triton Language Spec](https://triton-lang.org/main/python-api/triton.language.html) – `tl.load`, `tl.store`, pointer arithmetic
- [Mamba Paper (Appendix)](https://arxiv.org/abs/2312.00752) – Chunk-based selective scan algorithm (Section B.2)
- [flash-linear-attention Mamba kernel](https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/ops/triton/chunk_fuse/fn.py) – Production Triton reference

---

**Conclusion**: Pallas/Triton's current limitations make it unsuitable for Mamba's sequential selective scan via the `pl.pallas_call` abstraction. Phase 1+ will implement handwritten Triton kernels bypassing Pallas for performance-critical inner loops.
