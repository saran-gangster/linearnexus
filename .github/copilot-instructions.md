# LinearNexus AI Coding Agent Instructions

LinearNexus is a JAX/Flax NNx research stack for linear attention and selective state-space models (SSMs), targeting GPU/TPU. **Phase 0-1 focus**: Mamba-style selective SSM with pure JAX reference kernels optimized for constrained compute (≤12GB GPU/CPU). Pallas/Triton GPU kernels come in Phase 1+.

## Core Architecture

**Four-layer abstraction** (strict separation of concerns):
```
Models (LM heads, training) → Layers (NNx modules) → Core (utilities) → Kernels (computation)
```

- **Kernels** (`linearnexus/kernels/`): Protocol-based compute primitives implementing `SelectiveKernelProtocol` with `forward_chunk()` (parallel training) and `forward_recurrent()` (autoregressive inference). Currently pure JAX (`lax.scan` chunking); Pallas GPU/TPU in Phase 1.
- **Core** (`linearnexus/core/`): Shared cross-cutting utilities (`cache.py`, `conv.py`, `padding.py`, `gating.py`, `mode.py`) eliminating duplication. All layers reuse these pure functions for depthwise conv, state management, padding logic.
- **Layers** (`linearnexus/layers/`): Flax NNx modules wiring projections → core utilities → kernel invocations. Handle parameter initialization, shape transforms, mode selection, and state threading (`MambaLayerState` bundles conv/SSM caches).
- **Models** (`linearnexus/models/`): Full transformer stacks, LM heads (stubbed until Phase 1).
- **Registry** (`linearnexus/registry.py`): Maps `"mamba"` → `(MambaLayer, MambaConfig)` for automated tests/benchmarks. Add new features here for instant tooling integration.

**Key Pattern**: Chunk-based processing splits sequences into `chunk_size` tokens (default 64) for parallel intra-chunk computation + sequential inter-chunk state updates, balancing parallelism with recurrence.

## JAX/Flax NNx Specifics

- **Immutability**: JAX arrays are immutable. Use `array.at[idx].set(val)` not `array[idx] = val`.
- **PRNG keys**: Always split keys explicitly: `key, subkey = jax.random.split(key)`. Never reuse the same key.
- **NNx parameters**: `nnx.Param` for trainable weights, `nnx.Variable` for buffers. Access via `.value`.
- **State threading**: Layers return `(output, new_state)` tuples. Thread state through sequential calls for autoregressive generation.
- **Type hints**: Required. Use `jax.Array` for array types, `Optional[T]` for nullable args.

## Code Patterns

### Adding New Kernels
1. Create dataclasses: `*KernelParams` (learned), `*KernelInputs` (runtime), `*KernelState` (recurrent)
2. Implement `SelectiveKernelProtocol` methods
3. Write reference implementation first (pure JAX/`lax.scan`) before Triton/Pallas optimizations
4. Test parity: chunk vs recurrent modes must produce identical outputs (rtol=1e-4)

### Layer Structure
```python
class MyLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: MyConfig):
        self.proj = nnx.Linear(..., rngs=rngs)
        self.kernel = MyKernel(mode=KernelMode.CHUNK)
    
    def __call__(self, x, *, state=None, mode=KernelMode.CHUNK):
        # 1. Project inputs
        # 2. Prepare kernel inputs (watch tensor layouts!)
        # 3. Call kernel with params/inputs/state
        # 4. Project outputs
        return output, new_state
```

### Testing Requirements
- **Numerical parity**: `np.testing.assert_allclose(chunk_out, recurrent_out, rtol=1e-4)`
- **Gradient correctness**: Compare `jax.grad()` vs finite differences
- **Edge cases**: Empty sequences, single tokens, non-divisible chunk sizes
- **Use small shapes**: `hidden_size=32, seq_len=16` for fast iteration

## Current Constraints (Phase 0-1)

- **Compute**: Must run on CPU or 12GB GPU. Avoid >1GB activations.
- **Dependencies**: JAX ≥0.4.28, Flax ≥0.8.3. No Triton/Pallas yet (Phase 1+).
- **Scope**: Mamba reference kernel + layer only. Models/training in backlog until compute secured.

## File Organization & Entry Points

**Critical paths**:
- `kernels/base.py`: `SelectiveKernelProtocol`, `KernelMode`, `GridConfig` (kernel contracts)
- `kernels/mamba_reference.py`: Pure JAX Mamba kernel using `lax.scan` (Phase 0 implementation)
- `core/__init__.py`: Exports `depthwise_conv1d_causal`, `ConvState`, `RecurrentState`, `select_mode`, etc. — used by all layers
- `layers/mamba.py`: `MambaLayer` (NNx module), `MambaConfig`, `MambaLayerState` (cache bundle)
- `registry.py`: `KERNEL_REGISTRY`, `LAYER_REGISTRY`, `MODEL_REGISTRY` — add features here
- `tests/test_mamba_layer.py`: Parity tests (`test_chunk_and_recurrent_paths_align`, `test_attention_mask_zeroes_out_tokens`)
- `tests/helpers/parity.py`: Shared assertions (`assert_chunk_recurrent_parity`, `assert_mask_behavior`)
- `examples/run_mamba_reference.py`: Smoke test CLI with `--batch`, `--seq`, `--hidden`, `--chunk`

## Common Gotchas

1. **Layout mismatches**: Layers use `[batch, seq, hidden]`, kernels use `[batch, intermediate, seq]`. Always transpose explicitly and comment shapes.
2. **State initialization**: Use `*State.zeros()` or layer's `init_state()` method. Never create states manually.
3. **Padding**: Chunk kernels pad sequences to `chunk_size` multiples. Strip padding before returning.
4. **Discretization**: Mamba's `A_discrete = exp(a_log * delta)` is input-dependent per token. Never cache.
5. **Depthwise conv**: Use `feature_group_count=channels` in `conv_general_dilated` for efficiency.

## Development Workflow

**Setup** (first time):
```bash
pip install -e .                   # Core dependencies (JAX, Flax, etc.)
pip install -e .[dev]              # Adds pytest, black, ruff
```

**Test cycle** (fast feedback):
```bash
pytest tests/test_mamba_layer.py -v           # Parity tests
pytest tests/test_mamba_layer.py::test_chunk_and_recurrent_paths_align  # Single test
python examples/run_mamba_reference.py --batch 2 --seq 16 --hidden 32   # Smoke test
```

**Debug mode** (when tracing errors):
```python
# Add at top of script to disable JIT for clearer stack traces
import jax
jax.config.update('jax_disable_jit', True)
```

**Typical workflow**:
1. Add new feature → register in `linearnexus/registry.py`
2. Run parity tests → fix numerical issues (rtol=1e-4)
3. Smoke test with small shapes → validate on realistic inputs
4. Update `ARCHITECTURE.md` if kernel protocol changes

## Documentation Standards

- **Docstrings**: Required for public APIs. Include shape annotations in Args/Returns.
- **Inline comments**: Annotate tensor shapes at every transformation: `# [batch, seq, hidden]`
- **Protocols**: When adding kernels, update `ARCHITECTURE.md` section 4 if contracts change.
- **Tests**: Include one-line comments explaining what each test validates.

## Future Roadmap Awareness

- **Phase 1** (Weeks 3-6): Replace reference kernel with Pallas chunk kernel. Add Triton fallback.
- **Phase 2** (Weeks 7-12): Instrumentation, ablations, benchmarks. Prep for scale-out.
- **Phase 3** (Gated): Multi-mechanism support (RetNet, GLA, DeltaNet), distributed training. Blocked on compute access.

When suggesting features, check `ROADMAP.md` phase gates. Defer large-scale work (>4 GPUs, TPU-specific) until Phase 3.

## Documentation Resources (`docs/`)

The `docs/` folder contains specialized references—consult them **before** implementing features:

### When Adding New Layers/Kernels
- **`docs/adding_new_layers.md`**: Complete step-by-step guide for implementing new attention mechanisms. Covers kernel data structures, reference implementation, NNx layer wiring, registry integration, testing patterns, and Pallas backend addition. **Read this first** when adding GLA, RetNet, DeltaNet, or similar features.

### Flax NNx Development
- **`docs/flax_nnx_glossary.md`**: Quick lookup for NNx terms (`Ref`, `Param`, `GraphDef`, `split/merge`, etc.). Use when confused about NNx concepts.
- **`docs/flax_nnx_quick_reference.md`**: Practical NNx patterns (module definition, transforms, state management, training loops). Copy-paste ready code snippets for common tasks.
- **`docs/why_flax_nnx.md`**: Design philosophy and comparison with Linen. Explains **why** NNx uses eager initialization, mutable state, and Pythonic references.

### Pallas GPU Kernel Development (Phase 1+)
- **`docs/pallas_gpu_glossary.md`**: Hardware terminology reference (SMEM, GMEM, wgmma, TMA, barriers, etc.). Essential for understanding Pallas code.
- **`docs/pallas_gpu_reference.md`**: Comprehensive Pallas API guide (memory hierarchy, `pallas_call`, BlockSpec, TensorCore ops, pipelining, synchronization). Main resource for writing GPU kernels.
- **`docs/pallas_patterns.md`**: Production-ready patterns (double-buffering, warp specialization, persistent kernels, multi-GPU). Use when optimizing performance.

### Code Review & Architecture Decisions
- **`docs/code_review_fla.md`**: Deep technical review of flash-linear-attention codebase identifying architectural strengths and weaknesses. Read when making **cross-cutting design decisions** (e.g., how to structure multi-feature codebases, avoid duplication, centralize utilities). Informs LinearNexus's core/registry/feature-packaging strategy.

### Usage Workflow
1. **Before coding**: Read relevant doc (e.g., `adding_new_layers.md` → implement Mamba layer → follow checklist)
2. **During coding**: Reference glossaries for terminology, quick references for patterns
3. **During review**: Check `code_review_fla.md` for anti-patterns (duplicated cross-cutting code, scattered features, implicit registries)

## Key References

- Mamba paper (ICLR 2024): Selective SSM design, chunking algorithm
- `ARCHITECTURE.md`: System design, memory management, kernel fusion patterns
- `TECHNICAL_REFERENCE.md`: Equations, Pallas snippets, debugging tips
- `CONTRIBUTOR_GUIDE.md`: Math foundations, JAX primer, line-by-line kernel walkthrough
- `docs/` folder: Specialized how-to guides and references (see above)

---

*Focus*: Modularity through protocols, numerical correctness first (speed later), constrained-compute pragmatism.
