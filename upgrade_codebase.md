# LinearNexus Codebase Upgrade Plan

_Last updated: 2025-11-18_

This document captures a concrete plan to evolve LinearNexus so that, while learning from the strengths of the Flash Linear Attention (FLA) codebase, we deliberately avoid its structural weaknesses and leverage our JAX/Flax NNx/Pallas stack to be strictly more modular and portable.

The plan is written to be **actionable** under current compute constraints (CPU / 12GB GPU) and to scale cleanly once Pallas GPU/TPU kernels are introduced.

---

## 1. Goals

- Preserve and deepen our three-layer architecture:
  - **Kernels** (`linearnexus/kernels/*`): pure compute, hardware-aware, protocol-based.
  - **Layers** (`linearnexus/layers/*`): NNx modules, caching, projections, shape/layout management.
  - **Models** (`linearnexus/models/*`): future transformer stacks, training/eval glue.
- Learn from FLA’s strong patterns (rich tests, fused modules, clear layering) while **avoiding**:
  - Scattered cross-cutting logic (padding, gating, caching) across many files.
  - Feature code spread across multiple top-level directories with no registry.
  - Eager, heavyweight top-level imports.
  - Config duplication and drift across models.
  - Test/benchmark boilerplate repeated per feature.
- Use JAX-native advantages to surpass FLA:
  - First-class `jit`/`vmap`/`pjit` for composition.
  - Pallas backends for both GPU (Triton-like) and TPU (Mosaic).
  - NNx state threading and PRNG discipline.

---

## 2. High-Level Structure Changes

### 2.1 Introduce `linearnexus/core/` for Cross-Cutting Concerns

**Problem in FLA**: padding/unpadding, gating, cache management, and short-conv logic are duplicated across GLA/KDA/Mamba-like layers.

**Plan for LinearNexus**:

Add a `linearnexus/core/` package with the following initial modules:

- `core/cache.py`
  - Defines generic recurrent/cache state containers for layers:
    - `RecurrentState` (e.g., Mamba SSM state) with `zeros(...)` helpers.
    - Optional `ConvState` for depthwise conv buffers.
  - Small, shape-annotated helpers:
    - `init_recurrent_state(batch, channels, state_size, dtype)`.
    - `update_state(state, new_state)` (pure functional; no mutation).

- `core/padding.py`
  - JAX equivalents of FLA’s `get_unpad_data`, `pad_input`, and `index_first_axis`:
    - `compute_unpadded_indices(mask: Array) -> (indices, cu_seqlens)`.
    - `unpad(x, indices)`, `pad(x_flat, indices, batch, seq)`.
  - Shared by all future layers that support packed variable-length sequences.

- `core/gating.py`
  - Shared gating utilities for Mamba-style, GLA-style, and Delta-style mechanisms:
    - Low-rank gate projection helpers (e.g., `project_low_rank_gate(x, rank, out_dim)`).
    - Standardized gate transforms (e.g., `logsigmoid` scaling, clamping).
  - Keeps per-layer code focused on wiring, not repeated math.

- `core/conv.py`
  - Move `_depthwise_conv1d` from `layers/mamba.py` into a reusable, well-documented function:
    - `depthwise_conv1d_causal(inputs, weight, bias, cache=None) -> (output, new_cache)`.
  - Used by any selective SSM layers that need short convs.

This directly addresses FLA’s weakness of scattered cross-cutting code while keeping NNx modules thin and focused.

### 2.2 Feature-Centric Grouping Without Breaking the Three-Layer Stack

**Problem in FLA**: each feature (GLA, KDA, Mamba2, RWKV, etc.) lives across `ops/`, `layers/`, `models/`, and `tests/` with no single feature home.

**Plan for LinearNexus** (incremental, starting with Mamba):

- Keep the existing top-level layout (`kernels/`, `layers/`, `models/`) but introduce **feature namespaces**:
  - `linearnexus/kernels/mamba_reference.py` (already present).
  - Future: `linearnexus/kernels/mamba_pallas.py`, etc.
  - `linearnexus/layers/mamba.py` (already present).
  - `linearnexus/models/mamba.py` (stubbed for now).
- Add a **registry** (see §3) that groups these by feature so tooling/tests/docs can treat “Mamba” as a single unit.
- When we add more mechanisms (GLA, Delta, etc.), mirror this pattern:
  - `kernels/<feature>_*.py`, `layers/<feature>.py`, `models/<feature>.py`.

This preserves our clean three-layer separation but adds the feature-centric view FLA lacks, without a disruptive repo reshuffle.

---

## 3. Registries and Config System

### 3.1 Kernel/Layer/Model Registry

**Problem in FLA**: no explicit registry; features are discovered by imports and naming conventions.

**Plan**: create `linearnexus/registry.py` exposing simple dictionaries with string keys:

```python
from linearnexus.kernels.mamba_reference import MambaReferenceKernel
from linearnexus.layers.mamba import MambaLayer, MambaConfig

KERNEL_REGISTRY = {
    "mamba:reference": MambaReferenceKernel,
    # future: "mamba:pallas": MambaPallasKernel,
}

LAYER_REGISTRY = {
    "mamba": (MambaLayer, MambaConfig),
}

MODEL_REGISTRY = {
    # future: "mamba_lm_tiny": (MambaLMConfig, MambaLMModel),
}
```

Usage:

- Tests/benchmarks iterate `LAYER_REGISTRY` to run common parity and gradient suites.
- CLI/experiments create models from registry names instead of hard-coded imports.
- Docs generator walks the registries to build tables of supported mechanisms.

### 3.2 Unified Config Base

**Problem in FLA**: HF `configuration_*.py` files duplicate fields and defaults per model family.

**Plan**:

- Keep `MambaConfig` as a dataclass but introduce a small `ConfigBase` helper in `linearnexus/core/config.py`:
  - Provides `.to_dict()`, `.from_dict()`, and optional validation hooks.
  - Encodes shared knobs once (e.g., `chunk_size`, `dtype`, `init_scale`).
- Future: load configs from YAML while keeping type safety for NNx modules.

This avoids config drift while staying lighter-weight than full HF `PretrainedConfig` machinery.

---

## 4. Kernel Layer: Reference → Pallas Without FLA’s Pitfalls

### 4.1 Clarify `SelectiveKernelProtocol` for Pallas

Our current `SelectiveKernelProtocol` in `kernels/base.py` is JAX-only. To prepare for Pallas while avoiding tight coupling:

- Extend the protocol docstring to clarify that `forward_chunk` and `forward_recurrent` must be **pure functions** over JAX arrays, irrespective of whether the underlying implementation uses Pallas.
- For Pallas-backed kernels, keep the Pallas calls encapsulated inside the method; do **not** let NNx layers depend on Pallas types (`Ref`, `BlockSpec`).

### 4.2 GridConfig and Tuning Utilities

FLA relies on hand-tuned Triton heuristics. We can do better by:

- Expanding `GridConfig` to optionally carry Pallas `BlockSpec` hints for GPU/TPU, but still returning an abstract description to callers.
- Adding a small `kernels/tuning.py` with helper functions:
  - `estimate_grid_for_sequence(batch, seq, feature_dim, chunk_size)`.
  - Future: micro-benchmark-based tuner that caches best tile sizes per device.

This centralizes launch logic and keeps layers ignorant of backend details.

### 4.3 Reference vs Pallas Implementation Strategy

To avoid FLA’s partial duplication between `naive_*` and fused kernels:

- Maintain **one** canonical JAX reference kernel (`MambaReferenceKernel`), with the Pallas-backed kernel class delegating to it in debug/emulation modes.
- Ensure tests always compare Pallas kernels against the reference via the same `SelectiveKernelProtocol` API.

---

## 5. Layer Level: NNx Patterns and State Management

### 5.1 Slimming `MambaLayer` Using Core Utilities

Refactor `linearnexus/layers/mamba.py` to:

- Import and use `core.conv.depthwise_conv1d_causal` instead of local `_depthwise_conv1d`.
- Define `MambaLayerState` in `core/cache.py` or make it compose `RecurrentState` + `ConvState` rather than embedding raw arrays.
- Add small comments around all shape transforms (`[batch, seq, hidden] → [batch, intermediate, seq]`) to match our doc standards and avoid latent layout bugs.

This reduces per-layer complexity and prepares for additional selective SSM layers that reuse the same state patterns.

### 5.2 Mode Handling and Autoregressive Decoding

We already support `KernelMode.CHUNK` and `KernelMode.RECURRENT`. To align with FLA’s automatic mode switching without their duplication:

- Add a simple utility in `core/mode.py`:
  - `select_mode(seq_len, threshold=64) -> KernelMode`.
- Make `MambaLayer.__call__` default to `select_mode(seq_len)` when `mode` is `None`, so inference automatically uses recurrent mode for short sequences.

### 5.3 Future: Shared Attention/SSM Block Base

As we add more mechanisms, introduce a small `core/blocks.py` base class:

- Offers a common `__call__(x, *, state=None, attention_mask=None, mode=None, chunk_size=None)` signature.
- Handles mask application, varlen unpadding/repadding using `core.padding`.
- Concrete layers (Mamba, GLA, Delta) only implement their internal projections + kernel calls.

This addresses FLA’s weakness where every feature re-implements the same packed-varlen and mask logic.

---

## 6. Testing and Benchmarks

### 6.1 Shared Parity and Gradient Harness

FLA has deep tests but duplicates structure per feature. We can centralize:

- Add `tests/helpers/parity.py` with functions:
  - `assert_chunk_recurrent_parity(layer, inputs, *, atol=1e-4, rtol=1e-4)`.
  - `assert_mask_behavior(layer, inputs, mask)`.
- Use these helpers in `tests/test_mamba_layer.py` (and future tests) to avoid re-rolling loops each time.

### 6.2 Kernel-Level Tests

When we add Pallas kernels, create `tests/test_mamba_kernel.py` that:

- Generates small random batches.
- Compares `MambaReferenceKernel.forward_chunk` vs `MambaPallasKernel.forward_chunk` using `np.testing.assert_allclose`.
- Verifies gradients via `jax.grad` on a scalar loss.

All new kernels must pass these before being exposed through the registry.

### 6.3 Micro-Benchmarks (Optional, Post-Phase-1)

Add a simple `benchmarks/` script that:

- Runs the layer on fixed shapes (e.g., `batch=2, seq=128, hidden=256`).
- Logs wall clock and effective tokens/s.

This will help tune Pallas grid configs later, but is not required for immediate correctness.

---

## 7. Documentation and Tooling

### 7.1 Feature Mini-Docs

For each feature (starting with Mamba):

- Add a small `docs/features/mamba.md` describing:
  - Equations and architecture sketch.
  - Supported modes (chunk, recurrent).
  - Implementation notes (e.g., depthwise conv, selective scan).

### 7.2 Auto-Generated Summaries

Once the registry exists:

- Write a small script (e.g., `tools/generate_feature_table.py`) that walks `LAYER_REGISTRY` and emits a markdown table summarizing all features into `README.md` or `ARCHITECTURE.md`.

This avoids FLA’s manual doc duplication and keeps the repo self-describing as it grows.

---

## 8. Incremental Execution Plan

Given current constraints, upgrades should be incremental:

1. **Core Package & Refactors (Short-Term)**
   - Create `linearnexus/core/{cache,padding,gating,conv}.py`.
   - Move `_depthwise_conv1d` into `core.conv` and update `MambaLayer` to use it.
   - Keep behavior and tests identical (no functional changes beyond imports).

2. **Registry and Config Base**
   - Implement `linearnexus/registry.py` for Mamba kernel/layer.
   - Add `core/config.py` with a small `ConfigBase` and adapt `MambaConfig` to inherit or reuse it.

3. **Testing Helpers**
   - Add `tests/helpers/parity.py` and rewrite `test_mamba_layer.py` in terms of these helpers.

4. **Documentation Hooks**
   - Create `docs/features/mamba.md` and a lightweight generator script or a TODO note for future automation.

5. **Pallas Kernel Integration (Phase 1/2 Roadmap)**
   - Design a Pallas-backed Mamba kernel implementing `SelectiveKernelProtocol`.
   - Add kernel tests comparing Pallas vs reference.
   - Register it under a separate key (e.g., `"mamba:pallas"`) to allow easy A/B testing.

Throughout, we explicitly avoid FLA’s weaknesses by:

- Centralizing shared logic in `core/` instead of copying it per feature.
- Using registries to keep feature components discoverable and scriptable.
- Keeping `__init__.py` imports narrow (no heavy eager imports) so small users don’t pay cost for all features.

This plan gives us a clear path from the current Mamba-only reference implementation to a modular, multi-feature linear attention stack that is more coherent and portable than FLA’s Triton/PyTorch-only design, while remaining realistic for our current compute environment.