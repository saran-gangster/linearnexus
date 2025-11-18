# Adding New Layers to LinearNexus

**Last updated**: 2025-11-18

This guide walks through the complete process of adding a new linear attention or selective state-space mechanism to LinearNexus. It covers kernel implementation, layer wiring, Pallas backend integration, testing, and registry management—all while leveraging the shared `linearnexus/core/` runtime library to avoid code duplication.

---

## Table of Contents

1. [Overview: The LinearNexus Stack](#overview-the-linearnexus-stack)
2. [Step 1: Design Kernel Data Structures](#step-1-design-kernel-data-structures)
3. [Step 2: Implement Reference Kernel](#step-2-implement-reference-kernel)
4. [Step 3: Build the NNx Layer](#step-3-build-the-nnx-layer)
5. [Step 4: Register in `registry.py`](#step-4-register-in-registrypy)
6. [Step 5: Write Layer Tests](#step-5-write-layer-tests)
7. [Step 6: Add Pallas Backend (Optional)](#step-6-add-pallas-backend-optional)
8. [Step 7: Kernel-Level Parity Tests](#step-7-kernel-level-parity-tests)
9. [Best Practices & Anti-Patterns](#best-practices--anti-patterns)
10. [Example: Skeleton GLA Layer](#example-skeleton-gla-layer)

---

## Overview: The LinearNexus Stack

LinearNexus enforces a strict three-layer separation:

```
Kernels (linearnexus/kernels/)
    ↓ Pure JAX Array interfaces via SelectiveKernelProtocol
Core Runtime (linearnexus/core/)
    ↓ Shared utilities: cache, conv, padding, gating, mode selection
Layers (linearnexus/layers/)
    ↓ Flax NNx modules: projections, shape transforms, kernel wiring
Models (linearnexus/models/)
    ↓ Transformer blocks, LM heads, training/inference entry points
```

**Key principle**: Cross-cutting concerns (caching, convolution, gating, padding) live in `core/` and are reused by all layers. This eliminates duplication and keeps layers focused on mechanism-specific logic.

---

## Step 1: Design Kernel Data Structures

Before writing any code, define the mathematical interface for your kernel. You'll need three dataclasses:

### 1.1 Kernel Parameters (`*KernelParams`)

**What**: Learned weights derived from layer parameters (e.g., SSM `A` matrix, bias terms).

**Example** (Mamba):
```python
@dataclass
class MambaKernelParams:
    a_log: jax.Array  # [intermediate, ssm_state]
    d: jax.Array      # [intermediate]
```

**Guidelines**:
- Include only parameters needed inside the kernel computation.
- Use log-space representations when numerically appropriate (e.g., `a_log` for SSM decay).
- Keep shapes explicit in comments.

### 1.2 Kernel Inputs (`*KernelInputs`)

**What**: Runtime tensors passed to the kernel at each forward call.

**Example** (Mamba):
```python
@dataclass
class MambaKernelInputs:
    hidden: jax.Array  # [batch, intermediate, seq]
    delta: jax.Array   # [batch, intermediate, seq]
    B: jax.Array       # [batch, seq, ssm_state]
    C: jax.Array       # [batch, seq, ssm_state]
    gate: jax.Array    # [batch, intermediate, seq]
```

**Guidelines**:
- Use the layout expected by the kernel (often `[batch, intermediate, seq]` for selective SSMs, but `[batch, seq, heads, dim]` for attention-style mechanisms).
- The layer is responsible for transforming from standard NNx layout (`[batch, seq, hidden]`) to kernel layout.
- Document shapes thoroughly.

### 1.3 Kernel State (`*KernelState`)

**What**: Recurrent state carried between invocations during autoregressive generation.

**Example** (Mamba):
```python
@dataclass
class MambaKernelState:
    ssm: jax.Array  # [batch, intermediate, ssm_state]

    @classmethod
    def zeros(cls, batch_size: int, intermediate: int, ssm_state: int, dtype: jnp.dtype):
        return cls(ssm=jnp.zeros((batch_size, intermediate, ssm_state), dtype=dtype))
```

**Guidelines**:
- Provide a `zeros` classmethod for initialization.
- State should be **sufficient to resume** recurrent computation from any point in the sequence.
- For attention mechanisms that don't use recurrence, this can be empty or omitted.

---

## Step 2: Implement Reference Kernel

**Location**: `linearnexus/kernels/<feature>_reference.py`

### 2.1 Protocol Compliance

All kernels must implement `SelectiveKernelProtocol`:

```python
from linearnexus.kernels.base import SelectiveKernelProtocol, KernelMode, GridConfig

class YourReferenceKernel(SelectiveKernelProtocol):
    def forward_chunk(
        self,
        params: YourKernelParams,
        inputs: YourKernelInputs,
        state: Optional[YourKernelState],
        *,
        chunk_size: int,
    ) -> tuple[jax.Array, YourKernelState]:
        """Chunked forward pass for training."""
        ...

    def forward_recurrent(
        self,
        params: YourKernelParams,
        inputs: YourKernelInputs,
        state: Optional[YourKernelState],
    ) -> tuple[jax.Array, YourKernelState]:
        """Single-step recurrent forward for inference."""
        ...

    def get_grid_config(
        self,
        *,
        batch_size: int,
        seq_len: int,
        feature_dim: int,
    ) -> GridConfig:
        """Return grid configuration for Pallas kernels."""
        ...
```

### 2.2 Implementation Strategy

**Reference kernels use pure JAX only**—no Pallas, no external CUDA. They serve as the numerical ground truth.

#### Pattern 1: Chunk-Based Processing

Most selective SSMs naturally decompose into chunks:

```python
def forward_chunk(self, params, inputs, state, *, chunk_size: int):
    # 1. Cast to target dtype
    hidden = inputs.hidden.astype(self.dtype)
    delta = inputs.delta.astype(self.dtype)
    # ... cast other inputs

    batch_size, intermediate, seq_len = hidden.shape
    state = state or YourKernelState.zeros(batch_size, intermediate, state_dim, self.dtype)

    # 2. Pad to multiple of chunk_size
    num_chunks = math.ceil(seq_len / chunk_size)
    pad = num_chunks * chunk_size - seq_len
    if pad > 0:
        hidden = jnp.pad(hidden, ((0,0), (0,0), (0, pad)))
        # ... pad other inputs

    # 3. Reshape into chunks: [num_chunks, batch, intermediate, chunk_size]
    hidden_chunks = hidden.reshape(batch_size, intermediate, num_chunks, chunk_size)
    hidden_chunks = hidden_chunks.transpose(2, 0, 1, 3)  # [num_chunks, batch, intermediate, chunk_size]

    # 4. Define chunk scan function
    def chunk_scan(carry, chunk_inputs):
        recurrent_state = carry
        # a) Compute intra-chunk updates (parallel across chunk_size)
        # b) Update recurrent state (sequential across chunks)
        # c) Compute outputs
        return new_state, chunk_outputs

    # 5. Scan over chunks
    final_state, chunk_outputs = jax.lax.scan(
        chunk_scan,
        state.ssm,  # initial carry
        chunk_inputs,
    )

    # 6. Reshape outputs and strip padding
    outputs = chunk_outputs.transpose(1, 2, 0, 3).reshape(batch_size, intermediate, num_chunks * chunk_size)
    outputs = outputs[:, :, :seq_len]

    return outputs, YourKernelState(ssm=final_state)
```

#### Pattern 2: Recurrent as Chunk-of-1

For autoregressive generation:

```python
def forward_recurrent(self, params, inputs, state):
    return self.forward_chunk(params, inputs, state, chunk_size=1)
```

This ensures numerical parity between chunk and recurrent modes.

### 2.3 Numerical Stability

- Use log-space for exponentials: `a_discrete = jnp.exp(a_log * delta)` not `a ** delta`.
- Subtract max before softmax: `scores - jnp.max(scores, axis=-1, keepdims=True)`.
- Accumulate in float32 even when inputs are bfloat16.
- Clamp gates and deltas to reasonable ranges if needed.

### 2.4 Complete Example

See `linearnexus/kernels/mamba_reference.py` for a working reference implementing the Mamba selective SSM with chunking and recurrent modes.

---

## Step 3: Build the NNx Layer

**Location**: `linearnexus/layers/<feature>.py`

### 3.1 Config Definition

Inherit from `ConfigBase` for serialization:

```python
from dataclasses import dataclass
from linearnexus.core import ConfigBase

@dataclass
class YourConfig(ConfigBase):
    hidden_size: int = 256
    intermediate_size: int = 512
    state_size: int = 16
    num_heads: int = 8
    conv_kernel: int = 4
    chunk_size: int = 64
    use_bias: bool = True
    hidden_act: str = "silu"
```

**Shared fields** (consider extracting to `ConfigBase` if common across many layers):
- `chunk_size`
- `dtype` (if not inferred from inputs)
- Initialization scales

### 3.2 Layer State

Compose `core.cache` abstractions:

```python
from dataclasses import dataclass
from linearnexus.core import ConvState, RecurrentState
import jax.numpy as jnp

@dataclass
class YourLayerState:
    conv_state: ConvState | None  # If using depthwise conv
    ssm_state: RecurrentState | None  # For selective SSM
    position: jnp.int32

    @classmethod
    def initialize(
        cls,
        *,
        batch_size: int,
        conv_kernel: int,
        intermediate_size: int,
        state_size: int,
        dtype: jnp.dtype,
    ):
        return cls(
            conv_state=ConvState.zeros(
                batch_size=batch_size,
                kernel_size=conv_kernel,
                channels=intermediate_size,
                dtype=dtype,
            ),
            ssm_state=RecurrentState.zeros(
                batch_size=batch_size,
                channels=intermediate_size,
                state_size=state_size,
                dtype=dtype,
            ),
            position=jnp.array(0, dtype=jnp.int32),
        )
```

### 3.3 Layer Implementation

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Callable, Optional

from linearnexus.core import (
    ConfigBase,
    ConvState,
    RecurrentState,
    depthwise_conv1d_causal,
    select_mode,
)
from linearnexus.kernels import KernelMode

class YourLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: YourConfig):
        self.config = config

        # Projections (example: QKV-style or Mamba-style in/out)
        self.in_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        # Optional: depthwise conv weights
        if config.conv_kernel > 1:
            conv_rng = rngs.params()
            scale = 1.0 / jnp.sqrt(config.conv_kernel * config.intermediate_size)
            self.conv_weight = nnx.Param(
                jax.random.normal(conv_rng, (config.conv_kernel, config.intermediate_size)) * scale
            )
            self.conv_bias = nnx.Param(jnp.zeros((config.intermediate_size,)))
        else:
            self.conv_weight = None
            self.conv_bias = None

        # Kernel-specific parameters (e.g., SSM A, D)
        # ... initialize using rngs

        # Instantiate reference kernel
        from linearnexus.kernels.your_reference import YourReferenceKernel
        self.kernel = YourReferenceKernel(dtype=jnp.float32)

    def init_state(self, batch_size: int, dtype: jnp.dtype) -> YourLayerState:
        return YourLayerState.initialize(
            batch_size=batch_size,
            conv_kernel=self.config.conv_kernel,
            intermediate_size=self.config.intermediate_size,
            state_size=self.config.state_size,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,  # [batch, seq, hidden]
        *,
        state: Optional[YourLayerState] = None,
        attention_mask: Optional[jax.Array] = None,
        mode: Optional[KernelMode] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[jax.Array, YourLayerState]:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        chunk_size = chunk_size or self.config.chunk_size

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, dtype)

        # Auto-select mode
        if mode is None:
            mode = select_mode(seq_len, threshold=chunk_size)

        # Apply mask by zeroing inputs (simple pattern)
        if attention_mask is not None:
            mask_expanded = attention_mask[..., None]  # [batch, seq, 1]
            hidden_states = hidden_states * mask_expanded

        # Projections
        projected = self.in_proj(hidden_states)
        hidden, gate = jnp.split(projected, 2, axis=-1)

        # Optional: depthwise conv via core utility
        if self.conv_weight is not None:
            hidden, conv_cache = depthwise_conv1d_causal(
                hidden,
                self.conv_weight.value,
                self.conv_bias.value,
                cache=state.conv_state.buffer if state.conv_state else None,
            )
            new_conv_state = state.conv_state.update(conv_cache) if state.conv_state else None
        else:
            new_conv_state = state.conv_state

        # Prepare kernel inputs (transform shapes as needed)
        # Example: [batch, seq, intermediate] → [batch, intermediate, seq]
        hidden_kernel = jnp.swapaxes(hidden, 1, 2)
        gate_kernel = jnp.swapaxes(gate, 1, 2)

        kernel_inputs = YourKernelInputs(
            hidden=hidden_kernel,
            gate=gate_kernel,
            # ... other inputs
        )
        kernel_params = YourKernelParams(
            # ... extract from self parameters
        )
        kernel_state = YourKernelState(ssm=state.ssm_state.value) if state.ssm_state else None

        # Call kernel
        if mode == KernelMode.CHUNK:
            kernel_outputs, new_kernel_state = self.kernel.forward_chunk(
                kernel_params,
                kernel_inputs,
                kernel_state,
                chunk_size=chunk_size,
            )
        else:
            kernel_outputs, new_kernel_state = self.kernel.forward_recurrent(
                kernel_params,
                kernel_inputs,
                kernel_state,
            )

        # Transform back: [batch, intermediate, seq] → [batch, seq, intermediate]
        kernel_outputs = kernel_outputs.transpose(0, 2, 1)

        # Output projection
        output = self.out_proj(kernel_outputs)

        # Update state
        new_ssm_state = state.ssm_state.update(new_kernel_state.ssm) if state.ssm_state else None
        new_state = YourLayerState(
            conv_state=new_conv_state,
            ssm_state=new_ssm_state,
            position=state.position + seq_len,
        )

        return output, new_state
```

### 3.4 Key Layer Responsibilities

✅ **Do**:
- Handle all shape transformations between NNx layout and kernel layout.
- Use `core/` utilities for conv, caching, gating, padding.
- Apply `attention_mask` by zeroing inputs (simple, consistent pattern).
- Auto-select mode via `select_mode` when `mode` is `None`.
- Document all shape transforms inline: `# [batch, seq, hidden] → [batch, intermediate, seq]`.

❌ **Don't**:
- Import Pallas types (`Ref`, `BlockSpec`) in the layer.
- Reimplement depthwise conv or caching—use `core/`.
- Let kernel choice leak into layer logic (kernel is an opaque protocol implementer).

---

## Step 4: Register in `registry.py`

**Location**: `linearnexus/registry.py`

Add your kernel and layer:

```python
from linearnexus.kernels.your_reference import YourReferenceKernel
from linearnexus.layers.your_layer import YourLayer, YourConfig

KERNEL_REGISTRY["your_feature:reference"] = YourReferenceKernel
LAYER_REGISTRY["your_feature"] = (YourLayer, YourConfig)
```

**Why?**
- Tests can iterate `LAYER_REGISTRY` to auto-test all registered features.
- Benchmarks can measure all mechanisms without hardcoding imports.
- Docs generators can build feature tables automatically.

---

## Step 5: Write Layer Tests

**Location**: `tests/test_<feature>_layer.py`

Use shared helpers from `tests/helpers/parity.py`:

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.layers.your_layer import YourConfig, YourLayer
from tests.helpers.parity import assert_chunk_recurrent_parity, assert_mask_behavior


def test_chunk_and_recurrent_paths_align():
    """Ensures chunk and recurrent modes produce identical outputs."""
    rngs = nnx.Rngs(0)
    config = YourConfig(hidden_size=32, intermediate_size=32, state_size=8)
    layer = YourLayer(rngs, config)

    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (2, 16, config.hidden_size))

    # Shared helper handles both modes + comparison
    assert_chunk_recurrent_parity(layer, inputs, chunk_size=4, rtol=1e-4, atol=1e-4)


def test_attention_mask_zeroes_out_tokens():
    """Ensures masked tokens produce zero-like outputs."""
    rngs = nnx.Rngs(1)
    config = YourConfig(hidden_size=16, intermediate_size=16, state_size=4)
    layer = YourLayer(rngs, config)

    key = jax.random.PRNGKey(0)
    inputs = jax.random.normal(key, (1, 4, config.hidden_size))
    mask = jnp.array([[1.0, 1.0, 0.0, 0.0]], dtype=inputs.dtype)

    # Shared helper validates mask behavior
    assert_mask_behavior(layer, inputs, mask, rtol=1e-5, atol=1e-5)
```

**Run tests**:
```bash
pytest tests/test_<feature>_layer.py -v
```

---

## Step 6: Add Pallas Backend (Optional)

**When to add**: Once you have a working reference kernel and want GPU/TPU acceleration.

**Location**: `linearnexus/kernels/<feature>_pallas.py`

### 6.1 Implement `SelectiveKernelProtocol` with Pallas

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax
import jax.numpy as jnp

from linearnexus.kernels.base import SelectiveKernelProtocol, KernelMode, GridConfig

class YourPallasKernel(SelectiveKernelProtocol):
    def __init__(self, *, dtype=jnp.float32):
        self.dtype = dtype

    def forward_chunk(self, params, inputs, state, *, chunk_size: int):
        batch_size, intermediate, seq_len = inputs.hidden.shape

        # Define output shape
        out_shape = jax.ShapeDtypeStruct((batch_size, intermediate, seq_len), self.dtype)

        # Define kernel function
        def kernel_fn(
            hidden_ref, delta_ref, gate_ref,  # input refs
            state_ref,  # state ref
            output_ref,  # output ref
        ):
            # Load tiles into SMEM/registers
            hidden = hidden_ref[:]
            delta = delta_ref[:]
            gate = gate_ref[:]
            prev_state = state_ref[:]

            # Compute (using Pallas primitives)
            # ... selective SSM update logic
            result = ...  # computed output
            new_state = ...  # updated state

            # Store results
            output_ref[:] = result
            state_ref[:] = new_state

        # Define BlockSpecs for memory layout
        in_specs = [
            pl.BlockSpec((batch_size, intermediate, chunk_size), lambda i: (0, 0, i * chunk_size)),
            # ... specs for other inputs
        ]
        out_specs = pl.BlockSpec((batch_size, intermediate, chunk_size), lambda i: (0, 0, i * chunk_size))

        # Grid configuration
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        grid = (num_chunks,)

        # Call Pallas
        outputs = pl.pallas_call(
            kernel_fn,
            out_shape=out_shape,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        )(inputs.hidden, inputs.delta, inputs.gate, state.ssm)

        return outputs, YourKernelState(ssm=...)  # extract updated state

    def forward_recurrent(self, params, inputs, state):
        # For simplicity, delegate to chunk with size 1
        return self.forward_chunk(params, inputs, state, chunk_size=1)

    def get_grid_config(self, *, batch_size, seq_len, feature_dim):
        chunks = max(1, seq_len // 64)
        return GridConfig(block_shape=(feature_dim,), num_programs=(batch_size * chunks,))
```

### 6.2 Key Pallas Patterns

**BlockSpec anatomy**:
```python
pl.BlockSpec(
    block_shape=(128, 64),           # Shape of tile in SMEM
    index_map=lambda i, j: (i, j),   # Maps grid indices to tensor indices
)
```

**Mosaic GPU transforms** (for optimized memory access):
```python
plgpu.BlockSpec(
    block_shape=(128, 64),
    index_map=lambda i, j: (i, j),
    transforms=(
        plgpu.TilingTransform((8, 8)),   # Tile for tensor cores
        plgpu.SwizzleTransform(128),     # Avoid bank conflicts
    ),
)
```

**Software pipelining** (overlap memory + compute):
```python
pipeline = plgpu.emit_pipeline(
    kernel_body,
    in_specs=[...],
    grid=(num_chunks,),
    max_concurrent_steps=2,  # Number of stages in flight
)
```

### 6.3 Register Pallas Kernel

```python
# linearnexus/registry.py
from linearnexus.kernels.your_pallas import YourPallasKernel

KERNEL_REGISTRY["your_feature:pallas"] = YourPallasKernel
```

### 6.4 Switching Backends in Layers

**Option 1**: Pass kernel at construction:
```python
class YourLayer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, config: YourConfig, *, kernel=None):
        self.kernel = kernel or YourReferenceKernel(dtype=jnp.float32)
        # ... rest of init
```

**Option 2**: Choose via registry:
```python
from linearnexus.registry import KERNEL_REGISTRY

kernel_cls = KERNEL_REGISTRY[f"{config.feature_name}:{config.backend}"]
self.kernel = kernel_cls(dtype=config.dtype)
```

---

## Step 7: Kernel-Level Parity Tests

**Location**: `tests/test_<feature>_kernel.py`

Compare Pallas vs reference on small shapes:

```python
import jax
import jax.numpy as jnp
import numpy as np

from linearnexus.kernels.your_reference import YourReferenceKernel, YourKernelParams, YourKernelInputs, YourKernelState
from linearnexus.kernels.your_pallas import YourPallasKernel


def test_pallas_matches_reference():
    """Ensures Pallas kernel produces identical outputs to reference."""
    batch_size = 2
    intermediate = 32
    seq_len = 16
    state_size = 8

    # Generate random inputs
    key = jax.random.PRNGKey(0)
    inputs = YourKernelInputs(
        hidden=jax.random.normal(key, (batch_size, intermediate, seq_len)),
        delta=jax.random.normal(key, (batch_size, intermediate, seq_len)),
        gate=jax.random.normal(key, (batch_size, intermediate, seq_len)),
    )
    params = YourKernelParams(...)  # initialize parameters
    state = YourKernelState.zeros(batch_size, intermediate, state_size, jnp.float32)

    # Run both kernels
    ref_kernel = YourReferenceKernel(dtype=jnp.float32)
    pallas_kernel = YourPallasKernel(dtype=jnp.float32)

    ref_output, ref_state = ref_kernel.forward_chunk(params, inputs, state, chunk_size=8)
    pallas_output, pallas_state = pallas_kernel.forward_chunk(params, inputs, state, chunk_size=8)

    # Compare outputs
    np.testing.assert_allclose(pallas_output, ref_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pallas_state.ssm, ref_state.ssm, rtol=1e-4, atol=1e-4)


def test_pallas_gradients():
    """Ensures Pallas kernel gradients are correct."""
    # ... similar setup

    def loss_fn(params_dict):
        outputs, _ = pallas_kernel.forward_chunk(params_dict, inputs, state, chunk_size=8)
        return jnp.sum(outputs ** 2)

    # JAX auto-diff
    grad_auto = jax.grad(loss_fn)(params_as_dict)

    # Finite differences
    grad_numerical = numerical_gradient(loss_fn, params_as_dict)

    # Compare
    for key in grad_auto:
        np.testing.assert_allclose(grad_auto[key], grad_numerical[key], rtol=1e-3, atol=1e-3)
```

---

## Best Practices & Anti-Patterns

### ✅ Best Practices

1. **Reuse `core/` utilities**
   - Use `depthwise_conv1d_causal` for all depthwise convs.
   - Use `ConvState` and `RecurrentState` for state management.
   - Use `compute_unpadded_indices`, `unpad`, `pad` for packed sequences.
   - Use `low_rank_project` and `normalize_gate_logits` for gating patterns.

2. **Protocol-based kernel interface**
   - Kernels implement `SelectiveKernelProtocol`; layers don't know if it's JAX or Pallas.
   - Keep Pallas imports inside `kernels/*_pallas.py` only.

3. **Reference-first development**
   - Always implement and test a pure JAX reference kernel first.
   - Add Pallas only after reference is working and tested.

4. **Shared test helpers**
   - Use `tests/helpers/parity.py` for chunk/recurrent and mask tests.
   - Add registry-driven tests that iterate all registered layers.

5. **Explicit shape documentation**
   - Comment every shape transform: `# [batch, seq, hidden] → [batch, intermediate, seq]`.
   - Use descriptive variable names that include layout hints.

6. **Auto-mode selection**
   - Default `mode=None` and use `select_mode(seq_len, threshold=chunk_size)` to choose chunk vs recurrent automatically.

### ❌ Anti-Patterns (Learned from FLA)

1. **Don't duplicate cross-cutting logic**
   - ❌ Reimplementing depthwise conv in each layer.
   - ✅ Import `depthwise_conv1d_causal` from `core.conv`.

2. **Don't scatter feature code**
   - ❌ Ops in `ops/`, layer in `layers/`, tests in `tests/`, no connection.
   - ✅ Register in `registry.py` for automated tooling.

3. **Don't let layers depend on kernel internals**
   - ❌ `from jax.experimental.pallas import BlockSpec` in layer code.
   - ✅ Layers only see `SelectiveKernelProtocol` interface.

4. **Don't hardcode test logic per feature**
   - ❌ Copy-pasting chunk/recurrent loops in every test file.
   - ✅ Use `assert_chunk_recurrent_parity` helper.

5. **Don't duplicate config schemas**
   - ❌ Each config redefines `chunk_size`, `dtype`, etc.
   - ✅ Inherit from `ConfigBase` and share common fields.

6. **Don't use eager top-level imports**
   - ❌ `from linearnexus import *` imports all features at once.
   - ✅ Lazy imports or targeted exports per module.

---

## Example: Skeleton GLA Layer

Here's a minimal GLA (Gated Linear Attention) skeleton following all patterns:

### `linearnexus/kernels/gla_reference.py`

```python
"""Reference GLA kernel (pure JAX)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from .base import GridConfig, KernelMode, SelectiveKernelProtocol


@dataclass
class GLAKernelParams:
    """GLA kernel parameters."""
    # Add learned parameters (e.g., gate normalizers, etc.)
    pass


@dataclass
class GLAKernelInputs:
    """GLA kernel inputs."""
    q: jax.Array  # [batch, heads, seq, head_dim]
    k: jax.Array  # [batch, heads, seq, head_dim]
    v: jax.Array  # [batch, heads, seq, head_dim]
    g: jax.Array  # [batch, heads, seq] - gate logits


@dataclass
class GLAKernelState:
    """GLA recurrent state."""
    kv: jax.Array  # [batch, heads, head_dim, head_dim]

    @classmethod
    def zeros(cls, batch: int, heads: int, head_dim: int, dtype: jnp.dtype):
        return cls(kv=jnp.zeros((batch, heads, head_dim, head_dim), dtype=dtype))


class GLAReferenceKernel(SelectiveKernelProtocol):
    def __init__(self, *, dtype: jnp.dtype = jnp.float32):
        self.dtype = dtype

    def forward_chunk(
        self,
        params: GLAKernelParams,
        inputs: GLAKernelInputs,
        state: Optional[GLAKernelState],
        *,
        chunk_size: int,
    ) -> tuple[jax.Array, GLAKernelState]:
        # Cast inputs
        q = inputs.q.astype(self.dtype)
        k = inputs.k.astype(self.dtype)
        v = inputs.v.astype(self.dtype)
        g = inputs.g.astype(self.dtype)

        batch, heads, seq_len, head_dim = q.shape
        state = state or GLAKernelState.zeros(batch, heads, head_dim, self.dtype)

        # Pad to chunk_size
        num_chunks = math.ceil(seq_len / chunk_size)
        pad = num_chunks * chunk_size - seq_len
        if pad > 0:
            q = jnp.pad(q, ((0,0), (0,0), (0,pad), (0,0)))
            k = jnp.pad(k, ((0,0), (0,0), (0,pad), (0,0)))
            v = jnp.pad(v, ((0,0), (0,0), (0,pad), (0,0)))
            g = jnp.pad(g, ((0,0), (0,0), (0,pad)))

        # Reshape into chunks
        q = q.reshape(batch, heads, num_chunks, chunk_size, head_dim)
        k = k.reshape(batch, heads, num_chunks, chunk_size, head_dim)
        v = v.reshape(batch, heads, num_chunks, chunk_size, head_dim)
        g = g.reshape(batch, heads, num_chunks, chunk_size)

        def chunk_scan(carry_kv, chunk_data):
            q_c, k_c, v_c, g_c = chunk_data
            # Compute gated linear attention update
            # (simplified; real GLA has cumsum gating and normalization)
            g_exp = jnp.exp(g_c)[..., None, :]  # [batch, heads, chunk, 1]
            kv_local = jnp.einsum('bhcd,bhce->bhde', k_c, v_c * g_exp)
            kv_new = carry_kv + kv_local
            o_c = jnp.einsum('bhcd,bhde->bhce', q_c, kv_new)
            return kv_new, o_c

        final_kv, outputs = jax.lax.scan(
            chunk_scan,
            state.kv,
            (q, k, v, g),
        )

        outputs = outputs.reshape(batch, heads, num_chunks * chunk_size, head_dim)
        outputs = outputs[:, :, :seq_len, :]
        return outputs, GLAKernelState(kv=final_kv)

    def forward_recurrent(self, params, inputs, state):
        return self.forward_chunk(params, inputs, state, chunk_size=1)

    def get_grid_config(self, *, batch_size, seq_len, feature_dim):
        chunks = max(1, seq_len // 64)
        return GridConfig(block_shape=(feature_dim,), num_programs=(batch_size * chunks,))
```

### `linearnexus/layers/gla.py`

```python
"""GLA (Gated Linear Attention) layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.core import ConfigBase, RecurrentState, select_mode
from linearnexus.kernels import KernelMode
from linearnexus.kernels.gla_reference import (
    GLAKernelInputs,
    GLAKernelParams,
    GLAKernelState,
    GLAReferenceKernel,
)


@dataclass
class GLAConfig(ConfigBase):
    hidden_size: int = 256
    num_heads: int = 8
    head_dim: int = 32
    chunk_size: int = 64
    use_bias: bool = True


@dataclass
class GLALayerState:
    """GLA layer state."""
    ssm_state: RecurrentState
    position: jnp.int32

    @classmethod
    def initialize(cls, *, batch_size: int, num_heads: int, head_dim: int, dtype: jnp.dtype):
        return cls(
            ssm_state=RecurrentState.zeros(
                batch_size=batch_size,
                channels=num_heads,
                state_size=head_dim * head_dim,  # flattened KV state
                dtype=dtype,
            ),
            position=jnp.array(0, dtype=jnp.int32),
        )


class GLALayer(nnx.Module):
    """Gated Linear Attention layer."""

    def __init__(self, rngs: nnx.Rngs, config: GLAConfig):
        self.config = config

        # QKV projections
        self.qkv_proj = nnx.Linear(
            config.hidden_size,
            3 * config.num_heads * config.head_dim,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        # Gate projection
        self.g_proj = nnx.Linear(
            config.hidden_size,
            config.num_heads,
            use_bias=False,
            rngs=rngs,
        )
        # Output projection
        self.o_proj = nnx.Linear(
            config.num_heads * config.head_dim,
            config.hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        self.kernel = GLAReferenceKernel(dtype=jnp.float32)

    def init_state(self, batch_size: int, dtype: jnp.dtype) -> GLALayerState:
        return GLALayerState.initialize(
            batch_size=batch_size,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dtype=dtype,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        state: Optional[GLALayerState] = None,
        attention_mask: Optional[jax.Array] = None,
        mode: Optional[KernelMode] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[jax.Array, GLALayerState]:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        chunk_size = chunk_size or self.config.chunk_size

        if state is None:
            state = self.init_state(batch_size, dtype)
        if mode is None:
            mode = select_mode(seq_len, threshold=chunk_size)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[..., None]

        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.config.num_heads, self.config.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2).transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)

        # Gate projection
        g = self.g_proj(hidden_states)  # [batch, seq, heads]
        g = g.transpose(0, 2, 1)  # [batch, heads, seq]

        # Prepare kernel inputs
        kernel_inputs = GLAKernelInputs(q=q, k=k, v=v, g=g)
        kernel_params = GLAKernelParams()
        # Reshape state for kernel
        kv_state = state.ssm_state.value.reshape(batch_size, self.config.num_heads, self.config.head_dim, self.config.head_dim)
        kernel_state = GLAKernelState(kv=kv_state)

        # Call kernel
        if mode == KernelMode.CHUNK:
            outputs, new_kernel_state = self.kernel.forward_chunk(
                kernel_params, kernel_inputs, kernel_state, chunk_size=chunk_size
            )
        else:
            outputs, new_kernel_state = self.kernel.forward_recurrent(
                kernel_params, kernel_inputs, kernel_state
            )

        # Transform back: [batch, heads, seq, head_dim] → [batch, seq, heads * head_dim]
        outputs = outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        outputs = self.o_proj(outputs)

        # Update state
        new_ssm_value = new_kernel_state.kv.reshape(batch_size, self.config.num_heads, -1)
        new_state = GLALayerState(
            ssm_state=state.ssm_state.update(new_ssm_value),
            position=state.position + seq_len,
        )

        return outputs, new_state
```

### Register and test

```python
# linearnexus/registry.py
from linearnexus.kernels.gla_reference import GLAReferenceKernel
from linearnexus.layers.gla import GLAConfig, GLALayer

KERNEL_REGISTRY["gla:reference"] = GLAReferenceKernel
LAYER_REGISTRY["gla"] = (GLALayer, GLAConfig)
```

```python
# tests/test_gla_layer.py
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.layers.gla import GLAConfig, GLALayer
from tests.helpers.parity import assert_chunk_recurrent_parity


def test_gla_chunk_recurrent_parity():
    rngs = nnx.Rngs(0)
    config = GLAConfig(hidden_size=32, num_heads=4, head_dim=8)
    layer = GLALayer(rngs, config)

    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, (2, 16, config.hidden_size))

    assert_chunk_recurrent_parity(layer, inputs, chunk_size=4)
```

---

## Summary Checklist

When adding a new layer, ensure:

- [ ] Kernel data structures defined (`*KernelParams`, `*KernelInputs`, `*KernelState`)
- [ ] Reference kernel implements `SelectiveKernelProtocol` (pure JAX)
- [ ] Layer config inherits `ConfigBase`
- [ ] Layer state composes `core.cache` abstractions
- [ ] Layer uses `core/` utilities (conv, gating, padding, mode selection)
- [ ] Kernel and layer registered in `registry.py`
- [ ] Layer tests use `tests.helpers.parity`
- [ ] (Optional) Pallas kernel implemented and registered separately
- [ ] (Optional) Kernel parity tests compare Pallas vs reference
- [ ] All shapes documented inline with comments

---

**Questions or improvements?** Open a GitHub issue with the `documentation` label.
