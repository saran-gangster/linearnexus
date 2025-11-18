# LinearNexus System Architecture

**Version**: 1.0  
**Last Updated**: November 17, 2024

---

## Table of Contents

1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [System Layers](#system-layers)
4. [Kernel Architecture](#kernel-architecture)
5. [Memory Management](#memory-management)
6. [Execution Flow](#execution-flow)
7. [Performance Optimization Strategy](#performance-optimization-strategy)
8. [Design Patterns](#design-patterns)
9. [Technology Stack](#technology-stack)
10. [Comparison with Alternatives](#comparison-with-alternatives)

---

## Overview

LinearNexus is designed as a modular, high-performance framework for linear and hybrid attention mechanisms in JAX. The architecture emphasizes:

- **Hardware Portability**: Single codebase targeting GPU (via Mosaic) and TPU
- **Performance**: Kernel-level optimizations matching raw CUDA/TPU performance
- **Modularity**: Easy to extend with new attention mechanisms
- **Research-Friendly**: JAX transformations enable rapid experimentation
- **Production-Ready**: Distributed training, checkpointing, and monitoring

### Key Architectural Decisions

1. **Pallas for Kernel Implementation**: Provides hardware abstraction while maintaining performance
2. **Three-Tier Abstraction**: Kernels → Layers → Models for clean separation of concerns
3. **JAX-Native Design**: Leverage JAX's transformations (jit, grad, vmap, pmap)
4. **Functional Programming**: Immutable data structures and pure functions
5. **Configuration-Driven**: YAML/Python configs for model and training specifications

---

## Architectural Principles

### 1. Separation of Concerns

```
Models (High-level API)
   ↓
Layers (Attention mechanisms)
   ↓
Kernels (Low-level compute)
   ↓
Hardware (GPU/TPU)
```

Each layer has well-defined interfaces and responsibilities:

- **Kernels**: Raw computation, memory management, hardware optimization
- **Layers**: Attention logic, state management, gradient computation
- **Models**: Architecture composition, configuration, training/inference modes

### 2. Hardware Abstraction

Pallas provides a unified programming model:

```python
# Single kernel implementation
def attention_kernel(q_ref, k_ref, v_ref, o_ref):
    # ... computation ...
    pass

# Compiles to different backends
pl.pallas_call(
    attention_kernel,
    # ... specifications ...
)
# → Triton (GPU) or Mosaic (TPU) automatically
```

### 3. Composability

All components are composable through standard JAX transformations:

```python
# Automatic batching
batched_attention = jax.vmap(attention_layer)

# Automatic differentiation
grad_fn = jax.grad(loss_fn)

# Just-in-time compilation
fast_attention = jax.jit(attention_layer)

# Parallelization
parallel_attention = jax.pmap(attention_layer)
```

### 4. Configuration Management

Hierarchical configuration system:

```yaml
# model_config.yaml
model:
  type: "gla"
  hidden_size: 2048
  num_layers: 24
  num_heads: 16
  
  attention:
    expand_k: 0.5
    expand_v: 1.0
    mode: "chunk"
    chunk_size: 64
  
  training:
    batch_size: 32
    sequence_length: 2048
    mixed_precision: true
```

---

## System Layers

### Layer 1: Kernel Layer (`linearnexus/kernels/`)

**Responsibility**: Low-level compute primitives implementing `SelectiveKernelProtocol`

**Key Components**:

```
kernels/
├── base.py                # SelectiveKernelProtocol, GridConfig, KernelMode
├── mamba_reference.py     # Pure JAX reference kernel (lax.scan chunking)
├── mamba_pallas.py        # Future: Pallas GPU/TPU optimized kernel
└── tuning.py              # Future: Grid config estimation and tuning utilities
```

### Layer 1.5: Core Runtime Library (`linearnexus/core/`)

**Responsibility**: Shared cross-cutting utilities reused by all layers

**Key Components**:

```
core/
├── cache.py               # RecurrentState, ConvState abstractions
├── padding.py             # compute_unpadded_indices, unpad/pad for varlen
├── gating.py              # Low-rank gate projections, logsigmoid transforms
├── conv.py                # depthwise_conv1d_causal (reusable for SSM layers)
├── config.py              # ConfigBase with serialization/validation
└── mode.py                # select_mode utility for automatic chunk/recurrent switching
```

**Design Principle**: Eliminate duplication by centralizing logic that would otherwise be copied into every layer (padding, caching, gating, conv). All functions are pure, shape-annotated, and `vmap`/`jit`-friendly.

> **Phase 0 snapshot**: `linearnexus/kernels/mamba_reference.py` implements `SelectiveKernelProtocol` purely in JAX (`lax.scan` + chunked execution) so we can validate numerics before layering in Pallas/Triton kernels.

**Interface Design**:

```python
class AttentionKernel(ABC):
    """Base interface for attention kernels."""
    
    @abstractmethod
    def forward_kernel(
        self,
        q_ref: Ref,
        k_ref: Ref,
        v_ref: Ref,
        o_ref: Ref,
        **params
    ) -> None:
        """Forward pass kernel."""
        pass
    
    @abstractmethod
    def backward_kernel(
        self,
        q_ref: Ref,
        k_ref: Ref,
        v_ref: Ref,
        do_ref: Ref,
        dq_ref: Ref,
        dk_ref: Ref,
        dv_ref: Ref,
        **params
    ) -> None:
        """Backward pass kernel."""
        pass
    
    def get_grid_config(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int
    ) -> GridConfig:
        """Compute optimal grid configuration."""
        pass
```

### Layer 2: Attention Layer (`linearnexus/layers/`)

**Responsibility**: High-level attention mechanism implementation

**Key Components**:

```python
class LinearAttentionLayer(nn.Module):
    """Base class for linear attention layers."""
    
    mode: str  # 'chunk', 'fused_chunk', 'fused_recurrent'
    hidden_size: int
    num_heads: int
    head_dim: int
    
    # Projections
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear
    
    # Optional components
    gate_proj: Optional[nn.Linear]
    norm: Optional[nn.Module]
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        cache: Optional[Cache] = None,
    ) -> Tuple[jnp.ndarray, Optional[Cache]]:
        """Forward pass with caching support."""
        pass
```

    > **Phase 0 snapshot**: `linearnexus/layers/mamba.py` hosts the NNx Mamba block using `core.conv.depthwise_conv1d_causal` and `core.cache` abstractions. Projections, shape transforms, and kernel invocation are handled here; all reusable logic lives in `core/`.

**Features**:
- Multiple execution modes (chunk, recurrent) via `SelectiveKernelProtocol`
- Automatic mode selection using `core.mode.select_mode(seq_len)`
- State management (`core.cache.RecurrentState`, `ConvState`) for autoregressive generation
- Variable-length sequence support via `core.padding` helpers (future)
- Clean separation: layers wire components; `core/` provides reusable building blocks

### Layer 2.5: Registry and Feature Organization (`linearnexus/registry.py`)

**Responsibility**: Map feature names to kernel/layer/config classes for automated tooling

**Structure**:

```python
KERNEL_REGISTRY = {
    "mamba:reference": MambaReferenceKernel,
    "mamba:pallas": MambaPallasKernel,  # future
}

LAYER_REGISTRY = {
    "mamba": (MambaLayer, MambaConfig),
    # future: "gla", "deltanet", etc.
}

MODEL_REGISTRY = {
    # future: "mamba_lm_tiny": (MambaLMConfig, MambaLMModel),
}
```

**Usage**:
- Tests iterate `LAYER_REGISTRY` to run parity/gradient checks on all registered features
- Benchmarks script walks registry to measure throughput per mechanism
- Docs generator builds feature tables from registry metadata
- Eliminates hardcoded imports in training scripts

### Layer 3: Model Layer (`linearnexus/models/`)

**Responsibility**: Complete transformer models

**Architecture**:

```python
class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    attn_norm: nn.Module
    attention: LinearAttentionLayer
    mlp_norm: nn.Module
    mlp: nn.Module
    
    def __call__(self, x, **kwargs):
        # Pre-norm architecture
        residual = x
        x = self.attn_norm(x)
        x, cache = self.attention(x, **kwargs)
        x = residual + x
        
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x, cache


class TransformerModel(nn.Module):
    """Full transformer model."""
    
    embedding: nn.Embed
    layers: List[TransformerBlock]
    norm: nn.Module
    lm_head: nn.Linear
    
    def __call__(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        
        caches = []
        for layer in self.layers:
            x, cache = layer(x, **kwargs)
            caches.append(cache)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, caches
```

---

## Kernel Architecture

### Chunk-Based Processing

Linear attention naturally decomposes into chunk-wise computation:

```
Input Sequence: [T tokens]
         ↓
Chunk Division: [T/C chunks of C tokens]
         ↓
Parallel Processing: Process each chunk independently
         ↓
Recurrence: Update states between chunks
         ↓
Output Sequence: [T tokens]
```

**Implementation**:

```python
def chunk_attention_kernel(
    q_ref,  # [B, T, H, K] queries
    k_ref,  # [B, T, H, K] keys
    v_ref,  # [B, T, H, V] values
    h_ref,  # [B, num_chunks, H, K, V] chunk states
    o_ref,  # [B, T, H, V] output
):
    # Get chunk ID
    chunk_id = pl.program_id(0)
    
    # Load current chunk
    q_chunk = q_ref[chunk_id]
    k_chunk = k_ref[chunk_id]
    v_chunk = v_ref[chunk_id]
    
    # Intra-chunk attention (parallel)
    attn_chunk = compute_intra_chunk_attention(q_chunk, k_chunk, v_chunk)
    
    # Inter-chunk recurrence (sequential)
    h_prev = h_ref[chunk_id - 1] if chunk_id > 0 else 0
    h_curr = update_state(h_prev, k_chunk, v_chunk)
    h_ref[chunk_id] = h_curr
    
    # Combine intra-chunk and recurrent contributions
    o_chunk = attn_chunk + (q_chunk @ h_prev)
    o_ref[chunk_id] = o_chunk
```

### Memory Layout

**GPU (Mosaic Backend)**:

```
Global Memory (HBM)
      ↕ TMA (Tensor Memory Accelerator)
Shared Memory (SMEM) - 128KB per SM
      ↕
Registers - 256KB per SM
      ↕
Tensor Cores - MMA/WGMMA instructions
```

**Optimization Strategy**:
1. **Tile Inputs**: Break large tensors into tiles that fit in SMEM
2. **Software Pipelining**: Overlap memory transfers with computation
3. **Swizzling**: Optimize memory access patterns to avoid bank conflicts
4. **Warp Specialization**: Dedicate warps to memory vs. compute

**TPU (Mosaic Backend)**:

```
HBM (High Bandwidth Memory)
      ↕
VMEM (Vector Memory) - 32MB per core
      ↕
MXU (Matrix Unit) - 128x128 BF16 MAC per cycle
```

**Optimization Strategy**:
1. **Large Tiles**: TPU benefits from larger tile sizes (512x512)
2. **Pipeline Parallelism**: Multiple stages in flight simultaneously
3. **Layout Transforms**: Ensure optimal tensor layouts for MXU
4. **Minimize HBM Traffic**: Maximize reuse in VMEM

### Kernel Fusion

Fuse multiple operations to minimize memory round-trips:

```python
# Unfused (3 kernel launches, 3 memory round-trips)
q = q @ w_q      # Kernel 1
q = layernorm(q)  # Kernel 2
q = silu(q)      # Kernel 3

# Fused (1 kernel launch, 1 memory round-trip)
def fused_qkv_proj_kernel(x_ref, w_q_ref, w_k_ref, w_v_ref, 
                          q_ref, k_ref, v_ref):
    x = x_ref[:]
    q = (x @ w_q_ref[:])
    q = layernorm_inline(q)
    q = silu_inline(q)
    q_ref[:] = q
    # Similar for k, v...
```

**Commonly Fused Operations**:
- QKV projection + LayerNorm + Activation
- Attention + Output projection
- MLP + Gate projection
- LayerNorm + Linear
- Softmax + Masking

---

## Memory Management

### Memory Budget Calculation

**GPU Example (A100 - 80GB)**:

```python
def compute_memory_budget(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    dtype: jnp.dtype
) -> Dict[str, int]:
    bytes_per_elem = jnp.dtype(dtype).itemsize
    
    # Activations
    activation_memory = (
        batch_size * seq_len * hidden_size * 
        num_layers * bytes_per_elem * 2  # forward + backward
    )
    
    # Parameters
    param_memory = (
        num_layers * (
            4 * hidden_size * hidden_size +  # QKV + O projections
            8 * hidden_size * hidden_size    # MLP
        ) * bytes_per_elem
    )
    
    # Optimizer state (Adam)
    optimizer_memory = param_memory * 2  # momentum + variance
    
    # Gradients
    gradient_memory = param_memory
    
    total = (
        activation_memory + 
        param_memory + 
        optimizer_memory + 
        gradient_memory
    )
    
    return {
        'activations': activation_memory,
        'parameters': param_memory,
        'optimizer': optimizer_memory,
        'gradients': gradient_memory,
        'total': total,
    }
```

### Gradient Checkpointing

Trade compute for memory by recomputing activations:

```python
def transformer_block_with_checkpointing(x, params):
    """Transformer block with gradient checkpointing."""
    
    @jax.checkpoint  # Recompute in backward pass
    def attention_and_mlp(x):
        # Attention
        residual = x
        x = layer_norm(x)
        x = attention(x, params)
        x = residual + x
        
        # MLP
        residual = x
        x = layer_norm(x)
        x = mlp(x, params)
        x = residual + x
        
        return x
    
    return attention_and_mlp(x)
```

**Memory Savings**:
- Without checkpointing: O(num_layers × batch × seq_len × hidden_size)
- With checkpointing: O(sqrt(num_layers) × batch × seq_len × hidden_size)

---

## Execution Flow

### Training Flow

```
1. Data Loading
   ├─ Tokenization
   ├─ Batching
   ├─ Sequence packing
   └─ Prefetching
         ↓
2. Forward Pass
   ├─ Embedding lookup
   ├─ Attention layers (chunked)
   │  ├─ QKV projection
   │  ├─ Chunk attention kernel
   │  └─ Output projection
   ├─ MLP layers
   └─ LM head
         ↓
3. Loss Computation
   ├─ Cross-entropy (fused with LM head)
   └─ Auxiliary losses (optional)
         ↓
4. Backward Pass
   ├─ Gradient computation (auto-diff)
   ├─ Gradient checkpointing (recompute)
   └─ Gradient accumulation
         ↓
5. Optimization
   ├─ Gradient clipping
   ├─ Optimizer step (AdamW)
   └─ Learning rate schedule
         ↓
6. Logging & Checkpointing
   ├─ Metrics logging
   ├─ Model checkpointing
   └─ Validation (periodic)
```

### Inference Flow

```
1. Input Processing
   ├─ Tokenization
   └─ Batch preparation
         ↓
2. Prefill Phase (parallel)
   ├─ Process all input tokens
   ├─ Use chunk mode for efficiency
   └─ Initialize KV cache/state
         ↓
3. Decode Phase (autoregressive)
   ├─ Generate one token at a time
   ├─ Use fused recurrent mode
   ├─ Update state incrementally
   └─ Stopping criteria check
         ↓
4. Post-processing
   ├─ Detokenization
   └─ Output formatting
```

**State Management**:

```python
class GenerationCache:
    """Cache for autoregressive generation."""
    
    # Recurrent state per layer
    recurrent_states: List[jnp.ndarray]  # [num_layers, batch, heads, K, V]
    
    # Optional: Conv states for models with short convolutions
    conv_states: Optional[List[jnp.ndarray]]
    
    # Attention states for hybrid models
    attn_states: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]]  # KV cache
    
    def update(self, layer_idx: int, new_state: jnp.ndarray):
        """Update cache for a specific layer."""
        self.recurrent_states[layer_idx] = new_state
```

---

## Performance Optimization Strategy

### 1. Profiling & Bottleneck Analysis

**Tools**:
- JAX Profiler (TensorBoard integration)
- NVIDIA Nsight Compute (GPU)
- Pallas performance counters

**Key Metrics**:
- Kernel execution time
- Memory bandwidth utilization
- Compute utilization (TFLOPS achieved / peak)
- SMEM usage
- Register usage
- Warp occupancy

### 2. Kernel Optimization Techniques

**A. Memory Coalescing**:
```python
# Bad: Strided access
for i in range(seq_len):
    val = x[batch_id, i, head_id, :]  # Strided in batch dimension

# Good: Coalesced access
for i in range(seq_len):
    val = x[batch_id, head_id, i, :]  # Contiguous in last dimension
```

**B. Tiling for Cache Locality**:
```python
# Compute optimal tile size
tile_size = min(
    max_seq_len,
    int(sqrt(smem_size / (3 * head_dim * dtype_size)))  # 3 for Q, K, V
)
```

**C. Instruction-Level Parallelism**:
```python
# Unroll small loops to enable ILP
for i in range(0, seq_len, 4):  # Process 4 at a time
    v0 = load(x[i + 0])
    v1 = load(x[i + 1])
    v2 = load(x[i + 2])
    v3 = load(x[i + 3])
    # ... compute ...
```

### 3. Distributed Training Optimization

**Data Parallelism**:
```python
# Simple SPMD with pmap
@jax.pmap
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['input_ids'])
        loss = cross_entropy(logits, batch['labels'])
        return loss
    
    grads = jax.grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')  # All-reduce
    
    return state.apply_gradients(grads=grads)
```

**Tensor Parallelism**:
```python
# Shard attention heads across devices
mesh = jax.sharding.Mesh(devices, ('data', 'model'))

@functools.partial(
    jax.jit,
    in_shardings=(PartitionSpec('data', None, 'model', None),),  # Shard heads
    out_shardings=(PartitionSpec('data', None, 'model', None),)
)
def attention_layer(x):
    # Automatically handles cross-device communication
    return parallel_attention(x)
```

---

## Design Patterns

### 0. Core Utilities Pattern (Cross-Cutting Concerns)

**Problem**: Layers for different attention mechanisms (Mamba, GLA, DeltaNet) all need padding, gating, caching, and conv logic.

**Solution**: Centralize shared logic in `linearnexus/core/`:

```python
# core/conv.py
def depthwise_conv1d_causal(
    inputs: Array,  # [batch, seq, channels]
    weight: Array,  # [kernel_size, channels]
    bias: Optional[Array],
    *,
    cache: Optional[Array] = None,
) -> tuple[Array, Array]:
    """Reusable depthwise causal conv with optional cache warmstart."""
    # Implementation once, used by all SSM layers
    ...

# layers/mamba.py
from linearnexus.core.conv import depthwise_conv1d_causal

class MambaLayer(nnx.Module):
    def __call__(self, x, *, state=None, ...):
        conv_out, new_cache = depthwise_conv1d_causal(
            x, self.conv_weight.value, self.conv_bias.value, cache=state.conv_buffer
        )
        # Focus on Mamba-specific logic
```

**Benefits**:
- Eliminates duplication: one implementation, many consumers
- Easier to optimize: improve `core.conv` once, all layers benefit
- Clearer layer code: projections + kernel wiring only

### 1. Registry Pattern for Feature Discovery

**Problem**: Tests, benchmarks, and docs need to know what kernels/layers/models exist without hardcoding imports.

**Solution**: Explicit registry in `linearnexus/registry.py`:

```python
from linearnexus.registry import LAYER_REGISTRY

# Tests iterate all layers
for name, (layer_cls, config_cls) in LAYER_REGISTRY.items():
    test_parity(layer_cls, config_cls)

# Benchmarks measure all registered mechanisms
for name in LAYER_REGISTRY:
    benchmark_throughput(name)
```

**Benefits**:
- Automated test coverage: new features auto-included
- Documentation stays in sync: script walks registry
- Easier experimentation: swap mechanisms by name in configs

### 2. Factory Pattern for Kernels

```python
class KernelFactory:
    """Factory for creating optimized kernels based on hardware."""
    
    @staticmethod
    def create_attention_kernel(
        mechanism: str,
        mode: str,
        hardware: str = 'auto'
    ) -> AttentionKernel:
        """Create appropriate kernel based on mechanism and hardware."""
        
        if hardware == 'auto':
            hardware = detect_hardware()
        
        kernel_class = {
            ('retnet', 'chunk', 'gpu'): RetNetChunkGPUKernel,
            ('retnet', 'chunk', 'tpu'): RetNetChunkTPUKernel,
            ('gla', 'chunk', 'gpu'): GLAChunkGPUKernel,
            # ... more combinations
        }[(mechanism, mode, hardware)]
        
        return kernel_class()
```

### 2. Strategy Pattern for Execution Modes

```python
class ExecutionStrategy(ABC):
    """Base class for execution strategies."""
    
    @abstractmethod
    def forward(self, q, k, v, **kwargs):
        pass
    
    @abstractmethod
    def backward(self, dout, **kwargs):
        pass


class ChunkStrategy(ExecutionStrategy):
    """Chunk-based parallel execution."""
    
    def forward(self, q, k, v, chunk_size=64):
        # Chunk processing logic
        pass


class RecurrentStrategy(ExecutionStrategy):
    """Sequential recurrent execution."""
    
    def forward(self, q, k, v):
        # Recurrent processing logic
        pass


class AttentionLayer:
    def __init__(self, strategy: ExecutionStrategy):
        self.strategy = strategy
    
    def __call__(self, q, k, v):
        return self.strategy.forward(q, k, v)
```

### 3. Builder Pattern for Model Configuration

```python
class ModelBuilder:
    """Builder for constructing transformer models."""
    
    def __init__(self):
        self.config = {}
    
    def set_model_size(self, hidden_size, num_layers):
        self.config['hidden_size'] = hidden_size
        self.config['num_layers'] = num_layers
        return self
    
    def set_attention(self, mechanism, **kwargs):
        self.config['attention'] = {
            'mechanism': mechanism,
            **kwargs
        }
        return self
    
    def build(self) -> nn.Module:
        return TransformerModel(**self.config)


# Usage
model = (ModelBuilder()
    .set_model_size(hidden_size=2048, num_layers=24)
    .set_attention('gla', mode='chunk', expand_k=0.5)
    .build())
```

---

## Technology Stack

### Core Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"

# JAX ecosystem
jax = "^0.4.20"
jaxlib = "^0.4.20"
flax = "^0.7.5"
optax = "^0.1.7"

# Utilities
einops = "^0.7.0"
numpy = "^1.24.0"
pyyaml = "^6.0"

# Training
tensorboard = "^2.14.0"
wandb = "^0.15.0"

# Data
grain = "^0.1.0"  # Google's data loading library
tensorflow-datasets = "^4.9.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^23.7.0"
flake8 = "^6.1.0"
mypy = "^1.5.0"
```

### Hardware Requirements

**Minimum**:
- GPU: NVIDIA A100 (40GB) or equivalent
- TPU: TPU v3 or later
- RAM: 64GB system memory
- Storage: 500GB SSD

**Recommended**:
- GPU: NVIDIA H100 (80GB) or A100 (80GB)
- TPU: TPU v4 or v5
- RAM: 128GB system memory
- Storage: 1TB NVMe SSD

---

## Comparison with Alternatives

### vs. flash-linear-attention (PyTorch + Triton)

| Aspect | flash-linear-attention | LinearNexus |
|--------|------------------------|-------------|
| **Framework** | PyTorch | JAX + Flax NNx |
| **Backend** | Triton only | Pallas (Mosaic GPU/TPU + Triton) |
| **GPU Support** | ✓ | ✓ |
| **TPU Support** | ✗ | ✓ |
| **Performance** | Triton-optimized | Pallas-optimized (GPU/TPU) |
| **Transformations** | Manual | JAX auto-diff, vmap, pjit |
| **Composability** | Limited | Full JAX ecosystem |
| **Distributed** | PyTorch DDP/FSDP | JAX pjit/shard_map |
| **Cross-Cutting Code** | Duplicated per feature | Centralized in `core/` |
| **Feature Registry** | Implicit (imports) | Explicit (`registry.py`) |
| **Test Automation** | Manual per feature | Registry-driven harness |

### vs. Standard Transformers (Hugging Face)

| Aspect | Transformers | LinearNexus |
|--------|-------------|-------------|
| **Attention Complexity** | O(n²) | O(n) |
| **Long Context** | Limited | Excellent |
| **Inference Speed** | Standard | 3-10x faster |
| **Training Speed** | Standard | 2-5x faster |
| **Memory Usage** | High | Low |
| **Flexibility** | High-level | High-level + kernel-level |

### vs. JAX Libraries (Equinox, Haiku)

| Aspect | Equinox/Haiku | LinearNexus |
|--------|---------------|-------------|
| **Focus** | General NNs | Linear attention |
| **Custom Kernels** | Minimal | Extensive |
| **Performance** | Standard JAX | Optimized Pallas |
| **Attention Variants** | Few | Many (RetNet, GLA, etc.) |
| **Production Ready** | ✓ | In progress |

---

## Future Extensions

### Planned Features

1. **More Attention Mechanisms**:
   - Sparse attention patterns
   - Block-sparse attention
   - Mixture of attentions

2. **Advanced Optimizations**:
   - INT8 quantization
   - Flash decoding techniques
   - Speculative decoding

3. **Integration**:
   - Hugging Face Transformers compatibility
   - ONNX export
   - TensorFlow Lite conversion

4. **Tooling**:
   - Interactive kernel profiler
   - Automatic hyperparameter tuning
   - Neural architecture search

---

**Document Status**: Living Document  
**Last Review**: November 17, 2024  
**Next Review**: End of Phase 1

For architecture questions or suggestions, please open a GitHub issue with the `architecture` label.
