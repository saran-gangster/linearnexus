# LinearNexus Contributor Guide

**Target Audience**: New contributors wanting to understand the codebase architecture, selective state-space mathematics, JAX/Flax NNx patterns, and kernel design principles.

**Last Updated**: November 18, 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Codebase Architecture Deep Dive](#codebase-architecture-deep-dive)
4. [JAX & Flax NNx Primer](#jax--flax-nnx-primer)
5. [Kernel Implementation Walkthrough](#kernel-implementation-walkthrough)
6. [Layer Implementation Walkthrough](#layer-implementation-walkthrough)
7. [Testing Philosophy](#testing-philosophy)
8. [Development Workflow](#development-workflow)
9. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
10. [Next Steps for Contributors](#next-steps-for-contributors)

---

## Prerequisites

### Required Knowledge

**Must Have**:
- Python 3.10+ fundamentals (type hints, dataclasses, protocols)
- Linear algebra (matrix multiplication, eigenvalues, state-space representations)
- Basic understanding of neural network training (forward/backward passes, gradients)

**Should Have**:
- Familiarity with PyTorch or TensorFlow (helps understand design patterns)
- RNN/LSTM concepts (hidden states, sequential processing)
- Attention mechanisms (queries, keys, values)

**Nice to Have**:
- JAX transformations (`jit`, `grad`, `vmap`)
- GPU programming concepts (memory hierarchy, tiling)
- State-space models from control theory

### Environment Setup

```bash
# Clone and navigate
cd LinearNexus

# Install dependencies (JAX CPU for development)
pip install -e .[dev]

# Verify installation
python -c "import jax; import flax.nnx as nnx; print('JAX:', jax.__version__)"
pytest tests/test_mamba_layer.py
python examples/run_mamba_reference.py --batch 2 --seq 16 --hidden 64
```

**Expected Output**:
```
JAX: 0.4.28 (or later)
======================== 2 passed in ~8s ========================
Output shape: (2, 16, 64)
Sample (first token): [...]
```

---

## Mathematical Foundations

### The Attention Problem

**Standard (Softmax) Attention** has quadratic complexity:

```python
# Shape annotations: B=batch, N=seq_len, D=hidden_dim, K=head_dim
Q, K, V = project(x)  # [B, N, D] → [B, N, K] each

# Compute attention scores
scores = Q @ K.T  # [B, N, N] ← QUADRATIC in sequence length!
attn = softmax(scores)
output = attn @ V  # [B, N, N] @ [B, N, K] → [B, N, K]
```

**Problem**: For sequence length `N=4096`, the attention matrix is `4096²=16M` elements per head. Memory and compute explode.

### Linear Attention Insight

**Key Observation**: Matrix multiplication is **associative**:
```
(Q @ K.T) @ V  ==  Q @ (K.T @ V)
```

If we compute `K.T @ V` first, we get a state matrix of size `[K, V]` (typically 64×64 or 128×128) that's **independent of sequence length**!

```python
# Linear complexity version
state = K.T @ V  # [K, N] @ [N, V] → [K, V] ← constant size!
output = Q @ state  # [N, K] @ [K, V] → [N, V]
```

**Complexity**:
- Standard attention: `O(N² × D)` time, `O(N²)` memory
- Linear attention: `O(N × D²)` time, `O(D²)` memory

### State-Space Models (SSMs)

SSMs generalize linear attention by introducing **temporal dynamics**:

```python
# Discrete-time state-space equations
h[t] = A @ h[t-1] + B @ x[t]  # State update
y[t] = C @ h[t] + D @ x[t]     # Output equation
```

Where:
- `h[t]`: Hidden state at time `t` (shape: `[state_size]`)
- `A`: State transition matrix (how past influences present)
- `B`: Input projection (how current input updates state)
- `C`: Output projection (how state produces output)
- `D`: Direct feedthrough (skip connection)

**Why SSMs?**
- RNNs use fixed `A` (not input-dependent) → limited expressiveness
- Attention uses identity `A` (no temporal structure) → wastes capacity
- **Selective SSMs** (Mamba) make `A, B, C` input-dependent → best of both worlds

### Mamba's Selective Mechanism

Mamba introduces **selectivity** by computing `Δ, B, C` from the input:

```python
# Input-dependent parameters
Δ = softplus(dt_proj(time_step_proj(x)))  # Discretization step size
B = input_proj_B(x)  # Input coupling
C = input_proj_C(x)  # Output coupling

# Discretize continuous system
A_discrete = exp(A_log * Δ)  # A_log is learned (input-independent)
B_discrete = Δ * B

# Recurrent state update
h[t] = A_discrete * h[t-1] + B_discrete * x[t]
y[t] = C @ h[t] + D * x[t]
```

**Key Innovation**: `Δ` acts as a **learned forgetting factor**. Large `Δ` means "pay attention to current input", small `Δ` means "rely on past state".

---

## Codebase Architecture Deep Dive

### Directory Structure

```
linearnexus/
├── __init__.py              # Public API exports
├── kernels/                 # Low-level compute primitives
│   ├── __init__.py
│   ├── base.py             # Kernel protocol & config
│   └── mamba_reference.py  # Pure JAX selective scan
├── layers/                  # High-level layer modules
│   ├── __init__.py
│   └── mamba.py            # NNx Mamba layer
└── models/                  # Model builders (future)
    └── __init__.py

tests/
├── conftest.py             # Pytest configuration
└── test_mamba_layer.py     # Layer correctness tests

examples/
└── run_mamba_reference.py  # Minimal usage demo
```

### Layered Abstraction Philosophy

```
┌─────────────────────────────────────────────────┐
│  Models Layer (models/)                         │
│  - Full transformer stacks                      │
│  - Training loops                               │
│  - Checkpoint management                        │
└────────────────┬────────────────────────────────┘
                 │ uses
┌────────────────▼────────────────────────────────┐
│  Layers Layer (layers/)                         │
│  - Flax NNx modules                             │
│  - Weight initialization                        │
│  - Cache/state management                       │
│  - Mode switching (chunk vs recurrent)          │
└────────────────┬────────────────────────────────┘
                 │ calls
┌────────────────▼────────────────────────────────┐
│  Kernels Layer (kernels/)                       │
│  - Pure computation (no parameters)             │
│  - Chunk/recurrent implementations              │
│  - Grid configuration                           │
│  - Hardware-specific optimizations (future)     │
└─────────────────────────────────────────────────┘
```

**Design Principle**: Each layer has a **single responsibility** and **well-defined interfaces**. You can swap kernel implementations without touching layers, or swap layer architectures without touching kernels.

---

## JAX & Flax NNx Primer

### JAX Fundamentals

**JAX = NumPy + Autograd + XLA Compilation**

#### 1. Functional Programming

JAX arrays are **immutable**:

```python
import jax.numpy as jnp

# BAD: Tries to mutate (raises error)
x = jnp.array([1, 2, 3])
x[0] = 10  # ERROR: JAX arrays are immutable

# GOOD: Create new array
x = jnp.array([1, 2, 3])
x = x.at[0].set(10)  # Returns new array: [10, 2, 3]
```

**Why?** Immutability enables safe parallelization and automatic differentiation.

#### 2. Pseudo-Randomness

JAX uses **explicit PRNG keys** (no global random state):

```python
import jax.random as jr

# Create root key
key = jr.PRNGKey(0)

# Split key for independent random streams
key1, key2 = jr.split(key)
x = jr.normal(key1, (10,))  # Random normal
y = jr.normal(key2, (10,))  # Different random normal

# BAD: Reusing same key gives same random numbers
z = jr.normal(key1, (10,))  # z == x (deterministic!)
```

**Why?** Enables reproducibility and safe parallelization.

#### 3. JIT Compilation

`jax.jit` compiles Python functions to XLA (optimized machine code):

```python
import jax
import jax.numpy as jnp

def slow_function(x):
    return jnp.sum(x ** 2)

fast_function = jax.jit(slow_function)

# First call: compile + run (slow)
x = jnp.ones((1000,))
result = fast_function(x)  # ~100ms

# Subsequent calls: run compiled code (fast)
result = fast_function(x)  # ~0.1ms (1000x faster!)
```

**Constraints**:
- Function must be **pure** (no side effects, no global state)
- Array shapes must be **static** (or use `static_argnums`)

#### 4. Automatic Differentiation

```python
def loss_fn(params, x, y):
    pred = params['w'] @ x + params['b']
    return jnp.mean((pred - y) ** 2)

# Get gradient function
grad_fn = jax.grad(loss_fn)

params = {'w': jnp.ones((10, 5)), 'b': jnp.zeros((10,))}
x = jnp.ones((5,))
y = jnp.zeros((10,))

# Compute gradients w.r.t. params
grads = grad_fn(params, x, y)
# grads = {'w': [...], 'b': [...]}
```

### Flax NNx (The New API)

**NNx vs Linen (Old API)**:

| Aspect | Linen (Old) | NNx (New) |
|--------|-------------|-----------|
| **Style** | Functional (stateless) | Pythonic (stateful) |
| **Parameters** | Separate dict | Embedded in module |
| **Mutation** | Immutable trees | Mutable references |
| **Initialization** | Lazy (shape inference) | Eager (explicit shapes) |
| **Learning Curve** | Steep | Gentle |

#### NNx Module Example

```python
import flax.nnx as nnx
import jax.numpy as jnp

class MyModule(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        # Parameters are created here (like PyTorch)
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.bias = nnx.Param(jnp.zeros((output_dim,)))
    
    def __call__(self, x):
        # Forward pass (like PyTorch)
        return self.linear(x) + self.bias.value

# Usage
rngs = nnx.Rngs(0)  # Random number generator
model = MyModule(input_dim=10, output_dim=5, rngs=rngs)

# Parameters are accessible
print(model.linear.kernel.value.shape)  # (10, 5)

# Forward pass
x = jnp.ones((10,))
y = model(x)  # (5,)
```

#### Parameter Types

```python
# Trainable parameter
self.weight = nnx.Param(jnp.ones((10, 5)))

# Non-trainable buffer (e.g., running stats)
self.running_mean = nnx.Variable(jnp.zeros((5,)))

# Access values
w = self.weight.value  # Get underlying array
self.running_mean.value = new_mean  # Update (mutable!)
```

#### Training Loop Pattern

```python
import optax

# Create model and optimizer
model = MyModule(input_dim=10, output_dim=5, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Training step
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        pred = model(x)
        return jnp.mean((pred - y) ** 2)
    
    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Update parameters
    optimizer.update(grads)
    
    return loss

# Run training
for epoch in range(100):
    loss = train_step(model, optimizer, x_batch, y_batch)
```

---

## Kernel Implementation Walkthrough

### Protocol Definition (`kernels/base.py`)

```python
from typing import Protocol, Tuple
import jax

class SelectiveKernelProtocol(Protocol):
    """Contract all selective SSM kernels must satisfy."""
    
    def forward_chunk(
        self,
        params,      # Learned parameters (A_log, D)
        inputs,      # Runtime inputs (hidden, delta, B, C, gate)
        state,       # Recurrent state from previous chunk
        *,
        chunk_size: int,
    ) -> Tuple[jax.Array, object]:
        """Process sequence in chunks for training."""
    
    def forward_recurrent(
        self, 
        params, 
        inputs, 
        state
    ) -> Tuple[jax.Array, object]:
        """Process one token at a time for inference."""
    
    def get_grid_config(
        self, 
        *, 
        batch_size: int, 
        seq_len: int, 
        feature_dim: int
    ) -> GridConfig:
        """Return launch configuration for Pallas kernels."""
```

**Why Protocols?**
- Like interfaces in Java/TypeScript
- Type checker verifies all implementations conform
- Enables swapping kernel backends (reference → Triton → Mosaic)

### Reference Kernel (`kernels/mamba_reference.py`)

#### Data Structures

```python
@dataclass
class MambaKernelParams:
    """Learned parameters (input-independent)."""
    a_log: jax.Array  # [intermediate, ssm_state] - log of A matrix
    d: jax.Array      # [intermediate] - direct feedthrough

@dataclass
class MambaKernelInputs:
    """Runtime inputs (input-dependent)."""
    hidden: jax.Array  # [batch, intermediate, seq]
    delta: jax.Array   # [batch, intermediate, seq] - discretization step
    B: jax.Array       # [batch, seq, ssm_state] - input projection
    C: jax.Array       # [batch, seq, ssm_state] - output projection
    gate: jax.Array    # [batch, intermediate, seq] - gating mechanism

@dataclass
class MambaKernelState:
    """Recurrent state carried between chunks."""
    ssm: jax.Array  # [batch, intermediate, ssm_state]
```

**Design Note**: We separate `params` (cached, gradient-tracked) from `inputs` (recomputed each forward pass) for memory efficiency.

#### Chunk Processing Logic

```python
def forward_chunk(self, params, inputs, state, *, chunk_size):
    # 1. Extract and reshape inputs
    batch_size, intermediate, seq_len = inputs.hidden.shape
    ssm_state_dim = params.a_log.shape[-1]
    
    # 2. Pad sequence to multiple of chunk_size
    num_chunks = math.ceil(seq_len / chunk_size)
    pad = num_chunks * chunk_size - seq_len
    if pad > 0:
        # Zero-pad all tensors
        hidden = jnp.pad(inputs.hidden, ((0,0), (0,0), (0,pad)))
    
    # 3. Reshape into chunks [num_chunks, batch, intermediate, chunk_size]
    hidden_chunks = hidden.reshape(batch, intermediate, num_chunks, chunk_size)
    hidden_chunks = hidden_chunks.transpose(2, 0, 1, 3)  # [chunks, batch, ...]
    
    # 4. Discretize continuous system parameters
    a = -jnp.exp(params.a_log)  # Negative for stability
    d = params.d
    
    # 5. Process chunks sequentially with state handoff
    def chunk_scan(carry, chunk_inputs):
        ssm_state = carry
        h_chunk, delta_chunk, gate_chunk, B_chunk, C_chunk = chunk_inputs
        
        # Process each token in chunk
        def step(carry_inner, step_inputs):
            ssm_state_inner = carry_inner
            h_t, delta_t, gate_t, B_t, C_t = step_inputs
            
            # Discretize A and B (input-dependent!)
            A_discrete = jnp.exp(a[None, :, :] * delta_t[:, :, None])
            B_discrete = delta_t[:, :, None] * B_t[:, None, :]
            
            # State update: h[t] = A*h[t-1] + B*x[t]
            deltaB_u = B_discrete * h_t[:, :, None]
            ssm_state_inner = A_discrete * ssm_state_inner + deltaB_u
            
            # Output: y[t] = C @ h[t] + D * x[t]
            y = jnp.einsum("bis,bs->bi", ssm_state_inner, C_t)
            y = y + d[None, :] * h_t
            y = y * gate_t  # Gating
            
            return ssm_state_inner, y
        
        # Scan over tokens in chunk
        carry_out, outputs = jax.lax.scan(step, ssm_state, chunk_inputs)
        return carry_out, outputs
    
    # 6. Scan over chunks
    final_state, chunk_outputs = jax.lax.scan(chunk_scan, state.ssm, all_chunks)
    
    # 7. Reshape back and remove padding
    outputs = chunk_outputs.reshape(batch, intermediate, seq_len)
    return outputs, MambaKernelState(ssm=final_state)
```

#### Why Chunking?

**Problem**: Full sequence scan is sequential (can't parallelize).

**Solution**: Process `chunk_size` tokens in parallel, then update state:

```
Sequence: [token0, token1, ..., token63] [token64, ..., token127] ...
           \________________/              \___________________/
                Chunk 0                         Chunk 1
                  ↓                                ↓
            Parallel scan                     Parallel scan
                  ↓                                ↓
            State h[63] ─────────────────────→ State h[127] ───→ ...
```

**Tradeoff**:
- Larger chunks: More parallelism, but longer dependency chains
- Smaller chunks: Less parallelism, but faster state updates
- Sweet spot: **64-128 tokens** per chunk

---

## Layer Implementation Walkthrough

### Configuration (`layers/mamba.py`)

```python
@dataclass
class MambaConfig:
    hidden_size: int = 256          # Model dimension
    state_size: int = 16            # SSM state dimension
    conv_kernel: int = 4            # Causal conv kernel size
    use_conv_bias: bool = True
    intermediate_size: int = 256    # Projection dimension
    time_step_rank: int = 128       # Rank of Δ projection
    use_bias: bool = True
    hidden_act: str = "silu"        # Activation function
    chunk_size: int = 64            # Default chunk size
```

**Key Parameters**:
- `state_size`: Larger → more memory, better long-range modeling
- `intermediate_size`: Typically `1.5x` to `2x` hidden_size (like FFN expansion)
- `time_step_rank`: Controls Δ computation rank (lower → fewer params)

### Layer State Management

```python
@dataclass
class MambaLayerState:
    """State tracked for autoregressive generation."""
    conv_buffer: jax.Array  # [batch, conv_kernel-1, intermediate]
    ssm_state: jax.Array    # [batch, intermediate, state_size]
    position: jnp.int32     # Current token position
```

**Why Separate State?**
- Training: No state needed (each example is independent)
- Inference: State carries context from previous tokens
- KV-cache pattern: Similar to transformer caching but different shape

### Depthwise Convolution

```python
def _depthwise_conv1d(inputs, weight, bias, *, cache=None):
    """Causal conv1d with depthwise groups (one filter per channel)."""
    batch, seq_len, channels = inputs.shape
    kernel_size = weight.shape[0]
    
    # Prepend cache (previous tokens) for causality
    if cache is None:
        cache = jnp.zeros((batch, kernel_size-1, channels))
    conv_input = jnp.concatenate([cache, inputs], axis=1)
    
    # Depthwise convolution
    weight_dw = weight[:, None, :]  # [kernel, 1, channels]
    conv_output = jax.lax.conv_general_dilated(
        conv_input,
        weight_dw,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=channels,  # ← Depthwise (1 filter per channel)
    )
    
    # Update cache with last (kernel_size-1) tokens
    new_cache = conv_input[:, -(kernel_size-1):, :]
    return conv_output, new_cache
```

**Why Depthwise?**
- Regular conv: `kernel_size × in_channels × out_channels` params
- Depthwise: `kernel_size × channels` params (much fewer!)
- Similar to MobileNet design: cheap local feature extraction

### Forward Pass Flow

```python
def __call__(self, hidden_states, *, state=None, mode=KernelMode.CHUNK):
    # 1. Input projection (expand + split)
    projected = self.in_proj(hidden_states)  # [B, S, 2*I]
    hidden, gate = jnp.split(projected, 2, axis=-1)  # [B, S, I] each
    
    # 2. Causal convolution (local mixing)
    conv_out, conv_cache = _depthwise_conv1d(hidden, self.conv_weight.value, ...)
    conv_out = self.activation(conv_out)  # SiLU
    
    # 3. Compute selective parameters (Δ, B, C)
    x_proj_out = self.x_proj(conv_out)  # [B, S, rank + 2*state_size]
    time_step, B, C = jnp.split(x_proj_out, [...], axis=-1)
    delta = jax.nn.softplus(self.dt_proj(time_step))  # Δ > 0
    
    # 4. Prepare kernel inputs (transpose for [B, I, S] layout)
    hidden_kernel = jnp.swapaxes(conv_out, 1, 2)
    delta_kernel = jnp.swapaxes(delta, 1, 2)
    gate_kernel = jnp.swapaxes(self.activation(gate), 1, 2)
    
    kernel_inputs = MambaKernelInputs(
        hidden=hidden_kernel,
        delta=delta_kernel,
        B=B,
        C=C,
        gate=gate_kernel,
    )
    
    # 5. Call selective scan kernel
    kernel_outputs, new_kernel_state = self.kernel.forward_chunk(
        kernel_params, kernel_inputs, kernel_state, chunk_size=chunk_size
    )
    
    # 6. Output projection
    kernel_outputs = kernel_outputs.transpose(0, 2, 1)  # [B, S, I]
    contextualized = self.out_proj(kernel_outputs)  # [B, S, H]
    
    return contextualized, new_state
```

**Critical Details**:
1. **Split gating**: `hidden` goes through SSM, `gate` is direct activation
2. **Softplus on Δ**: Ensures positive discretization step
3. **Layout swap**: Kernel expects `[B, I, S]`, layer works in `[B, S, H]`
4. **State threading**: Input state → kernel → output state

---

## Testing Philosophy

### Test Categories

#### 1. Numerical Correctness

**Goal**: Verify kernel produces same output as reference implementation.

```python
def test_chunk_and_recurrent_paths_align():
    """Chunk mode and recurrent mode must give identical results."""
    rngs = nnx.Rngs(0)
    config = MambaConfig(hidden_size=32, ...)
    layer = MambaLayer(rngs, config)
    
    inputs = jax.random.normal(key, (2, 6, 32))
    
    # Chunk path (parallel)
    chunk_out, _ = layer(inputs, chunk_size=4)
    
    # Recurrent path (sequential)
    recurrent_out = _run_recurrent(layer, inputs)
    
    # Must match within numerical precision
    np.testing.assert_allclose(chunk_out, recurrent_out, rtol=1e-4, atol=1e-4)
```

**Why Important?**
- Ensures optimized kernel matches mathematical spec
- Catches layout bugs (transpose errors, etc.)
- Guards against future regressions

#### 2. Gradient Correctness

```python
def test_gradient_matches_finite_differences():
    """Auto-diff gradients must match numerical gradients."""
    def loss_fn(params):
        output = layer(inputs, params)
        return jnp.sum(output ** 2)
    
    # JAX auto-diff
    grad_auto = jax.grad(loss_fn)(params)
    
    # Finite differences
    grad_numerical = numerical_gradient(loss_fn, params, epsilon=1e-5)
    
    np.testing.assert_allclose(grad_auto, grad_numerical, rtol=1e-3)
```

#### 3. Behavioral Tests

```python
def test_attention_mask_zeroes_out_tokens():
    """Masked tokens should not influence output."""
    mask = jnp.array([[1.0, 1.0, 0.0, 0.0]])
    masked_out, _ = layer(inputs, attention_mask=mask)
    
    # Verify masked positions are zero-like
    assert jnp.allclose(masked_out[:, 2:, :], 0.0, atol=1e-5)
```

### Testing Best Practices

**1. Use Small Shapes**
```python
# Good: Fast, easy to debug
config = MambaConfig(hidden_size=32, state_size=8, ...)

# Bad: Slow, hard to inspect
config = MambaConfig(hidden_size=2048, state_size=64, ...)
```

**2. Test Edge Cases**
```python
# Empty sequence
test_empty_sequence(inputs.shape = (B, 0, H))

# Single token
test_single_token(inputs.shape = (B, 1, H))

# Non-divisible chunk size
test_padding(seq_len=17, chunk_size=8)  # Requires padding
```

**3. Parameterize Tests**
```python
@pytest.mark.parametrize("seq_len", [1, 7, 64, 127])
@pytest.mark.parametrize("chunk_size", [1, 8, 64])
def test_variable_lengths(seq_len, chunk_size):
    # Test all combinations
    ...
```

---

## Development Workflow

### 1. Local Development Loop

```bash
# Edit code
vim linearnexus/layers/mamba.py

# Run tests (fast feedback)
pytest tests/test_mamba_layer.py -v

# Run specific test
pytest tests/test_mamba_layer.py::test_chunk_and_recurrent_paths_align -v

# Run with coverage
pytest tests/ --cov=linearnexus --cov-report=html

# Check types (if configured)
mypy linearnexus/
```

### 2. Debugging JAX Code

**Enable Eager Mode** (disables JIT for easier debugging):
```python
import jax
jax.config.update('jax_disable_jit', True)

# Now you can use print() and pdb.set_trace()
```

**Print Inside JIT** (use `jax.debug.print`):
```python
@jax.jit
def my_function(x):
    jax.debug.print("x shape: {}, x[0]: {}", x.shape, x[0])
    return x * 2
```

**Check for NaN/Inf**:
```python
jax.config.update('jax_debug_nans', True)  # Raises error on NaN
jax.config.update('jax_debug_infs', True)  # Raises error on Inf
```

**Inspect Compiled Code**:
```python
# See XLA HLO (low-level representation)
print(jax.make_jaxpr(my_function)(x))
```

### 3. Performance Profiling

**TensorBoard Profiler**:
```bash
# Instrument code
from jax.profiler import trace
with trace("/tmp/jax-trace"):
    result = model(inputs)

# View in TensorBoard
tensorboard --logdir=/tmp/jax-trace
```

**Memory Profiling**:
```python
# Check peak memory usage
import jax
jax.profiler.start_trace("/tmp/trace")
result = model(inputs)
jax.profiler.stop_trace()

# Analyze HBM (GPU memory) usage
```

---

## Common Pitfalls & Solutions

### 1. Shape Mismatches

**Problem**: Tensors have wrong dimensions after operations.

```python
# Symptom
x = jnp.ones((32, 64, 128))  # [batch, seq, hidden]
y = linear(x)  # Expected [32, 64, 256]
# Error: got [32, 256, 64] instead!
```

**Solution**: Always annotate shapes in comments.
```python
# Input: [batch, seq, hidden]
x = inputs  # [32, 64, 128]

# Transpose to [batch, hidden, seq] for kernel
x = x.transpose(0, 2, 1)  # [32, 128, 64]

# Transpose back
output = output.transpose(0, 2, 1)  # [32, 64, 256]
```

### 2. Random Key Reuse

**Problem**: Same random numbers generated repeatedly.

```python
# BAD
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
y = jax.random.normal(key, (10,))  # y == x (deterministic!)
```

**Solution**: Always split keys.
```python
# GOOD
key = jax.random.PRNGKey(0)
key, subkey1 = jax.random.split(key)
key, subkey2 = jax.random.split(key)
x = jax.random.normal(subkey1, (10,))
y = jax.random.normal(subkey2, (10,))  # y != x
```

### 3. Forgetting `jax.jit` Constraints

**Problem**: Side effects inside JIT-compiled functions.

```python
# BAD
@jax.jit
def train_step(x):
    print(f"Processing {x.shape}")  # Only prints once (at trace time)!
    return x * 2
```

**Solution**: Use `jax.debug.print` or extract non-JIT logic.
```python
# GOOD
@jax.jit
def train_step(x):
    jax.debug.print("Processing {}", x.shape)
    return x * 2
```

### 4. Gradient Instability

**Problem**: Loss becomes NaN during training.

**Common Causes**:
- Large learning rate
- Missing softmax temperature scaling
- Overflow in exp() operations

**Solutions**:
```python
# 1. Log-space operations
log_probs = jax.nn.log_softmax(logits)  # Numerically stable
probs = jnp.exp(log_probs)

# 2. Gradient clipping
grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

# 3. Check intermediate values
jax.debug.print("max logit: {}", jnp.max(logits))
```

### 5. Memory Leaks in NNx

**Problem**: GPU memory grows over time.

**Solution**: Use optimizer correctly.
```python
# BAD: Creates new optimizer each step
def train_step(model, x, y):
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # Memory leak!
    ...

# GOOD: Create optimizer once
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
for batch in data:
    train_step(model, optimizer, batch)
```

---

## Next Steps for Contributors

### Beginner Tasks

1. **Add Activation Functions**
   - File: `layers/mamba.py`
   - Add support for `gelu`, `relu`, `swish` activations
   - Test: Verify forward pass works with each activation

2. **Implement Gradient Checkpointing**
   - File: `layers/mamba.py`
   - Wrap kernel call with `jax.checkpoint`
   - Measure memory reduction vs compute overhead

3. **Add Config Validation**
   - File: `layers/mamba.py`
   - Check `state_size > 0`, `chunk_size` divides sequences, etc.
   - Raise helpful errors for invalid configs

### Intermediate Tasks

4. **Optimize Conv1D**
   - File: `layers/mamba.py`
   - Current implementation is generic; specialize for small kernels
   - Benchmark: Compare against current implementation

5. **Add Multi-Head Support**
   - File: `layers/mamba.py`
   - Split `intermediate_size` into `num_heads × head_dim`
   - Enable per-head parallelism

6. **Implement Mamba-2 Kernel**
   - File: `kernels/mamba2_reference.py`
   - Follow Mamba-2 paper (SSD improvements)
   - Test: Parity with Mamba-1 on simple sequences

### Advanced Tasks

7. **Write Triton Kernel**
   - File: `kernels/mamba_triton.py`
   - Implement chunked selective scan in Triton
   - Target: 2-3× faster than reference on GPU

8. **Add Pallas Mosaic GPU Kernel**
   - File: `kernels/mamba_pallas.py`
   - Use software pipelining for overlapped memory/compute
   - Target: Match Triton speed, enable TPU port

9. **Build Full Transformer Stack**
   - File: `models/mamba_lm.py`
   - Stack `N` MambaLayers + LayerNorm + Embedding
   - Add training script with Optax

### Research Extensions

10. **Hybrid Attention**
    - Mix Mamba layers with softmax attention layers
    - Explore: Which layers benefit most from each mechanism?

11. **Long-Context Benchmarks**
    - Test on 32K, 64K, 128K token sequences
    - Compare memory/speed vs standard transformers

12. **Architecture Search**
    - Vary `state_size`, `intermediate_size`, `conv_kernel`
    - Find optimal configs for different sequence lengths

---

## Resources

### Papers

- **Mamba**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- **Mamba-2**: [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- **S4**: [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

### Documentation

- **JAX**: https://jax.readthedocs.io/
- **Flax NNx**: https://flax.readthedocs.io/en/latest/nnx/index.html
- **Pallas**: https://jax.readthedocs.io/en/latest/pallas/index.html
- **Triton**: https://triton-lang.org/

### Code References

- **flash-linear-attention** (PyTorch): https://github.com/sustcsonglin/flash-linear-attention
- **Mamba (Official)** (PyTorch): https://github.com/state-spaces/mamba
- **JAX Examples**: https://github.com/google/jax/tree/main/examples

---

## Getting Help

**Found a bug?** Open an issue with:
- Minimal reproduction (code + error message)
- Environment details (`jax.__version__`, GPU/CPU, etc.)
- Expected vs actual behavior

**Have a question?** Check:
1. This guide (search for keywords)
2. `TECHNICAL_REFERENCE.md` (API details)
3. `ARCHITECTURE.md` (system design)
4. GitHub Issues (maybe someone asked already)

**Want to contribute?** Start with:
1. Pick a task from "Next Steps" above
2. Open an issue titled "WIP: [Task Name]"
3. Submit PR when ready (include tests!)

---

**Welcome to LinearNexus!** We're building the future of efficient sequence modeling, and every contribution matters. Start small, ask questions, and iterate. The codebase is designed to be hackable—dive in and make it your own.

*Happy coding!* 🚀
