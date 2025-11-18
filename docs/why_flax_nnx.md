# Why Flax NNX? Design Philosophy

**Understanding the motivation, benefits, and improvements of Flax NNX over Flax Linen.**

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Key Improvements](#key-improvements)
3. [Linen vs NNX Comparison](#linen-vs-nnx-comparison)
4. [When to Use NNX](#when-to-use-nnx)

---

## Core Philosophy

### The Problem with Linen

In 2020, Flax released the **Linen API** with functional/lazy semantics:
- **Pros**: Concise code, automatic shape inference, aligned with Haiku
- **Cons**: Non-Pythonic semantics, surprising behavior, implementation complexity

**Example of Linen Complexity**:
```python
# Linen: Cannot access attributes until runtime
class Block(nn.Module):
    def setup(self):
        self.linear = nn.Dense(10)

block = Block()
block.linear  # AttributeError! (lazy initialization)
```

### NNX's Solution

**Central Idea**: Introduce **reference semantics** into JAX while retaining Linen's strengths.

**Three Principles**:

1. **Pythonic**: Regular Python semantics for modules, including mutability and shared references
2. **Simple**: Complex APIs simplified using Python idioms or removed entirely
3. **Better JAX Integration**: Custom transforms adopt JAX APIs, easier to use JAX transforms directly

---

## Key Improvements

### 1. Inspection (Eager Initialization)

**Linen Problem**: Lazy initialization makes modules hard to inspect.

```python
# Linen: Attributes not available at construction
class Block(nn.Module):
    def setup(self):
        self.linear = nn.Dense(10)

block = Block()
block.linear  # ERROR: AttributeError
```

**NNX Solution**: Eager initialization makes modules immediately inspectable.

```python
# NNX: Attributes available immediately
class Block(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(features, features, rngs=rngs)

block = Block(features=10, rngs=nnx.Rngs(0))
print(block.linear.kernel.shape)  # (10, 10) ✓
```

**Tradeoff**: No shape inference—both input and output shapes must be provided. This allows for more explicit and predictable behavior.

---

### 2. Running Computation (No .apply())

**Linen Problem**: Asymmetry between code inside vs outside `.apply()`.

```python
# Linen: Must use .apply() for all top-level computation
class AutoEncoder(nn.Module):
    def setup(self):
        self.encoder = nn.Dense(10)
        self.decoder = nn.Dense(2)
    
    def __call__(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)

model = AutoEncoder()
params = model.init(random.key(0), x)['params']

# Must use .apply() to call methods
y = model.apply({'params': params}, x)
z = model.apply({'params': params}, x, method='encode')

# Cannot call sub-modules directly (not initialized)
# model.decoder(z)  # ERROR!
```

**NNX Solution**: No special context—call methods directly.

```python
# NNX: Direct method calls
class AutoEncoder(nnx.Module):
    def __init__(self, input_dim: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.encoder = nnx.Linear(input_dim, embed_dim, rngs=rngs)
        self.decoder = nnx.Linear(embed_dim, input_dim, rngs=rngs)
    
    def __call__(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)

model = AutoEncoder(input_dim=2, embed_dim=10, rngs=nnx.Rngs(0))

# Call methods directly (no .apply())
y = model(x)
z = model.encode(x)

# Call sub-modules directly
y = model.decoder(z)  # ✓
```

**Benefit**: `__init__` and `__call__` are not treated differently from other class methods.

---

### 3. State Handling (Mutability)

**Linen Problem**: Complex state management for stateful layers.

```python
# Linen: Must manually handle state collections
class Block(nn.Module):
    train: bool
    
    def setup(self):
        self.linear = nn.Dense(10)
        self.bn = nn.BatchNorm(use_running_average=not self.train)
        self.dropout = nn.Dropout(0.1, deterministic=not self.train)
    
    def __call__(self, x):
        return nn.relu(self.dropout(self.bn(self.linear(x))))

model = Block(train=True)
vs = model.init(random.key(0), x)
params, batch_stats = vs['params'], vs['batch_stats']

# Complex: Must track mutable collections
y, updates = model.apply(
    {'params': params, 'batch_stats': batch_stats},
    x,
    rngs={'dropout': random.key(1)},
    mutable=['batch_stats'],  # Specify what's mutable
)
batch_stats = updates['batch_stats']  # Manual update
```

**NNX Solution**: Mutable state kept inside module.

```python
# NNX: State automatically managed
class Block(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.bn = nnx.BatchNorm(features, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
    
    def __call__(self, x):
        return nnx.relu(self.dropout(self.bn(self.linear(x))))

model = Block(features=10, rngs=nnx.Rngs(0))

# Just call it (state updated automatically)
model.train()  # Set training mode
y = model(x)

model.eval()  # Set eval mode
y = model(x)
```

**Benefits**:
- No need to change training code when adding stateful layers
- State updates happen in-place (Pythonic)
- Train/eval modes switched with `.train()` / `.eval()`

**Easy Implementation**:
```python
# Simplified BatchNorm implementation
class BatchNorm(nnx.Module):
    def __init__(self, features: int, mu: float = 0.95):
        self.scale = nnx.Param(jnp.ones((features,)))
        self.bias = nnx.Param(jnp.zeros((features,)))
        self.mean = nnx.BatchStat(jnp.zeros((features,)))
        self.var = nnx.BatchStat(jnp.ones((features,)))
        self.mu = mu  # Static (not a Variable)
    
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1)
        var = jnp.var(x, axis=-1)
        
        # EMA updates (mutable!)
        self.mean.value = self.mu * self.mean.value + (1 - self.mu) * mean
        self.var.value = self.mu * self.var.value + (1 - self.mu) * var
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        return x * self.scale.value + self.bias.value
```

---

### 4. Model Surgery (Pythonic References)

**Linen Problem**: Difficult due to lazy initialization and separated parameter structure.

```python
# Linen: Cannot replace sub-modules easily
class LoraLinear(nn.Module):
    linear: nn.Dense
    rank: int
    
    @nn.compact
    def __call__(self, x):
        A = self.param(random.normal, (x.shape[-1], self.rank))
        B = self.param(random.normal, (self.rank, self.linear.features))
        return self.linear(x) + x @ A @ B

model = Block(train=True)
# model.linear = LoraLinear(model.linear, rank=5)  # ERROR: not available

# Must manually update params dict
lora_params = model.linear.init(random.key(1), x)
lora_params['linear'] = params['linear']
params['linear'] = lora_params  # Complex manual surgery
```

**NNX Solution**: Direct replacement using Python semantics.

```python
# NNX: Direct sub-module replacement
class LoraLinear(nnx.Module):
    def __init__(self, linear: nnx.Linear, rank: int, *, rngs: nnx.Rngs):
        self.linear = linear
        self.A = nnx.Param(jax.random.normal(rngs(), (linear.in_features, rank)))
        self.B = nnx.Param(jax.random.normal(rngs(), (rank, linear.out_features)))
    
    def __call__(self, x):
        return self.linear(x) + x @ self.A.value @ self.B.value

model = Block(features=10, rngs=nnx.Rngs(0))

# Replace sub-module directly (Pythonic!)
model.linear = LoraLinear(model.linear, rank=5, rngs=nnx.Rngs(1))
```

**Generic Surgery**:
```python
# Replace all Linear layers with LoraLinear
rngs = nnx.Rngs(0)
model = Block(rngs=rngs)

for path, module in nnx.iter_graph(model):
    if isinstance(module, nnx.Module):
        for name, value in vars(module).items():
            if isinstance(value, nnx.Linear):
                setattr(module, name, LoraLinear(value, rank=5, rngs=rngs))
```

---

### 5. Transforms (JAX-Like APIs)

**Linen Problem**: Custom APIs that diverge from JAX, constrained usage patterns.

**Issues**:
1. Expose additional non-JAX APIs (confusing, divergent behavior)
2. Work on functions with specific signatures:
   - `flax.linen.Module` must be first argument
   - Accept `Module` args but not as return values
3. Can only be used inside `flax.linen.Module.apply`

**NNX Solution**: Equivalent to JAX transforms, but work with modules.

**Characteristics**:
1. Same API as JAX transforms
2. Accept/return `nnx.Module` on any argument
3. Can be used anywhere (including training loop)

**Example**:
```python
# NNX: vmap over module creation and application
class Weights(nnx.Module):
    def __init__(self, seed: int):
        key = jax.random.PRNGKey(seed)
        self.kernel = nnx.Param(jax.random.uniform(key, (2, 3)))
        self.bias = nnx.Param(jnp.zeros((3,)))

def vector_dot(weights: Weights, x: jax.Array):
    assert weights.kernel.value.ndim == 2
    assert x.ndim == 1
    return x @ weights.kernel.value + weights.bias.value

# Create stack of weights
seeds = jnp.arange(10)
weights = nnx.vmap(lambda seed: Weights(seed), in_axes=0, out_axes=0)(seeds)

# Apply to batch
x = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
y = nnx.vmap(vector_dot, in_axes=(0, 0), out_axes=1)(weights, x)
```

**Method Decorators**:
```python
# NNX: Use transforms as decorators
class WeightStack(nnx.Module):
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def __init__(self, seed: jax.Array):
        key = jax.random.PRNGKey(seed)
        self.kernel = nnx.Param(jax.random.uniform(key, (2, 3)))
        self.bias = nnx.Param(jnp.zeros((3,)))
    
    @nnx.vmap(in_axes=(0, 0), out_axes=1)
    def __call__(self, x: jax.Array):
        return x @ self.kernel.value + self.bias.value

weights = WeightStack(jnp.arange(10))
y = weights(x)
```

**Key Point**: `in_axes` and other APIs **do** affect how `nnx.Module` state is transformed (unlike Linen).

---

## Linen vs NNX Comparison

| Feature | Linen | NNX |
|---------|-------|-----|
| **Initialization** | Lazy (shape inference) | Eager (explicit shapes) |
| **Parameters** | Separate dict | Embedded in module |
| **State** | Immutable collections | Mutable attributes |
| **Method Calls** | Via `.apply()` | Direct |
| **Sub-modules** | Not accessible | Pythonic references |
| **Stateful Layers** | Manual collection tracking | Automatic |
| **Model Surgery** | Complex (dict surgery) | Simple (attribute assignment) |
| **Transforms** | Custom APIs, constrained | JAX-like, flexible |
| **Learning Curve** | Steep | Gentle |

---

## When to Use NNX

### Choose NNX if:
- ✅ You want Pythonic, intuitive APIs
- ✅ You need frequent model inspection/debugging
- ✅ You're building complex stateful models (RNNs, BatchNorm)
- ✅ You need flexible model surgery (LoRA, pruning, etc.)
- ✅ You want tight JAX integration
- ✅ You're starting a new project

### Consider Linen if:
- You have existing Linen codebase (migration takes time)
- You heavily rely on shape inference
- You need backward compatibility with old checkpoints

### Migration Path

For incremental migration, use the [NNX/Linen bridge](https://flax.readthedocs.io/en/latest/guides/bridge_guide.html) to mix both APIs in the same codebase.

---

## Example: Simple Training Loop

```python
from flax import nnx
import optax

class Model(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dmid, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.2, rngs=rngs)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)
    
    def __call__(self, x):
        x = nnx.relu(self.dropout(self.bn(self.linear(x))))
        return self.linear_out(x)

# Create model and optimizer
model = Model(2, 64, 3, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit  # Automatic state management
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)  # Call directly
        return ((y_pred - y) ** 2).mean()
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place updates
    
    return loss

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        loss = train_step(model, optimizer, x_batch, y_batch)
        # No need to reassign model/optimizer (stateful!)
```

**Key Simplifications**:
1. No `.apply()` calls
2. No manual state threading
3. No collection tracking
4. In-place updates (no returns needed)

---

## Resources

- **Why NNX**: https://flax.readthedocs.io/en/stable/why.html
- **NNX Basics**: https://flax.readthedocs.io/en/latest/nnx_basics.html
- **Migration Guide**: https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html
- **Transforms Guide**: https://flax.readthedocs.io/en/latest/guides/transforms.html

---

**Last Updated**: November 18, 2025
