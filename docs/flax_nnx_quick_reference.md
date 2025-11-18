# Flax NNX Quick Reference

**Quick reference guide for Flax NNX API - essential concepts and patterns for LinearNexus contributors.**

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Module Definition](#module-definition)
3. [Variable Types](#variable-types)
4. [Random Number Generation](#random-number-generation)
5. [Transforms](#transforms)
6. [State Management](#state-management)
7. [Common Patterns](#common-patterns)
8. [Migration Notes (0.10 → 0.11)](#migration-notes-010--011)

---

## Core Concepts

### Module

[`nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html) is a stateful, Pythonic dataclass for defining neural network layers.

**Key Characteristics**:
- **Eager initialization**: Parameters created immediately (no lazy shape inference)
- **Mutable state**: Variables stored as module attributes
- **Referential semantics**: Modules can reference and modify each other

```python
import flax.nnx as nnx

class MyLayer(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        # Parameters created here (eager)
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.bias = nnx.Param(jnp.zeros((out_features,)))
    
    def __call__(self, x):
        return self.linear(x) + self.bias.value

# Instantiate with explicit shapes
rngs = nnx.Rngs(0)
layer = MyLayer(in_features=10, out_features=5, rngs=rngs)
```

### GraphDef and State

**Split and Merge Pattern**: Separate static structure from dynamic state.

```python
# Split module into static + dynamic parts
graphdef, state = nnx.split(model)

# state: nnx.State (pytree of arrays)
# graphdef: nnx.GraphDef (static structure)

# Merge back to get module
model = nnx.merge(graphdef, state)
```

**Use Cases**:
- Checkpointing (save/load `state`)
- JAX transforms (pass `state` as pytree)
- Model surgery (modify structure)

---

## Module Definition

### Basic Pattern

```python
import flax.nnx as nnx
import jax.numpy as jnp

class Block(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        # Always pass rngs as keyword-only argument
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.bn = nnx.BatchNorm(features, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
    
    def __call__(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        return nnx.relu(x)
```

### Nested Modules

```python
class Encoder(nnx.Module):
    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
    
    def __call__(self, x):
        return self.linear(x)

class AutoEncoder(nnx.Module):
    def __init__(self, input_dim: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.encoder = Encoder(embed_dim, rngs=rngs)
        self.decoder = nnx.Linear(embed_dim, input_dim, rngs=rngs)
    
    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    # Call methods directly (no .apply() needed)
    def encode(self, x):
        return self.encoder(x)
```

---

## Variable Types

### Param (Trainable Parameters)

```python
self.weight = nnx.Param(jnp.ones((10, 5)))
self.bias = nnx.Param(jnp.zeros((5,)))

# Access value
w = self.weight.value
```

### BatchStat (Running Statistics)

```python
class MyBatchNorm(nnx.Module):
    def __init__(self, features: int):
        self.scale = nnx.Param(jnp.ones((features,)))
        self.bias = nnx.Param(jnp.zeros((features,)))
        # Non-trainable running stats
        self.mean = nnx.BatchStat(jnp.zeros((features,)))
        self.var = nnx.BatchStat(jnp.ones((features,)))
    
    def __call__(self, x):
        # Update running stats (mutable!)
        batch_mean = jnp.mean(x, axis=0)
        self.mean.value = 0.9 * self.mean.value + 0.1 * batch_mean
        # ... normalize using stats
```

### Variable (Generic Container)

```python
# Any stateful data
self.counter = nnx.Variable(jnp.array(0))
self.cache = nnx.Variable(jnp.zeros((batch_size, seq_len)))
```

### Custom Variable Types

```python
class MyCustomVar(nnx.Variable):
    pass

# Use in filters
self.custom = MyCustomVar(jnp.ones((10,)))
```

---

## Random Number Generation

### Rngs Object

[`nnx.Rngs`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/rnglib.html) manages PRNG keys with automatic splitting.

```python
# Create with single seed
rngs = nnx.Rngs(0)

# Create with multiple streams
rngs = nnx.Rngs(params=0, dropout=1)

# Fork creates independent copy
rngs_copy = rngs.fork()

# Split for vmap/scan
rngs_split = rngs.fork(splits=5)  # Creates 5 independent copies
```

### Module Construction

```python
class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        # Each layer gets forked copy (v0.11+)
        self.layer1 = nnx.Linear(10, 20, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
```

### Manual Key Usage

```python
class MyLayer(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.rngs = rngs
    
    def __call__(self, x):
        # Get next key from stream
        key = self.rngs()
        noise = jax.random.normal(key, x.shape)
        return x + noise
```

---

## Transforms

### NNx Transforms vs JAX Transforms

| Feature | JAX (`jax.jit`, `jax.grad`) | NNx (`nnx.jit`, `nnx.grad`) |
|---------|------------------------------|------------------------------|
| **Purity** | Pure functions only | Handles stateful modules |
| **Arguments** | Pytrees | Modules + pytrees |
| **State** | Manual threading | Automatic tracking |
| **Returns** | Must return updated state | In-place updates |

### nnx.jit

```python
@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        pred = model(x)
        return jnp.mean((pred - y) ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # In-place update
    return loss

# No need to return/reassign model
loss = train_step(model, optimizer, x_batch, y_batch)
```

### nnx.grad

```python
def loss_fn(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y) ** 2)

# Get gradients as nnx.State
grads = nnx.grad(loss_fn)(model, x, y)

# Update parameters
params = nnx.state(model, nnx.Param)
params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
nnx.update(model, params)
```

### nnx.vmap

```python
# Vectorize module creation
@nnx.vmap(in_axes=(0,), out_axes=0)
def create_layers(seed):
    return nnx.Linear(10, 10, rngs=nnx.Rngs(seed))

# Creates 5 independent layers with stacked weights
seeds = jnp.arange(5)
layers = create_layers(seeds)
```

### nnx.scan

```python
class RNN(nnx.Module):
    def __init__(self, hidden_size: int, *, rngs: nnx.Rngs):
        self.cell = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
    
    @nnx.scan(in_axes=(nnx.Carry, 1), out_axes=nnx.Carry)
    def __call__(self, carry, x_t):
        # carry: hidden state, x_t: input at time t
        h = self.cell(jnp.concatenate([carry, x_t], axis=-1))
        return h  # New carry (no need for dummy output)

rnn = RNN(hidden_size=64, rngs=nnx.Rngs(0))
h0 = jnp.zeros((batch, 64))
x_seq = jnp.ones((batch, seq_len, input_dim))
h_final = rnn(h0, x_seq)
```

### Scan Over Layers

```python
class MLP(nnx.Module):
    def __init__(self, num_layers: int, *, rngs: nnx.Rngs):
        # Use vmap to create stacked weights
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs):
            return Block(features=64, rngs=rngs)
        
        self.blocks = create_block(rngs.fork(splits=num_layers))
    
    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
    def __call__(self, x):
        # Scan over stacked blocks
        return self.blocks(x)
```

### Mixing NNx and JAX Transforms

```python
# Use nnx.jit with jax.grad
@nnx.jit
def train_step(model, x, y):
    def loss_fn(graphdef, state):
        model = nnx.merge(graphdef, state)
        return jnp.mean((model(x) - y) ** 2)
    
    # JAX grad requires pytrees
    grads = jax.grad(loss_fn, argnums=1)(*nnx.split(model))
    
    # Update using NNx API
    params = nnx.state(model, nnx.Param)
    params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    nnx.update(model, params)
```

---

## State Management

### Filter by Variable Type

```python
# Extract specific variable types
params = nnx.state(model, nnx.Param)  # Trainable only
batch_stats = nnx.state(model, nnx.BatchStat)  # Running stats
all_state = nnx.state(model)  # Everything

# Multiple filters
trainable = nnx.state(model, nnx.Param, MyCustomVar)
```

### Update Module State

```python
# Modify state dict
params = nnx.state(model, nnx.Param)
params = jax.tree.map(lambda p: p * 0.9, params)  # Decay

# Write back to model
nnx.update(model, params)
```

### Model Surgery

```python
# Replace sub-modules (Pythonic!)
model = TransformerModel(...)
for name, module in vars(model).items():
    if isinstance(module, nnx.Linear):
        # Replace with LoRA layer
        setattr(model, name, LoRALinear(module, rank=8, rngs=rngs))

# Iterate over graph
for path, module in nnx.iter_graph(model):
    if isinstance(module, nnx.BatchNorm):
        print(f"Found BatchNorm at {path}")
```

### Train/Eval Modes

```python
# Switch dropout/batchnorm behavior
model.train()  # Training mode
output = model(x)

model.eval()  # Evaluation mode
output = model(x)
```

---

## Common Patterns

### Training Loop with Optimizer

```python
import optax

# Create model and optimizer
model = Model(input_dim=10, output_dim=5, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        return jnp.mean((model(x) - y) ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # Updates model in-place
    return loss

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        loss = train_step(model, optimizer, x_batch, y_batch)
```

### Checkpointing

```python
import orbax.checkpoint as ocp

# Save
checkpointer = ocp.StandardCheckpointer()
state = nnx.state(model)
checkpointer.save(path, state)

# Load
state = checkpointer.restore(path)
nnx.update(model, state)
```

### Gradient Accumulation

```python
@nnx.jit
def accumulate_gradients(model, x_batches, y_batches):
    def loss_fn(model, x, y):
        return jnp.mean((model(x) - y) ** 2)
    
    # Accumulate over micro-batches
    grads_accum = None
    for x, y in zip(x_batches, y_batches):
        grads = nnx.grad(loss_fn)(model, x, y)
        if grads_accum is None:
            grads_accum = grads
        else:
            grads_accum = jax.tree.map(lambda a, g: a + g, grads_accum, grads)
    
    # Average
    n = len(x_batches)
    grads_accum = jax.tree.map(lambda g: g / n, grads_accum)
    return grads_accum
```

### Multi-Device Training (pmap)

```python
# Replicate model across devices
devices = jax.devices()
replicated_model = jax.device_put_replicated(model, devices)

@jax.pmap
def train_step_pmap(model, x, y):
    def loss_fn(model):
        return jnp.mean((model(x) - y) ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # All-reduce gradients
    grads = jax.lax.pmean(grads, axis_name='devices')
    
    params = nnx.state(model, nnx.Param)
    params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    nnx.update(model, params)
    return loss
```

---

## Migration Notes (0.10 → 0.11)

### RNG Changes

**v0.10**: Modules held shared reference to `Rngs` object.

**v0.11**: Modules hold **forked copy** of `Rngs`.

```python
# v0.11 (current)
class Model(nnx.Module):
    @nnx.split_rngs(splits=5)
    @nnx.vmap(in_axes=(0,))
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
    
    @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
    def __call__(self, x):
        return self.dropout(self.linear(x))

# Alternative: explicit fork
rngs = nnx.Rngs(0)
model = Model(rngs=rngs.fork(splits=5))
```

### Optimizer Changes

**v0.10**: Optimizer held reference to model.

**v0.11**: Optimizer takes model + grads as arguments.

```python
# v0.11 (current)
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)  # wrt required

@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # Pass model explicitly
    return loss
```

### Loading Old Checkpoints

```python
# Load v0.10 checkpoint and convert
checkpoint = checkpointer.restore(old_path)

@jax.jit
def fix_checkpoint(checkpoint, rngs: nnx.Rngs):
    # 1. Remove old RNG keys from checkpoint
    flat_paths = nnx.traversals.flatten_mapping(checkpoint)
    flat_paths = {
        path: value
        for path, value in flat_paths.items()
        if "rngs" not in path  # Drop old RNG state
    }
    checkpoint = nnx.traversals.unflatten_mapping(flat_paths)
    
    # 2. Initialize model with new RNGs
    model = Model(rngs=rngs)
    
    # 3. Overwrite params with checkpoint
    nnx.update(model, checkpoint)
    
    return nnx.state(model)

new_checkpoint = fix_checkpoint(checkpoint, rngs=nnx.Rngs(0))
checkpointer.save(new_path, new_checkpoint)
```

### Pytree Handling

**v0.11**: NNx modules are now pytrees.

```python
# When using jax.tree.* on structures containing modules
modules = [nnx.Linear(3, 3, rngs=rngs), nnx.BatchNorm(3, rngs=rngs)]

# Specify that NNx objects are leaves
type_names = jax.tree.map(
    lambda x: type(x).__name__,
    modules,
    is_leaf=lambda x: isinstance(x, nnx.Pytree)  # Treat as leaves
)
```

---

## Resources

- **Official Docs**: https://flax.readthedocs.io/en/stable/
- **Migration Guide**: https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html
- **NNx Basics**: https://flax.readthedocs.io/en/latest/nnx_basics.html
- **Glossary**: https://flax.readthedocs.io/en/stable/nnx_glossary.html
- **GitHub**: https://github.com/google/flax

---

**Last Updated**: November 18, 2025
