# Flax NNX Glossary

**Essential terminology for understanding Flax NNX - quick lookup reference.**

---

## Core Terms

### Filter

A way to extract only certain `nnx.Variable` objects out of a Flax NNX Module (`nnx.Module`). This is usually done by calling [`nnx.split`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/graph.html#flax.nnx.split) upon the module.

**Example**:
```python
# Extract only trainable parameters
params = nnx.state(model, nnx.Param)

# Extract batch statistics
batch_stats = nnx.state(model, nnx.BatchStat)

# Multiple filters
trainable = nnx.state(model, nnx.Param, MyCustomVar)
```

**Use Cases**:
- Separate parameters for optimizer
- Extract batch stats for evaluation
- Custom variable grouping

**Further Reading**: [Filter guide](https://flax.readthedocs.io/en/latest/guides/filters_guide.html)

---

### Folding In

In Flax, [folding in](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.fold_in.html) means generating a new [JAX PRNG](https://jax.readthedocs.io/en/latest/random-numbers.html) key, given an input PRNG key and integer.

**Why?** You want to generate a new key but still be able to use the original PRNG key afterwards.

**Alternative**: `jax.random.split` creates two PRNG keys (slower).

**Example**:
```python
key = jax.random.PRNGKey(0)

# Fold in integer to get new key
key1 = jax.random.fold_in(key, 0)
key2 = jax.random.fold_in(key, 1)

# Original key still usable
key3 = jax.random.fold_in(key, 2)
```

**Further Reading**: [Randomness/PRNG guide](https://flax.readthedocs.io/en/latest/guides/randomness.html)

---

### GraphDef

[`nnx.GraphDef`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/graph.html#flax.nnx.GraphDef) is a class that represents all the **static, stateless, and Pythonic** parts of a Flax Module (`nnx.Module`).

**What it Contains**:
- Module structure (class definition, nested modules)
- Non-array attributes (hyperparameters, strings, etc.)
- Computation graph

**What it Doesn't Contain**:
- Parameter arrays
- State arrays (batch stats, caches, etc.)
- PRNG state

**Example**:
```python
# Split module into static + dynamic
graphdef, state = nnx.split(model)

# graphdef: Structure
# state: Arrays (params, batch_stats, etc.)

# Serialize separately
save_json(graphdef)  # Small, structure only
save_arrays(state)    # Large, weights only

# Reconstruct
model = nnx.merge(graphdef, state)
```

---

### Merge

Refer to **Split and Merge** below.

---

### Module

[`nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html) is a dataclass that enables defining and initializing parameters in a referentially-transparent form. It is responsible for storing and updating `Variable` objects and parameters within itself.

**Key Characteristics**:
- **Stateful**: Variables stored as attributes
- **Eager**: Parameters initialized immediately
- **Pythonic**: Standard Python object semantics

**Example**:
```python
class MyModule(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.weight = nnx.Param(jnp.ones((features, features)))
    
    def __call__(self, x):
        return self.linear(x) + x @ self.weight.value

# Instantiate (parameters created immediately)
model = MyModule(features=64, rngs=nnx.Rngs(0))

# Access parameters directly
print(model.weight.value.shape)  # (64, 64)
```

---

### Params / Parameters

[`nnx.Param`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/variables.html#flax.nnx.Param) is a particular subclass of `nnx.Variable` that generally contains the **trainable weights**.

**Characteristics**:
- Tracked by optimizers
- Included in gradients
- Saved in checkpoints

**Example**:
```python
class MyLayer(nnx.Module):
    def __init__(self, din, dout):
        self.weight = nnx.Param(jnp.ones((din, dout)))
        self.bias = nnx.Param(jnp.zeros((dout,)))
    
    def __call__(self, x):
        return x @ self.weight.value + self.bias.value

# Extract all Param variables
params = nnx.state(layer, nnx.Param)
```

---

### PRNG States

A Flax `nnx.Module` can keep a reference of a [pseudorandom number generator (PRNG)](https://jax.readthedocs.io/en/latest/random-numbers.html) state object [`nnx.Rngs`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/rnglib.html#flax.nnx.Rngs) that can generate new [JAX PRNG](https://jax.readthedocs.io/en/latest/random-numbers.html) keys.

**Why?** These keys are used to generate random JAX arrays through [JAX's functional PRNGs](https://jax.readthedocs.io/en/latest/random-numbers.html). You can use a PRNG state with different seeds to add more fine-grained control to your model (for example, to have independent random numbers for parameters and dropout masks).

**Example**:
```python
# Single stream
rngs = nnx.Rngs(0)

# Multiple named streams
rngs = nnx.Rngs(params=0, dropout=1, noise=2)

# Use in module
class MyModule(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        # Automatically managed by layers
        self.dropout = nnx.Dropout(0.5, rngs=rngs)
    
    def __call__(self, x):
        # Get key from default stream
        key = self.rngs()
        noise = jax.random.normal(key, x.shape)
        return self.dropout(x) + noise
```

**Further Reading**: [Randomness/PRNG guide](https://flax.readthedocs.io/en/latest/guides/randomness.html)

---

### Split and Merge

[`nnx.split`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/graph.html#flax.nnx.split) is a way to represent an `nnx.Module` by two parts:

1. **Static** Flax NNX `GraphDef` that captures its Pythonic static information
2. One or more **Variable state(s)** that capture its [JAX arrays](https://jax.readthedocs.io/en/latest/key-concepts.html#jax-arrays-jax-array) in the form of [JAX pytrees](https://jax.readthedocs.io/en/latest/working-with-pytrees.html)

They can be merged back to the original `nnx.Module` using [`nnx.merge`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/graph.html#flax.nnx.merge).

**Example**:
```python
model = MyModel(...)

# Split into static + dynamic
graphdef, state = nnx.split(model)

# state is a pytree (can be passed to JAX transforms)
def pure_fn(state):
    # Use JAX transforms on state
    return jax.tree.map(lambda x: x * 2, state)

new_state = pure_fn(state)

# Merge back
model = nnx.merge(graphdef, new_state)
```

**Use Cases**:
- Checkpointing: Save `state` as pytree
- JAX transforms: Pass `state` to pure functions
- Model surgery: Modify `graphdef` structure

---

### Transformation

A Flax NNX transformation (transform) is a **wrapped version** of a [JAX transformation](https://flax.readthedocs.io/en/latest/guides/transforms.html) that allows the function that is being transformed to take the Flax NNX Module (`nnx.Module`) as input or output.

**Example**: A "lifted" version of [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) is [`nnx.jit`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html#flax.nnx.jit).

**Comparison**:

| Transform | JAX | NNx |
|-----------|-----|-----|
| **JIT** | `jax.jit` | `nnx.jit` |
| **Grad** | `jax.grad` | `nnx.grad` |
| **Vmap** | `jax.vmap` | `nnx.vmap` |
| **Scan** | `jax.lax.scan` | `nnx.scan` |

**Example**:
```python
# JAX: Must be pure function with pytrees
@jax.jit
def train_step(state, x, y):
    # state must be pytree
    # must return new state
    return new_state

# NNx: Can use stateful modules
@nnx.jit
def train_step(model, x, y):
    # model can be nnx.Module
    # in-place updates (no return needed)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
```

**Further Reading**: [Flax NNX transforms guide](https://flax.readthedocs.io/en/latest/guides/transforms.html)

---

### Variable

The weights / parameters / data / array [`nnx.Variable`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/variables.html#flax.nnx.Variable) residing in a Flax Module. Variables are defined inside modules as `nnx.Variable` or its subclasses.

**Built-in Subclasses**:
- `nnx.Param`: Trainable parameters
- `nnx.BatchStat`: Running statistics (BatchNorm mean/var)
- `nnx.Intermediary`: Intermediate activations (for logging)

**Example**:
```python
class MyModule(nnx.Module):
    def __init__(self):
        # Trainable
        self.weight = nnx.Param(jnp.ones((10, 10)))
        
        # Non-trainable
        self.running_mean = nnx.BatchStat(jnp.zeros((10,)))
        
        # Custom type
        self.cache = nnx.Variable(jnp.zeros((10,)))
    
    def __call__(self, x):
        # Access .value
        output = x @ self.weight.value
        
        # Update (mutable!)
        self.running_mean.value = 0.9 * self.running_mean.value + 0.1 * jnp.mean(x)
        
        return output
```

**Custom Variable Types**:
```python
# Define custom variable type
class MyCustomVar(nnx.Variable):
    pass

# Use in filtering
self.my_data = MyCustomVar(jnp.ones((10,)))

# Extract only custom vars
custom_state = nnx.state(model, MyCustomVar)
```

---

## Additional JAX Terms

For additional JAX terminology, refer to the [JAX glossary](https://jax.readthedocs.io/en/latest/glossary.html).

**Key JAX Concepts**:
- **Pytree**: Tree-like structure of containers (dicts, lists, tuples) holding arrays
- **Transformation**: Higher-order function like `jit`, `grad`, `vmap`
- **PRNG**: Pseudorandom number generator (functional in JAX)
- **JIT**: Just-in-time compilation to XLA
- **XLA**: Accelerated Linear Algebra compiler

---

## Quick Lookup Table

| Term | Category | Key Property | Use Case |
|------|----------|--------------|----------|
| `nnx.Module` | Structure | Stateful, Pythonic | Define layers |
| `nnx.Param` | Variable | Trainable | Model weights |
| `nnx.BatchStat` | Variable | Non-trainable | Running stats |
| `nnx.Rngs` | PRNG | Auto-splitting | Random numbers |
| `nnx.split` | State Mgmt | Module → GraphDef + State | Checkpointing |
| `nnx.merge` | State Mgmt | GraphDef + State → Module | Load checkpoint |
| `nnx.jit` | Transform | Stateful JIT | Compile training |
| `nnx.grad` | Transform | Stateful grad | Get gradients |
| `nnx.vmap` | Transform | Vectorize modules | Batch ops |
| `nnx.scan` | Transform | Sequential loop | RNNs, layer stacks |

---

## Resources

- **Full Glossary**: https://flax.readthedocs.io/en/stable/nnx_glossary.html
- **JAX Glossary**: https://jax.readthedocs.io/en/latest/glossary.html
- **NNx Basics**: https://flax.readthedocs.io/en/latest/nnx_basics.html
- **Filter Guide**: https://flax.readthedocs.io/en/latest/guides/filters_guide.html

---

**Last Updated**: November 18, 2025
