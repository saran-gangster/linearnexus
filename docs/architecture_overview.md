# Architecture Overview

**Understanding how LinearNexus is organized**

This document explains the structure of LinearNexus, how the pieces fit together, and where to find things when you want to modify the code.

---

## Table of Contents

1. [High-Level Structure](#high-level-structure)
2. [Core Modules](#core-modules)
3. [Data Flow](#data-flow)
4. [The Block Protocol](#the-block-protocol)
5. [How Models Are Built](#how-models-are-built)
6. [Extending LinearNexus](#extending-linearnexus)

---

## High-Level Structure

```
linearnexus/
├── __init__.py      # Public API exports
├── models.py        # LMModel, ModelConfig, presets
├── train.py         # Training loops (SFT, GRPO, PPO)
├── optim.py         # Optimizers (AdamW, Muon, Sophia)
├── data.py          # Tokenizers, datasets, dataloaders
├── generate.py      # Text generation utilities
│
├── modules/         # Neural network building blocks
│   ├── common.py    # Shared: MLP, RMSNorm, Embedding, RoPE
│   ├── attention/   # Attention mechanisms
│   │   └── causal.py    # CausalSelfAttention, KVCache
│   ├── ssm/         # State-space models
│   │   └── mamba.py     # MambaBlock, selective scan
│   ├── sparse/      # [Planned] Sparse attention
│   ├── linear_attn/ # [Planned] Linear attention variants
│   └── hybrid/      # [Planned] Hybrid block utilities
│
├── core/            # Low-level utilities
│   ├── cache.py     # ConvState, RecurrentState
│   ├── config.py    # ConfigBase dataclass
│   ├── conv.py      # Depthwise causal convolution
│   ├── mode.py      # KernelMode enum
│   └── ...
│
└── kernels/         # Compute kernels
    └── mamba_reference.py  # Pure JAX selective scan
```

---

## Core Modules

### `models.py` — Model Definition

The heart of LinearNexus. Contains:

```python
# Configuration
@dataclass
class ModelConfig:
    vocab_size: int = 50257      # Vocabulary size
    hidden_size: int = 768       # Model width (d_model)
    n_layer: int = 12            # Number of blocks
    n_head: int = 12             # Attention heads
    block_size: int = 1024       # Max sequence length
    block_pattern: list = ["attention"]  # Architecture pattern
    # ... more options

# The main model class
class LMModel(nnx.Module):
    def __init__(self, config, rngs):
        self.embed = Embedding(...)
        self.blocks = [create_block(pattern, ...) for ...]
        self.ln_f = RMSNorm(...)
        self.lm_head = nnx.Linear(...)
    
    def __call__(self, tokens, *, state=None, mask=None):
        x = self.embed(tokens)
        for block in self.blocks:
            x, state = block(x, state=state, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, state

# Factory function
def create_model(name_or_config, *, rngs=None):
    ...
```

**Key insight**: Models are lists of interchangeable blocks. The `block_pattern` controls which type of block goes where.

### `train.py` — Training Infrastructure

Three trainers for different training paradigms:

```python
# Supervised Fine-Tuning (standard next-token prediction)
class SFTTrainer:
    def train(self, dataloader):
        for batch in dataloader:
            loss, grads = self.compute_loss_and_grads(batch)
            self.optimizer.update(grads)

# Group Relative Policy Optimization (preference learning)
class GRPOTrainer:
    ...

# Proximal Policy Optimization (reinforcement learning)
class PPOTrainer:
    ...
```

### `optim.py` — Custom Optimizers

Beyond standard AdamW:

```python
def adamw(...) -> optax.GradientTransformation:
    """Standard AdamW with LLM-friendly defaults."""

def muon(...) -> optax.GradientTransformation:
    """Momentum with orthogonalization for stability."""

def sophia(...) -> optax.GradientTransformation:
    """Second-order Hessian-based optimizer."""

def create_optimizer(name, learning_rate, ...):
    """Factory with learning rate schedule and gradient clipping."""
```

### `data.py` — Data Pipeline

```python
# Tokenizers
class CharTokenizer:     # Character-level (simple, educational)
class BPETokenizer:      # Byte-pair encoding (production-ready)

# Datasets  
class TextDataset:       # Memory-mapped text file
    def __getitem__(self, idx):
        return tokens[idx:idx+seq_len], tokens[idx+1:idx+seq_len+1]

# Loading
class DataLoader:        # Batching with shuffling
    def __iter__(self):
        for batch_indices in self.sampler:
            yield self.collate([self.dataset[i] for i in batch_indices])
```

### `generate.py` — Text Generation

```python
def generate(model, prompt, max_tokens, *, temperature, top_k, top_p):
    """Core generation loop with sampling."""
    
def complete(model, tokenizer, text, max_tokens):
    """High-level: text in, text out."""

def generate_streaming(model, prompt, max_tokens):
    """Yields tokens one at a time."""

def batch_generate(model, prompts, max_tokens):
    """Generate from multiple prompts efficiently."""
```

---

## Data Flow

### Training Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Text File  │───▶│  Tokenizer   │───▶│  Dataset    │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ DataLoader  │
                                       └─────────────┘
                                              │
         ┌────────────────────────────────────┘
         ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Batch     │───▶│   Model     │───▶│    Loss     │
│ (x, y)      │    │ LMModel     │    │  CE Loss    │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                   ┌──────────────────────────┘
                   ▼
            ┌─────────────┐    ┌─────────────┐
            │  Gradients  │───▶│  Optimizer  │
            │  jax.grad() │    │  AdamW/etc  │
            └─────────────┘    └─────────────┘
                                      │
                                      ▼
                               ┌─────────────┐
                               │ Update Model│
                               │   Weights   │
                               └─────────────┘
```

### Generation Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Prompt    │───▶│  Tokenizer  │───▶│   Model     │
│  "Hello"    │    │  encode()   │    │  forward()  │
└─────────────┘    └──────────────┘   └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │   Logits    │
                                       │ [vocab_size]│
                                       └─────────────┘
                                              │
                   ┌──────────────────────────┘
                   ▼
            ┌─────────────┐    ┌─────────────┐
            │  Sampling   │───▶│ Next Token  │
            │ temp/top_k  │    │    ID       │
            └─────────────┘    └─────────────┘
                                      │
                                      ▼
                               ┌─────────────┐
                               │  Tokenizer  │───▶ "Hello world"
                               │  decode()   │
                               └─────────────┘
```

---

## The Block Protocol

All blocks (attention, Mamba, etc.) follow the same interface:

```python
class SomeBlock(nnx.Module):
    def __call__(
        self, 
        x: jax.Array,           # [batch, seq, hidden]
        *,
        state: Optional[BlockState] = None,  # For caching
        mask: Optional[jax.Array] = None,    # Attention mask
        mode: Optional[str] = None,          # "chunk" or "recurrent"
    ) -> tuple[jax.Array, Optional[BlockState]]:
        # Process input
        output = ...
        new_state = ...
        return output, new_state
```

**Why this matters**: You can swap block types without changing the model code!

```python
# These all work with the same LMModel:
blocks = [AttentionBlock(...)]   # GPT
blocks = [MambaBlock(...)]       # Mamba
blocks = [MambaBlock(...), MambaBlock(...), AttentionBlock(...)]  # Hybrid
```

---

## How Models Are Built

### Step 1: Configuration

```python
config = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layer=12,
    block_pattern=["mamba", "mamba", "attention"],
)
```

### Step 2: Block Pattern Expansion

The pattern repeats to fill `n_layer`:

```python
# pattern = ["mamba", "mamba", "attention"]
# n_layer = 12
# Result: ["mamba", "mamba", "attention", "mamba", "mamba", "attention", 
#          "mamba", "mamba", "attention", "mamba", "mamba", "attention"]
```

### Step 3: Block Creation

```python
for i, block_type in enumerate(expanded_pattern):
    if block_type == "attention":
        block = AttentionBlock(hidden_size, n_head, ...)
    elif block_type == "mamba":
        block = MambaBlock(hidden_size, ssm_state, ...)
    blocks.append(block)
```

### Step 4: Forward Pass

```python
def __call__(self, tokens, *, state=None):
    x = self.embed(tokens)
    
    block_states = state.block_states if state else {}
    new_states = {}
    
    for i, block in enumerate(self.blocks):
        block_state = block_states.get(i)
        x, new_block_state = block(x, state=block_state)
        new_states[i] = new_block_state
    
    x = self.ln_f(x)
    logits = self.lm_head(x)
    
    return logits, ModelState(block_states=new_states)
```

---

## Extending LinearNexus

### Adding a New Block Type

1. **Create the block** in `modules/yourtype/`:

```python
# modules/yourtype/block.py
class YourBlock(nnx.Module):
    def __init__(self, config, rngs):
        ...
    
    def __call__(self, x, *, state=None, mask=None, mode=None):
        output = ...
        new_state = ...
        return output, new_state
```

2. **Export it** in `modules/yourtype/__init__.py`:

```python
from .block import YourBlock
```

3. **Register in models.py**:

```python
def _create_block(block_type, config, rngs):
    if block_type == "attention":
        return AttentionBlock(...)
    elif block_type == "mamba":
        return MambaBlock(...)
    elif block_type == "yourtype":        # Add this
        return YourBlock(...)
```

4. **Use it**:

```python
config = ModelConfig(block_pattern=["yourtype", "attention"])
```

### Adding a New Optimizer

1. **Implement in `optim.py`**:

```python
def your_optimizer(learning_rate, ...) -> optax.GradientTransformation:
    def init_fn(params):
        return YourState(...)
    
    def update_fn(updates, state, params):
        new_updates = ...
        new_state = ...
        return new_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)
```

2. **Register in `get_optimizer`**:

```python
def get_optimizer(name, **kwargs):
    optimizers = {
        "adamw": adamw,
        "muon": muon,
        "sophia": sophia,
        "yours": your_optimizer,  # Add this
    }
    return optimizers[name](**kwargs)
```

### Adding a New Trainer

1. **Implement in `train.py`**:

```python
@dataclass
class YourConfig:
    learning_rate: float = 1e-4
    max_steps: int = 10000
    ...

class YourTrainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
    
    def train(self, dataloader):
        for step, batch in enumerate(dataloader):
            loss, grads = self.compute_loss(batch)
            self.optimizer.update(grads)
            
            if step >= self.config.max_steps:
                break
```

---

## File Size Guide

LinearNexus is intentionally small. Each file should be readable in one sitting:

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | ~450 | Model definition, configs, factory |
| `train.py` | ~500 | All training loops |
| `optim.py` | ~450 | All optimizers |
| `data.py` | ~350 | Data pipeline |
| `generate.py` | ~450 | Generation utilities |
| `modules/common.py` | ~300 | Shared building blocks |
| `modules/attention/causal.py` | ~350 | Attention mechanism |
| `modules/ssm/mamba.py` | ~400 | Mamba mechanism |

**Total**: ~3,000 lines of core code

---

## Design Principles

1. **Flat over nested**: Prefer fewer, larger files over deep hierarchies
2. **Explicit over implicit**: No magic registration, clear imports
3. **Composable**: Blocks are interchangeable, optimizers chain with optax
4. **Educational**: Code should teach, not obscure
5. **Practical**: Must run on consumer hardware (12GB GPU)

---

## See Also

- [GETTING_STARTED.md](../GETTING_STARTED.md) — Quick start tutorial
- [training_guide.md](./training_guide.md) — Deep dive into training
- [adding_new_layers.md](./adding_new_layers.md) — Extending with new architectures
