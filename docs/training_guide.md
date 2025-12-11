# Training Guide

**A comprehensive guide to training language models with LinearNexus**

This guide covers everything you need to know about training: from basic supervised fine-tuning to advanced RL-based methods.

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
3. [Understanding the Data Pipeline](#understanding-the-data-pipeline)
4. [Optimizers Deep Dive](#optimizers-deep-dive)
5. [Checkpointing and Resuming](#checkpointing-and-resuming)
6. [Monitoring Training](#monitoring-training)
7. [Advanced: GRPO Training](#advanced-grpo-training)
8. [Advanced: PPO Training](#advanced-ppo-training)
9. [Tips and Best Practices](#tips-and-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Training Overview

LinearNexus supports three training paradigms:

| Method | Use Case | Difficulty |
|--------|----------|------------|
| **SFT** | Next-token prediction, general pretraining | Beginner |
| **GRPO** | Preference learning, alignment | Intermediate |
| **PPO** | Reinforcement learning with rewards | Advanced |

For most users, **SFT is the right choice**.

---

## Supervised Fine-Tuning (SFT)

SFT trains a model to predict the next token given previous tokens. This is how GPT, Mamba, and most language models are trained.

### Basic Example

```python
from flax import nnx
from linearnexus import (
    LMModel, ModelConfig,
    CharTokenizer, TextDataset, DataLoader,
    SFTTrainer, SFTConfig,
    create_optimizer,
)

# 1. Setup data
tokenizer = CharTokenizer.from_file("data.txt")
dataset = TextDataset("data.txt", tokenizer, seq_len=256)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. Create model
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    n_layer=6,
    n_head=4,
)
model = LMModel(config, rngs=nnx.Rngs(42))

# 3. Setup training
optimizer = create_optimizer(
    "adamw",
    learning_rate=3e-4,
    total_steps=5000,
    warmup_steps=100,
    weight_decay=0.1,
)

train_config = SFTConfig(
    max_steps=5000,
    log_every=100,
    eval_every=500,
    save_every=1000,
    checkpoint_dir="checkpoints",
)

# 4. Train
trainer = SFTTrainer(model, optimizer, train_config)
trainer.train(iter(dataloader))
```

### SFTConfig Options

```python
@dataclass
class SFTConfig:
    # Training duration
    max_steps: int = 10000          # Total training steps
    
    # Logging
    log_every: int = 100            # Log loss every N steps
    eval_every: int = 500           # Run evaluation every N steps
    
    # Checkpointing
    save_every: int = 1000          # Save checkpoint every N steps
    checkpoint_dir: str = "checkpoints"
    keep_last_n: int = 3            # Keep only last N checkpoints
    
    # Optimization (override via optimizer)
    learning_rate: float = 3e-4     # Peak learning rate
    warmup_steps: int = 100         # LR warmup steps
    grad_clip: float = 1.0          # Gradient clipping norm
    
    # Optional
    wandb_project: str = None       # W&B project name
    wandb_run_name: str = None      # W&B run name
```

---

## Understanding the Data Pipeline

### Tokenizers

LinearNexus provides two tokenizers:

#### Character Tokenizer (Simple)

```python
from linearnexus import CharTokenizer

# Create from text file (learns vocabulary)
tokenizer = CharTokenizer.from_file("shakespeare.txt")

# Or from vocabulary string
tokenizer = CharTokenizer.from_vocab("abcdefghijklmnopqrstuvwxyz ")

# Use
tokens = tokenizer.encode("hello world")  # [7, 4, 11, 11, 14, ...]
text = tokenizer.decode(tokens)            # "hello world"

print(tokenizer.vocab_size)  # ~100 for ASCII text
```

#### BPE Tokenizer (Production)

```python
from linearnexus import BPETokenizer

# Uses tiktoken (GPT-2/GPT-4 tokenizer)
tokenizer = BPETokenizer(encoding="gpt2")

tokens = tokenizer.encode("Hello, world!")  # [15496, 11, 995, 0]
text = tokenizer.decode(tokens)

print(tokenizer.vocab_size)  # 50257 for GPT-2
```

### Datasets

```python
from linearnexus import TextDataset

# Memory-mapped for large files
dataset = TextDataset(
    path="data.txt",
    tokenizer=tokenizer,
    seq_len=256,          # Sequence length
)

# Each item is (input_tokens, target_tokens)
x, y = dataset[0]
# x = tokens[0:256]
# y = tokens[1:257]  (shifted by 1 for next-token prediction)
```

### DataLoader

```python
from linearnexus import DataLoader

loader = DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,         # Shuffle for training
    drop_last=True,       # Drop incomplete final batch
)

for batch_x, batch_y in loader:
    # batch_x: [batch_size, seq_len]
    # batch_y: [batch_size, seq_len]
    pass
```

### Custom Data Sources

For more control, create your own generator:

```python
def my_data_generator():
    while True:
        # Your custom logic
        x = jnp.array(...)  # [batch, seq_len]
        y = jnp.array(...)  # [batch, seq_len]
        yield x, y

trainer.train(my_data_generator())
```

---

## Optimizers Deep Dive

### AdamW (Default)

Standard choice for most training:

```python
from linearnexus import adamw, create_optimizer

# Direct use
optimizer = adamw(
    learning_rate=3e-4,
    b1=0.9,               # Momentum
    b2=0.95,              # Second moment (0.95 for stability)
    eps=1e-8,
    weight_decay=0.1,
)

# With schedule and clipping (recommended)
optimizer = create_optimizer(
    "adamw",
    learning_rate=3e-4,
    total_steps=10000,
    warmup_steps=100,
    weight_decay=0.1,
    grad_clip=1.0,
)
```

### Muon (Momentum Orthogonalization)

Better stability, especially for larger models:

```python
from linearnexus import muon, create_optimizer

optimizer = create_optimizer(
    "muon",
    learning_rate=1e-3,   # Can use higher LR
    momentum=0.9,
    total_steps=10000,
)
```

**When to use**: When training is unstable with AdamW, or for larger models.

### Sophia (Second-Order)

Uses Hessian information for efficient updates:

```python
from linearnexus import sophia, create_optimizer

optimizer = create_optimizer(
    "sophia",
    learning_rate=1e-4,   # Usually lower LR
    b1=0.9,
    b2=0.95,
    rho=0.04,             # Hessian EMA coefficient
    total_steps=10000,
)
```

**When to use**: When you want faster convergence per step (but each step is slower).

### Composing with Optax

All optimizers are Optax-compatible:

```python
import optax
from linearnexus import get_optimizer

# Get base optimizer
base = get_optimizer("muon", learning_rate=1e-3)

# Compose with additional transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),      # Gradient clipping
    optax.add_decayed_weights(0.1),       # Weight decay
    base,
)
```

### Learning Rate Schedules

```python
from linearnexus import cosine_schedule

# Cosine decay with warmup
schedule = cosine_schedule(
    init_lr=3e-4,         # Peak learning rate
    total_steps=10000,
    warmup_steps=100,     # Linear warmup
    min_lr=3e-5,          # Final learning rate (10% of peak)
)

# Use with optax
optimizer = optax.chain(
    optax.scale_by_adam(b1=0.9, b2=0.95),
    optax.scale_by_schedule(schedule),
    optax.scale(-1.0),  # Gradient descent
)
```

---

## Checkpointing and Resuming

### Saving Checkpoints

```python
from linearnexus import save_checkpoint

# During training (automatic with SFTTrainer)
save_checkpoint(
    model=model,
    optimizer_state=opt_state,
    step=1000,
    path="checkpoints/step_1000",
)
```

### Loading Checkpoints

```python
from linearnexus import load_checkpoint

# Load for continued training
model, opt_state, step = load_checkpoint(
    "checkpoints/step_1000",
    model=model,              # Template model
    optimizer=optimizer,       # Template optimizer
)

# Load just the model (for inference)
model, _, _ = load_checkpoint("checkpoints/step_1000", model=model)
```

### Resuming Training

```python
# Start or resume training
import os

checkpoint_path = "checkpoints/latest"
start_step = 0

if os.path.exists(checkpoint_path):
    model, opt_state, start_step = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
    )
    print(f"Resuming from step {start_step}")

# Continue training
trainer = SFTTrainer(model, optimizer, config)
trainer.train(dataloader, start_step=start_step)
```

---

## Monitoring Training

### Console Logging

```python
config = SFTConfig(
    log_every=100,  # Print loss every 100 steps
)
```

Output:
```
Step 100 | Loss: 4.2341 | LR: 0.000150 | Time: 1.23s
Step 200 | Loss: 3.8567 | LR: 0.000250 | Time: 1.21s
...
```

### Weights & Biases

```python
config = SFTConfig(
    wandb_project="my-llm",
    wandb_run_name="gpt-small-shakespeare",
)
```

### Custom Logging

```python
class MyTrainer(SFTTrainer):
    def log(self, step, loss, lr):
        # Custom logging logic
        print(f"Step {step}: {loss:.4f}")
        
        # Log to file, database, etc.
        with open("training.log", "a") as f:
            f.write(f"{step},{loss},{lr}\n")
```

### Evaluation During Training

```python
def evaluate(model, eval_loader, max_batches=10):
    """Compute validation loss."""
    total_loss = 0
    for i, (x, y) in enumerate(eval_loader):
        if i >= max_batches:
            break
        logits, _ = model(x)
        loss = cross_entropy_loss(logits, y)
        total_loss += loss
    return total_loss / min(i + 1, max_batches)

# In training loop
config = SFTConfig(eval_every=500)

# The trainer will call your eval function
trainer.eval_fn = lambda: evaluate(model, eval_loader)
```

---

## Advanced: GRPO Training

Group Relative Policy Optimization for preference learning.

```python
from linearnexus import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    max_steps=1000,
    group_size=4,           # Compare N responses per prompt
    beta=0.1,               # KL penalty coefficient
    learning_rate=1e-5,     # Lower LR for fine-tuning
)

trainer = GRPOTrainer(model, reference_model, optimizer, config)

# Data: (prompt, [response1, response2, ...], [reward1, reward2, ...])
trainer.train(preference_dataloader)
```

---

## Advanced: PPO Training

Proximal Policy Optimization with reward model.

```python
from linearnexus import PPOTrainer, PPOConfig

config = PPOConfig(
    max_steps=1000,
    ppo_epochs=4,           # PPO update epochs per batch
    clip_ratio=0.2,         # PPO clipping
    value_coef=0.5,         # Value loss coefficient
    entropy_coef=0.01,      # Entropy bonus
)

trainer = PPOTrainer(
    policy_model=model,
    value_model=value_head,
    reward_model=reward_model,
    optimizer=optimizer,
    config=config,
)

trainer.train(prompt_dataloader)
```

---

## Tips and Best Practices

### Hyperparameters

| Param | GPT Small | GPT Medium | Mamba |
|-------|-----------|------------|-------|
| Learning Rate | 3e-4 | 1e-4 | 1e-3 |
| Batch Size | 8-32 | 16-64 | 8-32 |
| Warmup Steps | 100-500 | 500-2000 | 100-500 |
| Weight Decay | 0.1 | 0.1 | 0.01 |
| Gradient Clip | 1.0 | 1.0 | 1.0 |

### Memory Optimization

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=4)

# Reduce model size
config = ModelConfig(hidden_size=256, n_layer=4)

# Use gradient accumulation (coming soon)
# config = SFTConfig(gradient_accumulation_steps=4)
```

### Speed Optimization

```python
# JIT compilation (automatic, but verify)
import jax
print(jax.devices())  # Check for GPU

# Increase batch size (if memory allows)
dataloader = DataLoader(dataset, batch_size=32)

# Reduce logging frequency
config = SFTConfig(log_every=500)
```

### Stability Tips

1. **Start with lower learning rate** and increase
2. **Use warmup**: `warmup_steps=max_steps // 10`
3. **Gradient clipping**: `grad_clip=1.0`
4. **Monitor for NaN**: Stop if loss becomes NaN
5. **Weight decay**: `0.1` for most cases

---

## Troubleshooting

### Loss is NaN

```python
# Lower learning rate
optimizer = create_optimizer("adamw", learning_rate=1e-5, ...)

# Add gradient clipping
optimizer = create_optimizer(..., grad_clip=0.5)

# Check data for issues
for x, y in dataloader:
    assert not jnp.any(jnp.isnan(x)), "NaN in input data!"
    break
```

### Out of Memory

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=2)

# Reduce model size
config = ModelConfig(hidden_size=128, n_layer=4)

# Reduce sequence length
dataset = TextDataset(path, tokenizer, seq_len=128)
```

### Training is Slow

```python
# Verify GPU is being used
import jax
print(jax.devices())  # Should show GPU

# Increase batch size
dataloader = DataLoader(dataset, batch_size=32)

# Reduce logging
config = SFTConfig(log_every=500, eval_every=2000)
```

### Loss Plateau

```python
# Try different optimizer
optimizer = create_optimizer("muon", learning_rate=5e-4, ...)

# Adjust learning rate schedule
optimizer = create_optimizer(
    "adamw",
    learning_rate=1e-3,     # Higher peak
    min_lr=1e-5,            # Lower final
    total_steps=20000,      # Longer training
)

# Add more data
dataset = TextDataset("more_data.txt", tokenizer, seq_len=512)
```

---

## Quick Reference

### Minimal Training Script

```python
from flax import nnx
from linearnexus import *

# Data
tok = CharTokenizer.from_file("data.txt")
data = TextDataset("data.txt", tok, seq_len=128)
loader = DataLoader(data, batch_size=8)

# Model  
cfg = ModelConfig(vocab_size=tok.vocab_size, hidden_size=128, n_layer=4)
model = LMModel(cfg, rngs=nnx.Rngs(0))

# Train
opt = create_optimizer("adamw", learning_rate=1e-3, total_steps=1000)
trainer = SFTTrainer(model, opt, SFTConfig(max_steps=1000))
trainer.train(iter(loader))
```

---

## See Also

- [GETTING_STARTED.md](../GETTING_STARTED.md) — Quick start
- [architecture_overview.md](./architecture_overview.md) — Code structure
- [adding_new_layers.md](./adding_new_layers.md) — Extending architectures
