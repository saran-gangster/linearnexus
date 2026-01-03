# Getting Started with LinearNexus

**A beginner-friendly guide to training your first language model**

LinearNexus is a minimal LLM training framework built on JAX. This guide will take you from zero to training your first model in under 10 minutes.

---

## Table of Contents

1. [What is LinearNexus?](#what-is-linearnexus)
2. [Installation](#installation)
3. [Your First Model in 5 Minutes](#your-first-model-in-5-minutes)
4. [Understanding the Code](#understanding-the-code)
5. [Key Concepts](#key-concepts)
6. [Next Steps](#next-steps)
7. [Troubleshooting](#troubleshooting)

---

## What is LinearNexus?

LinearNexus is a **nanoGPT-style** training framework that supports multiple model architectures:

| Architecture | What it is | Best for |
|--------------|------------|----------|
| **GPT** | Transformer with dense attention | General text, well-understood |
| **Mamba** | State-space model (no attention) | Long sequences, fast inference |
| **Hybrid** | Mix of both (Jamba-style) | Best of both worlds |

**Why use it?**
- 📖 **Educational**: Code is simple enough to read and understand
- 🔧 **Hackable**: Easy to modify and experiment
- 🚀 **Practical**: Runs on consumer GPUs (12GB)
- 🐍 **Pythonic**: Uses modern JAX/Flax patterns

---

## Installation

### Prerequisites

- Python 3.10 or newer
- A GPU (recommended) or CPU

### Step 1: Clone the repository

```bash
git clone https://github.com/saran-gangster/linearnexus.git
cd linearnexus
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install the package

```bash
# Basic installation
pip install -e .

# With all optional dependencies (BPE tokenizer, logging)
pip install -e ".[full]"
```

### Step 4: Verify installation

```bash
python -c "import linearnexus; print(f'LinearNexus v{linearnexus.__version__} installed!')"
```

You should see: `LinearNexus v0.2.0 installed!`

---

## Your First Model in 5 Minutes

Let's train a tiny character-level model on Shakespeare!

### Option A: Using the command line

```bash
# Download Shakespeare and train a small GPT
python train_lm.py --model gpt-small --download-shakespeare --max-steps 1000

# Generate some text
python sample.py --checkpoint checkpoints/latest --prompt "To be or not to be"
```

### Option B: Using Python directly

Create a file called `my_first_model.py`:

```python
"""Train a tiny language model on Shakespeare."""
import jax.numpy as jnp
from flax import nnx

from linearnexus import (
    LMModel, ModelConfig,
    CharTokenizer, TextDataset, DataLoader,
    SFTTrainer, SFTConfig,
    create_optimizer,
    generate,
    download_shakespeare,
)

# Step 1: Get some data
print("📚 Downloading Shakespeare...")
download_shakespeare("shakespeare.txt")

# Step 2: Create tokenizer and dataset
print("🔤 Creating tokenizer...")
tokenizer = CharTokenizer.from_file("shakespeare.txt")
print(f"   Vocabulary size: {tokenizer.vocab_size}")

dataset = TextDataset("shakespeare.txt", tokenizer, seq_len=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Step 3: Create a small model
print("🏗️ Building model...")
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=128,      # Small for fast training
    n_layer=4,            # Few layers
    n_head=4,             # Few attention heads
    block_size=128,       # Context length
    block_pattern=["attention"],  # Pure GPT
)
model = LMModel(config, rngs=nnx.Rngs(42))
print(f"   Parameters: {model.count_params():,}")

# Step 4: Setup training
print("⚙️ Setting up training...")
optimizer = create_optimizer(
    "adamw",
    learning_rate=1e-3,
    total_steps=500,
    warmup_steps=50,
)
trainer = SFTTrainer(
    model, 
    optimizer, 
    SFTConfig(max_steps=500, log_every=50)
)

# Step 5: Train!
print("🚀 Training...")
trainer.train(iter(dataloader))

# Step 6: Generate text
print("\n📝 Generating text...")
prompt = "ROMEO:"
prompt_tokens = jnp.array([tokenizer.encode(prompt)])

import jax
output_tokens = generate(
    model, 
    prompt_tokens, 
    max_tokens=200,
    temperature=0.8,
    key=jax.random.key(0),
)

output_text = tokenizer.decode(output_tokens[0].tolist())
print(output_text)
```

Run it:

```bash
python my_first_model.py
```

You'll see the training progress and then some generated Shakespeare-ish text!

---

## Understanding the Code

Let's break down what just happened:

### The Model

```python
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,  # How many unique tokens
    hidden_size=128,                   # Width of the model
    n_layer=4,                         # Depth (number of blocks)
    n_head=4,                          # Attention heads (for GPT)
    block_size=128,                    # Max sequence length
    block_pattern=["attention"],       # What type of blocks to use
)
```

The `block_pattern` controls architecture:
- `["attention"]` → Pure GPT (attention blocks)
- `["mamba"]` → Pure Mamba (SSM blocks)  
- `["mamba", "mamba", "attention"]` → Hybrid (repeats pattern)

### The Data Pipeline

```python
# Tokenizer: converts text ↔ numbers
tokenizer = CharTokenizer.from_file("shakespeare.txt")
"hello" → [104, 101, 108, 108, 111]

# Dataset: serves chunks of tokenized text
dataset = TextDataset("shakespeare.txt", tokenizer, seq_len=128)

# DataLoader: creates batches for training
dataloader = DataLoader(dataset, batch_size=4)
```

### The Training Loop

```python
# Optimizer: how to update weights
optimizer = create_optimizer("adamw", learning_rate=1e-3, ...)

# Trainer: manages the training loop
trainer = SFTTrainer(model, optimizer, config)
trainer.train(dataloader)  # Does gradient descent
```

### Generation

```python
# Start with prompt tokens
prompt_tokens = tokenizer.encode("ROMEO:")

# Generate more tokens
output = generate(model, prompt_tokens, max_tokens=200)

# Decode back to text
text = tokenizer.decode(output)
```

---

## Key Concepts

### What is JAX?

JAX is Google's machine learning library. Key ideas:

```python
import jax
import jax.numpy as jnp

# Arrays (like NumPy)
x = jnp.array([1, 2, 3])

# Automatic differentiation
def f(x):
    return x ** 2

grad_f = jax.grad(f)  # df/dx = 2x
grad_f(3.0)  # Returns 6.0

# JIT compilation (fast!)
@jax.jit
def fast_function(x):
    return x @ x.T
```

### What is Flax NNx?

Flax is a neural network library for JAX. NNx is its modern API:

```python
from flax import nnx

class MyLayer(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(features, features, rngs=rngs)
    
    def __call__(self, x):
        return nnx.relu(self.linear(x))

# Create and use
layer = MyLayer(64, rngs=nnx.Rngs(0))
output = layer(input_array)
```

### GPT vs Mamba

**GPT (Transformer)**:
- Uses **attention** to relate all tokens to each other
- Complexity: O(n²) with sequence length
- Great for understanding relationships
- Slower for long sequences

**Mamba (State-Space Model)**:
- Uses a **recurrent state** to process sequences
- Complexity: O(n) with sequence length  
- Great for long sequences
- Faster inference

**Hybrid (Jamba)**:
- Alternates between attention and SSM blocks
- Gets benefits of both
- Example: attention every 8th layer

---

## Next Steps

### 1. Experiment with architectures

```python
# Try Mamba
config = ModelConfig(..., block_pattern=["mamba"])

# Try hybrid (Jamba-style)
config = ModelConfig(..., block_pattern=["mamba"] * 7 + ["attention"])
```

### 2. Try different optimizers

```python
# Muon (good for stability)
optimizer = create_optimizer("muon", learning_rate=1e-3)

# Sophia (second-order, efficient)
optimizer = create_optimizer("sophia", learning_rate=1e-4)
```

### 3. Scale up

```python
# Medium model (~85M params)
config, _ = create_model("gpt-medium")

# Use presets
from linearnexus import GPT_SMALL, MAMBA_MEDIUM
```

### 4. Read the code!

The whole framework is designed to be readable:

```
linearnexus/
├── models.py      # ~400 lines: model definition
├── train.py       # ~500 lines: training loops
├── optim.py       # ~400 lines: optimizers
├── data.py        # ~300 lines: data loading
├── generate.py    # ~400 lines: text generation
└── modules/       # Building blocks
    ├── attention/ # Attention mechanism
    └── ssm/       # Mamba mechanism
```

---

## Troubleshooting

### "CUDA out of memory"

Reduce batch size or model size:

```python
config = ModelConfig(hidden_size=128, n_layer=4)  # Smaller
dataloader = DataLoader(dataset, batch_size=2)     # Smaller batches
```

### "No GPU found"

JAX will use CPU by default. To verify GPU:

```python
import jax
print(jax.devices())  # Should show 'cuda' or 'gpu'
```

Install GPU version: `pip install jax[cuda12]`

### "Module not found"

Make sure you installed in development mode:

```bash
pip install -e .
```

### Training is slow

- Use JIT compilation (enabled by default)
- Reduce logging frequency: `SFTConfig(log_every=100)`
- Use mixed precision (coming soon)

### Loss is NaN

- Reduce learning rate: `learning_rate=1e-4`
- Add gradient clipping: `grad_clip=1.0`
- Check your data for issues

---

## Getting Help

- 📖 **Documentation**: See `docs/` folder
- ✅ **Testing**: See `docs/testing.md` for the full test matrix
- 🐛 **Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions

Happy training! 🚀
