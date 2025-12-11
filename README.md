# LinearNexus

**Minimal LLM training framework in JAX** — nanoGPT-style simplicity with multi-architecture support.

Train GPT, Mamba, or hybrid models from scratch with custom optimizers (AdamW, Muon, Sophia) and modern training paradigms (SFT, GRPO, PPO).

> 🚀 **New to LinearNexus?** Start with [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step tutorial.

---

## Features

| Component | Description |
|-----------|-------------|
| **Architectures** | GPT (dense attention), Mamba (selective SSM), Hybrid (Jamba-style interleaved) |
| **Optimizers** | AdamW, Muon (momentum orthogonalization), Sophia (second-order) |
| **Training** | SFT, GRPO (group relative policy optimization), PPO |
| **Generation** | Temperature, top-k, top-p sampling with KV/SSM state caching |
| **Data** | Character-level + BPE tokenizers, memory-mapped datasets |

Built on JAX + Flax NNx for clean, functional design that's easy to hack.

---

## Quick Start

```bash
# Install
pip install -e .

# Download data and train GPT on Shakespeare
python train_lm.py --model gpt-small --download-shakespeare

# Train Mamba with Muon optimizer
python train_lm.py --model mamba-small --optimizer muon --lr 1e-3

# Train Jamba-style hybrid (every 8th layer is attention)
python train_lm.py --model jamba-small

# Generate text
python sample.py --checkpoint checkpoints/step_5000 --prompt "To be or not"
```

---

## Architecture

```
linearnexus/
├── models.py       # LMModel, configs, block pattern factory
├── optim.py        # AdamW, Muon, Sophia optimizers
├── train.py        # SFT, GRPO, PPO trainers
├── data.py         # Tokenizers, datasets, dataloaders
├── generate.py     # Sampling and generation utilities
├── modules/
│   ├── attention/  # CausalSelfAttention, KV cache
│   ├── ssm/        # MambaBlock, selective scan
│   ├── sparse/     # [Phase 2] Sliding window, block-sparse
│   ├── linear_attn/# [Phase 3] DeltaNet, GLA, RetNet
│   └── hybrid/     # [Phase 3] Jamba-style interleaved
└── core/           # Shared utilities (cache, conv, config)
```

---

## Model Presets

```python
from linearnexus import create_model, LMModel
import flax.nnx as nnx

# GPT-2 small (124M params style)
config, _ = create_model("gpt-small")
model = LMModel(config, rngs=nnx.Rngs(0))

# Mamba (pure SSM)
config, _ = create_model("mamba-small")

# Jamba-style hybrid (every 8th layer is attention)
config, _ = create_model("jamba-small")

# Custom hybrid pattern
from linearnexus.models import ModelConfig
config = ModelConfig(
    hidden_size=768,
    n_layers=12,
    block_pattern=["mamba", "mamba", "attention"],  # Repeats to fill n_layers
)
```

---

## Custom Optimizers

```python
from linearnexus import create_optimizer, get_optimizer
import optax

# Quick setup with schedules and clipping
optimizer = create_optimizer(
    "muon",
    learning_rate=1e-3,
    total_steps=10000,
    warmup_steps=100,
    grad_clip=1.0,
)

# Compose with optax
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    get_optimizer("sophia", learning_rate=1e-4),
)
```

---

## Training

```python
from linearnexus import (
    LMModel, ModelConfig,
    CharTokenizer, TextDataset, DataLoader,
    SFTTrainer, SFTConfig,
    create_optimizer,
)
import flax.nnx as nnx

# Setup
tokenizer = CharTokenizer.from_file("data.txt")
dataset = TextDataset("data.txt", tokenizer, seq_len=256)
loader = DataLoader(dataset, batch_size=4)

config = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_size=256, n_layers=6)
model = LMModel(config, rngs=nnx.Rngs(0))

optimizer = create_optimizer("adamw", learning_rate=3e-4, total_steps=5000)
trainer = SFTTrainer(model, optimizer, SFTConfig(max_steps=5000))

trainer.train(iter(loader))
```

---

## Generation

```python
from linearnexus import generate, complete
import jax.numpy as jnp

# Low-level generation
prompt = jnp.array([[1, 2, 3, 4]])  # Token IDs
output = generate(model, prompt, max_tokens=100, temperature=0.8, top_k=50)

# High-level completion
text = complete(model, tokenizer, "Once upon a time", max_tokens=100)
```

---

## Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| Phase 1 | ✅ Done | GPT + Mamba + SFT + AdamW/Muon/Sophia |
| Phase 2 | 🚧 Planned | Sliding window attention, GRPO refinements |
| Phase 3 | 📋 Backlog | DeltaNet, GLA, RetNet, full PPO, distributed |
| Future | 💡 Ideas | Custom Pallas GPU/TPU kernels, RLHF |

---

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | 🚀 **Start here!** Beginner tutorial |
| [docs/concepts.md](docs/concepts.md) | 📚 LLM fundamentals explained |
| [docs/architecture_overview.md](docs/architecture_overview.md) | How the codebase is organized |
| [docs/training_guide.md](docs/training_guide.md) | Deep dive into training |
| [docs/adding_new_blocks.md](docs/adding_new_blocks.md) | Extending with new architectures |
| [docs/flax_nnx_quick_reference.md](docs/flax_nnx_quick_reference.md) | Flax NNx patterns |

---

## Philosophy

LinearNexus follows [nanoGPT](https://github.com/karpathy/nanoGPT) principles:

1. **Minimal** — Read the code, understand it all
2. **Hackable** — Single files over deep hierarchies
3. **Educational** — Clear over clever
4. **Practical** — Works on consumer GPUs (12GB)

No magic, no hidden complexity, just JAX.

---

## Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.28
- Flax ≥ 0.8.3
- Optax ≥ 0.2.3

Optional: `tiktoken` for BPE, `wandb` for logging.

---

## License

MIT