# Checkpointing Guide

LinearNexus provides a robust checkpointing system for saving and loading models, optimizers, and training state.

## Quick Start

### Simple Save/Load

For basic use cases, use the convenience functions:

```python
from linearnexus import save_model, load_model

# Save model (auto-extracts config from model.config)
save_model("./my_checkpoint", model)

# Load model
model, config = load_model("./my_checkpoint")
```

### With Optimizer State

```python
from linearnexus import save_model, load_model, save_optimizer, load_optimizer
import optax

# Save both model and optimizer
save_model("./checkpoint", model)
save_optimizer("./checkpoint", optimizer)

# Load
model, config = load_model("./checkpoint")
tx = optax.adamw(1e-4)
optimizer = load_optimizer("./checkpoint", model, tx)
```

## CheckpointManager (Recommended)

For production training, use `CheckpointManager` which provides:
- Automatic cleanup of old checkpoints
- Best checkpoint tracking by metric
- Atomic saves (prevents corruption)
- Resume from interruption

### Basic Usage

```python
from linearnexus import CheckpointManager

# Initialize manager
manager = CheckpointManager(
    directory="./checkpoints",
    max_to_keep=5,           # Keep last 5 checkpoints
    best_metric="loss",      # Track best by loss
    best_mode="min",         # Lower is better
    best_to_keep=3,          # Keep 3 best
)

# During training
for step in range(1000):
    # ... training code ...
    
    if step % 100 == 0:
        manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            metrics={"loss": loss, "accuracy": acc},
        )
```

### Loading Checkpoints

```python
# Load most recent
state = manager.load_latest()
model = state.model
optimizer = state.optimizer
start_step = state.step

# Load best checkpoint
state = manager.load_best()

# Load specific step
state = manager.load_step(500)

# Direct path load
state = manager.load("./checkpoints/step_1000")
```

### Resume Training

```python
from linearnexus import CheckpointManager, LMModel, ModelConfig
import optax

# Initialize manager
manager = CheckpointManager("./checkpoints", max_to_keep=5)

# Try to resume
if manager.latest_step is not None:
    # Resume from checkpoint
    state = manager.load_latest(tx=optax.adamw(1e-4))
    model = state.model
    optimizer = state.optimizer
    start_step = state.step + 1
    print(f"Resuming from step {state.step}")
else:
    # Start fresh
    config = ModelConfig(...)
    model = LMModel(config, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4))
    start_step = 0

# Training loop
for step in range(start_step, max_steps):
    # ... training ...
    
    if step % save_interval == 0:
        manager.save(step=step, model=model, optimizer=optimizer)
```

## CheckpointState

When loading via `CheckpointManager`, you get a `CheckpointState` object:

```python
@dataclass
class CheckpointState:
    step: int                    # Training step
    model: nnx.Module            # Restored model
    optimizer: Optional[...]     # Optimizer (if loaded)
    config: Dict                 # Model config
    train_config: Dict           # Training config (if saved)
    metrics: Dict[str, float]    # Metrics at checkpoint time
    extra: Dict                  # Any extra data you saved
```

## File Format

Checkpoints are saved as directories containing:
```
step_1000/
├── model.npz          # Model weights (compressed numpy)
├── optimizer.npz      # Optimizer state (if saved)
├── optimizer_meta.pkl # Optimizer tree structure
├── config.json        # Model and training config
├── metrics.json       # Training metrics
├── structure.json     # Weight shapes (for debugging)
└── extra.json         # Additional user data
```

The `.npz` format is:
- 10-100x faster than JSON for large models
- Preserves full precision
- Compressed by default
- Cross-platform compatible

## Advanced Usage

### Save Extra Data

```python
manager.save(
    step=step,
    model=model,
    extra={
        "lr_schedule_state": scheduler.state_dict(),
        "best_eval_loss": 0.5,
        "wandb_run_id": "abc123",
    },
)

# Load
state = manager.load_latest()
scheduler.load_state_dict(state.extra["lr_schedule_state"])
```

### Custom Model Classes

```python
# If you have a custom model class
from my_models import CustomModel, CustomConfig

state = manager.load(
    "./checkpoint",
    model_cls=CustomModel,
    config_cls=CustomConfig,
)
```

### Different Optimizer for Fine-tuning

```python
# Load checkpoint but use different optimizer
state = manager.load_latest(
    tx=optax.adam(1e-5),  # New learning rate
)
```

The new API provides:
- Better performance (binary vs JSON)
- Automatic cleanup
- Best checkpoint tracking
- Atomic saves
- Full optimizer state preservation
