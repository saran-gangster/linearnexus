#!/usr/bin/env python3
"""Debug script to isolate NaN in Gated DeltaNet with benchmark-like training."""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from linearnexus.models import ModelConfig, LMModel
from linearnexus.data import CharTokenizer, TextDataset, DataLoader, download_shakespeare
from linearnexus.optim import create_optimizer
from linearnexus.train.base import cross_entropy_loss


def check_nan(name, tensor, verbose=False):
    """Check if tensor has NaN/Inf and print debug info."""
    has_nan = bool(jnp.any(jnp.isnan(tensor)))
    has_inf = bool(jnp.any(jnp.isinf(tensor)))
    
    if has_nan or has_inf or verbose:
        min_val = float(jnp.min(tensor))
        max_val = float(jnp.max(tensor))
        mean_val = float(jnp.mean(tensor))
        status = "❌ NaN" if has_nan else ("❌ Inf" if has_inf else "✓")
        print(f"  {name}: {status} | min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    return has_nan or has_inf


def debug_training():
    """Test gated_deltanet with benchmark-like training loop."""
    
    print("=" * 60)
    print("Gated DeltaNet Training Debug")
    print("=" * 60)
    
    # Use EXACT same config as benchmark_architectures.py
    config = ModelConfig(
        vocab_size=65,
        hidden_size=256,
        n_layers=6,  # Same as benchmark
        block_pattern=["gated_deltanet"],
        gated_deltanet_heads=4,
        gated_deltanet_v_heads=4,
        gated_deltanet_head_dim=64,
        gated_deltanet_expand_v=2.0,
        gated_deltanet_use_short_conv=True,
        gated_deltanet_use_gate=True,
    )
    
    batch_size = 8
    seq_len = 128
    max_steps = 25
    lr = 3e-4
    warmup_steps = 5
    
    print(f"\nConfig: hidden_size={config.hidden_size}, n_layers={config.n_layers}")
    print(f"Training: batch_size={batch_size}, seq_len={seq_len}, lr={lr}")
    
    # Create model
    print("\n--- Creating Model ---")
    rngs = nnx.Rngs(42)
    model = LMModel(config, rngs=rngs)
    print(f"Model params: {model.count_params():,}")
    
    # Download data
    print("\n--- Loading Data ---")
    data_path = download_shakespeare()
    tokenizer = CharTokenizer.from_file(data_path)
    dataset = TextDataset(data_path, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, seed=42)
    
    # Create optimizer
    optimizer = create_optimizer(
        "adamw",
        learning_rate=lr,
        total_steps=max_steps,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        grad_clip=1.0,
    )
    
    # Split model
    graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
    params = nnx.to_pure_dict(params_state)
    opt_state = optimizer.init(params)
    
    # Define loss function
    def loss_fn(params, batch, graphdef, params_state, rest_state):
        nnx.replace_by_pure_dict(params_state, params)
        merged_model = nnx.merge(graphdef, params_state, rest_state)
        logits, _ = merged_model(batch["input_ids"])
        return cross_entropy_loss(logits, batch["labels"])
    
    # JIT compile
    @jax.jit
    def train_step(params, opt_state, batch, graphdef, params_state, rest_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, graphdef, params_state, rest_state)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Training loop
    print("\n--- Training Loop ---")
    data_iter = iter(dataloader)
    
    for step in range(max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        params, opt_state, loss = train_step(params, opt_state, batch, graphdef, params_state, rest_state)
        loss_val = float(loss)
        
        print(f"  Step {step+1:3d}: loss={loss_val:.4f}", end="")
        
        if jnp.isnan(loss) or jnp.isinf(loss):
            print(" ❌ NaN/Inf detected!")
            
            # Debug: check params
            print("\n  Checking params for NaN...")
            def _check_params(obj, path=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _check_params(v, f"{path}.{k}" if path else k)
                elif hasattr(obj, 'shape'):
                    if bool(jnp.any(jnp.isnan(obj))) or bool(jnp.any(jnp.isinf(obj))):
                        print(f"    ❌ {path}: NaN/Inf | shape={obj.shape}")
            
            _check_params(params)
            return
        else:
            print(" ✓")
    
    print(f"\n✓ Training completed {max_steps} steps without NaN!")


if __name__ == "__main__":
    debug_training()
