#!/usr/bin/env python3
"""Debug script to isolate NaN in Gated DeltaNet (including gradients)."""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from linearnexus.models import ModelConfig, LMModel
from linearnexus.train.base import cross_entropy_loss

def check_nan(name, tensor, verbose=True):
    """Check if tensor has NaN/Inf and print debug info."""
    has_nan = bool(jnp.any(jnp.isnan(tensor)))
    has_inf = bool(jnp.any(jnp.isinf(tensor)))
    
    if has_nan or has_inf or verbose:
        min_val = float(jnp.min(tensor))
        max_val = float(jnp.max(tensor))
        mean_val = float(jnp.mean(tensor))
        status = "❌ NaN" if has_nan else ("❌ Inf" if has_inf else "✓")
        print(f"  {name}: {status} | shape={tensor.shape} | min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    
    return has_nan or has_inf


def check_grads_for_nan(grads, prefix=""):
    """Recursively check gradient dict for NaN."""
    found_nan = False
    
    def _check(obj, path=""):
        nonlocal found_nan
        if isinstance(obj, dict):
            for k, v in obj.items():
                _check(v, f"{path}.{k}" if path else k)
        elif hasattr(obj, 'shape'):  # It's an array
            has_nan = bool(jnp.any(jnp.isnan(obj)))
            has_inf = bool(jnp.any(jnp.isinf(obj)))
            if has_nan or has_inf:
                found_nan = True
                status = "❌ NaN" if has_nan else "❌ Inf"
                print(f"  {prefix}{path}: {status} | shape={obj.shape}")
    
    _check(grads)
    return found_nan


def debug_gated_deltanet():
    """Test gated_deltanet with gradient computation."""
    
    print("=" * 60)
    print("Gated DeltaNet Gradient Debug")
    print("=" * 60)
    
    # Create config
    config = ModelConfig(
        vocab_size=65,
        hidden_size=256,
        n_layers=2,
        block_pattern=["gated_deltanet"],
        gated_deltanet_heads=4,
        gated_deltanet_v_heads=4,
        gated_deltanet_head_dim=64,
        gated_deltanet_expand_v=2.0,
        gated_deltanet_use_short_conv=True,
        gated_deltanet_use_gate=True,
    )
    
    print(f"\nConfig: hidden_size={config.hidden_size}, n_layers={config.n_layers}")
    
    # Create model
    print("\n--- Creating Model ---")
    rngs = nnx.Rngs(42)
    model = LMModel(config, rngs=rngs)
    print(f"Model params: {model.count_params():,}")
    
    # Create batch
    batch_size, seq_len = 2, 64
    input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 65)
    labels = jax.random.randint(jax.random.key(1), (batch_size, seq_len), 0, 65)
    batch = {"input_ids": input_ids, "labels": labels}
    
    print(f"\n--- Test Forward Pass ---")
    logits, _ = model(input_ids)
    loss = cross_entropy_loss(logits, labels)
    print(f"Forward loss: {float(loss):.4f}")
    
    if jnp.isnan(loss) or jnp.isinf(loss):
        print("❌ Forward pass produces NaN/Inf loss!")
        return
    else:
        print("✓ Forward pass OK")
    
    # Split model for gradient computation
    print("\n--- Testing Gradient Computation ---")
    graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
    params = nnx.to_pure_dict(params_state)
    
    # Define loss function
    def loss_fn(params, batch, graphdef, params_state, rest_state):
        nnx.replace_by_pure_dict(params_state, params)
        merged_model = nnx.merge(graphdef, params_state, rest_state)
        logits, _ = merged_model(batch["input_ids"])
        return cross_entropy_loss(logits, batch["labels"])
    
    # Compute gradients
    print("Computing gradients...")
    try:
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, graphdef, params_state, rest_state)
        print(f"Loss from grad computation: {float(loss):.4f}")
        
        # Check gradients for NaN
        print("\n--- Checking Gradients for NaN ---")
        has_nan_grads = check_grads_for_nan(grads)
        
        if has_nan_grads:
            print("\n❌ GRADIENTS CONTAIN NaN/Inf!")
        else:
            print("\n✓ All gradients are finite!")
            
            # Try one optimization step
            print("\n--- Testing Optimizer Step ---")
            optimizer = optax.adamw(learning_rate=1e-4)
            opt_state = optimizer.init(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            # Check new params for NaN
            print("Checking updated params...")
            has_nan_params = check_grads_for_nan(new_params, prefix="param: ")
            
            if has_nan_params:
                print("\n❌ UPDATED PARAMS CONTAIN NaN!")
            else:
                print("\n✓ Updated params are finite!")
                
                # Compute loss with new params
                loss2 = loss_fn(new_params, batch, graphdef, params_state, rest_state)
                print(f"\nLoss after 1 step: {float(loss2):.4f}")
                
                if jnp.isnan(loss2) or jnp.isinf(loss2):
                    print("❌ Loss became NaN/Inf after 1 step!")
                else:
                    print("✓ Training step successful!")
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_gated_deltanet()
