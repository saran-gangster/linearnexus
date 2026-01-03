#!/usr/bin/env python3
"""Debug script to isolate NaN in Gated DeltaNet."""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from linearnexus.models import ModelConfig, LMModel

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

def debug_gated_deltanet():
    """Test gated_deltanet with debug output."""
    
    print("=" * 60)
    print("Gated DeltaNet NaN Debug")
    print("=" * 60)
    
    # Create config
    config = ModelConfig(
        vocab_size=65,
        hidden_size=256,
        n_layers=2,  # Use fewer layers for debug
        block_pattern=["gated_deltanet"],
        gated_deltanet_heads=4,
        gated_deltanet_v_heads=4,
        gated_deltanet_head_dim=64,
        gated_deltanet_expand_v=2.0,
        gated_deltanet_use_short_conv=True,
        gated_deltanet_use_gate=True,
    )
    
    print(f"\nConfig: hidden_size={config.hidden_size}, n_layers={config.n_layers}")
    print(f"GatedDeltaNet: heads={config.gated_deltanet_heads}, v_heads={config.gated_deltanet_v_heads}")
    print(f"head_dim={config.gated_deltanet_head_dim}, expand_v={config.gated_deltanet_expand_v}")
    
    # Create model
    print("\n--- Creating Model ---")
    rngs = nnx.Rngs(42)
    model = LMModel(config, rngs=rngs)
    print(f"Model params: {model.count_params():,}")
    
    # Check model parameters for NaN
    print("\n--- Checking Model Parameters ---")
    for block_idx, block in enumerate(model.blocks):
        if hasattr(block, 'A_log'):
            check_nan(f"Block {block_idx} A_log", block.A_log.value)
        if hasattr(block, 'dt_bias'):
            check_nan(f"Block {block_idx} dt_bias", block.dt_bias.value)
    
    # Create dummy input
    batch_size, seq_len = 2, 64
    input_ids = jax.random.randint(jax.random.key(0), (batch_size, seq_len), 0, 65)
    
    print(f"\n--- Forward Pass (batch={batch_size}, seq_len={seq_len}) ---")
    
    # Get embedding
    x = model.embed(input_ids)
    check_nan("Embedding output", x)
    
    # Pass through each block
    for block_idx, block in enumerate(model.blocks):
        print(f"\n--- Block {block_idx} ---")
        
        # Store input for debugging
        residual = x
        check_nan(f"  Input to block", x)
        
        # Manually trace through GatedDeltaNetBlock
        if hasattr(block, 'norm'):
            x_norm = block.norm(x)
            check_nan(f"  After norm", x_norm)
        
        if hasattr(block, 'q_proj'):
            q = block.q_proj(x_norm)
            check_nan(f"  q_proj", q)
        
        if hasattr(block, 'k_proj'):
            k = block.k_proj(x_norm)
            check_nan(f"  k_proj", k)
            
        if hasattr(block, 'v_proj'):
            v = block.v_proj(x_norm)
            check_nan(f"  v_proj", v)
        
        if hasattr(block, 'a_proj'):
            a = block.a_proj(x_norm)
            check_nan(f"  a_proj", a)
            
        if hasattr(block, 'b_proj'):
            b = block.b_proj(x_norm)
            check_nan(f"  b_proj", b)
            beta = jax.nn.sigmoid(b)
            check_nan(f"  beta=sigmoid(b)", beta)
        
        if hasattr(block, 'A_log') and hasattr(block, 'dt_bias'):
            A_exp = jnp.exp(block.A_log[...])
            check_nan(f"  exp(A_log)", A_exp)
            
            softplus_a = jax.nn.softplus(a + block.dt_bias[...])
            check_nan(f"  softplus(a + dt_bias)", softplus_a)
            
            g = -A_exp * softplus_a
            check_nan(f"  g (decay gate)", g)
        
        # Now run full block
        try:
            x, _ = block(residual)
            check_nan(f"  Block output", x)
        except Exception as e:
            print(f"  ❌ Block forward failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Final output
    print("\n--- Final Layers ---")
    x = model.final_norm(x)
    check_nan("After final_norm", x)
    
    logits = model.embed.unembed(x)
    check_nan("Logits", logits)
    
    print("\n--- Computing Loss ---")
    labels = jax.random.randint(jax.random.key(1), (batch_size, seq_len), 0, 65)
    
    # Cross entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    check_nan("Log probs", log_probs)
    
    one_hot = jax.nn.one_hot(labels, 65)
    loss = -jnp.sum(one_hot * log_probs) / (batch_size * seq_len)
    
    print(f"\nFinal loss: {float(loss):.4f}")
    
    if jnp.isnan(loss) or jnp.isinf(loss):
        print("\n❌ LOSS IS NaN/Inf!")
    else:
        print("\n✓ Loss is finite!")


if __name__ == "__main__":
    debug_gated_deltanet()
