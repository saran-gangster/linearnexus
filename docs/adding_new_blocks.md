# Adding New Block Types to LinearNexus

**Last updated**: 2025-11-30

This guide walks through adding a new block type (like attention or Mamba) to LinearNexus. The architecture is intentionally simple — you create a module, register it in `models.py`, and you're done.

---

## Table of Contents

1. [Overview](#overview)
2. [Step 1: Create Your Block Module](#step-1-create-your-block-module)
3. [Step 2: Register in models.py](#step-2-register-in-modelspy)
4. [Step 3: Add Tests](#step-3-add-tests)
5. [Example: Adding a Sliding Window Attention Block](#example-adding-a-sliding-window-attention-block)
6. [Best Practices](#best-practices)

---

## Overview

LinearNexus uses a simple **block protocol** — all blocks have the same interface:

```python
class SomeBlock(nnx.Module):
    def __call__(
        self,
        x: jax.Array,                          # [batch, seq, hidden]
        *,
        state: Optional[BlockState] = None,    # For caching
        mask: Optional[jax.Array] = None,      # Attention mask
        mode: Optional[str] = None,            # "chunk" or "recurrent"
    ) -> tuple[jax.Array, Optional[BlockState]]:
        ...
        return output, new_state
```

This means any block can be swapped for any other, enabling hybrid architectures.

### Directory Structure

```
linearnexus/
├── modules/
│   ├── common.py              # Shared: MLP, RMSNorm, Embedding
│   ├── attention/
│   │   ├── __init__.py
│   │   └── causal.py          # CausalSelfAttention, AttentionBlock
│   ├── ssm/
│   │   ├── __init__.py
│   │   └── mamba.py           # MambaBlock
│   ├── sparse/                # [Your new block type]
│   │   ├── __init__.py
│   │   └── sliding_window.py
│   └── linear_attn/           # [Future]
│       └── ...
└── models.py                  # Block registration
```

---

## Step 1: Create Your Block Module

Create a new file in the appropriate `modules/` subdirectory.

### Block Template

```python
"""Your new block type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from ..common import MLP, RMSNorm


@dataclass
class YourBlockState:
    """State for caching during generation."""
    # Add whatever state you need for autoregressive generation
    cache: jax.Array  # Example: [batch, max_seq, hidden]
    cache_index: int  # Current position in cache
    
    @classmethod
    def zeros(cls, batch_size: int, max_seq: int, hidden_size: int, dtype=jnp.float32):
        return cls(
            cache=jnp.zeros((batch_size, max_seq, hidden_size), dtype=dtype),
            cache_index=0,
        )


class YourBlock(nnx.Module):
    """Your block description.
    
    Args:
        hidden_size: Model dimension.
        your_param: Description.
        rngs: Flax NNx random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        your_param: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.your_param = your_param
        
        # Layer norm
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)
        
        # Your mechanism
        self.your_layer = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        
        # MLP (shared across all blocks)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = MLP(hidden_size, mlp_hidden, rngs=rngs)
        
        # Optional dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None
    
    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[YourBlockState] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[str] = None,
    ) -> tuple[jax.Array, Optional[YourBlockState]]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq, hidden].
            state: Optional state for caching.
            mask: Optional attention mask.
            mode: Processing mode ("chunk" or "recurrent").
            
        Returns:
            Tuple of (output, new_state).
        """
        # Pre-norm architecture
        residual = x
        x = self.ln1(x)
        
        # Your mechanism
        x, new_state = self._your_mechanism(x, state=state, mask=mask, mode=mode)
        
        if self.dropout:
            x = self.dropout(x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        if self.dropout:
            x = self.dropout(x)
        x = residual + x
        
        return x, new_state
    
    def _your_mechanism(
        self,
        x: jax.Array,
        *,
        state: Optional[YourBlockState],
        mask: Optional[jax.Array],
        mode: Optional[str],
    ) -> tuple[jax.Array, YourBlockState]:
        """Your core mechanism logic."""
        # Implement your mechanism here
        # Return output and updated state
        
        batch_size, seq_len, hidden = x.shape
        
        # Example: simple linear transform (replace with your logic)
        output = self.your_layer(x)
        
        # Example: state management
        if state is None:
            new_state = YourBlockState.zeros(batch_size, 1024, hidden, x.dtype)
        else:
            new_state = state  # Update as needed
        
        return output, new_state
    
    def init_state(
        self,
        batch_size: int,
        max_seq: int = 1024,
        dtype: jnp.dtype = jnp.float32,
    ) -> YourBlockState:
        """Initialize state for generation."""
        return YourBlockState.zeros(batch_size, max_seq, self.hidden_size, dtype)
```

### Export Your Block

Create `modules/yourtype/__init__.py`:

```python
"""Your block type exports."""

from .your_block import YourBlock, YourBlockState

__all__ = ["YourBlock", "YourBlockState"]
```

---

## Step 2: Register in models.py

Add your block to the `_create_block` function in `linearnexus/models.py`:

```python
def _create_block(
    block_type: str,
    config: ModelConfig,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create a block based on type string."""
    if block_type == "attention":
        from .modules.attention import AttentionBlock
        return AttentionBlock(
            hidden_size=config.hidden_size,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            block_size=config.block_size,
            dropout=config.dropout,
            rngs=rngs,
        )
    elif block_type == "mamba":
        from .modules.ssm import MambaBlock
        return MambaBlock(
            hidden_size=config.hidden_size,
            ssm_state=config.ssm_state,
            conv_size=config.conv_size,
            expand=config.ssm_expand,
            rngs=rngs,
        )
    # ADD YOUR BLOCK HERE:
    elif block_type == "yourtype":
        from .modules.yourtype import YourBlock
        return YourBlock(
            hidden_size=config.hidden_size,
            your_param=config.your_param,  # Add config field if needed
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown block type: {block_type}")
```

If your block needs new config parameters, add them to `ModelConfig`:

```python
@dataclass
class ModelConfig:
    # ... existing fields ...
    
    # Your new parameters
    your_param: int = 64
```

---

## Step 3: Add Tests

Create `tests/test_your_block.py`:

```python
"""Tests for your block type."""

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from linearnexus.modules.yourtype import YourBlock, YourBlockState


class TestYourBlock:
    """Test suite for YourBlock."""
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        rngs = nnx.Rngs(0)
        block = YourBlock(hidden_size=64, your_param=16, rngs=rngs)
        
        x = jnp.ones((2, 32, 64))  # [batch, seq, hidden]
        output, state = block(x)
        
        assert output.shape == x.shape
        assert state is not None
    
    def test_with_state(self):
        """Test that state is properly updated."""
        rngs = nnx.Rngs(0)
        block = YourBlock(hidden_size=64, your_param=16, rngs=rngs)
        
        x = jnp.ones((2, 1, 64))  # Single token
        state = block.init_state(batch_size=2)
        
        output1, state1 = block(x, state=state)
        output2, state2 = block(x, state=state1)
        
        # States should be different after processing
        assert output1.shape == (2, 1, 64)
        assert output2.shape == (2, 1, 64)
    
    def test_gradients_flow(self):
        """Test that gradients flow through the block."""
        rngs = nnx.Rngs(0)
        block = YourBlock(hidden_size=64, your_param=16, rngs=rngs)
        
        def loss_fn(block, x):
            output, _ = block(x)
            return jnp.mean(output ** 2)
        
        x = jnp.ones((2, 32, 64))
        loss, grads = nnx.value_and_grad(loss_fn)(block, x)
        
        # Check that gradients are non-zero
        assert loss > 0
        # grads is the gradient w.r.t. block parameters
    
    def test_in_model(self):
        """Test block works in full model."""
        from linearnexus import create_model, ModelConfig
        
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            n_layer=2,
            block_pattern=["yourtype"],  # Use your block
        )
        
        model = create_model(config, rngs=nnx.Rngs(0))
        
        x = jnp.ones((2, 16), dtype=jnp.int32)
        logits, state = model(x)
        
        assert logits.shape == (2, 16, 100)
```

Run tests:

```bash
pytest tests/test_your_block.py -v
```

---

## Example: Adding a Sliding Window Attention Block

Here's a complete example of adding sliding window (local) attention:

### 1. Create the module

`linearnexus/modules/sparse/sliding_window.py`:

```python
"""Sliding window (local) attention block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from ..common import MLP, RMSNorm


@dataclass
class SlidingWindowState:
    """KV cache for sliding window attention."""
    k_cache: jax.Array  # [batch, window_size, n_head, head_dim]
    v_cache: jax.Array  # [batch, window_size, n_head, head_dim]
    cache_index: int
    
    @classmethod
    def zeros(cls, batch_size: int, window_size: int, n_head: int, head_dim: int, dtype=jnp.float32):
        return cls(
            k_cache=jnp.zeros((batch_size, window_size, n_head, head_dim), dtype=dtype),
            v_cache=jnp.zeros((batch_size, window_size, n_head, head_dim), dtype=dtype),
            cache_index=0,
        )


class SlidingWindowAttention(nnx.Module):
    """Multi-head attention with sliding window (local attention).
    
    Only attends to the last `window_size` tokens, reducing complexity
    from O(n²) to O(n * window_size).
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        window_size: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.window_size = window_size
        self.head_dim = hidden_size // n_head
        
        # Projections
        self.q_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.o_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
    
    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[SlidingWindowState] = None,
        mask: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, SlidingWindowState]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to heads
        q = q.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        
        # Create sliding window mask
        # Each position can only attend to positions within window_size
        positions = jnp.arange(seq_len)
        window_mask = jnp.abs(positions[:, None] - positions[None, :]) < self.window_size
        causal_mask = positions[:, None] >= positions[None, :]
        combined_mask = window_mask & causal_mask  # [seq, seq]
        
        # Attention
        scale = self.head_dim ** -0.5
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        
        # Apply mask
        scores = jnp.where(combined_mask[None, None, :, :], scores, -1e9)
        
        attn = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        
        # Reshape and project
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        
        # Create state (simplified - full impl would update cache)
        new_state = SlidingWindowState.zeros(
            batch_size, self.window_size, self.n_head, self.head_dim, x.dtype
        )
        
        return output, new_state


class SlidingWindowBlock(nnx.Module):
    """Transformer block with sliding window attention."""
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        window_size: int = 256,
        *,
        mlp_ratio: float = 4.0,
        rngs: nnx.Rngs,
    ):
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)
        
        self.attn = SlidingWindowAttention(
            hidden_size, n_head, window_size, rngs=rngs
        )
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = MLP(hidden_size, mlp_hidden, rngs=rngs)
    
    def __call__(
        self,
        x: jax.Array,
        *,
        state: Optional[SlidingWindowState] = None,
        mask: Optional[jax.Array] = None,
        mode: Optional[str] = None,
    ) -> tuple[jax.Array, SlidingWindowState]:
        # Attention
        residual = x
        x = self.ln1(x)
        x, new_state = self.attn(x, state=state, mask=mask)
        x = residual + x
        
        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x, new_state
```

### 2. Export it

`linearnexus/modules/sparse/__init__.py`:

```python
"""Sparse attention modules."""

from .sliding_window import SlidingWindowBlock, SlidingWindowState

__all__ = ["SlidingWindowBlock", "SlidingWindowState"]
```

### 3. Register it

In `linearnexus/models.py`, add to `_create_block`:

```python
elif block_type == "sliding_window":
    from .modules.sparse import SlidingWindowBlock
    return SlidingWindowBlock(
        hidden_size=config.hidden_size,
        n_head=config.n_head,
        window_size=getattr(config, 'window_size', 256),
        rngs=rngs,
    )
```

### 4. Use it

```python
from linearnexus import create_model, ModelConfig

# Pure sliding window
config = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layer=12,
    n_head=12,
    block_pattern=["sliding_window"],
)

# Hybrid: sliding window + global attention
config = ModelConfig(
    ...
    block_pattern=["sliding_window"] * 3 + ["attention"],  # Global every 4th
)
```

---

## Best Practices

### 1. Follow the Block Protocol

Always return `(output, state)` tuple, even if state is None:

```python
def __call__(self, x, *, state=None, mask=None, mode=None):
    output = self.process(x)
    return output, None  # ✓ Always return tuple
```

### 2. Use Pre-Norm Architecture

Apply layer norm before the mechanism, not after:

```python
# ✓ Pre-norm (preferred)
x = self.ln(x)
x = self.mechanism(x)
x = residual + x

# ✗ Post-norm (avoid)
x = self.mechanism(x)
x = residual + x
x = self.ln(x)
```

### 3. Initialize State Lazily

Don't require state on first call:

```python
def __call__(self, x, *, state=None, ...):
    if state is None:
        state = self.init_state(x.shape[0])  # ✓ Create if needed
    ...
```

### 4. Support Both Training and Inference

Use the `mode` parameter to switch behavior:

```python
def __call__(self, x, *, state=None, mode=None):
    if mode == "recurrent" or x.shape[1] == 1:
        # Autoregressive: single token, use cache
        return self._forward_recurrent(x, state)
    else:
        # Training: full sequence, parallel
        return self._forward_parallel(x)
```

### 5. Document Shapes

Always document tensor shapes in docstrings and comments:

```python
def forward(self, x: jax.Array) -> jax.Array:
    """Process input.
    
    Args:
        x: Input tensor [batch, seq, hidden].
        
    Returns:
        Output tensor [batch, seq, hidden].
    """
    # [batch, seq, hidden] -> [batch, seq, n_head, head_dim]
    x = x.reshape(batch, seq, self.n_head, self.head_dim)
```

---

## See Also

- [architecture_overview.md](./architecture_overview.md) — Codebase structure
- [training_guide.md](./training_guide.md) — Training your new block
- [flax_nnx_quick_reference.md](./flax_nnx_quick_reference.md) — Flax NNx patterns
