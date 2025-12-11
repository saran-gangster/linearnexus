"""Common building blocks shared across all architecture families.

This module provides the fundamental neural network components used by
attention, SSM, and hybrid architectures:
- MLP: Feed-forward network with gated variants
- RMSNorm: Root mean square layer normalization
- Embedding: Token and position embeddings
- RotaryEmbedding: Rotary position embeddings (RoPE)
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

Array = jax.Array


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.
    
    Simpler and faster than LayerNorm, used in LLaMA, Mistral, etc.
    
    Args:
        dim: Hidden dimension to normalize.
        eps: Small constant for numerical stability.
        rngs: Random number generators for parameter initialization.
    """
    
    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))
    
    def __call__(self, x: Array) -> Array:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Normalized tensor of same shape.
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight.value


class LayerNorm(nnx.Module):
    """Standard Layer Normalization.
    
    Args:
        dim: Hidden dimension to normalize.
        eps: Small constant for numerical stability.
        rngs: Random number generators for parameter initialization.
    """
    
    def __init__(self, dim: int, *, eps: float = 1e-5, rngs: nnx.Rngs):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))
        self.bias = nnx.Param(jnp.zeros((dim,)))
    
    def __call__(self, x: Array) -> Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * self.weight.value + self.bias.value


# -----------------------------------------------------------------------------
# Feed-Forward Networks
# -----------------------------------------------------------------------------

class MLP(nnx.Module):
    """Feed-forward network with optional gating (SwiGLU, GeGLU, etc).
    
    Standard: x -> Linear -> Act -> Linear -> out
    Gated:    x -> [Linear_gate * Act(Linear_up)] -> Linear_down -> out
    
    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Hidden layer dimension (typically 4x hidden_size).
        activation: Activation function name ("silu", "gelu", "relu").
        use_gating: Whether to use gated activation (SwiGLU-style).
        bias: Whether to use bias in linear layers.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: str = "silu",
        use_gating: bool = True,
        bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.use_gating = use_gating
        self.activation = _get_activation(activation)
        
        if use_gating:
            # SwiGLU-style: gate and up projections
            self.gate_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=bias, rngs=rngs)
            self.up_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=bias, rngs=rngs)
        else:
            # Standard MLP: single up projection
            self.up_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=bias, rngs=rngs)
        
        self.down_proj = nnx.Linear(intermediate_size, hidden_size, use_bias=bias, rngs=rngs)
    
    def __call__(self, x: Array) -> Array:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq, hidden_size]
            
        Returns:
            Output tensor of shape [batch, seq, hidden_size]
        """
        if self.use_gating:
            # SwiGLU: gate * activation(up)
            gate = self.activation(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
        else:
            hidden = self.activation(self.up_proj(x))
        
        return self.down_proj(hidden)


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

class Embedding(nnx.Module):
    """Token embedding layer with optional weight tying.
    
    Args:
        vocab_size: Size of vocabulary.
        embed_dim: Embedding dimension.
        rngs: Random number generators.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Initialize with small random values
        scale = 1.0 / math.sqrt(embed_dim)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (vocab_size, embed_dim)) * scale
        )
    
    def __call__(self, token_ids: Array) -> Array:
        """Look up embeddings for token IDs.
        
        Args:
            token_ids: Integer tensor of shape [batch, seq]
            
        Returns:
            Embeddings of shape [batch, seq, embed_dim]
        """
        return self.weight.value[token_ids]
    
    def unembed(self, hidden: Array) -> Array:
        """Project hidden states to vocabulary logits (weight tying).
        
        Args:
            hidden: Hidden states of shape [batch, seq, embed_dim]
            
        Returns:
            Logits of shape [batch, seq, vocab_size]
        """
        return hidden @ self.weight.value.T


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding (RoPE).
    
    Encodes position information directly into attention QK computation.
    Used in LLaMA, Mistral, GPT-NeoX, etc.
    
    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base for frequency computation (default 10000).
        rngs: Random number generators (unused, for API consistency).
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        *,
        base: float = 10000.0,
        rngs: nnx.Rngs,
    ):
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        
        # Precompute sin/cos for all positions
        positions = jnp.arange(max_seq_len, dtype=jnp.float32)
        freqs = jnp.outer(positions, inv_freq)  # [max_seq_len, dim/2]
        
        # Store as variables (not trainable)
        self.cos_cached = nnx.Variable(jnp.cos(freqs))  # [max_seq_len, dim/2]
        self.sin_cached = nnx.Variable(jnp.sin(freqs))  # [max_seq_len, dim/2]
    
    def __call__(self, seq_len: int, offset: int = 0) -> Tuple[Array, Array]:
        """Get cos/sin embeddings for positions [offset, offset + seq_len).
        
        Args:
            seq_len: Sequence length to fetch.
            offset: Starting position (for incremental generation).
            
        Returns:
            Tuple of (cos, sin) each of shape [seq_len, dim/2]
        """
        return (
            self.cos_cached.value[offset : offset + seq_len],
            self.sin_cached.value[offset : offset + seq_len],
        )


def apply_rotary_emb(
    x: Array,
    cos: Array,
    sin: Array,
) -> Array:
    """Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor of shape [batch, seq, heads, head_dim]
        cos: Cosine embeddings of shape [seq, head_dim/2]
        sin: Sine embeddings of shape [seq, head_dim/2]
        
    Returns:
        Rotated tensor of same shape as x.
    """
    # Split into even/odd pairs
    x1 = x[..., ::2]   # [batch, seq, heads, head_dim/2]
    x2 = x[..., 1::2]  # [batch, seq, heads, head_dim/2]
    
    # Reshape cos/sin for broadcasting: [1, seq, 1, head_dim/2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    # Apply rotation
    rotated = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1)
    
    # Interleave back: [batch, seq, heads, head_dim]
    return rotated.reshape(x.shape)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _get_activation(name: str) -> Callable[[Array], Array]:
    """Get activation function by name."""
    activations = {
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,  # Alias
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "identity": lambda x: x,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(activations.keys())}")
    return activations[name]


def get_norm(
    norm_type: Literal["rmsnorm", "layernorm"],
    dim: int,
    *,
    eps: float = 1e-6,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Factory for normalization layers."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps, rngs=rngs)
    elif norm_type == "layernorm":
        return LayerNorm(dim, eps=eps, rngs=rngs)
    else:
        raise ValueError(f"Unknown norm type '{norm_type}'")
