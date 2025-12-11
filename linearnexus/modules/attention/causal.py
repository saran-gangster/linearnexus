"""Causal self-attention implementation.

Dense O(n²) attention with causal masking, optimized for clarity and
correctness. Supports MHA, GQA, and MQA variants with optional RoPE.

For efficient inference, use KVCache to avoid recomputing past keys/values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.common import MLP, RMSNorm, RotaryEmbedding, apply_rotary_emb, get_norm

Array = jax.Array


# -----------------------------------------------------------------------------
# KV Cache for Autoregressive Generation
# -----------------------------------------------------------------------------

@dataclass
class KVCache:
    """Key-Value cache for efficient autoregressive decoding.
    
    Stores past keys and values to avoid recomputation during generation.
    
    Attributes:
        keys: Cached keys of shape [batch, max_seq_len, n_kv_heads, head_dim]
        values: Cached values of shape [batch, max_seq_len, n_kv_heads, head_dim]
        position: Current position in the cache (number of tokens cached)
    """
    keys: Array    # [batch, max_seq_len, n_kv_heads, head_dim]
    values: Array  # [batch, max_seq_len, n_kv_heads, head_dim]
    position: int
    
    @classmethod
    def zeros(
        cls,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> "KVCache":
        """Create empty cache."""
        return cls(
            keys=jnp.zeros((batch_size, max_seq_len, n_kv_heads, head_dim), dtype=dtype),
            values=jnp.zeros((batch_size, max_seq_len, n_kv_heads, head_dim), dtype=dtype),
            position=0,
        )
    
    def update(self, keys: Array, values: Array) -> "KVCache":
        """Append new keys/values to cache.
        
        Args:
            keys: New keys of shape [batch, seq_len, n_kv_heads, head_dim]
            values: New values of shape [batch, seq_len, n_kv_heads, head_dim]
            
        Returns:
            Updated cache with new position.
        """
        seq_len = keys.shape[1]
        new_keys = jax.lax.dynamic_update_slice(
            self.keys, keys, (0, self.position, 0, 0)
        )
        new_values = jax.lax.dynamic_update_slice(
            self.values, values, (0, self.position, 0, 0)
        )
        return KVCache(
            keys=new_keys,
            values=new_values,
            position=self.position + seq_len,
        )
    
    def get(self) -> Tuple[Array, Array]:
        """Get cached keys/values up to current position.
        
        Returns:
            Tuple of (keys, values) sliced to [batch, position, n_kv_heads, head_dim]
        """
        return (
            jax.lax.dynamic_slice(self.keys, (0, 0, 0, 0), 
                                   (self.keys.shape[0], self.position, self.keys.shape[2], self.keys.shape[3])),
            jax.lax.dynamic_slice(self.values, (0, 0, 0, 0),
                                   (self.values.shape[0], self.position, self.values.shape[2], self.values.shape[3])),
        )


# -----------------------------------------------------------------------------
# Causal Self-Attention
# -----------------------------------------------------------------------------

class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention with GQA/MQA support.
    
    Implements scaled dot-product attention with causal masking:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) @ V
    
    Supports:
    - Multi-Head Attention (MHA): n_heads == n_kv_heads
    - Grouped-Query Attention (GQA): n_heads > n_kv_heads, n_heads % n_kv_heads == 0
    - Multi-Query Attention (MQA): n_kv_heads == 1
    
    Args:
        hidden_size: Model dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (for GQA/MQA).
        head_dim: Dimension per head (default: hidden_size // n_heads).
        bias: Whether to use bias in projections.
        dropout: Dropout probability (applied during training).
        use_rope: Whether to apply rotary position embeddings.
        max_seq_len: Maximum sequence length (for RoPE precomputation).
        rope_base: Base frequency for RoPE.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        *,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or (hidden_size // n_heads)
        self.dropout = dropout
        self.use_rope = use_rope
        
        # Validate GQA configuration
        if n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        self.n_rep = n_heads // self.n_kv_heads  # Repetition factor for GQA
        
        # Projections
        self.q_proj = nnx.Linear(hidden_size, n_heads * self.head_dim, use_bias=bias, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, self.n_kv_heads * self.head_dim, use_bias=bias, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, self.n_kv_heads * self.head_dim, use_bias=bias, rngs=rngs)
        self.o_proj = nnx.Linear(n_heads * self.head_dim, hidden_size, use_bias=bias, rngs=rngs)
        
        # RoPE embeddings
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len, base=rope_base, rngs=rngs)
        else:
            self.rope = None
        
        # For dropout during training
        self.rngs = rngs
    
    def __call__(
        self,
        x: Array,
        *,
        state: Optional[KVCache] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, Optional[KVCache]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
            state: Optional KV cache for incremental decoding.
            mask: Optional attention mask [batch, seq_len] or [batch, 1, seq_len, seq_len].
            mode: Ignored (for API compatibility with other block types).
            
        Returns:
            Tuple of (output, new_state) where output has same shape as x.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq, n_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq, n_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, n_kv_heads * head_dim]
        
        # Reshape to [batch, seq, n_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Determine position offset from cache
        offset = state.position if state is not None else 0
        
        # Apply RoPE
        if self.rope is not None:
            cos, sin = self.rope(seq_len, offset=offset)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        # Update KV cache
        if state is not None:
            state = state.update(k, v)
            k_full, v_full = state.get()  # [batch, total_len, n_kv_heads, head_dim]
        else:
            k_full, v_full = k, v
        
        # Repeat KV heads for GQA
        if self.n_rep > 1:
            # [batch, seq, n_kv_heads, head_dim] -> [batch, seq, n_heads, head_dim]
            k_full = jnp.repeat(k_full, self.n_rep, axis=2)
            v_full = jnp.repeat(v_full, self.n_rep, axis=2)
        
        # Compute attention scores
        # Transpose to [batch, n_heads, seq_q, head_dim] for batched matmul
        q = q.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_q, head_dim]
        k_full = k_full.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_kv, head_dim]
        v_full = v_full.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_kv, head_dim]
        
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_full) * scale  # [batch, n_heads, seq_q, seq_kv]
        
        # Apply causal mask
        kv_len = k_full.shape[2]
        causal_mask = jnp.tril(jnp.ones((seq_len, kv_len), dtype=jnp.bool_))
        # Shift causal mask for cached positions
        if offset > 0:
            # When generating, query position i can attend to all cached + current
            causal_mask = jnp.ones((seq_len, kv_len), dtype=jnp.bool_)
            causal_mask = causal_mask.at[:, offset + seq_len:].set(False)
            # Make sure we still respect causality within the new tokens
            for i in range(seq_len):
                causal_mask = causal_mask.at[i, offset + i + 1:].set(False)
        
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)
        
        # Apply optional padding mask
        if mask is not None:
            if mask.ndim == 2:
                # [batch, seq] -> [batch, 1, 1, seq]
                mask = mask[:, None, None, :]
            scores = jnp.where(mask, scores, -1e9)
        
        # Softmax and apply to values
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Optional dropout (only during training)
        # Note: In NNx, we'd check training mode; simplified here
        if self.dropout > 0.0:
            # Would apply dropout here in training mode
            pass
        
        # Apply attention to values
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v_full)
        
        # Reshape back: [batch, n_heads, seq, head_dim] -> [batch, seq, hidden]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, state
    
    def init_state(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> KVCache:
        """Initialize empty KV cache for generation."""
        return KVCache.zeros(batch_size, max_seq_len, self.n_kv_heads, self.head_dim, dtype)


# -----------------------------------------------------------------------------
# Attention Block (Attention + FFN with residuals)
# -----------------------------------------------------------------------------

class AttentionBlock(nnx.Module):
    """Transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual.
    
    This is the standard pre-norm transformer block used in LLaMA, GPT-3, etc.
    
    Args:
        hidden_size: Model dimension.
        n_heads: Number of attention heads.
        intermediate_size: FFN hidden dimension.
        n_kv_heads: Number of KV heads for GQA/MQA.
        head_dim: Per-head dimension.
        bias: Whether to use bias.
        dropout: Dropout probability.
        use_rope: Whether to use RoPE.
        max_seq_len: Maximum sequence length.
        rope_base: RoPE base frequency.
        mlp_activation: MLP activation function.
        norm_type: Normalization type ("rmsnorm" or "layernorm").
        norm_eps: Normalization epsilon.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        intermediate_size: int,
        *,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        mlp_activation: str = "silu",
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        norm_eps: float = 1e-6,
        rngs: nnx.Rngs,
    ):
        # Normalization layers
        self.attn_norm = get_norm(norm_type, hidden_size, eps=norm_eps, rngs=rngs)
        self.ffn_norm = get_norm(norm_type, hidden_size, eps=norm_eps, rngs=rngs)
        
        # Attention
        self.attn = CausalSelfAttention(
            hidden_size,
            n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bias=bias,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rngs=rngs,
        )
        
        # FFN
        self.ffn = MLP(
            hidden_size,
            intermediate_size,
            activation=mlp_activation,
            use_gating=True,
            bias=bias,
            rngs=rngs,
        )
    
    def __call__(
        self,
        x: Array,
        *,
        state: Optional[KVCache] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, Optional[KVCache]]:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            state: Optional KV cache for generation.
            mask: Optional attention mask.
            mode: Ignored (for API compatibility).
            
        Returns:
            Tuple of (output, new_state).
        """
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        x, state = self.attn(x, state=state, mask=mask, mode=mode)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, state
    
    def init_state(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> KVCache:
        """Initialize attention cache."""
        return self.attn.init_state(batch_size, max_seq_len, dtype)
