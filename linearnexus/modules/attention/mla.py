"""Multi-Head Latent Attention (MLA) implementation.

MLA is a memory-efficient attention variant from DeepSeek-V2 that projects
Q, K, V to low-rank latent spaces using LoRA-style compression, dramatically
reducing KV cache memory during inference.

Paper: https://arxiv.org/abs/2405.04434

Key ideas:
1. Compress Q/K/V into low-rank representations before attention
2. Split query/key into RoPE and non-RoPE (nope) components
3. Share a single low-rank KV projection across heads

This enables:
- Reduced KV cache from O(n_heads * head_dim) to O(kv_lora_rank)
- Lower memory bandwidth during autoregressive generation
- Competitive quality with standard MHA/GQA at much lower memory cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.modules.common import RMSNorm, MLP, RotaryEmbedding, get_norm

Array = jax.Array


# -----------------------------------------------------------------------------
# MLA Cache for Autoregressive Generation
# -----------------------------------------------------------------------------

@dataclass
class MLACache:
    """Cache for efficient MLA autoregressive decoding.
    
    Unlike standard KV cache, MLA caches the compressed latent representations
    and the RoPE key component, which are much smaller than full K/V tensors.
    
    Attributes:
        compressed_kv: Cached compressed KV of shape [batch, max_seq_len, kv_lora_rank]
        key_rope: Cached RoPE key of shape [batch, max_seq_len, 1, rope_head_dim]
        position: Current position in the cache (number of tokens cached)
    """
    compressed_kv: Array   # [batch, max_seq_len, kv_lora_rank]
    key_rope: Array        # [batch, max_seq_len, 1, rope_head_dim]
    position: int
    
    @classmethod
    def zeros(
        cls,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        rope_head_dim: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> "MLACache":
        """Create empty MLA cache."""
        return cls(
            compressed_kv=jnp.zeros((batch_size, max_seq_len, kv_lora_rank), dtype=dtype),
            key_rope=jnp.zeros((batch_size, max_seq_len, 1, rope_head_dim), dtype=dtype),
            position=0,
        )
    
    def update(
        self,
        compressed_kv: Array,
        key_rope: Array,
    ) -> "MLACache":
        """Append new compressed KV and RoPE key to cache.
        
        Args:
            compressed_kv: New compressed KV of shape [batch, seq_len, kv_lora_rank]
            key_rope: New RoPE key of shape [batch, seq_len, 1, rope_head_dim]
            
        Returns:
            Updated cache with new position.
        """
        seq_len = compressed_kv.shape[1]
        new_compressed_kv = jax.lax.dynamic_update_slice(
            self.compressed_kv, compressed_kv, (0, self.position, 0)
        )
        new_key_rope = jax.lax.dynamic_update_slice(
            self.key_rope, key_rope, (0, self.position, 0, 0)
        )
        return MLACache(
            compressed_kv=new_compressed_kv,
            key_rope=new_key_rope,
            position=self.position + seq_len,
        )
    
    def get(self) -> Tuple[Array, Array]:
        """Get cached compressed KV and RoPE key up to current position.
        
        Returns:
            Tuple of (compressed_kv, key_rope) sliced to current position.
        """
        return (
            jax.lax.dynamic_slice(
                self.compressed_kv, 
                (0, 0, 0),
                (self.compressed_kv.shape[0], self.position, self.compressed_kv.shape[2])
            ),
            jax.lax.dynamic_slice(
                self.key_rope,
                (0, 0, 0, 0),
                (self.key_rope.shape[0], self.position, self.key_rope.shape[2], self.key_rope.shape[3])
            ),
        )


# -----------------------------------------------------------------------------
# RoPE Helpers for MLA
# -----------------------------------------------------------------------------

def apply_rope_single(
    x: Array,
    cos: Array,
    sin: Array,
) -> Array:
    """Apply rotary embeddings to a single tensor.
    
    Args:
        x: Input tensor of shape [batch, heads, seq, rope_head_dim]
        cos: Cosine embeddings of shape [seq, rope_head_dim/2]
        sin: Sine embeddings of shape [seq, rope_head_dim/2]
        
    Returns:
        Rotated tensor of same shape as x.
    """
    # x: [batch, heads, seq, dim]
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    
    # Reshape cos/sin for broadcasting: [1, 1, seq, dim/2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    
    # Apply rotation
    rotated = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1)
    
    return rotated.reshape(x.shape)


def apply_rope_mla(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
) -> Tuple[Array, Array]:
    """Apply rotary embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch, n_heads, seq_q, rope_head_dim]
        k: Key tensor of shape [batch, 1, seq_k, rope_head_dim]  (shared across heads)
        cos: Cosine embeddings of shape [seq, rope_head_dim/2] - should match q's seq length
        sin: Sine embeddings of shape [seq, rope_head_dim/2]
        
    Returns:
        Tuple of rotated (query, key).
    """
    q_rotated = apply_rope_single(q, cos, sin)
    k_rotated = apply_rope_single(k, cos, sin)
    
    return q_rotated, k_rotated


# -----------------------------------------------------------------------------
# Multi-Head Latent Attention
# -----------------------------------------------------------------------------

class MultiHeadLatentAttention(nnx.Module):
    """Multi-Head Latent Attention (MLA) from DeepSeek-V2.
    
    MLA reduces memory by projecting Q, K, V into low-rank latent spaces.
    The key innovation is splitting attention dimensions into:
    - nope (non-positional): Regular attention without position encoding
    - rope: Rotary position embedding component
    
    This allows aggressive compression of the KV cache while maintaining
    position-aware attention.
    
    Args:
        hidden_size: Model hidden dimension (d_model).
        n_heads: Number of attention heads.
        v_head_dim: Value head dimension.
        nope_head_dim: Non-positional (no RoPE) head dimension for Q/K.
        rope_head_dim: RoPE head dimension for Q/K.
        q_lora_rank: LoRA rank for query compression.
        kv_lora_rank: LoRA rank for KV compression.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        rope_base: Base frequency for RoPE.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        *,
        v_head_dim: int,
        nope_head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_base: float = 10000.0,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.v_head_dim = v_head_dim
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.dropout = dropout
        
        # Derived dimensions
        # value_dim: total dimension for values across all heads
        self.value_dim = n_heads * v_head_dim
        # nope_dim: total non-positional dimension for Q/K
        self.nope_dim = n_heads * nope_head_dim
        # rope_dim: total RoPE dimension for Q
        self.rope_dim = n_heads * rope_head_dim
        
        # Query compression: x -> compressed -> (nope, rope)
        self.compress_q = nnx.Linear(hidden_size, q_lora_rank, use_bias=False, rngs=rngs)
        self.q_norm = RMSNorm(q_lora_rank, rngs=rngs)
        self.decompress_q_nope = nnx.Linear(q_lora_rank, self.nope_dim, use_bias=False, rngs=rngs)
        self.decompress_q_rope = nnx.Linear(q_lora_rank, self.rope_dim, use_bias=False, rngs=rngs)
        
        # KV compression: x -> compressed -> (k_nope, v)
        # Note: K uses the same nope_dim as Q for dot product compatibility
        self.compress_kv = nnx.Linear(hidden_size, kv_lora_rank, use_bias=False, rngs=rngs)
        self.kv_norm = RMSNorm(kv_lora_rank, rngs=rngs)
        self.decompress_k_nope = nnx.Linear(kv_lora_rank, self.nope_dim, use_bias=False, rngs=rngs)
        self.decompress_v = nnx.Linear(kv_lora_rank, self.value_dim, use_bias=False, rngs=rngs)
        
        # Separate K RoPE projection (shared across heads, from raw input)
        self.k_rope_proj = nnx.Linear(hidden_size, rope_head_dim, use_bias=False, rngs=rngs)
        
        # Output projection
        self.o_proj = nnx.Linear(self.value_dim, hidden_size, use_bias=False, rngs=rngs)
        
        # RoPE embeddings
        self.rope = RotaryEmbedding(rope_head_dim, max_seq_len, base=rope_base, rngs=rngs)
        
        self.rngs = rngs
    
    def __call__(
        self,
        x: Array,
        *,
        state: Optional[MLACache] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, Optional[MLACache]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
            state: Optional MLA cache for incremental decoding.
            mask: Optional attention mask [batch, seq_len].
            mode: Ignored (for API compatibility).
            
        Returns:
            Tuple of (output, new_state) where output has same shape as x.
        """
        batch_size, seq_len, _ = x.shape
        
        # Determine position offset from cache
        offset = state.position if state is not None else 0
        
        # === Query path ===
        # Compress query: [batch, seq, hidden] -> [batch, seq, q_lora_rank]
        compressed_q = self.compress_q(x)
        norm_q = self.q_norm(compressed_q)
        
        # Decompress to nope and rope components
        # [batch, seq, nope_dim], [batch, seq, rope_dim]
        q_nope = self.decompress_q_nope(norm_q)
        q_rope = self.decompress_q_rope(norm_q)
        
        # === Key-Value path ===
        # Compress KV: [batch, seq, hidden] -> [batch, seq, kv_lora_rank]
        compressed_kv = self.compress_kv(x)
        norm_kv = self.kv_norm(compressed_kv)
        
        # Decompress to k_nope and value
        k_nope = self.decompress_k_nope(norm_kv)  # [batch, seq, nope_dim]
        value = self.decompress_v(norm_kv)         # [batch, seq, value_dim]
        
        # K RoPE from raw input (shared single head)
        k_rope = self.k_rope_proj(x)  # [batch, seq, rope_head_dim]
        
        # === Reshape for multi-head attention ===
        # Q: [batch, seq, n_heads, *_head_dim]
        q_nope = q_nope.reshape(batch_size, seq_len, self.n_heads, self.nope_head_dim)
        q_rope = q_rope.reshape(batch_size, seq_len, self.n_heads, self.rope_head_dim)
        
        # K: k_nope per head, k_rope shared (single head)
        k_nope = k_nope.reshape(batch_size, seq_len, self.n_heads, self.nope_head_dim)
        k_rope = k_rope.reshape(batch_size, seq_len, 1, self.rope_head_dim)
        
        # V: [batch, seq, n_heads, v_head_dim]
        value = value.reshape(batch_size, seq_len, self.n_heads, self.v_head_dim)
        
        # === Update cache if provided ===
        if state is not None:
            state = state.update(compressed_kv, k_rope)
            # Get full cached compressed_kv and k_rope
            cached_compressed_kv, cached_k_rope = state.get()
            
            # Decompress cached KV
            cached_norm_kv = self.kv_norm(cached_compressed_kv)
            k_nope_full = self.decompress_k_nope(cached_norm_kv)
            value_full = self.decompress_v(cached_norm_kv)
            
            # Reshape cached tensors
            kv_len = cached_compressed_kv.shape[1]
            k_nope_full = k_nope_full.reshape(batch_size, kv_len, self.n_heads, self.nope_head_dim)
            k_rope_full = cached_k_rope  # Already [batch, kv_len, 1, rope_head_dim]
            value_full = value_full.reshape(batch_size, kv_len, self.n_heads, self.v_head_dim)
        else:
            k_nope_full = k_nope
            k_rope_full = k_rope
            value_full = value
            kv_len = seq_len
        
        # === Transpose for attention: [batch, heads, seq, dim] ===
        q_nope = q_nope.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_q, nope_head_dim]
        q_rope = q_rope.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_q, rope_head_dim]
        
        k_nope_full = k_nope_full.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_kv, nope_head_dim]
        k_rope_full = k_rope_full.transpose(0, 2, 1, 3)  # [batch, 1, seq_kv, rope_head_dim]
        value_full = value_full.transpose(0, 2, 1, 3)    # [batch, n_heads, seq_kv, v_head_dim]
        
        # === Apply RoPE ===
        # For query: apply RoPE with positions [offset, offset + seq_len)
        cos_q, sin_q = self.rope(seq_len, offset=offset)
        q_rope = apply_rope_single(q_rope, cos_q, sin_q)
        
        # For key: apply RoPE with positions [0, kv_len) 
        cos_k, sin_k = self.rope(kv_len, offset=0)
        k_rope_full = apply_rope_single(k_rope_full, cos_k, sin_k)
        
        # === Scale k_rope for head sharing (critical fix from reference) ===
        k_rope_full = k_rope_full / self.n_heads
        
        # === Combine nope and rope components ===
        # Q: [batch, n_heads, seq_q, nope_head_dim + rope_head_dim]
        # K: [batch, n_heads, seq_kv, nope_head_dim + rope_head_dim]
        full_head_dim = self.nope_head_dim + self.rope_head_dim
        
        query = jnp.concatenate([q_nope, q_rope], axis=-1)  # [batch, n_heads, seq_q, full_head_dim]
        
        # k_rope broadcasts across all heads
        key = jnp.concatenate([k_nope_full, jnp.broadcast_to(k_rope_full, k_nope_full.shape[:-1] + (self.rope_head_dim,))], axis=-1)
        
        # === Scaled dot-product attention ===
        scale = 1.0 / jnp.sqrt(full_head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) * scale  # [batch, n_heads, seq_q, seq_kv]
        
        # === Apply causal mask ===
        causal_mask = jnp.tril(jnp.ones((seq_len, kv_len), dtype=jnp.bool_))
        if offset > 0:
            # When generating, query position i can attend to all cached + current
            causal_mask = jnp.ones((seq_len, kv_len), dtype=jnp.bool_)
            causal_mask = causal_mask.at[:, offset + seq_len:].set(False)
            for i in range(seq_len):
                causal_mask = causal_mask.at[i, offset + i + 1:].set(False)
        
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)
        
        # Apply optional padding mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            scores = jnp.where(mask, scores, -1e9)
        
        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value_full)
        
        # Reshape back: [batch, n_heads, seq, v_head_dim] -> [batch, seq, value_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.value_dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, state
    
    def init_state(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> MLACache:
        """Initialize empty MLA cache for generation."""
        return MLACache.zeros(
            batch_size,
            max_seq_len,
            self.kv_lora_rank,
            self.rope_head_dim,
            dtype,
        )


# -----------------------------------------------------------------------------
# MLA Block (Attention + FFN with residuals)
# -----------------------------------------------------------------------------

class MLABlock(nnx.Module):
    """Transformer block with Multi-Head Latent Attention.
    
    Pre-norm architecture: LayerNorm -> MLA -> Residual -> LayerNorm -> FFN -> Residual.
    
    Args:
        hidden_size: Model hidden dimension.
        n_heads: Number of attention heads.
        intermediate_size: FFN hidden dimension.
        v_head_dim: Value head dimension.
        nope_head_dim: Non-positional head dimension.
        rope_head_dim: RoPE head dimension.
        q_lora_rank: Query compression rank.
        kv_lora_rank: KV compression rank.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
        rope_base: RoPE base frequency.
        mlp_activation: MLP activation function.
        norm_type: Normalization type.
        norm_eps: Normalization epsilon.
        rngs: Random number generators.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        intermediate_size: int,
        *,
        v_head_dim: int,
        nope_head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        dropout: float = 0.0,
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
        
        # Multi-Head Latent Attention
        self.attn = MultiHeadLatentAttention(
            hidden_size,
            n_heads,
            v_head_dim=v_head_dim,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            dropout=dropout,
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
            bias=False,
            rngs=rngs,
        )
    
    def __call__(
        self,
        x: Array,
        *,
        state: Optional[MLACache] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, Optional[MLACache]]:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor [batch, seq, hidden_size]
            state: Optional MLA cache for generation.
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
    ) -> MLACache:
        """Initialize MLA cache."""
        return self.attn.init_state(batch_size, max_seq_len, dtype)
