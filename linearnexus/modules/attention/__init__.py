"""Attention-based blocks (dense causal attention).

Implements GPT-style multi-head causal self-attention with support for:
- Multi-Head Attention (MHA)
- Grouped-Query Attention (GQA)
- Multi-Query Attention (MQA)
- Multi-Head Latent Attention (MLA) - memory-efficient variant from DeepSeek-V2
- Rotary position embeddings (RoPE)
- KV-cache for efficient autoregressive generation
"""

from .causal import CausalSelfAttention, AttentionBlock, KVCache
from .mla import MultiHeadLatentAttention, MLABlock, MLACache

__all__ = [
    # Standard attention
    "CausalSelfAttention",
    "AttentionBlock",
    "KVCache",
    # Multi-Head Latent Attention
    "MultiHeadLatentAttention",
    "MLABlock",
    "MLACache",
]
