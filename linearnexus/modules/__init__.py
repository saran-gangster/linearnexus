"""Neural network building blocks for LinearNexus.

Architecture families:
- attention/: Dense causal attention (GPT-style)
- sparse/: Sparse attention patterns (sliding window, block-sparse) [Phase 2]
- ssm/: State-space models (Mamba, RWKV, S4)
- linear_attn/: Linear attention variants (DeltaNet, GLA, RetNet)
- hybrid/: Mixed architecture blocks (Jamba-style interleaved) [Phase 3]

All blocks implement a unified interface:
    __call__(x, *, state=None, mask=None, mode=None) -> (output, new_state)
"""

from .common import (
    MLP,
    RMSNorm,
    LayerNorm,
    Embedding,
    RotaryEmbedding,
    apply_rotary_emb,
    get_norm,
)

from .attention import AttentionBlock, CausalSelfAttention, KVCache
from .attention import MultiHeadLatentAttention, MLABlock, MLACache
from .ssm import MambaBlock, MambaState
from .linear_attn import DeltaNetBlock, DeltaNetState

__all__ = [
    # Common
    "MLP",
    "RMSNorm",
    "LayerNorm", 
    "Embedding",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "get_norm",
    
    # Attention
    "AttentionBlock",
    "CausalSelfAttention",
    "KVCache",
    
    # Multi-Head Latent Attention
    "MultiHeadLatentAttention",
    "MLABlock",
    "MLACache",
    
    # SSM
    "MambaBlock",
    "MambaState",
    
    # Linear Attention
    "DeltaNetBlock",
    "DeltaNetState",
]
