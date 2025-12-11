"""Language model architectures with flexible block composition.

Supports multiple architecture families through a unified interface:
- GPT: Dense causal attention (LLaMA-style)
- Mamba: Selective state-space model
- Hybrid: Interleaved patterns (Jamba-style)

The key abstraction is `block_pattern`: a list of block types that defines
the model architecture. This enables:
- Pure architectures: ["attention"] * 12 (GPT) or ["mamba"] * 24 (Mamba)
- Hybrid patterns: ["mamba", "mamba", "attention"] * 8 (every 3rd is attention)
- Jamba-style: ["mamba"] * 7 + ["attention"] (every 8th is attention)

Example:
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        n_layers=12,
        block_pattern=["attention"],  # Pure GPT
    )
    model = LMModel(config, rngs=nnx.Rngs(0))
    logits, state = model(tokens)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from linearnexus.core import ConfigBase
from linearnexus.modules.common import Embedding, RMSNorm, get_norm
from linearnexus.modules.attention import AttentionBlock, KVCache
from linearnexus.modules.ssm import MambaBlock, MambaState, Mamba2Block, Mamba2State
from linearnexus.modules.linear_attn import (
    DeltaNetBlock, DeltaNetState,
    GatedDeltaNetBlock, GatedDeltaNetState,
    RWKV6Block, RWKV6State,
    RWKV7Block, RWKV7State,
    KDABlock, KDAState,
)

Array = jax.Array

# Type alias for block state (varies by block type)
BlockState = Union[KVCache, MambaState, Mamba2State, DeltaNetState, GatedDeltaNetState, RWKV6State, RWKV7State, KDAState, None]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig(ConfigBase):
    """Unified configuration for all model architectures.
    
    Core Parameters:
        vocab_size: Vocabulary size for embeddings.
        hidden_size: Model hidden dimension.
        n_layers: Number of transformer/SSM blocks.
        block_pattern: List of block types defining architecture.
            - ["attention"]: GPT-style dense attention
            - ["mamba"]: Mamba SSM blocks
            - ["deltanet"]: DeltaNet linear attention blocks
            - ["attention", "mamba"]: Alternating hybrid
            Pattern is repeated to fill n_layers.
    
    Attention Parameters (when using attention blocks):
        n_heads: Number of attention heads.
        n_kv_heads: Number of KV heads (for GQA/MQA). None = MHA.
        head_dim: Per-head dimension. None = hidden_size // n_heads.
        max_seq_len: Maximum sequence length (for RoPE, KV cache).
        rope_base: RoPE base frequency.
    
    SSM Parameters (when using Mamba blocks):
        state_size: SSM state dimension.
        conv_kernel: Causal convolution kernel size.
        time_step_rank: Rank of time-step projection.
    
    DeltaNet Parameters (when using DeltaNet blocks):
        deltanet_heads: Number of attention heads for DeltaNet.
        deltanet_expand_k: Key expansion ratio.
        deltanet_expand_v: Value expansion ratio.
        deltanet_use_beta: Whether to use learnable beta.
        deltanet_use_gate: Whether to use output gating.
        deltanet_use_short_conv: Whether to use short convolutions.
        deltanet_qk_activation: Activation for Q/K ("silu", "relu", etc.).
        deltanet_qk_norm: Normalization for Q/K ("l2" or "sum").
        deltanet_chunk_size: Chunk size for chunkwise algorithm.
    
    Shared Parameters:
        intermediate_size: FFN/SSM intermediate dimension.
        activation: Activation function name.
        use_bias: Whether to use bias in projections.
        dropout: Dropout probability.
        norm_type: Normalization type ("rmsnorm" or "layernorm").
        norm_eps: Normalization epsilon.
        tie_embeddings: Whether to tie input/output embeddings.
    """
    # Core
    vocab_size: int = 32000
    hidden_size: int = 768
    n_layers: int = 12
    block_pattern: List[str] = field(default_factory=lambda: ["attention"])
    
    # Attention
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    
    # SSM (Mamba1)
    state_size: int = 16
    conv_kernel: int = 4
    time_step_rank: Optional[int] = None
    
    # SSM (Mamba2)
    mamba2_heads: Optional[int] = None  # Number of SSM heads for Mamba2
    mamba2_head_dim: int = 64           # Head dimension for Mamba2
    mamba2_state_size: int = 128        # State size for Mamba2 (larger than Mamba1)
    mamba2_n_groups: int = 1            # Number of groups for B/C
    mamba2_chunk_size: int = 256        # Chunk size for SSD
    
    # DeltaNet (Linear Attention)
    deltanet_heads: Optional[int] = None       # Number of heads (default: n_heads)
    deltanet_expand_k: float = 1.0             # Key expansion ratio
    deltanet_expand_v: float = 1.0             # Value expansion ratio
    deltanet_use_beta: bool = True             # Learnable beta (learning rate)
    deltanet_use_gate: bool = False            # Output gating
    deltanet_use_short_conv: bool = True       # Short convolutions on Q,K,V
    deltanet_qk_activation: str = "silu"       # Activation for Q/K
    deltanet_qk_norm: str = "l2"               # Normalization for Q/K
    deltanet_chunk_size: int = 64              # Chunk size for chunkwise algorithm
    
    # Gated DeltaNet (Mamba2 gating + Delta rule)
    gated_deltanet_heads: Optional[int] = None         # Number of Q/K heads
    gated_deltanet_v_heads: Optional[int] = None       # Number of V heads (GVA if > heads)
    gated_deltanet_head_dim: int = 256                 # Head dimension (default Mamba2-style)
    gated_deltanet_expand_v: float = 2.0               # Value expansion factor
    gated_deltanet_use_short_conv: bool = True         # Short convolutions on Q,K,V
    gated_deltanet_use_gate: bool = True               # Output gating
    gated_deltanet_allow_neg_eigval: bool = False      # If True, beta in [0,2] instead of [0,1]
    gated_deltanet_use_qk_l2norm: bool = True          # L2 normalize Q and K
    
    # RWKV6 (Matrix-valued states with data-dependent decay)
    rwkv6_heads: Optional[int] = None              # Number of heads (default: n_heads)
    rwkv6_proj_low_rank_dim: int = 32              # Low-rank dim for time_maa projections
    rwkv6_gate_low_rank_dim: int = 64              # Low-rank dim for decay projection
    rwkv6_intermediate_size: Optional[int] = None  # FFN intermediate size (default: 4x hidden)
    
    # RWKV7 (DPLR - Diagonal Plus Low Rank transition)
    rwkv7_heads: Optional[int] = None              # Number of heads (default: n_heads)
    rwkv7_head_dim: int = 64                       # Head dimension
    rwkv7_value_dim: Optional[int] = None          # Value dimension (default: hidden_size)
    rwkv7_decay_low_rank_dim: Optional[int] = None # Low-rank dim for w (auto-computed if None)
    rwkv7_gate_low_rank_dim: Optional[int] = None  # Low-rank dim for g (auto-computed if None)
    rwkv7_a_low_rank_dim: Optional[int] = None     # Low-rank dim for a (auto-computed if None)
    rwkv7_v_low_rank_dim: Optional[int] = None     # Low-rank dim for v interpolation (auto-computed if None)
    
    # KDA (Kimi Delta Attention - per-dimension gating)
    kda_heads: Optional[int] = None               # Number of Q/K heads (default: hidden_size // head_dim)
    kda_v_heads: Optional[int] = None             # Number of V heads (GVA if > heads)
    kda_head_dim: int = 128                       # Head dimension (default: 128)
    kda_expand_v: float = 1.0                     # Value expansion factor
    kda_use_short_conv: bool = True               # Short convolutions on Q,K,V
    kda_allow_neg_eigval: bool = False            # If True, beta in [0,2] instead of [0,1]
    
    # Shared
    intermediate_size: Optional[int] = None
    activation: str = "silu"
    use_bias: bool = False
    dropout: float = 0.0
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    
    def __post_init__(self):
        """Compute derived parameters."""
        # Default intermediate size: 4x hidden for attention, 2x for Mamba/DeltaNet
        if self.intermediate_size is None:
            if "mamba" in self.block_pattern or "deltanet" in self.block_pattern or "gated_deltanet" in self.block_pattern:
                self.intermediate_size = self.hidden_size * 2
            else:
                self.intermediate_size = self.hidden_size * 4
        
        # Default head dim
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.n_heads
        
        # Default time step rank
        if self.time_step_rank is None:
            self.time_step_rank = self.hidden_size
        
        # Default DeltaNet heads
        if self.deltanet_heads is None:
            self.deltanet_heads = self.n_heads
        
        # Default Gated DeltaNet heads
        if self.gated_deltanet_heads is None:
            # Compute based on hidden_size / head_dim (like Mamba2)
            self.gated_deltanet_heads = max(1, self.hidden_size // self.gated_deltanet_head_dim)
        if self.gated_deltanet_v_heads is None:
            self.gated_deltanet_v_heads = self.gated_deltanet_heads
        
        # Default RWKV6 heads
        if self.rwkv6_heads is None:
            self.rwkv6_heads = self.n_heads
        if self.rwkv6_intermediate_size is None:
            self.rwkv6_intermediate_size = self.hidden_size * 4
        
        # Default RWKV7 heads
        if self.rwkv7_heads is None:
            self.rwkv7_heads = max(1, self.hidden_size // self.rwkv7_head_dim)
        if self.rwkv7_value_dim is None:
            self.rwkv7_value_dim = self.hidden_size
        
        # Default KDA heads
        if self.kda_heads is None:
            self.kda_heads = max(1, self.hidden_size // self.kda_head_dim)
        if self.kda_v_heads is None:
            self.kda_v_heads = self.kda_heads
    
    def get_block_types(self) -> List[str]:
        """Expand block_pattern to n_layers blocks."""
        pattern_len = len(self.block_pattern)
        return [self.block_pattern[i % pattern_len] for i in range(self.n_layers)]


# Preset configurations for common architectures
GPT_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=12,
    n_heads=12,
    block_pattern=["attention"],
    intermediate_size=3072,
)

GPT_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=24,
    n_heads=16,
    block_pattern=["attention"],
    intermediate_size=4096,
)

MAMBA_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    block_pattern=["mamba"],
    state_size=16,
    conv_kernel=4,
)

MAMBA_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=48,
    block_pattern=["mamba"],
    state_size=16,
    conv_kernel=4,
)

# Jamba-style: every 8th layer is attention
JAMBA_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "attention"],
    state_size=16,
)

# Mamba2 presets
MAMBA2_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    block_pattern=["mamba2"],
    mamba2_heads=12,
    mamba2_head_dim=64,
    mamba2_state_size=128,
    mamba2_n_groups=1,
)

MAMBA2_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=48,
    block_pattern=["mamba2"],
    mamba2_heads=16,
    mamba2_head_dim=64,
    mamba2_state_size=128,
    mamba2_n_groups=1,
)

# Jamba2-style: Mamba2 with occasional attention
JAMBA2_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["mamba2", "mamba2", "mamba2", "mamba2", "mamba2", "mamba2", "mamba2", "attention"],
    mamba2_heads=12,
    mamba2_head_dim=64,
    mamba2_state_size=128,
)

# DeltaNet presets
DELTANET_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["deltanet"],
    deltanet_heads=12,
    deltanet_expand_k=1.0,
    deltanet_expand_v=1.0,
    deltanet_use_beta=True,
    deltanet_use_gate=False,
    deltanet_use_short_conv=True,
    deltanet_qk_activation="silu",
    deltanet_qk_norm="l2",
    deltanet_chunk_size=64,
)

DELTANET_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=48,
    n_heads=16,
    block_pattern=["deltanet"],
    deltanet_heads=16,
    deltanet_expand_k=1.0,
    deltanet_expand_v=1.0,
    deltanet_use_beta=True,
    deltanet_use_gate=False,
    deltanet_use_short_conv=True,
    deltanet_qk_activation="silu",
    deltanet_qk_norm="l2",
    deltanet_chunk_size=64,
)

# Hybrid DeltaNet + Attention
DELTA_HYBRID_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["deltanet", "deltanet", "deltanet", "attention"],
    deltanet_heads=12,
    deltanet_use_short_conv=True,
)

# Gated DeltaNet presets (Mamba2 gating + Delta rule)
GATED_DELTANET_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    block_pattern=["gated_deltanet"],
    gated_deltanet_heads=3,          # 768 / 256 = 3
    gated_deltanet_v_heads=3,
    gated_deltanet_head_dim=256,
    gated_deltanet_expand_v=2.0,
    gated_deltanet_use_short_conv=True,
    gated_deltanet_use_gate=True,
    gated_deltanet_allow_neg_eigval=False,
    gated_deltanet_use_qk_l2norm=True,
)

GATED_DELTANET_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=48,
    block_pattern=["gated_deltanet"],
    gated_deltanet_heads=4,          # 1024 / 256 = 4
    gated_deltanet_v_heads=4,
    gated_deltanet_head_dim=256,
    gated_deltanet_expand_v=2.0,
    gated_deltanet_use_short_conv=True,
    gated_deltanet_use_gate=True,
)

# Hybrid Gated DeltaNet + Attention (every 4th is attention)
GATED_DELTA_HYBRID_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["gated_deltanet", "gated_deltanet", "gated_deltanet", "attention"],
    gated_deltanet_heads=3,
    gated_deltanet_v_heads=3,
    gated_deltanet_head_dim=256,
    gated_deltanet_use_gate=True,
)

# RWKV6 presets (Matrix-valued states with data-dependent decay)
RWKV6_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["rwkv6"],
    rwkv6_heads=12,
    rwkv6_proj_low_rank_dim=32,
    rwkv6_gate_low_rank_dim=64,
    rwkv6_intermediate_size=3072,  # 4x hidden
)

RWKV6_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=48,
    n_heads=16,
    block_pattern=["rwkv6"],
    rwkv6_heads=16,
    rwkv6_proj_low_rank_dim=32,
    rwkv6_gate_low_rank_dim=64,
    rwkv6_intermediate_size=4096,  # 4x hidden
)

# Hybrid RWKV6 + Attention (every 4th is attention)
RWKV6_HYBRID_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["rwkv6", "rwkv6", "rwkv6", "attention"],
    rwkv6_heads=12,
    rwkv6_proj_low_rank_dim=32,
    rwkv6_gate_low_rank_dim=64,
)

# RWKV7 presets (DPLR - Diagonal Plus Low Rank transition)
RWKV7_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["rwkv7"],
    rwkv7_heads=12,
    rwkv7_head_dim=64,
)

RWKV7_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=1024,
    n_layers=48,
    n_heads=16,
    block_pattern=["rwkv7"],
    rwkv7_heads=16,
    rwkv7_head_dim=64,
)

# Hybrid RWKV7 + Attention (every 4th is attention)
RWKV7_HYBRID_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=768,
    n_layers=24,
    n_heads=12,
    block_pattern=["rwkv7", "rwkv7", "rwkv7", "attention"],
    rwkv7_heads=12,
    rwkv7_head_dim=64,
)

# KDA presets (Kimi Delta Attention - per-dimension gating)
KDA_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=2048,
    n_layers=24,
    n_heads=16,
    block_pattern=["kda"],
    kda_heads=16,
    kda_v_heads=16,
    kda_head_dim=128,
    kda_expand_v=1.0,
    kda_use_short_conv=True,
    kda_allow_neg_eigval=False,
)

KDA_MEDIUM = ModelConfig(
    vocab_size=50257,
    hidden_size=4096,
    n_layers=48,
    n_heads=32,
    block_pattern=["kda"],
    kda_heads=32,
    kda_v_heads=32,
    kda_head_dim=128,
    kda_expand_v=1.0,
    kda_use_short_conv=True,
)

# Hybrid KDA + Attention (every 4th is attention)
KDA_HYBRID_SMALL = ModelConfig(
    vocab_size=50257,
    hidden_size=2048,
    n_layers=24,
    n_heads=16,
    block_pattern=["kda", "kda", "kda", "attention"],
    kda_heads=16,
    kda_v_heads=16,
    kda_head_dim=128,
)


# -----------------------------------------------------------------------------
# Model State (for generation)
# -----------------------------------------------------------------------------

@dataclass
class ModelState:
    """Composite state for all layers during generation.
    
    Attributes:
        block_states: List of per-block states (KVCache or MambaState).
        position: Current sequence position.
    """
    block_states: List[BlockState]
    position: int = 0
    
    def update_position(self, seq_len: int) -> "ModelState":
        """Advance position after processing tokens."""
        return ModelState(
            block_states=self.block_states,
            position=self.position + seq_len,
        )


# -----------------------------------------------------------------------------
# Block Factory
# -----------------------------------------------------------------------------

def _create_block(
    block_type: str,
    config: ModelConfig,
    rngs: nnx.Rngs,
    layer_idx: int = 0,
) -> nnx.Module:
    """Create a single block based on type."""
    
    if block_type == "attention":
        return AttentionBlock(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            intermediate_size=config.intermediate_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            bias=config.use_bias,
            dropout=config.dropout,
            use_rope=True,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            mlp_activation=config.activation,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "mamba":
        return MambaBlock(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            state_size=config.state_size,
            conv_kernel=config.conv_kernel,
            time_step_rank=config.time_step_rank,
            use_conv_bias=True,
            use_bias=config.use_bias,
            activation=config.activation,
            chunk_size=64,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "mamba2":
        # Determine number of heads
        num_heads = config.mamba2_heads
        if num_heads is None:
            # Default: derive from hidden_size / head_dim
            num_heads = config.hidden_size // config.mamba2_head_dim
        
        return Mamba2Block(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            head_dim=config.mamba2_head_dim,
            state_size=config.mamba2_state_size,
            n_groups=config.mamba2_n_groups,
            conv_kernel=config.conv_kernel,
            use_conv_bias=False,  # Mamba2 typically doesn't use conv bias
            use_bias=config.use_bias,
            activation=config.activation,
            chunk_size=config.mamba2_chunk_size,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "deltanet":
        return DeltaNetBlock(
            hidden_size=config.hidden_size,
            num_heads=config.deltanet_heads or config.n_heads,
            expand_k=config.deltanet_expand_k,
            expand_v=config.deltanet_expand_v,
            use_beta=config.deltanet_use_beta,
            use_gate=config.deltanet_use_gate,
            use_short_conv=config.deltanet_use_short_conv,
            conv_size=config.conv_kernel,
            conv_bias=False,
            qk_activation=config.deltanet_qk_activation,
            qk_norm=config.deltanet_qk_norm,
            chunk_size=config.deltanet_chunk_size,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "gated_deltanet":
        return GatedDeltaNetBlock(
            hidden_size=config.hidden_size,
            num_heads=config.gated_deltanet_heads,
            num_v_heads=config.gated_deltanet_v_heads,
            head_dim=config.gated_deltanet_head_dim,
            expand_v=config.gated_deltanet_expand_v,
            use_short_conv=config.gated_deltanet_use_short_conv,
            conv_size=config.conv_kernel,
            use_gate=config.gated_deltanet_use_gate,
            allow_neg_eigval=config.gated_deltanet_allow_neg_eigval,
            use_qk_l2norm=config.gated_deltanet_use_qk_l2norm,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "rwkv6":
        return RWKV6Block(
            hidden_size=config.hidden_size,
            num_heads=config.rwkv6_heads or config.n_heads,
            intermediate_size=config.rwkv6_intermediate_size,
            proj_low_rank_dim=config.rwkv6_proj_low_rank_dim,
            gate_low_rank_dim=config.rwkv6_gate_low_rank_dim,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "rwkv7":
        return RWKV7Block(
            hidden_size=config.hidden_size,
            num_heads=config.rwkv7_heads,
            head_dim=config.rwkv7_head_dim,
            value_dim=config.rwkv7_value_dim,
            decay_low_rank_dim=config.rwkv7_decay_low_rank_dim,
            gate_low_rank_dim=config.rwkv7_gate_low_rank_dim,
            a_low_rank_dim=config.rwkv7_a_low_rank_dim,
            v_low_rank_dim=config.rwkv7_v_low_rank_dim,
            layer_idx=layer_idx,
            n_layers=config.n_layers,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    elif block_type == "kda":
        return KDABlock(
            hidden_size=config.hidden_size,
            num_heads=config.kda_heads,
            num_v_heads=config.kda_v_heads,
            head_dim=config.kda_head_dim,
            expand_v=config.kda_expand_v,
            use_short_conv=config.kda_use_short_conv,
            allow_neg_eigval=config.kda_allow_neg_eigval,
            conv_size=config.conv_kernel,
            norm_eps=config.norm_eps,
            rngs=rngs,
        )
    
    else:
        raise ValueError(
            f"Unknown block type '{block_type}'. "
            f"Supported: 'attention', 'mamba', 'mamba2', 'deltanet', 'gated_deltanet', 'rwkv6', 'rwkv7', 'kda'"
        )


# -----------------------------------------------------------------------------
# Language Model
# -----------------------------------------------------------------------------

class LMModel(nnx.Module):
    """Language model with flexible block composition.
    
    Supports GPT, Mamba, and hybrid architectures through block_pattern.
    
    Architecture:
        tokens -> Embedding -> [Blocks] -> Norm -> LM Head -> logits
    
    Args:
        config: Model configuration.
        rngs: Random number generators.
    """
    
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        self.config = config
        
        # Token embeddings
        self.embed = Embedding(config.vocab_size, config.hidden_size, rngs=rngs)
        
        # Create blocks based on pattern
        block_types = config.get_block_types()
        self.blocks = []
        self.block_types = block_types
        
        for i, block_type in enumerate(block_types):
            block = _create_block(block_type, config, rngs, layer_idx=i)
            self.blocks.append(block)
        
        # Final normalization
        self.norm = get_norm(config.norm_type, config.hidden_size, eps=config.norm_eps, rngs=rngs)
        
        # LM head (optionally tied to embeddings)
        if config.tie_embeddings:
            self.lm_head = None  # Will use embed.unembed()
        else:
            self.lm_head = nnx.Linear(
                config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs
            )
    
    def __call__(
        self,
        input_ids: Array,
        *,
        state: Optional[ModelState] = None,
        mask: Optional[Array] = None,
        mode: Optional[Literal["chunk", "recurrent"]] = None,
    ) -> Tuple[Array, ModelState]:
        """Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len].
            state: Optional model state for generation.
            mask: Optional attention/padding mask [batch, seq_len].
            mode: Processing mode ("chunk" for training, "recurrent" for generation).
            
        Returns:
            Tuple of (logits [batch, seq_len, vocab_size], new_state).
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size)
        
        # Embed tokens
        hidden = self.embed(input_ids)  # [batch, seq, hidden]
        
        # Process through blocks
        new_block_states = []
        v_first = None  # For RWKV7 cross-layer v interpolation
        
        for i, block in enumerate(self.blocks):
            block_state = state.block_states[i] if state.block_states else None
            block_type = self.block_types[i]
            
            # RWKV7 has special v_first handling
            if block_type == "rwkv7":
                hidden, new_state, v_first = block(
                    hidden, state=block_state, v_first=v_first, mask=mask, mode=mode
                )
            else:
                hidden, new_state = block(hidden, state=block_state, mask=mask, mode=mode)
            
            new_block_states.append(new_state)
        
        # Final norm
        hidden = self.norm(hidden)
        
        # Project to vocab
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = self.embed.unembed(hidden)
        
        # Update model state
        new_model_state = ModelState(
            block_states=new_block_states,
            position=state.position + seq_len,
        )
        
        return logits, new_model_state
    
    def init_state(
        self,
        batch_size: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> ModelState:
        """Initialize empty generation state."""
        block_states = []
        
        for i, block in enumerate(self.blocks):
            block_type = self.block_types[i]
            
            if block_type == "attention":
                # KV cache for attention
                kv_cache = block.init_state(batch_size, self.config.max_seq_len, dtype)
                block_states.append(kv_cache)
            
            elif block_type == "mamba":
                # SSM state for Mamba
                ssm_state = block.init_state(batch_size, dtype)
                block_states.append(ssm_state)
            
            elif block_type == "mamba2":
                # SSM state for Mamba2
                ssm_state = block.init_state(batch_size, dtype)
                block_states.append(ssm_state)
            
            elif block_type == "rwkv6":
                # RWKV6 state
                rwkv_state = block.init_state(batch_size)
                block_states.append(rwkv_state)
            
            elif block_type == "rwkv7":
                # RWKV7 state
                rwkv_state = block.init_state(batch_size)
                block_states.append(rwkv_state)
            
            elif block_type == "kda":
                # KDA state
                kda_state = block.init_state(batch_size)
                block_states.append(kda_state)
            
            elif block_type in ("deltanet", "gated_deltanet"):
                # DeltaNet/GatedDeltaNet state
                block_state = block.init_state(batch_size)
                block_states.append(block_state)
            
            else:
                block_states.append(None)
        
        return ModelState(block_states=block_states, position=0)
    
    def count_params(self) -> int:
        """Count total number of parameters."""
        def count_module(module):
            total = 0
            for name, value in vars(module).items():
                if isinstance(value, nnx.Param):
                    total += value.value.size
                elif isinstance(value, nnx.Module):
                    total += count_module(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, nnx.Module):
                            total += count_module(item)
            return total
        return count_module(self)


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

def create_model(
    model_type: Union[str, ModelConfig],
    *,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> Union[LMModel, Tuple[ModelConfig, type]]:
    """Create model from preset name or configuration.
    
    Args:
        model_type: One of "gpt-small", "gpt-medium", "mamba-small", 
                   "mamba-medium", "mamba2-small", "mamba2-medium",
                   "jamba-small", "jamba2-small", "deltanet-small",
                   "deltanet-medium", "delta-hybrid-small", "rwkv6-small",
                   "rwkv6-medium", "rwkv6-hybrid-small", "custom", or a ModelConfig.
        rngs: Optional Flax NNx random number generators. If provided,
              returns an instantiated model. Otherwise returns (config, class).
        **kwargs: Override config parameters (only for string model_type).
        
    Returns:
        If rngs is provided: instantiated LMModel.
        If rngs is None: tuple of (config, model_class).
    
    Example:
        # Get config and class
        config, Model = create_model("gpt-small", hidden_size=512)
        model = Model(config, rngs=nnx.Rngs(0))
        
        # Or directly instantiate
        model = create_model("gpt-small", rngs=nnx.Rngs(0))
        
        # Or use preset config
        model = create_model(GPT_SMALL, rngs=nnx.Rngs(0))
    """
    presets = {
        "gpt-small": GPT_SMALL,
        "gpt-medium": GPT_MEDIUM,
        "mamba-small": MAMBA_SMALL,
        "mamba-medium": MAMBA_MEDIUM,
        "jamba-small": JAMBA_SMALL,
        "mamba2-small": MAMBA2_SMALL,
        "mamba2-medium": MAMBA2_MEDIUM,
        "jamba2-small": JAMBA2_SMALL,
        "deltanet-small": DELTANET_SMALL,
        "deltanet-medium": DELTANET_MEDIUM,
        "delta-hybrid-small": DELTA_HYBRID_SMALL,
        "rwkv6-small": RWKV6_SMALL,
        "rwkv6-medium": RWKV6_MEDIUM,
        "rwkv6-hybrid-small": RWKV6_HYBRID_SMALL,
        "rwkv7-small": RWKV7_SMALL,
        "rwkv7-medium": RWKV7_MEDIUM,
        "rwkv7-hybrid-small": RWKV7_HYBRID_SMALL,
        "kda-small": KDA_SMALL,
        "kda-medium": KDA_MEDIUM,
        "kda-hybrid-small": KDA_HYBRID_SMALL,
    }
    
    # Handle ModelConfig directly
    if isinstance(model_type, ModelConfig):
        config = model_type
    elif model_type in presets:
        # Start from preset and override
        base = presets[model_type]
        config_dict = base.to_dict()
        config_dict.update(kwargs)
        config = ModelConfig(**config_dict)
    elif model_type == "custom":
        config = ModelConfig(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(presets.keys())}, 'custom', or pass a ModelConfig"
        )
    
    # Return instantiated model if rngs provided
    if rngs is not None:
        return LMModel(config, rngs=rngs)
    
    return config, LMModel
