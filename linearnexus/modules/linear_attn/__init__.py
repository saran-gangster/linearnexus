"""Linear attention variants.

Available implementations:
- DeltaNetBlock: Delta rule linear attention (DeltaNet paper)
- GatedDeltaNetBlock: Gated DeltaNet (Mamba2 gating + Delta rule)
- RWKV6Block: RWKV-6 with matrix-valued states and data-dependent decay
- RWKV7Block: RWKV-7 (Goose) with DPLR transition matrices
- KDABlock: Kimi Delta Attention with per-dimension gating

Planned implementations:
- GLABlock: Gated Linear Attention
- RetNetBlock: Retention-based attention
"""

from .delta_net import (
    DeltaNetBlock,
    DeltaNetState,
    delta_rule_chunkwise,
    delta_rule_recurrent,
    delta_rule_step,
)

from .gated_deltanet import (
    GatedDeltaNetBlock,
    GatedDeltaNetState,
    gated_delta_rule_chunkwise,
    gated_delta_rule_recurrent,
    gated_delta_rule_step,
)

from .rwkv6 import (
    RWKV6Block,
    RWKV6State,
    rwkv6_recurrent,
    rwkv6_chunkwise,
    rwkv6_step,
    token_shift,
)

from .rwkv7 import (
    RWKV7Block,
    RWKV7State,
    dplr_delta_rule_recurrent,
    dplr_delta_rule_step,
    LoRA,
    l2_norm,
    gate_output_correction,
)

from .kda import (
    KDABlock,
    KDAState,
    kda_recurrent,
    kda_chunkwise,
    kda_step,
    kda_gate,
    FusedRMSNormGated,
)

__all__ = [
    # DeltaNet
    "DeltaNetBlock",
    "DeltaNetState",
    "delta_rule_chunkwise",
    "delta_rule_recurrent",
    "delta_rule_step",
    # Gated DeltaNet
    "GatedDeltaNetBlock",
    "GatedDeltaNetState",
    "gated_delta_rule_chunkwise",
    "gated_delta_rule_recurrent",
    "gated_delta_rule_step",
    # RWKV6
    "RWKV6Block",
    "RWKV6State",
    "rwkv6_recurrent",
    "rwkv6_chunkwise",
    "rwkv6_step",
    "token_shift",
    # RWKV7
    "RWKV7Block",
    "RWKV7State",
    "dplr_delta_rule_recurrent",
    "dplr_delta_rule_step",
    "LoRA",
    "l2_norm",
    "gate_output_correction",
    # KDA
    "KDABlock",
    "KDAState",
    "kda_recurrent",
    "kda_chunkwise",
    "kda_step",
    "kda_gate",
    "FusedRMSNormGated",
]
