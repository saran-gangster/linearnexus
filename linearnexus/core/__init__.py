"""Shared runtime utilities reused across LinearNexus layers."""

from .cache import ConvState, RecurrentState
from .config import ConfigBase
from .conv import depthwise_conv1d_causal
from .gating import low_rank_project, normalize_gate_logits
from .mode import select_mode
from .padding import compute_unpadded_indices, pad, unpad

__all__ = [
    "ConvState",
    "RecurrentState",
    "ConfigBase",
    "depthwise_conv1d_causal",
    "compute_unpadded_indices",
    "pad",
    "unpad",
    "low_rank_project",
    "normalize_gate_logits",
    "select_mode",
]
