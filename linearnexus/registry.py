"""Feature registries for kernels, layers, and models."""

from __future__ import annotations

from typing import Dict, Tuple, Type

from linearnexus.kernels import MambaPallasKernel, MambaReferenceKernel, SelectiveKernelProtocol
from linearnexus.layers.mamba import MambaConfig, MambaLayer

KernelCls = Type[SelectiveKernelProtocol]
LayerEntry = Tuple[Type[MambaLayer], Type[MambaConfig]]

KERNEL_REGISTRY: Dict[str, KernelCls] = {
    "mamba:reference": MambaReferenceKernel,
    "mamba:pallas": MambaPallasKernel,
}

LAYER_REGISTRY: Dict[str, LayerEntry] = {
    "mamba": (MambaLayer, MambaConfig),
}

MODEL_REGISTRY: Dict[str, Tuple[Type, Type]] = {}

__all__ = [
    "KERNEL_REGISTRY",
    "LAYER_REGISTRY",
    "MODEL_REGISTRY",
]
