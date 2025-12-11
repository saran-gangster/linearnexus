"""Kernel implementations.

Currently provides pure JAX reference kernels. Custom Pallas GPU/TPU
kernels are planned for future releases (see archive/pallas_kernels/).

Available kernels:
- MambaReferenceKernel: Pure JAX selective SSM using lax.scan
"""

from .mamba_reference import (
    MambaKernelInputs,
    MambaKernelParams,
    MambaKernelState,
    MambaReferenceKernel,
)

__all__ = [
    "MambaKernelInputs",
    "MambaKernelParams",
    "MambaKernelState",
    "MambaReferenceKernel",
]
