"""Kernel implementations.

Currently provides pure JAX reference kernels. Custom Pallas GPU/TPU
kernels are planned for future releases (see archive/pallas_kernels/).

Available kernels:
- MambaReferenceKernel: Pure JAX selective SSM using lax.scan
- Kernel backend utilities for Triton/Pallas integration
"""

from .mamba_reference import (
    MambaKernelInputs,
    MambaKernelParams,
    MambaKernelState,
    MambaReferenceKernel,
)
from .backend import KernelBackend, select_kernel_backend

__all__ = [
    "MambaKernelInputs",
    "MambaKernelParams",
    "MambaKernelState",
    "MambaReferenceKernel",
    "KernelBackend",
    "select_kernel_backend",
]
