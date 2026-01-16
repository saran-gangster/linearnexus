"""Kernel backend selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from typing import Optional

import jax


class KernelBackend(str, Enum):
    """Supported kernel backends."""

    REFERENCE = "reference"
    TRITON = "triton"
    PALLAS = "pallas"


@dataclass(frozen=True)
class BackendAvailability:
    """Availability flags for optional kernel backends."""

    triton: bool
    pallas: bool


def _has_triton() -> bool:
    try:
        import jax_triton  # noqa: F401
    except Exception:
        return False
    return any(device.platform == "gpu" for device in jax.devices())


def _has_pallas() -> bool:
    try:
        from jax.experimental import pallas  # noqa: F401
    except Exception:
        return False
    return True


def backend_availability() -> BackendAvailability:
    return BackendAvailability(triton=_has_triton(), pallas=_has_pallas())


def select_kernel_backend(preferred: Optional[str] = None) -> KernelBackend:
    """Selects a kernel backend based on preference and availability.

    Args:
        preferred: One of "auto", "reference", "triton", "pallas" or None.

    Returns:
        Selected KernelBackend.
    """
    env_pref = os.getenv("LINEARNEXUS_KERNEL_BACKEND", "auto").lower()
    if preferred is None:
        preferred = env_pref
    preferred = preferred.lower()

    availability = backend_availability()

    if preferred in {"auto", ""}:
        if availability.triton:
            return KernelBackend.TRITON
        if availability.pallas:
            return KernelBackend.PALLAS
        return KernelBackend.REFERENCE

    if preferred == KernelBackend.TRITON.value:
        return KernelBackend.TRITON if availability.triton else KernelBackend.REFERENCE
    if preferred == KernelBackend.PALLAS.value:
        return KernelBackend.PALLAS if availability.pallas else KernelBackend.REFERENCE
    return KernelBackend.REFERENCE
