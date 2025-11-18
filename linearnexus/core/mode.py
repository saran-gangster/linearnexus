"""Helpers for selecting kernel execution modes."""

from __future__ import annotations

from linearnexus.kernels import KernelMode


def select_mode(seq_len: int, *, threshold: int = 64) -> KernelMode:
    """Chooses chunk vs recurrent mode based on sequence length."""

    if seq_len <= threshold:
        return KernelMode.RECURRENT
    return KernelMode.CHUNK
