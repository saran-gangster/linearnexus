"""Triton kernel implementations (via JAX-Triton)."""

from .delta_rule import delta_rule_recurrent_triton

__all__ = [
    "delta_rule_recurrent_triton",
]
